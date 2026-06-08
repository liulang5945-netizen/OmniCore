"""
Taiji Trainer

Trainer for Taiji (态极), a natively trained AI life form.
Two-phase training: language modeling pretrain → ReAct tool-calling finetune.

Key features:
- Seed data + collected log data training
- Language modeling + tool classification joint training
- Generator mode, yields progress for API frontend
"""
import os
import json
import time
import logging
from typing import Optional, List, Dict, Any, Generator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from taiji.config import SPECIAL_TOKENS
from taiji.architecture import ModelSelf
from taiji.tokenizer import ModelSelfTokenizer
from taiji.loader import save_model

from taiji.data.seed_data import get_seed_react_data, get_seed_conversation_data

logger = logging.getLogger("ModelSelfTrainer")


class TextDataset(Dataset):
    """
    纯文本预训练数据集。
    用于两步训练法的第一步：语言建模预训练。
    输入格式：原始文本行，自动编码为 token 序列进行自回归语言建模。
    """

    def __init__(
        self,
        tokenizer: ModelSelfTokenizer,
        texts: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = []

        for text in texts:
            ids = [tokenizer.bos_token_id]
            ids += tokenizer._encode(text)
            ids += [tokenizer.eos_token_id]

            # 按 max_length 切分长文本
            for i in range(0, len(ids), max_length):
                chunk = ids[i:i + max_length]
                if len(chunk) >= 10:  # 忽略过短的片段
                    self.encoded.append(chunk)

        logger.info(f"Pretrain dataset created: {len(self.encoded)} chunks")

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        ids = self.encoded[idx]
        # 填充到 max_length
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            ids = ids[:self.max_length]

        # 自回归：输入 = 所有 token，标签 = 所有 token（忽略 pad）
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "tool_target": torch.tensor(-100, dtype=torch.long),
        }


class ReActDataset(Dataset):
    """
    ReAct 训练数据集

    将 ReAct 步骤和对话转换为训练样本:
    - 输入: 任务描述 + 工具描述 + 上下文
    - 目标: <think>思考</think><tool_call>工具 / <final_answer>回答
    
    修复: 每个步骤独立作为样本，不提前暴露后续步骤的正确答案。
    历史步骤只包含【用户原始输入 + 工具执行结果】，不包含"正确的思考"。
    """

    # 默认工具描述（与 react_engine.py _run_native 中的 tool_desc 格式一致）
    DEFAULT_TOOL_DESC = """可用工具:
- read_local_file: 读取文件内容。参数: input(str) 文件路径
- write_file: 创建文件。参数: input(str) "路径 | 内容"
- edit_file: 编辑文件。参数: input(str) "路径 | 旧文本 | 新文本"
- list_directory: 列出目录内容。参数: input(str) 目录路径
- execute_python: 执行 Python 代码。参数: input(str) Python 代码
- search: 搜索互联网。参数: input(str) 搜索关键词
- read_webpage: 读取网页内容。参数: input(str) URL
- create_project: 创建项目。参数: input(str) "类型 | 名称"
- analyze_code: 分析代码。参数: input(str) 文件路径
- install_dependency: 安装依赖。参数: input(str) 包名
- learn_knowledge: 学习知识。参数: input(str) 知识内容
- query_knowledge: 查询知识。参数: input(str) 查询关键词
- run_command: 运行系统命令。参数: input(str) 命令"""

    def __init__(
        self,
        tokenizer: ModelSelfTokenizer,
        react_data: List[dict],
        conv_data: List[dict],
        max_length: int = 512,
        tool_descriptions: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tool_descriptions = tool_descriptions or self.DEFAULT_TOOL_DESC
        self.samples = []

        # 处理 ReAct 数据
        for item in react_data:
            self.samples.extend(self._process_react(item))

        # 处理对话数据
        for item in conv_data:
            self.samples.extend(self._process_conversation(item))

        logger.info(f"ReAct dataset created: {len(self.samples)} samples")

    def _process_react(self, item: dict) -> list:
        """
        处理一条 ReAct 数据，生成多个训练样本。
        
        关键设计：训练 prompt 格式与 react_engine.py _run_native() 的推理格式完全一致。
        这样训练出来的模型在真实推理时看到的 prompt 分布与训练时完全匹配。
        
        推理格式（来自 react_engine.py 第 293-353 行）:
            prompt = f"[系统] 你是 OmniCore AI 助手。可用工具:\n{tool_desc}\n[用户] {task}\n[助手] "
            ...
            prompt += f"<think>{thought}</think><tool_call>{action} {args}\n<tool_result>{obs}</tool_result>\n"
        """
        samples = []
        task = item["task"]
        steps = item.get("steps", [])

        # 多模态上下文注入：让模型知道输入包含媒体文件
        modality = item.get("modality")
        media_path = item.get("media_path")
        if modality and media_path:
            media_name = os.path.basename(media_path)
            task = f"[{modality.upper()}: {media_name}] {task}"

        for i, step in enumerate(steps):
            thought = step.get("thought", "")
            action = step.get("action")
            action_args = step.get("action_args", {})
            final_answer = step.get("final_answer", "")
            # 模拟 observation（真实推理时由工具返回）
            observation = step.get("observation", f"{action} 执行完成。" if action else "")

            # ── 构建输入：格式与 _run_native() 完全一致 ──
            # Step 1: 系统提示 + 工具描述 + 用户任务
            input_parts = [
                f"[系统] 你是 OmniCore AI 助手。可用工具:\n{self.tool_descriptions}",
                f"[用户] {task}",
                "[助手] ",
            ]

            # Step 2: 历史步骤 — 用 <think>/<tool_call>/<tool_result> 格式
            # 与 react_engine.py 第 353 行的格式完全一致
            for j in range(i):
                prev_step = steps[j]
                prev_action = prev_step.get("action")
                prev_action_args = prev_step.get("action_args", {})
                # 历史步骤不注入正确答案的思考，只注入动作和观察
                if prev_action:
                    prev_args_str = json.dumps(prev_action_args, ensure_ascii=False)
                    prev_observation = prev_step.get("observation", f"{prev_action} 执行完成。")
                    input_parts.append(
                        f"<think>我需要使用 {prev_action} 工具来完成这一步。</think>"
                        f"<tool_call>{prev_action} {prev_args_str}\n"
                        f"<tool_result>{prev_observation}</tool_result>\n"
                    )

            input_text = "".join(input_parts)

            # ── 构建目标（只含当前这一步！）──
            if final_answer:
                target_text = f"<think>{thought}</think><final_answer>{final_answer}</final_answer>"
                tool_target = -100
            elif action:
                args_str = json.dumps(action_args, ensure_ascii=False)
                target_text = f"<think>{thought}</think><tool_call>{action} {args_str}"
                tool_id = self.tokenizer.get_tool_id(action)
                tool_target = tool_id if tool_id is not None else -100
            else:
                continue

            samples.append({
                "input_text": input_text,
                "target_text": target_text,
                "tool_target": tool_target,
            })

        return samples

    def _process_conversation(self, item: dict) -> list:
        """处理对话样本"""
        samples = []
        messages = item.get("messages", [])

        for i in range(len(messages)):
            if messages[i]["role"] != "assistant":
                continue

            input_parts = []
            for msg in messages[:i]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    input_parts.append(f"[系统] {content}")
                elif role == "user":
                    input_parts.append(f"[用户] {content}")
                elif role == "assistant":
                    input_parts.append(f"[助手] {content}")

            input_text = "\n".join(input_parts) + "\n[助手] "
            target_text = messages[i]["content"]

            samples.append({
                "input_text": input_text,
                "target_text": target_text,
                "tool_target": -100,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 编码输入：添加 BOS token
        input_ids = [self.tokenizer.bos_token_id]
        input_ids += self.tokenizer._encode(sample["input_text"])

        # 编码目标：添加 EOS token
        target_ids = self.tokenizer._encode(sample["target_text"])
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        # 将工具 token ID 转换为工具索引（使用查找表，不依赖连续性）
        tool_target = sample["tool_target"]
        if tool_target >= 0:
            all_tool_ids = sorted(self.tokenizer._tool_name_to_id.values())
            try:
                tool_idx = all_tool_ids.index(tool_target)
                if tool_idx >= 750:
                    tool_idx = -100
            except ValueError:
                tool_idx = -100
        else:
            tool_idx = tool_target

        # 滑动窗口截断：优先保留目标，输入从末尾截断
        if len(input_ids) + len(target_ids) > self.max_length:
            max_input_len = self.max_length - len(target_ids)
            if max_input_len > 0:
                input_ids = input_ids[-max_input_len:]
            else:
                target_ids = target_ids[:self.max_length]
                input_ids = []

        # 拼接
        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids

        # 填充到 max_length
        pad_len = self.max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tool_target": torch.tensor(tool_idx, dtype=torch.long),
        }

    def split(self, val_ratio: float = 0.1):
        """将数据集分为训练集和验证集"""
        import copy
        n = len(self.samples)
        split_idx = max(int(n * (1 - val_ratio)), 1)
        train_ds = copy.copy(self)
        train_ds.samples = self.samples[:split_idx]
        val_ds = copy.copy(self)
        val_ds.samples = self.samples[split_idx:]
        return train_ds, val_ds


class ModelSelfTrainer:
    """
    ModelSelf 训练器

    generator 模式，yield 进度供 API 前端使用。
    支持两步训练法：
    1. pretrain() — 纯语言建模预训练
    2. finetune() — ReAct 工具调用微调
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Args:
            model: ModelSelf（态极）
            tokenizer: ModelSelfTokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        self.scheduler = None
        self.warmup_steps = warmup_steps

        self.global_step = 0
        self.best_loss = float("inf")
        self.is_paused = False
        self.is_stopped = False
        self.loss_history = []

        # 早停配置
        self.early_stopping_patience = 3
        self.early_stopping_threshold = 0.001
        self._patience_counter = 0

        # 瓶颈检测配置
        self._bottleneck_enabled = True
        self._stagnant_window = 50       # 最近 N 步检测停滞
        self._stagnant_threshold = 0.01  # loss 波动阈值
        self._bottleneck_detected = False
        self._upgrade_suggested = False

    def _setup_scheduler(self, total_steps: int):
        """安全地设置学习率调度器"""
        # OneCycleLR 需要 total_steps > 5 才能安全工作（PyTorch 内部有除零风险）
        if total_steps <= 5:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=max(total_steps, 1),
            )
            return
        safe_warmup = min(self.warmup_steps, max(total_steps // 4, 1))
        # 确保 pct_start < 0.9 且剩余步数足够
        if safe_warmup >= total_steps - 2:
            safe_warmup = max(total_steps // 5, 1)
        pct_start = safe_warmup / total_steps
        # OneCycleLR 要求 0 < pct_start < 1
        if pct_start >= 1.0:
            pct_start = 0.3
        if pct_start <= 0.0:
            pct_start = 0.1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=pct_start,
        )

    def pretrain(
        self,
        dataset: TextDataset,
        num_epochs: int = 5,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints/pretrain",
        save_steps: int = 100,
        log_steps: int = 10,
        device: str = "cpu",
    ) -> Generator:
        """
        第一步：预训练（纯语言建模）。

        只训练语言头，不激活工具头。
        Yields:
            (fraction, description, loss_history, metrics_dict)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        total_steps = max(len(dataloader) * num_epochs // self.gradient_accumulation_steps, 1)

        self._setup_scheduler(total_steps)
        os.makedirs(save_dir, exist_ok=True)
        loss_history = []
        self._patience_counter = 0
        self.model.train()

        # 阶段一：只训练语言头，冻结其他头
        self.freeze_heads(exclude=["language"])
        logger.info("Pretrain: 冻结工具/感知/记忆/规划头，只训练语言头")

        # AMP 配置
        use_amp = device.startswith("cuda") and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        train_start = time.time()
        logger.info(f"Starting pretrain: {len(dataset)} chunks, {num_epochs} epochs, {total_steps} steps, AMP={use_amp}")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                if self.is_stopped:
                    return
                while self.is_paused:
                    time.sleep(0.1)
                    yield (self.global_step / max(total_steps, 1),
                           "预训练已暂停", loss_history, {"status": "paused"})

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = self.model(
                        input_ids,
                        targets=labels,
                        tool_head_active=False,
                    )
                    loss = output.loss / self.gradient_accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    if trainable_params:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    else:
                        grad_norm = torch.tensor(0.0)
                    if use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_steps == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        lr = self.scheduler.get_last_lr()[0]
                        loss_history.append(avg_loss)
                        self.loss_history = loss_history
                        elapsed = time.time() - train_start
                        eta = (elapsed / self.global_step) * (total_steps - self.global_step) if self.global_step > 0 else 0
                        samples_per_sec = self.global_step * batch_size / elapsed if elapsed > 0 else 0
                        yield (
                            self.global_step / total_steps,
                            f"预训练 Epoch {epoch+1} | Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}",
                            loss_history,
                            {
                                "loss": avg_loss, "lr": lr, "step": self.global_step, "phase": "pretrain",
                                "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                                "elapsed": round(elapsed, 1), "eta": round(eta, 1),
                                "samples_per_sec": round(samples_per_sec, 2),
                                "epoch": epoch + 1, "total_epochs": num_epochs,
                            },
                        )

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(save_dir, epoch_loss / max(num_batches, 1))

            # Epoch 结束：早停检查
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Pretrain epoch {epoch+1} done, avg loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < self.best_loss - self.early_stopping_threshold:
                self.best_loss = avg_epoch_loss
                self._patience_counter = 0
                self._save_checkpoint(save_dir, avg_epoch_loss)
            else:
                self._patience_counter += 1

            if self._patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping pretrain at epoch {epoch+1}")
                yield (1.0, f"早停: 连续 {self.early_stopping_patience} 轮 loss 无改善", loss_history,
                       {"status": "early_stopped", "phase": "pretrain", "loss": avg_epoch_loss})
                return

        self._save_checkpoint(save_dir, self.best_loss)
        self.loss_history = loss_history
        yield (1.0, f"预训练完成! Loss: {avg_epoch_loss:.4f}", loss_history, {"status": "completed", "phase": "pretrain"})

    def finetune(
        self,
        dataset: ReActDataset,
        num_epochs: int = 10,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints/finetune",
        save_steps: int = 50,
        log_steps: int = 5,
        device: str = "cpu",
    ) -> Generator:
        """
        第二步：ReAct 微调（联合训练语言头 + 工具头）。

        同时训练语言建模和工具调用。支持 AMP、早停、验证集。
        Yields:
            (fraction, description, loss_history, metrics_dict)
        """
        # 分割训练/验证集
        if len(dataset) >= 20:
            train_ds, val_ds = dataset.split(val_ratio=0.1)
        else:
            train_ds, val_ds = dataset, None

        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) % batch_size != 0)
        total_steps = max(len(dataloader) * num_epochs // self.gradient_accumulation_steps, 1)

        self._setup_scheduler(total_steps)
        os.makedirs(save_dir, exist_ok=True)
        loss_history = self.loss_history[:] if self.loss_history else []
        self._patience_counter = 0
        self.model.train()

        # 阶段二：训练语言头 + 工具头，冻结其他头
        self.freeze_heads(exclude=["language", "tool"])
        logger.info("Finetune: 冻结感知/记忆/规划头，训练语言头 + 工具头")

        # AMP 配置
        use_amp = device.startswith("cuda") and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        train_start = time.time()
        logger.info(f"Starting finetune: {len(train_ds)} train, {len(val_ds) if val_ds else 0} val, {num_epochs} epochs, {total_steps} steps, AMP={use_amp}")

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_tool_loss = 0
            num_batches = 0
            num_tool_batches = 0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                if self.is_stopped:
                    return
                while self.is_paused:
                    time.sleep(0.1)
                    yield (self.global_step / max(total_steps, 1),
                           "微调已暂停", loss_history, {"status": "paused"})

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                tool_targets = batch["tool_target"].to(device)

                has_tool = (tool_targets != -100).any().item()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = self.model(
                        input_ids,
                        targets=labels,
                        tool_targets=tool_targets if has_tool else None,
                        tool_head_active=has_tool,
                    )
                    loss = output.loss / self.gradient_accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    if trainable_params:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    else:
                        grad_norm = torch.tensor(0.0)
                    if use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_steps == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        lr = self.scheduler.get_last_lr()[0]
                        loss_history.append(avg_loss)
                        self.loss_history = loss_history
                        elapsed = time.time() - train_start
                        eta = (elapsed / self.global_step) * (total_steps - self.global_step) if self.global_step > 0 else 0
                        samples_per_sec = self.global_step * batch_size / elapsed if elapsed > 0 else 0

                        metrics = {
                            "loss": avg_loss, "lr": lr, "step": self.global_step,
                            "phase": "finetune",
                            "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                            "elapsed": round(elapsed, 1), "eta": round(eta, 1),
                            "samples_per_sec": round(samples_per_sec, 2),
                            "epoch": epoch + 1, "total_epochs": num_epochs,
                        }
                        if has_tool:
                            metrics["tool_active"] = True

                        yield (
                            self.global_step / total_steps,
                            f"微调 Epoch {epoch+1}/{num_epochs} | Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}",
                            loss_history,
                            metrics,
                        )

                        # 瓶颈检测
                        if self._bottleneck_enabled and not self._bottleneck_detected:
                            bn = self._check_bottleneck(loss_history)
                            if bn["bottleneck"]:
                                yield (
                                    self.global_step / total_steps,
                                    f"⚠️ 检测到训练瓶颈: {', '.join(bn['triggers'])}。建议升级模型以获得更好的效果。",
                                    loss_history,
                                    {
                                        "loss": avg_loss, "lr": lr,
                                        "step": self.global_step,
                                        "phase": "finetune",
                                        "bottleneck": True,
                                        "bottleneck_triggers": bn["triggers"],
                                        "should_upgrade": True,
                                    },
                                )

                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(save_dir, epoch_loss / max(num_batches, 1))

            # Epoch 结束
            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            # 验证集评估
            val_loss = None
            if val_ds and len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                val_loss = self._evaluate(val_loader, device)

            monitor_loss = val_loss if val_loss is not None else avg_epoch_loss
            logger.info(f"Finetune epoch {epoch+1} done, train_loss={avg_epoch_loss:.4f}, val_loss={val_loss if val_loss else 'N/A'}")

            # 早停检查
            if monitor_loss < self.best_loss - self.early_stopping_threshold:
                self.best_loss = monitor_loss
                self._patience_counter = 0
                self._save_checkpoint(save_dir, monitor_loss)
            else:
                self._patience_counter += 1

            # Epoch 完成事件
            epoch_metrics = {
                "loss": avg_epoch_loss, "step": self.global_step, "phase": "finetune",
                "epoch": epoch + 1, "total_epochs": num_epochs, "status": "epoch_end",
            }
            if val_loss is not None:
                epoch_metrics["val_loss"] = round(val_loss, 4)
            yield (
                (epoch + 1) / num_epochs,
                f"Epoch {epoch+1}/{num_epochs} 完成 | Loss: {avg_epoch_loss:.4f}" + (f" | Val: {val_loss:.4f}" if val_loss else ""),
                loss_history,
                epoch_metrics,
            )

            if self._patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping finetune at epoch {epoch+1}")
                yield (1.0, f"早停: 连续 {self.early_stopping_patience} 轮 loss 无改善", loss_history,
                       {"status": "early_stopped", "phase": "finetune", "loss": avg_epoch_loss,
                        "val_loss": val_loss, "best_loss": self.best_loss})
                return

        self._save_checkpoint(save_dir, self.best_loss)
        self.loss_history = loss_history
        yield (1.0, f"微调完成! Loss: {avg_epoch_loss:.4f}", loss_history,
               {"status": "completed", "phase": "finetune", "best_loss": self.best_loss})

    def train(
        self,
        dataset: ReActDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints",
        save_steps: int = 100,
        log_steps: int = 10,
        device: str = "cpu",
    ):
        """
        兼容旧接口的单步训练（直接 ReAct 微调，无预训练阶段）。
        
        Yields:
            (fraction, description, loss_history, metrics_dict)
        """
        return self.finetune(dataset, num_epochs, batch_size, save_dir, save_steps, log_steps, device)

    def _evaluate(self, dataloader, device: str) -> float:
        """在验证集上评估 loss"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                tool_targets = batch["tool_target"].to(device)
                has_tool = (tool_targets != -100).any().item()
                output = self.model(
                    input_ids,
                    targets=labels,
                    tool_targets=tool_targets if has_tool else None,
                    tool_head_active=has_tool,
                )
                total_loss += output.loss.item()
                count += 1
        self.model.train()
        return total_loss / max(count, 1)

    def _save_checkpoint(self, save_dir: str, loss: float):
        training_state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        checkpoint_dir = os.path.join(save_dir, f"step_{self.global_step}")
        save_model(self.model, self.tokenizer, checkpoint_dir, training_state)
        if loss < self.best_loss:
            self.best_loss = loss
            save_model(self.model, self.tokenizer, os.path.join(save_dir, "best"), training_state)

    def train_progressive(
        self,
        dataset,
        save_dir: str,
        num_epochs: int = 5,
        batch_size: int = 4,
        log_steps: int = 5,
        device: str = "cpu",
    ) -> Generator:
        """
        渐进式分阶段训练 — 逐头训练，确保每个头都得到充分学习。

        训练顺序：
        1. 语言头（pretrain）
        2. 工具头（finetune）
        3. 联合微调（全部头）

        Yields:
            (fraction, description, loss_history, metrics_dict)
        """
        total_phases = 3
        phase_idx = 0

        # === 阶段一：语言预训练 ===
        phase_idx += 1
        logger.info(f"=== 阶段 {phase_idx}/{total_phases}: 语言预训练 ===")
        yield (
            phase_idx / total_phases * 0.01,
            f"📚 阶段 {phase_idx}: 语言预训练（冻结其他头）",
            [],
            {"phase": "pretrain", "status": "starting"},
        )

        for frac, desc, hist, metrics in self.pretrain(
            dataset, num_epochs=max(num_epochs // 2, 1),
            batch_size=batch_size, save_dir=os.path.join(save_dir, "pretrain"),
            device=device, log_steps=log_steps,
        ):
            global_frac = (phase_idx - 1 + frac) / total_phases
            yield (global_frac, f"[预训练] {desc}", hist, metrics)

        # === 阶段二：工具微调（如果有 ReAct 数据）===
        phase_idx += 1
        logger.info(f"=== 阶段 {phase_idx}/{total_phases}: 工具微调 ===")
        yield (
            phase_idx / total_phases * 0.01,
            f"🔧 阶段 {phase_idx}: 工具微调（训练语言+工具头）",
            [],
            {"phase": "finetune", "status": "starting"},
        )

        for frac, desc, hist, metrics in self.finetune(
            dataset, num_epochs=num_epochs,
            batch_size=batch_size, save_dir=os.path.join(save_dir, "finetune"),
            device=device, log_steps=log_steps,
        ):
            global_frac = (phase_idx - 1 + frac) / total_phases
            yield (global_frac, f"[微调] {desc}", hist, metrics)

        # === 阶段三：联合微调 ===
        phase_idx += 1
        logger.info(f"=== 阶段 {phase_idx}/{total_phases}: 联合微调 ===")

        # 解冻所有头
        self.unfreeze_all()

        for frac, desc, hist, metrics in self.finetune(
            dataset, num_epochs=max(num_epochs // 2, 1),
            batch_size=batch_size, save_dir=os.path.join(save_dir, "joint"),
            device=device, log_steps=log_steps,
        ):
            global_frac = (phase_idx - 1 + frac) / total_phases
            yield (global_frac, f"[联合] {desc}", hist, metrics)

        yield (1.0, "✅ 渐进式训练完成", [], {"status": "completed"})

    def freeze_heads(self, exclude: list = None):
        """
        冻结所有头（除了 exclude 列表中的）。

        Args:
            exclude: 不冻结的头名称列表，如 ["language", "tool"]
        """
        exclude = exclude or []

        # 冻结 backbone（如果需要）
        # 注意：通常 backbone 不冻结，因为所有头共享 backbone

        head_map = {
            "tool": "tool_head",
            "perception": "perception_head",
            "memory": "memory_head",
            "plan": "plan_head",
        }

        for head_name, attr_name in head_map.items():
            head = getattr(self.model, attr_name, None)
            if head is None:
                continue

            requires_grad = head_name in exclude
            for param in head.parameters():
                param.requires_grad = requires_grad

            status = "训练中" if requires_grad else "已冻结"
            logger.info(f"  {head_name} 头: {status}")

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("所有参数已解冻")

    def get_head_losses(self, logits, targets, tool_logits, tool_targets,
                        tool_head_active, perception_logits=None, memory_logits=None,
                        plan_logits=None) -> dict:
        """
        计算各头的独立损失（用于监控和加权）。

        Returns:
            {"language": float, "tool": float, "total": float}
        """
        losses = {}

        # 语言损失
        if targets is not None:
            lang_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            losses["language"] = lang_loss.item()

        # 工具损失
        if tool_head_active and tool_logits is not None and tool_targets is not None:
            tool_loss = F.cross_entropy(tool_logits, tool_targets)
            losses["tool"] = tool_loss.item()

        return losses

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_stopped = True

    def _check_bottleneck(self, loss_history: list) -> dict:
        """
        检测训练瓶颈。

        瓶颈条件：
        1. loss 停滞：最近 N 步的 loss 波动 < 阈值
        2. loss 不再下降：最近 N 步的平均 loss >= 之前 N 步的平均 loss

        Returns:
            {"bottleneck": bool, "reason": str, "triggers": list}
        """
        if not self._bottleneck_enabled:
            return {"bottleneck": False, "triggers": []}

        triggers = []
        window = self._stagnant_window

        if len(loss_history) >= window * 2:
            recent = loss_history[-window:]
            previous = loss_history[-window * 2:-window]

            # 条件1: 波动太小（loss 平坦）
            recent_range = max(recent) - min(recent)
            if recent_range < self._stagnant_threshold:
                triggers.append("loss_stagnant")

            # 条件2: 不再下降（近期均值 >= 之前均值）
            recent_avg = sum(recent) / len(recent)
            prev_avg = sum(previous) / len(previous)
            if recent_avg >= prev_avg * 0.99:
                triggers.append("loss_not_improving")

        bottleneck = len(triggers) >= 1

        if bottleneck and not self._bottleneck_detected:
            self._bottleneck_detected = True
            logger.warning(f"⚠️ 训练瓶颈检测触发: {triggers}")

        return {
            "bottleneck": bottleneck,
            "triggers": triggers,
            "should_upgrade": bottleneck,
        }


def build_dataset(
    tokenizer: ModelSelfTokenizer,
    extra_react_data: Optional[List[dict]] = None,
    extra_conv_data: Optional[List[dict]] = None,
    max_length: int = 512,
) -> ReActDataset:
    """
    构建 ReAct 训练数据集。

    合并种子数据 + 多模态种子数据 + 额外收集的数据。
    """
    react_data = get_seed_react_data()
    conv_data = get_seed_conversation_data()

    # 多模态种子数据（教模型调用视觉/音频/视频工具）
    try:
        from taiji.multimodal_data_generator import generate_multimodal_seed_data
        react_data.extend(generate_multimodal_seed_data())
    except Exception:
        pass  # 多模态模块不可用时静默跳过

    if extra_react_data:
        react_data.extend(extra_react_data)
    if extra_conv_data:
        conv_data.extend(extra_conv_data)

    return ReActDataset(tokenizer, react_data, conv_data, max_length)