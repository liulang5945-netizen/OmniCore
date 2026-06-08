"""
态极 DPO 训练器
================

用 Constitutional AI 生成偏好数据，做 DPO 对齐。
不需要人工标注，不需要外部奖励模型。

流程：
  1. 对同一问题生成多个回答
  2. Constitutional AI 评分选出 best/worst
  3. 构成 (chosen, rejected) 配对
  4. DPO 训练：chosen 概率 > rejected 概率
"""
import os
import json
import time
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

logger = logging.getLogger("Taiji.DPO")


@dataclass
class PreferencePair:
    """偏好数据对"""
    prompt: str
    chosen: str    # 好回答
    rejected: str  # 差回答
    score_diff: float = 0.0  # 分数差异


class DPOTrainer:
    """
    DPO 训练器

    使用 Constitutional AI 生成偏好数据，然后用 DPO 损失训练模型。
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        beta: float = 0.1,
        learning_rate: float = 1e-6,
    ):
        """
        Args:
            model: 态极模型
            tokenizer: 分词器
            device: 设备
            beta: DPO 温度参数（越大越保守）
            learning_rate: 学习率（DPO 用小学习率）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.lr = learning_rate

    def generate_preference_data(
        self,
        prompts: List[str],
        num_samples: int = 4,
        temperature: float = 0.8,
    ) -> List[PreferencePair]:
        """
        生成偏好数据。

        对每个 prompt 生成多个回答，用 Constitutional AI 评分，
        选出 best 和 worst 构成配对。

        Args:
            prompts: 提示列表
            num_samples: 每个提示生成的回答数
            temperature: 生成温度（越高越多样）

        Returns:
            偏好数据对列表
        """
        from taiji.constitutional_ai import get_constitutional_ai
        from taiji.inference import NativeInferenceEngine

        critic = get_constitutional_ai()
        engine = NativeInferenceEngine(self.model, self.tokenizer, self.device)

        pairs = []

        for i, prompt in enumerate(prompts):
            logger.info(f"生成偏好数据: {i+1}/{len(prompts)}")

            # 生成多个回答
            responses = []
            for _ in range(num_samples):
                try:
                    response = engine.generate(
                        prompt,
                        max_new_tokens=256,
                        temperature=temperature,
                    )
                    # 评分
                    critique = critic.critique(response, {"task": prompt})
                    score = 1.0 - len(critique.violations) * 0.2  # 违规越少分越高
                    responses.append((response, max(score, 0.0)))
                except Exception as e:
                    logger.warning(f"生成失败: {e}")
                    continue

            if len(responses) < 2:
                continue

            # 选出 best 和 worst
            responses.sort(key=lambda x: x[1], reverse=True)
            chosen = responses[0][0]
            rejected = responses[-1][0]
            score_diff = responses[0][1] - responses[-1][1]

            # 只保留有明显差异的配对
            if score_diff > 0.1 and chosen != rejected:
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    score_diff=score_diff,
                ))

        logger.info(f"生成了 {len(pairs)} 个偏好数据对（来自 {len(prompts)} 个提示）")
        return pairs

    def train(
        self,
        preference_data: List[PreferencePair],
        num_epochs: int = 3,
        batch_size: int = 4,
        save_dir: str = None,
    ) -> Generator:
        """
        DPO 训练。

        Yields:
            (epoch, step, loss, metrics)
        """
        # 获取训练锁，防止并发训练导致 inplace 操作冲突
        _app_state = None
        try:
            from core.app_state import app_state
            _app_state = app_state
            if not app_state.try_start_training():
                logger.warning("其他训练正在进行，无法启动 DPO 训练")
                return
        except ImportError:
            pass

        try:
            yield from self._do_train(preference_data, num_epochs, batch_size, save_dir)
        finally:
            if _app_state is not None:
                _app_state.finish_training()

    def _do_train(
        self,
        preference_data: List[PreferencePair],
        num_epochs: int,
        batch_size: int,
        save_dir: str,
    ) -> Generator:
        """DPO 训练的实际逻辑（调用方已持有训练锁）"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()

        total_steps = len(preference_data) * num_epochs
        step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(preference_data), batch_size):
                batch = preference_data[i:i + batch_size]
                batch_loss = 0.0

                for pair in batch:
                    # 计算 chosen 的 log probability
                    chosen_logprob = self._compute_logprob(pair.prompt, pair.chosen)
                    # 计算 rejected 的 log probability
                    rejected_logprob = self._compute_logprob(pair.prompt, pair.rejected)

                    # DPO 损失
                    # L = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))
                    logit = self.beta * (chosen_logprob - rejected_logprob)
                    loss = -F.logsigmoid(logit)

                    batch_loss += loss

                # 反向传播
                batch_loss = batch_loss / len(batch)
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                step += 1
                epoch_loss += batch_loss.item()
                num_batches += 1

                yield (
                    epoch,
                    step,
                    batch_loss.item(),
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": batch_loss.item(),
                        "avg_loss": epoch_loss / num_batches,
                    },
                )

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"DPO Epoch {epoch+1}/{num_epochs} done, avg loss: {avg_loss:.4f}")

        # 保存
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            from taiji.loader import save_model
            save_path = os.path.join(save_dir, "dpo_best")
            save_model(self.model, self.tokenizer, save_path)
            logger.info(f"DPO 模型已保存到 {save_path}")

    def _compute_logprob(self, prompt: str, response: str) -> torch.Tensor:
        """
        计算给定 prompt 下 response 的 log probability。

        Args:
            prompt: 输入提示
            response: 目标回答

        Returns:
            log probability 标量
        """
        # 拼接 prompt + response
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # 只计算 response 部分的 log probability
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # 取 response 部分的 logits
        response_logits = logits[:, prompt_len - 1:-1, :]  # [1, resp_len, vocab]
        response_ids = input_ids[:, prompt_len:]  # [1, resp_len]

        # 计算 log probability
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        # 求和（总 log probability）
        total_logprob = token_log_probs.sum()

        return total_logprob


def save_preference_data(pairs: List[PreferencePair], path: str):
    """保存偏好数据到文件"""
    data = []
    for pair in pairs:
        data.append({
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "score_diff": pair.score_diff,
        })

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"偏好数据已保存到 {path}: {len(data)} 条")


def load_preference_data(path: str) -> List[PreferencePair]:
    """从文件加载偏好数据"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                pairs.append(PreferencePair(
                    prompt=item["prompt"],
                    chosen=item["chosen"],
                    rejected=item["rejected"],
                    score_diff=item.get("score_diff", 0.0),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"跳过无效行: {e}")

    logger.info(f"加载了 {len(pairs)} 个偏好数据对")
    return pairs
