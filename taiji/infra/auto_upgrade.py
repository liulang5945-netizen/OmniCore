"""
Taiji Auto-Upgrade Engine

Integrates capability evaluation, bottleneck detection, knowledge distillation, and hot-switching.

M3: Capability Evaluator + "Graduation" mechanism
M5: Knowledge Distillation Upgrader
M6: Complete auto-upgrade closed loop

Taiji is a natively trained AI life form — upgrades preserve the native architecture.
"""
import json
import logging
import os
import time
import threading
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("Taiji.AutoUpgrade")


class CapabilityEvaluator:
    """
    M3: 能力评估器
    评估态极在各类问题上的能力，判断是否"毕业"
    """
    
    CATEGORIES = ["general", "code", "knowledge", "creative", "math"]
    
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        # 能力分数 (0~1)
        self.scores = {cat: 0.1 for cat in self.CATEGORIES}
        # 评估历史
        self.eval_history = []
        self._load_scores()
    
    def _load_scores(self):
        if self.save_path and os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                self.scores = data.get("scores", self.scores)
                logger.info(f"已加载能力分数: {self.scores}")
            except Exception:
                pass
    
    def _save_scores(self):
        if self.save_path:
            try:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                with open(self.save_path, "w") as f:
                    json.dump({
                        "scores": self.scores,
                        "updated_at": time.time(),
                    }, f, indent=2)
            except Exception:
                pass
    
    def evaluate_response(self, category: str, taiji_response: str, 
                          reference_response: str) -> float:
        """
        评估态极的回答质量（与参考回答对比）
        返回 0~1 的分数
        """
        if not taiji_response or not reference_response:
            return 0.0
        
        # 简单评估：基于文本相似度
        # 实际应用中可以使用更复杂的评估方法
        score = 0.0
        
        # 1. 长度相似度（太短说明能力不足）
        len_ratio = min(len(taiji_response), len(reference_response)) / max(len(taiji_response), len(reference_response), 1)
        score += len_ratio * 0.3
        
        # 2. 关键词覆盖率
        ref_words = set(reference_response.split())
        taiji_words = set(taiji_response.split())
        if ref_words:
            coverage = len(ref_words & taiji_words) / len(ref_words)
            score += coverage * 0.4
        
        # 3. 不包含错误标记
        error_markers = ["抱歉", "无法", "失败", "sorry", "error", "cannot"]
        if not any(m in taiji_response.lower() for m in error_markers):
            score += 0.3
        
        return min(1.0, score)
    
    def update_score(self, category: str, new_score: float):
        """更新某类问题的能力分数（指数移动平均）"""
        if category not in self.scores:
            category = "general"
        
        old = self.scores[category]
        # 指数移动平均，权重 0.2 给新分数
        self.scores[category] = min(1.0, old * 0.8 + new_score * 0.2)
        
        self.eval_history.append({
            "category": category,
            "old_score": old,
            "new_score": self.scores[category],
            "raw_score": new_score,
            "timestamp": time.time(),
        })
        
        # 每 10 次评估保存一次
        if len(self.eval_history) % 10 == 0:
            self._save_scores()
    
    def is_graduated(self, category: str, threshold: float = 0.8) -> bool:
        """某类问题是否已"毕业"（可以独立回答）"""
        return self.scores.get(category, 0) >= threshold
    
    def get_average_capability(self) -> float:
        """平均能力分数"""
        return sum(self.scores.values()) / len(self.scores)
    
    def get_status(self) -> dict:
        return {
            "scores": self.scores.copy(),
            "average": round(self.get_average_capability(), 3),
            "total_evaluations": len(self.eval_history),
            "graduated": [c for c, s in self.scores.items() if s >= 0.8],
        }


class BottleneckDetector:
    """
    M4: 瓶颈检测器
    检测态极模型是否需要升级
    """
    
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        # 历史 loss 记录
        self.loss_history = []
        self.last_upgrade_check = 0
        self.check_interval = 86400  # 每天最多检查一次
        self._load_history()
    
    def _load_history(self):
        if self.save_path and os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                self.loss_history = data.get("loss_history", [])
            except Exception:
                pass
    
    def _save_history(self):
        if self.save_path:
            try:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                with open(self.save_path, "w") as f:
                    json.dump({
                        "loss_history": self.loss_history[-100:],  # 只保留最近100条
                        "last_check": self.last_upgrade_check,
                    }, f, indent=2)
            except Exception:
                pass
    
    def record_training_loss(self, loss: float, epoch: int):
        """记录训练 loss"""
        self.loss_history.append({
            "loss": loss,
            "epoch": epoch,
            "timestamp": time.time(),
        })
        if len(self.loss_history) % 5 == 0:
            self._save_history()
    
    def detect_bottleneck(self, capability_evaluator: CapabilityEvaluator) -> dict:
        """
        检测是否遇到瓶颈
        瓶颈条件（任意2个同时触发）：
        1. 训练 loss 停滞（最近 N 个 epoch 不下降）
        2. 能力分数饱和（不再上升）
        3. 已积累足够训练数据但能力仍低
        """
        now = time.time()
        if now - self.last_upgrade_check < self.check_interval:
            return {"bottleneck": False, "reason": "检查间隔未到"}
        
        self.last_upgrade_check = now
        
        triggers = []
        
        # 条件1: loss 停滞
        if len(self.loss_history) >= 10:
            recent_losses = [h["loss"] for h in self.loss_history[-10:]]
            if max(recent_losses) - min(recent_losses) < 0.01:
                triggers.append("loss_stagnant")
        
        # 条件2: 能力分数饱和
        avg_cap = capability_evaluator.get_average_capability()
        if avg_cap > 0.5 and len(capability_evaluator.eval_history) >= 20:
            recent_evals = capability_evaluator.eval_history[-20:]
            recent_scores = [e["new_score"] for e in recent_evals]
            if max(recent_scores) - min(recent_scores) < 0.05:
                triggers.append("capability_saturated")
        
        # 条件3: 模型太小（125M/350M）且已有一定能力
        if avg_cap > 0.4:
            triggers.append("model_too_small")
        
        bottleneck = len(triggers) >= 2
        
        result = {
            "bottleneck": bottleneck,
            "triggers": triggers,
            "trigger_count": len(triggers),
            "avg_capability": round(avg_cap, 3),
            "loss_count": len(self.loss_history),
            "eval_count": len(capability_evaluator.eval_history),
        }
        
        if bottleneck:
            logger.info(f"🚨 检测到瓶颈！触发条件: {triggers}")
        
        return result


class KnowledgeDistiller:
    """
    M5: 知识蒸馏器
    将旧模型的知识迁移到新模型
    """
    
    def __init__(self):
        self.distill_config = {
            "temperature": 2.0,     # 蒸馏温度
            "alpha": 0.3,           # 蒸馏损失权重（降低，让模型更多学习原始任务）
            "epochs": 10,           # 增加训练轮数
            "learning_rate": 5e-5,  # 降低学习率，训练更稳定
            "batch_size": 4,        # 减小 batch，适应更大模型
            "warmup_steps": 50,     # 学习率预热
            "max_length": 256,      # 序列长度
            "grad_accum": 4,        # 梯度累积步数
        }
    
    def distill(
        self,
        teacher_model,      # 旧模型（教师）
        student_model,      # 新模型（学生）
        tokenizer,          # 分词器
        training_data: list,  # 训练数据
        device: str = "cpu",
        progress_callback=None,
    ) -> dict:
        """
        执行知识蒸馏
        
        Args:
            teacher_model: 旧模型（125M）
            student_model: 新模型（350M/1B）
            tokenizer: 分词器
            training_data: 训练数据列表
            device: 设备
            progress_callback: 进度回调 (step, total, loss)
        
        Returns:
            {"status": "success"/"failed", "final_loss": float, ...}
        """
        import torch
        import torch.nn.functional as F

        # 获取训练锁，防止并发训练导致 GPU 内存冲突
        _app_state = None
        try:
            from core.app_state import app_state
            _app_state = app_state
            if not app_state.try_start_training():
                logger.warning("其他训练正在进行，无法启动蒸馏训练")
                return {"status": "skipped", "error": "其他训练正在进行"}
        except ImportError:
            pass

        cfg = self.distill_config
        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.AdamW(
            student_model.parameters(), lr=cfg["learning_rate"],
            weight_decay=0.01, betas=(0.9, 0.999)
        )

        # 学习率调度：warmup + cosine decay
        total_steps = cfg["epochs"] * max(len(training_data) // (cfg["batch_size"] * cfg.get("grad_accum", 1)), 1)
        warmup_steps = min(cfg.get("warmup_steps", 50), total_steps // 4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["learning_rate"],
            total_steps=max(total_steps, 1),
            pct_start=warmup_steps / max(total_steps, 1) if total_steps > 0 else 0.1,
        )

        step = 0
        final_loss = float("inf")
        best_loss = float("inf")
        patience_counter = 0
        grad_accum = cfg.get("grad_accum", 1)
        max_length = cfg.get("max_length", 256)
        
        try:
            for epoch in range(cfg["epochs"]):
                for i in range(0, len(training_data), cfg["batch_size"]):
                    batch = training_data[i:i + cfg["batch_size"]]

                    # 编码
                    texts = []
                    for sample in batch:
                        try:
                            if isinstance(sample, dict):
                                if "messages" in sample:
                                    # 对话格式（兼容 content 为 str 或 list）
                                    parts = []
                                    for m in sample["messages"]:
                                        if m.get("role") == "system":
                                            continue
                                        c = m.get("content", "")
                                        if isinstance(c, list):
                                            # 多模态格式：提取文本部分
                                            c = " ".join(
                                                item.get("text", "") if isinstance(item, dict) else str(item)
                                                for item in c
                                            )
                                        parts.append(str(c))
                                    text = " ".join(parts)
                                elif "task" in sample:
                                    # ReAct 格式
                                    text = str(sample["task"])
                                else:
                                    text = str(sample)
                            else:
                                text = str(sample)
                            if text.strip():
                                texts.append(text)
                        except Exception:
                            continue

                    if not texts:
                        continue
                    
                    # Tokenize
                    try:
                        inputs = tokenizer(
                            texts[0] if len(texts) == 1 else texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                        )
                        input_ids = inputs["input_ids"].to(device)
                    except Exception as e:
                        logger.warning(f"Tokenize 失败: {e}")
                        continue

                    if input_ids.numel() == 0:
                        continue

                    # 教师前向（不计算梯度）
                    try:
                        with torch.no_grad():
                            teacher_output = teacher_model(input_ids)
                            teacher_logits = teacher_output.logits
                    except Exception as e:
                        logger.warning(f"教师前向失败: {e}")
                        continue

                    # 学生前向
                    try:
                        student_output = student_model(input_ids)
                        student_logits = student_output.logits
                    except Exception as e:
                        logger.warning(f"学生前向失败: {e}")
                        continue
                    
                    # 对齐维度（学生和教师的 vocab_size 可能不同）
                    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
                    s_logits = student_logits[..., :min_vocab]
                    t_logits = teacher_logits[..., :min_vocab]
                    
                    # 蒸馏损失（KL 散度）
                    T = cfg["temperature"]
                    distill_loss = F.kl_div(
                        F.log_softmax(s_logits / T, dim=-1),
                        F.softmax(t_logits / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                    
                    # 原始任务损失（next token prediction）
                    targets = input_ids[:, 1:].contiguous()
                    shift_logits = s_logits[:, :-1, :].contiguous()
                    task_loss = F.cross_entropy(
                        shift_logits.view(-1, min_vocab),
                        targets.view(-1),
                        ignore_index=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id else 0,
                    )
                    
                    # 总损失
                    alpha = cfg["alpha"]
                    loss = (alpha * distill_loss + (1 - alpha) * task_loss) / grad_accum
                    loss.backward()

                    # 梯度累积
                    if (step + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    final_loss = loss.item() * grad_accum
                    step += 1

                    if progress_callback:
                        progress_callback(step, total_steps, final_loss)

                    if step % 10 == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"蒸馏进度: {step}/{total_steps} "
                            f"loss={final_loss:.4f}"
                        )
            
            return {
                "status": "success",
                "final_loss": final_loss,
                "total_steps": step,
                "config": cfg,
            }
        
        except Exception as e:
            logger.error(f"知识蒸馏失败: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "final_loss": final_loss,
                "completed_steps": step,
            }
        finally:
            if _app_state is not None:
                _app_state.finish_training()


class AutoUpgrader:
    """
    M6: 完整自动升级闭环
    整合瓶颈检测 → 硬件扫描 → 知识蒸馏 → 热切换
    """
    
    # 升级路线表（纯态极 ModelSelf 架构）
    UPGRADE_TABLE = {
        "125M": {"next": "350M", "hidden_size": 1024, "layers": 24, "heads": 16, "min_ram": 4},
        "350M": {"next": "1B",   "hidden_size": 2048, "layers": 22, "heads": 32, "min_ram": 8},
        "1B":   {"next": "3B",   "hidden_size": 3072, "layers": 26, "heads": 32, "min_ram": 16},
        "3B":   {"next": "7B",   "hidden_size": 4096, "layers": 32, "heads": 32, "min_ram": 24},
        "7B":   {"next": None},
    }
    
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or ""
        self.capability_evaluator = CapabilityEvaluator(
            save_path=os.path.join(save_dir, "capability_scores.json") if save_dir else None
        )
        self.bottleneck_detector = BottleneckDetector(
            save_path=os.path.join(save_dir, "bottleneck_history.json") if save_dir else None
        )
        self.distiller = KnowledgeDistiller()
        
        # 升级状态
        self.upgrade_state = "idle"  # idle / checking / distilling / switching / done / error
        self.upgrade_progress = 0    # 0~100
        self.upgrade_message = ""
        self.upgrade_error = ""
        
        # 上次升级检查时间
        self._last_check = 0
    
    def check_and_suggest_upgrade(self, hardware_info=None) -> dict:
        """
        检查是否需要升级，并给出建议
        返回升级建议，不自动执行
        """
        from taiji.config import ModelConfig
        
        # 当前模型大小（默认125M）
        current_size = self._detect_current_size()
        
        # 硬件检查
        if hardware_info is None:
            try:
                from core.hardware import analyze_hardware
                hardware_info = analyze_hardware()
            except Exception:
                hardware_info = None
        
        available_ram = 8
        available_vram = 0
        if hardware_info:
            available_ram = getattr(hardware_info, 'available_memory_gb', 8) or 8
            available_vram = getattr(hardware_info, 'vram_gb', 0) or 0
        
        # 瓶颈检测
        bottleneck = self.bottleneck_detector.detect_bottleneck(
            self.capability_evaluator
        )
        
        # 升级路线
        route = self.UPGRADE_TABLE.get(current_size)
        if not route or not route.get("next"):
            return {
                "should_upgrade": False,
                "reason": "已达到最大支持模型",
                "bottleneck": bottleneck,
                "current_size": current_size,
            }
        
        next_size = route["next"]
        next_info = self.UPGRADE_TABLE.get(next_size, {})
        min_ram = next_info.get("min_ram", 999)
        
        # 检查硬件是否支持
        can_gpu = available_vram >= min_ram * 0.5
        can_cpu = available_ram >= min_ram * 0.6
        can_upgrade = can_gpu or can_cpu
        
        if not can_upgrade:
            return {
                "should_upgrade": False,
                "reason": f"硬件不足（需要 {min_ram}GB，可用 {available_ram:.1f}GB）",
                "bottleneck": bottleneck,
                "current_size": current_size,
                "next_size": next_size,
            }
        
        should = bottleneck.get("bottleneck", False) or bottleneck.get("trigger_count", 0) >= 1
        
        return {
            "should_upgrade": should,
            "bottleneck": bottleneck,
            "current_size": current_size,
            "next_size": next_size,
            "upgrade_mode": "gpu" if can_gpu else "cpu_quantized",
            "hardware": {
                "ram_gb": round(available_ram, 1),
                "vram_gb": round(available_vram, 1),
            },
            "config": {
                "hidden_size": next_info.get("hidden_size", 768),
                "layers": next_info.get("layers", 12),
                "heads": next_info.get("heads", 12),
            },
            "message": (
                f"建议从 {current_size} 升级到 {next_size} "
                f"({'GPU模式' if can_gpu else 'CPU量化模式'})"
            ),
        }
    
    def start_upgrade(
        self,
        teacher_model,
        teacher_tokenizer,
        training_data: list,
        device: str = "cpu",
        progress_callback=None,
    ) -> dict:
        """
        执行自动升级（纯态极 ModelSelf 架构）

        Args:
            teacher_model: 当前模型（教师）
            teacher_tokenizer: 当前分词器
            training_data: 训练数据
            device: 设备
            progress_callback: 进度回调
        """
        import torch

        current_size = self._detect_current_size()
        route = self.UPGRADE_TABLE.get(current_size, {})
        if not route.get("next"):
            return {"status": "error", "message": "无法继续升级"}

        try:
            self.upgrade_state = "creating"
            self.upgrade_message = f"正在创建 {route['next']} 模型架构..."
            self.upgrade_progress = 5

            return self._upgrade_taiji(
                teacher_model, teacher_tokenizer, training_data,
                device, route, current_size, progress_callback
            )

        except Exception as e:
            logger.error(f"自动升级失败: {e}")
            self.upgrade_state = "error"
            self.upgrade_error = str(e)
            return {"status": "error", "message": str(e)}

    def _upgrade_taiji(
        self, teacher_model, teacher_tokenizer, training_data,
        device, route, current_size, progress_callback
    ) -> dict:
        """旧 ModelSelf 架构升级路径（向后兼容）"""
        import torch
        from taiji.config import ModelConfig
        from taiji.architecture import ModelSelf

        new_config = ModelConfig(
            hidden_size=route["hidden_size"],
            num_hidden_layers=route["layers"],
            num_attention_heads=route["heads"],
            num_key_value_heads=route["heads"],
            intermediate_size=route["hidden_size"] * 4,
        )
        new_model = ModelSelf(new_config)

        logger.info(f"已创建新模型: {route['next']} ({new_config.count_parameters()/1e6:.0f}M)")

        self.upgrade_state = "distilling"
        self.upgrade_message = "正在执行知识蒸馏..."
        self.upgrade_progress = 10

        def distill_progress(step, total, loss):
            pct = 10 + int(step / max(total, 1) * 70)
            self.upgrade_progress = min(pct, 80)
            self.upgrade_message = f"蒸馏训练中... loss={loss:.4f} ({step}/{total})"
            if progress_callback:
                progress_callback(self.upgrade_progress, self.upgrade_message)

        distill_result = self.distiller.distill(
            teacher_model=teacher_model,
            student_model=new_model,
            tokenizer=teacher_tokenizer,
            training_data=training_data,
            device=device,
            progress_callback=distill_progress,
        )

        if distill_result["status"] != "success":
            self.upgrade_state = "error"
            self.upgrade_error = distill_result.get("error", "蒸馏失败")
            return {"status": "error", "message": self.upgrade_error}

        self.upgrade_progress = 85
        self.upgrade_message = "蒸馏完成，正在保存新模型..."

        save_path = os.path.join(self.save_dir, "upgraded_models", f"taiji_{route['next']}")
        os.makedirs(save_path, exist_ok=True)

        # 保存模型权重
        torch.save({
            "model_state_dict": new_model.state_dict(),
            "config": {
                "hidden_size": new_config.hidden_size,
                "num_hidden_layers": new_config.num_hidden_layers,
                "num_attention_heads": new_config.num_attention_heads,
                "num_key_value_heads": new_config.num_key_value_heads,
                "intermediate_size": new_config.intermediate_size,
                "vocab_size": new_config.vocab_size,
            },
            "distill_result": distill_result,
            "upgraded_from": current_size,
            "upgraded_to": route["next"],
            "timestamp": time.time(),
        }, os.path.join(save_path, "model.pt"))

        # 保存 config.json（态极加载器需要）
        import json
        config_dict = {
            "hidden_size": new_config.hidden_size,
            "num_hidden_layers": new_config.num_hidden_layers,
            "num_attention_heads": new_config.num_attention_heads,
            "num_key_value_heads": new_config.num_key_value_heads,
            "intermediate_size": new_config.intermediate_size,
            "vocab_size": new_config.vocab_size,
            "base_vocab_size": 32000,
            "num_special_tokens": 1000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 4096,
        }
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # 复制分词器
        import shutil
        src_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer")
        dst_tokenizer = os.path.join(save_path, "tokenizer")
        if os.path.exists(src_tokenizer) and not os.path.exists(dst_tokenizer):
            shutil.copytree(src_tokenizer, dst_tokenizer)

        self.upgrade_progress = 95
        eval_result = self._evaluate_model(new_model, teacher_tokenizer, training_data[:10])

        self.upgrade_state = "done"
        self.upgrade_progress = 100
        self.upgrade_message = f"升级完成！新模型: {route['next']}"

        return {
            "status": "success",
            "from_size": current_size,
            "to_size": route["next"],
            "save_path": save_path,
            "distill_result": distill_result,
            "eval_result": eval_result,
            "message": self.upgrade_message,
        }

    def _detect_current_size(self) -> str:
        """检测当前态极模型大小（纯 ModelSelf 架构）"""
        try:
            from core.app_state import app_state
            if app_state.model is not None:
                total_params = sum(p.numel() for p in app_state.model.parameters())
                if total_params >= 3e9:
                    return "7B"
                elif total_params >= 1e9:
                    return "3B"
                elif total_params >= 500e6:
                    return "1B"
                elif total_params >= 200e6:
                    return "350M"
                else:
                    return "125M"
        except Exception:
            pass
        return "125M"
    
    def _evaluate_model(self, model, tokenizer, eval_data: list) -> dict:
        """简单评估新模型"""
        import torch
        
        model.eval()
        total_loss = 0
        count = 0
        
        try:
            with torch.no_grad():
                for sample in eval_data[:5]:
                    try:
                        if isinstance(sample, dict) and "messages" in sample:
                            parts = []
                            for m in sample["messages"]:
                                if m.get("role") == "system":
                                    continue
                                c = m.get("content", "")
                                if isinstance(c, list):
                                    c = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in c)
                                parts.append(str(c))
                            text = " ".join(parts)
                        elif isinstance(sample, dict) and "task" in sample:
                            text = str(sample["task"])
                        else:
                            continue
                        if not text.strip():
                            continue
                    except Exception:
                        continue
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                    input_ids = inputs["input_ids"]
                    output = model(input_ids)
                    
                    targets = input_ids[:, 1:].contiguous()
                    logits = output.logits[:, :-1, :].contiguous()
                    
                    min_v = min(logits.size(-1), 33000)
                    loss = torch.nn.functional.cross_entropy(
                        logits[..., :min_v].view(-1, min_v),
                        targets.view(-1),
                        ignore_index=0,
                    )
                    total_loss += loss.item()
                    count += 1
            
            avg_loss = total_loss / max(count, 1)
            return {
                "avg_loss": round(avg_loss, 4),
                "perplexity": round(2.718 ** avg_loss, 2),
                "eval_samples": count,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self) -> dict:
        """获取升级引擎状态"""
        return {
            "upgrade_state": self.upgrade_state,
            "upgrade_progress": self.upgrade_progress,
            "upgrade_message": self.upgrade_message,
            "upgrade_error": self.upgrade_error,
            "capability": self.capability_evaluator.get_status(),
            "current_size": self._detect_current_size(),
        }