"""
混合精度训练管理器

支持 BF16/FP16 混合精度训练，包含动态 loss scaling。
参考: PyTorch AMP / NVIDIA Apex

关键优化:
  - 前向: BF16/FP16（计算快 2x，显存省 50%）
  - 反向: FP16（梯度）
  - 权重更新: FP32（保持精度）
  - 动态 Loss Scaling（防止 FP16 下溢）
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger("Taiji.MixedPrecision")


class MixedPrecisionTrainer:
    """
    混合精度训练管理器。
    
    用法:
        trainer = MixedPrecisionTrainer(model, optimizer, dtype="bf16")
        
        for batch in dataloader:
            loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        dtype: str = "bf16",           # "bf16" 或 "fp16"
        init_scale: float = 2**16,     # 初始 loss scale
        growth_factor: float = 2.0,    # scale 增长因子
        backoff_factor: float = 0.5,   # scale 缩小因子
        growth_interval: int = 2000,   # 连续无溢出步数后增大 scale
        max_scale: float = 2**24,      # 最大 scale
    ):
        self.model = model
        self.optimizer = optimizer
        self.dtype = dtype

        # 检查 BF16 支持
        if dtype == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("GPU 不支持 BF16，回退到 FP16")
            self.dtype = "fp16"

        self._dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float16

        # 动态 Loss Scaling (仅 FP16 需要)
        self.use_scaler = (self.dtype == "fp16")
        self.scaler = torch.amp.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=self.use_scaler,
        )

        # 统计
        self.step_count = 0
        self.overflow_count = 0

    def train_step(self, input_ids, targets, loss_fn=None):
        """
        执行一步混合精度训练。

        Args:
            input_ids: [batch, seq]
            targets: [batch, seq]
            loss_fn: 自定义损失函数 (logits, targets) → loss
        
        Returns:
            loss: float
        """
        self.optimizer.zero_grad()

        # 前向: 混合精度上下文
        with torch.amp.autocast(device_type="cuda", dtype=self._dtype):
            logits = self.model(input_ids)
            if loss_fn is not None:
                loss = loss_fn(logits, targets)
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )

        # 反向: 使用 GradScaler (FP16) 或直接反向 (BF16)
        if self.use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.step_count += 1
        return loss.item()

    def get_stats(self):
        """获取训练统计"""
        return {
            "dtype": self.dtype,
            "step_count": self.step_count,
            "overflow_count": self.overflow_count,
            "current_scale": self.scaler.get_scale() if self.use_scaler else 1.0,
        }