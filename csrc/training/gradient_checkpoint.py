"""
梯度检查点 + 选择性激活重计算

参考: Megatron-LM v3 Activation Checkpointing

梯度检查点: 不保存中间激活值，反向传播时重新计算
  - 显存节省: 50-70%
  - 计算代价: ~30% 额外前向计算

选择性重计算 (Megatron-LM v3):
  - 重计算「计算量小但显存大」的操作 (attention scores)
  - 保存「计算量大但显存小」的操作 (FFN 中间结果)
  - 比标准梯度检查点少 ~15% 计算量
"""

import torch
from typing import Callable, Optional


class GradientCheckpointing:
    """
    梯度检查点管理器。
    
    用法:
        ckpt = GradientCheckpointing(model)
        ckpt.enable()
        
        # 训练循环中正常使用
        loss = model(input_ids).loss
        loss.backward()  # 自动在反向时重算激活值
    """

    def __init__(self, model: torch.nn.Module, selective: bool = True):
        self.model = model
        self.selective = selective
        self._enabled = False
        self._original_forwards = {}

    def enable(self):
        """启用梯度检查点"""
        if self._enabled:
            return

        # 使用 PyTorch 内置的 gradient checkpointing
        # 后续 Phase 可以替换为自定义实现
        for name, module in self.model.named_modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

        self._enabled = True

    def disable(self):
        """禁用梯度检查点"""
        if not self._enabled:
            return

        for name, module in self.model.named_modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False

        self._enabled = False

    @staticmethod
    def checkpoint_forward(
        forward_fn: Callable,
        *args,
        use_reentrant: bool = False,
        **kwargs
    ):
        """
        对单个模块应用梯度检查点。

        Args:
            forward_fn: 前向函数
            *args, **kwargs: 前向函数的参数
            use_reentrant: 是否使用 reentrant 模式
        
        Returns:
            前向输出
        """
        return torch.utils.checkpoint.checkpoint(
            forward_fn,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )


class SelectiveRecompute:
    """
    选择性激活重计算 (Megatron-LM v3)。
    
    策略:
      - 重计算: Attention scores (计算量小，显存 O(seq²))
      - 保存: FFN 中间结果 (计算量大，显存 O(seq × hidden))
    """

    @staticmethod
    def should_recompute(layer_idx: int, op_name: str) -> bool:
        """
        判断某个操作是否应该重计算。
        
        Args:
            layer_idx: 层索引
            op_name: 操作名称 ("attention", "ffn", "norm")
        
        Returns:
            True = 重计算（不保存激活值）
        """
        # Attention scores: 重计算（显存大，计算量小）
        if op_name == "attention_scores":
            return True
        # FFN 中间结果: 保存（显存小，计算量大）
        if op_name == "ffn_intermediate":
            return False
        # Norm 输出: 重计算（计算量极小）
        if op_name == "norm":
            return True
        return False

    @staticmethod
    def selective_checkpoint(
        forward_fn: Callable,
        x: torch.Tensor,
        op_name: str,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        选择性梯度检查点。
        
        如果 op_name 对应的操作应该重计算，则使用 checkpoint；
        否则直接调用 forward（保存激活值）。
        """
        if SelectiveRecompute.should_recompute(layer_idx, op_name):
            return torch.utils.checkpoint.checkpoint(
                forward_fn, x, use_reentrant=False
            )
        else:
            return forward_fn(x)