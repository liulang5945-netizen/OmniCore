"""
Taiji Training System — 融合训练组件

M8: 融合优化器 + 混合精度
  - FusedAdamW: 融合 AdamW 优化器
  - MixedPrecisionTrainer: BF16/FP16 混合精度训练

M9: 梯度检查点 + 选择性重计算
  - GradientCheckpointing: 梯度检查点管理器
  - SelectiveRecompute: 选择性激活重计算

用法:
    from csrc.training.fused_adamw import FusedAdamW
    from csrc.training.mixed_precision import MixedPrecisionTrainer
    from csrc.training.gradient_checkpoint import GradientCheckpointing
"""

def __getattr__(name):
    if name == "FusedAdamW":
        from .fused_adamw import FusedAdamW
        return FusedAdamW
    elif name == "MixedPrecisionTrainer":
        from .mixed_precision import MixedPrecisionTrainer
        return MixedPrecisionTrainer
    elif name == "GradientCheckpointing":
        from .gradient_checkpoint import GradientCheckpointing
        return GradientCheckpointing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")