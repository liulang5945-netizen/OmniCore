"""
Taiji CUDA Kernels — 融合算子集合

M2: Triton 融合前向 kernels
  - triton_rms_norm: 融合 RMSNorm (前向+反向)
  - triton_swiglu: 融合 SwiGLU FFN (前向+反向)
  - triton_rope: 融合 RoPE 旋转编码
  - triton_softmax: 融合 Softmax

M3: 融合交叉熵 + 采样
  - fused_cross_entropy: 不 materialize 概率矩阵的交叉熵
  - fused_top_p_sampling: 融合 Top-P 采样

M7: FlashAttention
  - flash_attention: Tiling + Online Softmax

用法:
    from csrc.cuda.triton_rms_norm import triton_rms_norm
    from csrc.cuda.triton_swiglu import triton_swiglu
    from csrc.cuda.triton_rope import triton_rope
    from csrc.cuda.triton_softmax import triton_softmax
    from csrc.cuda.fused_cross_entropy import fused_cross_entropy
    from csrc.cuda.fused_top_p_sampling import fused_top_p_sample
"""

# 懒加载：只在实际使用时导入（避免 Triton 未安装时报错）
def __getattr__(name):
    if name == "triton_rms_norm":
        from .triton_rms_norm import triton_rms_norm
        return triton_rms_norm
    elif name == "triton_swiglu":
        from .triton_swiglu import triton_swiglu
        return triton_swiglu
    elif name == "triton_rope":
        from .triton_rope import triton_rope
        return triton_rope
    elif name == "triton_softmax":
        from .triton_softmax import triton_softmax
        return triton_softmax
    elif name == "fused_cross_entropy":
        from .fused_cross_entropy import fused_cross_entropy
        return fused_cross_entropy
    elif name == "fused_top_p_sample":
        from .fused_top_p_sampling import fused_top_p_sample
        return fused_top_p_sample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")