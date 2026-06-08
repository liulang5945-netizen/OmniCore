"""
Taiji Architecture Upgrades — 模型架构前沿升级

M10: YaRN 位置编码扩展 (4K → 128K)
M11: Multi-Token Prediction heads
M12: MoE 层 (8 experts / top-2)
M13: MLA 注意力 (DeepSeek-V3 风格)
M14: Mamba 混合层 (可选)
"""

def __getattr__(name):
    if name == "YaRNRotaryEmbedding":
        from .yarn_rope import YaRNRotaryEmbedding
        return YaRNRotaryEmbedding
    elif name == "MultiTokenPredictionHead":
        from .multi_token_prediction import MultiTokenPredictionHead
        return MultiTokenPredictionHead
    elif name == "MoELayer":
        from .moe_layer import MoELayer
        return MoELayer
    elif name == "MultiHeadLatentAttention":
        from .mla_attention import MultiHeadLatentAttention
        return MultiHeadLatentAttention
    elif name == "MambaBlock":
        from .mamba_block import MambaBlock
        return MambaBlock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")