"""
Taiji Model Loader

Handles saving, loading, and initialization of Taiji (态极) models.
Taiji is a natively trained AI life form — model files use ModelSelf architecture.
"""
import os
import json
import logging
import torch
from typing import Optional, Tuple
from pathlib import Path

from .config import ModelConfig, SPECIAL_TOKENS
from .architecture import ModelSelf
from .tokenizer import ModelSelfTokenizer

logger = logging.getLogger("ModelSelf")


def load_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[ModelSelf, ModelSelfTokenizer]:
    """
    加载 ModelSelf 模型和分词器。

    Args:
        model_path: 模型目录路径 (包含 config.json, model.pt, tokenizer/)
        device: 设备
        dtype: 数据类型

    Returns:
        (model, tokenizer) 元组
    """
    model_path = os.path.abspath(model_path)

    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = _dict_to_config(config_dict)

    # 创建模型
    model = ModelSelf(config)

    # 加载权重
    weights_path = os.path.join(model_path, "model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(f"No weights found at {weights_path}, using random init")

    # 加载分词器
    tokenizer_path = os.path.join(model_path, "tokenizer")
    sp_model_path = os.path.join(model_path, "tokenizer", "sentencepiece.model")
    if not os.path.exists(sp_model_path):
        sp_model_path = None

    tokenizer = ModelSelfTokenizer.load(
        tokenizer_path if os.path.exists(tokenizer_path) else model_path,
        sp_model_path=sp_model_path,
    )

    # 初始化工具头 (如果有工具映射)
    if tokenizer._tool_name_to_id:
        model.set_num_tools(len(tokenizer._tool_name_to_id))

    # 移动到设备
    model = model.to(device=device, dtype=dtype)

    logger.info(f"Model loaded: {config.describe()}")
    return model, tokenizer


def save_model(
    model,
    tokenizer,
    save_path: str,
    training_state: dict = None,
):
    """
    保存态极模型和分词器。

    Args:
        model: ModelSelf（态极）
        tokenizer: ModelSelfTokenizer
        save_path: 保存目录
    """
    os.makedirs(save_path, exist_ok=True)
    config_dict = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_theta": model.config.rope_theta,
        "base_vocab_size": model.config.base_vocab_size,
        "num_special_tokens": model.config.num_special_tokens,
        "num_short_term_slots": model.config.num_short_term_slots,
        "num_long_term_slots": model.config.num_long_term_slots,
        "memory_dim": model.config.memory_dim,
    }
    if model.config.active_heads is not None:
        config_dict["active_heads"] = model.config.active_heads
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 保存权重（可附带训练状态）
    save_data = model.state_dict()
    if training_state:
        save_data["_training_state"] = training_state
    torch.save(save_data, os.path.join(save_path, "model.pt"))

    # 保存分词器
    tokenizer.save(os.path.join(save_path, "tokenizer"))

    logger.info(f"Model saved to {save_path}")


def create_model(
    size: str = "125m",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    active_heads: list = None,
) -> Tuple[ModelSelf, ModelSelfTokenizer]:
    """
    创建新的 ModelSelf 模型 (随机初始化)。

    Args:
        size: 模型大小 ("125m", "350m", "1b", "3b", "7b")
        device: 设备
        dtype: 数据类型
        active_heads: 活跃头列表 (如 ["language", "tool"])，None = 全部

    Returns:
        (model, tokenizer) 元组
    """
    # 获取配置
    config_map = {
        "125m": ModelConfig.size_125m,
        "350m": ModelConfig.size_350m,
        "1b": ModelConfig.size_1b,
        "3b": ModelConfig.size_3b,
        "7b": ModelConfig.size_7b,
    }
    if size not in config_map:
        raise ValueError(f"Unknown size: {size}. Available: {list(config_map.keys())}")

    config = config_map[size]()
    if active_heads is not None:
        config.active_heads = active_heads

    # 创建模型
    model = ModelSelf(config)
    model = model.to(device=device, dtype=dtype)

    # 创建分词器（自动查找 SentencePiece 模型）
    # 优先查找包内路径（兼容打包后的运行环境）
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sp_model_path = os.path.join(_pkg_dir, "tokenizer", "sentencepiece.model")
    if not os.path.exists(sp_model_path):
        # 回退：开发环境下的相对路径
        sp_model_path = os.path.join("taiji", "tokenizer", "sentencepiece.model")
    if os.path.exists(sp_model_path):
        tokenizer = ModelSelfTokenizer(sp_model_path=sp_model_path)
    else:
        tokenizer = ModelSelfTokenizer()

    logger.info(f"Created model: {config.describe()}")
    return model, tokenizer


def _dict_to_config(d: dict) -> ModelConfig:
    """字典转配置（默认值与 ModelConfig 保持一致）"""
    return ModelConfig(
        vocab_size=d.get("vocab_size", 33000),
        hidden_size=d.get("hidden_size", 768),
        intermediate_size=d.get("intermediate_size", 2048),
        num_hidden_layers=d.get("num_hidden_layers", 12),
        num_attention_heads=d.get("num_attention_heads", 12),
        num_key_value_heads=d.get("num_key_value_heads", 12),
        max_position_embeddings=d.get("max_position_embeddings", 4096),
        rms_norm_eps=d.get("rms_norm_eps", 1e-5),
        rope_theta=d.get("rope_theta", 500000.0),
        base_vocab_size=d.get("base_vocab_size", 32000),
        num_special_tokens=d.get("num_special_tokens", 1000),
        num_short_term_slots=d.get("num_short_term_slots", 20),
        num_long_term_slots=d.get("num_long_term_slots", 10),
        memory_dim=d.get("memory_dim", 64),
        active_heads=d.get("active_heads", None),
    )
