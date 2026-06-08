"""
Taiji Model Configuration

Defines the neural architecture sizes and special token vocabulary for Taiji (态极).
Taiji is a natively trained AI life form — NOT a fine-tuned pre-trained model.

Available sizes: 125M, 350M, 1B, 3B, 7B (all trained from scratch)
"""
from dataclasses import dataclass
from typing import Optional


# OmniCore 专用特殊 token IDs
SPECIAL_TOKENS = {
    # === 思考与输出 ===
    "think_start":      32000,  # <think>
    "think_end":        32001,  # </think>
    "tool_call":        32002,  # <tool_call>
    "tool_result":      32003,  # <tool_result>
    "answer":           32004,  # <final_answer>

    # === 感知系统 (32005-32009) ===
    "observe":          32005,  # <observe> — 环境观察开始
    "observe_end":      32006,  # </observe> — 环境观察结束
    "env_tree":         32007,  # <tree> — 文件树
    "env_state":        32008,  # <state> — 系统状态
    "env_result":       32009,  # <result> — 执行结果

    # === 记忆系统 (32100-32139) ===
    "mem_read":         32100,  # <mem_read> — 读取记忆
    "mem_write":        32101,  # <mem_write> — 写入记忆
    "mem_slot_base":    32102,  # 32102-32121 → 记忆槽 0-19 (短期)
    "mem_long_base":    32122,  # 32122-32131 → 长期记忆槽 0-9

    # === 规划系统 (32132-32139) ===
    "plan_start":       32132,  # <plan>
    "plan_end":         32133,  # </plan>
    "plan_step":        32134,  # <step>
    "plan_step_end":    32135,  # </step>
    "plan_done":        32136,  # <plan_done>
    "plan_replan":      32137,  # <replan> — 重新规划

    # === 反思系统 (32138-32149) ===
    "reflect_start":    32138,  # <reflect>
    "reflect_end":      32139,  # </reflect>
    "reflect_detect":   32140,  # <detect> — 错误检测
    "reflect_cause":    32141,  # <cause> — 原因分析
    "reflect_correct":  32142,  # <correct> — 纠正方案
    "reflect_confirm":  32143,  # <confirm> — 确认成功

    # === 工具名 token (32150-32899，支持最多 750 个工具) ===
    "tool_name_base":   32150,
}


def get_taiji_data_path(subdir: str) -> str:
    """
    获取态极持久化数据路径。优先存到当前模型目录下，确保数据和模型绑定。

    优先级：
      1. 当前加载的模型目录/{subdir}（如 taiji_350M/sleep_data/）
      2. 外部目录 taiji_data/{subdir}（兜底）

    这样打包不影响数据，且模型目录就是态极的"身份"。
    """
    import os

    # 优先：存到当前模型目录下
    try:
        from core.app_state import app_state
        model_path = getattr(app_state, "_loaded_model_name", "") or ""
        if model_path and os.path.isdir(model_path):
            # model_path 指向 taiji_350M/ 或 best/，用其父目录
            model_dir = model_path if os.path.exists(os.path.join(model_path, "config.json")) else os.path.dirname(model_path)
            data_path = os.path.join(model_dir, subdir)
            os.makedirs(data_path, exist_ok=True)
            return data_path
    except (ImportError, Exception):
        pass

    # 兜底：外部目录
    try:
        from core.config import get_external_path
        base = get_external_path(os.path.join("taiji_data", subdir))
    except ImportError:
        base = os.path.join(os.getcwd(), "taiji_data", subdir)
    os.makedirs(base, exist_ok=True)
    return base


@dataclass
class ModelConfig:
    """模型架构配置"""

    # === 基础参数 ===
    vocab_size: int = 33000           # 32000 基础词表 + 1000 特殊 token (感知/记忆/规划/反思/工具)
    hidden_size: int = 768            # 隐藏层维度
    intermediate_size: int = 2048     # FFN 中间层维度
    num_hidden_layers: int = 12       # Transformer 层数
    num_attention_heads: int = 12     # 注意力头数
    num_key_value_heads: int = 12     # KV 头数 (GQA)

    # === 活跃头配置 ===
    # 控制哪些多头分支实际创建和训练。未激活的头不消耗参数。
    # 可选: "language", "tool", "perception", "memory", "plan"
    active_heads: list = None         # None = 全部激活（向后兼容）

    # === 序列参数 ===
    max_position_embeddings: int = 4096

    # === 归一化 ===
    rms_norm_eps: float = 1e-5

    # === RoPE ===
    rope_theta: float = 500000.0

    # === 词表参数 ===
    base_vocab_size: int = 32000      # 基础 SentencePiece 词表大小
    num_special_tokens: int = 1000    # 特殊 token 数量 (扩展后，含 750 个工具位)

    # === 记忆系统参数 ===
    num_short_term_slots: int = 20    # 短期记忆槽数
    num_long_term_slots: int = 10     # 长期记忆槽数
    memory_dim: int = 64              # 记忆向量维度

    @classmethod
    def size_125m(cls) -> "ModelConfig":
        """125M 参数 — 轻量 Agent，适合 CPU 推理"""
        return cls(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
        )

    @classmethod
    def size_350m(cls) -> "ModelConfig":
        """350M 参数 — 标准 Agent"""
        return cls(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
        )

    @classmethod
    def size_1b(cls) -> "ModelConfig":
        """1B 参数 — 复杂推理 + Agent"""
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_3b(cls) -> "ModelConfig":
        """3B 参数 — 高级推理 + 多任务 Agent"""
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_7b(cls) -> "ModelConfig":
        """7B 参数 — 复杂多步推理 + 专业级 Agent"""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def count_parameters(self) -> int:
        """估算参数量"""
        embed = self.vocab_size * self.hidden_size
        # 每层: attention(Q+K+V+O) + FFN(W1+W_gate+W2) + 2*norm
        head_dim = self.head_dim
        attn = (
            self.hidden_size * self.num_attention_heads * head_dim +  # Q
            self.hidden_size * self.num_key_value_heads * head_dim +  # K
            self.hidden_size * self.num_key_value_heads * head_dim +  # V
            self.num_attention_heads * head_dim * self.hidden_size    # O
        )
        ffn = (
            self.hidden_size * self.intermediate_size +   # W1
            self.hidden_size * self.intermediate_size +   # W_gate
            self.intermediate_size * self.hidden_size     # W2
        )
        norm = self.hidden_size * 2  # attention_norm + ffn_norm
        per_layer = attn + ffn + norm
        return embed + per_layer * self.num_hidden_layers

    def describe(self) -> str:
        """返回人类可读的配置描述"""
        params = self.count_parameters()
        if params >= 1e9:
            size_str = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            size_str = f"{params/1e6:.0f}M"
        else:
            size_str = f"{params/1e3:.0f}K"
        return (
            f"ModelSelf-{size_str} | "
            f"hidden={self.hidden_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads} kv={self.num_key_value_heads} "
            f"ffn={self.intermediate_size} vocab={self.vocab_size}"
        )
