"""
Taiji Brain Architecture — Multi-Head Model v2

The core neural architecture of Taiji (态极), a natively trained AI life form.
This is NOT a fine-tuned pre-trained model — Taiji's backbone is trained from scratch.

Multi-head architecture:
- Language Head: text generation (thought, answer)
- Tool Head: tool selection + structured argument prediction (v2)
- Perception Head: environment state encoding
- Memory Head: attention-based retrieval + read/write gating (v2)
- Plan Head: action classification + hierarchical sub-goal prediction (v2)
- Shared Transformer backbone (ModelSelfBackbone)

v2 upgrades:
  1. ToolHead: pure classification → classification + structured argument prediction
  2. MemoryHead: MLP classification → Cross-Attention differentiable retrieval
  3. PlanHead: 8-class classification → action + steps + sub-goals + difficulty
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Dict, List
from dataclasses import dataclass

import logging
from .config import ModelConfig, SPECIAL_TOKENS
from .layers import RMSNorm, TransformerBlock

logger = logging.getLogger("ModelSelf")


@dataclass
class ModelOutput:
    """模型输出 — 支持多头"""
    logits: torch.Tensor              # 语言头输出 [batch, seq, vocab]
    tool_logits: Optional[torch.Tensor] = None      # 工具头输出 [batch, num_tools]
    perception_logits: Optional[torch.Tensor] = None # 感知头输出 [batch, env_tokens]
    memory_logits: Optional[torch.Tensor] = None     # 记忆头输出 [batch, mem_slots]
    plan_logits: Optional[torch.Tensor] = None       # 规划头输出 [batch, plan_actions]
    loss: Optional[torch.Tensor] = None
    kv_cache: Optional[list] = None


class ModelSelfBackbone(nn.Module):
    """
    共享 Transformer Backbone
    处理输入序列，输出隐藏状态
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 词嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            tokens: [batch, seq_len]
            kv_cache: KV 缓存列表
            use_cache: 是否返回 KV Cache
        Returns:
            hidden: [batch, seq_len, hidden_size]
            new_kv_cache: 新的 KV 缓存
        """
        bsz, seqlen = tokens.shape

        # 词嵌入 (带缩放)
        h = self.embedding(tokens) * math.sqrt(self.config.hidden_size)

        # 因果掩码 — 使用 triu 一次性生成，避免逐行循环
        start_pos = kv_cache[0][0].shape[1] if kv_cache is not None else 0
        total_len = start_pos + seqlen

        # 构建 [seqlen, total_len] 的因果掩码
        # 对于有 KV Cache 的情况: 每个 query 位置 i 可以看到 [0, start_pos+i] 范围
        mask = torch.full((seqlen, total_len), float("-inf"),
                          device=tokens.device, dtype=h.dtype)
        # 使用 triu 置零：允许看到当前位置及之前的 token
        # row i 对应 query position start_pos+i，可以 attend 到 [0, start_pos+i]
        row_indices = torch.arange(seqlen, device=tokens.device).unsqueeze(1)  # [seq, 1]
        col_indices = torch.arange(total_len, device=tokens.device).unsqueeze(0)  # [1, total]
        # col_indices <= start_pos + row_indices 表示可以 attend
        causal_mask = col_indices <= (start_pos + row_indices)
        mask.masked_fill_(causal_mask, 0.0)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total]

        # 逐层处理
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            h, layer_new_kv = layer(h, mask, layer_kv, use_cache)
            new_kv_cache.append(layer_new_kv)

        h = self.norm(h)
        return h, new_kv_cache if use_cache else None


class MultimodalProjector(nn.Module):
    """
    多模态投影层 — 将视觉/音频编码器的输出映射到 Transformer 的嵌入空间

    这是实现"端到端多模态理解"的关键组件。
    类似 LLaVA / Qwen-VL 的做法：
    图像 → CLIP → 投影层 → 注入 Transformer 序列

    当前版本：2 层 MLP + GELU（标准做法）
    未来可升级：Cross-Attention 注入（更强但更复杂）
    """
    def __init__(self, encoder_dim: int, hidden_size: int, num_tokens: int = 1):
        super().__init__()
        self.num_tokens = num_tokens
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: 编码器输出 [batch, encoder_dim] 或 [batch, seq, encoder_dim]
        Returns:
            projected: [batch, num_tokens, hidden_size]
        """
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)  # [batch, 1, encoder_dim]
        batch_size = encoder_output.shape[0]
        # 对序列维度取均值（如果有多帧）
        pooled = encoder_output.mean(dim=1)  # [batch, encoder_dim]
        projected = self.projector(pooled)  # [batch, hidden_size * num_tokens]
        projected = projected.view(batch_size, self.num_tokens, -1)  # [batch, num_tokens, hidden_size]
        return self.norm(projected)


class PerceptionHead(nn.Module):
    """感知头 — 理解环境状态"""
    def __init__(self, hidden_size: int, num_env_tokens: int = 200):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_env_tokens),
        )
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class MemoryHead(nn.Module):
    """
    记忆头 — 注意力检索 + 读写门控

    升级点（v2）:
    1. 原 MLP 分类 → Cross-Attention 检索：query 与所有记忆做注意力，可微分读取
    2. 读写门控：sigmoid 门决定读/写/遗忘
    3. 内容寻址：按语义相似度检索，而非按槽位 ID 索引

    设计：
    - 记忆存储: [num_slots, memory_dim] 的可学习矩阵
    - 读操作: query = W_q(hidden), 与记忆矩阵做 attention → 读出向量
    - 写操作: gate = sigmoid(W_g(hidden))，决定写入哪些槽位
    """

    def __init__(self, hidden_size: int, num_slots: int = 30, memory_dim: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim

        # 可学习的记忆矩阵（初始化为小随机值）
        self.memory_keys = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.02)

        # 查询投影: hidden → query
        self.query_proj = nn.Linear(hidden_size, memory_dim, bias=False)

        # 读门控: 决定读出多少信息
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # 写门控: 决定写入哪些槽位（multi-label）
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_slots),
        )

        # 值投影: hidden → 写入值
        self.value_proj = nn.Linear(hidden_size, memory_dim, bias=False)

        # 输出投影: 读出向量 → hidden_size（与主隐藏状态兼容）
        self.output_proj = nn.Linear(memory_dim, hidden_size, bias=False)

    def read(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力读取记忆。

        Args:
            hidden: [batch, hidden_size]

        Returns:
            read_out: [batch, hidden_size] — 读出的记忆向量
            attention_weights: [batch, num_slots] — 注意力权重
        """
        query = self.query_proj(hidden)  # [batch, memory_dim]

        # 注意力: query × keys^T
        scores = torch.matmul(query, self.memory_keys.t())  # [batch, num_slots]
        scores = scores / (self.memory_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_slots]

        # 加权读取
        read_raw = torch.matmul(attn_weights, self.memory_values)  # [batch, memory_dim]

        # 门控: 根据 hidden 和读出内容决定保留多少
        gate_input = torch.cat([hidden, read_raw], dim=-1)
        gate = self.read_gate(gate_input)  # [batch, 1]

        read_out = self.output_proj(read_raw * gate)  # [batch, hidden_size]
        return read_out, attn_weights

    def get_write_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        获取写入 logits（哪些槽位应被写入）。

        Args:
            hidden: [batch, hidden_size]

        Returns:
            write_logits: [batch, num_slots] — 每个槽位的写入 logit
        """
        return self.write_gate(hidden)

    def get_write_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        获取要写入的值向量。

        Args:
            hidden: [batch, hidden_size]

        Returns:
            value: [batch, memory_dim] — 要写入的向量
        """
        return self.value_proj(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        向后兼容接口：返回读出向量 + 槽位分数。

        Returns:
            memory_logits: [batch, num_slots] — 兼容旧接口的槽位分数
        """
        read_out, attn_weights = self.read(hidden)
        # 返回注意力权重作为 "槽位分数"（向后兼容）
        return attn_weights


class PlanHead(nn.Module):
    """
    规划头 — 分层规划：动作 + 步骤数 + 子目标预测

    升级点（v2）:
    1. 保留原始 8 动作分类（向后兼容）
    2. 新增 steps_predictor：预测"这个任务需要几步"（回归）
    3. 新增 subgoal_head：预测子目标 embedding（用于分层分解）
    4. 新增 difficulty_head：评估任务难度（指导是否需要详细规划）

    设计思路：
    - 动作头: 保持 8 类分类不变
    - 步数头: 回归预测 [1, max_steps] 的连续值
    - 子目标头: 输出 N 个子目标向量，每个向量可解码为子任务描述
    - 难度头: sigmoid 输出 0~1 难度分，高难度触发更详细的规划
    """

    MAX_SUBGOALS = 5  # 最多预测 5 个子目标

    def __init__(self, hidden_size: int, num_actions: int = 8):
        super().__init__()
        self.hidden_size = hidden_size

        # ── 动作分类（向后兼容）──
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_actions),
        )

        # ── 步数预测 ──
        self.steps_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus(),  # 保证输出 > 0
        )

        # ── 子目标预测 ──
        self.subgoal_proj = nn.Linear(hidden_size, hidden_size // 4, bias=False)
        # 生成 N 个子目标的 query
        self.subgoal_queries = nn.Parameter(
            torch.randn(self.MAX_SUBGOALS, hidden_size // 4) * 0.02
        )

        # ── 难度评估 ──
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        return_subgoals: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [batch, hidden_size]
            return_subgoals: 是否返回子目标（训练时用）

        Returns:
            {
                "plan_logits": [batch, num_actions],        # 动作分类
                "predicted_steps": [batch, 1],               # 预测步数
                "difficulty": [batch, 1],                    # 难度分
                "subgoals": [batch, MAX_SUBGOALS, hidden//4] # 子目标向量（可选）
            }
        """
        plan_logits = self.mlp(hidden)
        predicted_steps = self.steps_predictor(hidden)  # [batch, 1]
        difficulty = self.difficulty_head(hidden)  # [batch, 1]

        result = {
            "plan_logits": plan_logits,
            "predicted_steps": predicted_steps,
            "difficulty": difficulty,
        }

        if return_subgoals:
            # 子目标: 用 query 与 hidden 做交叉注意力
            hidden_proj = self.subgoal_proj(hidden)  # [batch, hidden//4]
            # subgoal_queries: [MAX_SUBGOALS, hidden//4]
            # 对每个子目标 query，计算与 hidden 的相似度作为权重
            subgoal_scores = torch.matmul(
                self.subgoal_queries, hidden_proj.unsqueeze(-1)
            ).squeeze(-1)  # [batch, MAX_SUBGOALS]
            subgoal_weights = F.softmax(subgoal_scores, dim=-1)
            # 加权生成子目标向量
            subgoals = torch.matmul(
                subgoal_weights.unsqueeze(1),  # [batch, 1, MAX_SUBGOALS]
                self.subgoal_queries.unsqueeze(0).expand(hidden.size(0), -1, -1),
            ).squeeze(1)  # [batch, hidden//4]
            result["subgoals"] = subgoals

        return result


class ToolHead(nn.Module):
    """
    工具头 — 工具选择 + 结构化参数预测

    升级点（v2）:
    1. 仍输出 tool_logits 用于工具分类（向后兼容）
    2. 新增 arg_predictor：从隐藏状态直接预测结构化参数，
       不再完全依赖 LM head 逐 token 生成文本再正则解析
    3. schema_encoder：将工具描述编码为向量，辅助参数预测

    参数预测采用固定槽位设计：
    - 4 个参数槽位，每个预测 (参数名 embedding, 参数类型, 参数值 embedding)
    - 覆盖绝大多数工具的参数需求（file_path, content, query, input 等）
    """

    # 常见参数类型
    ARG_TYPES = ["string", "number", "file_path", "boolean", "json"]
    NUM_ARG_SLOTS = 4  # 每个工具最多预测 4 个参数

    def __init__(self, hidden_size: int, num_tools: int):
        super().__init__()
        self.hidden_size = hidden_size

        # ── 工具分类（向后兼容）──
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_tools),
        )

        # ── 工具 schema 编码器 ──
        # 将工具描述文本编码为向量（训练时由外部注入，推理时可选）
        self.schema_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── 参数预测头 ──
        # 每个槽位独立预测参数类型
        self.arg_type_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, len(self.ARG_TYPES)),
            )
            for _ in range(self.NUM_ARG_SLOTS)
        ])
        # 参数值 embedding（解码时与 LM head 配合）
        self.arg_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, hidden_size),
            )
            for _ in range(self.NUM_ARG_SLOTS)
        ])
        # 参数存在性预测（哪些槽位有值）
        self.arg_presence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, self.NUM_ARG_SLOTS),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        schema_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [batch, hidden_size] — 最后一个 token 的隐藏状态
            schema_embeds: [batch, hidden_size] — 工具 schema 编码（可选）

        Returns:
            {
                "tool_logits": [batch, num_tools],           # 工具分类 logits
                "arg_presence": [batch, NUM_ARG_SLOTS],      # 参数存在性 logits
                "arg_types": list of [batch, num_arg_types], # 每个槽位的类型 logits
                "arg_values": list of [batch, hidden_size],  # 每个槽位的值 embedding
            }
        """
        tool_logits = self.mlp(hidden)

        # 参数存在性
        presence_logits = self.arg_presence(hidden)

        # 构建条件输入：如果有 schema，拼接；否则只用 hidden
        if schema_embeds is not None:
            schema_proj = self.schema_proj(schema_embeds)
            cond = torch.cat([hidden, schema_proj], dim=-1)  # [batch, hidden*2]
        else:
            cond = torch.cat([hidden, hidden], dim=-1)  # [batch, hidden*2]

        # 每个槽位独立预测
        arg_types = []
        arg_values = []
        for i in range(self.NUM_ARG_SLOTS):
            arg_types.append(self.arg_type_heads[i](cond))
            arg_values.append(self.arg_value_heads[i](cond))

        return {
            "tool_logits": tool_logits,
            "arg_presence": presence_logits,
            "arg_types": arg_types,
            "arg_values": arg_values,
        }


class ModelSelf(nn.Module):
    """
    ModelSelf 双头模型

    架构:
        输入 tokens
            ↓
        Embedding
            ↓
        Transformer Backbone (共享)
            ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    语言头 (LM Head)      工具头 (Tool Head)
        ↓                       ↓
    next token logits       tool logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 确定活跃头（None 或空列表 = 全部激活，向后兼容）
        active = config.active_heads
        if active is None or len(active) == 0:
            active_heads = {"language", "tool", "perception", "memory", "plan"}
        else:
            active_heads = set(active)
        self._active_heads = active_heads

        # 共享 backbone
        self.backbone = ModelSelfBackbone(config)

        # 语言头 (与 embedding 权重绑定) — 始终创建
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 工具头 (独立分类头，支持最多 750 个工具)
        self._max_tools = 750
        self._num_tools = 0
        if "tool" in active_heads:
            self.tool_head = ToolHead(config.hidden_size, self._max_tools)
        else:
            self.tool_head = None

        # 感知头
        if "perception" in active_heads:
            self.perception_head = PerceptionHead(config.hidden_size, num_env_tokens=200)
        else:
            self.perception_head = None

        # 记忆头: 20 短期 + 10 长期 = 30 槽
        if "memory" in active_heads:
            self.memory_head = MemoryHead(config.hidden_size, num_slots=30)
        else:
            self.memory_head = None

        # 规划头: 8 种规划动作
        if "plan" in active_heads:
            self.plan_head = PlanHead(config.hidden_size, num_actions=8)
        else:
            self.plan_head = None

        # 多模态投影层（视觉/音频 → Transformer 嵌入空间）
        # CLIP 输出维度 512，Whisper 输出维度 384
        self.vision_projector = MultimodalProjector(
            encoder_dim=512, hidden_size=config.hidden_size, num_tokens=4
        )
        self.audio_projector = MultimodalProjector(
            encoder_dim=384, hidden_size=config.hidden_size, num_tokens=4
        )

        # 权重绑定
        self.lm_head.weight = self.backbone.embedding.weight

        # 初始化
        self.apply(self._init_weights)

        logger.info(f"ModelSelf active heads: {sorted(active_heads)}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_num_tools(self, num_tools: int):
        """更新活跃工具数量（工具头始终按最大容量初始化，不重建）"""
        if self.tool_head is None:
            logger.warning("工具头未激活，无法设置工具数量")
            return
        if num_tools > self._max_tools:
            logger.warning(
                f"工具数量 {num_tools} 超过最大容量 {self._max_tools}，"
                f"多余工具将无法使用工具头"
            )
        self._num_tools = min(num_tools, self._max_tools)

    @property
    def num_tools(self) -> int:
        return self._num_tools

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        tool_targets: Optional[torch.Tensor] = None,
        tool_head_active: bool = False,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
        force_tool_head: bool = False,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            tokens: 输入 token [batch, seq_len]
            targets: 语言目标 token [batch, seq_len] (训练时)
            tool_targets: 工具目标 [batch] (训练时，工具头激活时)
            tool_head_active: 是否激活工具头
            kv_cache: KV 缓存
            use_cache: 是否返回 KV Cache
            vision_features: CLIP 视觉特征 [batch, 512]（可选）
            audio_features: Whisper 音频特征 [batch, 384]（可选）
        """
        # 如果有多模态特征，注入到序列中
        if vision_features is not None or audio_features is not None:
            hidden = self._forward_with_multimodal(
                tokens, vision_features, audio_features, kv_cache, use_cache
            )
            new_kv_cache = None  # 多模态模式下不使用缓存（简化）
        else:
            # 纯文本模式
            hidden, new_kv_cache = self.backbone(tokens, kv_cache, use_cache)

        # 语言头: 所有位置
        logits = self.lm_head(hidden)

        # 工具头: 只取最后一个 token（v2: 返回 dict，提取 tool_logits）
        tool_logits = None
        tool_arg_outputs = None
        if (tool_head_active or force_tool_head) and self.tool_head is not None and self._num_tools > 0:
            last_hidden = hidden[:, -1, :]  # [batch, hidden_size]
            tool_outputs = self.tool_head(last_hidden)  # dict
            full_tool_logits = tool_outputs["tool_logits"]  # [batch, max_tools]
            # 只返回活跃工具范围内的 logits
            tool_logits = full_tool_logits[:, :self._num_tools]
            # 保存参数预测输出（推理时使用）
            tool_arg_outputs = {
                "arg_presence": tool_outputs["arg_presence"],
                "arg_types": tool_outputs["arg_types"],
                "arg_values": tool_outputs["arg_values"],
            }

        # 计算损失
        loss = None
        if targets is not None or tool_targets is not None:
            loss = self._compute_loss(
                logits, targets, tool_logits, tool_targets, tool_head_active
            )

        # 感知头、记忆头、规划头: 仅在推理时由引擎调用，不在 forward 中自动计算
        # 避免不必要的计算开销，由 NativeAgentEngine 按需调用

        return ModelOutput(
            logits=logits,
            tool_logits=tool_logits,
            loss=loss,
            kv_cache=new_kv_cache,
        )

    def get_agent_outputs(self, hidden: torch.Tensor) -> Dict[str, Any]:
        """
        获取所有 Agent 头的输出（推理时按需调用）。
        未激活的头返回 None，不抛异常。
        """
        last_hidden = hidden[:, -1, :]
        result = {}

        if self.perception_head is not None:
            result["perception"] = self.perception_head(last_hidden)
        else:
            result["perception"] = None

        if self.memory_head is not None:
            mem_out, mem_attn = self.memory_head.read(last_hidden)
            result["memory_read"] = mem_out
            result["memory_attn"] = mem_attn
            result["memory_write_logits"] = self.memory_head.get_write_logits(last_hidden)
        else:
            result["memory_read"] = None
            result["memory_attn"] = None
            result["memory_write_logits"] = None

        if self.plan_head is not None:
            result["plan"] = self.plan_head(last_hidden)
        else:
            result["plan"] = None

        return result

    def get_tool_arg_outputs(
        self,
        hidden: torch.Tensor,
        schema_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        获取工具参数预测（推理时按需调用）。
        工具头未激活时返回空 dict。
        """
        if self.tool_head is None:
            return {}
        return self.tool_head(hidden, schema_embeds)

    def _forward_with_multimodal(
        self,
        tokens: torch.Tensor,
        vision_features: Optional[torch.Tensor],
        audio_features: Optional[torch.Tensor],
        kv_cache: Optional[list],
        use_cache: bool,
    ) -> torch.Tensor:
        """
        多模态前向传播：将视觉/音频特征注入 Transformer 序列。

        流程：
        1. 文本 tokens → 文本 embedding
        2. 视觉特征 → 投影 → 视觉 embedding
        3. 音频特征 → 投影 → 音频 embedding
        4. 拼接：[视觉 tokens] + [音频 tokens] + [文本 tokens]
        5. 送入 Transformer backbone
        """
        # 文本 embedding
        text_embeds = self.backbone.embedding(tokens)  # [batch, text_len, hidden]
        batch_size = text_embeds.shape[0]

        # 收集所有 embedding 段
        segments = []

        # 视觉 embedding
        if vision_features is not None and hasattr(self, 'vision_projector'):
            vision_embeds = self.vision_projector(vision_features)  # [batch, 4, hidden]
            segments.append(vision_embeds)

        # 音频 embedding
        if audio_features is not None and hasattr(self, 'audio_projector'):
            audio_embeds = self.audio_projector(audio_features)  # [batch, 4, hidden]
            segments.append(audio_embeds)

        # 文本 embedding
        segments.append(text_embeds)

        # 拼接所有段
        combined = torch.cat(segments, dim=1)  # [batch, total_len, hidden]

        # 位置编码需要调整（序列长度变了）
        # 创建新的位置 ID
        seq_len = combined.shape[1]
        position_ids = torch.arange(seq_len, device=combined.device).unsqueeze(0).expand(batch_size, -1)

        # 通过 Transformer 层（手动调用，不使用 backbone 的 forward 因为它处理的是 token IDs）
        hidden = combined
        for layer in self.backbone.layers:
            hidden = layer(hidden, position_ids=position_ids)

        hidden = self.backbone.final_norm(hidden)

        # 截取文本部分的 hidden state（用于语言头和工具头）
        # 视觉/音频 tokens 在前面，文本 tokens 在后面
        text_start = seq_len - tokens.shape[1]
        hidden = hidden[:, text_start:, :]  # 只保留文本部分

        return hidden

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        tool_logits: Optional[torch.Tensor],
        tool_targets: Optional[torch.Tensor],
        tool_head_active: bool,
    ) -> torch.Tensor:
        """计算联合损失"""
        total_loss = torch.tensor(0.0, device=logits.device)
        loss_count = 0

        # 语言损失
        if targets is not None:
            lang_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + lang_loss
            loss_count += 1

        # 工具损失
        if tool_head_active and tool_logits is not None and tool_targets is not None:
            tool_loss = F.cross_entropy(tool_logits, tool_targets)
            total_loss = total_loss + tool_loss
            loss_count += 1

        if loss_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_loss / loss_count

    def get_num_parameters(self) -> Dict[str, int]:
        """统计参数"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tool_head_params = (
            sum(p.numel() for p in self.tool_head.parameters())
            if self.tool_head is not None else 0
        )
        other_head_params = 0
        for head_name in ("perception_head", "memory_head", "plan_head"):
            head = getattr(self, head_name, None)
            if head is not None:
                other_head_params += sum(p.numel() for p in head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": total - tool_head_params - other_head_params,
            "tool_head": tool_head_params,
            "other_heads": other_head_params,
            "active_heads": sorted(self._active_heads),
        }

    def print_model_info(self):
        """打印模型信息"""
        params = self.get_num_parameters()
        print("\n" + "=" * 60)
        print("ModelSelf Architecture")
        print("=" * 60)
        print(self.config.describe())
        print(f"Active Heads:         {', '.join(params['active_heads'])}")
        print(f"Total Parameters:     {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"  Backbone:           {params['backbone']:,}")
        print(f"  Tool Head:          {params['tool_head']:,}")
        print(f"  Other Heads:        {params['other_heads']:,}")
        print(f"  Tools registered:   {self._num_tools}")
        print("=" * 60 + "\n")
