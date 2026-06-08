"""
M11: Multi-Token Prediction (MTP)

训练时同时预测 next 2-4 tokens，提升训练质量和推理速度。

参考: Meta "Better & Faster Large Language Models via Multi-token Prediction" (2024)

收益:
  1. 训练质量提升（更好的内部表征）
  2. 推理时可做 self-speculative decoding（加速 2-3x）
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MultiTokenPredictionHead(nn.Module):
    """
    多 Token 预测头。
    
    训练时同时预测 next 1 ~ num_mtp_tokens 个 token。
    推理时可选使用 speculative decoding。
    
    Args:
        hidden_size: 隐藏层维度
        vocab_size: 词表大小
        num_mtp_tokens: 预测的 token 数 (1=标准, 2-4=多token)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_mtp_tokens: int = 4,
    ):
        super().__init__()
        self.num_mtp_tokens = num_mtp_tokens
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 主预测头 (next token 1)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 额外预测头 (next token 2, 3, ...)
        if num_mtp_tokens > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_mtp_tokens - 1)
            ])
            # 对齐投影: 将 hidden state 投影到下一个位置的表示
            self.mtp_proj = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=False)
                for _ in range(num_mtp_tokens - 1)
            ])
        else:
            self.mtp_heads = nn.ModuleList()
            self.mtp_proj = nn.ModuleList()

        # 训练权重: 越远的 token 权重越小
        self.register_buffer(
            "mtp_weights",
            torch.tensor([1.0 / (2 ** i) for i in range(num_mtp_tokens)]),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden: [batch, seq_len, hidden_size] — backbone 输出
            targets: [batch, seq_len] — 训练目标 (可选)
        
        Returns:
            如果 targets=None: logits [batch, seq_len, vocab_size] (主头)
            如果 targets 提供: (logits, mtp_loss) 元组
        """
        batch, seq_len, _ = hidden.shape

        # 主预测
        logits = self.lm_head(hidden)  # [batch, seq, vocab]

        if targets is None or self.num_mtp_tokens <= 1:
            return logits

        # 多 token 预测损失
        mtp_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        total_weight = 0.0

        for i, (head, proj) in enumerate(zip(self.mtp_heads, self.mtp_proj)):
            # 预测 next (i+2) token
            # 需要将 hidden 向右移动 i+1 位
            offset = i + 1
            if offset >= seq_len:
                break

            # 投影 + 预测
            shifted_hidden = proj(hidden[:, :seq_len - offset, :])
            mtp_logits = head(shifted_hidden)

            # 目标: targets[:, offset:]
            mtp_targets = targets[:, offset:]

            # 交叉熵损失
            loss = torch.nn.functional.cross_entropy(
                mtp_logits.reshape(-1, self.vocab_size),
                mtp_targets.reshape(-1),
                ignore_index=-100,
            )

            weight = self.mtp_weights[i + 1].item()
            mtp_loss = mtp_loss + weight * loss
            total_weight += weight

        if total_weight > 0:
            mtp_loss = mtp_loss / total_weight

        return logits, mtp_loss

    def speculative_generate(
        self,
        hidden: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Self-speculative decoding。
        
        用 MTP heads 快速生成候选 token，主头一次性验证。
        
        Args:
            hidden: [1, 1, hidden_size] — 当前 token 的隐藏状态
        
        Returns:
            candidate_ids: [num_mtp_tokens] — 候选 token IDs
        """
        candidates = []

        # 主头生成第一个 token
        logits = self.lm_head(hidden)[:, -1, :]  # [1, vocab]
        token1 = self._sample(logits, temperature, top_p)
        candidates.append(token1)

        # MTP heads 生成后续候选
        current_hidden = hidden
        for i, (head, proj) in enumerate(zip(self.mtp_heads, self.mtp_proj)):
            # 投影
            proj_hidden = proj(current_hidden[:, -1:, :])
            mtp_logits = head(proj_hidden)[:, -1, :]
            token = self._sample(mtp_logits, temperature, top_p)
            candidates.append(token)

        return torch.tensor(candidates, device=hidden.device)

    def _sample(self, logits, temperature, top_p):
        """Top-P 采样"""
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1).item()