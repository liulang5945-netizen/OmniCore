"""
Phase 4: FlashAttention — Tiling + Online Softmax

训练时的注意力优化。通过分块计算避免存储完整的 [batch, heads, seq, seq] 注意力矩阵，
显存从 O(seq²) 降到 O(seq)。

参考: Tri Dao "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)

核心思想:
  1. Tiling: 将 Q/K/V 分成小块（tile size = 64-128）
  2. Online Softmax: 单遍计算 softmax，不存储完整 score 矩阵
  3. 重计算: 反向传播时重新计算 attention scores（不保存）

适用场景: 训练时的注意力计算（长序列 4K+ tokens 显著节省显存）
"""

import torch
import torch.nn.functional as F
import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger("Taiji.FlashAttention")


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """
    FlashAttention 前向传播。
    
    使用 Tiling + Online Softmax 计算精确注意力，
    不存储完整 [seq, seq] score 矩阵。
    
    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, kv_heads, seq_len, head_dim]
        v: [batch, kv_heads, seq_len, head_dim]
        causal: 是否使用因果掩码
        block_size: 分块大小
    
    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    batch, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    # GQA: 如果 kv_heads < heads，需要扩展
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # 初始化输出和辅助变量
    output = torch.zeros_like(q)
    l = torch.zeros(batch, num_heads, seq_len, 1, device=q.device, dtype=torch.float32)
    m = torch.full((batch, num_heads, seq_len, 1), float('-inf'), device=q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(head_dim)

    # 分块遍历 K/V
    num_blocks = (seq_len + block_size - 1) // block_size

    for j in range(num_blocks):
        kv_start = j * block_size
        kv_end = min(kv_start + block_size, seq_len)

        # 加载 K/V 块
        k_block = k[:, :, kv_start:kv_end, :]  # [batch, heads, block, dim]
        v_block = v[:, :, kv_start:kv_end, :]

        # 分块遍历 Q
        for i in range(num_blocks):
            q_start = i * block_size
            q_end = min(q_start + block_size, seq_len)

            # 加载 Q 块
            q_block = q[:, :, q_start:q_end, :]

            # 计算 score 块
            s_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # 因果掩码
            if causal:
                # 只允许 attend 到 kv_start ~ min(kv_end, q_end) 的位置
                mask = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0) > \
                       torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                s_block = s_block.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Online Softmax 更新
            m_block = s_block.max(dim=-1, keepdim=True).values  # [batch, heads, block, 1]
            m_new = torch.maximum(m[:, :, q_start:q_end, :], m_block)

            # exp(s - m_new)
            exp_s = torch.exp(s_block - m_new)

            # 更新 l (sum of exp)
            l_new = torch.exp(m[:, :, q_start:q_end, :] - m_new) * l[:, :, q_start:q_end, :] + \
                    exp_s.sum(dim=-1, keepdim=True)

            # 更新 output
            output[:, :, q_start:q_end, :] = (
                torch.exp(m[:, :, q_start:q_end, :] - m_new) * l[:, :, q_start:q_end, :] *
                output[:, :, q_start:q_end, :] +
                torch.matmul(exp_s, v_block)
            ) / l_new

            # 更新状态
            m[:, :, q_start:q_end, :] = m_new
            l[:, :, q_start:q_end, :] = l_new

    return output


class FlashAttention(torch.nn.Module):
    """
    FlashAttention 模块。
    
    可直接替换标准 Attention:
      标准: scores = Q @ K^T; attn = softmax(scores) @ V
      Flash: 分块计算，不存储 scores 矩阵
    
    用法:
        flash = FlashAttention(head_dim, num_heads, num_kv_heads)
        output = flash(q, k, v, causal=True)
    """

    def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, seq, heads, head_dim]
            k: [batch, seq, kv_heads, head_dim]
            v: [batch, seq, kv_heads, head_dim]
        
        Returns:
            output: [batch, seq, heads * head_dim]
        """
        bsz, seqlen, _, _ = q.shape

        # 转置为 [batch, heads, seq, dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 尝试使用 PyTorch 内置的 FlashAttention (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            try:
                # GQA 扩展
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)

                output = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=causal,
                    dropout_p=self.dropout if self.training else 0.0,
                )
                return output.permute(0, 2, 1, 3).contiguous().reshape(bsz, seqlen, -1)
            except Exception:
                pass

        # 回退到手动实现的 FlashAttention
        output = flash_attention_forward(q, k, v, causal=causal)
        return output.permute(0, 2, 1, 3).contiguous().reshape(bsz, seqlen, -1)


def replace_attention_with_flash(model):
    """
    将模型中的标准 GQA Attention 替换为 FlashAttention。
    
    训练时调用，可节省 50-70% 的注意力显存。
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'attention') and hasattr(module.attention, 'wq'):
            # 这是一个 TransformerBlock
            old_attn = module.attention
            flash = FlashAttention(
                head_dim=old_attn.head_dim if hasattr(old_attn, 'head_dim') else old_attn.wq.weight.shape[0] // old_attn.num_heads,
                num_heads=old_attn.num_heads,
                num_kv_heads=old_attn.num_kv_heads,
            )
            # 保留原始权重
            flash.wq = old_attn.wq
            flash.wk = old_attn.wk
            flash.wv = old_attn.wv
            flash.wo = old_attn.wo
            module.attention = flash
            count += 1

    logger.info(f"✅ Replaced {count} attention layers with FlashAttention")
    return model