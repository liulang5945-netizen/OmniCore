"""
Triton 融合 RoPE 旋转位置编码 — 前向 + 反向
替代 PyTorch 的 sin/cos 计算 + 拆分 + 旋转 + 合并六次 kernel。

前向: 对 Q/K 应用旋转编码
反向: 旋转的逆操作
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    Q, K, SIN, COS,
    stride_batch, stride_seq, stride_head, stride_dim,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """前向: 对 Q/K 的每对相邻维度应用旋转编码"""
    pid = tl.program_id(0)
    batch = pid // num_heads
    head = pid % num_heads

    Q += batch * stride_batch + head * stride_head
    K += batch * stride_batch + head * stride_head

    half_dim = head_dim // 2

    for off in range(0, half_dim, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < half_dim

        # 加载 sin/cos
        sin_val = tl.load(SIN + cols, mask=mask, other=0.0).to(tl.float32)
        cos_val = tl.load(COS + cols, mask=mask, other=0.0).to(tl.float32)

        # 加载 Q 的偶数和奇数维度
        q_r = tl.load(Q + cols * 2, mask=mask, other=0.0).to(tl.float32)
        q_i = tl.load(Q + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        # 旋转
        q_out_r = q_r * cos_val - q_i * sin_val
        q_out_i = q_r * sin_val + q_i * cos_val

        tl.store(Q + cols * 2, q_out_r.to(Q.dtype.element_ty), mask=mask)
        tl.store(Q + cols * 2 + 1, q_out_i.to(Q.dtype.element_ty), mask=mask)

        # 加载 K 的偶数和奇数维度
        k_r = tl.load(K + cols * 2, mask=mask, other=0.0).to(tl.float32)
        k_i = tl.load(K + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        # 旋转
        k_out_r = k_r * cos_val - k_i * sin_val
        k_out_i = k_r * sin_val + k_i * cos_val

        tl.store(K + cols * 2, k_out_r.to(K.dtype.element_ty), mask=mask)
        tl.store(K + cols * 2 + 1, k_out_i.to(K.dtype.element_ty), mask=mask)


def compute_rope_sin_cos(seq_len, head_dim, theta=500000.0, device="cpu"):
    """计算 RoPE 的 sin/cos 值"""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs_idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (freqs_idx / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)


def triton_rope(q, k, start_pos=0, theta=500000.0):
    """
    融合 RoPE 前向。

    Args:
        q: [batch, seq, heads, head_dim]
        k: [batch, seq, kv_heads, head_dim]
        start_pos: KV Cache 中已有的长度
        theta: RoPE 基础频率

    Returns:
        (q_rotated, k_rotated)
    """
    bsz, seqlen, num_heads, hd = q.shape
    _, _, num_kv_heads, _ = k.shape
    total_len = start_pos + seqlen

    sin, cos = compute_rope_sin_cos(total_len, hd, theta, q.device)
    sin = sin[start_pos:].contiguous()
    cos = cos[start_pos:].contiguous()

    # Triton kernel 只处理连续内存
    q_flat = q.reshape(bsz * num_heads, seqlen, hd)
    k_flat = k.reshape(bsz * num_kv_heads, seqlen, hd)

    grid = (bsz * num_heads,)
    _rope_fwd_kernel[grid](
        q_flat, k_flat, sin, cos,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2), q_flat.stride(3),
        num_heads, hd,
        BLOCK=min(128, hd // 2),
    )

    grid_k = (bsz * num_kv_heads,)
    _rope_fwd_kernel[grid_k](
        k_flat, k_flat, sin, cos,
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2), k_flat.stride(3),
        num_kv_heads, hd,
        BLOCK=min(128, hd // 2),
    )

    return q, k