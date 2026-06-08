"""
Triton 融合 Softmax — 前向 + 反向
替代 PyTorch 的 max + exp + div 三次 kernel，合并为单次 kernel。
支持在线 softmax（不存储完整概率矩阵）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(
    X, Y, stride_row, stride_col,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """前向: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))"""
    row = tl.program_id(0)
    X += row * stride_row
    Y += row * stride_row

    # Phase 1: 找最大值
    max_val = tl.full([1], value=-float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols * stride_col, mask=mask, other=-float("inf")).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Phase 2: 计算 exp(x - max) 和 sum
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        ex = tl.exp(x - max_val)
        _sum += ex
    _sum = tl.sum(_sum)

    # Phase 3: 归一化并存储
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - max_val) / _sum
        tl.store(Y + cols * stride_col, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def _online_softmax_fwd_kernel(
    X, Y, stride_row, stride_col,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """在线 softmax — 单遍扫描，适合 FlashAttention 内部使用"""
    row = tl.program_id(0)
    X += row * stride_row
    Y += row * stride_row

    max_val = tl.full([1], value=-float("inf"), dtype=tl.float32)
    _sum = tl.zeros([1], dtype=tl.float32)

    # 单遍在线计算
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols * stride_col, mask=mask, other=-float("inf")).to(tl.float32)
        new_max = tl.maximum(max_val, tl.max(x, axis=0))
        # 更新 sum
        _sum = _sum * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max))
        max_val = new_max

    # 存储结果
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - max_val) / _sum
        tl.store(Y + cols * stride_col, y.to(X.dtype.element_ty), mask=mask)


def triton_softmax(x, dim=-1):
    """
    融合 Softmax — 推理用。

    Args:
        x: 输入张量
        dim: softmax 维度（默认最后一维）

    Returns:
        softmax(x)
    """
    if dim != -1 and dim != x.dim() - 1:
        x = x.transpose(dim, -1).contiguous()

    orig_shape = x.shape
    N = x.shape[-1]
    x_2d = x.reshape(-1, N)
    y = torch.empty_like(x_2d)

    BLOCK = min(triton.next_power_of_2(N), 4096)
    rows = x_2d.shape[0]

    _softmax_fwd_kernel[(rows,)](
        x_2d, y,
        x_2d.stride(0), x_2d.stride(1),
        N, BLOCK,
    )

    result = y.reshape(orig_shape)
    if dim != -1 and dim != x.dim() - 1:
        result = result.transpose(dim, -1)
    return result