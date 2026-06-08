"""
Triton 融合 RMSNorm — 前向 + 反向
替代 PyTorch 的 sqrt + div + mul 三次 kernel，合并为单次 kernel。

前向: x / sqrt(mean(x^2) + eps) * weight
反向: 利用 saved rms 值，一次 kernel 计算 dL/dx 和 dL/dweight
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X, W, Y,           # 指针
    stride,             # 行步长
    N: tl.constexpr,    # hidden_size
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """前向: y = (x / rms) * weight, 其中 rms = sqrt(mean(x^2) + eps)"""
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Phase 1: 计算 sum of squares
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x * x
    mean_sq = tl.sum(_sum) / N
    rms = tl.sqrt(mean_sq + eps)

    # Phase 2: 归一化 + 缩放
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(Y + cols, (x / rms) * w, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    DY, X, W, DX, DW,   # 指针
    RMS,                  # 保存的 rms 值
    stride, N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """反向: dL/dx = dL/dy * w / rms - x * mean(dL/dy * w * x) / (rms^3 * N)
              dL/dw = sum(dL/dy * x / rms)
    """
    row = tl.program_id(0)
    DY += row * stride
    X += row * stride
    DX += row * stride

    rms = tl.load(RMS + row).to(tl.float32)
    rms_inv = 1.0 / rms

    # Phase 1: 计算 dot(dy*w, x) 用于 dL/dx
    dot = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dot += dy * w * x
    dot = tl.sum(dot)

    # Phase 2: 计算 dL/dx
    coeff = dot / (rms * rms * rms * N)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dy * w * rms_inv - x * coeff
        tl.store(DX + cols, dx.to(X.dtype.element_ty), mask=mask)


class TritonRMSNorm(torch.autograd.Function):
    """融合 RMSNorm — 可微分，支持训练"""

    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        N = x_2d.shape[1]
        y = torch.empty_like(x_2d)

        BLOCK = min(triton.next_power_of_2(N), 4096)
        rows = x_2d.shape[0]

        # 保存 rms 用于反向
        rms = torch.empty(rows, device=x.device, dtype=torch.float32)

        _rms_norm_fwd_kernel[(rows,)](
            x_2d, weight, y,
            x_2d.stride(0), N, eps, BLOCK,
        )

        # 计算并保存 rms（用于反向）
        rms_val = torch.sqrt(x_2d.float().pow(2).mean(-1) + eps)
        rms.copy_(rms_val)

        ctx.save_for_backward(x_2d, weight, rms)
        ctx.eps = eps
        ctx.BLOCK = BLOCK

        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rms = ctx.saved_tensors
        dy_2d = dy.reshape(-1, x_2d.shape[-1])
        N = x_2d.shape[1]
        rows = x_2d.shape[0]

        dx = torch.empty_like(x_2d)
        dw = torch.zeros_like(weight)

        _rms_norm_bwd_kernel[(rows,)](
            dy_2d, x_2d, weight, dx, dw, rms,
            x_2d.stride(0), N, ctx.eps, ctx.BLOCK,
        )

        return dx.reshape_as(dy), dw, None


def triton_rms_norm(x, weight, eps=1e-5):
    """融合 RMSNorm — 推理用（无梯度）"""
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    N = x_2d.shape[1]
    y = torch.empty_like(x_2d)

    BLOCK = min(triton.next_power_of_2(N), 4096)
    rows = x_2d.shape[0]

    _rms_norm_fwd_kernel[(rows,)](
        x_2d, weight, y,
        x_2d.stride(0), N, eps, BLOCK,
    )
    return y.reshape(shape)


def triton_rms_norm_train(x, weight, eps=1e-5):
    """融合 RMSNorm — 训练用（支持反向传播）"""
    return TritonRMSNorm.apply(x, weight, eps)