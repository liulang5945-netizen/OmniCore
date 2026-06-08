"""
Triton 融合交叉熵损失 — 前向 + 反向
不 materialize 完整 [batch, seq, vocab] 概率矩阵，节省大量显存。

原始 PyTorch:
  logits [batch, seq, 33000] → softmax → cross_entropy
  需要存储完整概率矩阵: batch × seq × 33000 × 4 bytes

融合:
  单个 kernel 直接从 logits 计算 loss，只保存 max_val 和 sum_exp 用于反向
  显存占用: batch × seq × 2 × 4 bytes (节省 99.99%)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_fwd_kernel(
    LOGITS, TARGETS, LOSSES, MAX_LOGITS, SUM_EXP,
    stride_logits_row, stride_logits_col,
    V: tl.constexpr,       # vocab_size
    BLOCK: tl.constexpr,
):
    """前向: loss = -logits[target] + log(sum(exp(logits - max(logits))))"""
    row = tl.program_id(0)
    LOGITS += row * stride_logits_row
    TARGETS += row

    # Phase 1: 找最大值 (数值稳定)
    max_val = tl.full([1], value=-float("inf"), dtype=tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(LOGITS + cols * stride_logits_col, mask=mask, other=-float("inf")).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Phase 2: 计算 log_sum_exp
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(LOGITS + cols * stride_logits_col, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.exp(x - max_val)
    _sum = tl.sum(_sum)
    log_sum_exp = tl.log(_sum) + max_val

    # Phase 3: 计算 loss
    target_id = tl.load(TARGETS).to(tl.int32)
    target_logit = tl.load(LOGITS + target_id * stride_logits_col).to(tl.float32)
    loss = -target_logit + log_sum_exp

    # 保存用于反向
    tl.store(LOSSES + row, loss)
    tl.store(MAX_LOGITS + row, max_val)
    tl.store(SUM_EXP + row, _sum)


@triton.jit
def _cross_entropy_bwd_kernel(
    LOGITS, TARGETS, GRAD_OUTPUT,
    MAX_LOGITS, SUM_EXP,
    GRAD_LOGITS,
    stride_logits_row, stride_logits_col,
    stride_grad_row, stride_grad_col,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """反向: grad[i] = grad_output * (softmax(logit[i]) - (i == target ? 1 : 0))"""
    row = tl.program_id(0)
    LOGITS += row * stride_logits_row
    TARGETS += row
    GRAD_LOGITS += row * stride_grad_row

    max_val = tl.load(MAX_LOGITS + row).to(tl.float32)
    sum_exp = tl.load(SUM_EXP + row).to(tl.float32)
    grad = tl.load(GRAD_OUTPUT + row).to(tl.float32)
    target_id = tl.load(TARGETS).to(tl.int32)

    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(LOGITS + cols * stride_logits_col, mask=mask, other=0.0).to(tl.float32)
        prob = tl.exp(x - max_val) / sum_exp
        # grad = grad_output * (prob - (i == target))
        target_one_hot = tl.where(cols == target_id, 1.0, 0.0)
        g = grad * (prob - target_one_hot)
        tl.store(GRAD_LOGITS + cols * stride_grad_col, g.to(GRAD_LOGITS.dtype.element_ty), mask=mask)


class FusedCrossEntropy(torch.autograd.Function):
    """融合交叉熵 — 不 materialize 概率矩阵"""

    @staticmethod
    def forward(ctx, logits, targets):
        """
        Args:
            logits: [batch * seq, vocab_size] (float32)
            targets: [batch * seq] (int64)
        Returns:
            loss: scalar (mean over all tokens)
        """
        B, V = logits.shape
        losses = torch.empty(B, device=logits.device, dtype=torch.float32)
        max_logits = torch.empty(B, device=logits.device, dtype=torch.float32)
        sum_exp = torch.empty(B, device=logits.device, dtype=torch.float32)

        BLOCK = min(triton.next_power_of_2(V), 4096)

        _cross_entropy_fwd_kernel[(B,)](
            logits, targets, losses, max_logits, sum_exp,
            logits.stride(0), logits.stride(1),
            V, BLOCK,
        )

        ctx.save_for_backward(logits, targets, max_logits, sum_exp)
        ctx.BLOCK = BLOCK

        return losses.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, max_logits, sum_exp = ctx.saved_tensors
        B, V = logits.shape
        grad_logits = torch.empty_like(logits)

        # grad_output 是 scalar，需要广播到每个 token
        grad_per_token = torch.full((B,), grad_output.item() / B,
                                     device=logits.device, dtype=torch.float32)

        _cross_entropy_bwd_kernel[(B,)](
            logits, targets, grad_per_token,
            max_logits, sum_exp,
            grad_logits,
            logits.stride(0), logits.stride(1),
            grad_logits.stride(0), grad_logits.stride(1),
            V, ctx.BLOCK,
        )

        return grad_logits, None


def fused_cross_entropy(logits, targets):
    """
    融合交叉熵损失。

    Args:
        logits: [batch * seq, vocab_size] 或 [batch, seq, vocab_size]
        targets: [batch * seq] 或 [batch, seq] (int64)

    Returns:
        loss: scalar
    """
    orig_shape = logits.shape
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

    return FusedCrossEntropy.apply(logits.contiguous(), targets.contiguous())