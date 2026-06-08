"""
融合 TransformerBlock — 将 RMSNorm + RoPE + Attention + SwiGLU 串联为一个融合前向。

目标: 减少 kernel launch 开销，将多个小操作合并。
Phase 2 核心组件。
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger("Taiji.FusedBlock")

# 懒加载 Triton kernels
_triton_rms_norm = None
_triton_swiglu = None
_triton_rope = None
_triton_softmax = None


def _load_kernels():
    global _triton_rms_norm, _triton_swiglu, _triton_rope, _triton_softmax
    if _triton_rms_norm is not None:
        return True
    try:
        from .triton_rms_norm import triton_rms_norm
        from .triton_swiglu import triton_swiglu
        from .triton_rope import triton_rope
        from .triton_softmax import triton_softmax
        _triton_rms_norm = triton_rms_norm
        _triton_swiglu = triton_swiglu
        _triton_rope = triton_rope
        _triton_softmax = triton_softmax
        return True
    except ImportError:
        return False


class FusedTransformerBlock(torch.nn.Module):
    """
    融合 Transformer 块。
    
    将 Pre-Norm Transformer Block 的所有操作用 Triton kernel 实现:
      x → RMSNorm → QKV投影 → RoPE → KV Cache → GQA Attention → 输出投影 → Residual
      x → RMSNorm → SwiGLU FFN → Residual
    
    使用方式:
      block = FusedTransformerBlock(config, layer_idx)
      output, new_kv = block(x, kv_cache, use_cache=True)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size,
                 rms_norm_eps=1e-5, rope_theta=500000.0, layer_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.layer_idx = layer_idx
        self.qpk = num_heads // num_kv_heads

        # Attention weights
        self.wq = torch.nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # FFN weights (SwiGLU)
        self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_gate = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

        # Norm weights
        self.attn_norm = torch.nn.Parameter(torch.ones(hidden_size))
        self.ffn_norm = torch.nn.Parameter(torch.ones(hidden_size))

        self._triton_ok = _load_kernels()

    def _rms_norm(self, x, weight):
        if self._triton_ok:
            return _triton_rms_norm(x, weight, self.eps)
        rms = torch.sqrt(torch.mean(x.pow(2), -1, True) + self.eps)
        return weight * (x / rms)

    def _apply_rope(self, q, k, start_pos):
        if self._triton_ok:
            return _triton_rope(q, k, start_pos, self.rope_theta)
        # ATen fallback
        seq_len = q.size(1)
        hd = q.size(3)
        pos = torch.arange(start_pos, start_pos + seq_len, device=q.device, dtype=torch.float32)
        freqs_idx = torch.arange(0, hd, 2, device=q.device, dtype=torch.float32)
        freqs = 1.0 / (self.rope_theta ** (freqs_idx / hd))
        angles = torch.outer(pos, freqs)
        sin, cos = torch.sin(angles), torch.cos(angles)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        q_r, q_i = q[..., ::2], q[..., 1::2]
        k_r, k_i = k[..., ::2], k[..., 1::2]
        q_out = torch.stack([q_r * cos - q_i * sin, q_r * sin + q_i * cos], -1).flatten(-2).contiguous()
        k_out = torch.stack([k_r * cos - k_i * sin, k_r * sin + k_i * cos], -1).flatten(-2).contiguous()
        return q_out, k_out

    def _swiglu(self, x):
        if self._triton_ok:
            return _triton_swiglu(x, self.w1.weight, self.w_gate.weight, self.w2.weight)
        x_flat = x.reshape(-1, self.hidden_size)
        gate = torch.mm(x_flat, self.w_gate.weight.t())
        up = torch.mm(x_flat, self.w1.weight.t())
        hidden = F.silu(gate) * up
        return torch.mm(hidden, self.w2.weight.t()).reshape(x.shape)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        融合前向。
        
        Args:
            x: [batch, seq, hidden]
            kv_cache: (K, V) tuple or None
            use_cache: 是否返回新 KV Cache
        
        Returns:
            output: [batch, seq, hidden]
            new_kv: (K, V) or None
        """
        bsz, seqlen, _ = x.shape

        # ── Attention 部分 ──
        # RMSNorm
        h_normed = self._rms_norm(x, self.attn_norm)

        # QKV 投影
        xq = self.wq(h_normed).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        xk = self.wk(h_normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = self.wv(h_normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # RoPE
        start_pos = kv_cache[0].size(1) if kv_cache is not None and kv_cache[0] is not None else 0
        xq, xk = self._apply_rope(xq, xk, start_pos)

        # KV Cache 拼接
        if kv_cache is not None and kv_cache[0] is not None:
            k_full = torch.cat([kv_cache[0], xk], dim=1)
            v_full = torch.cat([kv_cache[1], xv], dim=1)
        else:
            k_full = xk
            v_full = xv

        new_kv = (k_full, v_full) if use_cache else None

        # GQA: 扩展 KV heads
        if self.qpk > 1:
            k_full = k_full.repeat_interleave(self.qpk, dim=2)
            v_full = v_full.repeat_interleave(self.qpk, dim=2)

        # Attention scores
        xq = xq.permute(0, 2, 1, 3)
        k_full = k_full.permute(0, 2, 1, 3)
        v_full = v_full.permute(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(xq, k_full.transpose(-2, -1)) * scale

        # 因果掩码
        total_len = k_full.size(2)
        mask = torch.full((seqlen, total_len), float("-inf"), device=x.device, dtype=x.dtype)
        row_idx = torch.arange(seqlen, device=x.device).unsqueeze(1)
        col_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
        causal = col_idx <= (total_len - seqlen + row_idx)
        mask.masked_fill_(causal, 0.0)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax (Triton or ATen)
        if self._triton_ok:
            scores = _triton_softmax(scores.to(torch.float32), dim=-1).to(xq.dtype)
        else:
            scores = torch.softmax(scores.to(torch.float32), dim=-1).to(xq.dtype)

        # Attention output
        attn_out = torch.matmul(scores, v_full)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(bsz, seqlen, -1)
        attn_out = self.wo(attn_out)

        # Residual
        h = x + attn_out

        # ── FFN 部分 ──
        h_normed = self._rms_norm(h, self.ffn_norm)
        ffn_out = self._swiglu(h_normed)
        h = h + ffn_out

        return h, new_kv


def replace_with_fused_blocks(model, config):
    """
    将模型中的标准 TransformerBlock 替换为 FusedTransformerBlock。
    
    Args:
        model: ModelSelf 模型
        config: ModelConfig
    
    Returns:
        替换后的模型
    """
    device = next(model.parameters()).device

    for i in range(config.num_hidden_layers):
        old_layer = model.backbone.layers[i]

        fused = FusedTransformerBlock(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            layer_idx=i,
        ).to(device)

        # 复制权重
        fused.wq.weight.data.copy_(old_layer.attention.wq.weight.data)
        fused.wk.weight.data.copy_(old_layer.attention.wk.weight.data)
        fused.wv.weight.data.copy_(old_layer.attention.wv.weight.data)
        fused.wo.weight.data.copy_(old_layer.attention.wo.weight.data)
        fused.w1.weight.data.copy_(old_layer.feed_forward.w1.weight.data)
        fused.w_gate.weight.data.copy_(old_layer.feed_forward.w_gate.weight.data)
        fused.w2.weight.data.copy_(old_layer.feed_forward.w2.weight.data)
        fused.attn_norm.data.copy_(old_layer.attention_norm.weight.data)
        fused.ffn_norm.data.copy_(old_layer.ffn_norm.weight.data)

        model.backbone.layers[i] = fused

    logger.info(f"✅ Replaced {config.num_hidden_layers} layers with FusedTransformerBlock")
    return model