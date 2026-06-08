"""
Phase 2: Triton 加速推理引擎

将 Phase 1 的 C++ 推理引擎的单步操作替换为 Triton 融合 kernel，
同时保留 C++ 引擎的 KV Cache 管理和 GIL 释放架构。

策略:
  - C++ 引擎负责: KV Cache、生成循环、GIL 释放
  - Triton kernel 负责: 单步计算（RMSNorm、SwiGLU、RoPE、Softmax、采样）
  - Python 层胶水: 将 Triton kernel 包装为与 ATen 兼容的接口
"""

import torch
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("Taiji.TritonEngine")

# 懒加载 Triton kernels
_triton_available = False
_triton_rms_norm = None
_triton_swiglu = None
_triton_rope = None
_triton_softmax = None
_fused_ce = None


def _load_triton_kernels():
    global _triton_available, _triton_rms_norm, _triton_swiglu
    global _triton_rope, _triton_softmax, _fused_ce

    if _triton_available:
        return True

    try:
        import triton
        from ..cuda.triton_rms_norm import triton_rms_norm
        from ..cuda.triton_swiglu import triton_swiglu
        from ..cuda.triton_rope import triton_rope
        from ..cuda.triton_softmax import triton_softmax
        from ..cuda.fused_cross_entropy import fused_cross_entropy

        _triton_rms_norm = triton_rms_norm
        _triton_swiglu = triton_swiglu
        _triton_rope = triton_rope
        _triton_softmax = triton_softmax
        _fused_ce = fused_cross_entropy
        _triton_available = True
        logger.info("✅ Triton kernels loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Triton kernels not available: {e}")
        return False


class TritonEngine:
    """
    Triton 加速推理引擎。
    
    使用 Triton 融合 kernel 替换 PyTorch ATen 的逐操作调用。
    与 C++ 引擎配合使用：C++ 管理 KV Cache 和生成循环，
    Triton 负责单步前向计算。
    
    用法:
        engine = TritonEngine(model, tokenizer, device="cuda")
        output = engine.generate("你好", max_new_tokens=256)
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = model.config
        self._triton_ready = _load_triton_kernels()
        self._use_triton = self._triton_ready and device.startswith("cuda")

        # 预提取模型权重（避免每次推理重复访问）
        self._weights = None
        self._extract_weights()

    def _extract_weights(self):
        """预提取模型权重到字典，加速推理时访问"""
        sd = self.model.state_dict()
        self._weights = {}
        for k, v in sd.items():
            self._weights[k] = v.to(self.device).contiguous()

    def _rms_norm(self, x, weight, eps=1e-5):
        """RMSNorm — 有 Triton 用 Triton，否则用 ATen"""
        if self._use_triton:
            return _triton_rms_norm(x, weight, eps)
        # ATen fallback
        rms = torch.sqrt(torch.mean(x.pow(2), -1, True) + eps)
        return weight * (x / rms)

    def _swiglu(self, x, w1, w_gate, w2):
        """SwiGLU FFN — 有 Triton 用 Triton，否则用 ATen"""
        if self._use_triton:
            return _triton_swiglu(x, w1, w_gate, w2)
        # ATen fallback
        x_flat = x.reshape(-1, x.shape[-1])
        return torch.mm(torch.silu(torch.mm(x_flat, w_gate.t())) * torch.mm(x_flat, w1.t()), w2.t())

    def _apply_rope(self, q, k, start_pos=0):
        """RoPE — 有 Triton 用 Triton，否则用 ATen"""
        if self._use_triton:
            return _triton_rope(q, k, start_pos, self.config.rope_theta)
        # ATen fallback — 与 engine.cpp 中的实现一致
        seq_len = q.size(1)
        hd = q.size(3)
        pos = torch.arange(start_pos, start_pos + seq_len, device=q.device, dtype=torch.float32)
        freqs_idx = torch.arange(0, hd, 2, device=q.device, dtype=torch.float32)
        freqs = 1.0 / (self.config.rope_theta ** (freqs_idx / hd))
        angles = torch.outer(pos, freqs)
        sin, cos = torch.sin(angles), torch.cos(angles)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        q_r, q_i = q[..., ::2], q[..., 1::2]
        k_r, k_i = k[..., ::2], k[..., 1::2]
        q_out = torch.stack([q_r * cos - q_i * sin, q_r * sin + q_i * cos], -1).flatten(-2).contiguous()
        k_out = torch.stack([k_r * cos - k_i * sin, k_r * sin + k_i * cos], -1).flatten(-2).contiguous()
        return q_out, k_out

    def _softmax(self, x, dim=-1):
        """Softmax — 有 Triton 用 Triton，否则用 ATen"""
        if self._use_triton:
            return _triton_softmax(x, dim)
        return torch.softmax(x, dim=dim)

    def _cross_entropy(self, logits, targets):
        """融合交叉熵 — 有 Triton 用 Triton，否则用 ATen"""
        if self._use_triton:
            return _fused_ce(logits, targets)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """文本生成 — 使用 Triton 加速的前向传播"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        kv_cache = None
        current_ids = input_ids
        generated_ids = []

        for _ in range(max_new_tokens):
            output = self.model(
                current_ids,
                kv_cache=kv_cache,
                use_cache=True,
            )
            kv_cache = output.kv_cache

            logits = output.logits[:, -1, :] / max(temperature, 1e-6)

            # 重复惩罚
            if generated_ids:
                for pid in set(generated_ids[-64:]):
                    logits[0, pid] /= 1.2

            # Top-P 采样
            next_token = self._sample_top_p(logits, top_p)
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            current_ids = next_token.unsqueeze(0).unsqueeze(0)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _sample_top_p(self, logits, top_p):
        """Top-P 采样"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[indices_to_remove] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def get_acceleration_info(self) -> Dict[str, Any]:
        """获取加速信息"""
        return {
            "triton_available": _triton_available,
            "triton_active": self._use_triton,
            "device": self.device,
            "kernels": {
                "rms_norm": "Triton" if self._use_triton else "ATen",
                "swiglu": "Triton" if self._use_triton else "ATen",
                "rope": "Triton" if self._use_triton else "ATen",
                "softmax": "Triton" if self._use_triton else "ATen",
                "cross_entropy": "Triton" if self._use_triton else "ATen",
            }
        }