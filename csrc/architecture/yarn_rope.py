"""
M10: YaRN 位置编码扩展

训练时 4K 上下文，推理时可外推到 128K。
无需重新训练，通过调整 RoPE 频率实现。

参考: "YaRN: Efficient Context Window Extension of Large Language Models" (2024)

核心思想:
  1. 频率分段: 高频维度保持不变，低频维度拉伸
  2. 注意力温度: 长序列时适当降低温度
  3. NTK-aware: 动态调整 base frequency
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN 旋转位置编码。
    
    训练时 max_seq_len=4096，推理时可外推到 scale_factor * 4096。
    
    Args:
        dim: 每个头的维度
        max_seq_len: 训练时的最大序列长度
        theta: RoPE 基础频率
        scale_factor: 外推倍数 (如 32 表示 4K → 128K)
        beta_fast: 高频/低频分界线 (NTK-aware)
        beta_slow: 最低频率缩放系数
        attn_factor: 注意力温度缩放
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 500000.0,
        scale_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scale_factor = scale_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor

        # 计算频率
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

        # YaRN: 计算每个频率的缩放因子
        if scale_factor > 1.0:
            self._compute_yarn_scaling()
        else:
            self._yarn_freqs = freqs

        # 缓存
        self._cache = {}
        self._max_cache_size = 8

    def _compute_yarn_scaling(self):
        """
        计算 YaRN 频率缩放。
        
        高频维度（变化快）: 保持不变
        低频维度（变化慢）: 按 scale_factor 拉伸
        中间维度: 平滑过渡
        """
        low_freq_wavelen = self.max_seq_len / self.beta_fast
        high_freq_wavelen = self.max_seq_len / self.beta_slow

        freqs = self.freqs
        wavelen = 2 * math.pi / freqs

        # 分段缩放
        yarn_freqs = torch.where(
            wavelen > low_freq_wavelen,
            freqs / self.scale_factor,  # 低频: 拉伸
            torch.where(
                wavelen < high_freq_wavelen,
                freqs,  # 高频: 保持不变
                # 中间: 线性插值
                freqs * (1.0 - self._smooth_factor(wavelen, low_freq_wavelen, high_freq_wavelen))
                + (freqs / self.scale_factor) * self._smooth_factor(wavelen, low_freq_wavelen, high_freq_wavelen),
            ),
        )

        self._yarn_freqs = yarn_freqs

    def _smooth_factor(self, wavelen, low, high):
        """平滑过渡因子"""
        return (wavelen - low) / (high - low)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 sin/cos 值"""
        key = (seq_len, device, dtype)
        if key in self._cache:
            return self._cache[key]

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = self._yarn_freqs.to(device)
        angles = torch.outer(pos, freqs)

        sin = torch.sin(angles).to(dtype)
        cos = torch.cos(angles).to(dtype)

        # 注意力温度缩放
        if self.attn_factor != 1.0:
            cos = cos * self.attn_factor

        # LRU 缓存
        if len(self._cache) >= self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = (sin, cos)

        return sin, cos