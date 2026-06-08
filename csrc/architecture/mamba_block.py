"""
M14: Mamba 混合层 — 状态空间模型 (SSM)

长序列 O(n) 效率，替代 Transformer 的 O(n²) 注意力。
混合架构: 奇数层 Transformer + 偶数层 Mamba。

参考: Mamba (Gu & Dao, 2023), Jamba (AI21, 2024)

优势:
  - 推理时 O(n) 时间复杂度，O(1) 显存（固定大小状态）
  - 适合长序列处理（16K+ tokens）
  - 与 Transformer 层交替使用，兼顾效率和质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaBlock(nn.Module):
    """
    Mamba 选择性状态空间模型块。
    
    核心: 选择性扫描 (Selective Scan)
      - 输入依赖的 A, B, C 矩阵
      - 动态选择记住/遗忘什么信息
    
    Args:
        hidden_size: 隐藏层维度
        d_state: SSM 状态维度 (默认 16)
        d_conv: 局部卷积宽度 (默认 4)
        expand: 内部扩展因子 (默认 2)
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        d_inner = int(expand * hidden_size)

        # 输入投影: hidden → 2 * d_inner (x 和 z 两条路径)
        self.in_proj = nn.Linear(hidden_size, d_inner * 2, bias=False)

        # 1D 因果卷积
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM 参数
        # A: 对角矩阵 (log space)
        self.A_log = nn.Parameter(torch.log(torch.randn(d_inner, d_state).abs()))
        # B, C, D: 输入依赖 (通过线性投影)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)  # B, C, dt
        self.D = nn.Parameter(torch.ones(d_inner))
        # dt 投影
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(d_inner, hidden_size, bias=False)

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        residual = x
        x = self.norm(x)

        bsz, seqlen, _ = x.shape

        # 输入投影
        xz = self.in_proj(x)  # [batch, seq, 2*d_inner]
        x_path, z_path = xz.chunk(2, dim=-1)

        # 1D 因果卷积
        x_conv = x_path.transpose(1, 2)  # [batch, d_inner, seq]
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]  # 因果: 截断到 seqlen
        x_conv = x_conv.transpose(1, 2)  # [batch, seq, d_inner]
        x_conv = F.silu(x_conv)

        # SSM 参数 (输入依赖)
        x_proj = self.x_proj(x_conv)  # [batch, seq, 2*d_state + d_inner]
        B_proj = x_proj[:, :, :self.d_state]
        C_proj = x_proj[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_proj[:, :, 2*self.d_state:]))

        # 选择性扫描
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        y = self._selective_scan(x_conv, A, B_proj, C_proj, dt)

        # 门控 + D 跳跃连接
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z_path)  # 门控

        # 输出投影
        output = self.out_proj(y)

        return output + residual

    def _selective_scan(self, x, A, B, C, dt):
        """
        选择性扫描 — Mamba 的核心。
        
        递推: h_t = A_bar * h_{t-1} + B_bar * x_t
              y_t = C_t * h_t
        
        A_bar = exp(dt * A), B_bar = dt * B (离散化)
        """
        bsz, seqlen, d_inner = x.shape
        d_state = A.shape[1]

        # 初始化状态
        h = torch.zeros(bsz, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seqlen):
            # dt 离散化
            dt_t = dt[:, t, :].unsqueeze(-1)  # [batch, d_inner, 1]
            B_t = B[:, t, :].unsqueeze(1)     # [batch, 1, d_state]
            C_t = C[:, t, :].unsqueeze(1)     # [batch, 1, d_state]
            x_t = x[:, t, :].unsqueeze(-1)    # [batch, d_inner, 1]

            # A_bar = exp(dt * A)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # [batch, d_inner, d_state]

            # B_bar = dt * B
            B_bar = dt_t * B_t  # [batch, d_inner, d_state]

            # 状态更新
            h = A_bar * h + B_bar * x_t

            # 输出
            y_t = (h * C_t).sum(dim=-1)  # [batch, d_inner]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [batch, seq, d_inner]