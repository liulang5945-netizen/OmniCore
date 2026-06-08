"""
M12: Mixture of Experts (MoE) 层

7B+ 模型的高效扩展路径。
总参数量大但每次只激活 top-K 个 expert，推理速度接近小模型。

参考: DeepSeek-V3 (MoE + Shared Expert), Mixtral 8x7B

设计:
  - num_experts 个独立 SwiGLU FFN
  - Router 线性层决定每个 token 分配给哪些 expert
  - 1 个 shared expert 始终激活 (DeepSeek-V3 设计)
  - Auxiliary loss 防止路由坍塌
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MoELayer(nn.Module):
    """
    Mixture of Experts 层。
    
    Args:
        hidden_size: 隐藏层维度
        intermediate_size: 每个 expert 的 FFN 中间维度
        num_experts: expert 总数
        top_k: 每个 token 激活的 expert 数
        use_shared_expert: 是否使用 shared expert (DeepSeek-V3)
        aux_loss_weight: auxiliary loss 权重 (防止路由坍塌)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        use_shared_expert: bool = True,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.aux_loss_weight = aux_loss_weight

        # Router: 决定每个 token 分配给哪些 expert
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert pool: 每个 expert 是一个 SwiGLU FFN
        expert_intermediate = intermediate_size // num_experts * top_k
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, expert_intermediate)
            for _ in range(num_experts)
        ])

        # Shared expert (始终激活)
        if use_shared_expert:
            self.shared_expert = SwiGLUExpert(hidden_size, intermediate_size // 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, hidden_size]
        
        Returns:
            output: [batch, seq, hidden_size]
            aux_loss: scalar (auxiliary load balancing loss)
        """
        batch, seq, hidden = x.shape
        x_flat = x.reshape(-1, hidden)  # [batch*seq, hidden]

        # Router 决策
        router_logits = self.router(x_flat)  # [batch*seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化

        # 分发到 experts
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [batch*seq]
            expert_weights = top_k_probs[:, k:k+1]  # [batch*seq, 1]

            for e_idx in range(self.num_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

        # Shared expert (始终激活)
        if self.use_shared_expert:
            shared_output = self.shared_expert(x_flat)
            output = output + shared_output

        # Auxiliary loss (负载均衡)
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)

        return output.reshape(batch, seq, hidden), aux_loss

    def _compute_aux_loss(self, router_probs, top_k_indices):
        """
        计算 auxiliary load balancing loss。
        防止所有 token 都路由到同一个 expert。
        """
        # 每个 expert 被选中的频率
        expert_freq = torch.zeros(self.num_experts, device=router_probs.device)
        for e in range(self.num_experts):
            expert_freq[e] = (top_k_indices == e).float().mean()

        # 理想频率: 均匀分布
        ideal_freq = torch.ones_like(expert_freq) / self.num_experts

        # 辅助损失: 频率与理想分布的 KL 散度
        aux_loss = self.aux_loss_weight * F.kl_div(
            expert_freq.log(), ideal_freq, reduction='batchmean'
        )
        return aux_loss


class SwiGLUExpert(nn.Module):
    """单个 Expert — SwiGLU FFN"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))