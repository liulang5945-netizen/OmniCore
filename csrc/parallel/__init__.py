"""
Taiji Parallel — 多 GPU 并行支持

M17: Tensor Parallelism + Pipeline Parallelism
  - TensorParallel: 单层内切分权重到多 GPU
  - PipelineParallel: 层间切分到多 GPU

适用于 7B+ 模型在多 GPU 上训练/推理。
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TensorParallel:
    """
    张量并行 — 单层内切分权重到多 GPU。
    
    线性层切分: Y = X @ W^T
      GPU 0: Y0 = X @ W0^T  (W 的前半部分)
      GPU 1: Y1 = X @ W1^T  (W 的后半部分)
      → all_gather → Y = [Y0, Y1]
    
    用法:
        tp = TensorParallel(num_gpus=2)
        model = tp.parallelize_model(model)
    """

    def __init__(self, num_gpus: int = 2):
        self.num_gpus = num_gpus
        self.devices = [f"cuda:{i}" for i in range(num_gpus)]

    def parallelize_linear(self, linear: nn.Linear, axis: int = 0) -> nn.ModuleList:
        """
        切分线性层到多 GPU。
        
        Args:
            linear: 原始线性层
            axis: 切分轴 (0=按输出维度切分, 1=按输入维度切分)
        
        Returns:
            ModuleList of 切分后的线性层
        """
        weight = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None

        chunks = torch.chunk(weight, self.num_gpus, dim=axis)
        bias_chunks = torch.chunk(bias, self.num_gpus, dim=0) if bias is not None else [None] * self.num_gpus

        parallel_layers = nn.ModuleList()
        for i, (w_chunk, b_chunk) in enumerate(zip(chunks, bias_chunks)):
            layer = nn.Linear(w_chunk.shape[1], w_chunk.shape[0], bias=b_chunk is not None)
            layer.weight.data = w_chunk.to(self.devices[i])
            if b_chunk is not None:
                layer.bias.data = b_chunk.to(self.devices[i])
            parallel_layers.append(layer)

        return parallel_layers

    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """并行化整个模型（标记需要并行的层）"""
        # 实际实现需要更复杂的逻辑来处理模型结构
        # 这里提供框架
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                # 标记需要并行化的层
                module._tp_axis = 0
        return model

    def all_gather_output(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """收集所有 GPU 的输出"""
        return torch.cat(outputs, dim=-1)


class PipelineParallel:
    """
    流水线并行 — 层间切分到多 GPU。
    
    层分配:
      GPU 0: layers 0-10 (前半部分)
      GPU 1: layers 11-21 (后半部分)
    
    用法:
        pp = PipelineParallel(num_gpus=2, num_layers=22)
        model = pp.distribute_model(model)
    """

    def __init__(self, num_gpus: int = 2, num_layers: int = 22):
        self.num_gpus = num_gpus
        self.num_layers = num_layers
        self.devices = [f"cuda:{i}" for i in range(num_gpus)]

        # 计算每 GPU 的层数
        layers_per_gpu = num_layers // num_gpus
        self.layer_assignment = []
        for i in range(num_gpus):
            start = i * layers_per_gpu
            end = start + layers_per_gpu if i < num_gpus - 1 else num_layers
            self.layer_assignment.append((start, end))

    def distribute_layers(self, layers: nn.ModuleList) -> List[nn.ModuleList]:
        """
        将层分配到各 GPU。
        
        Args:
            layers: 原始层列表
        
        Returns:
            每个 GPU 的层列表
        """
        distributed = []
        for gpu_idx, (start, end) in enumerate(self.layer_assignment):
            gpu_layers = nn.ModuleList()
            for i in range(start, end):
                layer = layers[i].to(self.devices[gpu_idx])
                gpu_layers.append(layer)
            distributed.append(gpu_layers)
        return distributed

    def pipeline_forward(
        self,
        distributed_layers: List[nn.ModuleList],
        hidden: torch.Tensor,
        micro_batch_size: int = 1,
    ) -> torch.Tensor:
        """
        流水线前向传播。
        
        简单实现: 顺序执行各 GPU 的层。
        生产级实现需要 GPipe/1F1B 调度。
        """
        for gpu_idx, layers in enumerate(distributed_layers):
            device = self.devices[gpu_idx]
            hidden = hidden.to(device)
            for layer in layers:
                hidden = layer(hidden)
        return hidden