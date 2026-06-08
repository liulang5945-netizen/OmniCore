"""
Phase 3: CUDA Graphs — 消除 kernel launch 开销

对于小模型（125M），单次 forward 只需 ~1ms，
但 CUDA kernel launch 开销就有 ~0.5ms（每次 launch ~5-10μs × 50-100 个 kernel）。
CUDA Graphs 将整个计算图录制一次，后续直接 replay，launch 开销降到 ~0.01ms。

适用场景:
  - 自回归推理（每步 forward 结构完全相同）
  - KV Cache 固定大小的分桶推理

限制:
  - 输入 shape 必须固定（通过分桶解决）
  - 首次录制有额外开销
  - 不支持动态控制流
"""

import torch
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger("Taiji.CudaGraphs")


class CUDAGraphWrapper:
    """
    CUDA Graph 包装器。
    
    将模型的 forward pass 录制为 CUDA Graph，
    后续推理直接 replay，消除 kernel launch 开销。
    
    分桶策略: 将序列长度分为 [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    每个桶录制一个独立的 CUDA Graph。
    
    用法:
        wrapper = CUDAGraphWrapper(model, device="cuda")
        wrapper.warmup(max_seq_len=2048)  # 预录制所有桶
        
        # 推理
        output = wrapper.forward(input_ids, kv_cache)
    """

    # 预定义的序列长度桶
    BUCKET_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self._graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self._warmed_up = False

    def _get_bucket_size(self, seq_len: int) -> int:
        """找到大于等于 seq_len 的最小桶"""
        for bucket in self.BUCKET_SIZES:
            if bucket >= seq_len:
                return bucket
        return seq_len  # 超出预定义范围，不使用 graph

    @torch.inference_mode()
    def warmup(self, max_seq_len: int = 2048):
        """
        预录制 CUDA Graphs。
        
        对每个桶大小:
          1. 创建固定大小的输入 tensor
          2. 执行一次 warmup forward（初始化所有内部状态）
          3. 录制 CUDA Graph
          4. 后续 replay 使用录制好的 graph
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA 不可用，跳过 CUDA Graphs 预录制")
            return

        logger.info(f"预录制 CUDA Graphs (max_seq_len={max_seq_len})...")

        for bucket in self.BUCKET_SIZES:
            if bucket > max_seq_len:
                break

            try:
                self._record_graph(bucket)
                logger.debug(f"  ✅ bucket={bucket}")
            except Exception as e:
                logger.warning(f"  ❌ bucket={bucket}: {e}")

        self._warmed_up = True
        logger.info(f"✅ CUDA Graphs 预录制完成 ({len(self._graphs)} 个桶)")

    def _record_graph(self, seq_len: int):
        """录制单个 CUDA Graph"""
        # 创建固定大小的输入
        static_input = torch.zeros(1, seq_len, device=self.device, dtype=torch.long)
        static_kv = None

        # Warmup: 执行几次 forward 初始化所有内部状态
        for _ in range(3):
            output = self.model(static_input, use_cache=True)

        # 录制 CUDA Graph
        graph = torch.cuda.CUDAGraph()
        static_output = None

        with torch.cuda.graph(graph):
            static_output = self.model(static_input, use_cache=True)

        self._graphs[seq_len] = (graph, static_input, static_output)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = True):
        """
        使用 CUDA Graph 执行 forward。
        
        如果输入大小匹配已录制的桶，直接 replay；
        否则回退到普通 forward。
        """
        if not self._warmed_up or not self._graphs:
            return self.model(input_ids, use_cache=use_cache)

        seq_len = input_ids.shape[1]
        bucket = self._get_bucket_size(seq_len)

        if bucket not in self._graphs:
            # 超出录制范围，回退到普通 forward
            return self.model(input_ids, use_cache=use_cache)

        graph, static_input, static_output = self._graphs[bucket]

        # 将输入复制到静态 tensor
        if seq_len == bucket:
            static_input.copy_(input_ids)
        else:
            # 序列长度小于桶大小，需要 padding
            static_input.zero_()
            static_input[:, :seq_len].copy_(input_ids)

        # Replay CUDA Graph
        graph.replay()

        return static_output

    def clear(self):
        """清除所有录制的 graphs"""
        self._graphs.clear()
        self._warmed_up = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BucketedKVCacheManager:
    """
    分桶 KV Cache 管理器。
    
    为每个桶预分配固定大小的 KV Cache tensor，
    避免动态分配导致的 CUDA Graph 录制失败。
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int = 4096, device: str = "cuda"):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # 预分配 KV Cache
        self.k_cache = torch.zeros(
            num_layers, 1, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16
        )
        self.v_cache = torch.zeros(
            num_layers, 1, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16
        )
        self.current_len = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """追加 KV 到缓存"""
        seq_len = k.shape[2]
        pos = self.current_len
        self.k_cache[layer_idx, :, :, pos:pos + seq_len, :] = k
        self.v_cache[layer_idx, :, :, pos:pos + seq_len, :] = v

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 KV Cache"""
        return (
            self.k_cache[layer_idx, :, :, :self.current_len, :],
            self.v_cache[layer_idx, :, :, :self.current_len, :],
        )

    def advance(self, n: int = 1):
        """前进 n 个位置"""
        self.current_len += n

    def reset(self):
        """重置缓存"""
        self.current_len = 0