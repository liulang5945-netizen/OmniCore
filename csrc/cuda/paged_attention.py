"""
Phase 5: PagedAttention — 像操作系统管理虚拟内存一样管理 KV Cache

核心问题:
  标准 KV Cache 预分配 max_seq_len 的连续显存，
  但实际生成的 token 远少于 max_seq_len，导致大量显存浪费。

PagedAttention 解决方案:
  1. 将 KV Cache 分成固定大小的页（默认每页 16 个 token）
  2. 按需分配页（生成 1 个 token 只分配 1/16 页）
  3. 用 Block Table 映射逻辑页 → 物理页
  4. 请求结束后释放所有页

显存利用率: 从 ~30% 提升到 ~95%

参考: vLLM "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
"""

import torch
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger("Taiji.PagedAttention")

PAGE_SIZE = 16  # 每页存储 16 个 token 的 KV


@dataclass
class PhysicalPage:
    """物理页 — 存储一页 KV 数据"""
    k_data: torch.Tensor  # [num_kv_heads, PAGE_SIZE, head_dim]
    v_data: torch.Tensor
    ref_count: int = 0    # 引用计数
    is_free: bool = True


class PagedKVCacheManager:
    """
    PagedAttention KV Cache 管理器。
    
    管理物理页池，为每个请求维护逻辑页→物理页映射。
    
    用法:
        manager = PagedKVCacheManager(num_layers=12, num_kv_heads=12, head_dim=64)
        
        # 新请求
        req_id = manager.allocate_request()
        
        # 生成一个 token
        manager.append_token(req_id, layer=0, k=k_tensor, v=v_tensor)
        
        # 获取 KV Cache 用于注意力计算
        k, v = manager.get_kv(req_id, layer=0)
        
        # 请求结束
        manager.free_request(req_id)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = PAGE_SIZE,
        max_num_pages: int = 1024,
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.device = device
        self.dtype = dtype

        # 物理页池: [num_layers, num_pages, num_kv_heads, page_size, head_dim]
        self.k_pool = torch.zeros(
            num_layers, max_num_pages, num_kv_heads, page_size, head_dim,
            device=device, dtype=dtype,
        )
        self.v_pool = torch.zeros_like(self.k_pool)

        # 空闲页栈
        self._free_pages: List[int] = list(range(max_num_pages))

        # 活跃请求: req_id → (block_table, current_len)
        # block_table: List[int] — 逻辑页→物理页映射
        self._requests: Dict[int, Tuple[List[int], int]] = {}
        self._next_req_id = 0

        logger.info(f"PagedKVCacheManager: {max_num_pages} pages × {page_size} tokens = "
                     f"{max_num_pages * page_size} total tokens capacity")

    def allocate_request(self) -> int:
        """为新请求分配初始页"""
        req_id = self._next_req_id
        self._next_req_id += 1

        # 分配第一页
        first_page = self._allocate_page()
        self._requests[req_id] = ([first_page], 0)
        return req_id

    def append_token(
        self,
        req_id: int,
        layer: int,
        k: torch.Tensor,  # [num_kv_heads, head_dim]
        v: torch.Tensor,
    ):
        """追加一个 token 的 KV 到指定请求和层"""
        if req_id not in self._requests:
            raise ValueError(f"Request {req_id} not found")

        block_table, current_len = self._requests[req_id]
        page_idx = current_len // self.page_size
        offset = current_len % self.page_size

        # 需要新页
        if page_idx >= len(block_table):
            new_page = self._allocate_page()
            block_table.append(new_page)

        # 写入 KV 数据
        phys_page = block_table[page_idx]
        self.k_pool[layer, phys_page, :, offset, :] = k.to(self.dtype)
        self.v_pool[layer, phys_page, :, offset, :] = v.to(self.dtype)

        # 更新长度（只在第 0 层更新一次）
        if layer == 0:
            self._requests[req_id] = (block_table, current_len + 1)

    def get_kv(
        self,
        req_id: int,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定请求和层的完整 KV Cache。
        
        Returns:
            k: [num_kv_heads, total_len, head_dim]
            v: [num_kv_heads, total_len, head_dim]
        """
        block_table, current_len = self._requests[req_id]
        num_full_pages = current_len // self.page_size
        last_page_len = current_len % self.page_size

        k_parts = []
        v_parts = []

        for i, phys_page in enumerate(block_table):
            if i < num_full_pages:
                # 完整页
                k_parts.append(self.k_pool[layer, phys_page])
                v_parts.append(self.v_pool[layer, phys_page])
            elif last_page_len > 0:
                # 最后一页（部分）
                k_parts.append(self.k_pool[layer, phys_page, :, :last_page_len, :])
                v_parts.append(self.v_pool[layer, phys_page, :, :last_page_len, :])

        if not k_parts:
            return (
                torch.zeros(self.num_kv_heads, 0, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(self.num_kv_heads, 0, self.head_dim, device=self.device, dtype=self.dtype),
            )

        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    def free_request(self, req_id: int):
        """释放请求的所有页"""
        if req_id not in self._requests:
            return

        block_table, _ = self._requests.pop(req_id)
        for phys_page in block_table:
            self._free_page(phys_page)

    def get_stats(self) -> Dict:
        """获取内存使用统计"""
        total_pages = self.max_num_pages
        free_pages = len(self._free_pages)
        used_pages = total_pages - free_pages
        return {
            "total_pages": total_pages,
            "used_pages": used_pages,
            "free_pages": free_pages,
            "utilization": used_pages / total_pages if total_pages > 0 else 0,
            "active_requests": len(self._requests),
            "total_token_capacity": total_pages * self.page_size,
            "page_size": self.page_size,
        }

    def _allocate_page(self) -> int:
        """分配一个物理页"""
        if not self._free_pages:
            raise RuntimeError("No free pages available! Increase max_num_pages.")
        page_id = self._free_pages.pop()
        # 清零（确保干净）
        self.k_pool[:, page_id].zero_()
        self.v_pool[:, page_id].zero_()
        return page_id

    def _free_page(self, page_id: int):
        """释放一个物理页"""
        self._free_pages.append(page_id)
        self.k_pool[:, page_id].zero_()
        self.v_pool[:, page_id].zero_()


class PagedAttention:
    """
    PagedAttention 计算模块。
    
    使用分页 KV Cache 执行注意力计算。
    与 PagedKVCacheManager 配合使用。
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qpk = num_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,           # [batch, seq, num_heads, head_dim]
        cache_manager: PagedKVCacheManager,
        req_id: int,
        layer: int,
    ) -> torch.Tensor:
        """
        使用 Paged KV Cache 执行注意力。
        
        Args:
            q: Query tensor [batch=1, seq=1, num_heads, head_dim]
            cache_manager: PagedKVCacheManager 实例
            req_id: 请求 ID
            layer: 层索引
        
        Returns:
            output: [batch=1, seq=1, num_heads * head_dim]
        """
        # 获取该请求该层的完整 KV Cache
        k, v = cache_manager.get_kv(req_id, layer)
        # k: [num_kv_heads, total_len, head_dim]
        # v: [num_kv_heads, total_len, head_dim]

        total_len = k.size(1)
        if total_len == 0:
            return torch.zeros(1, 1, self.num_heads * self.head_dim,
                               device=q.device, dtype=q.dtype)

        # 转换维度
        q = q.squeeze(0).squeeze(0)  # [num_heads, head_dim]
        q = q.unsqueeze(0)  # [1, num_heads, head_dim]

        k = k.unsqueeze(0)  # [1, num_kv_heads, total_len, head_dim]
        v = v.unsqueeze(0)

        # GQA 扩展
        if self.qpk > 1:
            k = k.repeat_interleave(self.qpk, dim=1)
            v = v.repeat_interleave(self.qpk, dim=1)

        # 注意力计算
        scale = self.head_dim ** -0.5
        q = q.transpose(0, 1)  # [num_heads, 1, head_dim]
        k = k.transpose(0, 1)  # [num_heads, total_len, head_dim]
        v = v.transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [num_heads, 1, head_dim]

        return attn_output.transpose(0, 1).reshape(1, 1, -1)