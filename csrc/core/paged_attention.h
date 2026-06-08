#pragma once
/**
 * PagedAttention 基础设施 (Phase 5+)
 * ================================
 * vLLM 风格的分页 KV Cache 管理。
 * 
 * 启用方式：在 ModelConfig 中设置 use_paged_attention=true，
 *           并在 CMakeLists.txt 中定义 USE_PAGED_ATTENTION。
 */

#include <torch/torch.h>
#include <vector>
#include <stdexcept>

namespace taiji {

// PagedAttention: 每页固定 token 数
struct PagedAttentionConfig {
    static constexpr int BLOCK_SIZE = 16;   // 每页 16 tokens
    static constexpr int NUM_BLOCKS = 5000; // 预分配 5000 页 (~80K tokens)
};

// 块表：记录单个序列使用了哪些物理块
struct BlockTable {
    std::vector<int> block_indices;  // 物理块索引列表
    int num_tokens = 0;              // 当前序列总 token 数
    
    int num_blocks() const { return static_cast<int>(block_indices.size()); }
    int capacity() const { return num_blocks() * PagedAttentionConfig::BLOCK_SIZE; }
};

// 块分配器：管理物理块池
class BlockAllocator {
public:
    BlockAllocator() = default;
    BlockAllocator(int num_blocks, int block_size, int num_kv_heads, int head_dim, torch::Device device);
    
    // 分配一个块，返回块索引，-1 表示块池已满
    int allocate();
    
    // 释放块
    void free(int block_idx);
    
    // 释放块列表
    void free_blocks(const std::vector<int>& block_indices);
    
    // 剩余可用块数
    int num_free() const { return num_free_; }
    
    // 获取指定块的 K/V tensor [block_size, num_kv_heads, head_dim]
    torch::Tensor get_k_block(int block_idx) const;
    torch::Tensor get_v_block(int block_idx) const;
    
    bool empty() const { return num_blocks_ == 0; }

private:
    int num_blocks_ = 0;
    int block_size_ = 0;
    torch::Tensor k_pool_;  // [num_blocks, block_size, num_kv_heads, head_dim]
    torch::Tensor v_pool_;  // [num_blocks, block_size, num_kv_heads, head_dim]
    std::vector<bool> free_map_;
    int num_free_ = 0;
};

// ================================
// BlockAllocator inline implementation
// ================================

inline BlockAllocator::BlockAllocator(int num_blocks, int block_size, int num_kv_heads, int head_dim, torch::Device device)
    : num_blocks_(num_blocks), block_size_(block_size), num_free_(num_blocks) 
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    k_pool_ = torch::empty({num_blocks, block_size, num_kv_heads, head_dim}, opts);
    v_pool_ = torch::empty({num_blocks, block_size, num_kv_heads, head_dim}, opts);
    free_map_.resize(num_blocks, true);
}

inline int BlockAllocator::allocate() {
    for (int i = 0; i < num_blocks_; ++i) {
        if (free_map_[i]) {
            free_map_[i] = false;
            --num_free_;
            return i;
        }
    }
    return -1;  // 池已满
}

inline void BlockAllocator::free(int block_idx) {
    if (block_idx < 0 || block_idx >= num_blocks_) return;
    if (!free_map_[block_idx]) {
        free_map_[block_idx] = true;
        ++num_free_;
    }
}

inline void BlockAllocator::free_blocks(const std::vector<int>& block_indices) {
    for (int idx : block_indices) {
        if (idx >= 0 && idx < num_blocks_ && !free_map_[idx]) {
            free_map_[idx] = true;
            ++num_free_;
        }
    }
}

inline torch::Tensor BlockAllocator::get_k_block(int block_idx) const {
    // [block_size, num_kv_heads, head_dim]
    return k_pool_.slice(0, block_idx, block_idx + 1).squeeze(0);
}

inline torch::Tensor BlockAllocator::get_v_block(int block_idx) const {
    return v_pool_.slice(0, block_idx, block_idx + 1).squeeze(0);
}

// 辅助函数：从块表 gather KV 到连续 buffer
// dst: [total_tokens, num_kv_heads, head_dim]
// pool: [num_blocks, block_size, num_kv_heads, head_dim]
inline void gather_kv_from_blocks(
    const torch::Tensor& pool,
    const BlockTable& table,
    torch::Tensor& dst)
{
    int pos = 0;
    for (int block_idx : table.block_indices) {
        if (pos >= table.num_tokens) break;
        int copy_len = std::min(PagedAttentionConfig::BLOCK_SIZE, table.num_tokens - pos);
        dst.slice(0, pos, pos + copy_len).copy_(pool.slice(0, block_idx, block_idx + 1).squeeze(0).slice(0, 0, copy_len));
        pos += copy_len;
    }
}

} // namespace taiji
