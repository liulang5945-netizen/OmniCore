"""
KV Cache 预分配管理优化脚本
- 修改 engine.h 添加预分配字段
- 修改 engine.cpp 的 reset_cache() 和 attention_forward()
"""

import re

# ============================
# 1. 修改 engine.h
# ============================
with open('csrc/core/engine.h', 'r', encoding='utf-8') as f:
    h_content = f.read()

# 在 kv_cache_ 声明附近添加预分配字段
old_kv_decl = '''    // KV Cache: 每层一对 (K, V)
    // K: [batch, cached_len, kv_heads, head_dim]
    // V: [batch, cached_len, kv_heads, head_dim]
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache_;
    int cache_len_ = 0;'''

new_kv_decl = '''    // KV Cache: 每层一对 (K, V)
    // K: [batch, cached_len, kv_heads, head_dim]
    // V: [batch, cached_len, kv_heads, head_dim]
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache_;
    int cache_len_ = 0;

    // KV Cache 预分配 buffer 管理
    static constexpr int KV_CACHE_INITIAL_CAPACITY = 4096;
    int kv_cache_capacity_ = 0;  // 当前预分配的序列长度
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache_buffers_;  // 预分配 buffer'''

if old_kv_decl in h_content:
    h_content = h_content.replace(old_kv_decl, new_kv_decl)
    print('[OK] engine.h: 添加 KV Cache 预分配字段')
else:
    print('[WARN] engine.h: 未匹配到 KV Cache 声明')

with open('csrc/core/engine.h', 'w', encoding='utf-8') as f:
    f.write(h_content)

# ============================
# 2. 修改 engine.cpp
# ============================
with open('csrc/core/engine.cpp', 'r', encoding='utf-8') as f:
    cpp_content = f.read()

# 修改 reset_cache() — 预分配 KV Cache buffer
old_reset = '''void TaijiEngine::reset_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_cache_.clear();
    kv_cache_.resize(config_.num_hidden_layers);
    cache_len_ = 0;
    generated_history_.clear();
}'''

new_reset = '''void TaijiEngine::reset_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_cache_.clear();
    kv_cache_.resize(config_.num_hidden_layers);
    cache_len_ = 0;
    generated_history_.clear();

    // 预分配 KV Cache buffer（按 max_position_embeddings 或初始容量）
    int capacity = std::max(config_.max_position_embeddings, KV_CACHE_INITIAL_CAPACITY);
    int num_layers = config_.num_hidden_layers;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    kv_cache_buffers_.clear();
    kv_cache_buffers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto k_buf = torch::empty({1, capacity, num_kv_heads, head_dim}, opts);
        auto v_buf = torch::empty({1, capacity, num_kv_heads, head_dim}, opts);
        kv_cache_buffers_.push_back({k_buf, v_buf});
    }
    kv_cache_capacity_ = capacity;
}'''

if old_reset in cpp_content:
    cpp_content = cpp_content.replace(old_reset, new_reset)
    print('[OK] engine.cpp: reset_cache() -> 预分配 KV buffer')
else:
    print('[WARN] engine.cpp: reset_cache() 未匹配')

# 修改 attention_forward 中的 KV Cache 拼接逻辑
# 找到：
#   // KV Cache 拼接
#   torch::Tensor k_full, v_full;
#   if (kv_cache_in != nullptr && kv_cache_in->first.defined()) {
#       k_full = torch::cat({kv_cache_in->first, xk}, 1);
#       v_full = torch::cat({kv_cache_in->second, xv}, 1);
#   } else {
#       k_full = xk;
#       v_full = xv;
#   }

old_kv_cat = '''    // KV Cache 拼接
    torch::Tensor k_full, v_full;
    if (kv_cache_in != nullptr && kv_cache_in->first.defined()) {
        k_full = torch::cat({kv_cache_in->first, xk}, 1);
        v_full = torch::cat({kv_cache_in->second, xv}, 1);
    } else {
        k_full = xk;
        v_full = xv;
    }'''

new_kv_prealloc = '''    // KV Cache 拼接 — 预分配 buffer + in-place 写入
    torch::Tensor k_full, v_full;
    auto& k_buf = kv_cache_buffers_[layer_idx].first;
    auto& v_buf = kv_cache_buffers_[layer_idx].second;

    if (kv_cache_in != nullptr && kv_cache_in->first.defined()) {
        int cached_len = kv_cache_in->first.size(1);
        // In-place 写入新 KV 到预分配 buffer
        k_buf.slice(1, cached_len, cached_len + seqlen).copy_(xk);
        v_buf.slice(1, cached_len, cached_len + seqlen).copy_(xv);
        // 返回有效范围的 slice（与预分配 buffer 共享内存）
        k_full = k_buf.slice(1, 0, cached_len + seqlen);
        v_full = v_buf.slice(1, 0, cached_len + seqlen);
    } else {
        // 第一次 forward，直接写入 buffer 头部
        k_buf.slice(1, 0, seqlen).copy_(xk);
        v_buf.slice(1, 0, seqlen).copy_(xv);
        k_full = k_buf.slice(1, 0, seqlen);
        v_full = v_buf.slice(1, 0, seqlen);
    }'''

if old_kv_cat in cpp_content:
    cpp_content = cpp_content.replace(old_kv_cat, new_kv_prealloc)
    print('[OK] engine.cpp: attention_forward() -> 预分配 KV buffer')
else:
    print('[WARN] engine.cpp: KV Cache 拼接未匹配')

with open('csrc/core/engine.cpp', 'w', encoding='utf-8') as f:
    f.write(cpp_content)

print('Done.')
