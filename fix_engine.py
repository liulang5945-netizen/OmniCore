import sys

with open('e:/OmniCore/csrc/core/engine.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 替换 attention_forward
old_attention = '''// Attention 计算
at::Tensor attention_forward
torch::Tensor attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v
)
{
    // q, k, v shape: [batch, num_heads, seq_len, head_dim]
    int64_t head_dim = q.size(-1);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    at::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
    at::Tensor attn   = torch::softmax(scores, -1);
    at::Tensor out    = torch::matmul(attn, v);
    return out;
}'''

new_attention = '''// Attention 计算 — 使用 PyTorch 2.0+ Flash Attention (scaled_dot_product_attention)
torch::Tensor attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool causal = true
)
{
    // q, k, v shape: [batch, num_heads, seq_len, head_dim]
    try {
        // PyTorch 2.0+ 原生 Flash Attention，自动选择最优后端
        auto opts = torch::nn::functional::ScaledDotProductAttentionFuncOptions()
            .is_causal(causal)
            .enable_flash(true)
            .enable_math(false)
            .enable_mem_eff(true);
        return torch::nn::functional::scaled_dot_product_attention(q, k, v, opts);
    } catch (...) {
        int64_t head_dim = q.size(-1);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        at::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
        at::Tensor attn   = torch::softmax(scores, -1);
        at::Tensor out    = torch::matmul(attn, v);
        return out;
    }
}'''

if old_attention in content:
    content = content.replace(old_attention, new_attention)
    print('[OK] attention_forward -> Flash Attention')
else:
    print('[WARN] attention_forward not matched')

# 2. 替换 KV Cache
old_kvcache = '''    // KV Cache 拼接到已有缓存上
    past_kv_pairs[ decode_layer_index*2 + 0 ] = torch::cat({past_kv_pairs[ decode_layer_index*2 + 0 ],
                                                           k_input_for_layer }, 2);
    past_kv_pairs[ decode_layer_index*2 + 1 ] = torch::cat({past_kv_pairs[ decode_layer_index*2 + 1 ],
                                                           v_input_for_layer }, 2);'''

new_kvcache = '''    // KV Cache 更新 — 预分配 + in-place 填充
    auto& kv_k = past_kv_pairs[ decode_layer_index*2 + 0 ];
    auto& kv_v = past_kv_pairs[ decode_layer_index*2 + 1 ];
    int64_t orig_len = kv_k.size(2);
    int64_t new_len  = k_input_for_layer.size(2);
    int64_t required = orig_len + new_len;
    
    // 检查并扩展缓存容量(2倍增长策略)
    if (kv_k.size(2) < required) {
        int64_t alloc_len = std::max(required, orig_len * 2);
        auto new_k = torch::zeros({kv_k.size(0), kv_k.size(1), alloc_len, kv_k.size(3)},
                                   torch::TensorOptions().dtype(kv_k.dtype()).device(kv_k.device()));
        auto new_v = torch::zeros({kv_v.size(0), kv_v.size(1), alloc_len, kv_v.size(3)},
                                   torch::TensorOptions().dtype(kv_v.dtype()).device(kv_v.device()));
        new_k.slice(2, 0, orig_len).copy_(kv_k);
        new_v.slice(2, 0, orig_len).copy_(kv_v);
        kv_k = new_k;
        kv_v = new_v;
    }
    // In-place 写入新 token
    kv_k.slice(2, orig_len, orig_len + new_len).copy_(k_input_for_layer);
    kv_v.slice(2, orig_len, orig_len + new_len).copy_(v_input_for_layer);'''

if old_kvcache in content:
    content = content.replace(old_kvcache, new_kvcache)
    print('[OK] KV Cache -> pre-alloc + in-place')
else:
    print('[WARN] KV Cache not matched')

# 3. 添加 CUDA Graphs 标记到 generate()
old_generate = '    // 循环生成 token\n    for (int i = 0; i < generation_params.max_new_tokens; ++i)\n    {'
new_generate = '''    // CUDA Graphs 建议: 对于大量重复小 batch 推理，可捕获 CUDA Graph 降低 launch overhead
    bool use_cuda_graphs = generation_params.max_new_tokens > 5 && xlu_output.device().is_cuda();
    if (use_cuda_graphs) {
        std::cerr << "[INFO] CUDA Graphs suggestion: consider enabling for generation loop" << std::endl;
    }

    // 循环生成 token
    for (int i = 0; i < generation_params.max_new_tokens; ++i)
    {'''

if old_generate in content:
    content = content.replace(old_generate, new_generate)
    print('[OK] generate() CUDA Graphs hint added')
else:
    print('[WARN] generate() not matched')

with open('e:/OmniCore/csrc/core/engine.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done.')
