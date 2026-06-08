/**
 * Taiji CUDA Inference Engine — 核心实现
 *
 * Phase 1: 基础推理循环（libtorch 侧，后续替换为融合 CUDA kernel）
 * 所有推理操作使用 libtorch 的 ATen API，保证正确性。
 * Phase 2+ 将逐步替换为自定义 CUDA kernel 以获得极致性能。
 */

#include "engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace taiji {

// ════════════════════════════════════════════════════════════════
// 构造 & 初始化
// ════════════════════════════════════════════════════════════════

TaijiEngine::TaijiEngine(const ModelConfig& config, const std::string& device)
    : config_(config)
    , device_(device)
    , loaded_(false)
    , cache_len_(0)
    , rope_cache_len_(0)
    , rng_(torch::make_generator<torch::CPUGeneratorImpl>())
{
    // 预分配 KV Cache 容器
    kv_cache_.resize(config.num_hidden_layers);
}

// ════════════════════════════════════════════════════════════════
// 权重管理
// ════════════════════════════════════════════════════════════════

void TaijiEngine::load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 转换为 unordered_map
    std::unordered_map<std::string, torch::Tensor> dict;
    for (const auto& [k, v] : state_dict) {
        dict[k] = v;
    }

    // 验证
    auto missing = WeightLoader::validate(dict, config_);
    if (!missing.empty()) {
        std::cerr << "[TaijiEngine] ⚠️ 缺失 " << missing.size() << " 个权重:" << std::endl;
        for (const auto& m : missing) {
            std::cerr << "  - " << m << std::endl;
        }
    }

    // 加载
    weights_ = WeightLoader::load_from_state_dict(dict, config_, device_);
    loaded_ = true;
    cache_len_ = 0;

    std::cout << "[TaijiEngine] ✅ 权重已加载到 " << device_
              << " | " << config_.describe() << std::endl;
}

int64_t TaijiEngine::num_parameters() const {
    return config_.count_parameters();
}

void TaijiEngine::set_num_tools(int n) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.num_tools = std::min(n, config_.max_tools);
}

// ════════════════════════════════════════════════════════════════
// KV Cache 管理
// ════════════════════════════════════════════════════════════════

void TaijiEngine::reset_cache() {
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
}

int TaijiEngine::cache_seq_len() const {
    return cache_len_;
}

// ════════════════════════════════════════════════════════════════
// 内部前向传播组件
// ════════════════════════════════════════════════════════════════

torch::Tensor TaijiEngine::embedding_forward(const torch::Tensor& input_ids) {
    // input_ids: [batch, seq_len] (int64)
    // weights_.embedding: [vocab_size, hidden_size]
    // output: [batch, seq_len, hidden_size] * sqrt(hidden_size)
    auto h = torch::embedding(weights_.embedding, input_ids);
    h = h * std::sqrt(static_cast<float>(config_.hidden_size));
    return h;
}

torch::Tensor TaijiEngine::rms_norm_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    float eps)
{
    // x: [..., hidden_size], weight: [hidden_size]
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // Phase 2: 替换为融合 CUDA kernel
    auto rms = torch::sqrt(torch::mean(x.pow(2), -1, true) + eps);
    return weight * (x / rms);
}

std::pair<torch::Tensor, torch::Tensor> TaijiEngine::get_rope_sin_cos(
    int seq_len, int start_pos)
{
    int total_len = start_pos + seq_len;
    int hd = config_.head_dim();

    // 检查缓存
    if (rope_cache_len_ >= total_len && rope_cache_.first.defined()) {
        // 切片
        auto sin_full = rope_cache_.first;
        auto cos_full = rope_cache_.second;
        if (start_pos > 0) {
            return {sin_full.slice(0, start_pos, total_len),
                    cos_full.slice(0, start_pos, total_len)};
        }
        if (seq_len < total_len) {
            return {sin_full.slice(0, 0, seq_len),
                    cos_full.slice(0, 0, seq_len)};
        }
        return rope_cache_;
    }

    // 计算频率
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(device_));
    auto pos = torch::arange(total_len, opts);
    auto freqs_idx = torch::arange(0, hd, 2, opts).slice(0, 0, hd / 2);
    auto freqs = 1.0 / torch::pow(
        torch::tensor(config_.rope_theta, opts),
        freqs_idx / static_cast<float>(hd));
    auto angles = torch::outer(pos, freqs);

    rope_cache_ = {torch::sin(angles), torch::cos(angles)};
    rope_cache_len_ = total_len;

    auto sin_full = rope_cache_.first;
    auto cos_full = rope_cache_.second;
    if (start_pos > 0) {
        return {sin_full.slice(0, start_pos, total_len),
                cos_full.slice(0, start_pos, total_len)};
    }
    return {sin_full.slice(0, 0, seq_len), cos_full.slice(0, 0, seq_len)};
}

void TaijiEngine::apply_rope(
    torch::Tensor& q,
    torch::Tensor& k,
    int start_pos)
{
    // q, k: [batch, seq, heads, head_dim]
    int seq_len = q.size(1);
    int hd = q.size(3);
    auto [sin, cos] = get_rope_sin_cos(seq_len, start_pos);

    // sin, cos: [seq, hd/2] → [1, seq, 1, hd/2]
    sin = sin.unsqueeze(0).unsqueeze(2);
    cos = cos.unsqueeze(0).unsqueeze(2);

    // 拆分奇偶维度
    auto q_r = q.index({"...", torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)});
    auto q_i = q.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
    auto k_r = k.index({"...", torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)});
    auto k_i = k.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});

    // 旋转
    auto q_out_r = q_r * cos - q_i * sin;
    auto q_out_i = q_r * sin + q_i * cos;
    auto k_out_r = k_r * cos - k_i * sin;
    auto k_out_i = k_r * sin + k_i * cos;

    // 交错合并回去
    q = torch::stack({q_out_r, q_out_i}, -1).flatten(-2).contiguous();
    k = torch::stack({k_out_r, k_out_i}, -1).flatten(-2).contiguous();
}

std::pair<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>
TaijiEngine::attention_forward(
    const torch::Tensor& x,
    const LayerWeights& layer,
    int layer_idx,
    const std::pair<torch::Tensor, torch::Tensor>* kv_cache_in,
    bool use_cache)
{
    // x: [batch, seq, hidden]
    int bsz = x.size(0);
    int seqlen = x.size(1);
    int hd = config_.head_dim();
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int qpk = num_heads / num_kv_heads;  // queries per KV

    // QKV 投影
    auto xq = torch::mm(x.reshape({-1, config_.hidden_size}), layer.wq.t())
                .reshape({bsz, seqlen, num_heads, hd});
    auto xk = torch::mm(x.reshape({-1, config_.hidden_size}), layer.wk.t())
                .reshape({bsz, seqlen, num_kv_heads, hd});
    auto xv = torch::mm(x.reshape({-1, config_.hidden_size}), layer.wv.t())
                .reshape({bsz, seqlen, num_kv_heads, hd});

    // RoPE
    int start_pos = (kv_cache_in != nullptr && kv_cache_in->first.defined())
                    ? kv_cache_in->first.size(1) : 0;
    apply_rope(xq, xk, start_pos);

    // KV Cache 拼接 — 预分配 buffer + in-place 写入
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
    }

    // 保存新 KV Cache
    std::pair<torch::Tensor, torch::Tensor> new_kv;
    if (use_cache) {
        new_kv = {k_full, v_full};
    }

    int total_len = k_full.size(1);

    // GQA: 扩展 KV heads 到 Q heads
    if (qpk > 1) {
        k_full = k_full.repeat_interleave(qpk, 2);
        v_full = v_full.repeat_interleave(qpk, 2);
    }

    // 转置为 [batch, heads, seq, dim]
    xq = xq.permute({0, 2, 1, 3});
    k_full = k_full.permute({0, 2, 1, 3});
    v_full = v_full.permute({0, 2, 1, 3});

    // Flash Attention (PyTorch 2.0+ scaled_dot_product_attention)
    at::Tensor attn_out;
    if (seqlen == total_len) {
        // Prefill: use causal mask for training / prefill
        attn_out = at::scaled_dot_product_attention(
            xq, k_full, v_full,
            c10::optional<at::Tensor>(),
            0.0,
            true
        );
    } else {
        // Decode: all past tokens visible, no causal mask needed
        attn_out = at::scaled_dot_product_attention(
            xq, k_full, v_full
        );
    }
    // 保持原输出布局: [batch, seq, heads, dim] -> [batch, seq, heads*dim]
    attn_out = attn_out.permute({0, 2, 1, 3}).contiguous()
               .reshape({bsz, seqlen, -1});
    // 输出投影
    auto output = torch::mm(attn_out.reshape({-1, num_heads * hd}), layer.wo.t())
                   .reshape({bsz, seqlen, config_.hidden_size});

    return {output, new_kv};
}

torch::Tensor TaijiEngine::swiglu_forward(
    const torch::Tensor& x,
    const LayerWeights& layer)
{
    // SwiGLU: w2(silu(w_gate(x)) * w1(x))
    auto x_flat = x.reshape({-1, config_.hidden_size});
    auto gate = torch::mm(x_flat, layer.w_gate.t());
    auto up = torch::mm(x_flat, layer.w1.t());
    auto hidden = torch::silu(gate) * up;
    return torch::mm(hidden, layer.w2.t()).reshape({x.size(0), x.size(1), config_.hidden_size});
}

std::pair<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>
TaijiEngine::transformer_block_forward(
    const torch::Tensor& x,
    const LayerWeights& layer,
    int layer_idx,
    const std::pair<torch::Tensor, torch::Tensor>* kv_cache_in,
    bool use_cache)
{
    // Pre-Norm Transformer Block
    // x → RMSNorm → Attention → + residual
    //   → RMSNorm → SwiGLU → + residual

    auto h_normed = rms_norm_forward(x, layer.attn_norm, config_.rms_norm_eps);
    auto [attn_out, new_kv] = attention_forward(h_normed, layer, layer_idx, kv_cache_in, use_cache);
    auto h = x + attn_out;

    auto ffn_normed = rms_norm_forward(h, layer.ffn_norm, config_.rms_norm_eps);
    auto ffn_out = swiglu_forward(ffn_normed, layer);
    h = h + ffn_out;

    return {h, new_kv};
}

torch::Tensor TaijiEngine::lm_head_forward(const torch::Tensor& hidden) {
    // hidden: [batch, seq, hidden_size] → logits: [batch, seq, vocab_size]
    auto h = hidden.reshape({-1, config_.hidden_size});
    auto logits = torch::mm(h, weights_.lm_head.t());
    return logits.reshape({hidden.size(0), hidden.size(1), config_.vocab_size});
}

torch::Tensor TaijiEngine::make_causal_mask(int seq_len, int total_len, torch::Device device) {
    // [1, 1, seq_len, total_len] 下三角掩码
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto mask = torch::full({seq_len, total_len}, -std::numeric_limits<float>::infinity(), opts);
    auto row_idx = torch::arange(seq_len, opts).unsqueeze(1);
    auto col_idx = torch::arange(total_len, opts).unsqueeze(0);
    auto causal = col_idx <= (total_len - seq_len + row_idx);
    mask.masked_fill_(causal, 0.0f);
    return mask.unsqueeze(0).unsqueeze(0);
}

int64_t TaijiEngine::sample_top_p(const torch::Tensor& logits, float temperature, float top_p) {
    // logits: [vocab_size]
    auto scaled = logits / std::max(temperature, 1e-6f);

    // Top-P 采样
    auto [sorted_logits, sorted_indices] = torch::sort(scaled, /*dim=*/-1, /*descending=*/true);
    auto cumulative_probs = torch::cumsum(torch::softmax(sorted_logits, -1), -1);

    // 移除累积概率超过 top_p 的 token
    auto sorted_mask = cumulative_probs > top_p;
    sorted_mask.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None)},
                           sorted_mask.index({"...", torch::indexing::Slice(torch::indexing::None, -1)}).clone());
    sorted_mask.index_put_({"...", 0}, false);

    // 将被移除的 token 设为 -inf
    auto indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask);
    scaled.index_fill_(-1, torch::where(indices_to_remove)[0], -std::numeric_limits<float>::infinity());

    // 采样
    auto probs = torch::softmax(scaled, -1);
    auto token = torch::multinomial(probs, 1, false, rng_);
    return token.item<int64_t>();
}

// ════════════════════════════════════════════════════════════════
// 推理接口
// ════════════════════════════════════════════════════════════════

torch::Tensor TaijiEngine::forward_ids(
    const std::vector<int64_t>& input_ids,
    bool use_cache)
{
    if (!loaded_) {
        throw std::runtime_error("TaijiEngine: 权重未 loaded, call load_state_dict() first");
    }

    torch::NoGradGuard no_grad;
    auto device = torch::Device(device_);

    auto opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto ids = torch::tensor(input_ids, opts).unsqueeze(0);  // [1, seq]
    auto h = embedding_forward(ids);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> new_kv_cache;

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        const auto* kv_in = (use_cache && kv_cache_[i].first.defined()) ? &kv_cache_[i] : nullptr;
        auto [h_out, new_kv] = transformer_block_forward(
            h, weights_.layers[i], i, kv_in, use_cache);
        h = h_out;
        if (use_cache) {
            new_kv_cache.push_back(new_kv);
        }
    }

    // 最终 norm
    h = rms_norm_forward(h, weights_.final_norm, config_.rms_norm_eps);

    // 更新 KV Cache
    if (use_cache) {
        kv_cache_ = new_kv_cache;
        cache_len_ = kv_cache_[0].first.size(1);
    }

    // LM Head
    return lm_head_forward(h);
}

torch::Tensor TaijiEngine::forward_step(
    const std::vector<int64_t>& input_ids,
    bool use_cache)
{
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::Device(device_));
    auto ids = torch::tensor(input_ids, opts).unsqueeze(0);  // [1, seq]
    auto logits = forward_ids(input_ids, use_cache);
    return logits[0].index({-1});  // [vocab_size] 最后一个 token 的 logits
}

std::vector<int64_t> TaijiEngine::generate(
    const std::vector<int64_t>& input_ids,
    const GenerateConfig& gen_config)
{
    if (!loaded_) {
        throw std::runtime_error("TaijiEngine: 权重未加载");
    }

    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::Device(device_));

    // 初始化输入
    auto ids_tensor = torch::tensor(input_ids, opts).unsqueeze(0);  // [1, seq]
    int prompt_len = ids_tensor.size(1);

    // 重置缓存（新生成开始）
    reset_cache();

    // 首次前向：处理整个 prompt
    auto logits = forward_ids(input_ids, /*use_cache=*/true);

    std::vector<int64_t> generated;
    generated.reserve(gen_config.max_new_tokens);

    // 自回归生成循环
    // NOTE: 如需极致性能优化，可考虑:
    //   1. CUDA Graphs 捕获静态拓扑降低 kernel launch overhead
    //   2. 预分配 KV Cache buffer 避免 decode 时反复 cat
    //   3.使用 torch.compile / torch.fx 做 graph 级别优化
    auto last_logits = logits[0].index({-1});  // [vocab_size]

    for (int step = 0; step < gen_config.max_new_tokens; ++step) {
        // 重复惩罚
        if (gen_config.repetition_penalty > 1.0f && !generated.empty()) {
            // 获取最近生成的 token（去重）
            auto recent = generated;
            // 简单实现：对最近 64 个 token 施加惩罚
            int pen_start = std::max(0, (int)recent.size() - 64);
            for (int i = pen_start; i < (int)recent.size(); ++i) {
                last_logits[recent[i]] /= gen_config.repetition_penalty;
            }
        }

        // 采样
        int64_t next_token = sample_top_p(last_logits, gen_config.temperature, gen_config.top_p);

        // EOS 检查
        if (gen_config.eos_token_id >= 0 && next_token == gen_config.eos_token_id) {
            break;
        }
        // 特殊 token 检查
        if (gen_config.stop_on_token >= 0 && next_token == gen_config.stop_on_token) {
            break;
        }

        generated.push_back(next_token);
        generated_history_.push_back(next_token);

        // 下一步前向
        std::vector<int64_t> next_ids = {next_token};
        auto next_logits = forward_ids(next_ids, /*use_cache=*/true);
        last_logits = next_logits[0].index({-1});
    }

    return generated;
}

std::vector<std::vector<int64_t>> TaijiEngine::generate_batched(
    const std::vector<int64_t>& input_ids,
    const GenerateConfig& gen_config)
{
    // 与 generate 相同，但每 tokens_per_yield 个 token 返回一批
    if (!loaded_) {
        throw std::runtime_error("TaijiEngine: 权重未加载");
    }

    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::Device(device_));

    reset_cache();

    auto logits = forward_ids(input_ids, true);
    auto last_logits = logits[0].index({-1});

    std::vector<std::vector<int64_t>> batches;
    std::vector<int64_t> current_batch;
    int batch_size = std::max(1, gen_config.tokens_per_yield);

    for (int step = 0; step < gen_config.max_new_tokens; ++step) {
        if (gen_config.repetition_penalty > 1.0f && !current_batch.empty()) {
            auto& hist = current_batch;
            int pen_start = std::max(0, (int)hist.size() - 64);
            for (int i = pen_start; i < (int)hist.size(); ++i) {
                last_logits[hist[i]] /= gen_config.repetition_penalty;
            }
        }

        int64_t next_token = sample_top_p(last_logits, gen_config.temperature, gen_config.top_p);

        if (gen_config.eos_token_id >= 0 && next_token == gen_config.eos_token_id) break;
        if (gen_config.stop_on_token >= 0 && next_token == gen_config.stop_on_token) break;

        current_batch.push_back(next_token);

        if ((int)current_batch.size() >= batch_size) {
            batches.push_back(std::move(current_batch));
            current_batch.clear();
        }

        std::vector<int64_t> next_ids = {next_token};
        auto next_logits = forward_ids(next_ids, true);
        last_logits = next_logits[0].index({-1});
    }

    // 剩余 token
    if (!current_batch.empty()) {
        batches.push_back(std::move(current_batch));
    }

    return batches;
}

} // namespace taiji