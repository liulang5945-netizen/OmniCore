#pragma once
/**
 * Taiji CUDA Inference Engine — 核心引擎
 *
 * 整个推理循环在 C++ 侧运行，通过 pybind11 暴露给 Python。
 * 推理时释放 GIL，训练时可选持有。
 *
 * Phase 1: 基础推理循环（libtorch 侧，后续 Phase 2+ 添加融合 CUDA kernel）
 * Phase 2: Triton 融合前向 kernels
 * Phase 3: 融合交叉熵 + 采样
 * Phase 4: CUDA Graphs + 全 5 个 Head
 * Phase 5: PagedAttention
 */

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>

#include "model_config.h"
#include "weight_loader.h"
#include "paged_attention.h"

namespace taiji {

class TaijiEngine {
public:
    /**
     * 构造引擎。
     * @param config 模型配置
     * @param device 目标设备 ("cpu" 或 "cuda:0")
     */
    explicit TaijiEngine(const ModelConfig& config, const std::string& device = "cpu");

    ~TaijiEngine() = default;

    // ═══════════════════════════════════════════════════════════
    // 权重管理
    // ═══════════════════════════════════════════════════════════

    /**
     * 从 Python dict 加载权重（运行时调用）。
     * Python 侧: engine.load_state_dict(model.state_dict())
     */
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict);

    /**
     * 权重是否已加载。
     */
    bool is_loaded() const { return loaded_; }

    /**
     * 获取模型配置。
     */
    const ModelConfig& config() const { return config_; }

    /**
     * 获取设备。
     */
    const std::string& device() const { return device_; }

    /**
     * 统计参数量。
     */
    int64_t num_parameters() const;

    // ═══════════════════════════════════════════════════════════
    // 推理接口
    // ═══════════════════════════════════════════════════════════

    /**
     * 自回归生成 — 完整循环在 C++ 侧，释放 GIL。
     *
     * @param input_ids 输入 token ID 列表
     * @param config 生成配置
     * @return 生成的 token ID 列表（不含输入）
     */
    std::vector<int64_t> generate(
        const std::vector<int64_t>& input_ids,
        const GenerateConfig& gen_config
    );

    /**
     * 流式生成 — 按批次返回 token IDs。
     *
     * @param input_ids 输入 token ID 列表
     * @param config 生成配置
     * @return 每批 token ID 列表的向量
     */
    std::vector<std::vector<int64_t>> generate_batched(
        const std::vector<int64_t>& input_ids,
        const GenerateConfig& gen_config
    );

    /**
     * 单步前向传播 — 返回最后一个 token 的 logits。
     * 用于 Python 侧的自定义推理循环。
     *
     * @param input_ids 输入 token IDs
     * @param use_cache 是否使用/返回 KV Cache
     * @return logits [1, vocab_size]
     */
    torch::Tensor forward_step(
        const std::vector<int64_t>& input_ids,
        bool use_cache = true
    );

    /**
     * 完整前向传播 — 所有位置的 logits。
     * 用于训练。
     *
     * @param input_ids [batch, seq_len] 输入 token 张量
     * @param use_cache 是否返回 KV Cache
     * @return logits [batch, seq_len, vocab_size]
     */
    torch::Tensor forward_ids(
        const std::vector<int64_t>& input_ids,
        bool use_cache = false
    );

    // ═══════════════════════════════════════════════════════════
    // KV Cache 管理
    // ═══════════════════════════════════════════════════════════

    /// 重置 KV Cache（新对话开始时调用）
    void reset_cache();

    /// 获取当前 KV Cache 的序列长度
    int cache_seq_len() const;

    // ═══════════════════════════════════════════════════════════
    // 工具注册
    // ═══════════════════════════════════════════════════════════

    /// 设置活跃工具数量（更新 tool_head）
    void set_num_tools(int n);

private:
    // ═══════════════════════════════════════════════════════════
    // 内部前向传播组件
    // ═══════════════════════════════════════════════════════════

    /// Embedding 前向
    torch::Tensor embedding_forward(const torch::Tensor& input_ids);

    /// RMSNorm 前向 (Phase 2: 替换为融合 kernel)
    torch::Tensor rms_norm_forward(
        const torch::Tensor& x,
        const torch::Tensor& weight,
        float eps
    );

    /// RoPE 前向 (Phase 2: 替换为融合 kernel)
    void apply_rope(
        torch::Tensor& q,    // [batch, seq, heads, head_dim]
        torch::Tensor& k,
        int start_pos
    );

    /// GQA Attention 前向 (Phase 2: 替换为融合 kernel)
    std::pair<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>
    attention_forward(
        const torch::Tensor& x,       // [batch, seq, hidden]
        const LayerWeights& layer,
        int layer_idx,
        const std::pair<torch::Tensor, torch::Tensor>* kv_cache_in,  // 可选输入 KV
        bool use_cache
    );

    /// SwiGLU FFN 前向 (Phase 2: 替换为融合 kernel)
    torch::Tensor swiglu_forward(
        const torch::Tensor& x,
        const LayerWeights& layer
    );

    /// 单层 Transformer 前向
    std::pair<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>
    transformer_block_forward(
        const torch::Tensor& x,
        const LayerWeights& layer,
        int layer_idx,
        const std::pair<torch::Tensor, torch::Tensor>* kv_cache_in,
        bool use_cache
    );

    /// LM Head 前向
    torch::Tensor lm_head_forward(const torch::Tensor& hidden);

    /// 因果掩码生成
    torch::Tensor make_causal_mask(int seq_len, int total_len, torch::Device device);

    /// Top-P 采样 (Phase 3: 替换为融合 kernel)
    int64_t sample_top_p(const torch::Tensor& logits, float temperature, float top_p);

    /// RoPE 缓存
    std::pair<torch::Tensor, torch::Tensor> get_rope_sin_cos(int seq_len, int start_pos);

    // ═══════════════════════════════════════════════════════════
    // 成员变量
    // ═══════════════════════════════════════════════════════════

    ModelConfig config_;
    std::string device_;
    bool loaded_ = false;

    // 模型权重（C++ 侧持有）
    ModelWeights weights_;

    // KV Cache: 每层一对 (K, V)
    // K: [batch, cached_len, kv_heads, head_dim]
    // V: [batch, cached_len, kv_heads, head_dim]
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache_;
    int cache_len_ = 0;

    // KV Cache 预分配 buffer 管理
    static constexpr int KV_CACHE_INITIAL_CAPACITY = 4096;
    int kv_cache_capacity_ = 0;  // 当前预分配的序列长度
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache_buffers_;  // 预分配 buffer

    // RoPE 缓存
    std::pair<torch::Tensor, torch::Tensor> rope_cache_;
    int rope_cache_len_ = 0;

    // 线程安全
    std::mutex mutex_;

    // 重复惩罚用的历史 token
    std::vector<int64_t> generated_history_;

    // 随机数生成器
    torch::Generator rng_;

    // PagedAttention (Phase 5+): 可选启用分页 KV Cache
    bool use_paged_attention_ = false;
    BlockAllocator block_allocator_;  // 物理块池
    std::vector<BlockTable> block_tables_;  // 每层一个块表
};

} // namespace taiji