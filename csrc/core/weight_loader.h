#pragma once
/**
 * Taiji Weight Loader
 *
 * 从 PyTorch state_dict (Python dict of name → Tensor) 加载权重到 C++ 引擎。
 * 零拷贝：直接使用 tensor.data_ptr()，不复制数据。
 *
 * 支持:
 *   - PyTorch .pt / .pth checkpoint
 *   - safetensors 格式 (通过 Python 侧预加载)
 *   - 运行时从 Python dict 传入
 */

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "model_config.h"

namespace taiji {

/// 每层 Transformer 的权重集合
struct LayerWeights {
    // Attention
    torch::Tensor wq;    // [hidden, heads * head_dim]
    torch::Tensor wk;    // [hidden, kv_heads * head_dim]
    torch::Tensor wv;    // [hidden, kv_heads * head_dim]
    torch::Tensor wo;    // [heads * head_dim, hidden]
    torch::Tensor attn_norm;  // [hidden]

    // FFN (SwiGLU)
    torch::Tensor w1;      // [hidden, intermediate]
    torch::Tensor w_gate;  // [hidden, intermediate]
    torch::Tensor w2;      // [intermediate, hidden]
    torch::Tensor ffn_norm; // [hidden]
};

/// 工具头权重
struct ToolHeadWeights {
    torch::Tensor mlp_weight_0;     // [hidden, hidden//2]
    torch::Tensor mlp_bias_0;
    torch::Tensor mlp_weight_1;     // [hidden//2, num_tools]
    torch::Tensor mlp_bias_1;
};

/// 记忆头权重
struct MemoryHeadWeights {
    torch::Tensor memory_keys;      // [num_slots, memory_dim]
    torch::Tensor memory_values;    // [num_slots, memory_dim]
    torch::Tensor query_proj;       // [hidden, memory_dim]
    torch::Tensor read_gate_w0;
    torch::Tensor read_gate_b0;
    torch::Tensor read_gate_w1;
    torch::Tensor read_gate_b1;
    torch::Tensor write_gate_w0;
    torch::Tensor write_gate_b0;
    torch::Tensor write_gate_w1;
    torch::Tensor write_gate_b1;
    torch::Tensor value_proj;       // [hidden, memory_dim]
    torch::Tensor output_proj;      // [memory_dim, hidden]
};

/// 规划头权重
struct PlanHeadWeights {
    torch::Tensor mlp_weight_0;
    torch::Tensor mlp_bias_0;
    torch::Tensor mlp_weight_1;
    torch::Tensor mlp_bias_1;
};

/// 感知头权重
struct PerceptionHeadWeights {
    torch::Tensor mlp_weight_0;
    torch::Tensor mlp_bias_0;
    torch::Tensor mlp_weight_1;
    torch::Tensor mlp_bias_1;
};

/// 全部模型权重
struct ModelWeights {
    // Embedding
    torch::Tensor embedding;        // [vocab_size, hidden_size]
    torch::Tensor final_norm;       // [hidden_size]

    // LM Head (与 embedding 权重绑定)
    torch::Tensor lm_head;          // [hidden_size, vocab_size] (通常与 embedding 共享)

    // 各层权重
    std::vector<LayerWeights> layers;

    // 多头权重
    ToolHeadWeights        tool_head;
    MemoryHeadWeights      memory_head;
    PlanHeadWeights        plan_head;
    PerceptionHeadWeights  perception_head;
};


class WeightLoader {
public:
    /**
     * 从 Python 传入的 state_dict 加载权重。
     *
     * state_dict 是 name → Tensor 的映射，名称与 PyTorch model.state_dict() 一致。
     * 例如: "backbone.embedding.weight", "backbone.layers.0.attention.wq.weight"
     *
     * @param state_dict 权重字典
     * @param config 模型配置
     * @param device 目标设备 ("cpu" 或 "cuda:0")
     * @return 完整的 ModelWeights
     */
    static ModelWeights load_from_state_dict(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const ModelConfig& config,
        const std::string& device = "cpu"
    );

    /**
     * 验证权重完整性：检查所有必需的权重是否存在、shape 是否匹配。
     * @return 缺失/不匹配的权重名称列表（空 = 完整）
     */
    static std::vector<std::string> validate(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const ModelConfig& config
    );

    /**
     * 将权重转移到目标设备，并确保 contiguous。
     */
    static torch::Tensor to_device(torch::Tensor t, const std::string& device);

private:
    /// 从 state_dict 提取一层的权重
    static LayerWeights load_layer(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        int layer_idx,
        const std::string& device
    );

    /// 从 state_dict 提取工具头权重
    static ToolHeadWeights load_tool_head(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& device
    );

    /// 从 state_dict 提取记忆头权重
    static MemoryHeadWeights load_memory_head(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& device
    );

    /// 从 state_dict 提取规划头权重
    static PlanHeadWeights load_plan_head(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& device
    );

    /// 从 state_dict 提取感知头权重
    static PerceptionHeadWeights load_perception_head(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& device
    );

    /// 辅助: 从 dict 中查找 key，支持前缀模糊匹配
    static torch::Tensor find_tensor(
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& key
    );
};

} // namespace taiji