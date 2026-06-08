/**
 * Taiji Weight Loader — 实现
 *
 * 从 PyTorch state_dict 加载权重到 C++ ModelWeights。
 * 权重名称与 taiji/architecture.py 的 model.state_dict() 一一对应。
 */

#include "weight_loader.h"
#include <algorithm>
#include <sstream>
#include <iostream>

namespace taiji {

// ════════════════════════════════════════════════════════════════
// 公开接口
// ════════════════════════════════════════════════════════════════

ModelWeights WeightLoader::load_from_state_dict(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const ModelConfig& config,
    const std::string& device)
{
    ModelWeights weights;

    // ── Embedding & Norm ──
    weights.embedding = to_device(find_tensor(state_dict, "backbone.embedding.weight"), device);
    weights.final_norm = to_device(find_tensor(state_dict, "backbone.norm.weight"), device);

    // lm_head 与 embedding 权重绑定 (weight tying)
    auto lm_head_w = find_tensor(state_dict, "lm_head.weight");
    if (lm_head_w.defined()) {
        weights.lm_head = to_device(lm_head_w, device);
    } else {
        // 权重绑定: lm_head 共享 embedding
        weights.lm_head = weights.embedding;
    }

    // ── 逐层加载 ──
    weights.layers.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        weights.layers.push_back(load_layer(state_dict, i, device));
    }

    // ── 多头加载 ──
    weights.tool_head = load_tool_head(state_dict, device);
    weights.memory_head = load_memory_head(state_dict, device);
    weights.plan_head = load_plan_head(state_dict, device);
    weights.perception_head = load_perception_head(state_dict, device);

    return weights;
}

std::vector<std::string> WeightLoader::validate(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const ModelConfig& config)
{
    std::vector<std::string> missing;

    // 基础权重
    auto check = [&](const std::string& key) {
        if (state_dict.find(key) == state_dict.end()) {
            missing.push_back(key);
        }
    };

    check("backbone.embedding.weight");
    check("backbone.norm.weight");

    // 逐层检查
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        std::string prefix = "backbone.layers." + std::to_string(i);
        check(prefix + ".attention.wq.weight");
        check(prefix + ".attention.wk.weight");
        check(prefix + ".attention.wv.weight");
        check(prefix + ".attention.wo.weight");
        check(prefix + ".attention_norm.weight");
        check(prefix + ".feed_forward.w1.weight");
        check(prefix + ".feed_forward.w_gate.weight");
        check(prefix + ".feed_forward.w2.weight");
        check(prefix + ".ffn_norm.weight");
    }

    // 多头检查（可选：不报错，只警告）
    // tool_head, memory_head, plan_head, perception_head 的权重
    // 可能不存在（如果模型未初始化这些头）

    return missing;
}

torch::Tensor WeightLoader::to_device(torch::Tensor t, const std::string& device) {
    if (!t.defined()) return t;
    // 确保 contiguous + 转移到目标设备
    auto opts = torch::TensorOptions().device(torch::Device(device));
    return t.contiguous().to(opts);
}

// ════════════════════════════════════════════════════════════════
// 内部实现
// ════════════════════════════════════════════════════════════════

LayerWeights WeightLoader::load_layer(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    int layer_idx,
    const std::string& device)
{
    std::string prefix = "backbone.layers." + std::to_string(layer_idx);

    LayerWeights lw;
    // Attention
    lw.wq        = to_device(find_tensor(state_dict, prefix + ".attention.wq.weight"), device);
    lw.wk        = to_device(find_tensor(state_dict, prefix + ".attention.wk.weight"), device);
    lw.wv        = to_device(find_tensor(state_dict, prefix + ".attention.wv.weight"), device);
    lw.wo        = to_device(find_tensor(state_dict, prefix + ".attention.wo.weight"), device);
    lw.attn_norm = to_device(find_tensor(state_dict, prefix + ".attention_norm.weight"), device);

    // FFN (SwiGLU)
    lw.w1        = to_device(find_tensor(state_dict, prefix + ".feed_forward.w1.weight"), device);
    lw.w_gate    = to_device(find_tensor(state_dict, prefix + ".feed_forward.w_gate.weight"), device);
    lw.w2        = to_device(find_tensor(state_dict, prefix + ".feed_forward.w2.weight"), device);
    lw.ffn_norm  = to_device(find_tensor(state_dict, prefix + ".ffn_norm.weight"), device);

    return lw;
}

ToolHeadWeights WeightLoader::load_tool_head(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& device)
{
    std::string prefix = "tool_head.mlp";
    ToolHeadWeights th;

    th.mlp_weight_0 = to_device(find_tensor(state_dict, prefix + ".0.weight"), device);
    th.mlp_bias_0   = to_device(find_tensor(state_dict, prefix + ".0.bias"), device);
    th.mlp_weight_1 = to_device(find_tensor(state_dict, prefix + ".2.weight"), device);
    th.mlp_bias_1   = to_device(find_tensor(state_dict, prefix + ".2.bias"), device);

    return th;
}

MemoryHeadWeights WeightLoader::load_memory_head(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& device)
{
    std::string prefix = "memory_head";
    MemoryHeadWeights mh;

    mh.memory_keys   = to_device(find_tensor(state_dict, prefix + ".memory_keys"), device);
    mh.memory_values = to_device(find_tensor(state_dict, prefix + ".memory_values"), device);
    mh.query_proj    = to_device(find_tensor(state_dict, prefix + ".query_proj.weight"), device);

    // Read gate
    mh.read_gate_w0 = to_device(find_tensor(state_dict, prefix + ".read_gate.0.weight"), device);
    mh.read_gate_b0 = to_device(find_tensor(state_dict, prefix + ".read_gate.0.bias"), device);
    mh.read_gate_w1 = to_device(find_tensor(state_dict, prefix + ".read_gate.2.weight"), device);
    mh.read_gate_b1 = to_device(find_tensor(state_dict, prefix + ".read_gate.2.bias"), device);

    // Write gate
    mh.write_gate_w0 = to_device(find_tensor(state_dict, prefix + ".write_gate.0.weight"), device);
    mh.write_gate_b0 = to_device(find_tensor(state_dict, prefix + ".write_gate.0.bias"), device);
    mh.write_gate_w1 = to_device(find_tensor(state_dict, prefix + ".write_gate.2.weight"), device);
    mh.write_gate_b1 = to_device(find_tensor(state_dict, prefix + ".write_gate.2.bias"), device);

    // Value proj & output proj
    mh.value_proj   = to_device(find_tensor(state_dict, prefix + ".value_proj.weight"), device);
    mh.output_proj  = to_device(find_tensor(state_dict, prefix + ".output_proj.weight"), device);

    return mh;
}

PlanHeadWeights WeightLoader::load_plan_head(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& device)
{
    std::string prefix = "plan_head.mlp";
    PlanHeadWeights ph;

    ph.mlp_weight_0 = to_device(find_tensor(state_dict, prefix + ".0.weight"), device);
    ph.mlp_bias_0   = to_device(find_tensor(state_dict, prefix + ".0.bias"), device);
    ph.mlp_weight_1 = to_device(find_tensor(state_dict, prefix + ".2.weight"), device);
    ph.mlp_bias_1   = to_device(find_tensor(state_dict, prefix + ".2.bias"), device);

    return ph;
}

PerceptionHeadWeights WeightLoader::load_perception_head(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& device)
{
    std::string prefix = "perception_head.mlp";
    PerceptionHeadWeights ph;

    ph.mlp_weight_0 = to_device(find_tensor(state_dict, prefix + ".0.weight"), device);
    ph.mlp_bias_0   = to_device(find_tensor(state_dict, prefix + ".0.bias"), device);
    ph.mlp_weight_1 = to_device(find_tensor(state_dict, prefix + ".2.weight"), device);
    ph.mlp_bias_1   = to_device(find_tensor(state_dict, prefix + ".2.bias"), device);

    return ph;
}

torch::Tensor WeightLoader::find_tensor(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& key)
{
    auto it = state_dict.find(key);
    if (it != state_dict.end()) {
        return it->second;
    }
    // 返回 undefined tensor（调用方检查 .defined()）
    return torch::Tensor();
}

} // namespace taiji