#pragma once
/**
 * Taiji Model Configuration — C++ 侧
 *
 * 动态支持 125M → 350M → 1B → 3B → 7B → 13B+ 所有规模。
 * 与 taiji/config.py 的 ModelConfig 保持一一对应。
 */

#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <torch/torch.h>

namespace taiji {

struct ModelConfig {
    // ═══════════ 基础参数 ═══════════
    int vocab_size          = 33000;    // 32000 基础词表 + 1000 特殊 token
    int hidden_size         = 768;
    int intermediate_size   = 2048;     // FFN 中间层维度
    int num_hidden_layers   = 12;
    int num_attention_heads = 12;
    int num_key_value_heads = 12;       // GQA: KV heads ≤ Q heads

    // ═══════════ 序列参数 ═══════════
    int max_position_embeddings = 4096;

    // ═══════════ 归一化 ═══════════
    float rms_norm_eps = 1e-5f;

    // ═══════════ RoPE ═══════════
    float rope_theta = 500000.0f;

    // ═══════════ 多头 Agent 参数 ═══════════
    int num_tools          = 0;     // 工具头活跃工具数（运行时动态设置）
    int max_tools          = 750;   // 工具头最大容量
    int num_plan_actions   = 8;     // 规划头动作数
    int num_env_tokens     = 200;   // 感知头环境 token 数
    int num_memory_slots   = 30;    // 记忆头槽位数（20 短期 + 10 长期）
    int memory_dim         = 64;    // 记忆向量维度

    // ═══════════ MoE 参数（M12 添加）═══════════
    int num_experts        = 1;     // 1 = Dense, >1 = MoE
    int num_experts_per_tok = 1;    // top-K routing

    // ═══════════ MTP 参数（M11 添加）═══════════
    int num_mtp_heads      = 1;     // 1 = 单 token 预测, 4 = 多 token 预测

    // ═══════════ MLA 参数（M13 添加）═══════════
    int latent_dim         = 0;     // 0 = 标准 GQA, >0 = MLA
    int rope_dim           = 64;    // MLA 中 RoPE 作用的维度

    // ═══════════ 计算属性 ═══════════

    /// 每个注意力头的维度
    int head_dim() const {
        return hidden_size / num_attention_heads;
    }

    /// 每个 KV head 对应多少个 query head
    int num_queries_per_kv() const {
        return num_attention_heads / num_key_value_heads;
    }

    /// 是否使用 GQA（KV heads < Q heads）
    bool use_gqa() const {
        return num_key_value_heads < num_attention_heads;
    }

    /// 是否使用 MoE
    bool use_moe() const {
        return num_experts > 1;
    }

    /// 是否使用 MLA
    bool use_mla() const {
        return latent_dim > 0;
    }

    /// 是否使用 MTP
    bool use_mtp() const {
        return num_mtp_heads > 1;
    }

    /// 估算参数量
    long long count_parameters() const {
        long long embed = (long long)vocab_size * hidden_size;
        // 每层: attention(Q+K+V+O) + FFN(W1+W_gate+W2) + 2*norm
        int hd = head_dim();
        long long attn = (long long)hidden_size * num_attention_heads * hd    // Q
                       + (long long)hidden_size * num_key_value_heads * hd    // K
                       + (long long)hidden_size * num_key_value_heads * hd    // V
                       + (long long)num_attention_heads * hd * hidden_size;   // O
        long long ffn = (long long)hidden_size * intermediate_size    // W1
                      + (long long)hidden_size * intermediate_size    // W_gate
                      + (long long)intermediate_size * hidden_size;   // W2
        long long norm = (long long)hidden_size * 2;  // attention_norm + ffn_norm
        long long per_layer = attn + ffn + norm;

        // MoE: 每个 expert 有自己的 FFN
        if (use_moe()) {
            per_layer = attn + ffn * num_experts + norm;
        }

        long long total = embed + per_layer * num_hidden_layers;

        // 多头参数（固定，不随模型规模变化大）
        total += (long long)hidden_size * max_tools;        // tool_head
        total += (long long)hidden_size * num_plan_actions;  // plan_head
        total += (long long)hidden_size * 200;               // perception_head
        total += (long long)num_memory_slots * memory_dim * 2; // memory_head keys+values
        total += (long long)hidden_size * num_memory_slots;  // memory_head projections

        return total;
    }

    /// 人类可读描述
    std::string describe() const {
        long long params = count_parameters();
        std::string size_str;
        if (params >= (long long)1e9)
            size_str = std::to_string(params / (long long)1e9) + "B";
        else if (params >= (long long)1e6)
            size_str = std::to_string(params / (long long)1e6) + "M";
        else
            size_str = std::to_string(params / (long long)1e3) + "K";

        return "ModelSelf-" + size_str
             + " | hidden=" + std::to_string(hidden_size)
             + " layers=" + std::to_string(num_hidden_layers)
             + " heads=" + std::to_string(num_attention_heads)
             + " kv=" + std::to_string(num_key_value_heads)
             + " ffn=" + std::to_string(intermediate_size)
             + " vocab=" + std::to_string(vocab_size);
    }

    // ═══════════ 预定义配置工厂 ═══════════

    static ModelConfig size_125m() {
        ModelConfig c;
        c.hidden_size         = 768;
        c.intermediate_size   = 2048;
        c.num_hidden_layers   = 12;
        c.num_attention_heads = 12;
        c.num_key_value_heads = 12;
        return c;
    }

    static ModelConfig size_350m() {
        ModelConfig c;
        c.hidden_size         = 1024;
        c.intermediate_size   = 2816;
        c.num_hidden_layers   = 24;
        c.num_attention_heads = 16;
        c.num_key_value_heads = 16;
        return c;
    }

    static ModelConfig size_1b() {
        ModelConfig c;
        c.hidden_size         = 2048;
        c.intermediate_size   = 5504;
        c.num_hidden_layers   = 22;
        c.num_attention_heads = 32;
        c.num_key_value_heads = 4;
        return c;
    }

    static ModelConfig size_3b() {
        ModelConfig c;
        c.hidden_size         = 3200;
        c.intermediate_size   = 8640;
        c.num_hidden_layers   = 26;
        c.num_attention_heads = 32;
        c.num_key_value_heads = 4;
        return c;
    }

    static ModelConfig size_7b() {
        ModelConfig c;
        c.hidden_size         = 4096;
        c.intermediate_size   = 11008;
        c.num_hidden_layers   = 32;
        c.num_attention_heads = 32;
        c.num_key_value_heads = 4;
        return c;
    }

    static ModelConfig size_13b() {
        ModelConfig c;
        c.hidden_size         = 5120;
        c.intermediate_size   = 13824;
        c.num_hidden_layers   = 40;
        c.num_attention_heads = 40;
        c.num_key_value_heads = 8;
        return c;
    }

    /// 从字符串解析配置
    static ModelConfig from_string(const std::string& size) {
        if (size == "125m") return size_125m();
        if (size == "350m") return size_350m();
        if (size == "1b")   return size_1b();
        if (size == "3b")   return size_3b();
        if (size == "7b")   return size_7b();
        if (size == "13b")  return size_13b();
        throw std::invalid_argument("Unknown model size: " + size
            + ". Available: 125m, 350m, 1b, 3b, 7b, 13b");
    }
};

// ═══════════ 推理配置 ═══════════

struct GenerateConfig {
    int   max_new_tokens      = 256;
    float temperature         = 0.7f;
    float top_p               = 0.9f;
    float repetition_penalty  = 1.2f;
    int   eos_token_id        = -1;
    int   stop_on_token       = -1;    // tool_call / answer token
    int   batch_size          = 1;     // 连续批处理并发数
    int   tokens_per_yield    = 8;     // 流式生成每批 token 数
};

// ═══════════ 模型输出 ═══════════

struct ModelOutputForward {
    torch::Tensor logits;                      // 语言头 [batch, seq, vocab]
    torch::Tensor tool_logits;                 // 工具头 [batch, num_tools]
    torch::Tensor arg_presence;                // 参数存在性 [batch, 4]
    std::vector<torch::Tensor> arg_types;      // 每个槽位类型
    std::vector<torch::Tensor> arg_values;     // 每个槽位值
    torch::Tensor memory_attn;                 // 记忆注意力权重
    torch::Tensor perception_logits;           // 感知头
    torch::Tensor plan_logits;                 // 规划头
    torch::Tensor predicted_steps;             // 预测步数
    torch::Tensor difficulty;                  // 难度分
    torch::Tensor loss;                        // 训练损失

    // KV Cache (每层一对 K, V)
    std::vector<std::pair<torch::Tensor, torch::Tensor>> kv_cache;

    bool has_loss       = false;
    bool has_tool       = false;
    bool has_memory     = false;
    bool has_plan       = false;
    bool has_perception = false;
};

} // namespace taiji