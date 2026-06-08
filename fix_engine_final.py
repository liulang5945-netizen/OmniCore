import re

with open('csrc/core/engine.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the stray "------- REPLACE" line
content = content.replace('------- REPLACE\n\n\n    // 输出投影', '    // 输出投影')

# 2. Fix torch::nn::functional::scaled_dot_product_attention -> at::scaled_dot_product_attention
# Prefill case (causal=true)
content = content.replace(
        '''attn_out = torch::nn::functional::scaled_dot_product_attention(
            xq, k_full, v_full,
            torch::nn::functional::ScaledDotProductAttentionFuncOptions()
                .is_causal(true)
        );''',
        '''attn_out = at::scaled_dot_product_attention(
            xq, k_full, v_full,
            c10::optional<at::Tensor>(),
            0.0,
            true
        );''')

# Decode case (no causal mask)
content = content.replace(
        '''attn_out = torch::nn::functional::scaled_dot_product_attention(
            xq, k_full, v_full
        );''',
        '''attn_out = at::scaled_dot_product_attention(
            xq, k_full, v_full
        );''')

with open('csrc/core/engine.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed.")
