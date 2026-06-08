"""
模型注册数据定义
包含量化级别、模型条目 dataclass 以及完整的 GGUF + HF 模型注册表
单独提取以避免 model_registry.py 过于臃肿
"""
from dataclasses import dataclass, field
from typing import List, Optional

# ======================== 量化级别定义 ========================

@dataclass
class QuantInfo:
    """量化级别信息"""
    name: str
    bytes_per_param: float
    quality_score: int
    description: str

QUANT_LEVELS = {
    "Q2_K":    QuantInfo("Q2_K",    0.35, 3, "极限压缩，质量损失大"),
    "Q3_K_S":  QuantInfo("Q3_K_S",  0.45, 4, "小模型适中压缩"),
    "Q3_K_M":  QuantInfo("Q3_K_M",  0.50, 5, "中等压缩，均衡"),
    "Q4_K_S":  QuantInfo("Q4_K_S",  0.58, 6, "4-bit 小版，质量尚可"),
    "Q4_K_M":  QuantInfo("Q4_K_M",  0.65, 7, "✅ 推荐 — 质量与内存最佳平衡"),
    "Q5_K_M":  QuantInfo("Q5_K_M",  0.80, 8, "5-bit 中等，质量较好"),
    "Q6_K":    QuantInfo("Q6_K",    0.90, 8, "6-bit 高质量"),
    "Q8_0":    QuantInfo("Q8_0",    1.10, 9, "8-bit 几乎无损"),
    "F16":     QuantInfo("F16",     2.00, 10, "半精度浮点，原始质量"),
    "BF16":    QuantInfo("BF16",    2.00, 10, "BFloat16 — 完整模型，可微调训练"),
    "full":    QuantInfo("full",    2.00, 10, "HF 完整模型，用于微调训练"),
}

# ======================== 模型条目定义 ========================

@dataclass
class ModelVariant:
    """单个模型变体（特定量化版本）"""
    quant: str
    vram_gb: float
    hf_filename: str
    is_recommended: bool = False

@dataclass
class ModelEntry:
    """模型条目"""
    family: str
    name: str
    params_b: float
    description: str
    hf_repo: str
    variants: List[ModelVariant]
    tags: List[str] = field(default_factory=list)
    model_type: str = "gguf"

    def get_vram(self, quant: str) -> Optional[float]:
        for v in self.variants:
            if v.quant == quant:
                return v.vram_gb
        return None

    def recommended_variant(self) -> Optional[ModelVariant]:
        for v in self.variants:
            if v.is_recommended:
                return v
        return self.variants[0] if self.variants else None

    @property
    def hf_train_repo(self) -> str:
        return self.hf_repo.replace("-GGUF", "")

# ======================== 文件大小估算 ========================

def estimate_file_size_mb(params_b: float, quant: str) -> float:
    """根据参数量和量化级别估算 GGUF 文件大小（MB）"""
    quant_info = QUANT_LEVELS.get(quant, QuantInfo(quant, 0.65, 5, ""))
    bpp = quant_info.bytes_per_param
    size_mb = params_b * bpp * 953.67431640625
    return round(size_mb, 1)

def format_file_size(size_mb: float) -> str:
    """将文件大小格式化为人类可读字符串"""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.0f} MB"

# ======================== GGUF 模型注册表 ========================

MODEL_REGISTRY: List[ModelEntry] = [

    # === 1-3B 级别（6-8GB 内存） ===

    ModelEntry("Qwen2.5", "Qwen2.5-0.5B-Instruct", 0.5,
               "阿里通义0.5B，超轻量级，适合低配设备",
               "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.4, "qwen2.5-0.5b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   0.6, "qwen2.5-0.5b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "轻量"]),

    ModelEntry("Qwen2.5", "Qwen2.5-1.5B-Instruct", 1.5,
               "阿里通义1.5B，小巧实用，中文对话流畅",
               "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.1, "qwen2.5-1.5b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q5_K_M", 1.3, "qwen2.5-1.5b-instruct-q5_k_m.gguf"),
                   ModelVariant("Q8_0",   1.7, "qwen2.5-1.5b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "轻量"]),

    ModelEntry("Llama 3.2", "Llama-3.2-1B-Instruct", 1.0,
               "Meta Llama 3.2 1B，超快推理速度",
               "unsloth/Llama-3.2-1B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.8, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "通用", "轻量"]),

    ModelEntry("Llama 3.2", "Llama-3.2-3B-Instruct", 3.0,
               "Meta Llama 3.2 3B，小模型中的性能标杆",
               "unsloth/Llama-3.2-3B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.2, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   3.5, "Llama-3.2-3B-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "通用"]),

    ModelEntry("Phi-3", "Phi-3-mini-3.8B", 3.8,
               "微软 Phi-3 3.8B，小尺寸大能力，低配首选",
               "microsoft/Phi-3-mini-4k-instruct-gguf",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "Phi-3-mini-4k-instruct-q4.gguf", True),
                   ModelVariant("Q8_0",   4.5, "Phi-3-mini-4k-instruct-q8.gguf"),
               ],
               tags=["英文", "轻量", "高效"]),

    ModelEntry("Qwen2.5", "Qwen2.5-3B-Instruct", 3.0,
               "阿里通义3B，中等规模中文最佳",
               "Qwen/Qwen2.5-3B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.2, "qwen2.5-3b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q5_K_M", 2.7, "qwen2.5-3b-instruct-q5_k_m.gguf"),
                   ModelVariant("Q8_0",   3.5, "qwen2.5-3b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "通用"]),

    ModelEntry("Gemma 2", "Gemma-2-2B", 2.0,
               "Google Gemma 2 2B，轻量且安全",
               "unsloth/gemma-2-2b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.5, "gemma-2-2b-it-Q4_K_M.gguf", True),
               ],
               tags=["英文", "安全"]),

    ModelEntry("TinyLlama", "TinyLlama-1.1B-Chat", 1.1,
               "超轻量 Llama 架构，适合嵌入式设备和快速原型",
               "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.9, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   1.4, "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
               ],
               tags=["英文", "轻量", "嵌入式"]),

    ModelEntry("StableLM", "StableLM-Zephyr-3B", 3.0,
               "Stability AI 的 Zephyr 微调版，对话流畅度高",
               "TheBloke/stablelm-zephyr-3b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.2, "stablelm-zephyr-3b.Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   3.5, "stablelm-zephyr-3b.Q8_0.gguf"),
               ],
               tags=["英文", "对话"]),

    # === 4-6B 级别（8-12GB 内存） ===

    ModelEntry("MiniCPM", "MiniCPM3-4B", 4.0,
               "面壁智能 MiniCPM3 4B，小尺寸强性能",
               "unsloth/MiniCPM3-4B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "MiniCPM3-4B-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   4.6, "MiniCPM3-4B-Q8_0.gguf"),
               ],
               tags=["中文", "轻量", "高效"]),

    ModelEntry("Phi-3.5", "Phi-3.5-mini-3.8B", 3.8,
               "微软 Phi-3.5 3.8B，多语言增强版",
               "microsoft/Phi-3.5-mini-instruct-gguf",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "Phi-3.5-mini-instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "多语言", "轻量"]),

    ModelEntry("Yi-1.5", "Yi-1.5-6B-Chat", 6.0,
               "零一万物 Yi-1.5 6B，中英双语优秀",
               "TheBloke/Yi-1.5-6B-Chat-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "yi-1.5-6b-chat.Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 5.0, "yi-1.5-6b-chat.Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   6.8, "yi-1.5-6b-chat.Q8_0.gguf"),
               ],
               tags=["中文", "英文", "通用"]),

    ModelEntry("InternLM2", "InternLM2-Chat-7B", 7.0,
               "上海AI实验室书生浦语7B，中文能力强",
               "TheBloke/internlm2-chat-7b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "internlm2-chat-7b.Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   8.0, "internlm2-chat-7b.Q8_0.gguf"),
               ],
               tags=["中文", "通用", "学术"]),

    ModelEntry("Baichuan2", "Baichuan2-7B-Chat", 7.0,
               "百川智能Baichuan2 7B，中文理解力强",
               "TheBloke/Baichuan2-7B-Chat-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "baichuan2-7b-chat.Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   8.0, "baichuan2-7b-chat.Q8_0.gguf"),
               ],
               tags=["中文", "通用"]),

    # === 7-8B 级别（8-16GB 内存） ===

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-1.5B", 1.5,
               "DeepSeek R1 蒸馏版 1.5B，推理能力强",
               "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.1, "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", True),
               ],
               tags=["中文", "推理", "轻量"]),

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-7B", 7.0,
               "DeepSeek R1 蒸馏 7B，中文推理王者，强烈推荐",
               "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
               variants=[
                   ModelVariant("Q3_K_M", 4.0, "DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf"),
                   ModelVariant("Q4_K_M", 4.8, "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 5.8, "DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"),
                   ModelVariant("Q6_K",   6.5, "DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"),
                   ModelVariant("Q8_0",   8.0, "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"),
                   ModelVariant("F16",    14.0, "DeepSeek-R1-Distill-Qwen-7B-F16.gguf"),
               ],
               tags=["中文", "推理", "推荐"]),

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Llama-8B", 8.0,
               "DeepSeek R1 蒸馏 Llama 8B，英文推理更强",
               "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   9.0, "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf"),
               ],
               tags=["英文", "推理"]),

    ModelEntry("Qwen2.5", "Qwen2.5-7B-Instruct", 7.0,
               "阿里通义 7B，中文综合能力出色",
               "Qwen/Qwen2.5-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q3_K_M", 4.0, "qwen2.5-7b-instruct-q3_k_m.gguf"),
                   ModelVariant("Q4_K_M", 4.8, "qwen2.5-7b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q5_K_M", 5.8, "qwen2.5-7b-instruct-q5_k_m.gguf"),
                   ModelVariant("Q6_K",   6.5, "qwen2.5-7b-instruct-q6_k.gguf"),
                   ModelVariant("Q8_0",   8.0, "qwen2.5-7b-instruct-q8_0.gguf"),
                   ModelVariant("F16",    14.0, "qwen2.5-7b-instruct-f16.gguf"),
               ],
               tags=["中文", "通用", "推荐"]),

    ModelEntry("Mistral", "Mistral-7B-Instruct-v0.3", 7.0,
               "Mistral 7B v0.3，开源最强通用 7B",
               "Mistral/Mistral-7B-Instruct-v0.3-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 5.8, "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   8.0, "Mistral-7B-Instruct-v0.3-Q8_0.gguf"),
               ],
               tags=["英文", "通用", "高效"]),

    ModelEntry("Llama 3.1", "Llama-3.1-8B-Instruct", 8.0,
               "Meta Llama 3.1 8B，通用对话标杆",
               "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 6.5, "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   9.0, "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
                   ModelVariant("F16",    16.0, "Meta-Llama-3.1-8B-Instruct-F16.gguf"),
               ],
               tags=["英文", "通用", "推荐"]),

    ModelEntry("Gemma 2", "Gemma-2-9B", 9.0,
               "Google Gemma 2 9B，安全且高质量",
               "unsloth/gemma-2-9b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 6.0, "gemma-2-9b-it-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   10.0, "gemma-2-9b-it-Q8_0.gguf"),
               ],
               tags=["英文", "安全"]),

    ModelEntry("Qwen2.5", "Qwen2.5-Coder-7B", 7.0,
               "阿里通义代码 7B，专为代码生成优化",
               "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "qwen2.5-coder-7b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   8.0, "qwen2.5-coder-7b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "代码"]),

    # === 14-20B 级别（16-32GB 内存） ===

    ModelEntry("Qwen2.5", "Qwen2.5-14B-Instruct", 14.0,
               "阿里通义 14B，高配置下的中文顶级选择",
               "Qwen/Qwen2.5-14B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 9.5, "qwen2.5-14b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q5_K_M", 11.5, "qwen2.5-14b-instruct-q5_k_m.gguf"),
                   ModelVariant("Q6_K",   13.0, "qwen2.5-14b-instruct-q6_k.gguf"),
               ],
               tags=["中文", "旗舰"]),

    ModelEntry("Phi-3", "Phi-3-medium-14B", 14.0,
               "微软 Phi-3 14B，大尺寸强能力",
               "microsoft/Phi-3-medium-4k-instruct-gguf",
               variants=[
                   ModelVariant("Q4_K_M", 9.5, "Phi-3-medium-4k-instruct-q4.gguf", True),
                   ModelVariant("Q6_K",   13.0, "Phi-3-medium-4k-instruct-q6.gguf"),
               ],
               tags=["英文", "能力"]),

    ModelEntry("Mistral", "Mixtral-8x7B-Instruct", 46.0,
               "Mistral 8x7B MoE，等效 12B 活跃参数，需 24GB+",
               "Mistral/Mixtral-8x7B-Instruct-v0.1-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 24.0, "mixtral-8x7b-instruct-v0.1-Q4_K_M.gguf", True),
                   ModelVariant("Q3_K_M", 18.0, "mixtral-8x7b-instruct-v0.1-Q3_K_M.gguf"),
               ],
               tags=["英文", "MoE", "旗舰"]),

    # === 32B+ 级别（24GB+ 内存） ===

    ModelEntry("Qwen2.5", "Qwen2.5-32B-Instruct", 32.0,
               "阿里通义 32B，旗舰级中文模型，需 32GB+",
               "Qwen/Qwen2.5-32B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 22.0, "qwen2.5-32b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q3_K_M", 18.0, "qwen2.5-32b-instruct-q3_k_m.gguf"),
               ],
               tags=["中文", "旗舰", "顶级"]),

    ModelEntry("Gemma 2", "Gemma-2-27B", 27.0,
               "Google Gemma 2 27B，需 32GB+",
               "unsloth/gemma-2-27b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 20.0, "gemma-2-27b-it-Q4_K_M.gguf", True),
               ],
               tags=["英文", "旗舰"]),

    # === 补充模型 (4-9B) ===

    ModelEntry("Yi", "Yi-6B", 6.0,
               "零一万物 Yi-6B，中英双语均衡",
               "unsloth/Yi-6B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "Yi-6B-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 5.0, "Yi-6B-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   7.0, "Yi-6B-Q8_0.gguf"),
               ],
               tags=["中文", "通用"]),

    ModelEntry("ChatGLM", "ChatGLM3-6B", 6.0,
               "智谱 ChatGLM3 6B，中文对话经典",
               "unsloth/chatglm3-6b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "chatglm3-6b-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   7.0, "chatglm3-6b-Q8_0.gguf"),
               ],
               tags=["中文", "经典"]),

    ModelEntry("GLM-4", "GLM-4-9B-Chat", 9.0,
               "智谱 GLM-4 9B，中文理解生成出色，128K 上下文",
               "unsloth/glm-4-9b-chat-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 6.0, "glm-4-9b-chat-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 7.0, "glm-4-9b-chat-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   10.0, "glm-4-9b-chat-Q8_0.gguf"),
               ],
               tags=["中文", "推荐"]),

    ModelEntry("OpenChat", "OpenChat-3.6-8B", 8.0,
               "OpenChat 3.6 8B，对话流畅度顶级",
               "unsloth/openchat-3.6-8b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "openchat-3.6-8b-Q4_K_M.gguf", True),
               ],
               tags=["英文", "对话"]),

    ModelEntry("OpenCode", "OpenCode-8B", 8.0,
               "OpenCode 8B，开源本地 Copilot 替代",
               "unsloth/OpenCode-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "OpenCode-8B-Q4_K_M.gguf", True),
               ],
               tags=["英文", "代码"]),

    ModelEntry("CodeQwen", "CodeQwen2.5-7B-Instruct", 7.0,
               "阿里 CodeQwen 2.5 7B，专业代码模型",
               "Qwen/CodeQwen2.5-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "codeqwen2.5-7b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   8.0, "codeqwen2.5-7b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "代码"]),

    ModelEntry("Gemma 2", "Gemma-2-2B-JPN", 2.0,
               "Gemma 2 2B 日语优化版",
               "unsloth/gemma-2-2b-jpn-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.5, "gemma-2-2b-jpn-it-Q4_K_M.gguf", True),
               ],
               tags=["日语", "轻量", "安全"]),

    # === 补充模型 (12-15B) ===

    ModelEntry("Mistral", "Mistral-Nemo-12B", 12.0,
               "Mistral Nemo 12B，NVIDIA 合作，128K 上下文，全能型",
               "unsloth/Mistral-Nemo-Instruct-2407-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 8.2, "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   13.0, "Mistral-Nemo-Instruct-2407-Q8_0.gguf"),
               ],
               tags=["英文", "推荐", "全能"]),

    ModelEntry("Qwen2.5", "Qwen2.5-Coder-14B", 14.0,
               "阿里代码 14B，最强开源本地代码模型",
               "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 9.5, "qwen2.5-coder-14b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q5_K_M", 11.5, "qwen2.5-coder-14b-instruct-q5_k_m.gguf"),
               ],
               tags=["中文", "代码"]),

    ModelEntry("StarCoder2", "StarCoder2-15B", 15.0,
               "StarCoder2 15B，600+ 编程语言，GitHub 训练",
               "bigcode/starcoder2-15b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 10.0, "starcoder2-15b-Q4_K_M.gguf", True),
               ],
               tags=["英文", "代码"]),

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-14B", 14.0,
               "DeepSeek R1 14B，深度推理，数学和逻辑能力接近 32B 级",
               "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 9.5, "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 11.5, "DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf"),
               ],
               tags=["中文", "推理", "推荐"]),

    # === 补充模型 (72B) ===

    ModelEntry("Qwen2.5", "Qwen2.5-72B-Instruct", 72.0,
               "通义千问 2.5 终极版 72B，直逼 GPT-4o，需 48GB+",
               "Qwen/Qwen2.5-72B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q2_K",   30.0, "qwen2.5-72b-instruct-q2_k.gguf"),
                   ModelVariant("Q3_K_M", 38.0, "qwen2.5-72b-instruct-q3_k_m.gguf"),
                   ModelVariant("Q4_K_M", 48.0, "qwen2.5-72b-instruct-q4_k_m.gguf", True),
               ],
               tags=["中文", "旗舰", "顶级"]),

    # === 新增热门模型 2025 ===

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-32B", 32.0,
               "DeepSeek R1 蒸馏 32B，推理能力接近 GPT-4，需 24GB+",
               "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
               variants=[
                   ModelVariant("Q3_K_M", 18.0, "DeepSeek-R1-Distill-Qwen-32B-Q3_K_M.gguf"),
                   ModelVariant("Q4_K_M", 22.0, "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 26.0, "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M.gguf"),
               ],
               tags=["中文", "推理", "旗舰", "推荐"]),

    ModelEntry("Llama 3.2", "Llama-3.2-11B-Vision", 11.0,
               "Meta Llama 3.2 11B 视觉模型，图文理解",
               "unsloth/Llama-3.2-11B-Vision-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 7.5, "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   12.5, "Llama-3.2-11B-Vision-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "视觉", "多模态"]),

    ModelEntry("Qwen2.5", "Qwen2.5-Coder-1.5B", 1.5,
               "阿里通义代码 1.5B，低配设备代码补全神器",
               "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.1, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   1.7, "qwen2.5-coder-1.5b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "代码", "轻量"]),

    ModelEntry("DeepSeek", "DeepSeek-Coder-V2-Lite-16B", 16.0,
               "DeepSeek Coder V2 Lite 16B，开源代码模型顶级，需 16GB+",
               "unsloth/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 11.0, "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q3_K_M", 9.0, "DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf"),
               ],
               tags=["英文", "代码", "旗舰"]),

    ModelEntry("Phi-4", "Phi-4-14B", 14.0,
               "微软 Phi-4 14B，合成数据训练，推理能力出色",
               "unsloth/phi-4-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 9.5, "phi-4-Q4_K_M.gguf", True),
                   ModelVariant("Q6_K",   13.0, "phi-4-Q6_K.gguf"),
               ],
               tags=["英文", "推理", "推荐"]),

    ModelEntry("InternLM3", "InternLM3-8B-Instruct", 8.0,
               "上海AI实验室书生浦语3 8B，中文能力出色",
               "unsloth/internlm3-8b-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "internlm3-8b-instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 6.5, "internlm3-8b-instruct-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   9.0, "internlm3-8b-instruct-Q8_0.gguf"),
               ],
               tags=["中文", "通用", "推荐"]),

    ModelEntry("Mistral", "Mistral-Small-24B", 24.0,
               "Mistral Small 24B，企业级文本生成，需 20GB+",
               "unsloth/Mistral-Small-24B-Instruct-2501-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 16.0, "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", True),
                   ModelVariant("Q3_K_M", 13.0, "Mistral-Small-24B-Instruct-2501-Q3_K_M.gguf"),
               ],
               tags=["英文", "旗舰", "全能"]),

    ModelEntry("Qwen2.5", "Qwen2.5-Math-7B", 7.0,
               "阿里通义数学 7B，数学推理专项优化",
               "Qwen/Qwen2.5-Math-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "qwen2.5-math-7b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   8.0, "qwen2.5-math-7b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "数学", "推理"]),

    ModelEntry("CodeLlama", "CodeLlama-13B-Instruct", 13.0,
               "Meta CodeLlama 13B，经典开源代码模型",
               "TheBloke/CodeLlama-13B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 8.8, "codellama-13b-instruct.Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 10.5, "codellama-13b-instruct.Q5_K_M.gguf"),
               ],
               tags=["英文", "代码"]),

    ModelEntry("Qwen2.5", "Qwen2.5-0.5B-Coder", 0.5,
               "阿里通义代码 0.5B，嵌入式/树莓派代码助手",
               "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.4, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", True),
               ],
               tags=["中文", "代码", "嵌入式"]),

    ModelEntry("Gemma 3", "Gemma-3-4B", 4.0,
               "Google Gemma 3 4B，新一代轻量多语言模型",
               "unsloth/gemma-3-4b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "gemma-3-4b-it-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   4.6, "gemma-3-4b-it-Q8_0.gguf"),
               ],
               tags=["英文", "多语言", "轻量"]),

    ModelEntry("DeepSeek", "DeepSeek-V3-0324-Lite", 16.0,
               "DeepSeek V3 0324 轻量版，MoE高效推理",
               "unsloth/DeepSeek-V3-0324-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 11.0, "DeepSeek-V3-0324-Q4_K_M.gguf", True),
               ],
               tags=["中文", "MoE", "旗舰", "推荐"]),

    # === 2025 Q2 新增模型 ===

    ModelEntry("SmolLM2", "SmolLM2-1.7B-Instruct", 1.7,
               "HuggingFace SmolLM2 1.7B，超轻量高性能",
               "bartowski/SmolLM2-1.7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.2, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   1.9, "SmolLM2-1.7B-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "轻量", "边缘"]),

    ModelEntry("SmolLM2", "SmolLM2-360M-Instruct", 0.36,
               "HuggingFace SmolLM2 360M，微型高性能演示模型",
               "bartowski/SmolLM2-360M-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.3, "SmolLM2-360M-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   0.5, "SmolLM2-360M-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "微型", "演示"]),

    ModelEntry("Phi-3.5", "Phi-3.5-MoE-6.6B", 6.6,
               "微软 Phi-3.5 MoE 6.6B，混合专家高效推理",
               "microsoft/Phi-3.5-MoE-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.5, "Phi-3.5-MoE-instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 5.4, "Phi-3.5-MoE-instruct-Q5_K_M.gguf"),
               ],
               tags=["英文", "MoE", "高效"]),

    ModelEntry("Phi-4", "Phi-4-mini-3.8B", 3.8,
               "微软 Phi-4 mini 3.8B，合成数据训练的轻量推理模型",
               "unsloth/phi-4-mini-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "phi-4-mini-instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   4.5, "phi-4-mini-instruct-Q8_0.gguf"),
               ],
               tags=["英文", "推理", "轻量"]),

    ModelEntry("Command R", "Command-R-7B-2025", 7.0,
               "Cohere Command R 7B，多语言 RAG 优化",
               "CohereForAI/c4ai-command-r7b-12-2024-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "c4ai-command-r7b-12-2024-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   8.0, "c4ai-command-r7b-12-2024-Q8_0.gguf"),
               ],
               tags=["英文", "多语言", "RAG"]),

    ModelEntry("Falcon3", "Falcon3-7B-Instruct", 7.0,
               "TII Falcon3 7B，阿联酋开源，性能超越 Llama3.1 同级",
               "tiiuae/Falcon3-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "Falcon3-7B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   8.0, "Falcon3-7B-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "通用", "推荐"]),

    ModelEntry("Falcon3", "Falcon3-10B-Instruct", 10.0,
               "TII Falcon3 10B，更大的Falcon3，性能更强",
               "tiiuae/Falcon3-10B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 7.0, "Falcon3-10B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   11.5, "Falcon3-10B-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "通用"]),

    ModelEntry("Granite Code", "Granite-3.1-8B-Instruct", 8.0,
               "IBM Granite 3.1 8B，企业级代码和通用模型，Apache 2.0",
               "ibm-granite/granite-3.1-8b-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "granite-3.1-8b-instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   9.0, "granite-3.1-8b-instruct-Q8_0.gguf"),
               ],
               tags=["英文", "代码", "企业"]),

    ModelEntry("Granite Code", "Granite-3.1-2B-Instruct", 2.0,
               "IBM Granite 3.1 2B，小型企业级模型，Apache 2.0",
               "ibm-granite/granite-3.1-2b-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.5, "granite-3.1-2b-instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   2.5, "granite-3.1-2b-instruct-Q8_0.gguf"),
               ],
               tags=["英文", "代码", "轻量"]),

    ModelEntry("Solar Pro", "Solar-10.7B", 10.7,
               "Upstage Solar 10.7B，韩国AI公司旗舰",
               "upstage/SOLAR-10.7B-Instruct-v1.0-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 6.5, "SOLAR-10.7B-Instruct-v1.0-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 8.0, "SOLAR-10.7B-Instruct-v1.0-Q5_K_M.gguf"),
               ],
               tags=["英文", "韩文", "多语言"]),

    ModelEntry("EXAONE", "EXAONE-4.0-7.8B", 7.8,
               "LG AI EXAONE 4.0 7.8B，韩国LG自研，韩英双语出色",
               "LGAI-EXAONE/EXAONE-4.0-7.8B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.3, "EXAONE-4.0-7.8B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   9.0, "EXAONE-4.0-7.8B-Instruct-Q8_0.gguf"),
               ],
               tags=["韩文", "英文", "双语"]),

    ModelEntry("Aya Expanse", "Aya-Expanse-8B", 8.0,
               "Cohere Aya Expanse 8B，支持23种语言的多语言模型",
               "CohereForAI/aya-expanse-8b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "aya-expanse-8b-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   9.0, "aya-expanse-8b-Q8_0.gguf"),
               ],
               tags=["多语言", "23语言"]),

    ModelEntry("Hermes 3", "Hermes-3-Llama-3.1-8B", 8.0,
               "Nous Research Hermes 3，指令遵循能力出色",
               "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 6.5, "Hermes-3-Llama-3.1-8B-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   9.0, "Hermes-3-Llama-3.1-8B-Q8_0.gguf"),
               ],
               tags=["英文", "指令遵循", "推荐"]),

    ModelEntry("Hermes 3", "Hermes-3-Llama-3.2-3B", 3.0,
               "Nous Research Hermes 3 3B，小尺寸指令遵循",
               "NousResearch/Hermes-3-Llama-3.2-3B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.2, "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   3.5, "Hermes-3-Llama-3.2-3B-Q8_0.gguf"),
               ],
               tags=["英文", "指令遵循", "轻量"]),

    ModelEntry("Nemotron", "Nemotron-Mini-4B", 4.0,
               "NVIDIA Nemotron Mini 4B，NVIDIA 自研小模型",
               "nvidia/Nemotron-Mini-4B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   4.6, "Nemotron-Mini-4B-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "高效", "NVIDIA"]),

    ModelEntry("Olmo 2", "OLMo-2-7B-Instruct", 7.0,
               "AI2 OLMo 2 7B，完全开源（含训练数据）",
               "allenai/OLMo-2-7B-1124-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "OLMo-2-7B-1124-Instruct-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   8.0, "OLMo-2-7B-1124-Instruct-Q8_0.gguf"),
               ],
               tags=["英文", "研究", "完全开源"]),

    ModelEntry("Gemma 3", "Gemma-3-12B", 12.0,
               "Google Gemma 3 12B，新一代多模态多语言模型",
               "unsloth/gemma-3-12b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 8.2, "gemma-3-12b-it-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 10.0, "gemma-3-12b-it-Q5_K_M.gguf"),
                   ModelVariant("Q8_0",   13.5, "gemma-3-12b-it-Q8_0.gguf"),
               ],
               tags=["英文", "多模态", "安全"]),

    ModelEntry("Gemma 3", "Gemma-3-1B", 1.0,
               "Google Gemma 3 1B，最轻量Gemma 3，手机/边缘可用",
               "unsloth/gemma-3-1b-it-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.8, "gemma-3-1b-it-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   1.3, "gemma-3-1b-it-Q8_0.gguf"),
               ],
               tags=["英文", "边缘", "轻量"]),

    ModelEntry("Qwen2.5-VL", "Qwen2.5-VL-7B", 7.0,
               "阿里通义视觉语言模型 7B，图文多模态理解",
               "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "qwen2.5-vl-7b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   8.0, "qwen2.5-vl-7b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "视觉", "多模态"]),

    ModelEntry("Qwen2.5-VL", "Qwen2.5-VL-3B", 3.0,
               "阿里通义视觉语言模型 3B，轻量图文理解",
               "Qwen/Qwen2.5-VL-3B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.2, "qwen2.5-vl-3b-instruct-q4_k_m.gguf", True),
                   ModelVariant("Q8_0",   3.5, "qwen2.5-vl-3b-instruct-q8_0.gguf"),
               ],
               tags=["中文", "视觉", "轻量"]),

    ModelEntry("LLaMA-Mesh", "LLaMA-Mesh-8B", 8.0,
               "NVIDIA LLaMA-Mesh 8B，3D建模与代码生成",
               "NVIDIA/Llama-3.1-LLaMA-Mesh-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "Llama-3.1-LLaMA-Mesh-8B-Q4_K_M.gguf", True),
               ],
               tags=["英文", "3D", "代码"]),

    ModelEntry("Dolphin3", "Dolphin3.0-Llama-3.1-8B", 8.0,
               "Dolphin 3.0 8B，未审查版本 Llama 3.1",
               "cognitivecomputations/Dolphin3.0-Llama-3.1-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "Dolphin3.0-Llama-3.1-8B-Q4_K_M.gguf", True),
                   ModelVariant("Q8_0",   9.0, "Dolphin3.0-Llama-3.1-8B-Q8_0.gguf"),
               ],
               tags=["英文", "创意", "未审查"]),

    ModelEntry("TokyoTech", "LLM-jp-3-13B", 13.0,
               "日本东京工业大学 LLM-jp 3 13B，日语最优化大模型",
               "llm-jp/llm-jp-3-13b-instruct3-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 8.8, "llm-jp-3-13b-instruct3-Q4_K_M.gguf", True),
                   ModelVariant("Q5_K_M", 10.5, "llm-jp-3-13b-instruct3-Q5_K_M.gguf"),
               ],
               tags=["日语", "通用", "推荐"]),

    # === Q3 扩充热门模型 ===
    ModelEntry("Mistral", "Mistral-7B-Instruct-v0.3-Q4_K_M", 7.0,
               "Mistral 7B Instruct v0.3 Q4_K_M 量化",
               "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", True),
               ],
               tags=["英文", "通用", "高效"]),

    ModelEntry("Mixtral", "Mixtral-8x7B-Instruct-v0.1-Q4_K_M", 46.7,
               "Mixtral 8x7B MoE Q4_K_M 量化",
               "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 26.0, "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", True),
               ],
               tags=["英文", "MoE", "旗舰"]),

    ModelEntry("CodeLlama", "CodeLlama-7B-Instruct-Q4_K_M", 7.0,
               "Meta CodeLlama 7B 代码指令版 Q4_K_M 量化",
               "TheBloke/CodeLlama-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "codellama-7b-instruct.Q4_K_M.gguf", True),
               ],
               tags=["英文", "代码"]),

    ModelEntry("CodeLlama", "CodeLlama-13B-Instruct-Q4_K_M", 13.0,
               "Meta CodeLlama 13B 代码指令版 Q4_K_M 量化",
               "TheBloke/CodeLlama-13B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 7.8, "codellama-13b-instruct.Q4_K_M.gguf", True),
               ],
               tags=["英文", "代码"]),

    ModelEntry("DeepSeekMath", "DeepSeekMath-7B-Instruct-Q4_K_M", 7.0,
               "DeepSeek Math 7B 数学推理优化版 Q4_K_M 量化",
               "mradermacher/DeepSeek-Math-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.3, "DeepSeek-Math-7B-Instruct.Q4_K_M.gguf", True),
               ],
               tags=["英文", "数学", "推理"]),

    ModelEntry("Zephyr", "Zephyr-7B-Beta-Q4_K_M", 7.0,
               "HuggingFace Zephyr 7B Beta Q4_K_M 量化",
               "TheBloke/zephyr-7B-beta-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "zephyr-7b-beta.Q4_K_M.gguf", True),
               ],
               tags=["英文", "对话"]),

    ModelEntry("OpenChat", "OpenChat-3.5-7B-Q4_K_M", 7.0,
               "OpenChat 3.5 7B Q4_K_M 量化",
               "TheBloke/openchat-3.5-7B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.2, "openchat-3.5-7b.Q4_K_M.gguf", True),
               ],
               tags=["英文", "对话"]),

    ModelEntry("Nous Hermes", "Hermes-2-Theta-Llama-3-8B-Q4_K_M", 8.0,
               "Nous Hermes 2 Theta Llama-3 8B Q4_K_M 量化",
               "NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.9, "Hermes-2-Theta-Llama-3-8B-Q4_K_M.gguf", True),
               ],
               tags=["英文", "Agent", "指令遵循"]),

    ModelEntry("Nous Hermes", "Hermes-2-Pro-Llama-3-8B-Q4_K_M", 8.0,
               "Nous Hermes 2 Pro Llama-3 8B Q4_K_M 量化",
               "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.9, "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf", True),
               ],
               tags=["英文", "函数调用", "Agent"]),

    ModelEntry("Dolphin", "Dolphin-2.9-Llama-3-8B-Q4_K_M", 8.0,
               "Dolphin 2.9 Llama-3 8B Q4_K_M 量化",
               "cognitivecomputations/dolphin-2.9-llama3-8b-gguf",
               variants=[
                   ModelVariant("Q4_K_M", 4.9, "dolphin-2.9-llama3-8b-Q4_K_M.gguf", True),
               ],
               tags=["英文", "创意", "未审查"]),

    ModelEntry("StableLM", "StableLM-Zephyr-3B-Q4_K_M", 3.0,
               "Stability AI StableLM Zephyr 3B Q4_K_M 量化",
               "TheBloke/stablelm-zephyr-3b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.9, "stablelm-zephyr-3b.Q4_K_M.gguf", True),
               ],
               tags=["英文", "对话", "轻量"]),

    ModelEntry("Yi-1.5", "Yi-1.5-9B-Chat-Q4_K_M", 9.0,
               "零一万物 Yi-1.5 9B 对话版 Q4_K_M 量化",
               "lmstudio-community/Yi-1.5-9B-Chat-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "Yi-1.5-9B-Chat-Q4_K_M.gguf", True),
               ],
               tags=["中文", "英文", "通用"]),

    ModelEntry("Yi-1.5", "Yi-1.5-34B-Chat-Q4_K_M", 34.0,
               "零一万物 Yi-1.5 34B 对话版 Q4_K_M 量化",
               "lmstudio-community/Yi-1.5-34B-Chat-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 20.5, "Yi-1.5-34B-Chat-Q4_K_M.gguf", True),
               ],
               tags=["中文", "英文", "旗舰"]),

    ModelEntry("LLaVA", "LLaVA-v1.6-Mistral-7B-Q4_K_M", 7.0,
               "LLaVA 1.6 多模态视觉模型 Q4_K_M 量化",
               "MaziyarPanahi/llava-v1.6-mistral-7b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.5, "llava-v1.6-mistral-7b.Q4_K_M.gguf", True),
               ],
               tags=["英文", "视觉", "多模态"]),

    ModelEntry("LLaVA", "LLaVA-v1.6-34B-Q4_K_M", 34.0,
               "LLaVA 1.6 多模态视觉模型 34B Q4_K_M 量化",
               "MaziyarPanahi/llava-v1.6-34b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 20.0, "llava-v1.6-34b.Q4_K_M.gguf", True),
               ],
               tags=["英文", "视觉", "多模态"]),

    ModelEntry("Granite Code", "Granite-3.1-8B-Instruct-Q4_K_M", 8.0,
               "IBM Granite 3.1 8B Q4_K_M 量化",
               "ibm-granite/granite-3.1-8b-instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "granite-3.1-8b-instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "代码", "企业"]),

    ModelEntry("Aya Expanse", "Aya-Expanse-8B-Q4_K_M", 8.0,
               "Cohere Aya Expanse 8B Q4_K_M 量化",
               "CohereForAI/aya-expanse-8b-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.5, "aya-expanse-8b-Q4_K_M.gguf", True),
               ],
               tags=["多语言", "23语言"]),

    ModelEntry("EXAONE", "EXAONE-4.0-7.8B-Q4_K_M", 7.8,
               "LG AI EXAONE 4.0 7.8B Q4_K_M 量化",
               "LGAI-EXAONE/EXAONE-4.0-7.8B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 5.3, "EXAONE-4.0-7.8B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["韩文", "英文", "双语"]),

    ModelEntry("Llama 3.2", "Meta-Llama-3-8B-Instruct-Q4_K_M", 8.0,
               "Meta Llama 3 8B Q4_K_M 量化",
               "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.9, "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "通用", "推荐"]),

    ModelEntry("Llama 3.2", "Meta-Llama-3-70B-Instruct-Q4_K_M", 70.0,
               "Meta Llama 3 70B Q4_K_M 量化",
               "lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 40.0, "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "旗舰", "顶级"]),

    ModelEntry("Falcon3", "Falcon3-7B-Instruct-Q4_K_M", 7.0,
               "TII Falcon3 7B Q4_K_M 量化",
               "tiiuae/Falcon3-7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "Falcon3-7B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "高效", "推荐"]),

    ModelEntry("Falcon3", "Falcon3-10B-Instruct-Q4_K_M", 10.0,
               "TII Falcon3 10B Q4_K_M 量化",
               "tiiuae/Falcon3-10B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 7.0, "Falcon3-10B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "高效"]),

    ModelEntry("Command R", "Command-R-7B-2024-Q4_K_M", 7.0,
               "Cohere Command R 7B Q4_K_M 量化",
               "CohereForAI/c4ai-command-r7b-12-2024-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "c4ai-command-r7b-12-2024-Q4_K_M.gguf", True),
               ],
               tags=["英文", "多语言", "RAG"]),

    ModelEntry("Solar Pro", "Solar-Pro-22B-Q4_K_M", 22.0,
               "Upstage Solar Pro 22B Q4_K_M 量化",
               "upstage/SOLAR-10.7B-Instruct-v1.0-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 14.5, "SOLAR-10.7B-Instruct-v1.0-Q4_K_M.gguf", True),
               ],
               tags=["英文", "韩文", "多语言"]),

    ModelEntry("Olmo 2", "OLMo-2-7B-Instruct-Q4_K_M", 7.0,
               "AI2 OLMo 2 7B Q4_K_M 量化",
               "allenai/OLMo-2-7B-1124-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 4.8, "OLMo-2-7B-1124-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "研究", "完全开源"]),

    ModelEntry("Nemotron", "Nemotron-Mini-4B-Instruct-Q4_K_M", 4.0,
               "NVIDIA Nemotron Mini 4B Q4_K_M 量化",
               "nvidia/Nemotron-Mini-4B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 2.8, "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "高效", "NVIDIA"]),

    ModelEntry("SmolLM2", "SmolLM2-1.7B-Instruct-Q4_K_M", 1.7,
               "HuggingFace SmolLM2 1.7B Q4_K_M 量化",
               "bartowski/SmolLM2-1.7B-Instruct-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 1.2, "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", True),
               ],
               tags=["英文", "轻量", "边缘"]),

    ModelEntry("TinyLlama", "TinyLlama-1.1B-Chat-Q4_K_M", 1.1,
               "TinyLlama 1.1B Q4_K_M 量化",
               "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
               variants=[
                   ModelVariant("Q4_K_M", 0.7, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", True),
               ],
               tags=["英文", "轻量", "嵌入式"]),

]

# ======================== HuggingFace 原版模型（适合微调训练） ========================

MODEL_REGISTRY_HF: List[ModelEntry] = [
    ModelEntry("Qwen2.5", "Qwen2.5-0.5B-Instruct (HF)", 0.5,
               "阿里通义0.5B，HuggingFace原版，支持微调训练",
               "Qwen/Qwen2.5-0.5B-Instruct",
               variants=[ModelVariant("BF16", 1.5, "full", True)],
               tags=["中文", "轻量", "可微调"], model_type="huggingface"),

    ModelEntry("Qwen2.5", "Qwen2.5-1.5B-Instruct (HF)", 1.5,
               "阿里通义1.5B，HuggingFace原版，支持微调训练",
               "Qwen/Qwen2.5-1.5B-Instruct",
               variants=[ModelVariant("BF16", 3.5, "full", True)],
               tags=["中文", "轻量", "可微调"], model_type="huggingface"),

    ModelEntry("Qwen2.5", "Qwen2.5-3B-Instruct (HF)", 3.0,
               "阿里通义3B，HuggingFace原版，支持微调训练",
               "Qwen/Qwen2.5-3B-Instruct",
               variants=[ModelVariant("BF16", 7.0, "full", True)],
               tags=["中文", "通用", "可微调"], model_type="huggingface"),

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-1.5B (HF)", 1.5,
               "DeepSeek R1蒸馏版1.5B，HuggingFace原版",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
               variants=[ModelVariant("BF16", 3.5, "full", True)],
               tags=["推理", "轻量", "可微调"], model_type="huggingface"),

    ModelEntry("DeepSeek", "DeepSeek-R1-Distill-Qwen-7B (HF)", 7.0,
               "DeepSeek R1蒸馏版7B，HuggingFace原版",
               "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
               variants=[ModelVariant("BF16", 16.0, "full", True)],
               tags=["推理", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("Llama 3.2", "Llama-3.2-1B-Instruct (HF)", 1.0,
               "Meta Llama 3.2 1B，HuggingFace原版",
               "meta-llama/Llama-3.2-1B-Instruct",
               variants=[ModelVariant("BF16", 2.5, "full", True)],
               tags=["英文", "通用", "可微调"], model_type="huggingface"),

    ModelEntry("Llama 3.2", "Llama-3.2-3B-Instruct (HF)", 3.0,
               "Meta Llama 3.2 3B，HuggingFace原版",
               "meta-llama/Llama-3.2-3B-Instruct",
               variants=[ModelVariant("BF16", 7.0, "full", True)],
               tags=["英文", "通用", "可微调"], model_type="huggingface"),

    ModelEntry("Gemma 2", "Gemma-2-2B (HF)", 2.0,
               "Google Gemma 2 2B，HuggingFace原版",
               "google/gemma-2-2b-it",
               variants=[ModelVariant("BF16", 5.0, "full", True)],
               tags=["英文", "安全", "可微调"], model_type="huggingface"),

    ModelEntry("Phi-3", "Phi-3-mini-3.8B (HF)", 3.8,
               "微软Phi-3 3.8B，HuggingFace原版",
               "microsoft/Phi-3-mini-4k-instruct",
               variants=[ModelVariant("BF16", 9.0, "full", True)],
               tags=["英文", "轻量", "高效", "可微调"], model_type="huggingface"),

    ModelEntry("Qwen2.5", "Qwen2.5-7B-Instruct (HF)", 7.0,
               "阿里通义7B，HuggingFace原版，中文能力强",
               "Qwen/Qwen2.5-7B-Instruct",
               variants=[ModelVariant("BF16", 16.0, "full", True)],
               tags=["中文", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("Qwen2.5", "Qwen2.5-14B-Instruct (HF)", 14.0,
               "阿里通义14B，高质量中文模型",
               "Qwen/Qwen2.5-14B-Instruct",
               variants=[ModelVariant("BF16", 30.0, "full", True)],
               tags=["中文", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("Yi-1.5", "Yi-1.5-6B-Chat (HF)", 6.0,
               "零一万物 6B，中英双语，HuggingFace原版",
               "01-ai/Yi-1.5-6B-Chat",
               variants=[ModelVariant("BF16", 14.0, "full", True)],
               tags=["中文", "英文", "可微调"], model_type="huggingface"),

    ModelEntry("Yi-1.5", "Yi-1.5-9B-Chat (HF)", 9.0,
               "零一万物 9B，中英双语旗舰",
               "01-ai/Yi-1.5-9B-Chat",
               variants=[ModelVariant("BF16", 20.0, "full", True)],
               tags=["中文", "英文", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("InternLM2", "InternLM2-Chat-7B (HF)", 7.0,
               "上海AI实验室书生浦语7B，HuggingFace原版",
               "internlm/internlm2-chat-7b",
               variants=[ModelVariant("BF16", 16.0, "full", True)],
               tags=["中文", "学术", "可微调"], model_type="huggingface"),

    ModelEntry("Mistral", "Mistral-7B-Instruct-v0.3 (HF)", 7.0,
               "Mistral AI 7B v0.3，HuggingFace原版",
               "mistralai/Mistral-7B-Instruct-v0.3",
               variants=[ModelVariant("BF16", 16.0, "full", True)],
               tags=["英文", "通用", "可微调"], model_type="huggingface"),

    ModelEntry("Llama 3", "Llama-3.1-8B-Instruct (HF)", 8.0,
               "Meta Llama 3.1 8B，HuggingFace原版",
               "meta-llama/Llama-3.1-8B-Instruct",
               variants=[ModelVariant("BF16", 18.0, "full", True)],
               tags=["英文", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("Phi-3", "Phi-3-medium-14B (HF)", 14.0,
               "微软Phi-3 Medium 14B，高效中型模型",
               "microsoft/Phi-3-medium-4k-instruct",
               variants=[ModelVariant("BF16", 30.0, "full", True)],
               tags=["英文", "高效", "可微调"], model_type="huggingface"),

    ModelEntry("Gemma 2", "Gemma-2-9B (HF)", 9.0,
               "Google Gemma 2 9B，强大英文模型",
               "google/gemma-2-9b-it",
               variants=[ModelVariant("BF16", 20.0, "full", True)],
               tags=["英文", "旗舰", "可微调"], model_type="huggingface"),

    ModelEntry("ChatGLM", "ChatGLM3-6B (HF)", 6.0,
               "智谱AI ChatGLM3 6B，中文对话优秀",
               "THUDM/chatglm3-6b",
               variants=[ModelVariant("BF16", 14.0, "full", True)],
               tags=["中文", "对话", "可微调"], model_type="huggingface"),
]
