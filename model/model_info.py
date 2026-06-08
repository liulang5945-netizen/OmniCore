"""
模型详细信息数据库
包含每个模型的中文全名、详细介绍、能力评分、适用场景等
与 model_registry.py 配合使用，保持注册表轻量
"""

# ======================== 模型详细描述 ========================

MODEL_DETAILS = {
    "Qwen2.5-0.5B-Instruct": {
        "full_name_cn": "通义千问 2.5 0.5B 指令版",
        "long_description": (
            "阿里云通义千问系列最小的模型，参数仅 0.5B，"
            "可以在 2GB 内存的设备上流畅运行。"
            "虽然体积小，但中文对话能力出色，适合嵌入式设备、边缘计算等资源受限场景。"
            "支持 32K 上下文窗口，可处理较长的对话。"
        ),
        "strengths": ["超低资源占用", "中文流畅", "启动极快"],
        "scores": {
            "中文对话": 5, "英文对话": 3, "代码生成": 2, "数学推理": 2,
            "逻辑推理": 2, "知识问答": 3, "创意写作": 3,
            "翻译": 4, "指令遵循": 5, "长文本处理": 4,
        },
        "recommended_for": ["嵌入式设备", "学习测试", "低配电脑入门", "边缘计算"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    },
    "Qwen2.5-1.5B-Instruct": {
        "full_name_cn": "通义千问 2.5 1.5B 指令版",
        "long_description": (
            "通义千问 1.5B 版本，在小巧的体积中提供了不错的中文对话能力。"
            "适合 4GB 内存设备，日常文字处理、简单翻译、基础问答均可胜任。"
            "32K 上下文，响应速度快，是中低配电脑的理想选择。"
        ),
        "strengths": ["中文优质", "体积小巧", "响应快速", "32K上下文"],
        "scores": {
            "中文对话": 7, "英文对话": 4, "代码生成": 3, "数学推理": 3,
            "逻辑推理": 3, "知识问答": 5, "创意写作": 4,
            "翻译": 5, "指令遵循": 6, "长文本处理": 5,
        },
        "recommended_for": ["日常办公", "学习助手", "文字翻译", "中低配电脑"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
    },
    "Llama-3.2-1B-Instruct": {
        "full_name_cn": "Llama 3.2 1B 指令版",
        "long_description": (
            "Meta 推出的轻量级模型，1B 参数却有令人惊喜的英文能力。"
            "推理速度极快，适合需要低延迟的实时交互场景。"
            "支持 128K 上下文 (需足够内存)，是英文问答、摘要的首选小模型。"
        ),
        "strengths": ["英文优质", "超快速度", "128K上下文", "Meta官方维护"],
        "scores": {
            "中文对话": 2, "英文对话": 7, "代码生成": 3, "数学推理": 3,
            "逻辑推理": 3, "知识问答": 5, "创意写作": 4,
            "翻译": 4, "指令遵循": 7, "长文本处理": 6,
        },
        "recommended_for": ["英文对话", "实时交互", "文本摘要", "学习LLM"],
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
    },
    "Llama-3.2-3B-Instruct": {
        "full_name_cn": "Llama 3.2 3B 指令版",
        "long_description": (
            "Meta Llama 3.2 3B 版本，号称 3B 级最强模型。"
            "英文理解和生成能力强，在 3B 以下尺寸的各类评测中表现优异。"
            "适合对英文能力要求较高的轻量级应用。"
        ),
        "strengths": ["英文最佳", "性能标杆", "推理迅速", "广泛社区支持"],
        "scores": {
            "中文对话": 2, "英文对话": 8, "代码生成": 5, "数学推理": 4,
            "逻辑推理": 4, "知识问答": 6, "创意写作": 5,
            "翻译": 5, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["英文写作助手", "代码补全", "学习研究", "轻量服务器"],
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    },
    "Phi-3-mini-3.8B": {
        "full_name_cn": "微软 Phi-3 Mini 3.8B",
        "long_description": (
            "微软 Phi-3 系列小模型，在 3.8B 参数下实现了令人惊讶的能力，"
            "甚至接近部分 7B 模型水平。采用高质量数据训练，英文推理和代码能力突出。"
            "适合资源有限但追求性能的场景。"
        ),
        "strengths": ["小身材大能力", "代码能力", "英文推理强", "微软背书"],
        "scores": {
            "中文对话": 3, "英文对话": 8, "代码生成": 7, "数学推理": 6,
            "逻辑推理": 7, "知识问答": 6, "创意写作": 5,
            "翻译": 4, "指令遵循": 8, "长文本处理": 5,
        },
        "recommended_for": ["代码辅助", "英文写作", "教育研究", "低配笔记本"],
        "hf_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
    },
    "Qwen2.5-3B-Instruct": {
        "full_name_cn": "通义千问 2.5 3B 指令版",
        "long_description": (
            "通义千问 3B 版本，是 3B 级别中文能力最强的模型之一。"
            "相比 1.5B 版本能力有质的飞跃，中文问答流畅自然，"
            "在知识问答、翻译、代码理解等方面表现优秀。32K 上下文。"
        ),
        "strengths": ["中文天花板", "能力均衡", "32K上下文", "阿里生态"],
        "scores": {
            "中文对话": 8, "英文对话": 5, "代码生成": 5, "数学推理": 4,
            "逻辑推理": 4, "知识问答": 6, "创意写作": 6,
            "翻译": 6, "指令遵循": 7, "长文本处理": 5,
        },
        "recommended_for": ["中文日常使用", "翻译工具", "内容创作", "办公助手"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    },
    "Gemma-2-2B": {
        "full_name_cn": "Google Gemma 2 2B",
        "long_description": (
            "Google 开源的 Gemma 2 2B 模型，在极小的体积下保持了优秀的安全对齐和英文能力。"
            "基于 Gemini 技术蒸馏，遵循安全准则，适合对内容安全要求高的场景。"
        ),
        "strengths": ["安全对齐", "英文优质", "极小体积", "Google出品"],
        "scores": {
            "中文对话": 2, "英文对话": 7, "代码生成": 4, "数学推理": 3,
            "逻辑推理": 4, "知识问答": 5, "创意写作": 4,
            "翻译": 3, "指令遵循": 8, "长文本处理": 4,
        },
        "recommended_for": ["内容安全场景", "英文教育", "移动端"],
        "hf_url": "https://huggingface.co/google/gemma-2-2b-it",
    },
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "full_name_cn": "DeepSeek R1 蒸馏 通义 1.5B",
        "long_description": (
            "DeepSeek R1 推理模型蒸馏到 1.5B 通义千问，保留了 R1 的链式推理能力。"
            "虽是极小模型，但数学推理和逻辑推理能力突出，超过同尺寸普通模型。"
            "适合低资源设备上进行数学计算和逻辑分析。"
        ),
        "strengths": ["数学推理强", "逻辑链式思考", "极小体积", "性价比高"],
        "scores": {
            "中文对话": 5, "英文对话": 4, "代码生成": 3, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 4, "创意写作": 3,
            "翻译": 4, "指令遵循": 5, "长文本处理": 3,
        },
        "recommended_for": ["数学辅导", "逻辑推理", "低配设备", "学生助手"],
        "hf_url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "full_name_cn": "DeepSeek R1 蒸馏 通义 7B",
        "long_description": (
            "⭐ 强烈推荐！DeepSeek R1 推理能力蒸馏到通义千问 7B，"
            "是目前中文推理能力最强的 7B 级模型。擅长逐步推理、数学计算、代码分析，"
            "在复杂逻辑问题上表现远超同尺寸通用模型。适用 8GB+ 内存设备，"
            "采用 Q4_K_M 量化仅需 4.8GB 显存。"
        ),
        "strengths": ["中文推理王者", "数学优秀", "代码能力", "逻辑链", "高性价比"],
        "scores": {
            "中文对话": 8, "英文对话": 7, "代码生成": 8, "数学推理": 9,
            "逻辑推理": 9, "知识问答": 7, "创意写作": 6,
            "翻译": 7, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["编程辅助", "数学学习", "复杂推理", "科研助手", "主流电脑首选"],
        "hf_url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B",
        "paper_url": "https://arxiv.org/abs/2501.12948",
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "full_name_cn": "DeepSeek R1 蒸馏 Llama 8B",
        "long_description": (
            "DeepSeek R1 蒸馏到 Llama 3.1 8B，继承了 R1 的强大推理能力，"
            "同时保留了 Llama 的优秀英文能力。英文推理和代码能力出色，"
            "适合需要中英双语推理的用户。"
        ),
        "strengths": ["英文推理", "代码生成", "双语能力", "逻辑分析"],
        "scores": {
            "中文对话": 6, "英文对话": 8, "代码生成": 8, "数学推理": 8,
            "逻辑推理": 9, "知识问答": 7, "创意写作": 7,
            "翻译": 7, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["英文编程", "学术研究", "双语用户", "数据分析"],
        "hf_url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B",
        "paper_url": "https://arxiv.org/abs/2501.12948",
    },
    "Qwen2.5-7B-Instruct": {
        "full_name_cn": "通义千问 2.5 7B 指令版",
        "long_description": (
            "阿里通义千问 2.5 的 7B 旗舰版本，中文综合能力在 7B 级别中名列前茅。"
            "在对话、翻译、知识问答、代码生成等各方面表现均衡出色。"
            "支持 32K/128K 上下文，适合多数日常和办公场景。"
        ),
        "strengths": ["中文全能", "能力均衡", "生态完善", "阿里维护"],
        "scores": {
            "中文对话": 9, "英文对话": 7, "代码生成": 7, "数学推理": 6,
            "逻辑推理": 6, "知识问答": 8, "创意写作": 8,
            "翻译": 9, "指令遵循": 8, "长文本处理": 7,
        },
        "recommended_for": ["中文办公", "内容创作", "翻译助手", "通用助手"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    },
    "Mistral-7B-Instruct-v0.3": {
        "full_name_cn": "Mistral 7B v0.3",
        "long_description": (
            "法国 Mistral AI 的经典 7B 模型 v0.3 版本，在英文通用能力和效率上表现出色。"
            "训练高效，推理速度快，被广泛用作开源基础模型。"
            "代码生成能力强，适合作为英文编程助手。"
        ),
        "strengths": ["英文优秀", "代码能力", "高效推理", "开源生态好"],
        "scores": {
            "中文对话": 4, "英文对话": 9, "代码生成": 8, "数学推理": 6,
            "逻辑推理": 7, "知识问答": 7, "创意写作": 8,
            "翻译": 6, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["英文编程", "英文写作", "研究对比", "服务器部署"],
        "hf_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    },
    "Llama-3.1-8B-Instruct": {
        "full_name_cn": "Llama 3.1 8B 指令版",
        "long_description": (
            "Meta Llama 3.1 8B 是目前最受欢迎的 8B 开源模型之一，"
            "英文综合能力和指令遵循表现均为同类标杆。"
            "支持 128K 上下文和多语言 (支持 8 种语言)，社区生态最完善。"
        ),
        "strengths": ["英文标杆", "指令遵循优秀", "128K上下文", "最大社区"],
        "scores": {
            "中文对话": 4, "英文对话": 9, "代码生成": 7, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 8, "创意写作": 8,
            "翻译": 6, "指令遵循": 9, "长文本处理": 8,
        },
        "recommended_for": ["英文全面助手", "长篇处理", "Agent开发", "主流电脑"],
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    },
    "Gemma-2-9B": {
        "full_name_cn": "Google Gemma 2 9B",
        "long_description": (
            "Google Gemma 2 9B 在安全对齐和英文能力上表现出众。"
            "采用先进的训练技术和安全过滤，输出内容更加可控和安全。"
            "8192 上下文窗口，适合商业应用中对内容安全要求较高的场景。"
        ),
        "strengths": ["安全对齐", "英文流畅", "内容可控", "Google技术"],
        "scores": {
            "中文对话": 3, "英文对话": 9, "代码生成": 7, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 7, "创意写作": 8,
            "翻译": 5, "指令遵循": 9, "长文本处理": 6,
        },
        "recommended_for": ["商业应用", "内容审核", "教育平台", "英文场景"],
        "hf_url": "https://huggingface.co/google/gemma-2-9b-it",
    },
    "Qwen2.5-Coder-7B": {
        "full_name_cn": "通义千问 2.5 代码 7B",
        "long_description": (
            "阿里专为代码生成优化训练的模型，在代码补全、代码解释、"
            "Bug 修复、算法实现等方面表现出色。支持 100+ 编程语言，"
            "是开发者本地编程助手的绝佳选择。中文提示也可获得良好效果。"
        ),
        "strengths": ["代码专精", "多语言编程", "Bug修复", "阿里维护"],
        "scores": {
            "中文对话": 6, "英文对话": 6, "代码生成": 10, "数学推理": 7,
            "逻辑推理": 8, "知识问答": 6, "创意写作": 5,
            "翻译": 6, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["编程开发", "代码审查", "技术学习", "软件工程"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
    },
    "Qwen2.5-14B-Instruct": {
        "full_name_cn": "通义千问 2.5 14B 指令版",
        "long_description": (
            "通义千问 14B 是中高端配置的最佳中文选择，各项能力较 7B 版本显著提升。"
            "中文写作、知识问答、复杂推理能力达到高水准。"
            "适合 16GB+ 内存的中高配电脑，体验接近商业大模型的中文水平。"
        ),
        "strengths": ["中文顶级", "高智商", "知识渊博", "写作优秀"],
        "scores": {
            "中文对话": 10, "英文对话": 8, "代码生成": 8, "数学推理": 8,
            "逻辑推理": 8, "知识问答": 9, "创意写作": 9,
            "翻译": 9, "指令遵循": 9, "长文本处理": 8,
        },
        "recommended_for": ["专业创作", "高级助手", "企业应用", "高配电脑"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
    },
    "Phi-3-medium-14B": {
        "full_name_cn": "微软 Phi-3 Medium 14B",
        "long_description": (
            "微软 Phi-3 中型版本，在 14B 参数下能力非常接近 GPT-3.5 水平。"
            "英文推理、代码生成和数学能力突出，特别是代码能力在 14B 级中属顶尖。"
            "需要 16GB+ 内存，适合对英文和代码要求较高的用户。"
        ),
        "strengths": ["代码顶尖", "英文推理", "数学能力", "接近GPT3.5"],
        "scores": {
            "中文对话": 4, "英文对话": 9, "代码生成": 9, "数学推理": 8,
            "逻辑推理": 8, "知识问答": 8, "创意写作": 7,
            "翻译": 6, "指令遵循": 9, "长文本处理": 7,
        },
        "recommended_for": ["英文编程", "高级研究", "代码测试", "中高配电脑"],
        "hf_url": "https://huggingface.co/microsoft/Phi-3-medium-4k-instruct",
    },
    "Mixtral-8x7B-Instruct": {
        "full_name_cn": "Mixtral 8x7B MoE",
        "long_description": (
            "Mistral 的混合专家 (MoE) 模型，总参数 46B，但每次推理仅激活约 12B 参数。"
            "英文能力接近 70B 大模型水平，多语言能力和代码能力超强。"
            "需要有 24GB+ 显存或内存的大配置设备，是高配置用户的顶级选择。"
        ),
        "strengths": ["MoE架构", "近70B能力", "多语言", "代码超强"],
        "scores": {
            "中文对话": 6, "英文对话": 10, "代码生成": 10, "数学推理": 9,
            "逻辑推理": 9, "知识问答": 9, "创意写作": 9,
            "翻译": 8, "指令遵循": 9, "长文本处理": 8,
        },
        "recommended_for": ["高端工作站", "专业编程", "学术研究", "24GB+设备"],
        "hf_url": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    },
    "Qwen2.5-32B-Instruct": {
        "full_name_cn": "通义千问 2.5 32B 指令版",
        "long_description": (
            "⭐ 旗舰天花板！通义千问 32B 是目前最强的开源中文模型之一，"
            "综合能力与 GPT-4o-mini 看齐。在中文理解、创作、专业知识方面的表现"
            "极为出色。需要 32GB+ 内存的最佳选择，真正达到商业大模型水平。"
        ),
        "strengths": ["中文巅峰", "商用级别", "全能型", "知识深度"],
        "scores": {
            "中文对话": 10, "英文对话": 9, "代码生成": 9, "数学推理": 9,
            "逻辑推理": 9, "知识问答": 10, "创意写作": 10,
            "翻译": 10, "指令遵循": 10, "长文本处理": 9,
        },
        "recommended_for": ["企业级部署", "专业工作站", "需要最高品质", "32GB+设备"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
    },
    "Gemma-2-27B": {
        "full_name_cn": "Google Gemma 2 27B",
        "long_description": (
            "Google 旗舰级开源模型，27B 参数展现出色的英文综合能力。"
            "在安全对齐、指令遵循和创意写作方面表现顶级。"
            "需 32GB+ 内存运行，是大配置英文用户的顶级选择。"
        ),
        "strengths": ["英文旗舰", "安全可控", "创意写作", "研发级"],
        "scores": {
            "中文对话": 4, "英文对话": 10, "代码生成": 9, "数学推理": 9,
            "逻辑推理": 9, "知识问答": 9, "创意写作": 10,
            "翻译": 7, "指令遵循": 10, "长文本处理": 8,
        },
        "recommended_for": ["企业英文场景", "内容创作平台", "研究机构", "32GB+设备"],
        "hf_url": "https://huggingface.co/google/gemma-2-27b-it",
    },
    # ========== 新增模型 ==========
    "DeepSeek-R1-Distill-Qwen-14B": {
        "full_name_cn": "DeepSeek R1 蒸馏 通义 14B",
        "long_description": (
            "DeepSeek R1 推理能力蒸馏到通义千问 14B，获得了更强大的推理深度。"
            "数学和逻辑推理能力接近 32B 级水平，是 16GB+ 设备上最强的推理模型。"
        ),
        "strengths": ["深度推理", "数学卓越", "代码精进", "性价比推理王"],
        "scores": {
            "中文对话": 8, "英文对话": 7, "代码生成": 9, "数学推理": 10,
            "逻辑推理": 10, "知识问答": 8, "创意写作": 7,
            "翻译": 7, "指令遵循": 8, "长文本处理": 7,
        },
        "recommended_for": ["数学竞赛", "算法研究", "编程竞赛", "16GB+设备"],
        "hf_url": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B",
        "paper_url": "https://arxiv.org/abs/2501.12948",
    },
    "CodeQwen2.5-7B-Instruct": {
        "full_name_cn": "CodeQwen 2.5 7B",
        "long_description": (
            "阿里专门优化的代码生成模型，在代码补全、代码重构、API 使用等方面"
            "进行了专项强化。支持 Python、JavaScript、Java、Go 等主流语言，"
            "适合作为 IDE 集成的本地编程助手。"
        ),
        "strengths": ["代码专精", "IDE友好", "代码重构", "多语言编程"],
        "scores": {
            "中文对话": 6, "英文对话": 7, "代码生成": 10, "数学推理": 7,
            "逻辑推理": 8, "知识问答": 6, "创意写作": 5,
            "翻译": 6, "指令遵循": 8, "长文本处理": 7,
        },
        "recommended_for": ["专业编程", "IDE集成", "代码审查", "技术文档"],
        "hf_url": "https://huggingface.co/Qwen/CodeQwen2.5-7B-Instruct",
    },
    "OpenCode-8B": {
        "full_name_cn": "OpenCode 8B",
        "long_description": (
            "专为开源代码开发训练的模型，在 Python 和 JavaScript 上的表现"
            "尤为突出。支持完整的代码理解和生成流程，可作为本地 Copilot 替代方案。"
        ),
        "strengths": ["代码专项", "Python专家", "开源友好", "Copilot替代"],
        "scores": {
            "中文对话": 4, "英文对话": 7, "代码生成": 10, "数学推理": 6,
            "逻辑推理": 7, "知识问答": 5, "创意写作": 3,
            "翻译": 5, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["开源开发", "Python编程", "本地Copilot", "代码学习"],
        "hf_url": "https://huggingface.co/opencode/OpenCode-8B",
    },
    "Yi-6B": {
        "full_name_cn": "零一万物 Yi-6B",
        "long_description": (
            "零一万物推出的 6B 双语模型，中文和英文能力均衡。"
            "在 6B 级别中中英能力都相当不错，被广泛用于中文应用场景。"
            "支持 4K 上下文，对内存要求友好。"
        ),
        "strengths": ["中英平衡", "创业团队", "质量稳定", "社区活跃"],
        "scores": {
            "中文对话": 8, "英文对话": 8, "代码生成": 6, "数学推理": 5,
            "逻辑推理": 6, "知识问答": 7, "创意写作": 7,
            "翻译": 8, "指令遵循": 7, "长文本处理": 5,
        },
        "recommended_for": ["中英双语场景", "通用助手", "学习研究"],
        "hf_url": "https://huggingface.co/01-ai/Yi-6B",
    },
    "Mistral-Nemo-12B": {
        "full_name_cn": "Mistral Nemo 12B",
        "long_description": (
            "Mistral 与 NVIDIA 合作推出的 12B 模型，在通用能力和多语言上表现优异。"
            "128K 上下文长度，英文和代码能力顶级。"
            "是 16GB 设备上最值得考虑的全能型模型之一。"
        ),
        "strengths": ["128K上下文", "多语言", "NVIDIA合作", "全能型"],
        "scores": {
            "中文对话": 5, "英文对话": 9, "代码生成": 9, "数学推理": 8,
            "逻辑推理": 8, "知识问答": 8, "创意写作": 8,
            "翻译": 7, "指令遵循": 9, "长文本处理": 9,
        },
        "recommended_for": ["长文本处理", "文档分析", "研究助手", "16GB+设备"],
        "hf_url": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
    },
    "Qwen2.5-Coder-14B": {
        "full_name_cn": "通义千问 2.5 代码 14B",
        "long_description": (
            "代码专用模型的 14B 升级版，代码能力大幅提升。"
            "能够理解复杂项目结构、进行跨文件分析、重构大片代码，"
            "是目前最强的开源本地代码模型之一。"
        ),
        "strengths": ["代码天花板", "项目理解", "跨文件分析", "大型重构"],
        "scores": {
            "中文对话": 7, "英文对话": 8, "代码生成": 10, "数学推理": 8,
            "逻辑推理": 9, "知识问答": 7, "创意写作": 6,
            "翻译": 7, "指令遵循": 9, "长文本处理": 8,
        },
        "recommended_for": ["大型项目开发", "代码架构", "企业开发", "20GB+设备"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct",
    },
    "StarCoder2-15B": {
        "full_name_cn": "StarCoder2 15B",
        "long_description": (
            "BigCode 推出的代码生成模型，在 600+ 编程语言上训练。"
            "专精于代码补全和代码分析，是编程竞赛和批量代码处理的利器。"
            "GitHub 代码数据预训练，对常见代码模式理解深入。"
        ),
        "strengths": ["600+语言", "GitHub训练", "代码分析", "社区驱动"],
        "scores": {
            "中文对话": 3, "英文对话": 6, "代码生成": 10, "数学推理": 7,
            "逻辑推理": 8, "知识问答": 5, "创意写作": 2,
            "翻译": 4, "指令遵循": 7, "长文本处理": 7,
        },
        "recommended_for": ["编程竞赛", "代码库分析", "批量处理", "高配开发者"],
        "hf_url": "https://huggingface.co/bigcode/starcoder2-15b",
    },
    "MiniCPM3-4B": {
        "full_name_cn": "MiniCPM3 4B",
        "long_description": (
            "面壁智能推出的高效小型模型，在 4B 参数下实现了接近 7B 模型的能力。"
            "中文表现优异，数学和推理能力突出，端侧部署的标杆模型。"
            "特别针对低资源设备优化，4GB 内存即可流畅运行。"
        ),
        "strengths": ["超高效", "端侧标杆", "中文优秀", "数学推理好"],
        "scores": {
            "中文对话": 8, "英文对话": 6, "代码生成": 6, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 7, "创意写作": 6,
            "翻译": 7, "指令遵循": 8, "长文本处理": 6,
        },
        "recommended_for": ["端侧设备", "手机部署", "中文助手", "低配电脑"],
        "hf_url": "https://huggingface.co/openbmb/MiniCPM3-4B",
    },
    "GLM-4-9B-Chat": {
        "full_name_cn": "智谱 GLM-4 9B Chat",
        "long_description": (
            "智谱 AI 推出的 GLM-4 系列 9B 模型，中文理解和生成能力极为出色。"
            "继承 GLM-4 的优秀血统，在中文知识问答、阅读理解、"
            "内容创作方面表现一流。128K 上下文，适合长文档处理。"
        ),
        "strengths": ["中文深厚", "GLM血统", "128K上下文", "知识丰富"],
        "scores": {
            "中文对话": 10, "英文对话": 7, "代码生成": 7, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 9, "创意写作": 9,
            "翻译": 8, "指令遵循": 9, "长文本处理": 9,
        },
        "recommended_for": ["中文专业写作", "长文档处理", "知识问答", "办公主力"],
        "hf_url": "https://huggingface.co/THUDM/glm-4-9b-chat",
    },
    "ChatGLM3-6B": {
        "full_name_cn": "智谱 ChatGLM3 6B",
        "long_description": (
            "经典的 ChatGLM3 模型，在中文对话领域享有盛誉。"
            "中文对话流畅、知识面广，特别适合中文聊天和轻度办公场景。"
            "社区成熟，衍生工具丰富，是入门级中文模型的经典之选。"
        ),
        "strengths": ["中文经典", "社区成熟", "工具丰富", "稳定可靠"],
        "scores": {
            "中文对话": 9, "英文对话": 6, "代码生成": 5, "数学推理": 5,
            "逻辑推理": 5, "知识问答": 7, "创意写作": 7,
            "翻译": 7, "指令遵循": 8, "长文本处理": 5,
        },
        "recommended_for": ["中文聊天", "入门体验", "教育研究", "社区工具体验"],
        "hf_url": "https://huggingface.co/THUDM/chatglm3-6b",
    },
    "OpenChat-3.6-8B": {
        "full_name_cn": "OpenChat 3.6 8B",
        "long_description": (
            "社区最受欢迎的对话模型之一，基于 Llama 架构进行了深度微调。"
            "在对话流畅度和指令遵循上达到了 8B 模型的顶尖水平。"
            "适合需要自然流畅对话体验的用户。"
        ),
        "strengths": ["对话流畅", "指令遵循好", "社区喜爱", "自然交互"],
        "scores": {
            "中文对话": 6, "英文对话": 9, "代码生成": 7, "数学推理": 5,
            "逻辑推理": 6, "知识问答": 7, "创意写作": 8,
            "翻译": 6, "指令遵循": 9, "长文本处理": 6,
        },
        "recommended_for": ["聊天机器人", "对话系统", "客服替代", "英文助手"],
        "hf_url": "https://huggingface.co/openchat/openchat-3.6-8b-20240522",
    },
    "Phi-3.5-mini-3.8B": {
        "full_name_cn": "微软 Phi-3.5 Mini 3.8B",
        "long_description": (
            "Phi-3 的升级版 3.5，在原有优秀基础上改进了多语言能力和推理深度。"
            "3.8B 小巧体积展现 7B 级实力，特别适合对体积和性能都有要求的场景。"
        ),
        "strengths": ["升级版", "多语言", "小体积大能力", "微软出品"],
        "scores": {
            "中文对话": 4, "英文对话": 8, "代码生成": 7, "数学推理": 7,
            "逻辑推理": 7, "知识问答": 7, "创意写作": 5,
            "翻译": 5, "指令遵循": 9, "长文本处理": 6,
        },
        "recommended_for": ["轻量开发", "移动设备", "嵌入式应用", "资源受限场景"],
        "hf_url": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
    },
    "Gemma-2-2B-JPN": {
        "full_name_cn": "Gemma 2 2B 日语优化版",
        "long_description": (
            "Gemma 2 的日语优化版本，在保持英文能力的同时大幅提升了日语处理能力。"
            "是日语学习者和日本用户的小型理想模型。"
        ),
        "strengths": ["日语优化", "双语", "小体积", "安全"],
        "scores": {
            "中文对话": 2, "英文对话": 7, "代码生成": 4, "数学推理": 3,
            "逻辑推理": 3, "知识问答": 5, "创意写作": 4,
            "翻译": 6, "指令遵循": 8, "长文本处理": 4,
        },
        "recommended_for": ["日语学习", "日本市场", "双语需求"],
        "hf_url": "https://huggingface.co/google/gemma-2-2b-jpn-it",
    },
    "Qwen2.5-72B-Instruct": {
        "full_name_cn": "通义千问 2.5 72B 指令版",
        "long_description": (
            "通义千问 2.5 终极版，72B 参数的综合能力直逼 GPT-4o。"
            "需要 48GB+ 内存。中文理解与创作能力达到顶尖水平，"
            "知识储备极深，是高配置数据中心或工作站顶配才可运行的终极模型。"
        ),
        "strengths": ["终极中文", "类GPT4", "知识渊博", "商用顶级"],
        "scores": {
            "中文对话": 10, "英文对话": 10, "代码生成": 10, "数学推理": 10,
            "逻辑推理": 10, "知识问答": 10, "创意写作": 10,
            "翻译": 10, "指令遵循": 10, "长文本处理": 10,
        },
        "recommended_for": ["专业数据中心", "高配工作站", "需要最高品质", "48GB+设备"],
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
    },
}

# ======================== 硬件检测增强 ========================

def _subprocess_no_window_flags():
    """子进程标志：Windows 上隐藏控制台黑窗口"""
    import sys
    import subprocess
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def detect_vulkan_support() -> bool:
    """检测是否支持 Vulkan GPU 加速"""
    try:
        import subprocess
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10,
            creationflags=_subprocess_no_window_flags(),
        )
        return result.returncode == 0
    except Exception:
        pass
    # 备选：检查 Vulkan SDK 路径
    for path in os.environ.get("PATH", "").split(os.pathsep):
        if "vulkan" in path.lower():
            return True
    return False


def detect_rocm_support() -> bool:
    """检测是否支持 AMD ROCm"""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, timeout=5,
            creationflags=_subprocess_no_window_flags(),
        )
        return result.returncode == 0
    except Exception:
        pass
    return False


def get_gpu_model_name() -> str:
    """获取 GPU 型号名称"""
    try:
        import subprocess
        # Windows
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            capture_output=True, text=True, timeout=10,
            creationflags=_subprocess_no_window_flags(),
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip() and l.strip() != "Name"]
            return lines[0] if lines else "Unknown"
    except Exception:
        pass
    # Linux
    try:
        result = subprocess.run(
            ["lspci", "|", "grep", "-i", "VGA"],
            capture_output=True, text=True, shell=True, timeout=5,
            creationflags=_subprocess_no_window_flags(),
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def get_vram_detailed_gb() -> float:
    """获取更精确的显存大小"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    try:
        # Windows DXGI 查询
        import subprocess
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "AdapterRAM"],
            capture_output=True, text=True, timeout=10,
            creationflags=_subprocess_no_window_flags(),
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024**3)
    except Exception:
        pass
    return 0