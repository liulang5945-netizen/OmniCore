"""
ModelSelf 完整训练流程编排

顺序执行：
1. 自动生成训练数据（使用 data_generator）
2. 创建模型（随机初始化或加载 checkpoint）
3. 第一步：预训练 — 语言建模
4. 第二步：ReAct 微调 — 工具调用 + 对话

用法：
    python -m taiji.train_pipeline --size 125m --device cpu --num_react 5000 --num_conv 1000
    python -m taiji.train_pipeline --size 350m --device cuda --resume ./taiji_checkpoints/pretrain/best
"""
import os
import sys
import json
import argparse
import logging
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")


def generate_data(
    num_react: int = 5000,
    num_conv: int = 1000,
    output_dir: str = "taiji/training_data",
) -> tuple:
    """
    生成训练数据并导出。

    Returns:
        (react_samples, conv_samples)
    """
    from taiji.data_generator import (
        generate_bulk_react_data,
        generate_bulk_conversation_data,
        export_data,
    )

    logger.info(f"Generating {num_react} ReAct samples...")
    react = generate_bulk_react_data(num_react)

    logger.info(f"Generating {num_conv} conversation samples...")
    conv = generate_bulk_conversation_data(num_conv)

    export_data(react, conv, output_dir)
    return react, conv


def create_or_load_model(
    size: str = "125m",
    device: str = "cpu",
    resume: Optional[str] = None,
) -> tuple:
    """
    创建新模型或加载 checkpoint。

    Returns:
        (model, tokenizer)
    """
    from taiji.loader import create_model, load_model
    from taiji.config import ModelConfig

    # 默认只启用语言头+工具头（感知/记忆/规划头无训练数据）
    default_active_heads = ["language", "tool"]

    if resume and os.path.exists(resume):
        logger.info(f"Loading checkpoint from {resume}")
        model, tokenizer = load_model(resume, device=device)
        return model, tokenizer

    logger.info(f"Creating new model: {size}")
    model, tokenizer = create_model(size, device=device, active_heads=default_active_heads)
    return model, tokenizer


def register_tools(tokenizer, model):
    """
    向分词器和模型注册所有工具。
    """
    tool_names = [
        "read_local_file", "write_file", "edit_file", "list_directory",
        "execute_python", "search", "read_webpage", "create_project",
        "analyze_code", "install_dependency", "learn_knowledge",
        "query_knowledge", "run_command",
    ]
    for name in tool_names:
        tokenizer.register_tool(name)
    model.set_num_tools(len(tokenizer._tool_name_to_id))
    logger.info(f"Registered {len(tool_names)} tools")
    return tool_names


def build_pretrain_dataset(
    tokenizer,
    source: str = "auto",
    max_length: int = 512,
) -> "TextDataset":
    """
    构建预训练数据集。
    
    Args:
        tokenizer: ModelSelfTokenizer
        source: "auto" 使用内置模板文本，或指向 .txt 文件的路径
        max_length: 序列最大长度

    Returns:
        TextDataset
    """
    from taiji.trainer import TextDataset

    if source and source != "auto" and os.path.exists(source):
        # 从外部文本文件加载
        with open(source, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} lines from {source}")
    else:
        # 使用内置的种子数据文本 + 自动生成的对话文本
        from taiji.seed_data import get_seed_conversation_data
        from taiji.data_generator import generate_bulk_conversation_data

        # 种子数据
        texts = []
        for item in get_seed_conversation_data():
            for msg in item.get("messages", []):
                texts.append(msg["content"])

        # 自动生成补充
        extra = generate_bulk_conversation_data(500)
        for item in extra:
            for msg in item.get("messages", []):
                texts.append(msg["content"])

        # 代码语料（内置 Python 代码片段）
        code_snippets = [
            """
def fibonacci(n):
    '''计算斐波那契数列的前 n 项'''
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

def quick_sort(arr):
    '''快速排序算法'''
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def is_prime(n):
    '''判断素数'''
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
            """
Transformer 是一种深度学习架构，由 Google 在 2017 年提出。
核心组件包括：
1. 自注意力机制（Self-Attention）：让模型关注输入序列中的不同位置
2. 多头注意力（Multi-Head Attention）：并行多个注意力头
3. 前馈网络（FFN）：对每个位置独立处理
4. 位置编码（Positional Encoding）：注入序列顺序信息

Python 是一种高级、解释型、通用的编程语言。
主要特点包括简洁易读、动态类型、丰富的标准库、跨平台。

机器学习是让计算机从数据中自动学习规律的技术。
三大类型：监督学习、无监督学习、强化学习。

Docker 是一个容器化平台，可以将应用及其依赖打包成容器。
核心概念：镜像（Image）、容器（Container）、Dockerfile。
""",
        ]
        texts.extend(code_snippets)

        logger.info(f"Built pretrain corpus: {len(texts)} texts")

    return TextDataset(tokenizer, texts, max_length=max_length)


def run_pipeline(args):
    """执行完整训练流程"""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Step 0: 创建/加载模型
    # Step 0.5: 训练 SentencePiece 分词器（如果不存在）
    sp_model_path = os.path.join("taiji", "tokenizer", "sentencepiece.model")
    if not os.path.exists(sp_model_path):
        _train_sentencepiece_if_needed()
    else:
        logger.info(f"SentencePiece 模型已存在: {sp_model_path}")

    model, tokenizer = create_or_load_model(args.size, device, args.resume)
    register_tools(tokenizer, model)
    model.print_model_info()

    # Step 1: 生成训练数据
    if args.skip_data_gen:
        from taiji.data_generator import load_exported_data
        react, conv = load_exported_data("taiji/training_data")
        logger.info(f"Loaded {len(react)} react + {len(conv)} conv samples from disk")
    else:
        react, conv = generate_data(args.num_react, args.num_conv)

    # Step 2: 预训练（如果未指定跳过）
    if not args.skip_pretrain:
        from taiji.trainer import ModelSelfTrainer

        logger.info("=" * 60)
        logger.info("STEP 1: Pre-training (Language Modeling)")
        logger.info("=" * 60)

        pretrain_dataset = build_pretrain_dataset(
            tokenizer,
            source=args.pretrain_corpus,
            max_length=args.max_length,
        )
        logger.info(f"Pretrain dataset size: {len(pretrain_dataset)} chunks")

        trainer = ModelSelfTrainer(
            model, tokenizer,
            learning_rate=args.pretrain_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.grad_accum,
        )

        for progress, desc, loss_hist, metrics in trainer.pretrain(
            pretrain_dataset,
            num_epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            save_dir=os.path.join(args.save_dir, "pretrain"),
            save_steps=args.save_steps,
            log_steps=args.log_steps,
            device=device,
        ):
            logger.info(f"  [{progress*100:.0f}%] {desc}")
            if metrics.get("status") == "completed":
                _loss = metrics.get('loss', None)
                logger.info(f"✅ Pretrain completed! Final loss: {_loss:.4f}" if isinstance(_loss, (int, float)) else "✅ Pretrain completed!")

    # Step 3: ReAct 微调
    if not args.skip_finetune:
        from taiji.trainer import ModelSelfTrainer, build_dataset

        logger.info("=" * 60)
        logger.info("STEP 2: ReAct Fine-tuning")
        logger.info("=" * 60)

        # 构建数据集（使用种子数据 + 自动生成的数据）
        finetune_dataset = build_dataset(
            tokenizer,
            extra_react_data=react,
            extra_conv_data=conv,
            max_length=args.max_length,
        )
        logger.info(f"Finetune dataset size: {len(finetune_dataset)} samples")

        trainer = ModelSelfTrainer(
            model, tokenizer,
            learning_rate=args.finetune_lr,
            weight_decay=args.weight_decay,
            warmup_steps=max(args.warmup_steps // 2, 10),
            gradient_accumulation_steps=args.grad_accum,
        )

        for progress, desc, loss_hist, metrics in trainer.finetune(
            finetune_dataset,
            num_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            save_dir=os.path.join(args.save_dir, "finetune"),
            save_steps=args.save_steps,
            log_steps=args.log_steps,
            device=device,
        ):
            logger.info(f"  [{progress*100:.0f}%] {desc}")
            if metrics.get("status") == "completed":
                _loss = metrics.get('loss', None)
                logger.info(f"✅ Finetune completed! Final loss: {_loss:.4f}" if isinstance(_loss, (int, float)) else "✅ Finetune completed!")

    logger.info("🎉 Training pipeline complete!")
    logger.info(f"Final model saved to: {os.path.join(args.save_dir, 'finetune', 'best')}")


def _train_sentencepiece_if_needed():
    """训练 SentencePiece BPE 分词器（如果尚未存在）"""
    import subprocess
    import glob

    sp_model_path = os.path.join("taiji", "tokenizer", "sentencepiece.model")
    if os.path.exists(sp_model_path):
        return

    logger.info("SentencePiece 模型不存在，开始训练...")
    os.makedirs(os.path.join("taiji", "tokenizer"), exist_ok=True)

    # 收集训练语料
    corpus_file = os.path.join("taiji", "tokenizer", "sp_corpus.txt")
    texts = []

    # 1. 项目中的代码和文档
    for pattern in ["*.py", "*.md", "*.json", "*.txt"]:
        for fpath in glob.glob(os.path.join("taiji", pattern)):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception:
                pass

    # 2. 种子数据的对话文本
    try:
        from taiji.seed_data import get_seed_conversation_data
        for item in get_seed_conversation_data():
            for msg in item.get("messages", []):
                texts.append(msg["content"])
    except Exception:
        pass

    # 3. 内置中文通用语料（确保分词器覆盖中文）
    general_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
        "自然语言处理是人工智能和语言学领域的分支学科，探讨如何处理及运用自然语言。",
        "机器学习是一门多学科交叉专业，涵盖概率论知识、统计学知识、近似理论知识和复杂算法知识。",
        "Python是一种广泛使用的高级编程语言，由吉多·范罗苏姆创造，第一版发布于1991年。",
        "PyTorch是一个基于Python的可续计算包，提供两个高级功能：具有强大的GPU加速的张量计算，以及包含自动求导系统的深度神经网络。",
        "Transformer模型是一种采用自注意力机制的深度学习模型，最初应用于自然语言处理领域。",
        "态极是一个原生训练的AI生命体，不是微调的预训练模型。态极有自己的大脑结构、生命系统和进化能力。",
        "OmniCore是一个本地AI助手工作站，支持多种AI后端，提供Agent模式、RAG知识库、模型微调训练等功能。",
        "当用户需要读取文件时，应该使用read_local_file工具。当用户需要创建文件时，应该使用write_file工具。",
        "ReAct是一种推理和行动交替进行的Agent框架，模型先思考再选择工具执行，然后观察结果继续推理。",
    ]
    texts.extend(general_texts)

    if len(texts) < 10:
        logger.warning("语料不足，跳过 SentencePiece 训练，将使用字符级回退")
        return

    # 写入语料文件
    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in texts:
            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 5:
                    f.write(line + "\n")

    logger.info(f"语料文件已生成: {corpus_file} ({len(texts)} 段文本)")

    # 训练 SentencePiece
    try:
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=os.path.join("taiji", "tokenizer", "sentencepiece"),
            vocab_size=32000,
            model_type="bpe",
            character_coverage=0.9995,
            num_threads=4,
            max_sentence_length=4096,
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2,
            user_defined_symbols=[
                "<think>", "</think>", "<tool_call>", "<tool_result>",
                "<final_answer>", "<observe>", "</observe>",
                "<mem_read>", "<mem_write>",
                "<plan>", "</plan>", "<step>", "</step>",
                "<reflect>", "</reflect>",
            ],
        )
        logger.info(f"✅ SentencePiece 模型训练完成: {sp_model_path}")
    except ImportError:
        logger.warning("sentencepiece 未安装，将使用字符级回退分词器")
    except Exception as e:
        logger.error(f"SentencePiece 训练失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="ModelSelf Training Pipeline")

    # 模型参数
    parser.add_argument("--size", type=str, default="125m", choices=["125m", "350m", "1b"],
                        help="模型大小")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="训练设备")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（路径）")

    # 数据参数
    parser.add_argument("--num_react", type=int, default=5000,
                        help="生成 ReAct 样本数量")
    parser.add_argument("--num_conv", type=int, default=1000,
                        help="生成对话样本数量")
    parser.add_argument("--skip_data_gen", action="store_true",
                        help="跳过数据生成，使用已导出的数据")
    parser.add_argument("--pretrain_corpus", type=str, default="auto",
                        help="预训练语料路径（.txt），默认 auto 使用内置文本")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="序列最大长度")
    parser.add_argument("--pretrain_lr", type=float, default=3e-4,
                        help="预训练学习率")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="微调学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="预热步数")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="梯度累积步数")

    # 流程控制
    parser.add_argument("--pretrain_epochs", type=int, default=5,
                        help="预训练轮数")
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="微调轮数")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存间隔步数")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="日志间隔步数")
    parser.add_argument("--save_dir", type=str, default="./taiji_checkpoints",
                        help="checkpoint 保存目录")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="跳过预训练步骤（直接微调）")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="跳过微调步骤（仅预训练）")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()