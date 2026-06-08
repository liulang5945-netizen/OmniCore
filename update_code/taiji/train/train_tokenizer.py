"""
训练 SentencePiece 分词器

用 OmniCore 的种子数据 + 通用文本训练一个小型分词器。
词表大小 32000，与 LLaMA 兼容。
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("TrainTokenizer")


def create_training_corpus(output_path: str):
    """创建分词器训练语料"""
    from taiji.seed_data import get_seed_react_data, get_seed_conversation_data

    lines = []

    # 从种子数据中提取文本
    for item in get_seed_react_data():
        lines.append(item["task"])
        for step in item.get("steps", []):
            if step.get("thought"):
                lines.append(step["thought"])
            if step.get("final_answer"):
                lines.append(step["final_answer"])

    for item in get_seed_conversation_data():
        for msg in item.get("messages", []):
            lines.append(msg["content"])

    # 添加通用中文和英文文本
    general_texts = [
        "你好，我是 OmniCore AI 助手。",
        "我可以帮你完成文件操作、代码执行、信息搜索等任务。",
        "Python 是一种高级编程语言，广泛应用于数据科学和人工智能。",
        "机器学习是人工智能的一个分支，让计算机从数据中学习。",
        "深度学习使用神经网络进行学习，是机器学习的子领域。",
        "快速排序是一种高效的排序算法，平均时间复杂度为 O(n log n)。",
        "递归是一种编程技巧，指函数调用自身。",
        "装饰器是 Python 的一种设计模式，使用 @ 语法糖。",
        "Hello, I am OmniCore AI assistant.",
        "I can help you with file operations, code execution, and information search.",
        "Python is a high-level programming language widely used in data science.",
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks for learning.",
        "Quick sort is an efficient sorting algorithm.",
        "Recursion is a programming technique where a function calls itself.",
        "Decorators are a design pattern in Python using @ syntax.",
        "def hello_world():",
        "print('Hello World')",
        "import os",
        "import sys",
        "from typing import List, Dict",
        "class MyClass:",
        "    def __init__(self):",
        "        self.value = 0",
        "    def method(self):",
        "        return self.value",
        "for i in range(10):",
        "    print(i)",
        "if __name__ == '__main__':",
        "    main()",
        "try:",
        "    result = do_something()",
        "except Exception as e:",
        "    print(f'Error: {e}')",
        "with open('file.txt', 'r') as f:",
        "    content = f.read()",
        "json.dumps({'key': 'value'})",
        "os.path.join('dir', 'file.txt')",
        "torch.tensor([1, 2, 3])",
        "model.forward(input_ids)",
        "optimizer.step()",
        "loss.backward()",
    ]
    lines.extend(general_texts)

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")

    logger.info(f"Created training corpus: {len(lines)} lines -> {output_path}")
    return len(lines)


def train_sentencepiece(input_path: str, output_dir: str, vocab_size: int = 32000):
    """训练 SentencePiece 模型"""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "sentencepiece")

    logger.info(f"Training SentencePiece model (vocab_size={vocab_size})...")

    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    model_path = model_prefix + ".model"
    logger.info(f"SentencePiece model saved to: {model_path}")
    return model_path


def test_tokenizer(model_path: str):
    """测试训练好的分词器"""
    from taiji.tokenizer import ModelSelfTokenizer

    tokenizer = ModelSelfTokenizer(sp_model_path=model_path)

    # 测试编码解码
    test_texts = [
        "你好，我是 OmniCore AI 助手。",
        "def hello_world():\n    print('Hello')",
        "请搜索 Python 教程",
        "I can help you with file operations.",
    ]

    for text in test_texts:
        result = tokenizer(text)
        decoded = tokenizer.decode(result["input_ids"][0], skip_special_tokens=True)
        print(f"  原文: {text[:40]}...")
        print(f"  编码: {result['input_ids'].shape}")
        print(f"  解码: {decoded[:40]}...")
        print()

    # 测试特殊 token
    tokenizer.register_tool("search")
    tokenizer.register_tool("read_file")

    special_text = "<think>test</think><tool_call>search"
    result = tokenizer(special_text)
    ids = result["input_ids"][0].tolist()
    print(f"  Special token IDs: {ids}")
    print(f"  Has think_start (32000): {32000 in ids}")
    print(f"  Has think_end (32001): {32001 in ids}")
    print(f"  Has tool_call (32002): {32002 in ids}")
    print(f"  Has search (32010): {32010 in ids}")


if __name__ == "__main__":
    print("=" * 60)
    print("Training ModelSelf Tokenizer")
    print("=" * 60)
    print()

    # 1. 创建训练语料
    corpus_path = "taiji/tokenizer_corpus.txt"
    num_lines = create_training_corpus(corpus_path)

    # 2. 训练 SentencePiece (用较小的词表，因为语料较小)
    model_path = train_sentencepiece(corpus_path, "taiji/tokenizer", vocab_size=3000)

    # 3. 测试
    print("\n--- Testing Tokenizer ---\n")
    test_tokenizer(model_path)

    # 清理语料文件
    os.remove(corpus_path)
    print("\n" + "=" * 60)
    print("Tokenizer training completed!")
    print(f"Model: {model_path}")
    print("=" * 60)
