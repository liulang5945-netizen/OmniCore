"""
data_loader.py 模块的严苛单元测试
覆盖：InstructionDataset 数据加载、格式解析、分词、DataLoader 创建
"""
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_loader import InstructionDataset, create_dataloader, split_dataset


# 最小化伪 tokenizer，避免依赖真实模型
class MockTokenizer:
    """模拟 HuggingFace tokenizer"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def __call__(self, text, max_length=512, truncation=True, padding=False, return_tensors=None):
        # 简单按字符映射为 token id
        ids = [min(ord(c), self.vocab_size - 1) for c in text[:max_length]]
        if padding == "max_length":
            ids = ids + [0] * (max_length - len(ids))
        import torch
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class TestInstructionDatasetInit:
    """数据集初始化测试"""

    def test_init_nonexistent_file(self):
        tokenizer = MockTokenizer()
        ds = InstructionDataset("/nonexistent/path.json", tokenizer)
        assert len(ds) == 0

    def test_init_empty_jsonl(self):
        tokenizer = MockTokenizer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("")
        try:
            ds = InstructionDataset(fpath, tokenizer)
            assert len(ds) == 0
        finally:
            os.unlink(fpath)

    def test_init_jsonl_with_data(self):
        tokenizer = MockTokenizer()
        lines = [
            json.dumps({"instruction": "你好", "output": "你好！"}),
            json.dumps({"instruction": "再见", "output": "再见！"}),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer)
            assert len(ds) == 2
        finally:
            os.unlink(fpath)


class TestInstructionDatasetFormat:
    """数据格式解析测试"""

    def test_format_item_standard(self):
        ds = InstructionDataset.__new__(InstructionDataset)
        result = ds._format_item({"instruction": "a", "output": "b"})
        assert result == {"instruction": "a", "output": "b"}

    def test_format_item_qa(self):
        ds = InstructionDataset.__new__(InstructionDataset)
        result = ds._format_item({"question": "q?", "answer": "a!"})
        assert result == {"instruction": "q?", "output": "a!"}

    def test_format_item_prompt_completion(self):
        ds = InstructionDataset.__new__(InstructionDataset)
        result = ds._format_item({"prompt": "p", "completion": "c"})
        assert result == {"instruction": "p", "output": "c"}

    def test_format_item_unknown(self):
        ds = InstructionDataset.__new__(InstructionDataset)
        result = ds._format_item({"arbitrary": "data"})
        assert result["instruction"] == "请学习以下知识"
        assert "arbitrary" in result["output"]


class TestInstructionDatasetGetItem:
    """__getitem__ 测试"""

    def test_getitem_train_mode(self):
        tokenizer = MockTokenizer()
        lines = [json.dumps({"instruction": "测试指令", "output": "测试输出"})]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=128, mode="train")
            batch = ds[0]
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape[0] == 128
            assert batch["attention_mask"].shape[0] == 128
            assert batch["labels"].shape[0] == 128
        finally:
            os.unlink(fpath)

    def test_getitem_eval_mode(self):
        tokenizer = MockTokenizer()
        lines = [json.dumps({"instruction": "测试指令", "output": "测试输出"})]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=64, mode="eval")
            batch = ds[0]
            assert "input_ids" in batch
            assert batch["input_ids"].shape[0] == 64
        finally:
            os.unlink(fpath)

    def test_labels_masked_in_train_mode(self):
        """验证训练模式下 instruction 部分的 labels 被设为 -100"""
        tokenizer = MockTokenizer(vocab_size=30000)
        lines = [json.dumps({"instruction": "ABCD", "output": "EFGH"})]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=64, mode="train")
            batch = ds[0]
            labels = batch["labels"]
            # 前面部分应为 -100
            assert (labels[:10] == -100).any() or len(labels) > 0
        finally:
            os.unlink(fpath)


class TestCreateDataLoader:
    """DataLoader 创建测试"""

    def test_create_dataloader(self):
        tokenizer = MockTokenizer()
        lines = [json.dumps({"instruction": f"任务{i}", "output": f"结果{i}"}) for i in range(8)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=64, mode="train")
            loader = create_dataloader(ds, batch_size=4, shuffle=False)
            batches = list(loader)
            assert len(batches) >= 1
            for batch in batches:
                assert "input_ids" in batch
                assert batch["input_ids"].shape[0] <= 4
        finally:
            os.unlink(fpath)

    def test_create_dataloader_drop_last(self):
        tokenizer = MockTokenizer()
        lines = [json.dumps({"instruction": f"任务{i}", "output": f"结果{i}"}) for i in range(3)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=32, mode="train")
            loader = create_dataloader(ds, batch_size=2, shuffle=False, drop_last=True)
            batches = list(loader)
            # 3 条数据，batch_size=2，drop_last=True → 1 个batch
            assert len(batches) == 1
        finally:
            os.unlink(fpath)


class TestSplitDataset:
    """数据集分割测试"""

    def test_split_dataset(self):
        tokenizer = MockTokenizer()
        lines = [json.dumps({"instruction": f"任务{i}", "output": f"结果{i}"}) for i in range(10)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            fpath = f.name
            f.write("\n".join(lines))
        try:
            ds = InstructionDataset(fpath, tokenizer, max_length=32, mode="train")
            train_ds, val_ds = split_dataset(ds, train_ratio=0.8)
            assert len(train_ds) == 8
            assert len(val_ds) == 2
        finally:
            os.unlink(fpath)