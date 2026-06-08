"""
数据集加载与预处理模块
支持多种文件格式，为微调提供标准化的 InstructionDataset
支持合并多个文件为单一数据集
"""
import json
import logging
import os
import re
from typing import List, Union, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

from tools.file_parser import parse_file_to_text

logger = logging.getLogger("DataLoader")


class InstructionDataset(Dataset):
    """
    指令微调数据集
    支持从多种文件格式解析，自动转换为标准化的 instruction 格式
    支持传入单个文件路径、文件路径列表、或预解析的文本字符串

    数据样例:
        { "instruction": "请解释...", "output": "这是..." }
        或自由文本格式，自动拼接为 instruction + output
    """

    def __init__(self, file_path: Optional[Union[str, List[str]]] = None,
                 tokenizer=None, max_length: int = 512, mode: str = "train",
                 raw_text: Optional[str] = None, file_name: str = "",
                 pre_tokenize: bool = True, max_pre_tokenize: int = 5000):
        """
        Args:
            max_pre_tokenize: 预编码样本数上限。超过此阈值自动降级为
                惰性 tokenize（__getitem__ 按需编码），避免大数据集 OOM。
                设为 0 表示始终惰性编码，设为 float('inf') 表示始终预编码。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = []
        self._encoded_data = None
        self._max_pre_tokenize = max_pre_tokenize

        # 优先使用预解析文本（由外部先完成 PDF OCR 等耗时操作）
        if raw_text is not None and raw_text.strip():
            ext = os.path.splitext(file_name)[1].lower() if file_name else ""
            if ext == ".jsonl":
                self._load_jsonl(raw_text)
            elif ext == ".json":
                self._load_json(raw_text)
            else:
                self._load_raw_text(raw_text)
            logger.info(f"数据集加载完成: {file_name or '预解析文本'} -> {len(self.data)} 条")
        elif file_path is None:
            logger.warning("未提供文件路径或预解析文本，数据集为空")
        elif isinstance(file_path, list):
            for fp in file_path:
                if os.path.exists(fp):
                    self._load_data(fp)
                else:
                    logger.warning(f"文件不存在: {fp}，跳过")
        elif isinstance(file_path, str):
            if os.path.exists(file_path):
                self._load_data(file_path)
            else:
                logger.warning(f"文件不存在: {file_path}，使用空数据集")
        else:
            logger.warning(f"无效的文件路径类型: {type(file_path)}")

        # 预编码全部样本，避免 __getitem__ 中实时 tokenize 阻塞训练循环
        # 当数据集超过 max_pre_tokenize 阈值时，自动降级为惰性编码防止 OOM
        if pre_tokenize and self.tokenizer is not None and len(self.data) > 0:
            if len(self.data) <= self._max_pre_tokenize:
                self._pre_tokenize()
            else:
                logger.warning(
                    f"⚠️ 数据集 {len(self.data)} 条 > 阈值 {self._max_pre_tokenize}，"
                    f"跳过预编码，使用惰性 tokenize（按需编码，节省内存）"
                )

    def _load_data(self, file_path: str):
        """加载并解析数据文件"""
        ext = os.path.splitext(file_path)[1].lower()
        raw_text = parse_file_to_text(file_path)

        if not raw_text.strip():
            logger.warning(f"文件解析结果为空: {os.path.basename(file_path)}")
            return

        before_count = len(self.data)
        if ext == ".jsonl":
            self._load_jsonl(raw_text)
        elif ext == ".json":
            self._load_json(raw_text)
        else:
            self._load_raw_text(raw_text)

        added = len(self.data) - before_count
        logger.info(f"数据集加载完成: {os.path.basename(file_path)} -> +{added} 条 (总计 {len(self.data)} 条)")

        if added == 0:
            logger.warning(
                f"⚠️ 文件 {os.path.basename(file_path)} 解析后未添加任何数据！"
                f"请检查文件格式是否正确。支持的格式：.jsonl / .json（数组或单个对象）/ .txt / .md / .csv"
            )

    def _load_jsonl(self, text: str):
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                formatted = self._format_item(item)
                if formatted:
                    self.data.append(formatted)
            except json.JSONDecodeError:
                continue

    def _load_json(self, text: str):
        try:
            items = json.loads(text)
            if isinstance(items, list):
                for item in items:
                    formatted = self._format_item(item)
                    if formatted:
                        self.data.append(formatted)
            elif isinstance(items, dict):
                if "instruction" in items or "output" in items or "question" in items or "prompt" in items:
                    formatted = self._format_item(items)
                    if formatted:
                        self.data.append(formatted)
                else:
                    for key, value in items.items():
                        self.data.append({
                            "instruction": str(key),
                            "output": str(value),
                        })
            else:
                logger.warning(f"JSON 数据格式不支持: {type(items).__name__}")
        except json.JSONDecodeError:
            self._load_raw_text(text)

    def _load_raw_text(self, text: str):
        """
        加载纯文本格式
        按段落拆分，自动生成指令/输出对
        支持中英文冒号、全角半角分隔符，首句切分回退
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            # 优先：中文全角冒号分割
            if "：" in para:
                parts = para.split("：", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    self.data.append({"instruction": parts[0].strip(), "output": parts[1].strip()})
                    continue
            # 其次：英文半角冒号分割
            if ":" in para:
                parts = para.split(":", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    self.data.append({"instruction": parts[0].strip(), "output": parts[1].strip()})
                    continue
            # 回退：按第一句拆分
            match = re.match(r'^([^。！？.!?\n]+)[。！？.!?]?\s*(.*)', para, re.DOTALL)
            if match and match.group(2).strip():
                self.data.append({"instruction": match.group(1).strip(), "output": match.group(2).strip()})
            else:
                self.data.append({"instruction": "请学习以下知识", "output": para})

    def _format_item(self, item: dict) -> dict:
        """标准化数据格式，返回 None 表示跳过无效数据"""
        if not isinstance(item, dict):
            logger.warning(f"跳过非字典类型数据: {type(item).__name__}")
            return None
        if "instruction" in item and "output" in item:
            return item
        if "question" in item and "answer" in item:
            return {"instruction": item["question"], "output": item["answer"]}
        if "prompt" in item and "completion" in item:
            return {"instruction": item["prompt"], "output": item["completion"]}
        if "input" in item and "output" in item:
            return {"instruction": item["input"], "output": item["output"]}
        return {"instruction": "请学习以下知识", "output": json.dumps(item, ensure_ascii=False)}

    def _pre_tokenize(self):
        """
        预编码全部样本，将 tokenization 从 DataLoader 热路径中移出。
        避免每个 batch 在 __getitem__ 中实时调用 tokenizer 阻塞训练循环，
        同时消除每个 epoch 重复 tokenize 的开销。
        """
        logger.info(f"🔧 预编码数据集 ({len(self.data)} 条, max_length={self.max_length})...")
        self._encoded_data = []
        for idx, item in enumerate(self.data):
            encoded = self._encode_item(item)
            self._encoded_data.append(encoded)
            if (idx + 1) % 500 == 0:
                logger.info(f"  预编码进度: {idx + 1}/{len(self.data)}")
        logger.info(f"✅ 预编码完成: {len(self._encoded_data)} 条样本")

    def pre_tokenize_with_progress(self, progress_callback=None):
        """
        批量预编码全部样本，支持进度回调。
        与 _pre_tokenize() 功能相同，但通过 progress_callback 逐条报告进度，
        避免一次性阻塞 GIL 导致 SSE 连接超时。

        progress_callback(fraction, current, total)  -- 每编码一条样本时调用
        """
        logger.info(f"🔧 预编码数据集 ({len(self.data)} 条, max_length={self.max_length})...")
        self._encoded_data = []
        total = len(self.data)
        for idx, item in enumerate(self.data):
            encoded = self._encode_item(item)
            self._encoded_data.append(encoded)
            if progress_callback is not None:
                progress_callback((idx + 1) / max(1, total), idx + 1, total)
            elif (idx + 1) % 500 == 0:
                logger.info(f"  预编码进度: {idx + 1}/{total}")
        logger.info(f"✅ 预编码完成: {len(self._encoded_data)} 条样本")

    def _encode_item(self, item: dict) -> dict:
        """编码单条数据，返回预计算的 tensor"""
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        if self.mode == "train":
            text = (
                f"### Instruction:\n{instruction}\n"
                f"### Response:\n{output}\n"
            )
            prefix_text = f"### Instruction:\n{instruction}\n### Response:\n"
            prefix_ids = self.tokenizer(
                prefix_text,
                max_length=self.max_length,
                truncation=True,
            )["input_ids"]
            response_start_token = len(prefix_ids)
        else:
            text = f"### Instruction:\n{instruction}\n### Response:\n"
            response_start_token = 0

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        labels = input_ids.clone()
        # 1) Mask padding 位置的 labels（attention_mask=0 处不参与 loss 计算）
        pad_mask = attention_mask == 0
        labels[pad_mask] = -100
        # 2) Mask instruction 前缀（只让模型预测 response 部分）
        if self.mode == "train" and response_start_token > 0:
            # 安全保护：确保至少保留 10% 的 token 作为 target（即 unmasked labels）
            cutoff = min(response_start_token, self.max_length)
            min_unmasked = max(1, self.max_length // 10)  # 至少保留 10%
            if self.max_length - cutoff < min_unmasked:
                cutoff = self.max_length - min_unmasked
            labels[:cutoff] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._encoded_data is not None:
            return self._encoded_data[idx]
        # 惰性编码：按需 tokenize（大数据集降级路径）
        # 使用简单缓存避免同一 idx 在 shuffle 后被重复编码
        if not hasattr(self, '_lazy_cache'):
            self._lazy_cache = {}
        if idx not in self._lazy_cache:
            # 缓存容量保护：防止缓存无限增长（最多缓存 2048 条）
            if len(self._lazy_cache) > 2048:
                self._lazy_cache.clear()
            self._lazy_cache[idx] = self._encode_item(self.data[idx])
        return self._lazy_cache[idx]


def _unwrap_dataset(dataset: Dataset) -> Dataset:
    """
    穿透 Subset / ConcatDataset 等包装，获取最底层的原始 Dataset。
    用于检查原始 dataset 上的自定义属性（如 _encoded_data）。
    
    仅在 torch.utils.data.Subset/ConcatDataset 时递归穿透，
    其他类型直接返回原 dataset。
    """
    from torch.utils.data import Subset, ConcatDataset
    
    while True:
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        elif isinstance(dataset, ConcatDataset):
            # ConcatDataset 内含多个子数据集，取第一个（假设同构）
            if len(dataset.datasets) > 0:
                dataset = dataset.datasets[0]
            else:
                break
        else:
            break
    return dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    创建 DataLoader（自动优化性能）

    - 自动设置合理的 num_workers（数据已预编码，安全）
    - 自动开启 pin_memory 加速 GPU 传输
    - Windows 下强制 num_workers=0 防止 multiprocessing 子进程弹窗
    """
    if len(dataset) == 0:
        logger.error("⚠️ 数据集为空，无法创建 DataLoader！请检查数据文件格式。")
        raise ValueError("数据集为空，无法开始训练。请确保上传了正确的 .jsonl / .json / .txt 格式数据文件。")
    effective_batch = min(batch_size, len(dataset))
    if effective_batch < batch_size:
        logger.warning(f"Batch size ({batch_size}) 大于数据集大小 ({len(dataset)})，自动调整为 {effective_batch}")

    # 自动设置 num_workers（数据已预编码，子进程只做 tensor 搬运，非常安全）
    if num_workers == 0:
        # 穿透 Subset/ConcatDataset 检查底层是否有 _encoded_data
        raw_dataset = _unwrap_dataset(dataset)
        if getattr(raw_dataset, '_encoded_data', None) is not None:
            # 数据已经在内存中预编码，开启多进程反而会带来昂贵的序列化/IPC开销，导致 GPU 饥饿
            num_workers = 0
            logger.info("DataLoader: 数据已全部预编码，禁用多进程(num_workers=0)以消除 IPC 瓶颈")
        else:
            suggested = max(1, min(4, (os.cpu_count() or 2) // 2))
            num_workers = min(2, suggested) if os.name == 'nt' else suggested
            logger.info(f"DataLoader: 自动 num_workers={num_workers}, pin_memory=True")

    # ══════════════════════════════════════════════════════════════════
    # Windows 安全保护：在 PyQt6 QWebEngine 嵌入式环境中，任何
    # num_workers > 0 都会通过 multiprocessing.spawn 创建子进程，
    # 这些子进程不受 subprocess 补丁保护，会弹出独立的控制台窗口。
    # 且预编码数据已全在内存中，多进程搬运只有 IPC 开销无收益。
    # ══════════════════════════════════════════════════════════════════
    if os.name == 'nt' and num_workers > 0:
        logger.warning(
            f"⚠️ Windows 嵌入式环境：强制 num_workers={num_workers}→0，"
            f"防止 multiprocessing 子进程弹出控制台窗口"
        )
        num_workers = 0

    # pin_memory 加速 GPU 传输（CPU 无副作用）
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    return DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=shuffle and effective_batch > 1,
        num_workers=num_workers,
        pin_memory=has_cuda,
        drop_last=drop_last and effective_batch > 1,
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.9):
    """将数据集分割为训练集和验证集"""
    if len(dataset) < 2:
        logger.warning("数据集过小（<2），无法分割验证集，跳过验证")
        return dataset, None
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    return random_split(dataset, [train_size, val_size])


def create_merged_dataset(file_paths: List[str], tokenizer, max_length: int = 512,
                          mode: str = "train") -> InstructionDataset:
    """将多个文件合并为单个数据集"""
    logger.info(f"合并 {len(file_paths)} 个文件为单一数据集...")
    return InstructionDataset(file_paths, tokenizer, max_length, mode)
