"""
数据集质量检查器
================
对训练数据集进行全面的质量检查，包括格式检测、字段验证、统计分析和异常检测。
支持 JSONL / CSV / Alpaca / ShareGPT / Vicuna 格式。
"""
import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger("DatasetChecker")


class DatasetQualityChecker:
    """数据集质量检查器"""

    MAX_PREVIEW_LINES = 1000  # 最多预览行数

    def check(self, file_path: str) -> dict:
        """
        全面检查数据集质量

        Returns:
            {
                "valid": bool,
                "format": str,
                "total_samples": int,
                "fields": list,
                "stats": {...},
                "warnings": list,
                "errors": list,
            }
        """
        if not os.path.exists(file_path):
            return {"valid": False, "errors": ["文件不存在"], "warnings": []}

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".jsonl" or ext == ".json":
                return self._check_jsonl(file_path)
            elif ext == ".csv":
                return self._check_csv(file_path)
            else:
                return self._check_jsonl(file_path)  # 尝试 JSONL
        except Exception as e:
            return {"valid": False, "errors": [f"解析失败: {str(e)}"], "warnings": []}

    def _check_jsonl(self, file_path: str) -> dict:
        """检查 JSONL / JSON 格式数据集"""
        samples = []
        errors = []
        warnings = []
        line_errors = 0

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 尝试 JSON 数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "data" in data:
                samples = data["data"]
        except json.JSONDecodeError:
            pass

        # 尝试 JSONL（每行一个 JSON）
        if not samples:
            for i, line in enumerate(content.strip().split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    samples.append(obj)
                except json.JSONDecodeError:
                    line_errors += 1
                    if line_errors <= 3:
                        warnings.append(f"第 {i} 行 JSON 解析失败")

        if not samples:
            return {"valid": False, "errors": ["无法解析数据集内容"], "warnings": warnings}

        # 分析样本
        return self._analyze_samples(samples, warnings)

    def _check_csv(self, file_path: str) -> dict:
        """检查 CSV 格式数据集"""
        import csv
        samples = []
        warnings = []

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(dict(row))

        if not samples:
            return {"valid": False, "errors": ["CSV 文件为空"], "warnings": []}

        return self._analyze_samples(samples, warnings)

    def _analyze_samples(self, samples: list, warnings: list) -> dict:
        """分析样本质量"""
        total = len(samples)
        fields = list(samples[0].keys()) if samples else []

        # 检测数据集格式
        fmt = "unknown"
        if any(k in fields for k in ("instruction", "output")):
            fmt = "alpaca"
        elif any(k in fields for k in ("conversations", "chat")):
            fmt = "sharegpt"
        elif "messages" in fields:
            fmt = "openai_chat"
        elif any(k in fields for k in ("input", "output")):
            fmt = "input_output"
        elif "text" in fields:
            fmt = "text_only"

        # 字段统计
        field_counts = {f: sum(1 for s in samples if f in s and s[f]) for f in fields}

        # 文本长度统计
        input_lengths = []
        output_lengths = []
        empty_count = 0
        duplicate_check = set()
        duplicate_count = 0
        oversized = 0
        MAX_LEN = 4096

        for i, sample in enumerate(samples):
            # 提取输入文本
            text = ""
            for key in ("instruction", "input", "prompt", "question", "text"):
                if key in sample and sample[key]:
                    text = str(sample[key])
                    break
            if not text and "messages" in sample:
                msgs = sample["messages"]
                if isinstance(msgs, list) and msgs:
                    text = str(msgs[0].get("content", ""))

            # 提取输出文本
            out_text = ""
            for key in ("output", "response", "answer", "completion"):
                if key in sample and sample[key]:
                    out_text = str(sample[key])
                    break

            if text:
                input_lengths.append(len(text))
            if out_text:
                output_lengths.append(len(out_text))

            # 空样本检测
            if not text and not out_text:
                empty_count += 1

            # 重复检测
            hash_key = hash(text[:200] if text else "")
            if hash_key in duplicate_check:
                duplicate_count += 1
            duplicate_check.add(hash_key)

            # 超长检测
            total_len = len(text) + len(out_text)
            if total_len > MAX_LEN:
                oversized += 1

        # 生成警告
        if empty_count > 0:
            warnings.append(f"发现 {empty_count} 条空白样本")
        if duplicate_count > 0:
            warnings.append(f"发现 {duplicate_count} 条重复样本（前200字符相同）")
        if oversized > 0:
            warnings.append(f"发现 {oversized} 条超长样本（>{MAX_LEN} 字符）")
        if total < 10:
            warnings.append("数据集样本过少（<10），可能导致过拟合")

        def _percentile(data, p):
            if not data:
                return 0
            data_sorted = sorted(data)
            idx = int(len(data_sorted) * p / 100)
            return data_sorted[min(idx, len(data_sorted) - 1)]

        return {
            "valid": total > 0,
            "format": fmt,
            "total_samples": total,
            "fields": fields,
            "field_coverage": field_counts,
            "stats": {
                "input_length": {
                    "avg": round(sum(input_lengths) / max(len(input_lengths), 1)),
                    "min": min(input_lengths) if input_lengths else 0,
                    "max": max(input_lengths) if input_lengths else 0,
                    "p50": _percentile(input_lengths, 50),
                    "p90": _percentile(input_lengths, 90),
                    "count": len(input_lengths),
                },
                "output_length": {
                    "avg": round(sum(output_lengths) / max(len(output_lengths), 1)),
                    "min": min(output_lengths) if output_lengths else 0,
                    "max": max(output_lengths) if output_lengths else 0,
                    "p50": _percentile(output_lengths, 50),
                    "p90": _percentile(output_lengths, 90),
                    "count": len(output_lengths),
                },
                "empty_samples": empty_count,
                "duplicate_samples": duplicate_count,
                "oversized_samples": oversized,
            },
            "warnings": warnings,
            "errors": [],
        }