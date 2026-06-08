"""
ModelSelf 分词器

复用 SentencePiece 基础词表 (32000)，增加特殊 token:
- 结构标记: <think></think><tool_call><tool_result><final_answer>
- 工具名: 动态注册，ID 从 32010 开始
"""
import os
import json
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch

from .config import SPECIAL_TOKENS

logger = logging.getLogger("ModelSelf")


class ModelSelfTokenizer:
    """
    ModelSelf 分词器

    底层使用 SentencePiece，上层管理特殊 token。
    兼容 HuggingFace tokenizer 接口，可直接用于 BaseInferenceEngine。
    """

    def __init__(self, sp_model_path: Optional[str] = None):
        """
        Args:
            sp_model_path: SentencePiece 模型文件路径 (.model)
                           如果为 None，使用字符级回退分词器
        """
        self.sp = None
        self._tool_name_to_id: Dict[str, int] = {}
        self._id_to_tool_name: Dict[int, str] = {}
        self._next_tool_id = SPECIAL_TOKENS["tool_name_base"]

        # 尝试加载 SentencePiece
        if sp_model_path and os.path.exists(sp_model_path):
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(sp_model_path)
                logger.info(f"Loaded SentencePiece model: {sp_model_path}")
            except ImportError:
                logger.warning("sentencepiece not installed, using char-level fallback")
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece: {e}")

        if self.sp is None:
            logger.info("Using character-level fallback tokenizer")

        # 特殊 token 文本 → ID 映射
        self._special_text_to_id = {
            "<think>": SPECIAL_TOKENS["think_start"],
            "</think>": SPECIAL_TOKENS["think_end"],
            "<tool_call>": SPECIAL_TOKENS["tool_call"],
            "<tool_result>": SPECIAL_TOKENS["tool_result"],
            "<final_answer>": SPECIAL_TOKENS["answer"],
        }
        self._special_id_to_text = {v: k for k, v in self._special_text_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        # 返回模型配置的词表大小，确保 tokenizer 和模型一致
        # SP 基础词表 32000 + 特殊 token 区域（固定 1000 个）
        return 33000

    def register_tool(self, tool_name: str) -> int:
        """
        注册一个工具名，分配特殊 token ID。

        Args:
            tool_name: 工具名称 (如 "search", "read_file")

        Returns:
            分配的 token ID
        """
        if tool_name in self._tool_name_to_id:
            return self._tool_name_to_id[tool_name]

        tool_id = self._next_tool_id
        self._next_tool_id += 1
        self._tool_name_to_id[tool_name] = tool_id
        self._id_to_tool_name[tool_id] = tool_name

        # 同时注册到特殊 token 映射中，这样编码时能识别
        self._special_text_to_id[tool_name] = tool_id
        self._special_id_to_text[tool_id] = tool_name

        logger.info(f"Registered tool token: {tool_name} -> {tool_id}")
        return tool_id

    def get_tool_id(self, tool_name: str) -> Optional[int]:
        """获取工具的 token ID"""
        return self._tool_name_to_id.get(tool_name)

    def get_tool_name(self, tool_id: int) -> Optional[str]:
        """从 token ID 获取工具名"""
        return self._id_to_tool_name.get(tool_id)

    def get_all_tool_ids(self) -> Dict[str, int]:
        """获取所有工具映射"""
        return dict(self._tool_name_to_id)

    def encode(self, text: str) -> List[int]:
        """
        编码文本为 token IDs（公开接口）

        Args:
            text: 输入文本

        Returns:
            token IDs 列表
        """
        return self._encode(text)

    def __call__(
        self,
        text,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        编码文本 (兼容 HuggingFace 接口，支持单条和批量)

        Args:
            text: 输入文本（str 或 List[str]）
            return_tensors: "pt" 返回 PyTorch 张量
            padding: 是否填充
            truncation: 是否截断
            max_length: 最大长度

        Returns:
            {"input_ids": Tensor, "attention_mask": Tensor}
        """
        # 支持批量输入
        if isinstance(text, list):
            batch_ids = []
            for t in text:
                ids = self._encode(t)
                if truncation and max_length:
                    ids = ids[:max_length]
                batch_ids.append(ids)
            # padding 到最大长度
            if padding:
                max_len = max_length or max(len(ids) for ids in batch_ids)
                batch_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in batch_ids]
            input_ids = torch.tensor(batch_ids, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        ids = self._encode(text)

        if truncation and max_length:
            ids = ids[:max_length]

        if padding and max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _encode(self, text: str) -> List[int]:
        """编码文本为 token IDs"""
        ids = []

        # 先处理特殊 token
        remaining = text
        while remaining:
            # 找到最近的特殊 token
            earliest_pos = len(remaining)
            earliest_token = None

            for special_text in self._special_text_to_id:
                pos = remaining.find(special_text)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special_text

            if earliest_token is None:
                # 没有更多特殊 token，编码剩余文本
                ids.extend(self._encode_text(remaining))
                break

            # 编码特殊 token 之前的文本
            if earliest_pos > 0:
                ids.extend(self._encode_text(remaining[:earliest_pos]))

            # 添加特殊 token ID
            ids.append(self._special_text_to_id[earliest_token])
            remaining = remaining[earliest_pos + len(earliest_token):]

        return ids

    def _encode_text(self, text: str) -> List[int]:
        """编码普通文本"""
        if not text:
            return []

        if self.sp is not None:
            return self.sp.EncodeAsIds(text)
        else:
            # 字符级回退: 使用 UTF-8 字节编码，确保可逆性
            # 使用 ID 范围 [3, 260] 共 258 个位置映射 256 个字节值
            # 这样任何 Unicode 字符都能通过 UTF-8 字节流编码/解码
            ids = []
            for byte_val in text.encode("utf-8"):
                ids.append(byte_val + 3)  # [3, 258]
            return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """
        解码 token IDs 为文本

        Args:
            ids: token IDs (Tensor 或 list)
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, int):
            ids = [ids]

        text_parts = []
        sp_ids = []  # 收集 SentencePiece 范围内的 ID

        for id in ids:
            # 特殊 token
            if id in self._special_id_to_text:
                # 先 flush 收集的 SP IDs
                if sp_ids and self.sp is not None:
                    try:
                        text_parts.append(self.sp.DecodeIds(sp_ids))
                    except Exception:
                        for sid in sp_ids:
                            text_parts.append(f"[{sid}]")
                    sp_ids = []
                if not skip_special_tokens:
                    text_parts.append(self._special_id_to_text[id])
                continue

            # 工具名 token
            if id in self._id_to_tool_name:
                # 先 flush 收集的 SP IDs
                if sp_ids and self.sp is not None:
                    try:
                        text_parts.append(self.sp.DecodeIds(sp_ids))
                    except Exception:
                        for sid in sp_ids:
                            text_parts.append(f"[{sid}]")
                    sp_ids = []
                if not skip_special_tokens:
                    text_parts.append(f"[tool:{self._id_to_tool_name[id]}]")
                continue

            # 普通 token - 检查是否在 SP 范围内
            if self.sp is not None and id < self.sp.GetPieceSize():
                sp_ids.append(id)
            elif self.sp is not None:
                # 超出 SP 范围，flush 后跳过
                if sp_ids:
                    try:
                        text_parts.append(self.sp.DecodeIds(sp_ids))
                    except Exception:
                        for sid in sp_ids:
                            text_parts.append(f"[{sid}]")
                    sp_ids = []
                # 跳过超出范围的 token
                continue
            else:
                # 无 SP 回退: UTF-8 字节解码
                if 3 <= id <= 258:
                    sp_ids.append(id)  # 收集到 byte_ids 中
                # 超出范围的跳过

        # flush 剩余的 SP IDs / byte IDs
        if sp_ids:
            if self.sp is not None:
                try:
                    text_parts.append(self.sp.DecodeIds(sp_ids))
                except Exception:
                    for sid in sp_ids:
                        text_parts.append(f"[{sid}]")
            else:
                # 无 SP 回退模式: 将收集的 byte_ids 还原为 UTF-8 文本
                try:
                    byte_vals = [id - 3 for id in sp_ids if 0 <= id - 3 <= 255]
                    text_parts.append(bytes(byte_vals).decode("utf-8", errors="replace"))
                except Exception:
                    for sid in sp_ids:
                        text_parts.append(f"[{sid}]")

        return "".join(text_parts)

    def save(self, path: str):
        """保存分词器配置"""
        os.makedirs(path, exist_ok=True)
        config = {
            "tool_mappings": self._tool_name_to_id,
            "next_tool_id": self._next_tool_id,
            "special_tokens": self._special_text_to_id,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, sp_model_path: Optional[str] = None) -> "ModelSelfTokenizer":
        """加载分词器"""
        tokenizer = cls(sp_model_path)

        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 恢复工具映射
            for name, tid in config.get("tool_mappings", {}).items():
                tokenizer._tool_name_to_id[name] = tid
                tokenizer._id_to_tool_name[tid] = name
            tokenizer._next_tool_id = config.get("next_tool_id", SPECIAL_TOKENS["tool_name_base"])

        return tokenizer
