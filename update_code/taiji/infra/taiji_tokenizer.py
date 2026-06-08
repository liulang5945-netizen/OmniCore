"""
态极分词器 v2 — 基于 HuggingFace tokenizer

包装 Qwen2.5 等标准 HF tokenizer，同时管理态极特殊 token。
与 ModelSelfTokenizer 接口完全兼容，可无缝替换。

设计理念：
  - 底层用 HF tokenizer（151936 词表，BPE，多语言能力强）
  - 特殊 token 注入到 HF 词表末尾，不干扰原有词表
  - 工具名 token 动态注册（与 ModelSelfTokenizer 相同机制）
  - __call__ / decode / register_tool 接口完全对齐
"""
import json
import logging
import os
from typing import Optional, Dict, List, Any

import torch

from taiji.config import SPECIAL_TOKENS

logger = logging.getLogger("ModelSelf.TaijiTokenizer")


# 态极特殊 token 文本 → config.py 中定义的语义名称
SPECIAL_TOKEN_TEXTS = {
    "<think>": "think_start",
    "</think>": "think_end",
    "<tool_call>": "tool_call",
    "<tool_result>": "tool_result",
    "<final_answer>": "answer",
    # 感知
    "<observe>": "observe",
    "</observe>": "observe_end",
    # 记忆
    "<mem_read>": "mem_read",
    "<mem_write>": "mem_write",
    # 规划
    "<plan>": "plan_start",
    "</plan>": "plan_end",
    "<step>": "plan_step",
    "</step>": "plan_step_end",
    "<plan_done>": "plan_done",
    "<replan>": "plan_replan",
    # 反思
    "<reflect>": "reflect_start",
    "</reflect>": "reflect_end",
    "<detect>": "reflect_detect",
    "<cause>": "reflect_cause",
    "<correct>": "reflect_correct",
    "<confirm>": "reflect_confirm",
}


class TaijiTokenizer:
    """
    态极分词器 — 基于 HuggingFace tokenizer

    接口与 ModelSelfTokenizer 完全兼容：
    - __call__(text, return_tensors="pt") → {"input_ids": Tensor, "attention_mask": Tensor}
    - decode(ids, skip_special_tokens=True) → str
    - register_tool(tool_name) → int
    - get_tool_id / get_tool_name / get_all_tool_ids
    - save / load
    """

    def __init__(self, hf_tokenizer=None, hf_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Args:
            hf_tokenizer: 已加载的 HF tokenizer 实例（优先）
            hf_model_name: HF 模型名（hf_tokenizer 为 None 时从这里加载）
        """
        if hf_tokenizer is not None:
            self.hf = hf_tokenizer
        else:
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

        # 确保有 pad_token
        if self.hf.pad_token is None:
            self.hf.pad_token = self.hf.eos_token

        # 特殊 token 映射（文本 ↔ ID）
        self._special_text_to_id: Dict[str, int] = {}
        self._special_id_to_text: Dict[int, str] = {}
        self._register_special_tokens()

        # 工具名 token 映射
        self._tool_name_to_id: Dict[str, int] = {}
        self._id_to_tool_name: Dict[int, str] = {}

        logger.info(
            f"TaijiTokenizer 初始化: {hf_model_name}, "
            f"vocab={self.hf.vocab_size}, special_tokens={len(self._special_text_to_id)}"
        )

    def _register_special_tokens(self):
        """将态极特殊 token 注入 HF 词表"""
        # 收集所有需要添加的特殊 token
        new_tokens = []
        for token_text in SPECIAL_TOKEN_TEXTS:
            if token_text not in self.hf.get_vocab():
                new_tokens.append(token_text)

        if new_tokens:
            # 添加到词表末尾
            num_added = self.hf.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info(f"注入 {num_added} 个态极特殊 token 到词表")

        # 建立映射
        for token_text in SPECIAL_TOKEN_TEXTS:
            token_id = self.hf.convert_tokens_to_ids(token_text)
            if token_id != self.hf.unk_token_id:
                self._special_text_to_id[token_text] = token_id
                self._special_id_to_text[token_id] = token_text

    # ── 属性（兼容 ModelSelfTokenizer）──

    @property
    def pad_token_id(self) -> int:
        return self.hf.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.hf.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self.hf.bos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.hf)

    # ── 工具注册（与 ModelSelfTokenizer 相同）──

    def register_tool(self, tool_name: str) -> int:
        """注册工具名，分配 token ID"""
        if tool_name in self._tool_name_to_id:
            return self._tool_name_to_id[tool_name]

        # 添加到 HF 词表
        if tool_name not in self.hf.get_vocab():
            self.hf.add_tokens([tool_name], special_tokens=False)

        tool_id = self.hf.convert_tokens_to_ids(tool_name)
        self._tool_name_to_id[tool_name] = tool_id
        self._id_to_tool_name[tool_id] = tool_name

        logger.debug(f"Registered tool: {tool_name} -> {tool_id}")
        return tool_id

    def get_tool_id(self, tool_name: str) -> Optional[int]:
        return self._tool_name_to_id.get(tool_name)

    def get_tool_name(self, tool_id: int) -> Optional[str]:
        return self._id_to_tool_name.get(tool_id)

    def get_all_tool_ids(self) -> Dict[str, int]:
        return dict(self._tool_name_to_id)

    # ── 编码（兼容 HF 接口）──

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """编码文本（接口与 ModelSelfTokenizer / HF tokenizer 一致）"""
        enc = self.hf(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return enc

    def encode(self, text: str) -> List[int]:
        """编码文本为 token ID 列表"""
        return self.hf.encode(text, add_special_tokens=False)

    def _encode(self, text: str) -> List[int]:
        """兼容 ModelSelfTokenizer 的 _encode 接口"""
        return self.encode(text)

    # ── 解码（兼容 HF 接口）──

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """解码 token IDs 为文本"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.hf.decode(ids, skip_special_tokens=skip_special_tokens)

    # ── 持久化 ──

    def save(self, path: str):
        """保存分词器配置（HF tokenizer + 态极工具映射）"""
        os.makedirs(path, exist_ok=True)
        # 保存 HF tokenizer
        self.hf.save_pretrained(path)
        # 保存态极扩展配置
        taiji_config = {
            "tool_mappings": self._tool_name_to_id,
            "special_tokens": self._special_text_to_id,
        }
        with open(os.path.join(path, "taiji_tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(taiji_config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TaijiTokenizer":
        """加载分词器"""
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokenizer = cls(hf_tokenizer=hf)

        # 恢复态极扩展配置
        taiji_path = os.path.join(path, "taiji_tokenizer.json")
        if os.path.exists(taiji_path):
            with open(taiji_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for name, tid in config.get("tool_mappings", {}).items():
                tokenizer._tool_name_to_id[name] = tid
                tokenizer._id_to_tool_name[tid] = name

        return tokenizer
