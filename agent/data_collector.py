"""
训练数据收集器

从 OmniCore 的实际使用中收集训练数据:
- ReAct 引擎的 AgentStep 日志
- 对话历史
- 工具调用记录

收集的数据用于训练 ModelSelf 模型。
"""
import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque

from core.utils import get_external_path

logger = logging.getLogger("DataCollector")


class DataCollector:
    """
    训练数据收集器

    单例模式，在后台收集 OmniCore 的使用数据。
    不影响正常运行，数据异步写入磁盘。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.data_dir = get_external_path("training_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # 内存缓冲区
        self._buffer: deque = deque(maxlen=10000)
        self._write_lock = threading.Lock()

        # 统计
        self.stats = {
            "react_steps": 0,
            "conversations": 0,
            "tool_calls": 0,
            "writes": 0,
        }

        logger.info(f"DataCollector initialized, data dir: {self.data_dir}")

    def collect_react_step(
        self,
        task: str,
        thought: str,
        action: Optional[str],
        action_args: Optional[dict],
        observation: Optional[str],
        is_final: bool,
        final_answer: Optional[str] = None,
    ):
        """
        收集一个 ReAct 步骤。

        Args:
            task: 用户任务
            thought: 模型思考
            action: 工具名 (最终回答时为 None)
            action_args: 工具参数
            observation: 工具返回结果
            is_final: 是否为最终步骤
            final_answer: 最终回答
        """
        entry = {
            "type": "react_step",
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "thought": thought,
            "action": action,
            "action_args": action_args,
            "observation": observation,
            "is_final": is_final,
            "final_answer": final_answer,
        }
        self._buffer.append(entry)
        self.stats["react_steps"] += 1
        if action:
            self.stats["tool_calls"] += 1

    def collect_conversation(
        self,
        user_msg: str,
        assistant_msg: str,
        system_prompt: Optional[str] = None,
    ):
        """
        收集一条对话。

        Args:
            user_msg: 用户消息
            assistant_msg: 助手回复
            system_prompt: 系统提示
        """
        entry = {
            "type": "conversation",
            "timestamp": datetime.now().isoformat(),
            "messages": []
        }
        if system_prompt:
            entry["messages"].append({"role": "system", "content": system_prompt})
        entry["messages"].append({"role": "user", "content": user_msg})
        entry["messages"].append({"role": "assistant", "content": assistant_msg})

        self._buffer.append(entry)
        self.stats["conversations"] += 1

    def flush(self):
        """将缓冲区数据写入磁盘"""
        if not self._buffer:
            return

        with self._write_lock:
            entries = list(self._buffer)
            self._buffer.clear()

        # 按日期分文件
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(self.data_dir, f"data_{date_str}.jsonl")

        with open(filepath, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.stats["writes"] += len(entries)
        logger.info(f"Flushed {len(entries)} entries to {filepath}")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "buffer_size": len(self._buffer),
            "data_dir": self.data_dir,
        }

    def load_all_data(self) -> List[dict]:
        """加载所有收集的数据"""
        all_data = []
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_data.append(json.loads(line))
        return all_data

    def load_as_training_data(self) -> tuple:
        """
        加载收集的数据并聚合为 ReActDataset 可直接使用的格式。

        将散落的 react_step 条目按 task 聚合为完整的任务步骤序列。
        将 conversation 条目直接作为对话数据。

        Returns:
            (react_data: List[dict], conv_data: List[dict])
            react_data 格式: [{"task": "...", "steps": [{"thought": "...", "action": "...", ...}, ...]}]
            conv_data 格式: [{"messages": [...]}]
        """
        raw = self.load_all_data()

        # 按 task 分组聚合 ReAct 步骤
        react_tasks: Dict[str, List[dict]] = {}
        conv_data = []

        for entry in raw:
            if entry.get("type") == "react_step":
                task = entry.get("task", "")
                if task not in react_tasks:
                    react_tasks[task] = []
                react_tasks[task].append({
                    "thought": entry.get("thought", ""),
                    "action": entry.get("action"),
                    "action_args": entry.get("action_args", {}),
                    "observation": entry.get("observation", ""),
                    "final_answer": entry.get("final_answer"),
                })
            elif entry.get("type") == "conversation":
                conv_data.append({"messages": entry.get("messages", [])})

        # 转换为 ReActDataset 期望的格式
        react_data = []
        for task, steps in react_tasks.items():
            react_data.append({"task": task, "steps": steps})

        logger.info(f"Loaded training data: {len(react_data)} react tasks, {len(conv_data)} conversations")
        return react_data, conv_data


def get_collector() -> DataCollector:
    """获取全局数据收集器实例"""
    return DataCollector()