"""
三层记忆系统 v2 (Memory Manager)
=================================
短期记忆 - 对话上下文（最近 N 条消息）
工作记忆 - 当前任务状态（KV 存储）
长期记忆 - 持久化存储（语义检索）

v2 新增：
情景记忆 - 任务执行历史的结构化记录
语义记忆 - 向量化长期知识（复用 RAG embedding）
记忆压缩器 - LLM 摘要 + 重要度评分
"""
import json
import logging
import math
import os
import time
import threading
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MemoryManager")


# ======================== 短期记忆 ========================

class ShortTermMemory:
    """短期记忆：对话上下文"""

    def __init__(self, max_messages: int = 50):
        self._messages: deque = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str, metadata: dict = None):
        self._messages.append({
            "role": role,
            "content": content[:2000],
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def get_context(self, last_n: int = 20) -> list:
        messages = list(self._messages)
        return messages[-last_n:] if len(messages) > last_n else messages

    def get_formatted_context(self, last_n: int = 10) -> str:
        messages = self.get_context(last_n)
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prefix = "用户" if role == "user" else ("助手" if role == "assistant" else role)
            lines.append(f"{prefix}: {content[:200]}")
        return "\n".join(lines)

    def clear(self):
        self._messages.clear()

    def count(self) -> int:
        return len(self._messages)


# ======================== 工作记忆 ========================

class WorkingMemory:
    """工作记忆：当前任务状态（KV 存储）"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def get_all(self) -> dict:
        return dict(self._data)

    def list_keys(self) -> list:
        return list(self._data.keys())

    def clear(self):
        self._data.clear()


# ======================== 长期记忆 ========================

class LongTermMemory:
    """长期记忆：持久化存储"""

    def __init__(self, storage_path: str = ""):
        self._entries: List[dict] = []
        self._storage_path = storage_path
        self._load()

    def _load(self):
        try:
            if self._storage_path and os.path.exists(self._storage_path):
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
                logger.info(f"已加载 {len(self._entries)} 条长期记忆")
        except Exception as e:
            logger.warning(f"加载长期记忆失败: {e}")
            self._entries = []

    def _save(self):
        try:
            if self._storage_path:
                os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
                with open(self._storage_path, "w", encoding="utf-8") as f:
                    json.dump({"entries": self._entries[-500:]}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存长期记忆失败: {e}")

    def store(self, text: str, category: str = "general", metadata: dict = None):
        entry = {
            "text": text[:1000],
            "category": category,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._entries.append(entry)
        self._save()

    def search(self, query: str, top_k: int = 5, category: str = None) -> list:
        if not self._entries:
            return []
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            if category and entry.get("category", "") != category:
                continue
            text = entry.get("text", "").lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def list_entries(self, category: str = None, limit: int = 50) -> list:
        entries = self._entries
        if category:
            entries = [e for e in entries if e.get("category", "") == category]
        return entries[-limit:]

    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self._save()


# ======================== 情景记忆 (v2 新增) ========================

@dataclass
class Episode:
    """情景记录"""
    task: str
    steps: int = 0
    result_summary: str = ""
    tools_used: List[str] = field(default_factory=list)
    duration_ms: float = 0
    success: bool = False
    timestamp: float = 0
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)


class EpisodicMemory:
    """情景记忆：记录任务执行历史的结构化摘要"""

    def __init__(self, storage_path: str = ""):
        self._episodes: List[dict] = []
        self._storage_path = storage_path
        self._load()

    def _load(self):
        try:
            if self._storage_path and os.path.exists(self._storage_path):
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._episodes = data.get("episodes", [])
                logger.info(f"已加载 {len(self._episodes)} 条情景记忆")
        except Exception as e:
            logger.warning(f"加载情景记忆失败: {e}")
            self._episodes = []

    def _save(self):
        try:
            if self._storage_path:
                os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
                with open(self._storage_path, "w", encoding="utf-8") as f:
                    json.dump({"episodes": self._episodes[-200:]}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存情景记忆失败: {e}")

    def add_episode(self, episode: Episode):
        entry = asdict(episode)
        entry["timestamp"] = entry.get("timestamp") or time.time()
        self._episodes.append(entry)
        self._save()
        logger.debug(f"情景记忆已记录: {episode.task[:50]}...")

    def search_by_text(self, query: str, top_k: int = 5) -> list:
        """关键词搜索情景"""
        if not self._episodes:
            return []
        query_lower = query.lower()
        scored = []
        for ep in self._episodes:
            text = (ep.get("task", "") + " " + ep.get("result_summary", "")).lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_recent(self, limit: int = 10, success_only: bool = False) -> list:
        episodes = self._episodes
        if success_only:
            episodes = [e for e in episodes if e.get("success")]
        return episodes[-limit:]

    def get_by_tool(self, tool_name: str, limit: int = 10) -> list:
        return [e for e in self._episodes if tool_name in e.get("tools_used", [])][-limit:]

    def get_stats(self) -> dict:
        if not self._episodes:
            return {"total": 0, "success_rate": 0, "top_tools": []}
        total = len(self._episodes)
        success = sum(1 for e in self._episodes if e.get("success"))
        tool_counter = Counter()
        for ep in self._episodes:
            for t in ep.get("tools_used", []):
                tool_counter[t] += 1
        return {
            "total": total,
            "success_rate": round(success / max(total, 1), 2),
            "top_tools": tool_counter.most_common(5),
            "avg_duration_ms": round(sum(e.get("duration_ms", 0) for e in self._episodes) / total, 1),
        }

    def count(self) -> int:
        return len(self._episodes)

    def clear(self):
        self._episodes.clear()
        self._save()


# ======================== 语义记忆 (v2 新增) ========================

class SemanticMemory:
    """语义记忆：向量化长期知识存储（复用 RAG embedding 能力）"""

    def __init__(self, storage_path: str = ""):
        self._entries: List[dict] = []
        self._embeddings = None
        self._storage_path = storage_path
        self._embedder = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            if self._storage_path and os.path.exists(self._storage_path):
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
                # 加载 embeddings
                emb_path = self._storage_path.replace(".json", "_embeddings.npy")
                if os.path.exists(emb_path):
                    import numpy as np
                    self._embeddings = np.load(emb_path)
                logger.info(f"已加载 {len(self._entries)} 条语义记忆")
        except Exception as e:
            logger.warning(f"加载语义记忆失败: {e}")
            self._entries = []

    def _save(self):
        try:
            if self._storage_path:
                os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
                # 只保留最近 500 条
                entries_to_save = self._entries[-500:]
                with open(self._storage_path, "w", encoding="utf-8") as f:
                    json.dump({"entries": entries_to_save}, f, indent=2, ensure_ascii=False)
                # 保存 embeddings
                if self._embeddings is not None:
                    import numpy as np
                    emb_path = self._storage_path.replace(".json", "_embeddings.npy")
                    if len(entries_to_save) == self._embeddings.shape[0]:
                        np.save(emb_path, self._embeddings[-len(entries_to_save):])
        except Exception as e:
            logger.warning(f"保存语义记忆失败: {e}")

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = os.environ.get("OMNICORE_EMBEDDING_MODEL",
                                            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                self._embedder = SentenceTransformer(model_name, device="cpu")
            except Exception as e:
                logger.warning(f"语义记忆 embedding 模型加载失败: {e}")
        return self._embedder

    def store(self, text: str, category: str = "general", importance: float = 0.5):
        """存储并索引一条记忆"""
        entry = {
            "text": text[:500],
            "category": category,
            "importance": importance,
            "timestamp": time.time(),
        }
        self._entries.append(entry)

        # 生成 embedding
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                import numpy as np
                vec = embedder.encode([text[:500]], normalize_embeddings=True)
                if self._embeddings is not None:
                    self._embeddings = np.vstack([self._embeddings, vec])
                else:
                    self._embeddings = vec
            except Exception as e:
                logger.debug(f"语义记忆 embedding 生成失败: {e}")

        self._save()

    def search(self, query: str, top_k: int = 5, min_importance: float = 0.0) -> list:
        """语义搜索记忆"""
        if not self._entries:
            return []

        # 尝试向量检索
        if self._embeddings is not None and len(self._embeddings) == len(self._entries):
            try:
                import numpy as np
                embedder = self._get_embedder()
                if embedder is not None:
                    q_vec = embedder.encode([query], normalize_embeddings=True)
                    scores = np.dot(self._embeddings, q_vec[0])
                    # 按重要度过滤
                    indices = np.argsort(scores)[::-1]
                    results = []
                    for idx in indices:
                        if len(results) >= top_k:
                            break
                        entry = self._entries[idx]
                        if entry.get("importance", 0) >= min_importance and scores[idx] > 0.05:
                            results.append({**entry, "score": float(scores[idx])})
                    return results
            except Exception as e:
                logger.debug(f"语义检索失败，回退到关键词: {e}")

        # 回退：关键词匹配
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            if entry.get("importance", 0) < min_importance:
                continue
            text = entry.get("text", "").lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append({**entry, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self._embeddings = None
        self._save()


# ======================== 记忆压缩器 (v2 新增) ========================

class MemoryCompressor:
    """记忆压缩器：对旧记忆进行摘要压缩"""

    def score_importance(self, text: str, category: str = "general") -> float:
        """基于启发式规则评估记忆重要度 (0.0 ~ 1.0)"""
        score = 0.5

        # 长度因素
        if len(text) > 200:
            score += 0.1
        if len(text) > 500:
            score += 0.1

        # 关键词提升
        important_keywords = ["错误", "失败", "成功", "重要", "关键", "注意", "bug", "error", "重要",
                              "密码", "API", "配置", "部署"]
        for kw in important_keywords:
            if kw.lower() in text.lower():
                score += 0.05

        # 类别因素
        category_weights = {"error": 0.2, "task_result": 0.15, "user_preference": 0.1,
                           "general": 0.0, "debug": 0.1}
        score += category_weights.get(category, 0)

        return min(max(score, 0.0), 1.0)

    def compress_entries(self, entries: list, target_count: int) -> list:
        """
        压缩条目：保留高重要度条目，合并低重要度条目

        Args:
            entries: [{"text": ..., "importance": ..., "timestamp": ...}, ...]
            target_count: 目标条目数

        Returns:
            压缩后的条目列表
        """
        if len(entries) <= target_count:
            return entries

        # 按重要度降序排列
        sorted_entries = sorted(entries, key=lambda e: e.get("importance", 0), reverse=True)

        # 保留前 target_count 条高重要度条目
        kept = sorted_entries[:target_count]

        # 剩余条目按类别分组摘要
        remaining = sorted_entries[target_count:]
        if remaining:
            categories = {}
            for entry in remaining:
                cat = entry.get("category", "general")
                categories.setdefault(cat, []).append(entry)

            for cat, cat_entries in categories.items():
                texts = [e.get("text", "")[:100] for e in cat_entries[:5]]
                summary_text = f"[压缩摘要:{cat}] " + "; ".join(texts)
                kept.append({
                    "text": summary_text[:500],
                    "category": cat,
                    "importance": 0.3,
                    "timestamp": time.time(),
                    "compressed_from": len(cat_entries),
                })

        return kept


# ======================== 记忆管理器 v2 ========================

class MemoryManager:
    """三层记忆系统管理器 v2（新增情景记忆 + 语义记忆 + 压缩器）"""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "user_data", "agent_memory")
        os.makedirs(data_dir, exist_ok=True)

        self.short_term = ShortTermMemory(max_messages=50)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(storage_path=os.path.join(data_dir, "long_term.json"))
        self.episodic = EpisodicMemory(storage_path=os.path.join(data_dir, "episodes.json"))
        self.semantic = SemanticMemory(storage_path=os.path.join(data_dir, "semantic.json"))
        self._compressor = MemoryCompressor()

    def add_message(self, role: str, content: str, metadata: dict = None):
        self.short_term.add_message(role, content, metadata)

    def get_context(self, last_n: int = 20) -> list:
        return self.short_term.get_context(last_n)

    def get_formatted_context(self, last_n: int = 10) -> str:
        return self.short_term.get_formatted_context(last_n)

    def remember(self, text: str, category: str = "general"):
        """存储到长期记忆"""
        self.long_term.store(text, category)

    def remember_important(self, text: str, category: str = "general"):
        """存储到语义记忆（带向量索引）"""
        importance = self._compressor.score_importance(text, category)
        self.semantic.store(text, category, importance)

    def recall(self, query: str, top_k: int = 5) -> list:
        """从长期记忆中检索"""
        return self.long_term.search(query, top_k)

    def recall_semantic(self, query: str, top_k: int = 5) -> list:
        """从语义记忆中向量检索"""
        return self.semantic.search(query, top_k)

    def remember_episode(self, task: str, steps: int, result_summary: str,
                         tools_used: list, success: bool, duration_ms: float):
        """从任务执行中自动记录情景"""
        importance = self._compressor.score_importance(result_summary, "task_result")
        episode = Episode(
            task=task,
            steps=steps,
            result_summary=result_summary[:300],
            tools_used=tools_used,
            duration_ms=duration_ms,
            success=success,
            importance=importance,
        )
        self.episodic.add_episode(episode)

    def recall_similar_tasks(self, query: str, top_k: int = 3) -> list:
        """召回相似任务的历史经验"""
        return self.episodic.search_by_text(query, top_k)

    def set_working(self, key: str, value: Any):
        self.working.set(key, value)

    def get_working(self, key: str, default: Any = None) -> Any:
        return self.working.get(key, default)

    def auto_compress(self):
        """自动压缩旧记忆"""
        # 压缩长期记忆
        if self.long_term.count() > 300:
            entries = self.long_term._entries
            compressed = self._compressor.compress_entries(entries, 200)
            self.long_term._entries = compressed
            self.long_term._save()
            logger.info(f"长期记忆已压缩: {len(entries)} → {len(compressed)}")

        # 压缩语义记忆
        if self.semantic.count() > 300:
            entries = self.semantic._entries
            compressed = self._compressor.compress_entries(entries, 200)
            self.semantic._entries = compressed
            self.semantic._save()
            logger.info(f"语义记忆已压缩: {len(entries)} → {len(compressed)}")

    def get_status(self) -> dict:
        return {
            "short_term_count": self.short_term.count(),
            "working_keys": self.working.list_keys(),
            "long_term_count": self.long_term.count(),
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "episodic_stats": self.episodic.get_stats(),
        }

    def clear_all(self):
        self.short_term.clear()
        self.working.clear()
        self.long_term.clear()
        self.episodic.clear()
        self.semantic.clear()


# 全局单例
memory = MemoryManager()