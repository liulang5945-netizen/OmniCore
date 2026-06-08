"""
态极事件总线 (Event Bus)
========================

态极的循环系统 — 引擎间通信的血液。

发布-订阅模式，引擎间不直接调用，而是通过事件通信。
这样每个引擎都是独立的，可以自由组合和替换。
"""
import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Taiji.Events")


@dataclass
class Event:
    """一个事件"""
    event_type: str
    data: Dict[str, Any]
    source: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# 预定义事件类型
class EventType:
    # 需求变化
    NEED_CHANGED = "need_changed"
    # 用户交互
    INTERACTION = "interaction"
    INTERACTION_SUCCESS = "interaction_success"
    INTERACTION_FAILURE = "interaction_failure"
    # 生命活动
    FEED_COMPLETE = "feed_complete"
    SLEEP_COMPLETE = "sleep_complete"
    PLAY_COMPLETE = "play_complete"
    # 成长
    EVOLUTION = "evolution"
    UPGRADE_COMPLETE = "upgrade_complete"
    # 模型
    MODEL_LOADED = "model_loaded"
    MODEL_SWITCH = "model_switch"
    # 安全
    SAFETY_ALERT = "safety_alert"
    RESOURCE_WARNING = "resource_warning"
    # 系统
    ERROR = "error"
    HEALTH_CHECK = "health_check"
    LIFE_STARTED = "life_started"
    LIFE_STOPPED = "life_stopped"


class EventBus:
    """
    态极事件总线

    发布-订阅模式的事件系统，解耦引擎间的直接依赖。
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[Event] = []
        self._max_history = 500
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event_type: str, data: Dict[str, Any] = None, source: str = ""):
        """
        发布事件。

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源（引擎名）
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
        )

        with self._lock:
            self._history.append(event)
            # 限制历史长度
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            callbacks = self._subscribers.get(event_type, []).copy()

        # 在锁外调用回调，避免死锁
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error for {event_type}: {e}")

    def get_history(self, n: int = 50, event_type: str = None) -> List[dict]:
        """获取事件历史"""
        with self._lock:
            events = self._history[-n:]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return [
                {
                    "type": e.event_type,
                    "data": e.data,
                    "source": e.source,
                    "timestamp": e.timestamp,
                }
                for e in events
            ]

    def get_subscriber_count(self) -> Dict[str, int]:
        """获取每个事件类型的订阅者数量"""
        with self._lock:
            return {k: len(v) for k, v in self._subscribers.items()}


# ─── 全局实例 ─────────────────────────────────────

_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus