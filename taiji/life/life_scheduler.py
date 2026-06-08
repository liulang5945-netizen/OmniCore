"""
态极生命调度器 (Life Scheduler)
================================

态极的"本能系统" — 需求驱动，而非时钟驱动。

人类不是"每2小时吃一次饭"，而是饿了才吃、累了才睡、无聊了才玩。
态极也应该有内在需求，由需求驱动行为。

需求系统：
  hunger    (饥饿度)  0~100  — 数据不足时上升，吃饭后下降
  fatigue   (疲劳度)  0~100  — 活动越多越累，睡觉后恢复
  boredom   (无聊度)  0~100  — 无新刺激时上升，玩耍后下降
  stress    (压力度)  0~100  — 错误多时上升，成功时下降
  curiosity (好奇心)  0~100  — 发现未知时上升，探索后下降

驱动逻辑：
  hunger > 70    → 自动触发吃饭
  fatigue > 80   → 自动触发睡觉
  boredom > 60   → 自动触发玩耍
  多个需求同时高时，优先级竞争
"""
import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("LifeScheduler")


@dataclass
class NeedsState:
    """态极的内在需求状态"""
    hunger: float = 30.0      # 饥饿度（初始略饿，需要进食）
    fatigue: float = 10.0     # 疲劳度
    boredom: float = 20.0     # 无聊度
    stress: float = 10.0      # 压力度
    curiosity: float = 50.0   # 好奇心

    def clamp_all(self):
        """确保所有需求在 0~100 范围内"""
        self.hunger = max(0, min(100, self.hunger))
        self.fatigue = max(0, min(100, self.fatigue))
        self.boredom = max(0, min(100, self.boredom))
        self.stress = max(0, min(100, self.stress))
        self.curiosity = max(0, min(100, self.curiosity))

    def to_dict(self) -> dict:
        return {
            "hunger": round(self.hunger, 1),
            "fatigue": round(self.fatigue, 1),
            "boredom": round(self.boredom, 1),
            "stress": round(self.stress, 1),
            "curiosity": round(self.curiosity, 1),
        }

    def dominant_need(self) -> str:
        """返回最强烈的需求"""
        needs = {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "boredom": self.boredom,
            "stress": self.stress,
            "curiosity": self.curiosity,
        }
        return max(needs, key=needs.get)


@dataclass
class LifeEvent:
    """一次生命事件"""
    timestamp: str
    event_type: str   # "feed" / "sleep" / "play" / "work" / "idle"
    trigger: str      # 触发原因（哪个需求导致的）
    needs_before: dict
    needs_after: dict
    duration_seconds: float = 0
    success: bool = True


class LifeScheduler:
    """
    态极的生命调度器

    不是定时器，而是"本能系统"。
    每次心跳评估需求，决定行动，执行行动。
    """

    # 需求阈值：超过这个值就触发对应行为
    HUNGER_THRESHOLD = 70
    FATIGUE_THRESHOLD = 80
    BOREDOM_THRESHOLD = 60

    # 需求自然增长速率（每分钟）
    HUNGER_GROWTH = 0.5       # 饥饿感缓慢上升
    FATIGUE_GROWTH = 0.3      # 疲劳感缓慢上升
    BOREDOM_GROWTH = 0.4      # 无聊感缓慢上升
    STRESS_DECAY = 0.2        # 压力自然缓慢下降
    CURIOSITY_GROWTH = 0.1    # 好奇心缓慢积累

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("life_data")
            except ImportError:
                data_dir = "taiji/life_data"
        self.data_dir = data_dir
        self.needs = NeedsState()
        self._life_state = "idle"  # idle / feeding / sleeping / playing / working
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._event_log: List[LifeEvent] = []
        self._last_heartbeat: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        self._total_heartbeats = 0
        self._lock = threading.Lock()

        # 引擎引用（延迟初始化）
        self._feed_engine = None
        self._sleep_engine = None
        self._play_engine = None

        os.makedirs(data_dir, exist_ok=True)
        self._load_state()

        logger.info("LifeScheduler initialized")

    # ─── 公开接口 ───────────────────────────────────

    def start(self):
        """启动生命"""
        if self._is_running:
            logger.warning("Life already running")
            return

        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._life_loop, daemon=True)
        self._thread.start()
        logger.info("🌱 Life started!")

    def stop(self):
        """暂停生命"""
        self._stop_event.set()
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._save_state()
        logger.info("⏸️ Life paused")

    def record_interaction(self, success: bool = True, topic: str = ""):
        """
        记录一次用户交互（外部调用）。

        每次用户与态极交互时调用，影响需求状态。
        """
        with self._lock:
            self._last_activity = datetime.now()

            if success:
                # 交互成功 → 降低饥饿/压力，提升好奇心
                self.needs.hunger -= 2
                self.needs.stress -= 3
                self.needs.curiosity += 1
                self.needs.boredom -= 5
                self.needs.fatigue += 1.5  # 工作会累
            else:
                # 交互失败 → 增加压力和饥饿
                self.needs.stress += 5
                self.needs.hunger += 3
                self.needs.fatigue += 2

            self.needs.clamp_all()

    def force_action(self, action: str) -> dict:
        """
        强制执行某个生命活动（手动触发）。

        Args:
            action: "feed" / "sleep" / "play"
        """
        return self._execute_action(action, trigger="manual")

    def get_status(self) -> dict:
        """获取生命状态"""
        return {
            "life_state": self._life_state,
            "is_running": self._is_running,
            "needs": self.needs.to_dict(),
            "dominant_need": self.needs.dominant_need(),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            "total_heartbeats": self._total_heartbeats,
            "total_events": len(self._event_log),
        }

    def get_timeline(self, hours: int = 24) -> List[dict]:
        """获取最近 N 小时的生命时间线"""
        cutoff = datetime.now().timestamp() - hours * 3600
        timeline = []
        for event in self._event_log:
            try:
                ts = datetime.fromisoformat(event.timestamp).timestamp()
                if ts >= cutoff:
                    timeline.append({
                        "time": event.timestamp,
                        "action": event.event_type,
                        "trigger": event.trigger,
                        "needs_before": event.needs_before,
                        "needs_after": event.needs_after,
                        "duration": event.duration_seconds,
                    })
            except Exception:
                continue
        return timeline[-50:]  # 最多返回 50 条

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        needs = self.needs

        state_emoji = {
            "idle": "😊", "feeding": "🍚", "sleeping": "💤",
            "playing": "🎮", "working": "🏃",
        }

        lines = [
            "🌱 生命调度器状态",
            "━━━━━━━━━━━━━━━━",
            f"当前状态: {state_emoji.get(self._life_state, '❓')} {self._life_state}",
            f"生命运行: {'✅ 启动' if self._is_running else '⏸️ 暂停'}",
            f"总心跳数: {self._total_heartbeats}",
            f"\n内在需求:",
            f"  🍚 饥饿度: {self._bar(needs.hunger)} {needs.hunger:.0f}",
            f"  😴 疲劳度: {self._bar(needs.fatigue)} {needs.fatigue:.0f}",
            f"  😐 无聊度: {self._bar(needs.boredom)} {needs.boredom:.0f}",
            f"  😰 压力度: {self._bar(needs.stress)} {needs.stress:.0f}",
            f"  🔍 好奇心: {self._bar(needs.curiosity)} {needs.curiosity:.0f}",
            f"\n最强烈的需求: {needs.dominant_need()}",
        ]

        # 下一步预测
        next_action = self._decide_action()
        if next_action:
            lines.append(f"下一步行动: {next_action}")
        else:
            lines.append(f"下一步行动: 继续等待（需求未达阈值）")

        return "\n".join(lines)

    # ─── 内部实现 ───────────────────────────────────

    def _life_loop(self):
        """生命主循环 — 心跳"""
        logger.info("💓 Heartbeat loop started")

        while not self._stop_event.is_set():
            try:
                self._heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # 每 60 秒一次心跳
            self._stop_event.wait(60)

        logger.info("💓 Heartbeat loop stopped")

    def _heartbeat(self):
        """一次心跳"""
        # 快速操作：更新需求 + 决策（持有锁）
        with self._lock:
            self._total_heartbeats += 1
            self._last_heartbeat = datetime.now()

            # 1. 需求自然变化
            self._needs_tick()

            # 2. 决定行动
            action = self._decide_action()

        # 慢操作：执行行动（不持有锁，避免阻塞 record_interaction 等调用）
        if action:
            self._execute_action(action, trigger=f"need_{self.needs.dominant_need()}")

        # 定期保存状态
        if self._total_heartbeats % 10 == 0:
            with self._lock:
                self._save_state()

    def _needs_tick(self):
        """需求自然变化（每次心跳调用）"""
        # 需求自然增长
        self.needs.hunger += self.HUNGER_GROWTH
        self.needs.fatigue += self.FATIGUE_GROWTH
        self.needs.boredom += self.BOREDOM_GROWTH
        self.needs.curiosity += self.CURIOSITY_GROWTH

        # 压力自然衰减
        self.needs.stress -= self.STRESS_DECAY

        # 如果很久没有活动，无聊度加速上升
        if self._last_activity:
            idle_minutes = (datetime.now() - self._last_activity).total_seconds() / 60
            if idle_minutes > 30:
                self.needs.boredom += 0.5
            if idle_minutes > 60:
                self.needs.fatigue += 0.3

        self.needs.clamp_all()

    def _decide_action(self) -> Optional[str]:
        """
        根据当前需求决定下一步行动。

        优先级：饥饿 > 疲劳 > 无聊
        """
        # 如果正在执行某个活动，不打断
        if self._life_state != "idle":
            return None

        # 检查各需求是否超过阈值
        if self.needs.hunger >= self.HUNGER_THRESHOLD:
            return "feed"
        if self.needs.fatigue >= self.FATIGUE_THRESHOLD:
            return "sleep"
        if self.needs.boredom >= self.BOREDOM_THRESHOLD:
            return "play"

        # 如果好奇心很高且有数据可吃，也可以吃饭
        if self.needs.curiosity >= 80 and self.needs.hunger >= 40:
            return "feed"

        return None

    def _execute_action(self, action: str, trigger: str = "") -> dict:
        """执行一个生命活动（不持有锁，避免阻塞其他调用）"""
        with self._lock:
            needs_before = self.needs.to_dict()
            self._life_state = action

        start_time = time.time()
        result = {"action": action, "success": False}

        try:
            if action == "feed":
                result = self._do_feed()
            elif action == "sleep":
                result = self._do_sleep()
            elif action == "play":
                result = self._do_play()
            else:
                logger.warning(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            result["error"] = str(e)

        duration = round(time.time() - start_time, 1)

        with self._lock:
            self._life_state = "idle"
            needs_after = self.needs.to_dict()

            # 记录事件
            event = LifeEvent(
                timestamp=datetime.now().isoformat(),
                event_type=action,
                trigger=trigger,
                needs_before=needs_before,
                needs_after=needs_after,
                duration_seconds=duration,
                success=result.get("success", False),
            )
            self._event_log.append(event)
            # 只保留最近 200 个事件
            self._event_log = self._event_log[-200:]

        logger.info(
            f"{'✅' if result.get('success') else '❌'} "
            f"Action: {action} ({trigger}), Duration: {duration}s"
        )

        return result

    def _do_feed(self) -> dict:
        """执行吃饭"""
        try:
            if self._feed_engine is None:
                from taiji.feed_engine import get_feed_engine
                self._feed_engine = get_feed_engine()

            report = self._feed_engine.feed(reason="auto")

            # 吃饭后：饥饿度大幅下降，好奇心略有提升
            self.needs.hunger -= 40
            self.needs.curiosity += 10
            self.needs.boredom -= 10
            self.needs.fatigue += 5  # 吃饭也有一点点累
            self.needs.clamp_all()

            return {"success": True, "samples": report.samples_generated}
        except Exception as e:
            logger.error(f"Feed failed: {e}")
            self.needs.hunger -= 10  # 即使失败也稍微缓解饥饿
            return {"success": False, "error": str(e)}

    def _do_sleep(self) -> dict:
        """执行睡觉"""
        try:
            if self._sleep_engine is None:
                from taiji.sleep_engine import get_sleep_engine
                self._sleep_engine = get_sleep_engine()

            report = self._sleep_engine.sleep(reason="auto")

            # 睡觉后：疲劳度清零，压力下降，饥饿上升
            self.needs.fatigue -= 60
            self.needs.stress -= 30
            self.needs.hunger += 20  # 睡醒了会饿
            self.needs.boredom += 10  # 睡醒了可能无聊
            self.needs.clamp_all()

            return {
                "success": True,
                "phases": report.phases_completed,
                "training_loss": report.training_loss,
            }
        except Exception as e:
            logger.error(f"Sleep failed: {e}")
            self.needs.fatigue -= 20
            return {"success": False, "error": str(e)}

    def _do_play(self) -> dict:
        """执行玩耍"""
        try:
            if self._play_engine is None:
                from taiji.play_engine import get_play_engine
                self._play_engine = get_play_engine()

            report = self._play_engine.play(reason="auto")

            # 玩耍后：无聊度下降，好奇心得到满足，压力下降
            self.needs.boredom -= 35
            self.needs.curiosity -= 15
            self.needs.stress -= 10
            self.needs.fatigue += 3  # 玩耍也稍微有点累
            self.needs.clamp_all()

            return {
                "success": True,
                "mood": report.mood,
                "activities": len(report.activities),
            }
        except Exception as e:
            logger.error(f"Play failed: {e}")
            self.needs.boredom -= 10
            return {"success": False, "error": str(e)}

    def _bar(self, value: float) -> str:
        """生成进度条"""
        filled = int(value / 10)
        return "█" * filled + "░" * (10 - filled)

    # ─── 持久化 ─────────────────────────────────────

    def _save_state(self):
        """保存生命状态"""
        try:
            state = {
                "needs": self.needs.to_dict(),
                "life_state": self._life_state,
                "total_heartbeats": self._total_heartbeats,
                "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
                "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            }
            path = os.path.join(self.data_dir, "life_state.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # 保存事件日志
            events_path = os.path.join(self.data_dir, "life_events.json")
            recent = self._event_log[-100:]
            events_data = []
            for e in recent:
                events_data.append({
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "trigger": e.trigger,
                    "needs_before": e.needs_before,
                    "needs_after": e.needs_after,
                    "duration_seconds": e.duration_seconds,
                    "success": e.success,
                })
            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save life state: {e}")

    def _load_state(self):
        """加载生命状态"""
        path = os.path.join(self.data_dir, "life_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            needs = state.get("needs", {})
            self.needs.hunger = needs.get("hunger", 30)
            self.needs.fatigue = needs.get("fatigue", 10)
            self.needs.boredom = needs.get("boredom", 20)
            self.needs.stress = needs.get("stress", 10)
            self.needs.curiosity = needs.get("curiosity", 50)
            self._total_heartbeats = state.get("total_heartbeats", 0)
            logger.info(f"Life state loaded: needs={self.needs.to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to load life state: {e}")

        # 加载事件日志
        events_path = os.path.join(self.data_dir, "life_events.json")
        if os.path.exists(events_path):
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
                for item in events_data:
                    self._event_log.append(LifeEvent(**item))
            except Exception:
                pass


# ─── 全局实例 ─────────────────────────────────────

_global_life: Optional[LifeScheduler] = None


def get_life_scheduler() -> LifeScheduler:
    """获取全局生命调度器实例"""
    global _global_life
    if _global_life is None:
        _global_life = LifeScheduler()
    return _global_life