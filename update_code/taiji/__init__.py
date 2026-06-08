"""
Taiji Core (态极) — Unified Entry Point

Assembles all organs into a complete living being.
One line to start Taiji:
    from taiji import TaijiCore
    taiji = TaijiCore(model, tokenizer)
    taiji.start_life()

Taiji is a natively trained AI life form, NOT a fine-tuned pre-trained model.
"""
import logging
from typing import Optional, Dict, Any

from taiji.loader import load_model, save_model, create_model
from taiji.architecture import ModelSelf
from taiji.core.inference import NativeInferenceEngine

# 态极多模态引擎（旧接口，向后兼容）
try:
    from taiji.multimodal.multimodal_engine import TaijiMultimodalEngine
except ImportError:
    TaijiMultimodalEngine = None

logger = logging.getLogger("TaijiCore")


class TaijiCore:
    """
    态极核心 - 完整的生命体

    将大脑、心脏、消化系统、睡眠系统、玩耍系统、
    进化系统、记忆系统、免疫系统组装成一个完整的生命。
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        action_provider=None,
        data_collector=None,
    ):
        """
        创建态极生命体。

        Args:
            model: 态极的大脑（ModelSelf）
            tokenizer: 语言中枢（分词器）
            device: 计算设备（cpu/cuda）
            action_provider: 手脚（ActionProvider，可选）
            data_collector: 数据收集器（可选）
        """
        from taiji.life.body import BodyCore
        from taiji.infra.events import EventBus
        from taiji.safety.safety import SafetyGuard

        # 骨骼：资源管理
        self.body = BodyCore()
        self.body.set_model(model)
        self.body.set_tokenizer(tokenizer)
        self.body.set_device(device)
        if action_provider:
            self.body.set_action_provider(action_provider)
        if data_collector:
            self.body.set_data_collector(data_collector)

        # 循环系统：事件总线
        self.events = EventBus()

        # 免疫系统
        self.safety = SafetyGuard()

        # 生命系统（延迟初始化）
        self._life = None
        self._feed = None
        self._sleep = None
        self._play = None
        self._evolution = None
        self._auto_upgrade = None

        logger.info("TaijiCore created")

    # ── 生命系统属性（延迟初始化）──

    @property
    def life(self):
        """生命调度器（心跳）"""
        if self._life is None:
            from taiji.life.life_scheduler import LifeScheduler
            self._life = LifeScheduler()
        return self._life

    @property
    def feed(self):
        """喂养引擎（吃饭）— 使用全局单例，确保 API 端点和桥接层操作同一实例"""
        if self._feed is None:
            from taiji.life.feed_engine import get_feed_engine
            self._feed = get_feed_engine()
        return self._feed

    @property
    def sleep(self):
        """睡眠引擎（睡觉）— 使用全局单例"""
        if self._sleep is None:
            from taiji.life.sleep_engine import get_sleep_engine
            self._sleep = get_sleep_engine()
        return self._sleep

    @property
    def play(self):
        """玩耍引擎（娱乐）— 使用全局单例"""
        if self._play is None:
            from taiji.life.play_engine import get_play_engine
            self._play = get_play_engine()
        return self._play

    @property
    def evolution(self):
        """进化引擎— 使用全局单例"""
        if self._evolution is None:
            from taiji.life.evolution_engine import get_evolution_engine
            self._evolution = get_evolution_engine()
        return self._evolution

    # ── 生命控制 ──

    def start_life(self):
        """启动态极的生命（启动心跳循环）"""
        self.life.start()
        self.events.publish("life_started", source="taikicore")
        logger.info("Taiji life started")

    def stop_life(self):
        """暂停态极的生命"""
        self.life.stop()
        self.events.publish("life_stopped", source="taikicore")
        logger.info("Taiji life stopped")

    def record_interaction(self, success: bool = True, topic: str = ""):
        """记录一次用户交互（影响需求状态）"""
        self.life.record_interaction(success=success, topic=topic)
        self.events.publish(
            "interaction_success" if success else "interaction_failure",
            {"topic": topic},
            source="interaction",
        )

    # ── 感知和行动 ──

    def think(self, prompt: str, **kwargs) -> str:
        """思考（调用推理引擎生成文本）"""
        model = self.body.model
        tokenizer = self.body.tokenizer
        if model is None or tokenizer is None:
            return "[无模型，无法思考]"

        try:
            # 缓存推理引擎，避免每次创建
            if not hasattr(self, '_inference_engine') or self._inference_engine is None:
                self._inference_engine = NativeInferenceEngine(model, tokenizer)
            # 模型热切换后需要重建引擎
            elif self._inference_engine.model is not model:
                self._inference_engine = NativeInferenceEngine(model, tokenizer)
            result = self._inference_engine.generate(prompt, **kwargs)
            return self.safety.validate_output(result)
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return f"[思考失败: {e}]"

    def see(self, image_path: str) -> str:
        """看（调用视觉编码器理解图像）"""
        try:
            from taiji.multimodal.vision_encoder import TaijiVisionEncoder
            hidden_size = self.body.model.hidden_size if self.body.model else 768
            encoder = TaijiVisionEncoder(hidden_size=hidden_size)
            return encoder.describe_image_simple(image_path)
        except Exception as e:
            logger.error(f"See failed: {e}")
            return f"[视觉处理失败: {e}]"

    # ── 记忆 ──

    def remember(self, key: str, value: str):
        """记住信息"""
        try:
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()
            wm.remember(key, value, source="user_input")
            logger.debug(f"Remembered: {key}")
        except Exception as e:
            logger.error(f"Remember failed: {e}")

    def recall(self, key: str) -> Optional[str]:
        """回忆信息"""
        try:
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()
            return wm.recall(key)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return None

    # ── 手动触发生命活动 ──

    def do_feed(self) -> dict:
        """手动触发吃饭"""
        if not self.safety.rate_limit("feed"):
            return {"success": False, "reason": "频率限制"}
        report = self.feed.feed(reason="manual")
        self.events.publish("feed_complete", {"samples": report.samples_generated}, source="feed")
        return {
            "items_fed": report.items_fed,
            "samples_generated": report.samples_generated,
            "avg_quality": report.avg_quality,
        }

    def do_sleep(self) -> dict:
        """手动触发睡觉"""
        if not self.safety.rate_limit("sleep"):
            return {"success": False, "reason": "频率限制"}
        report = self.sleep.sleep(reason="manual")
        self.events.publish("sleep_complete", {"loss": report.training_loss}, source="sleep")
        return {
            "phases": report.phases_completed,
            "training_loss": report.training_loss,
            "health": report.health_status,
        }

    def do_play(self) -> dict:
        """手动触发玩耍"""
        if not self.safety.rate_limit("play"):
            return {"success": False, "reason": "频率限制"}
        report = self.play.play(reason="manual")
        self.events.publish("play_complete", {"mood": report.mood}, source="play")
        return {
            "activities": len(report.activities),
            "mood": report.mood,
            "traits": report.personality_traits_discovered,
        }

    # ── 状态查询 ──

    def get_status(self) -> dict:
        """获取态极完整状态"""
        return {
            "body": self.body.get_status(),
            "life": self.life.get_status(),
            "needs": self.life.needs.to_dict(),
            "safety": self.safety.get_status(),
            "events": {
                "total": len(self.events.get_history(1000)),
                "subscribers": self.events.get_subscriber_count(),
            },
            "feed": self.feed.get_status(),
            "sleep": self.sleep.get_status(),
            "play": self.play.get_status(),
            "evolution": {
                "phase": self.evolution.metrics.current_phase,
                "tasks_completed": self.evolution.metrics.tasks_completed,
            },
        }

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        needs = status["needs"]

        lines = [
            "态极生命状态",
            "============",
            f"大脑: {'就绪' if status['body']['has_model'] else '未加载'}",
            f"心跳: {'运行中' if status['life']['is_running'] else '暂停'}",
            f"免疫: {status['safety']['threat_level']}",
            "",
            "内在需求:",
            f"  饥饿: {needs['hunger']:.0f}/100",
            f"  疲劳: {needs['fatigue']:.0f}/100",
            f"  无聊: {needs['boredom']:.0f}/100",
            f"  压力: {needs['stress']:.0f}/100",
            f"  好奇: {needs['curiosity']:.0f}/100",
            "",
            f"吃饭次数: {status['feed']['total_feeds']}",
            f"睡觉次数: {status['sleep']['total_sleeps']}",
            f"玩耍次数: {status['play']['total_plays']}",
            f"进化阶段: {status['evolution']['phase']}",
        ]

        return "\n".join(lines)

    # ── 导出（生殖系统）──

    def export(self, path: str):
        """将态极导出为独立包"""
        if self.body.model and self.body.tokenizer:
            save_model(self.body.model, self.body.tokenizer, path)
            logger.info(f"Taiji exported to {path}")
        else:
            logger.warning("No model/tokenizer to export")

    # ── 加载（类方法）──

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "TaijiCore":
        """从磁盘加载态极"""
        model, tokenizer = load_model(model_path, device=device)
        taiji = cls(model=model, tokenizer=tokenizer, device=device)
        logger.info(f"Taiji loaded from {model_path}")
        return taiji
