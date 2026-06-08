"""
OmniCore ↔ 态极桥接层
=======================

连接 OmniCore 产品层和态极灵魂层。
OmniCore 通过这个桥接层使用态极，而不是直接导入 taiji。

使用方式：
    from core.taiji_bridge import get_taiji_bridge
    bridge = get_taiji_bridge()
    bridge.initialize(model, tokenizer)
    bridge.start_life()
"""
import logging
from typing import Optional

logger = logging.getLogger("TaijiBridge")


class TaijiBridge:
    """
    OmniCore 与态极的桥接层。

    翻译两个世界：
    - OmniCore 的 app_state → 态极的 BodyCore
    - OmniCore 的 API 请求 → 态极的生命活动
    - 态极的事件 → OmniCore 的状态更新
    """

    def __init__(self):
        self._taiji = None
        self._initialized = False

    def initialize(self, model=None, tokenizer=None, device: str = "cpu"):
        """
        初始化态极。

        Args:
            model: 模型（从 app_state.model 获取）
            tokenizer: 分词器（从 app_state.tokenizer 获取）
            device: 计算设备
        """
        from taiji import TaijiCore

        self._taiji = TaijiCore(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        self._initialized = True
        logger.info("TaijiBridge initialized")

    @property
    def taiji(self):
        """获取态极实例"""
        return self._taiji

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def start_life(self):
        """启动态极生命"""
        if self._taiji:
            self._taiji.start_life()

    def stop_life(self):
        """暂停态极生命"""
        if self._taiji:
            self._taiji.stop_life()

    def record_interaction(self, success: bool = True, topic: str = ""):
        """记录用户交互"""
        if self._taiji:
            self._taiji.record_interaction(success=success, topic=topic)

    def get_status(self) -> dict:
        """获取态极完整状态"""
        if self._taiji:
            return self._taiji.get_status()
        return {"initialized": False}

    def get_summary(self) -> str:
        """获取状态摘要"""
        if self._taiji:
            return self._taiji.get_summary()
        return "态极未初始化"

    def get_needs(self) -> dict:
        """获取需求状态"""
        if self._taiji:
            return self._taiji.life.needs.to_dict()
        return {}

    def do_feed(self) -> dict:
        """手动触发吃饭"""
        if self._taiji:
            return self._taiji.do_feed()
        return {"success": False, "reason": "态极未初始化"}

    def do_sleep(self) -> dict:
        """手动触发睡觉"""
        if self._taiji:
            return self._taiji.do_sleep()
        return {"success": False, "reason": "态极未初始化"}

    def do_play(self) -> dict:
        """手动触发玩耍"""
        if self._taiji:
            return self._taiji.do_play()
        return {"success": False, "reason": "态极未初始化"}

    def update_model(self, model, tokenizer):
        """更新态极的模型（热切换）"""
        if self._taiji:
            self._taiji.body.set_model(model)
            self._taiji.body.set_tokenizer(tokenizer)
            logger.info("Taiji model updated via bridge")

    def cleanup(self):
        """清理资源"""
        if self._taiji:
            self._taiji.body.cleanup()
            self._taiji = None
            self._initialized = False


# ─── 全局实例 ─────────────────────────────────────

_global_bridge: Optional[TaijiBridge] = None


def get_taiji_bridge() -> TaijiBridge:
    """获取全局桥接层实例"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = TaijiBridge()
    return _global_bridge