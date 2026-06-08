"""
OmniCore ↔ 态极桥接层（态极已剥离，保留兼容接口）
"""
import logging
from typing import Optional

logger = logging.getLogger("TaijiBridge")


class TaijiBridge:
    """态极桥接层（态极模块已剥离，此接口返回空）"""

    def __init__(self):
        self._taiji = None
        self._initialized = False

    def initialize(self, model=None, tokenizer=None, device: str = "cpu"):
        """态极已剥离，无法初始化"""
        logger.warning("态极模块已剥离，TaijiBridge 无法初始化。如需态极功能，请安装态极模块。")
        self._initialized = False

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