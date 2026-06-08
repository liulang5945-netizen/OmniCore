"""
OmniCore 兼容层
让旧版 API/前端仍能调用态极引擎

提供：
- 旧接口 → 态极接口的映射
- 状态同步
- 向后兼容
"""
import logging
from typing import Optional

logger = logging.getLogger("Taiji.Shell")


class OmniCoreCompat:
    """
    OmniCore 兼容适配器
    将旧的 OmniCore API 调用映射到态极引擎
    """

    def __init__(self):
        self._taiji_engine = None
        self._legacy_handlers = {}

    def register_taiji_engine(self, engine):
        """注册态极引擎"""
        self._taiji_engine = engine
        logger.info("态极引擎已注册到兼容层")

    def get_taiji_engine(self):
        """获取态极引擎"""
        return self._taiji_engine

    def is_available(self) -> bool:
        """检查态极引擎是否可用"""
        return self._taiji_engine is not None

    def map_legacy_request(self, legacy_request: dict) -> dict:
        """
        将旧版 OmniCore 请求格式映射到态极格式

        旧格式:
            {"prompt": "...", "model": "...", "max_tokens": 512}

        态极格式:
            {"input": "...", "config": {"max_tokens": 512}}
        """
        return {
            "input": legacy_request.get("prompt", ""),
            "config": {
                "max_tokens": legacy_request.get("max_tokens", 512),
                "temperature": legacy_request.get("temperature", 0.7),
                "stream": legacy_request.get("stream", False),
            },
        }


# 全局兼容层实例
compat_layer = OmniCoreCompat()
