"""
配置管理器
线程安全的设置读写，带缓存和并发保护
"""
import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("SettingsManager")


class SettingsManager:
    """线程安全的配置管理器"""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._cache: Optional[dict] = None

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

    def _read_from_disk(self) -> dict:
        """从磁盘读取"""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取设置文件失败: {e}")
        return {}

    def _write_to_disk(self, data: dict):
        """写入磁盘"""
        self._ensure_dir()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self, key: str = None, default: Any = None) -> Any:
        """读取设置。key=None 返回全部设置。"""
        with self._lock:
            if self._cache is None:
                self._cache = self._read_from_disk()
            if key is None:
                return dict(self._cache)
            return self._cache.get(key, default)

    def set(self, key: str, value: Any):
        """写入单个设置（线程安全）"""
        with self._lock:
            if self._cache is None:
                self._cache = self._read_from_disk()
            self._cache[key] = value
            self._write_to_disk(self._cache)

    def update(self, data: dict):
        """批量更新设置"""
        with self._lock:
            if self._cache is None:
                self._cache = self._read_from_disk()
            self._cache.update(data)
            self._write_to_disk(self._cache)

    def delete(self, key: str) -> bool:
        """删除单个设置"""
        with self._lock:
            if self._cache is None:
                self._cache = self._read_from_disk()
            if key in self._cache:
                del self._cache[key]
                self._write_to_disk(self._cache)
                return True
            return False

    def reload(self):
        """强制从磁盘重新加载"""
        with self._lock:
            self._cache = self._read_from_disk()

    def save(self):
        """持久化当前缓存到文件"""
        with self._lock:
            if self._cache is not None:
                self._write_to_disk(self._cache)


# 全局实例
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """获取全局设置管理器实例"""
    global _settings_manager
    if _settings_manager is None:
        from core.utils import get_external_path
        _settings_manager = SettingsManager(get_external_path("app_settings.json"))
    return _settings_manager