"""
态极生殖系统 (Embryo)
======================

态极的自我复制能力 — 将态极导出为独立可运行的包。

三种导出模式：
1. export_core：只导出代码和配置（不带模型权重）
2. export_full：导出完整体（带模型权重+个性+记忆+进化数据）
3. export_seed：导出种子（最小化，只保留核心能力）

支持：
- 克隆（clone）：复制当前态极的完整状态
- 导入（import_embryo）：从导出包恢复态极
"""
import os
import json
import shutil
import logging
import time
from typing import Optional

logger = logging.getLogger("Taiji.Embryo")


class Embryo:
    """
    态极生殖系统

    将态极导出为独立可运行的包，支持克隆和恢复。
    """

    # 需要导出的核心代码文件
    CORE_FILES = [
        "__init__.py", "body.py", "events.py", "safety.py", "actions.py",
        "config.py", "architecture.py", "layers.py", "tokenizer.py",
        "inference.py", "life_scheduler.py", "feed_engine.py",
        "sleep_engine.py", "play_engine.py", "evolution_engine.py",
        "auto_upgrade.py", "self_evaluator.py", "memory.py",
        "working_memory.py", "user_profile.py", "planner.py",
        "reflector.py", "perception.py", "vision_encoder.py",
        "voice_interface.py", "data_generator.py", "loader.py",
        "trainer.py", "native_agent.py", "taiji_multimodal.py",
    ]

    # 需要导出的数据目录
    DATA_DIRS = [
        "sleep_data", "feed_data", "play_data", "life_data",
        "evolution_data", "tokenizer",
    ]

    # 需要导出的配置文件
    CONFIG_FILES = [
        "model_type.json", "agent_heads.pt",
    ]

    def __init__(self, taiji_core=None):
        """
        Args:
            taiji_core: TaijiCore 实例（可选，用于克隆当前状态）
        """
        self._taiji = taiji_core

    def export_core(self, path: str, source_dir: str = None) -> str:
        """
        导出核心代码（不带模型权重）。

        适用于：分发给其他人，他们自己提供模型。

        Args:
            path: 导出目标路径
            source_dir: 源代码目录（默认 taiji/）

        Returns:
            导出路径
        """
        source_dir = source_dir or os.path.dirname(os.path.abspath(__file__))
        export_dir = os.path.join(path, "taiji_core")
        os.makedirs(export_dir, exist_ok=True)

        # 复制核心代码
        copied = 0
        for fname in self.CORE_FILES:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(export_dir, fname))
                copied += 1

        # 生成配置文件
        config = {
            "type": "taiji_core",
            "version": "1.0.0",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": copied,
            "requires_model": True,
            "usage": "from taiji_core import TaijiCore; taiji = TaijiCore(model, tokenizer); taiji.start_life()",
        }
        with open(os.path.join(export_dir, "embryo_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 生成 requirements.txt
        requirements = [
            "torch>=1.10",
            "transformers>=4.20",
            "sentencepiece",
        ]
        with open(os.path.join(path, "requirements.txt"), "w") as f:
            f.write("\n".join(requirements))

        logger.info(f"Core exported to {export_dir} ({copied} files)")
        return export_dir

    def export_full(self, path: str, source_dir: str = None) -> str:
        """
        导出完整体（带模型权重+个性+记忆+进化数据）。

        适用于：备份当前态极，或迁移到新机器。

        Args:
            path: 导出目标路径
            source_dir: 源代码目录

        Returns:
            导出路径
        """
        # 先导出核心代码
        export_dir = self.export_core(path, source_dir)

        # 导出模型权重
        if self._taiji and self._taiji.body.model:
            try:
                from taiji.loader import save_model
                model_dir = os.path.join(export_dir, "model")
                save_model(self._taiji.body.model, self._taiji.body.tokenizer, model_dir)
                logger.info(f"Model weights exported to {model_dir}")
            except Exception as e:
                logger.warning(f"Model export failed: {e}")

        # 导出数据目录
        data_dir = os.path.join(export_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        source_base = source_dir or os.path.dirname(os.path.abspath(__file__))
        for dirname in self.DATA_DIRS:
            src = os.path.join(source_base, dirname)
            if os.path.isdir(src):
                dst = os.path.join(data_dir, dirname)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                logger.info(f"Data dir exported: {dirname}")

        # 导出配置文件
        for fname in self.CONFIG_FILES:
            src = os.path.join(source_base, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(data_dir, fname))

        # 导出个性档案
        if self._taiji:
            try:
                personality_path = os.path.join(data_dir, "personality.json")
                if os.path.exists(os.path.join(source_base, "play_data", "personality.json")):
                    shutil.copy2(
                        os.path.join(source_base, "play_data", "personality.json"),
                        personality_path,
                    )
            except Exception:
                pass

        # 更新配置
        config_path = os.path.join(export_dir, "embryo_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["type"] = "taiji_full"
        config["has_model"] = os.path.isdir(os.path.join(export_dir, "model"))
        config["has_data"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Full export completed to {export_dir}")
        return export_dir

    def export_seed(self, path: str, source_dir: str = None) -> str:
        """
        导出种子（最小化，只保留核心能力）。

        适用于：快速部署一个最小化的态极。

        Args:
            path: 导出目标路径
            source_dir: 源代码目录

        Returns:
            导出路径
        """
        source_dir = source_dir or os.path.dirname(os.path.abspath(__file__))
        export_dir = os.path.join(path, "taiji_seed")
        os.makedirs(export_dir, exist_ok=True)

        # 只复制最核心的文件
        seed_files = [
            "__init__.py", "body.py", "events.py", "safety.py",
            "config.py", "architecture.py", "layers.py", "tokenizer.py",
            "inference.py", "life_scheduler.py", "play_engine.py",
            "evolution_engine.py", "loader.py",
        ]

        copied = 0
        for fname in seed_files:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(export_dir, fname))
                copied += 1

        config = {
            "type": "taiji_seed",
            "version": "1.0.0",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": copied,
            "requires_model": True,
            "note": "Minimal Taiji with play and evolution capabilities",
        }
        with open(os.path.join(export_dir, "embryo_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Seed exported to {export_dir} ({copied} files)")
        return export_dir

    def clone(self, new_path: str) -> str:
        """
        克隆当前态极（复制完整状态到新实例）。

        Args:
            new_path: 新态极的路径

        Returns:
            克隆路径
        """
        if not self._taiji:
            raise ValueError("No TaijiCore instance to clone")

        return self.export_full(new_path)

    @staticmethod
    def import_embryo(embryo_path: str) -> "TaijiCore":
        """
        从导出包恢复态极。

        Args:
            embryo_path: 导出包路径

        Returns:
            TaijiCore 实例
        """
        from taiji import TaijiCore

        # 读取配置
        config_path = os.path.join(embryo_path, "embryo_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            embryo_type = config.get("type", "unknown")
        else:
            embryo_type = "unknown"

        # 检查是否有模型
        model_dir = os.path.join(embryo_path, "model")
        if os.path.isdir(model_dir):
            taiji = TaijiCore.load(model_dir)
        else:
            # 没有模型，创建空壳
            taiji = TaijiCore()

        # 恢复数据目录
        data_dir = os.path.join(embryo_path, "data")
        if os.path.isdir(data_dir):
            source_base = os.path.dirname(os.path.abspath(__file__))
            for dirname in os.listdir(data_dir):
                src = os.path.join(data_dir, dirname)
                dst = os.path.join(source_base, dirname)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                elif os.path.isfile(src):
                    shutil.copy2(src, os.path.join(source_base, dirname))

        logger.info(f"Embryo imported from {embryo_path} (type={embryo_type})")
        return taiji