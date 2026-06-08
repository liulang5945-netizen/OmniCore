"""
训练配置推荐引擎
================
基于硬件规格和模型大小，自动推荐最优训练配置。
支持三档预设：低配 / 中配 / 高配。
"""
import logging
import os
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("TrainingRecommender")


class TrainingRecommender:
    """基于硬件的训练配置推荐引擎"""

    PRESETS = {
        "low": {
            "label": "低配 (保底)",
            "description": "适合 8GB 以下显存，优先保证不 OOM",
            "use_lora": True,
            "lora_r": 8,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "max_length": 256,
            "warmup_steps": 5,
        },
        "mid": {
            "label": "中配 (推荐)",
            "description": "平衡速度和质量",
            "use_lora": True,
            "lora_r": 16,
            "load_in_4bit": False,
            "load_in_8bit": True,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "max_length": 512,
            "warmup_steps": 10,
        },
        "high": {
            "label": "高配 (最佳)",
            "description": "需要 24GB+ 显存，最佳训练质量",
            "use_lora": True,
            "lora_r": 32,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "max_length": 1024,
            "warmup_steps": 20,
        },
    }

    def detect_hardware(self) -> dict:
        """检测硬件信息"""
        info = {
            "ram_gb": 0,
            "gpu_name": "无",
            "gpu_vram_gb": 0,
            "cpu_cores": os.cpu_count() or 4,
            "has_gpu": False,
            "has_cuda": False,
        }

        try:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass

        try:
            import torch
            info["has_cuda"] = torch.cuda.is_available()
            if info["has_cuda"]:
                info["has_gpu"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
                )
        except Exception:
            pass

        return info

    def estimate_model_params(self, model_path: str) -> float:
        """估算模型参数量（单位：B）"""
        # 方法1: 从路径名猜
        name_lower = model_path.lower()
        for pattern, params in [
            ("70b", 70.0), ("34b", 34.0), ("13b", 13.0),
            ("7b", 7.0), ("3b", 3.0), ("1.5b", 1.5), ("1b", 1.0),
            ("0.5b", 0.5), ("500m", 0.5), ("350m", 0.35),
        ]:
            if pattern in name_lower:
                return params

        # 方法2: 读取 config.json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                hidden = config.get("hidden_size", 4096)
                layers = config.get("num_hidden_layers", 32)
                vocab = config.get("vocab_size", 32000)
                intermediate = config.get("intermediate_size", hidden * 4)
                # 近似公式
                params = (vocab * hidden + layers * (4 * hidden**2 + 2 * hidden * intermediate + 13 * hidden)) / 1e9
                return round(params, 1)
            except Exception:
                pass

        return 7.0  # 默认假设 7B

    def recommend(self, model_path: str, dataset_size: int,
                  preset: str = "mid") -> dict:
        """
        推荐训练配置

        Args:
            model_path: 模型路径
            dataset_size: 数据集样本数
            preset: 预设级别 "low" / "mid" / "high"

        Returns:
            推荐配置字典
        """
        hw = self.detect_hardware()
        params_b = self.estimate_model_params(model_path)
        base = dict(self.PRESETS.get(preset, self.PRESETS["mid"]))

        # 基于硬件调整
        if hw["gpu_vram_gb"] > 0:
            # 有 GPU
            if hw["gpu_vram_gb"] < 8:
                base = dict(self.PRESETS["low"])
                base["warnings"] = ["显存不足 8GB，已自动切换到低配模式"]
            elif hw["gpu_vram_gb"] < 16 and preset == "high":
                base = dict(self.PRESETS["mid"])
                base["warnings"] = ["显存不足 16GB，已自动切换到中配模式"]
            elif hw["gpu_vram_gb"] >= 24 and preset == "low":
                base["warnings"] = ["显存充足，建议使用中配或高配模式"]
        else:
            # 无 GPU，强制低配
            base = dict(self.PRESETS["low"])
            base["batch_size"] = 1
            base["gradient_accumulation_steps"] = 32
            base["max_length"] = 128
            base["load_in_4bit"] = False
            base["load_in_8bit"] = False
            base["warnings"] = ["未检测到 GPU，将使用 CPU 训练（速度较慢）"]

        # 大模型调整
        if params_b > 13:
            base["lora_r"] = min(base["lora_r"], 16)
            base["batch_size"] = max(1, base["batch_size"] // 2)
            base.setdefault("warnings", []).append(
                f"模型约 {params_b}B 参数，已自动降低 batch_size"
            )

        # 估算训练时间
        effective_batch = base["batch_size"] * base["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(dataset_size / effective_batch)
        total_steps = steps_per_epoch * 3  # 默认 3 epochs
        # 粗略估算：GPU 2-5ms/step，CPU 50-200ms/step
        ms_per_step = 3 if hw["has_gpu"] else 100
        estimated_hours = (total_steps * ms_per_step / 1000 / 3600) * params_b / 7.0

        # 估算显存占用
        if base["load_in_4bit"]:
            vram_gb = params_b * 0.5 + 1.5
        elif base["load_in_8bit"]:
            vram_gb = params_b * 1.0 + 2.0
        else:
            vram_gb = params_b * 2.0 + 3.0

        result = {
            **base,
            "preset": preset,
            "hardware": hw,
            "model_params_b": params_b,
            "dataset_size": dataset_size,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_time_hours": round(estimated_hours, 1),
            "estimated_vram_gb": round(vram_gb, 1),
            "effective_batch_size": effective_batch,
        }
        return result

    def get_all_presets(self, model_path: str, dataset_size: int) -> dict:
        """获取所有预设配置"""
        return {
            preset: self.recommend(model_path, dataset_size, preset)
            for preset in ["low", "mid", "high"]
        }