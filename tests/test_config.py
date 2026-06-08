"""
config.py 模块的严苛单元测试
覆盖：TrainingConfig 各项默认值、resolve_device、auto_configure_for_hardware、get_config、路径函数
"""
import os
import sys
import json
import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import TrainingConfig, get_config, get_external_path, get_internal_path, HF_ENDPOINT


class TestTrainingConfigDefaults:
    """测试 TrainingConfig 默认值"""

    def test_default_model_name(self):
        cfg = TrainingConfig()
        assert cfg.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def test_default_device_is_auto(self):
        cfg = TrainingConfig()
        assert cfg.device == "auto"

    def test_default_epochs_is_3(self):
        cfg = TrainingConfig()
        assert cfg.num_epochs == 3

    def test_default_batch_size_is_4(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 4

    def test_default_learning_rate(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 2e-4

    def test_default_lora_disabled(self):
        cfg = TrainingConfig()
        assert cfg.use_lora is False

    def test_default_quant_4bit_disabled(self):
        cfg = TrainingConfig()
        assert cfg.load_in_4bit is False

    def test_default_quant_8bit_disabled(self):
        cfg = TrainingConfig()
        assert cfg.load_in_8bit is False

    def test_default_model_type_is_huggingface(self):
        cfg = TrainingConfig()
        assert cfg.model_type == "huggingface"

    def test_default_gguf_path_empty(self):
        cfg = TrainingConfig()
        assert cfg.gguf_path == ""

    def test_default_n_gpu_layers(self):
        cfg = TrainingConfig()
        assert cfg.n_gpu_layers == -1

    def test_default_n_ctx(self):
        cfg = TrainingConfig()
        assert cfg.n_ctx == 2048

    def test_default_max_length(self):
        cfg = TrainingConfig()
        assert cfg.max_length == 512

    def test_default_gradient_accumulation(self):
        cfg = TrainingConfig()
        assert cfg.gradient_accumulation_steps == 1

    def test_default_lora_r(self):
        cfg = TrainingConfig()
        assert cfg.lora_r == 8

    def test_default_lora_alpha(self):
        cfg = TrainingConfig()
        assert cfg.lora_alpha == 32


class TestTrainingConfigDevice:
    """测试设备解析"""

    def test_resolve_device_explicit(self):
        cfg = TrainingConfig()
        cfg.device = "cuda:0"
        assert cfg.resolve_device() == "cuda:0"

    def test_resolve_device_auto_not_implemented(self):
        """auto 模式下，至少返回一个有效字符串"""
        cfg = TrainingConfig()
        device = cfg.resolve_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu") or device.startswith("dml")

    def test_get_torch_dtype_returns_auto(self):
        cfg = TrainingConfig()
        assert cfg.get_torch_dtype() == "auto"


class TestTrainingConfigRAMDetection:
    """测试内存检测（仅验证不崩溃）"""

    def test_get_total_ram_gb_returns_positive_number(self):
        ram = TrainingConfig.get_total_ram_gb()
        assert ram > 0
        assert isinstance(ram, float)

    def test_detect_low_memory_system_returns_bool(self):
        result = TrainingConfig.detect_low_memory_system()
        assert isinstance(result, bool)


class TestTrainingConfigAutoConfigure:
    """测试自动硬件配置"""

    def test_auto_configure_no_error(self):
        """确保 auto_configure 不抛异常"""
        cfg = TrainingConfig()
        try:
            cfg.auto_configure_for_hardware()
        except Exception as e:
            pytest.fail(f"auto_configure_for_hardware 抛出异常: {e}")

    def test_auto_configure_changes_batch_size_on_low_ram(self):
        """低内存检测到时应调整 batch_size"""
        original_detect = TrainingConfig.detect_low_memory_system
        try:
            # 强制模拟低内存
            # 注意：detect_low_memory_system 是 @staticmethod，
            # 替换后通过实例调用会传入 self，因此 lambda 必须接受 1 个参数
            TrainingConfig.detect_low_memory_system = lambda _: True
            cfg = TrainingConfig()
            cfg.batch_size = 8
            cfg.auto_configure_for_hardware()
            # 低内存 + load_in_4bit 和 load_in_8bit 均为 False 时自动启用 4-bit
            assert cfg.load_in_4bit is True or cfg.load_in_8bit is True
            assert cfg.batch_size <= 2
        finally:
            TrainingConfig.detect_low_memory_system = original_detect

    def test_auto_configure_no_change_on_high_ram(self):
        """高内存检测到时应不变"""
        original_detect = TrainingConfig.detect_low_memory_system
        try:
            TrainingConfig.detect_low_memory_system = lambda _: False
            cfg = TrainingConfig()
            cfg.load_in_4bit = False
            cfg.batch_size = 4
            cfg.auto_configure_for_hardware()
            assert cfg.load_in_4bit is False
            assert cfg.batch_size == 4
        finally:
            TrainingConfig.detect_low_memory_system = original_detect


class TestGetConfig:
    """测试 get_config"""

    def test_get_config_without_args_returns_default(self):
        cfg = get_config(None)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def test_get_config_with_args(self):
        """使用 argparse.Namespace 模拟命令行参数"""
        import argparse
        args = argparse.Namespace(
            model_name="test-model",
            cache_dir="/test/cache",
            gguf_path="/test/model.gguf",
            batch_size=8,
            num_epochs=5,
            learning_rate=0.001,
            load_in_4bit=True,
            load_in_8bit=False,
            use_lora=True,
            lora_r=16,
            lora_alpha=64,
            output_dir="/test/output",
            dataset_path="/test/data.jsonl",
            n_gpu_layers=32,
            n_ctx=4096,
        )
        cfg = get_config(args)
        assert cfg.model_name == "test-model"
        assert cfg.cache_dir == "/test/cache"
        assert cfg.batch_size == 8
        assert cfg.num_epochs == 5
        assert cfg.learning_rate == 0.001
        assert cfg.load_in_4bit is True
        assert cfg.use_lora is True
        assert cfg.lora_r == 16
        assert cfg.n_gpu_layers == 32
        assert cfg.n_ctx == 4096


class TestPathFunctions:
    """测试路径工具函数"""

    def test_get_external_path_returns_str(self):
        path = get_external_path("test")
        assert isinstance(path, str)
        assert path.endswith("test")

    def test_get_internal_path_returns_str(self):
        path = get_internal_path("test")
        assert isinstance(path, str)
        assert path.endswith("test")

    def test_get_external_path_with_subdir(self):
        path = get_external_path("data/subdir/file.txt")
        assert "data" in path
        assert "file.txt" in path

    def test_hf_endpoint_is_defined(self):
        assert len(HF_ENDPOINT) > 0
        assert HF_ENDPOINT.startswith("http")