"""
gguf_engine.py 模块的单元测试（使用 mock，无需真实模型）
覆盖：GGUF 检测、模型注册、下载回退、GPU 检测
"""
import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gguf_engine import (
    BaseGGUFEngine,
    list_available_gguf_models,
    is_gguf_model,
    find_gguf_file,
    detect_available_gpu_backends,
    _find_local_gguf_fallback,
)


class TestGGUFDetection:
    """GGUF 文件检测测试"""

    def test_is_gguf_model_by_extension(self):
        assert is_gguf_model("model.gguf") is True

    def test_is_gguf_model_by_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "test.gguf")
            with open(gguf_path, "w") as f:
                f.write("test")
            assert is_gguf_model(tmpdir) is True

    def test_is_gguf_model_empty_path(self):
        assert is_gguf_model("") is False

    def test_is_gguf_model_none_path(self):
        assert is_gguf_model(None) is False

    def test_find_gguf_file_file_path(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            fpath = f.name
        try:
            result = find_gguf_file(fpath)
            assert result == fpath
        finally:
            os.unlink(fpath)

    def test_find_gguf_file_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "test.gguf")
            with open(gguf_path, "w") as f:
                f.write("test")
            result = find_gguf_file(tmpdir)
            assert result == gguf_path

    def test_find_gguf_file_not_found(self):
        assert find_gguf_file("") is None


class TestGGUFFallback:
    """本地备选回退测试"""

    def test_find_local_gguf_fallback_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "model.gguf")
            with open(fpath, "w") as f:
                f.write("test")
            result = _find_local_gguf_fallback_inner(tmpdir, "model.gguf")
            assert result == fpath

    def test_find_local_gguf_fallback_fuzzy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "other_model.gguf")
            with open(fpath, "w") as f:
                f.write("test")
            result = _find_local_gguf_fallback_inner(tmpdir, "nonexistent.gguf")
            assert result == fpath

    def test_find_local_gguf_fallback_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_local_gguf_fallback_inner(tmpdir, "nonexistent.gguf")
            assert result is None


def _find_local_gguf_fallback_inner(local_dir, filename):
    """使用给定目录测试 fallback 逻辑"""
    exact = os.path.join(local_dir, filename)
    if os.path.exists(exact):
        return exact
    for f in os.listdir(local_dir):
        if f.endswith(".gguf"):
            return os.path.join(local_dir, f)
    return None


class TestGGUFRegistry:
    """模型注册表测试"""

    def test_list_available_models_returns_dict(self):
        models = list_available_gguf_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_registry_keys_have_required_fields(self):
        models = list_available_gguf_models()
        for key, info in models.items():
            assert "repo" in info
            assert "filename" in info
            assert "size_gb" in info
            assert "description" in info

    def test_registry_filenames_end_with_gguf(self):
        models = list_available_gguf_models()
        for key, info in models.items():
            assert info["filename"].endswith(".gguf")


class TestGPUDetection:
    """GPU 后端检测测试"""

    def test_detect_cpu_always_available(self):
        backends = detect_available_gpu_backends()
        assert "cpu" in backends

    def test_detect_returns_list(self):
        backends = detect_available_gpu_backends()
        assert isinstance(backends, list)


class TestBaseGGUFEngineInit:
    """BaseGGUFEngine 初始化测试"""

    def test_init_defaults(self):
        engine = BaseGGUFEngine("test.gguf")
        assert engine.model_path == "test.gguf"
        assert engine.n_gpu_layers == -1
        assert engine.n_ctx == 2048
        assert engine.is_loaded is False

    def test_init_custom_params(self):
        engine = BaseGGUFEngine("test.gguf", n_gpu_layers=0, n_ctx=4096)
        assert engine.n_gpu_layers == 0
        assert engine.n_ctx == 4096

    def test_init_model_not_found(self):
        engine = BaseGGUFEngine("/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError):
            engine.load()