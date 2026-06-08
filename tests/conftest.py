"""
测试共享 Fixtures
"""
import json
import os
import sys
import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """创建 FastAPI 测试客户端（缺少 torch 时自动跳过）"""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch 未安装，跳过需要 API 客户端的测试")
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


@pytest.fixture
def sample_jsonl(tmp_path):
    """创建临时 JSONL 测试数据集"""
    path = tmp_path / "test.jsonl"
    data = [{"instruction": f"问题{i}", "output": f"答案{i}"} for i in range(20)]
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data), encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_alpaca(tmp_path):
    """创建 Alpaca 格式测试数据集"""
    path = tmp_path / "alpaca.json"
    data = [
        {"instruction": "解释机器学习", "input": "", "output": "机器学习是AI的一个分支..."},
        {"instruction": "写一个排序算法", "input": "Python", "output": "def sort(arr):..."},
    ] * 10
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """临时持久化目录"""
    d = tmp_path / "persist"
    d.mkdir()
    return str(d)