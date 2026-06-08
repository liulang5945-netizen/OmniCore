"""
utils.py 模块的严苛单元测试
覆盖：路径函数、JSON 安全读写、Toast HTML 生成、历史规范化、查询过滤
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import (
    get_external_path,
    get_internal_path,
    ensure_dir,
    safe_json_load,
    safe_json_save,
    safe_remove,
    build_toast_html,
    normalize_history,
    filter_by_query,
)

class TestPathFunctions:
    """路径工具测试"""

    def test_get_external_path_dev(self):
        path = get_external_path("test_dir")
        assert path.endswith("test_dir")
        assert os.path.isabs(path)

    def test_get_internal_path_dev(self):
        path = get_internal_path("test_dir")
        assert path.endswith("test_dir")
        assert os.path.isabs(path)

    def test_ensure_dir_creates_and_returns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "a", "b", "c")
            result = ensure_dir(target)
            assert result == target
            assert os.path.isdir(target)

    def test_ensure_dir_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(tmpdir)
            assert result == tmpdir


class TestSafeJSON:
    """JSON 安全读写测试"""

    def test_safe_json_load_non_existent_returns_default(self):
        result = safe_json_load("/nonexistent/path/file.json", default={"a": 1})
        assert result == {"a": 1}

    def test_safe_json_load_default_is_empty_dict(self):
        result = safe_json_load("/nonexistent/path/file.json")
        assert result == {}

    def test_safe_json_save_and_load_roundtrip(self):
        data = {"key": "value", "number": 42, "nested": {"x": True}}
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.json")
            safe_json_save(fpath, data)
            loaded = safe_json_load(fpath)
            assert loaded == data

    def test_safe_json_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "corrupt.json")
            with open(fpath, "w") as f:
                f.write("{invalid json")
            result = safe_json_load(fpath, default={"fallback": True})
            assert result == {"fallback": True}

    def test_safe_json_save_nested_dirs(self):
        data = {"deep": "nested"}
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sub", "dir", "data.json")
            safe_json_save(fpath, data)
            assert os.path.exists(fpath)
            loaded = safe_json_load(fpath)
            assert loaded == data


class TestSafeRemove:
    """安全文件删除测试"""

    def test_safe_remove_existing(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fpath = f.name
            f.write(b"test")
        assert os.path.exists(fpath)
        safe_remove(fpath)
        assert not os.path.exists(fpath)

    def test_safe_remove_non_existent(self):
        # 不应抛异常
        safe_remove("/nonexistent/file.txt")


class TestToastHTML:
    """Toast HTML 生成测试"""

    def test_info_toast(self):
        html = build_toast_html("测试消息", "info")
        assert "测试消息" in html
        assert "e0f2fe" in html  # info background
        assert "toastIn" in html

    def test_success_toast(self):
        html = build_toast_html("成功", "success")
        assert "成功" in html
        assert "dcfce7" in html

    def test_error_toast(self):
        html = build_toast_html("错误", "error")
        assert "错误" in html
        assert "fee2e2" in html

    def test_warning_toast(self):
        html = build_toast_html("警告", "warning")
        assert "警告" in html
        assert "fef9c3" in html

    def test_default_type_is_info(self):
        html = build_toast_html("msg", "invalid_type")
        assert "e0f2fe" in html


class TestNormalizeHistory:
    """聊天历史规范化测试"""

    def test_normalize_empty_list(self):
        result = normalize_history([])
        assert result == []

    def test_normalize_none(self):
        result = normalize_history(None)
        assert result == []

    def test_normalize_list_format(self):
        history = [["你好", "你好！有什么可以帮你的？"], ["再见", "再见！"]]
        result = normalize_history(history)
        assert len(result) == 2
        assert result[0] == ["你好", "你好！有什么可以帮你的？"]
        assert result[1] == ["再见", "再见！"]

    def test_normalize_dict_format(self):
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"},
        ]
        result = normalize_history(history)
        assert len(result) == 2
        assert result[0] == ["问题1", "回答1"]
        assert result[1] == ["问题2", "回答2"]

    def test_normalize_dict_skip_unpaired(self):
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"},
        ]
        result = normalize_history(history)
        assert len(result) == 1
        assert result[0] == ["问题2", "回答2"]

    def test_normalize_mixed_format_handled(self):
        history = [{"role": "unknown", "content": "x"}]
        result = normalize_history(history)
        assert result == []


class TestFilterByQuery:
    """查询过滤测试"""

    def test_filter_empty_query_returns_all(self):
        items = ["apple", "banana", "cherry"]
        result = filter_by_query(items, "")
        assert result == items

    def test_filter_by_exact(self):
        items = ["apple", "banana", "cherry"]
        result = filter_by_query(items, "ban")
        assert result == ["banana"]

    def test_filter_case_insensitive(self):
        items = ["Apple", "BANANA", "cherry"]
        result = filter_by_query(items, "apple")
        assert result == ["Apple"]

    def test_filter_no_match(self):
        items = ["apple", "banana"]
        result = filter_by_query(items, "xyz")
        assert result == []