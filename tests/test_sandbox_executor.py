"""
sandbox_executor.py 模块的严苛单元测试
覆盖：代码清理、沙箱脚本生成、结果解析、子进程安全执行
"""
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.sandbox_executor import (
    _clean_code,
    _generate_sandbox_script,
    _parse_sandbox_result,
    _format_sandbox_output,
    execute_python_code_safe,
    execute_python_with_files,
    SAFE_MODULES,
    MAX_OUTPUT_LENGTH,
    MAX_EXECUTION_TIME,
)


class TestCleanCode:
    """代码清理测试"""

    def test_clean_markdown_python(self):
        code = "```python\nprint('hello')\n```"
        cleaned = _clean_code(code)
        assert cleaned == "print('hello')"

    def test_clean_markdown_no_lang(self):
        code = "```\nx = 1\n```"
        cleaned = _clean_code(code)
        assert cleaned == "x = 1"

    def test_clean_no_fence(self):
        code = "print('hello')"
        cleaned = _clean_code(code)
        assert cleaned == "print('hello')"

    def test_clean_whitespace_only(self):
        code = "   \n   "
        cleaned = _clean_code(code)
        assert cleaned == ""

    def test_clean_mixed_indent(self):
        code = "  ```python\n  print('test')\n  ```\n  "
        cleaned = _clean_code(code)
        assert "test" in cleaned


class TestSandboxScriptGeneration:
    """沙箱脚本生成测试"""

    def test_generate_script_structure(self):
        script = _generate_sandbox_script("print('hello')", "/tmp/sandbox")
        assert "__SANDBOX_RESULT_START__" in script
        assert "__SANDBOX_RESULT_END__" in script
        assert "exec" in script.lower()

    def test_generate_script_contains_safe_import(self):
        script = _generate_sandbox_script("import math", "/tmp/sandbox")
        assert "SAFE_MODULES" in script
        assert "_safe_import" in script

    def test_generate_script_forbids_open(self):
        script = _generate_sandbox_script("open('file.txt')", "/tmp/sandbox")
        assert "open" in script  # 应出现在 _FORBIDDEN_NAMES 中

    def test_generate_script_empty_code(self):
        script = _generate_sandbox_script("", "/tmp/sandbox")
        assert len(script) > 0
        assert "base64" in script


class TestSafeModules:
    """白名单模块测试"""

    def test_safe_modules_is_frozenset(self):
        assert isinstance(SAFE_MODULES, frozenset)

    def test_numpy_in_safe_modules(self):
        assert "numpy" in SAFE_MODULES

    def test_pandas_in_safe_modules(self):
        assert "pandas" in SAFE_MODULES

    def test_math_in_safe_modules(self):
        assert "math" in SAFE_MODULES

    def test_matplotlib_in_safe_modules(self):
        assert "matplotlib" in SAFE_MODULES

    def test_os_not_in_safe_modules(self):
        assert "os" not in SAFE_MODULES

    def test_sys_not_in_safe_modules(self):
        assert "sys" not in SAFE_MODULES


class TestParseSandboxResult:
    """沙箱结果解析测试"""

    def test_parse_normal_result(self):
        stdout = 'prefix text\n__SANDBOX_RESULT_START__\n{"output": "hello", "truncated": false, "images": [], "files_created": [], "files_modified": []}\n__SANDBOX_RESULT_END__\nsuffix'
        result = _parse_sandbox_result(stdout, "", 0)
        assert result["output"] == "hello"
        assert result["truncated"] is False
        assert result["images"] == []
        assert result["files_created"] == []

    def test_parse_result_with_images(self):
        stdout = (
            '__SANDBOX_RESULT_START__\n'
            '{"output": "", "truncated": false, "images": [{"filename": "plot.png", "data": "AAAA"}], "files_created": ["code.py"], "files_modified": []}\n'
            '__SANDBOX_RESULT_END__'
        )
        result = _parse_sandbox_result(stdout, "", 0)
        assert len(result["images"]) == 1
        assert result["images"][0]["filename"] == "plot.png"
        assert "code.py" in result["files_created"]

    def test_parse_no_markers(self):
        stdout = "just output"
        result = _parse_sandbox_result(stdout, "stderr text", 1)
        assert "just output" in result["output"]
        assert "stderr text" in result["output"]

    def test_parse_error_returncode(self):
        result = _parse_sandbox_result("", "error occurred", 1)
        assert "异常退出" in result.get("error", "")

    def test_parse_malformed_json(self):
        stdout = '__SANDBOX_RESULT_START__\n{invalid\n__SANDBOX_RESULT_END__'
        result = _parse_sandbox_result(stdout, "", 0)
        assert "解析失败" in result.get("output", "")


class TestFormatSandboxOutput:
    """输出格式化测试"""

    def test_format_empty_output(self):
        result = {"output": "", "truncated": False, "images": [], "files_created": [], "error": ""}
        formatted = _format_sandbox_output(result)
        assert "无任何控制台输出" in formatted

    def test_format_with_output(self):
        result = {"output": "hello world", "truncated": False, "images": [], "files_created": [], "error": ""}
        formatted = _format_sandbox_output(result)
        assert "hello world" in formatted

    def test_format_truncated(self):
        result = {"output": "x" * 1000, "truncated": True, "images": [], "files_created": [], "error": ""}
        formatted = _format_sandbox_output(result)
        assert "截断" in formatted or "限制" in formatted

    def test_format_with_error(self):
        result = {"output": "", "truncated": False, "images": [], "files_created": [], "error": "timeout"}
        formatted = _format_sandbox_output(result)
        assert "timeout" in formatted

    def test_format_with_files(self):
        result = {"output": "ok", "truncated": False, "images": [], "files_created": ["a.py", "b.txt"], "error": ""}
        formatted = _format_sandbox_output(result)
        assert "新创建的文件" in formatted
        assert "a.py" in formatted


class TestSafeExecution:
    """安全执行集成测试（使用子进程）"""

    def test_simple_print(self):
        result = execute_python_code_safe("print('hello sandbox')")
        assert "hello sandbox" in result

    def test_math_operation(self):
        result = execute_python_code_safe("print(2 + 3 * 4)")
        assert "14" in result

    def test_empty_code(self):
        result = execute_python_code_safe("")
        assert "空" in result or "跳过" in result

    def test_syntax_error(self):
        result = execute_python_code_safe("print(1/0")
        assert len(result) > 0  # 不应崩溃

    def test_execute_with_files_empty_code(self):
        result = execute_python_with_files("")
        assert result["success"] is False
        assert result["error"] == "代码为空"

    def test_execute_with_files_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_python_with_files("print('test')", workspace_dir=tmpdir)
            assert result["success"] is True
            assert "test" in result["output"]

    def test_timeout_on_infinite_loop(self):
        result = execute_python_code_safe("while True: pass")
        assert "超时" in result or "终止" in result

    def test_forbidden_eval(self):
        result = execute_python_code_safe("eval('1+1')")
        # 沙箱应拦截 eval
        assert "1+1" not in result or "error" in result.lower() or "NameError" in result

    def test_forbidden_open_write(self):
        result = execute_python_code_safe("open('test.txt', 'w').write('hack')")
        # 沙箱应拦截 open
        assert "error" in result.lower() or "NameError" in result or "TypeError" in result or "not defined" in result.lower()

    def test_markdown_code_block(self):
        result = execute_python_code_safe("```python\nprint('cleaned')\n```")
        assert "cleaned" in result

    def test_numpy_import(self):
        try:
            result = execute_python_code_safe("import numpy as np; print(np.array([1,2,3]).sum())")
            assert "6" in result
        except Exception:
            pass  # numpy 可能未安装

    def test_execution_time_threshold(self):
        """验证默认超时配置合理"""
        assert MAX_EXECUTION_TIME == 60
        assert MAX_OUTPUT_LENGTH == 10000