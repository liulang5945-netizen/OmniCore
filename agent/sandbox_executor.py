"""
安全的 Python 沙箱执行器（子进程隔离版）

设计原理：
- 将用户/LLM 生成的代码放入独立的子进程执行
- 子进程通过安全限制模块白名单防护
- 主进程与子进程通过 stdin/stdout 通信
- 严格的资源限制（时间、输出大小）
- 不依赖 exec() 沙箱逃逸防护（真正的进程级隔离）
- 支持文件变化追踪（对独立开发任务至关重要）

使用方法：
    result = execute_python_code_safe(code_string)
    result = execute_python_with_files(code_string)  # 增强版，返回文件变更
"""
import json
import logging
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("SafeSandbox")

# ======================== 沙箱安全性配置 ========================

# 白名单模块（仅在子进程中使用，防止恶意加载）
SAFE_MODULES = frozenset({
    "math", "random", "statistics", "itertools", "collections",
    "functools", "operator", "re", "string", "datetime",
    "json", "csv", "io", "textwrap", "base64",
    "urllib", "urllib.request", "urllib.parse", "requests", "bs4",
    "typing", "enum", "copy", "hashlib", "decimal", "fractions",
    "numpy", "pandas",
    "matplotlib", "matplotlib.pyplot",
    "glob", "uuid",  # 文件操作支持
})

# 最大输出字符数
MAX_OUTPUT_LENGTH = 10000

# 最大执行时间（秒）
MAX_EXECUTION_TIME = 60  # 增强：从30秒提高到60秒，支持更复杂的代码

# 最大生成图片数
MAX_IMAGES = 10

# Matplotlib 后端
MPL_BACKEND = "Agg"


def _build_safety_preamble(safe_modules_repr: str) -> list:
    """生成沙箱安全策略代码段：白名单模块 + builtins 过滤 + 安全 import 钩子"""
    lines = []
    lines.append("import base64, io, json, os, sys, textwrap, traceback, uuid")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# ===== 沙箱安全策略 =====")
    lines.append("")
    lines.append(f"_SAFE_MODULES = {safe_modules_repr}")
    lines.append("")
    lines.append(f'os.environ["MPLBACKEND"] = "{MPL_BACKEND}"')
    lines.append("import matplotlib")
    lines.append(f'matplotlib.use("{MPL_BACKEND}")')
    lines.append("")
    lines.append("# 禁用 os 模块中的危险函数")
    lines.append("for _dangerous in ['system', 'popen', 'exec', 'execv', 'execve', 'execvp', 'execvpe', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe', 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'popen2', 'popen3', 'popen4']:")
    lines.append("    if hasattr(os, _dangerous):")
    lines.append("        def _block(*a, _n=_dangerous, **kw): raise PermissionError(f'os.{_n} is disabled in sandbox')")
    lines.append("        setattr(os, _dangerous, _block)")
    lines.append("del _dangerous, _block")
    lines.append("")
    lines.append("# 限制危险操作")
    lines.append("_FORBIDDEN_NAMES = frozenset([")
    lines.append('    "exec", "eval", "compile", "__import__", "open",')
    lines.append('    "input", "breakpoint", "help", "exit", "quit",')
    lines.append("])")
    lines.append("")
    lines.append("# 安全 builtins 构建")
    lines.append("_raw_builtins = __builtins__")
    lines.append("if isinstance(_raw_builtins, dict):")
    lines.append("    _safe_builtins = dict(_raw_builtins)")
    lines.append("else:")
    lines.append("    _safe_builtins = dict(_raw_builtins.__dict__)")
    lines.append("")
    lines.append("for _name in _FORBIDDEN_NAMES:")
    lines.append("    _safe_builtins.pop(_name, None)")
    lines.append("")
    lines.append("# 安全的 import 钩子")
    lines.append('_original_import = _safe_builtins.get("__import__", __import__)')
    lines.append("")
    lines.append("def _safe_import(name, *args, **kwargs):")
    lines.append('    _top_level = name.split(".")[0]')
    lines.append("    if _top_level in _SAFE_MODULES:")
    lines.append("        return _original_import(name, *args, **kwargs)")
    lines.append('    raise ImportError(f"安全沙箱禁止导入非白名单模块: {name}")')
    lines.append("")
    lines.append('_safe_builtins["__import__"] = _safe_import')
    lines.append("")
    return lines


def _build_env_setup(sandbox_dir_repr: str, safe_builtins: str = "_safe_builtins") -> list:
    """生成执行环境配置代码段：工作目录、globals 字典"""
    lines = []
    lines.append("# ===== 执行环境 =====")
    lines.append("")
    lines.append(f"_sandbox_dir = {sandbox_dir_repr}")
    lines.append("os.makedirs(_sandbox_dir, exist_ok=True)")
    lines.append("os.chdir(_sandbox_dir)")
    lines.append("")
    lines.append("_original_open = open")
    lines.append("def _safe_open(file, mode='r', *args, **kwargs):")
    lines.append("    _p = os.path.abspath(file)")
    lines.append("    _w = os.path.abspath(_sandbox_dir)")
    lines.append("    if not _p.startswith(_w):")
    lines.append("        raise PermissionError(f'Sandbox violation: path traversal blocked.')")
    lines.append("    return _original_open(file, mode, *args, **kwargs)")
    lines.append("io.open = _safe_open")
    lines.append("pathlib.Path.open = lambda self, *a, **kw: _safe_open(str(self), *a, **kw) if 'pathlib' in sys.modules else None")
    lines.append(f"{safe_builtins}['open'] = _safe_open")
    lines.append("")
    lines.append("_safe_globals = {")
    lines.append('    "__name__": "__main__",')
    lines.append(f'    "__builtins__": {safe_builtins},')
    lines.append('    "__file__": str(Path(_sandbox_dir) / "sandbox_script.py"),')
    lines.append("}")
    lines.append("")
    return lines


def _build_matplotlib_preamble() -> list:
    """生成 matplotlib 预处理代码段：字体配置 + show 重定向为 savefig"""
    lines = []
    lines.append("# ===== matplotlib 预处理 =====")
    lines.append("")
    lines.append('_preamble = """')
    lines.append('import matplotlib.pyplot as plt')
    lines.append('import uuid')
    lines.append('plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "sans-serif"]')
    lines.append('plt.rcParams["axes.unicode_minus"] = False')
    lines.append('def _custom_show(*args, **kwargs):')
    lines.append('    fname = f"plot_{uuid.uuid4().hex[:6]}.png"')
    lines.append('    plt.savefig(fname)')
    lines.append('    print(f"[IMAGE GENERATED: {fname}]")')
    lines.append('    plt.clf()')
    lines.append('plt.show = _custom_show')
    lines.append('"""')
    lines.append("")
    return lines


def _build_execution_core(encoded_code: str) -> list:
    """生成用户代码解码与执行代码段：解码 markdown 代码块 + exec"""
    lines = []
    lines.append("# ===== 解码并执行用户代码 =====")
    lines.append("")
    lines.append("_output_buf = io.StringIO()")
    lines.append("")
    lines.append("try:")
    lines.append(f'    _user_code = base64.b64decode("{encoded_code}").decode("utf-8")')
    lines.append("")
    lines.append("    # 去除 markdown 代码块标记")
    lines.append("    _user_code = _user_code.strip()")
    lines.append('    if _user_code.startswith("```python"):')
    lines.append("        _user_code = _user_code[9:]")
    lines.append('    elif _user_code.startswith("```"):')
    lines.append("        _user_code = _user_code[3:]")
    lines.append('    if _user_code.endswith("```"):')
    lines.append("        _user_code = _user_code[:-3]")
    lines.append("    _user_code = _user_code.strip()")
    lines.append("")
    lines.append('    _full_code = _preamble + "\\n" + _user_code')
    lines.append("")
    lines.append("    sys.stdout = _output_buf")
    lines.append("    sys.stderr = _output_buf")
    lines.append("")
    lines.append("    exec(_full_code, _safe_globals)")
    lines.append("")
    lines.append("except SystemExit:")
    lines.append("    pass")
    lines.append("except Exception:")
    lines.append("    print(traceback.format_exc())")
    lines.append("")
    lines.append("finally:")
    lines.append("    sys.stdout = sys.__stdout__")
    lines.append("    sys.stderr = sys.__stderr__")
    lines.append("")
    return lines


def _build_file_tracking() -> list:
    """生成文件变更追踪代码段：执行前后文件快照对比"""
    lines = []
    lines.append("# ===== 记录文件变更 =====")
    lines.append("")
    lines.append("_before_files = set()")
    lines.append("if os.path.exists(_sandbox_dir):")
    lines.append("    for _root, _dirs, _files in os.walk(_sandbox_dir):")
    lines.append("        for _f in _files:")
    lines.append("            _rel = os.path.relpath(os.path.join(_root, _f), _sandbox_dir)")
    lines.append("            if not _rel.startswith('_sandbox_'):")
    lines.append("                _before_files.add(_rel)")
    lines.append("")
    return lines


def _build_result_output() -> list:
    """生成结果收集与 JSON 输出代码段：捕获输出、文件、图片、序列化"""
    lines = []
    lines.append("# ===== 输出结果 =====")
    lines.append("")
    lines.append("result = {")
    lines.append(f'    "output": _output_buf.getvalue()[:{MAX_OUTPUT_LENGTH}],')
    lines.append(f'    "truncated": len(_output_buf.getvalue()) > {MAX_OUTPUT_LENGTH},')
    lines.append("}")
    lines.append("")
    lines.append("# 检测新创建的文件")
    lines.append("_new_files = []")
    lines.append("if os.path.exists(_sandbox_dir):")
    lines.append("    for _root, _dirs, _files in os.walk(_sandbox_dir):")
    lines.append("        for _f in _files:")
    lines.append("            _rel = os.path.relpath(os.path.join(_root, _f), _sandbox_dir)")
    lines.append("            if not _rel.startswith('_sandbox_') and _rel not in _before_files:")
    lines.append("                _new_files.append(_rel)")
    lines.append('result["files_created"] = sorted(_new_files)')
    lines.append('result["files_modified"] = []')
    lines.append("")
    lines.append("# 收集生成的新图片")
    lines.append("import glob")
    lines.append("images = []")
    lines.append('for ext in ["*.png", "*.jpg", "*.jpeg"]:')
    lines.append("    for f in glob.glob(ext):")
    lines.append("        try:")
    lines.append('            with open(f, "rb") as _img_f:')
    lines.append("                import base64 as _b64")
    lines.append("                images.append({")
    lines.append('                    "filename": f,')
    lines.append('                    "data": _b64.b64encode(_img_f.read()).decode("utf-8"),')
    lines.append("                })")
    lines.append("        except Exception:")
    lines.append("            pass")
    lines.append(f'result["images"] = images[:{MAX_IMAGES}]')
    lines.append("")
    lines.append("# 以 JSON 格式输出结果到 stdout")
    lines.append('print("__SANDBOX_RESULT_START__")')
    lines.append("print(json.dumps(result, ensure_ascii=False))")
    lines.append('print("__SANDBOX_RESULT_END__")')
    return lines


def _generate_sandbox_script(code: str, sandbox_dir: str) -> str:
    """
    生成在子进程中执行的沙箱脚本。
    脚本通过 base64 嵌入用户代码，避免转义问题。
    子进程的 __builtins__ 经过安全过滤。

    内部拆分（便于维护和测试）：
      _build_safety_preamble()    → 安全策略 + builtins 过滤
      _build_env_setup()          → 工作目录 + globals
      _build_matplotlib_preamble()→ matplotlib 图片保存
      _build_execution_core()     → 解码 + exec 用户代码
      _build_file_tracking()      → 执行前后文件快照对比
      _build_result_output()      → JSON 序列化输出
    """
    import base64
    encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    safe_modules_repr = sorted(SAFE_MODULES)
    sandbox_dir_repr = repr(sandbox_dir)

    # 使用字符串拼接而非 f-string 模板，避免缩进/引号混乱
    lines = []
    lines.extend(_build_safety_preamble(safe_modules_repr))
    lines.extend(_build_env_setup(sandbox_dir_repr))
    lines.extend(_build_matplotlib_preamble())
    lines.extend(_build_execution_core(encoded_code))
    lines.extend(_build_file_tracking())
    lines.extend(_build_result_output())

    return "\n".join(lines)


def _clean_code(code: str) -> str:
    """清理代码，去除 markdown 代码块标记"""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _parse_sandbox_result(stdout: str, stderr: str, returncode: int) -> dict:
    """解析沙箱执行结果 JSON"""
    # 处理 Windows 非 UTF-8 编码导致 subprocess 返回 None 的问题
    if stdout is None:
        stdout = ""
    if stderr is None:
        stderr = ""

    start_marker = "__SANDBOX_RESULT_START__"
    end_marker = "__SANDBOX_RESULT_END__"

    result = {
        "output": "",
        "images": [],
        "files_created": [],
        "files_modified": [],
        "error": "",
        "truncated": False,
    }

    if start_marker in stdout and end_marker in stdout:
        json_start = stdout.index(start_marker) + len(start_marker)
        json_end = stdout.index(end_marker)
        json_str = stdout[json_start:json_end].strip()

        try:
            data = json.loads(json_str)
            result["output"] = data.get("output", "")
            result["truncated"] = data.get("truncated", False)
            result["images"] = data.get("images", [])
            result["files_created"] = data.get("files_created", [])
            result["files_modified"] = data.get("files_modified", [])
        except json.JSONDecodeError:
            result["output"] = f"沙箱结果解析失败\n原始输出:\n{stdout[:2000]}"
    else:
        if returncode != 0:
            result["error"] = f"沙箱进程异常退出 (代码 {returncode})"
        result["output"] = stdout[:2000]
        if stderr:
            result["output"] += f"\n[标准错误]\n{stderr[:1000]}"

    return result


def _format_sandbox_output(result: dict) -> str:
    """将沙箱结果格式化为文本"""
    output_text = result.get("output", "")
    truncated = result.get("truncated", False)
    images = result.get("images", [])
    files_created = result.get("files_created", [])
    error = result.get("error", "")

    parts = []

    if error:
        parts.append(f"⚠️ {error}")

    if not output_text.strip() and not images and not files_created:
        if not error:
            parts.append("代码执行成功，无任何控制台输出。")
    else:
        if output_text.strip():
            parts.append(output_text)

    if truncated:
        parts.append("\n输出过长已截断（限制 10000 字符）")

    # 报告文件变更
    if files_created:
        parts.append(f"\n📄 新创建的文件 ({len(files_created)}):")
        for f in files_created[:10]:
            parts.append(f"  - {f}")

    # 嵌入图片
    for img in images:
        filename = img.get("filename", "")
        img_data = img.get("data", "")
        if filename and img_data:
            ext = "jpeg" if filename.lower().endswith("jpg") else filename.lower().split(".")[-1]
            parts.append(f"\n![{filename}](data:image/{ext};base64,{img_data})\n")

    return "\n".join(parts)


def _execute_in_subprocess(sandbox_script: str, sandbox_dir: Path) -> tuple:
    """在子进程中执行沙箱脚本并返回 stdout, stderr, returncode"""
    import subprocess

    tmp_script = sandbox_dir / f"_sandbox_{uuid.uuid4().hex}.py"
    try:
        tmp_script.write_text(sandbox_script, encoding="utf-8")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # 使用 Popen 获取进程句柄以便应用 Job Object 内存限制
        proc = subprocess.Popen(
            [sys.executable, str(tmp_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(sandbox_dir),
            creationflags=creationflags,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )

        # ── Windows Job Object 内存硬限制 ──
        _apply_memory_limit_to_process(proc, max_mb=512)

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=MAX_EXECUTION_TIME)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise

        # 手动以 UTF-8 解码，错误用 replace 替换
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        return stdout, stderr, proc.returncode

    except subprocess.TimeoutExpired:
        raise  # 重新抛出，由上层处理
    finally:
        try:
            if tmp_script.exists():
                tmp_script.unlink()
        except Exception:
            pass


def _apply_memory_limit_to_process(proc, max_mb: int = 512):
    """
    为子进程设置 Windows Job Object 内存硬限制。

    当子进程内存超过 max_mb 时，Windows 内核会直接终止该进程，
    防止恶意/错误代码无限分配内存导致系统崩溃。

    仅在 Windows 上生效（Linux/macOS 可后续扩展 resource.setrlimit）。
    """
    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32

    # Job Object 信息类常量
    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = max_mb * 1024 * 1024

        kernel32.SetInformationJobObject(
            hJob,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )

        kernel32.AssignProcessToJobObject(hJob, proc._handle)
        logger.debug(f"沙箱子进程内存限制: {max_mb}MB (Job Object)")
    except Exception:
        # 非关键错误，静默忽略
        pass


def execute_python_code_safe(code: str, workspace_dir: Optional[str] = None) -> str:
    """
    在隔离的子进程中安全执行 Python 代码。

    安全机制：
    1. 子进程隔离：代码在独立 Python 子进程中执行
    2. 模块白名单：只允许导入 SAFE_MODULES 中的模块
    3. 资源限制：执行时间 <= 60 秒，输出 <= 10000 字符
    4. 文件系统隔离：在独立沙箱目录中执行
    5. 禁止危险 builtins：exec/eval/compile/open 等被移除

    Returns:
        格式化的执行结果文本（包含输出、图片、文件变更信息）
    """
    code = _clean_code(code)
    if not code:
        return "代码为空，跳过执行。"

    # 确定沙箱工作目录
    if workspace_dir is None:
        from core.utils import get_external_path
        sandbox_dir = Path(get_external_path("agent_workspace"))
    else:
        sandbox_dir = Path(workspace_dir)

    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # 记录执行前的文件快照（用于检测新文件）
    before_files = set()
    if sandbox_dir.exists():
        try:
            before_files = set(
                str(p.relative_to(sandbox_dir))
                for p in sandbox_dir.rglob("*")
                if p.is_file() and not p.name.startswith("_sandbox_")
            )
        except Exception:
            pass

    # 生成并执行沙箱脚本
    sandbox_script = _generate_sandbox_script(code, str(sandbox_dir))

    try:
        stdout, stderr, returncode = _execute_in_subprocess(sandbox_script, sandbox_dir)
        result = _parse_sandbox_result(stdout, stderr, returncode)

        # 检测新创建的文件
        try:
            after_files = set(
                str(p.relative_to(sandbox_dir))
                for p in sandbox_dir.rglob("*")
                if p.is_file() and not p.name.startswith("_sandbox_")
            )
            new_files = after_files - before_files
            # 过滤掉图片文件（已包含在 images 中）
            non_image_new = [
                f for f in sorted(new_files)
                if not f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            result["files_created"] = non_image_new
        except Exception:
            pass

        return _format_sandbox_output(result)

    except subprocess.TimeoutExpired:
        return f"代码执行超时（超过 {MAX_EXECUTION_TIME} 秒），已终止。请简化代码。"
    except Exception as e:
        logger.error(f"沙箱执行失败: {e}\n{traceback.format_exc()}")
        return f"沙箱执行器异常: {e}"


def execute_python_with_files(code: str, workspace_dir: Optional[str] = None) -> Dict:
    """
    增强版安全执行：返回结构化结果（包含输出、文件变更列表、图片等）。
    适用于需要知道代码创建了哪些文件的独立开发场景。

    Returns:
        dict: {
            "output": str,          # 控制台输出文本
            "files_created": list,  # 新创建的文件路径列表
            "images": list,         # 生成的图片 base64 数据
            "success": bool,        # 执行是否成功
            "error": str,           # 错误信息（如果有）
        }
    """
    code = _clean_code(code)
    if not code:
        return {"output": "代码为空", "files_created": [], "images": [], "success": False, "error": "代码为空"}

    if workspace_dir is None:
        from core.utils import get_external_path
        sandbox_dir = Path(get_external_path("agent_workspace"))
    else:
        sandbox_dir = Path(workspace_dir)

    sandbox_dir.mkdir(parents=True, exist_ok=True)

    before_files = set()
    if sandbox_dir.exists():
        try:
            before_files = set(
                str(p.relative_to(sandbox_dir))
                for p in sandbox_dir.rglob("*")
                if p.is_file() and not p.name.startswith("_sandbox_")
            )
        except Exception:
            pass

    sandbox_script = _generate_sandbox_script(code, str(sandbox_dir))

    try:
        stdout, stderr, returncode = _execute_in_subprocess(sandbox_script, sandbox_dir)
        result = _parse_sandbox_result(stdout, stderr, returncode)

        # 检测新创建的文件
        try:
            after_files = set(
                str(p.relative_to(sandbox_dir))
                for p in sandbox_dir.rglob("*")
                if p.is_file() and not p.name.startswith("_sandbox_")
            )
            new_files = after_files - before_files
            non_image_new = [
                f for f in sorted(new_files)
                if not f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            result["files_created"] = non_image_new
        except Exception:
            pass

        return {
            "output": result.get("output", ""),
            "files_created": result.get("files_created", []),
            "images": result.get("images", []),
            "success": returncode == 0 and not result.get("error"),
            "error": result.get("error", ""),
        }

    except subprocess.TimeoutExpired:
        return {"output": "", "files_created": [], "images": [], "success": False, "error": f"执行超时（{MAX_EXECUTION_TIME}秒）"}
    except Exception as e:
        return {"output": "", "files_created": [], "images": [], "success": False, "error": str(e)}