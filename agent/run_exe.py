"""
本地程序运行插件（安全重构版）
让 Agent 拥有调用外部 .exe 软件或 CMD 命令的能力

⚠️ 安全设计：
- 严格的白名单命令机制，非白名单命令一律拒绝执行
- 禁止 shell=True，使用参数列表形式防止命令注入
- 可配置的安全策略，默认只允许最安全的命令
"""
import subprocess
import shlex
import sys
import logging
from typing import List, Tuple, Optional
from langchain_core.tools import Tool

logger = logging.getLogger("RunExe")

# ======================== 安全配置 ========================

# 白名单：仅允许执行以下命令（精确匹配程序名，不包含路径）
COMMAND_WHITELIST = frozenset({
    "ping", "tracert", "ipconfig", "netstat", "nslookup",
    "whoami", "systeminfo", "ver", "dir", "echo",
    "type", "find", "findstr",
    # 常用开发与网络测试工具
    "python", "pip", "node", "npm", "git", "curl", "wget", "mkdir", "md", "copy", "xcopy",
    # Java 生态
    "java", "javac", "mvn", "gradle", "gradlew",
    # Go / Rust / C/C++
    "go", "cargo", "rustc", "rustup", "gcc", "g++", "make", "cmake",
    # 容器与虚拟化
    "docker", "docker-compose",
    # 版本管理与包管理
    "pnpm", "yarn", "bun", "conda", "pip3",
    # 文本处理
    "cat", "head", "tail", "wc", "sort", "uniq", "grep", "sed", "awk",
    # 系统工具
    "env", "set", "export", "where", "which", "ls", "pwd", "cd", "tree",
    # 编辑器
    "code", "notepad",
    # 测试工具
    "pytest", "unittest", "jest", "mocha",
})

# 危险的命令模式 - 匹配命令名（非参数）
DANGEROUS_COMMANDS = frozenset({
    "rm", "del", "rd", "rmdir", "format", "diskpart",
    "reg", "regedit", "sc", "wmic", "powershell",
    "shutdown", "reboot", "taskkill",
    "net", "attrib", "cacls", "icacls", "takeown",
    "mshta", "rundll32", "certutil", "bitsadmin",
})

# 每个命令允许的最大参数数量
MAX_ARGS = 10

# 每个参数的最大长度
MAX_ARG_LENGTH = 200


def _is_safe_command(command: str) -> Tuple[bool, str]:
    """
    检查命令是否在白名单中且参数安全
    
    Returns:
        (is_safe, reason) 元组
    """
    if not command or not command.strip():
        return False, "命令为空"
    
    try:
        # 使用 shlex.split 安全解析命令行
        parts = shlex.split(command)
    except ValueError as e:
        return False, f"命令格式错误: {e}"
    
    if not parts:
        return False, "命令为空"
    
    program = parts[0].lower().strip()
    
    # 提取程序基本名称（去掉路径）
    import os
    base_program = os.path.basename(program).replace('"', '').replace("'", '').lower()
    
    # 检查是否在危险命令列表中
    if base_program in DANGEROUS_COMMANDS:
        return False, f"命令 '{base_program}' 被列入危险黑名单，已拦截"
    
    # 检查是否在白名单中
    if base_program not in COMMAND_WHITELIST:
        return False, f"命令 '{base_program}' 不在安全白名单中。\n允许的命令: {', '.join(sorted(COMMAND_WHITELIST))}"
    
    # 检查参数数量
    if len(parts) > MAX_ARGS + 1:  # +1 因为 parts[0] 是命令本身
        return False, f"参数数量过多 ({len(parts) - 1})，超出限制 ({MAX_ARGS})"
    
    # 检查参数长度
    for i, arg in enumerate(parts[1:], 1):
        if len(arg) > MAX_ARG_LENGTH:
            return False, f"参数 {i} 过长 ({len(arg)} 字符)，超出限制 ({MAX_ARG_LENGTH})"
        
        # 检查危险符号（阻止命令注入）
        dangerous_chars = {'|', ';', '&', '`', '$', '(', ')', '{', '}', '<', '>',
                          '\n', '\r'}
        if any(c in arg for c in dangerous_chars):
            return False, f"参数 {i} 包含危险字符（管道、重定向、子shell等），已拦截"
    
    return True, ""


def run_local_program(command: str) -> str:
    """
    执行本地命令行或 exe 程序（安全版本）
    
    安全措施：
    1. 严格命令白名单
    2. 危险命令黑名单
    3. 参数数量限制
    4. 单个参数长度限制
    5. 禁止危险字符注入
    """
    try:
        # 安全检查
        is_safe, reason = _is_safe_command(command)
        if not is_safe:
            logger.warning(f"安全拦截: {reason} | 命令: {command[:100]}")
            return f"⛔ 安全沙箱已拦截此命令\n原因: {reason}"
        
        # 使用 shlex.split 解析命令为参数列表（安全，无 shell 注入）
        parts = shlex.split(command)
        program = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        logger.info(f"执行白名单命令: {command[:100]}")
        
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # 使用参数列表形式调用 subprocess（不启用 shell=True）
        result = subprocess.run(
            [program] + args,
            shell=False,              # 禁用 shell 以防止命令注入
            capture_output=True,
            text=True,
            timeout=15,               # 15 秒超时保护
            creationflags=creationflags
        )
        
        # 截断过长的输出（防止输出过大导致 LLM 上下文爆炸）
        max_output = 2000
        stdout = result.stdout[:max_output]
        stderr = result.stderr[:max_output]
        
        if stdout and len(result.stdout) > max_output:
            stdout += f"\n...(输出过长已截断，原始 {len(result.stdout)} 字符)"
        if stderr and len(result.stderr) > max_output:
            stderr += f"\n...(输出过长已截断，原始 {len(result.stderr)} 字符)"
        
        if result.returncode == 0:
            return f"✅ 执行成功:\n{stdout}" if stdout else "✅ 执行成功（无输出）"
        else:
            return f"❌ 执行报错 (错误码 {result.returncode}):\n{stderr}"
            
    except FileNotFoundError:
        return f"❌ 找不到程序 '{command.split()[0] if command.strip() else ''}'，请检查路径是否正确"
    except subprocess.TimeoutExpired:
        return "⚠️ 执行超时（超过15秒），程序可能已在后台运行或卡死。"
    except Exception as e:
        logger.error(f"运行命令失败: {e}")
        return f"❌ 启动程序失败: {e}"


# 暴露给 Agent 的工具列表
TOOLS = [
    Tool(
        name="run_local_program",
        description=(
            """运行本地电脑上的安全白名单命令。
            支持的命令: ping, ipconfig, netstat, systeminfo, dir, echo 等。
            输入必须是可执行的命令字符串（例如 'ipconfig' 或 'ping 127.0.0.1'）。
            ⚠️ 仅允许执行安全白名单中的命令。"""
        ),
        func=run_local_program
    )
]
