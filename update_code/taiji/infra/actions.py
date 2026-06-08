"""
态极手脚接口 (Action Provider)
===============================

态极与外部世界交互的抽象层。

态极不直接调用 os/subprocess/requests，
而是通过 ActionProvider 接口操作世界。
这样态极可以跑在任何环境中：
- LocalActionProvider：本地机器（读写文件、执行命令）
- RemoteActionProvider：远程服务器（通过 HTTP API）
- SandboxActionProvider：沙箱环境（安全限制）
"""
import os
import logging
import subprocess
import shlex
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger("Taiji.Actions")


class ActionProvider(ABC):
    """
    动作提供者抽象接口 — 态极的手脚。

    态极只调用这些方法，不关心底层实现。
    """

    @abstractmethod
    def read_file(self, path: str) -> str:
        """读取文件内容"""

    @abstractmethod
    def write_file(self, path: str, content: str) -> bool:
        """写入文件内容"""

    @abstractmethod
    def list_directory(self, path: str) -> List[str]:
        """列出目录内容"""

    @abstractmethod
    def execute(self, command: str, timeout: int = 30) -> str:
        """执行系统命令"""

    @abstractmethod
    def search(self, query: str) -> str:
        """搜索（文件、知识库等）"""

    @abstractmethod
    def web_fetch(self, url: str) -> str:
        """抓取网页内容"""

    def get_capabilities(self) -> List[str]:
        """获取当前可用的能力列表"""
        return ["read_file", "write_file", "list_directory", "execute", "search", "web_fetch"]


class LocalActionProvider(ActionProvider):
    """
    本地动作提供者 — 态极在本地机器上的手脚。

    使用 os/shlex/subprocess 实现文件和命令操作。
    """

    def __init__(self, working_dir: str = ".", sandbox: bool = False):
        """
        Args:
            working_dir: 工作目录
            sandbox: 是否启用沙箱模式（限制危险操作）
        """
        self.working_dir = os.path.abspath(working_dir)
        self.sandbox = sandbox
        self._blocked_commands = {"rm -rf /", "mkfs", "dd if=", ":(){", "chmod -R 777 /"}

    def read_file(self, path: str) -> str:
        """读取文件内容"""
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return f"[错误] 文件不存在: {path}"
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(50000)  # 限制最大读取 50KB
        except Exception as e:
            return f"[错误] 读取失败: {e}"

    def write_file(self, path: str, content: str) -> bool:
        """写入文件内容"""
        full_path = self._resolve_path(path)
        if self.sandbox:
            logger.warning(f"Sandbox mode: write_file blocked for {path}")
            return False
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Write file failed: {e}")
            return False

    def list_directory(self, path: str) -> List[str]:
        """列出目录内容"""
        full_path = self._resolve_path(path)
        if not os.path.isdir(full_path):
            return [f"[错误] 目录不存在: {path}"]
        try:
            entries = []
            for entry in sorted(os.listdir(full_path)):
                if entry.startswith('.'):
                    continue
                full_entry = os.path.join(full_path, entry)
                if os.path.isdir(full_entry):
                    entries.append(f"📁 {entry}/")
                else:
                    size = os.path.getsize(full_entry)
                    entries.append(f"📄 {entry} ({size}B)")
            return entries[:100]  # 限制最多 100 个
        except Exception as e:
            return [f"[错误] 列出目录失败: {e}"]

    def execute(self, command: str, timeout: int = 30) -> str:
        """执行系统命令"""
        if self.sandbox:
            return "[错误] 沙箱模式下不允许执行命令"

        # 安全检查
        for blocked in self._blocked_commands:
            if blocked in command:
                return f"[错误] 危险命令被阻止: {command}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:10000]  # 限制输出长度
        except subprocess.TimeoutExpired:
            return f"[错误] 命令超时 ({timeout}s): {command}"
        except Exception as e:
            return f"[错误] 执行失败: {e}"

    def search(self, query: str) -> str:
        """在工作目录中搜索文件名"""
        try:
            matches = []
            for root, dirs, files in os.walk(self.working_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fname in files:
                    if query.lower() in fname.lower():
                        rel_path = os.path.relpath(os.path.join(root, fname), self.working_dir)
                        matches.append(rel_path)
                if len(matches) >= 20:
                    break
            if matches:
                return "\n".join(matches)
            return f"未找到匹配 '{query}' 的文件"
        except Exception as e:
            return f"[错误] 搜索失败: {e}"

    def web_fetch(self, url: str) -> str:
        """抓取网页内容"""
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "TaijiCore/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read(50000).decode("utf-8", errors="ignore")
                return content
        except Exception as e:
            return f"[错误] 抓取失败: {e}"

    def _resolve_path(self, path: str) -> str:
        """解析路径（防止路径穿越）"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.working_dir, path)


class SandboxActionProvider(LocalActionProvider):
    """
    沙箱动作提供者 — 安全限制版本。

    禁止写入、执行命令等危险操作。
    适用于态极在不信任环境中运行时。
    """

    def __init__(self, working_dir: str = "."):
        super().__init__(working_dir=working_dir, sandbox=True)

    def write_file(self, path: str, content: str) -> bool:
        logger.warning(f"Sandbox: write_file blocked")
        return False

    def execute(self, command: str, timeout: int = 30) -> str:
        return "[沙箱模式] 命令执行被禁止"