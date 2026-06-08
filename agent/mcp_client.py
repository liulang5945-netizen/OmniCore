"""
MCP 客户端 (MCP Client)
========================
实现 Model Context Protocol (MCP) 的 JSON-RPC 通信。
通过 stdio（标准输入/输出）与 MCP 服务器进程通信。

MCP 协议核心流程：
1. 启动 MCP 服务器子进程
2. 发送 initialize 请求
3. 发送 initialized 通知
4. 调用 tools/list 获取工具列表
5. 调用 tools/call 执行工具
6. 关闭时发送 shutdown
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("MCPClient")


@dataclass
class MCPToolInfo:
    """MCP 服务器返回的工具信息"""
    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    id: str                     # 唯一标识
    name: str                   # 显示名称
    command: str                # 启动命令 (如 "cmd", "npx", "node")
    args: List[str] = field(default_factory=list)       # 命令参数
    env: Dict[str, str] = field(default_factory=dict)   # 环境变量
    enabled: bool = True
    npm_package: str = ""       # npm 包名（用于市场展示）
    description: str = ""       # 描述
    icon: str = "🧩"            # 图标
    category: str = "通用"       # 分类

    @classmethod
    def from_cline_config(cls, server_id: str, config: dict) -> "MCPServerConfig":
        """从 Cline MCP 配置格式创建"""
        return cls(
            id=server_id,
            name=server_id.split("/")[-1] if "/" in server_id else server_id,
            command=config.get("command", ""),
            args=config.get("args", []),
            env=config.get("env", {}),
            enabled=not config.get("disabled", False),
        )


class MCPClient:
    """
    MCP 协议客户端
    通过 stdio 与 MCP 服务器子进程通信
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._tools: List[MCPToolInfo] = []
        self._initialized = False
        self._lock = threading.Lock()
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._response_events: Dict[int, threading.Event] = {}
        self._responses: Dict[int, dict] = {}

    # ======================== 生命周期 ========================

    def start(self) -> bool:
        """启动 MCP 服务器子进程并初始化连接"""
        if self._process and self._process.poll() is None:
            logger.info(f"MCP 服务器 '{self.config.id}' 已在运行")
            return True

        try:
            cmd = [self.config.command] + self.config.args
            env = os.environ.copy()
            env.update(self.config.env)

            logger.info(f"启动 MCP 服务器: {' '.join(cmd)}")

            # 在 Windows 上需要 shell=True 来执行 cmd /c 命令
            use_shell = sys.platform == "win32" and self.config.command.lower() in ("cmd", "powershell")

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=use_shell,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            self._running = True

            # 启动读取线程
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            # 初始化 MCP 协议
            if not self._initialize():
                logger.error(f"MCP 服务器 '{self.config.id}' 初始化失败")
                self.stop()
                return False

            # 获取工具列表
            self._discover_tools()
            self._initialized = True

            logger.info(f"MCP 服务器 '{self.config.id}' 启动成功，发现 {len(self._tools)} 个工具")
            return True

        except FileNotFoundError as e:
            logger.error(f"MCP 服务器命令未找到: {self.config.command} - {e}")
            return False
        except Exception as e:
            logger.error(f"启动 MCP 服务器 '{self.config.id}' 失败: {e}")
            return False

    def stop(self):
        """停止 MCP 服务器"""
        self._running = False
        self._initialized = False

        if self._process and self._process.poll() is None:
            try:
                # 发送 shutdown 请求
                self._send_notification("shutdown", {})
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        self._tools.clear()
        logger.info(f"MCP 服务器 '{self.config.id}' 已停止")

    def is_running(self) -> bool:
        """检查服务器是否在运行"""
        return self._running and self._process is not None and self._process.poll() is None

    # ======================== MCP 协议 ========================

    def _initialize(self) -> bool:
        """发送 MCP initialize 请求"""
        try:
            response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                },
                "clientInfo": {
                    "name": "OmniCore-Agent",
                    "version": "1.0.0",
                },
            }, timeout=30)

            if response and "result" in response:
                # 发送 initialized 通知
                self._send_notification("notifications/initialized", {})
                return True

            logger.error(f"initialize 响应异常: {response}")
            return False
        except Exception as e:
            logger.error(f"initialize 失败: {e}")
            return False

    def _discover_tools(self):
        """获取 MCP 服务器提供的工具列表"""
        try:
            response = self._send_request("tools/list", {}, timeout=15)
            if response and "result" in response:
                tools_data = response["result"].get("tools", [])
                self._tools = []
                for t in tools_data:
                    self._tools.append(MCPToolInfo(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                    ))
                logger.info(f"发现 {len(self._tools)} 个 MCP 工具: {[t.name for t in self._tools]}")
            else:
                logger.warning(f"tools/list 响应异常: {response}")
        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")

    def call_tool(self, name: str, arguments: dict = None) -> str:
        """调用 MCP 工具"""
        if not self.is_running():
            return f"❌ MCP 服务器 '{self.config.id}' 未运行"

        try:
            response = self._send_request("tools/call", {
                "name": name,
                "arguments": arguments or {},
            }, timeout=60)

            if response and "result" in response:
                result = response["result"]
                # MCP 工具返回格式: {"content": [{"type": "text", "text": "..."}]}
                content = result.get("content", [])
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                texts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                texts.append(f"[图片: {item.get('mimeType', 'unknown')}]")
                            else:
                                texts.append(str(item))
                        else:
                            texts.append(str(item))
                    return "\n".join(texts) if texts else "✅ 工具执行完成（无文本输出）"
                elif isinstance(content, str):
                    return content
                return str(result)

            error = response.get("error", {}) if response else {}
            return f"❌ MCP 工具调用失败: {error.get('message', '未知错误')}"

        except Exception as e:
            return f"❌ MCP 工具调用异常: {e}"

    # ======================== JSON-RPC 通信 ========================

    def _send_request(self, method: str, params: dict, timeout: float = 30) -> Optional[dict]:
        """发送 JSON-RPC 请求并等待响应"""
        with self._lock:
            self._request_id += 1
            request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        event = threading.Event()
        self._response_events[request_id] = event
        self._responses.pop(request_id, None)

        try:
            self._write_message(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            self._response_events.pop(request_id, None)
            return None

        # 等待响应
        if event.wait(timeout=timeout):
            response = self._responses.pop(request_id, None)
            self._response_events.pop(request_id, None)
            return response
        else:
            logger.error(f"请求 {method} (id={request_id}) 超时 ({timeout}s)")
            self._response_events.pop(request_id, None)
            self._responses.pop(request_id, None)
            return None

    def _send_notification(self, method: str, params: dict):
        """发送 JSON-RPC 通知（无响应）"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            self._write_message(message)
        except Exception as e:
            logger.error(f"发送通知失败: {e}")

    def _write_message(self, message: dict):
        """写入消息到 stdin"""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("进程未运行")

        data = json.dumps(message, ensure_ascii=False)
        # MCP 使用 \n 分隔的 JSON-RPC
        raw = data.encode("utf-8") + b"\n"
        self._process.stdin.write(raw)
        self._process.stdin.flush()

    def _read_loop(self):
        """持续从 stdout 读取响应"""
        while self._running and self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError:
                    logger.debug(f"非 JSON 输出: {line_str[:200]}")
                    continue

                # 处理响应
                if "id" in message and message["id"] is not None:
                    msg_id = message["id"]
                    if msg_id in self._response_events:
                        self._responses[msg_id] = message
                        self._response_events[msg_id].set()

                # 处理通知（服务器主动发送的消息）
                elif "method" in message:
                    self._handle_notification(message)

            except Exception as e:
                if self._running:
                    logger.debug(f"读取循环异常: {e}")
                break

        logger.debug(f"MCP 读取循环结束: {self.config.id}")

    def _handle_notification(self, message: dict):
        """处理服务器发来的通知"""
        method = message.get("method", "")
        params = message.get("params", {})
        logger.debug(f"MCP 通知 [{self.config.id}]: {method}")

        # 处理工具列表变更通知
        if method == "notifications/tools/list_changed":
            logger.info(f"MCP 服务器 '{self.config.id}' 工具列表变更，重新获取")
            self._discover_tools()

    # ======================== 工具信息 ========================

    def get_tools(self) -> List[MCPToolInfo]:
        """获取已发现的工具列表"""
        return self._tools.copy()

    def get_tool_names(self) -> List[str]:
        """获取工具名列表"""
        return [t.name for t in self._tools]

    def get_server_info(self) -> dict:
        """获取服务器状态信息"""
        return {
            "id": self.config.id,
            "name": self.config.name,
            "running": self.is_running(),
            "initialized": self._initialized,
            "tools_count": len(self._tools),
            "tools": [{"name": t.name, "description": t.description} for t in self._tools],
            "command": f"{self.config.command} {' '.join(self.config.args)}",
            "npm_package": self.config.npm_package,
            "description": self.config.description,
            "icon": self.config.icon,
            "category": self.config.category,
        }


def create_client_from_cline_config(server_id: str, config: dict) -> MCPClient:
    """从 Cline MCP 配置创建客户端"""
    server_config = MCPServerConfig.from_cline_config(server_id, config)
    return MCPClient(server_config)


def create_client_from_market(server_id: str, market_entry: dict, workspace_path: str = "") -> MCPClient:
    """从市场配置创建客户端"""
    args = []
    template_args = market_entry.get("args_template", [])
    for arg in template_args:
        if arg == "{workspace_path}":
            args.append(workspace_path or os.getcwd())
        else:
            args.append(arg)

    config = MCPServerConfig(
        id=server_id,
        name=market_entry.get("name", server_id),
        command=market_entry.get("command", "cmd"),
        args=market_entry.get("args", []) + args,
        env=market_entry.get("env", {}),
        npm_package=market_entry.get("npm_package", ""),
        description=market_entry.get("description", ""),
        icon=market_entry.get("icon", "🧩"),
        category=market_entry.get("category", "通用"),
    )
    return MCPClient(config)