"""
Web 终端 WebSocket 路由
======================
提供基于 WebSocket 的交互式终端会话。
支持 Windows (cmd/PowerShell) 和 Linux (bash/zsh)。

安全措施：
- JWT 认证（通过 query 参数传递 token）
- 并发终端数量限制（默认 3）
- 空闲超时自动断开（默认 300 秒）
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("ApiServer.Terminal")
router = APIRouter()

# ======================== 安全配置 ========================

MAX_CONCURRENT_TERMINALS = 3
IDLE_TIMEOUT_SECONDS = 300  # 5 分钟无输入自动断开

# 当前活跃终端计数
_active_terminals = 0
_terminals_lock = threading.Lock()


def _acquire_terminal_slot() -> bool:
    """尝试获取终端槽位，成功返回 True"""
    global _active_terminals
    with _terminals_lock:
        if _active_terminals >= MAX_CONCURRENT_TERMINALS:
            return False
        _active_terminals += 1
        return True


def _release_terminal_slot():
    """释放终端槽位"""
    global _active_terminals
    with _terminals_lock:
        _active_terminals = max(0, _active_terminals - 1)


def _verify_ws_token(ws) -> bool:
    """验证 WebSocket 连接的 JWT token（通过 query 参数）"""
    try:
        token = ws.query_params.get("token", "")
        if not token:
            logger.warning("WebSocket 终端连接被拒绝: 缺少 token")
            return False

        from core.security import AuthManager
        auth = AuthManager()
        if not auth.enabled:
            return True  # 未启用认证时放行

        payload = auth.verify_token(token)
        if not payload:
            logger.warning("WebSocket 终端连接被拒绝: token 无效或已过期")
            return False
        return True
    except ImportError:
        return True  # 安全模块不可用时放行
    except Exception as e:
        logger.warning(f"WebSocket 认证异常: {e}")
        return False


def _get_default_shell() -> tuple:
    """获取系统默认 shell 命令和参数"""
    if sys.platform == "win32":
        powershell = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        if os.path.exists(powershell):
            return powershell, ["-NoLogo", "-NoProfile"]
        return "cmd.exe", []
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, []


async def _read_stream(stream, ws: WebSocket, prefix: str):
    """从子进程流读取并发送到 WebSocket"""
    try:
        while True:
            data = await stream.read(4096)
            if not data:
                break
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = str(data)
            await ws.send_text(json.dumps({"type": prefix, "data": text}))
    except Exception:
        pass


@router.websocket("/ws/terminal")
async def terminal_websocket(ws: WebSocket):
    """
    Web 终端 WebSocket 端点（带认证和并发限制）

    客户端连接时需在 URL 中传递 token:
        ws://host/ws/terminal?token=JWT_TOKEN

    客户端消息格式:
        {"type": "input", "data": "ls -la\\n"}
        {"type": "resize", "cols": 80, "rows": 24}
        {"type": "ping"}

    服务端消息格式:
        {"type": "output", "data": "..."}
        {"type": "exit", "code": 0}
    """
    # ── 1. 认证检查 ──
    if not _verify_ws_token(ws):
        await ws.accept()
        await ws.send_text(json.dumps({
            "type": "error",
            "data": "认证失败，请先登录"
        }))
        await ws.close(code=4001, reason="Unauthorized")
        return

    # ── 2. 并发限制检查 ──
    if not _acquire_terminal_slot():
        await ws.accept()
        await ws.send_text(json.dumps({
            "type": "error",
            "data": f"终端并发数已达上限（{MAX_CONCURRENT_TERMINALS}），请关闭其他终端后重试"
        }))
        await ws.close(code=4002, reason="Too many terminals")
        return

    await ws.accept()
    logger.info("Web 终端 WebSocket 连接已建立")

    process = None
    stdin_writer = None
    last_input_time = time.time()

    try:
        shell_cmd, shell_args = _get_default_shell()

        # 获取工作目录
        from core.utils import get_external_path
        work_dir = os.getcwd()
        try:
            _settings_path = get_external_path("app_settings.json")
            if os.path.exists(_settings_path):
                with open(_settings_path, "r", encoding="utf-8") as _f:
                    _settings = json.load(_f)
                _custom = _settings.get("workspace_path", "")
                if _custom and os.path.isdir(_custom):
                    work_dir = _custom
        except Exception:
            pass

        # 创建子进程
        process = await asyncio.create_subprocess_exec(
            shell_cmd, *shell_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env={**os.environ, "TERM": "xterm-256color"},
        )

        stdin_writer = process.stdin

        # 启动 stdout 和 stderr 读取任务
        stdout_task = asyncio.create_task(_read_stream(process.stdout, ws, "output"))
        stderr_task = asyncio.create_task(_read_stream(process.stderr, ws, "output"))

        # 发送欢迎消息
        welcome = f"\r\n\x1b[32m✓ OmniCore Terminal\x1b[0m | {shell_cmd} | PID: {process.pid}\r\n"
        await ws.send_text(json.dumps({"type": "output", "data": welcome}))

        # 主循环：接收客户端消息
        async def read_ws():
            nonlocal stdin_writer, last_input_time
            while True:
                try:
                    raw = await ws.receive_text()
                    last_input_time = time.time()  # 更新最后输入时间
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "input":
                        data = msg.get("data", "")
                        if stdin_writer and not stdin_writer.is_closing():
                            stdin_writer.write(data.encode("utf-8"))
                            await stdin_writer.drain()

                    elif msg_type == "resize":
                        cols = msg.get("cols", 80)
                        rows = msg.get("rows", 24)
                        try:
                            if hasattr(process, 'transport') and hasattr(process.transport, 'get_extra_info'):
                                pty_fd = process.transport.get_extra_info('pty')
                                if pty_fd is not None:
                                    import fcntl
                                    import struct
                                    import termios
                                    winsize = struct.pack("HHHH", rows, cols, 0, 0)
                                    fcntl.ioctl(pty_fd, termios.TIOCSWINSZ, winsize)
                        except Exception:
                            pass

                    elif msg_type == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    if stdin_writer and not stdin_writer.is_closing():
                        stdin_writer.write(raw.encode("utf-8"))
                        await stdin_writer.drain()

        # 空闲超时检查
        async def idle_watchdog():
            while True:
                await asyncio.sleep(30)
                idle_seconds = time.time() - last_input_time
                if idle_seconds > IDLE_TIMEOUT_SECONDS:
                    logger.warning(f"终端空闲超时 ({idle_seconds:.0f}s)，自动断开")
                    try:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "data": f"空闲超过 {IDLE_TIMEOUT_SECONDS} 秒，终端已自动断开"
                        }))
                    except Exception:
                        pass
                    if process and process.returncode is None:
                        process.terminate()
                    return

        ws_task = asyncio.create_task(read_ws())
        watchdog_task = asyncio.create_task(idle_watchdog())

        # 等待进程结束或 WebSocket 断开
        done, pending = await asyncio.wait(
            [ws_task, stdout_task, stderr_task, asyncio.create_task(process.wait()), watchdog_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        exit_code = process.returncode if process.returncode is not None else -1
        try:
            await ws.send_text(json.dumps({"type": "exit", "code": exit_code}))
        except Exception:
            pass

    except WebSocketDisconnect:
        logger.info("Web 终端 WebSocket 客户端断开")
    except Exception as e:
        logger.error(f"Web 终端 WebSocket 异常: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "data": "终端异常断开"}))
        except Exception:
            pass
    finally:
        _release_terminal_slot()
        if process and process.returncode is None:
            try:
                if sys.platform == "win32":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(process.wait(), timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        logger.info("Web 终端会话结束")