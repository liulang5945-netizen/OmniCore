"""
OmniCore FastAPI 应用核心
定义 FastAPI 应用实例、中间件、CORS、模型加载生命周期

════════════════════════════════════════════════════════════════
全局子进程窗口抑制补丁（必须在所有 import 之前生效）
防止 PyQt6 QWebEngine 嵌入式环境下任何子进程弹窗：
- subprocess.Popen / run / call / check_call / check_output
- multiprocessing.Process / spawn / forkserver
- os.system / os.popen
════════════════════════════════════════════════════════════════
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Optional

base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import TrainingConfig
from core.app_state import app_state
from core.utils import get_external_path, get_internal_path

logger = logging.getLogger("ApiServer")

# ======================== 请求频率限制中间件 ========================

_RATELIMIT_WHITELIST_PREFIXES = (
    "/api/chat/history/",
    "/api/models/download",
    "/api/models/download_cancel",
    "/api/models/download_progress",
    "/api/models/downloaded",
    "/api/settings/",
    "/api/system/",
    "/api/workspace/",
)

# ======================== JWT 认证中间件 ========================

class JWTAuthMiddleware:
    """JWT 认证中间件 (Pure ASGI)"""

    PUBLIC_PATHS = {
        "/api/auth/login",
        "/api/auth/status",
        "/api/auth/enable",
        "/api/health",
        "/",
    }
    PUBLIC_PREFIXES = ("/assets", "/workspace_data", "/ws/")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in self.PUBLIC_PATHS or any(path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await self.app(scope, receive, send)

        try:
            from core.security import AuthManager
            auth = AuthManager()
            if not auth.enabled:
                return await self.app(scope, receive, send)

            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode("utf-8")
            
            async def send_401(msg):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": json.dumps({"detail": msg}).encode("utf-8")})

            if not auth_header.startswith("Bearer "):
                return await send_401("未认证，请先登录")
                
            token = auth_header[7:]
            payload = auth.verify_token(token)
            if not payload:
                return await send_401("Token 已过期或无效")
                
            scope["state"] = scope.get("state", {})
            scope["state"]["user"] = payload
            
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"JWT 认证异常: {e}")

        return await self.app(scope, receive, send)


class RateLimitMiddleware:
    """简单内存频率限制中间件 (Pure ASGI)"""
    WINDOW_SECONDS = 60

    def __init__(self, app):
        self.app = app
        self._requests = defaultdict(list)

    @staticmethod
    def _is_whitelisted(path: str) -> bool:
        return path.startswith(_RATELIMIT_WHITELIST_PREFIXES)

    @staticmethod
    def _get_category(path: str, method: str) -> str:
        if "/api/chat/stream" in path:
            return "stream"
        elif method in ("POST", "DELETE", "PUT", "PATCH"):
            return "write"
        else:
            return "read"

    def _get_bucket_key(self, client_ip: str, path: str, method: str) -> str:
        category = self._get_category(path, method)
        return f"{client_ip}:{category}"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if self._is_whitelisted(path):
            return await self.app(scope, receive, send)
            
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        method = scope.get("method", "GET")
        
        bucket_key = self._get_bucket_key(client_ip, path, method)
        category = self._get_category(path, method)
        now = time.time()
        limits = {"stream": 100, "write": 300, "read": 600}
        max_req = limits.get(category, 600)
        
        timestamps = self._requests[bucket_key]
        timestamps[:] = [t for t in timestamps if now - t < self.WINDOW_SECONDS]
        
        if len(timestamps) >= max_req:
            retry_after = int(self.WINDOW_SECONDS - (now - timestamps[0]))
            
            async def send_429():
                await send({"type": "http.response.start", "status": 429, "headers": [(b"content-type", b"application/json"), (b"retry-after", str(retry_after).encode("utf-8"))]})
                await send({"type": "http.response.body", "body": json.dumps({"detail": f"请求过于频繁，请 {retry_after} 秒后重试", "retry_after": retry_after}).encode("utf-8")})
            
            return await send_429()

        timestamps.append(now)
        return await self.app(scope, receive, send)


# ======================== 模型加载后台任务 ========================
# 已提取到 core/model_loader.py，保持向后兼容

def _load_model_background():
    """在后台线程中加载模型（委托给 core.model_loader）"""
    from core.model_loader import load_model_on_startup
    load_model_on_startup()


def get_startup_download_progress():
    """获取启动下载进度（兼容旧接口）"""
    from core.model_loader import startup_download_progress
    return startup_download_progress


# ======================== FastAPI 应用实例 ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时在后台线程加载模型"""
    thread = threading.Thread(target=_load_model_background, daemon=True)
    thread.start()
    logger.info("模型加载已在后台线程启动")
    yield

app = FastAPI(title="OmniCore API", lifespan=lifespan)

# CORS 配置
ALLOWED_ORIGINS = os.environ.get(
    "OMNICORE_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(JWTAuthMiddleware)


# ======================== 注册路由模块 ========================

# 注意：延迟导入避免循环依赖
def _register_routers():
    from .routes_chat import router as chat_router
    from .training import router as training_router
    from .routes_rag import router as rag_router
    from .routes_models import router as models_router
    from .routes_system import router as system_router
    from .routes_settings import router as settings_router
    from .routes_update import router as update_router
    from .routes_model_switch import router as model_switch_router
    from .routes_agent import router as agent_router
    from .routes_agent_workspace import router as agent_workspace_router
    from .routes_agent_mcp import router as agent_mcp_router
    from .routes_agent_memory import router as agent_memory_router
    from .routes_terminal import router as terminal_router
    from .routes_auth import router as auth_router
    from .routes_workflows import router as workflows_router
    from .routes_plugins import router as plugins_router
    from .routes_taiji import router as taiji_router
    from .routes_taiji_model import router as taiji_model_router
    from .training.taiji_train import router as taiji_train_router

    app.include_router(auth_router)
    app.include_router(workflows_router)
    app.include_router(plugins_router)
    app.include_router(chat_router)
    app.include_router(training_router)
    app.include_router(rag_router)
    app.include_router(models_router)
    app.include_router(system_router)
    app.include_router(settings_router)
    app.include_router(update_router)
    app.include_router(model_switch_router)
    app.include_router(agent_router)
    app.include_router(agent_workspace_router)
    app.include_router(agent_mcp_router)
    app.include_router(agent_memory_router)
    app.include_router(terminal_router)
    app.include_router(taiji_router)
    app.include_router(taiji_model_router)
    app.include_router(taiji_train_router)

_register_routers()


# ======================== 挂载前端静态页面 ========================

external_dist = get_external_path("update_frontend")
internal_dist = get_internal_path(os.path.join("frontend", "dist"))

ws_dir = get_external_path("agent_workspace")
os.makedirs(ws_dir, exist_ok=True)
app.mount("/workspace_data", StaticFiles(directory=ws_dir), name="workspace_data")

if os.path.exists(os.path.join(external_dist, "index.html")):
    dist_path = external_dist
else:
    dist_path = internal_dist

if os.path.exists(dist_path):
    assets_path = os.path.join(dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{catchall:path}")
    async def serve_spa(catchall: str):
        if catchall.startswith("api/") or catchall.startswith("ws/"):
            return JSONResponse(
                {"status": "error", "message": f"端点不存在: /{catchall}"},
                status_code=404,
            )
        file_path = os.path.join(dist_path, catchall)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_path, "index.html"))


# ======================== 在 api_server.py 导入兼容 ========================
# get_startup_download_progress 已在上方定义（委托给 core.model_loader）
