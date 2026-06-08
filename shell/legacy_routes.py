"""
OmniCore 旧路由映射
将旧版 API 路由映射到态极引擎

使用方式：
    from shell.legacy_routes import setup_legacy_routes
    setup_legacy_routes(app)
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger("Taiji.Shell.Routes")
router = APIRouter()


def setup_legacy_routes(app):
    """将旧路由注册到 FastAPI 应用"""
    app.include_router(router, tags=["legacy-compat"])
    logger.info("OmniCore 兼容路由已注册")


@router.get("/api/legacy/status")
def legacy_status():
    """旧版状态检查接口"""
    from shell.compat import compat_layer
    return {
        "legacy_compat": True,
        "taiji_available": compat_layer.is_available(),
        "message": "OmniCore 兼容层运行中，推荐使用 /api/taiji/ 系列接口",
    }
