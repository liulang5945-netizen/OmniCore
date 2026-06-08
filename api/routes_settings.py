"""
设置管理 API 路由
从 routes_system.py 拆分：所有 /api/settings/* 端点
"""
import json
import logging
import os

from fastapi import APIRouter, HTTPException

from core.app_state import app_state
from core.config import get_config
from core.memory_watchdog import get_memory_status_dict, force_memory_refresh
from core.utils import get_external_path

logger = logging.getLogger("ApiServer.Settings")
router = APIRouter()

SETTINGS_PATH = get_external_path("app_settings.json")


@router.get("/api/settings")
def get_all_settings():
    """获取所有设置"""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"读取设置失败: {e}")
        return {}


@router.post("/api/settings")
async def save_all_settings(req: dict):
    """保存所有设置（批量合并）"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update(req)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return {"status": "ok", "message": "设置已保存"}
    except Exception as e:
        logger.warning(f"保存设置失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置保存失败: {e}")


@router.post("/api/settings/model")
async def set_model(req: dict):
    """切换模型名称/路径"""
    model_name = req.get("model_name", "")
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="模型名称不能为空")
    settings = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    settings["model_name"] = model_name.strip()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "message": f"模型已设为 {model_name}，重启后生效"}


@router.post("/api/settings/device")
async def set_device(req: dict):
    """切换推理设备"""
    device = req.get("device", "auto")
    settings = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    settings["device"] = device
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "message": f"设备已设为 {device}，重启后生效"}


@router.post("/api/settings/quant")
async def set_quant(req: dict):
    """切换 4-bit / 8-bit 量化设置"""
    load_in_4bit = req.get("load_in_4bit", False)
    load_in_8bit = req.get("load_in_8bit", False)
    settings = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    settings["load_in_4bit"] = load_in_4bit
    settings["load_in_8bit"] = load_in_8bit
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    quant_mode = "4-bit" if load_in_4bit else "8-bit" if load_in_8bit else "无"
    return {"status": "ok", "message": f"量化模式已设为 {quant_mode}，重启后生效"}


@router.post("/api/settings/gguf")
async def set_gguf_settings(req: dict):
    """切换 GGUF 模型设置"""
    settings = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    for key in ["model_type", "gguf_path", "n_gpu_layers", "n_ctx"]:
        if key in req:
            settings[key] = req[key]
    if req.get("gguf_path"):
        settings["model_type"] = "gguf"
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "message": "GGUF 设置已保存，重启后生效"}


@router.get("/api/settings/gguf_models")
async def list_gguf_models():
    """列出可下载的 GGUF 模型"""
    try:
        from model.gguf_engine import list_available_gguf_models
        models = list_available_gguf_models()
        result = []
        for key, info in models.items():
            result.append({
                "key": key,
                "name": info["filename"],
                "size_gb": info["size_gb"],
                "description": info["description"],
            })
        return {"models": result}
    except Exception as e:
        logger.error(f"列出 GGUF 模型失败: {e}")
        return {"models": [], "error": str(e)}


@router.post("/api/settings/download_gguf")
async def download_gguf(req: dict):
    """下载 GGUF 模型（后台）"""
    model_key = req.get("model_key", "")
    if not model_key:
        raise HTTPException(status_code=400, detail="模型键名不能为空")

    from model.gguf_engine import download_gguf_model
    download_dir = get_external_path("gguf_models")

    try:
        path = download_gguf_model(model_key, download_dir)
        settings = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
        settings["model_type"] = "gguf"
        settings["gguf_path"] = path
        settings["n_gpu_layers"] = -1
        settings["n_ctx"] = 2048
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return {"status": "ok", "path": path, "message": f"模型已下载到: {path}，重启后生效"}
    except Exception as e:
        logger.error(f"下载 GGUF 模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/system/current_model")
def get_current_model():
    """返回当前实际加载的模型信息 + 待生效设置"""
    try:
        config = get_config()
        model_type = getattr(config, "model_type", None)
        gguf_path = getattr(config, "gguf_path", "")
        model_name = getattr(config, "model_name", "")
        n_gpu_layers = getattr(config, "n_gpu_layers", 0)
        device = getattr(config, "device", "auto")
        load_in_4bit = getattr(config, "load_in_4bit", False)
        load_in_8bit = getattr(config, "load_in_8bit", False)

        actual_loaded_name = getattr(app_state, '_loaded_model_name', '') or ''
        if actual_loaded_name:
            if actual_loaded_name.endswith('.gguf'):
                effective_path = actual_loaded_name
                effective_type = "gguf"
            elif model_type == "self":
                effective_path = actual_loaded_name
                effective_type = "self"
            else:
                effective_path = actual_loaded_name
                effective_type = "huggingface"
        else:
            effective_path = gguf_path or model_name
            if model_type == "self":
                effective_type = "self"
            else:
                effective_type = "gguf" if (model_type == "gguf" or gguf_path) else "huggingface"

        pending_settings = {}
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                pending_model_type = saved.get("model_type", "")
                pending_gguf_path = saved.get("gguf_path", "")
                pending_model_name = saved.get("model_name", "")
                pending_path = pending_gguf_path or pending_model_name

                is_pending = False
                if pending_model_type and pending_model_type != (model_type or ""):
                    if not (pending_model_type == "gguf" and effective_type == "gguf"):
                        if not (pending_model_type == "self" and effective_type == "self"):
                            is_pending = True
                if pending_path and pending_path != effective_path:
                    is_pending = True
                if is_pending and app_state.startup_complete and effective_path:
                    if pending_path == effective_path:
                        is_pending = False

                pending_settings = {
                    "has_pending": is_pending,
                    "pending_model_type": pending_model_type or effective_type,
                    "pending_model_path": pending_path or effective_path,
                    "pending_n_gpu_layers": saved.get("n_gpu_layers", n_gpu_layers),
                    "pending_n_ctx": saved.get("n_ctx", getattr(config, "n_ctx", 2048)),
                    "needs_restart": is_pending,
                }
            except Exception:
                pending_settings = {"has_pending": False, "needs_restart": False}
        else:
            pending_settings = {"has_pending": False, "needs_restart": False}

        return {
            "status": "ok",
            "model_type": effective_type,
            "model_path": effective_path,
            "model_name": model_name,
            "gguf_path": gguf_path,
            "device": device,
            "n_gpu_layers": n_gpu_layers,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "loaded": bool(effective_path and app_state.startup_complete),
            "pending_settings": pending_settings,
        }
    except Exception as e:
        logger.warning(f"获取当前模型信息失败: {e}")
        return {"status": "error", "message": str(e), "loaded": False}


# ======================== 内存状态 ========================

@router.get("/api/system/memory")
def get_memory_status():
    """获取当前系统内存实时状态（读取 MemoryWatchdog 单例缓存，零开销）"""
    return get_memory_status_dict()


@router.post("/api/system/memory/refresh")
def refresh_memory_status():
    """强制刷新内存状态缓存（用于关键决策点）"""
    force_memory_refresh()
    return get_memory_status_dict()