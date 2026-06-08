"""
模型市场、模型下载、GGUF 导出等 API 路由
"""
import json
import logging
import os
import threading
import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from core.app_state import app_state
from core.config import TrainingConfig, save_config
from core.utils import get_external_path

from .models import GGUFExportRequest

logger = logging.getLogger("ApiServer.Models")
router = APIRouter()

# 全局下载任务跟踪
_active_downloads: dict = {}
_download_tasks: dict = {}


# ======================== GGUF 导出 ========================

@router.get("/api/model/gguf_quants")
def list_gguf_quants():
    """列出可用的 GGUF 量化选项"""
    from model.gguf_exporter import list_quant_options
    return {"quants": list_quant_options()}


# POST /api/model/export_gguf 已移至 api/training/publish.py（避免重复路由）


# ======================== 模型列表与市场 ========================

@router.get("/api/models/list")
def list_all_models():
    """获取所有可用模型列表（含硬件推荐）"""
    hardware = None
    recommendations = []
    recommended_keys = set()

    try:
        from model.model_registry import (
            get_all_models, analyze_hardware, recommend_models,
            QUANT_LEVELS, estimate_file_size_mb, format_file_size,
        )
        hardware = analyze_hardware()
        recommendations = recommend_models(hardware, top_k=5)
        recommended_keys = {r.model.name for r in recommendations}
    except Exception as hw_err:
        logger.warning(f"硬件检测失败: {hw_err}")
        try:
            from model.model_registry import get_all_models, QUANT_LEVELS, estimate_file_size_mb, format_file_size
        except Exception:
            logger.error("无法导入模型注册表")
            return {"status": "error", "models": [], "recommendations": [], "error": "无法加载模型注册表"}

    try:
        gguf_dir = get_external_path("gguf_models")
        os.makedirs(gguf_dir, exist_ok=True)
        downloaded_files = {}
        if os.path.isdir(gguf_dir):
            for root, dirs, files in os.walk(gguf_dir):
                for fname in files:
                    if fname.endswith(".gguf"):
                        downloaded_files[fname] = os.path.join(root, fname)

        cache_dir = get_external_path("model_cache")
        hf_downloaded_repos = {}
        if os.path.isdir(cache_dir):
            for entry_name in os.listdir(cache_dir):
                epath = os.path.join(cache_dir, entry_name)
                if os.path.isdir(epath):
                    repo_candidates = [entry_name]
                    if "_" in entry_name:
                        repo_candidates.append(entry_name.replace("_", "/", 1))
                    resolved_path = ""
                    if os.path.exists(os.path.join(epath, "config.json")):
                        resolved_path = epath
                    else:
                        snapshots = os.path.join(epath, "snapshots")
                        if os.path.isdir(snapshots):
                            for snap in os.listdir(snapshots):
                                snap_path = os.path.join(snapshots, snap)
                                if os.path.isdir(snap_path) and os.path.exists(os.path.join(snap_path, "config.json")):
                                    resolved_path = snap_path
                                    break
                    if resolved_path:
                        for rc in repo_candidates:
                            hf_downloaded_repos[rc] = resolved_path

        models_json = []
        for entry in get_all_models():
            variants = []
            for v in entry.variants:
                quant_info = QUANT_LEVELS.get(v.quant)
                variants.append({
                    "quant": v.quant,
                    "vram_gb": v.vram_gb,
                    "is_recommended": v.is_recommended,
                    "quality_score": quant_info.quality_score if quant_info else 5,
                })
            recommended_vram = None
            recommended_quant = None
            for v in entry.variants:
                if v.is_recommended:
                    recommended_vram = v.vram_gb
                    recommended_quant = v.quant
                    break
            if recommended_vram is None and entry.variants:
                recommended_vram = entry.variants[0].vram_gb
                recommended_quant = entry.variants[0].quant

            is_downloaded = False
            downloaded_path = ""
            for v in entry.variants:
                if v.hf_filename in downloaded_files:
                    is_downloaded = True
                    downloaded_path = downloaded_files[v.hf_filename]
                    break
            if not is_downloaded:
                check_repos = [entry.hf_repo]
                if hasattr(entry, 'hf_train_repo') and entry.hf_train_repo:
                    check_repos.append(entry.hf_train_repo)
                for repo in check_repos:
                    if repo in hf_downloaded_repos:
                        is_downloaded = True
                        downloaded_path = hf_downloaded_repos[repo]
                        break

            file_size_mb = 0
            file_size_display = ""
            if recommended_quant and entry.params_b > 0:
                file_size_mb = estimate_file_size_mb(entry.params_b, recommended_quant)
                file_size_display = format_file_size(file_size_mb)

            models_json.append({
                "key": entry.hf_repo,
                "name": entry.name,
                "hf_id": entry.hf_repo,
                "family": entry.family,
                "size": entry.params_b,
                "params_b": entry.params_b,
                "quant": recommended_quant or "",
                "vram_gb": recommended_vram,
                "file_size_mb": file_size_mb,
                "file_size_display": file_size_display,
                "desc": entry.description,
                "description": entry.description,
                "tags": entry.tags,
                "variants": variants,
                "recommended": entry.name in recommended_keys,
                "downloaded": is_downloaded,
                "path": downloaded_path,
                "memory_estimate": f"~{recommended_vram:.1f}GB VRAM" if recommended_vram else "",
                "url": f"https://huggingface.co/{entry.hf_repo}" if entry.hf_repo else "",
                "hf_train_repo": entry.hf_train_repo if entry.model_type == "gguf" else "",
            })

        recs_json = []
        for r in recommendations:
            recs_json.append({
                "model_name": r.model.name,
                "quant": r.quant,
                "vram_gb": r.vram_gb,
                "score": r.score,
                "reason": r.reason,
                "detail": getattr(r, 'detail', r.reason),
                "category": getattr(r, 'category', 'gguf'),
                "can_finetune": getattr(r, 'can_finetune', False),
                "finetune_note": getattr(r, 'finetune_note', ''),
            })

        hw_info = {}
        if hardware:
            hw_info = {
                "total_ram_gb": hardware.total_ram_gb,
                "vram_gb": hardware.vram_gb,
                "gpu_backends": hardware.gpu_backends,
                "cpu_cores": hardware.cpu_cores,
                "available_memory_gb": hardware.available_memory_gb,
                "tier": getattr(hardware, 'tier', 'medium'),
            }

        return {
            "status": "ok",
            "models": models_json,
            "hardware": hw_info,
            "recommendations": recs_json,
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return {"status": "error", "models": [], "recommendations": [], "error": str(e)}


@router.get("/api/models/installed")
def list_installed_models():
    """列出已安装的本地模型"""
    try:
        from model.model_registry import get_all_models
        gguf_dir = get_external_path("gguf_models")
        os.makedirs(gguf_dir, exist_ok=True)

        installed = []
        for root, dirs, files in os.walk(gguf_dir):
            for fname in files:
                if fname.endswith(".gguf"):
                    fpath = os.path.join(root, fname)
                    size_gb = os.path.getsize(fpath) / (1024**3) if os.path.exists(fpath) else 0
                    matched_name = fname
                    for entry in get_all_models():
                        for v in entry.variants:
                            if v.hf_filename == fname:
                                matched_name = entry.name
                                break
                    installed.append({
                        "filename": fname,
                        "path": fpath,
                        "size_gb": round(size_gb, 1),
                        "model_name": matched_name,
                    })

        return {"models": installed}
    except Exception as e:
        logger.error(f"列出已安装模型失败: {e}")
        return {"models": []}


# ======================== 模型下载 ========================

@router.post("/api/models/download_hf")
async def download_hf_training(req: dict):
    """下载 HuggingFace 原版模型用于微调训练"""
    try:
        model_key = req.get("model_key", "")
        quant = req.get("quant", "Q4_K_M")
        save_dir = req.get("save_dir", "")
        if not save_dir:
            save_dir = get_external_path("model_cache")
        os.makedirs(save_dir, exist_ok=True)

        from model.model_registry import get_all_models
        from model.model_downloader import ModelDownloader, DownloadProgress

        entry = None
        for e in get_all_models():
            if e.name == model_key or e.hf_repo == model_key:
                entry = e
                break
        if not entry:
            return {"status": "error", "message": f"未找到模型: {model_key}"}

        train_repo = entry.hf_train_repo
        task_id = f"hf_{model_key}_{quant}"

        if task_id in _active_downloads:
            return {"status": "error", "message": "该模型正在下载中"}

        downloader = ModelDownloader(save_dir=save_dir, mirror=True)
        _active_downloads[task_id] = downloader
        from model.model_downloader import DownloadProgress
        init_prog = DownloadProgress(filename=f"{model_key} (HF)")
        init_prog.status = "downloading"
        _download_tasks[task_id] = {
            "model_name": model_key,
            "quant": quant,
            "filename": "HF Training Model",
            "progress": init_prog,
            "status": "downloading",
            "error_message": "",
            "file_path": "",
        }

        def on_progress(prog):
            _download_tasks[task_id]["progress"] = prog
            _download_tasks[task_id]["status"] = prog.status
            if hasattr(prog, 'downloaded_mb'):
                _download_tasks[task_id]["downloaded_mb"] = prog.downloaded_mb
                _download_tasks[task_id]["total_mb"] = prog.total_mb

        def _do():
            try:
                model_dir = downloader.download_hf_model(
                    repo_id=train_repo,
                    model_name=model_key,
                    progress_callback=on_progress,
                )
                if model_dir and downloader.progress.status == "completed":
                    _download_tasks[task_id]["status"] = "completed"
                    _download_tasks[task_id]["file_path"] = model_dir
                elif downloader.progress.status == "cancelled":
                    _download_tasks[task_id]["status"] = "cancelled"
                    _download_tasks[task_id]["error_message"] = "下载已取消"
                else:
                    _download_tasks[task_id]["status"] = "error"
            except Exception as e:
                import traceback
                _download_tasks[task_id]["status"] = "error"
                _download_tasks[task_id]["error_message"] = f"{e}\n{traceback.format_exc()}"
            finally:
                _active_downloads.pop(task_id, None)

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        return {
            "status": "ok",
            "task_id": task_id,
            "message": f"开始下载 HF 模型: {entry.name} (来自 {train_repo})",
            "repo": train_repo,
            "model_name": entry.name,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/models/download")
async def download_model(req: dict):
    """增强版模型下载端点，支持断点续传、镜像加速等"""
    try:
        model_name = req.get("model_name", "")
        quant = req.get("quant", "Q4_K_M")
        save_dir = req.get("save_dir", "")
        mirror = req.get("mirror", True)
        verify_ssl = req.get("verify_ssl", True)

        from model.model_registry import get_model_download_info, get_all_models
        from model.model_downloader import ModelDownloader, DownloadProgress

        info = get_model_download_info(model_name, quant)
        if not info:
            for entry in get_all_models():
                if entry.hf_repo == model_name:
                    for v in entry.variants:
                        if v.quant == quant:
                            info = {
                                "repo": entry.hf_repo,
                                "filename": v.hf_filename,
                                "vram_gb": v.vram_gb,
                                "parameters_b": entry.params_b,
                                "family": entry.family,
                                "description": entry.description,
                            }
                            break
                    break
        if not info:
            return {"status": "error", "message": f"未找到模型: {model_name} / {quant}"}

        if not save_dir:
            save_dir = get_external_path("gguf_models")
        os.makedirs(save_dir, exist_ok=True)

        downloader = ModelDownloader(save_dir=save_dir, mirror=mirror, verify_ssl=verify_ssl)

        task_id = f"{model_name}_{quant}"
        if task_id in _active_downloads:
            return {"status": "error", "message": "该模型正在下载中，请勿重复操作"}

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        expected_path = os.path.join(save_dir, safe_name, info["filename"])

        _active_downloads[task_id] = downloader
        _download_tasks[task_id] = {
            "model_name": model_name,
            "quant": quant,
            "filename": info["filename"],
            "progress": downloader.progress,
            "status": "downloading",
            "error_message": "",
            "file_path": expected_path,
        }

        def on_progress(prog: DownloadProgress):
            _download_tasks[task_id]["progress"] = prog
            _download_tasks[task_id]["status"] = prog.status

        def _do_download():
            try:
                is_hf_model = info.get("filename") == "full"
                if is_hf_model:
                    model_dir = downloader.download_hf_model(
                        repo_id=info["repo"],
                        model_name=model_name,
                        progress_callback=on_progress,
                    )
                    if model_dir and downloader.progress.status == "completed":
                        _download_tasks[task_id]["status"] = "completed"
                        _download_tasks[task_id]["file_path"] = model_dir
                        _download_tasks[task_id]["progress"] = downloader.progress
                    elif downloader.progress.status == "cancelled":
                        _download_tasks[task_id]["status"] = "cancelled"
                        _download_tasks[task_id]["error_message"] = "下载已取消"
                    else:
                        _download_tasks[task_id]["status"] = "error"
                else:
                    file_path = downloader.download_file(
                        repo_id=info["repo"],
                        filename=info["filename"],
                        model_name=model_name,
                        progress_callback=on_progress,
                    )
                    if file_path and downloader.progress.status == "completed":
                        _download_tasks[task_id]["status"] = "completed"
                        _download_tasks[task_id]["file_path"] = file_path
                        _download_tasks[task_id]["progress"] = downloader.progress
                        # 自动保存设置并热加载
                        settings_path = get_external_path("app_settings.json")
                        settings = {}
                        if os.path.exists(settings_path):
                            with open(settings_path, "r", encoding="utf-8") as f:
                                settings = json.load(f)
                        settings["model_type"] = "gguf"
                        settings["gguf_path"] = file_path
                        settings["n_gpu_layers"] = -1
                        settings["n_ctx"] = 2048
                        with open(settings_path, "w", encoding="utf-8") as f:
                            json.dump(settings, f, ensure_ascii=False, indent=2)
                        # 自动热加载
                        try:
                            from model.model_setup import load_gguf_model
                            from core.config import TrainingConfig, save_config
                            config = TrainingConfig()
                            config.model_type = "gguf"
                            config.gguf_path = file_path
                            config.n_gpu_layers = settings.get("n_gpu_layers", -1)
                            config.n_ctx = settings.get("n_ctx", 2048)
                            engine, tokenizer = load_gguf_model(config)
                            if engine is not None and tokenizer is not None:
                                app_state.model = engine.model if hasattr(engine, 'model') else engine
                                app_state.tokenizer = tokenizer
                                app_state._loaded_model_name = file_path
                                app_state.startup_complete = True
                                app_state._inference_engine = engine
                                app_state.update_model(
                                    engine.model if hasattr(engine, 'model') else engine,
                                    tokenizer, engine, file_path
                                )
                                save_config(config, os.path.join(get_external_path("checkpoints"), "training_config.json"))
                                logger.info(f"✅ GGUF 模型下载后自动热加载成功: {file_path}")
                        except Exception as hot_reload_err:
                            logger.warning(f"⚠️ 自动热加载失败: {hot_reload_err}")
                    elif downloader.progress.status == "cancelled":
                        _download_tasks[task_id]["status"] = "cancelled"
                        _download_tasks[task_id]["error_message"] = "下载已取消"
                    else:
                        _download_tasks[task_id]["status"] = "error"
            except Exception as e:
                import traceback
                _download_tasks[task_id]["status"] = "error"
                _download_tasks[task_id]["error_message"] = f"{e}\n{traceback.format_exc()}"
            finally:
                _active_downloads.pop(task_id, None)

        t = threading.Thread(target=_do_download, daemon=True)
        t.start()

        return {
            "status": "ok",
            "task_id": task_id,
            "message": f"开始下载 {info['filename']}",
            "filename": info["filename"],
            "quant": quant,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/models/download_cancel")
def cancel_download(req: dict):
    """取消正在进行的下载"""
    task_id = req.get("task_id", "")
    if task_id in _active_downloads:
        try:
            downloader = _active_downloads[task_id]
            downloader.cancel()
            if task_id in _download_tasks:
                _download_tasks[task_id]["status"] = "cancelled"
                _download_tasks[task_id]["error_message"] = "下载已取消"
            return {"status": "ok", "message": "下载已取消"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    if task_id in _download_tasks:
        task = _download_tasks[task_id]
        if task.get("status") not in ("completed", "cancelled"):
            task["status"] = "cancelled"
            task["error_message"] = "下载已取消（任务已结束）"
        return {"status": "ok", "message": "下载任务已清理"}
    return {"status": "error", "message": "未找到下载任务"}


@router.post("/api/models/download_pause")
def pause_download(req: dict):
    """暂停下载"""
    task_id = req.get("task_id", "")
    if task_id in _active_downloads:
        try:
            _active_downloads[task_id].pause()
            if task_id in _download_tasks:
                _download_tasks[task_id]["status"] = "paused"
            return {"status": "ok", "message": "下载已暂停"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "未找到下载任务"}


@router.post("/api/models/download_resume")
def resume_download(req: dict):
    """恢复下载"""
    task_id = req.get("task_id", "")
    if task_id in _active_downloads:
        try:
            _active_downloads[task_id].resume()
            if task_id in _download_tasks:
                _download_tasks[task_id]["status"] = "downloading"
            return {"status": "ok", "message": "下载已恢复"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "未找到下载任务"}


@router.get("/api/models/download_progress")
def get_download_progress(task_id: str = ""):
    """获取模型下载进度"""
    if task_id and task_id in _download_tasks:
        task = _download_tasks[task_id]
        prog = task.get("progress")
        status = task.get("status") or (prog.status if prog else "unknown")
        error_message = task.get("error_message") or (prog.error_message if prog else "")
        file_path = task.get("file_path", "")
        return {
            "progress": {
                "filename": prog.filename if prog else "",
                "total_mb": round(prog.total_mb, 1) if prog else 0,
                "downloaded_mb": round(prog.downloaded_mb, 1) if prog else 0,
                "percent": round(prog.percent, 1) if prog else 0,
                "speed_mbps": round(prog.speed_mbps, 2) if prog else 0,
                "eta_seconds": round(prog.eta_seconds, 0) if prog else 0,
                "status": status,
                "error_message": error_message,
                "file_path": file_path,
            }
        }
    all_tasks = {}
    for tid, task in _download_tasks.items():
        prog = task.get("progress")
        status = task.get("status") or (prog.status if prog else "unknown")
        all_tasks[tid] = {
            "model_name": task["model_name"],
            "quant": task["quant"],
            "filename": prog.filename if prog else task.get("filename", ""),
            "percent": round(prog.percent, 1) if prog else 0,
            "status": status,
        }
    return {"active_downloads": all_tasks}


@router.get("/api/models/downloaded")
def list_downloaded_models():
    """列出所有已下载的模型"""
    try:
        from model.model_downloader import list_downloaded_models
        downloaded = list_downloaded_models(save_dir=get_external_path("gguf_models"))
        gguf_dir = get_external_path("gguf_models")
        if os.path.exists(gguf_dir):
            recorded_paths = {d.get("file_path", "") for d in downloaded}
            for root, dirs, files in os.walk(gguf_dir):
                for fname in files:
                    if fname.endswith(".gguf"):
                        fpath = os.path.join(root, fname)
                        if fpath not in recorded_paths:
                            size_gb = os.path.getsize(fpath) / (1024**3)
                            downloaded.append({
                                "model_name": fname.replace(".gguf", ""),
                                "quant": "unknown",
                                "file_path": fpath,
                                "filename": fname,
                                "parameters_b": 0,
                                "family": "GGUF量化",
                                "description": "GGUF 量化模型（仅推理，不可微调训练）",
                                "model_type": "gguf",
                            })

        model_cache_dir = get_external_path("model_cache")
        if os.path.exists(model_cache_dir):
            seen_paths = {d.get("file_path", "") for d in downloaded}
            for item in os.listdir(model_cache_dir):
                item_path = os.path.join(model_cache_dir, item)
                if os.path.isdir(item_path):
                    resolved = ""
                    if os.path.exists(os.path.join(item_path, "config.json")):
                        resolved = item_path
                    else:
                        snapshots = os.path.join(item_path, "snapshots")
                        if os.path.isdir(snapshots):
                            snaps = sorted(os.listdir(snapshots), reverse=True)
                            for snap in snaps:
                                snap_path = os.path.join(snapshots, snap)
                                if os.path.isdir(snap_path) and os.path.exists(os.path.join(snap_path, "config.json")):
                                    resolved = snap_path
                                    break
                    if resolved and resolved not in seen_paths:
                        total_size = 0
                        for root, dirs, files in os.walk(resolved):
                            for f in files:
                                try:
                                    total_size += os.path.getsize(os.path.join(root, f))
                                except Exception:
                                    pass
                        size_gb = total_size / (1024**3)
                        display_name = item.replace("_", "/", 1) if "_" in item else item
                        downloaded.append({
                            "model_name": item,
                            "quant": "hf-full",
                            "file_path": resolved,
                            "filename": item,
                            "display_name": display_name,
                            "parameters_b": 0,
                            "family": "HF可微调",
                            "description": "HuggingFace 完整模型（可用于微调训练）",
                            "size_gb": round(size_gb, 2),
                            "model_type": "huggingface",
                        })
                        seen_paths.add(resolved)

        for d in downloaded:
            if d.get("size_gb"):
                continue
            fpath = d.get("file_path", "")
            if os.path.exists(fpath):
                d["size_gb"] = round(os.path.getsize(fpath) / (1024**3), 2)
            else:
                d["size_gb"] = 0

        return {"models": downloaded}
    except Exception as e:
        logger.error(f"列出已下载模型失败: {e}")
        return {"models": [], "error": str(e)}


@router.get("/api/models/info")
def get_model_info(model_name: str = ""):
    """获取单个模型的详细信息"""
    try:
        from model.model_registry import get_model_by_name, QUANT_LEVELS
        entry = get_model_by_name(model_name)
        if not entry:
            return {"status": "error", "message": f"未找到模型: {model_name}"}
        variants = []
        for v in entry.variants:
            qinfo = QUANT_LEVELS.get(v.quant)
            variants.append({
                "quant": v.quant,
                "vram_gb": v.vram_gb,
                "is_recommended": v.is_recommended,
                "quality_score": qinfo.quality_score if qinfo else 5,
                "quality_desc": qinfo.description if qinfo else "",
            })
        return {
            "name": entry.name,
            "family": entry.family,
            "params_b": entry.params_b,
            "description": entry.description,
            "tags": entry.tags,
            "variants": variants,
            "hf_repo": entry.hf_repo,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/api/models/recommend")
def get_model_recommendations(force_refresh: bool = False):
    """获取硬件感知的模型推荐"""
    try:
        from model.model_registry import (
            analyze_hardware, recommend_models,
            recommend_for_quantization_focus, QUANT_LEVELS,
        )
        hardware = analyze_hardware()
        default_recs = recommend_models(hardware, top_k=8)
        quant_recs = recommend_for_quantization_focus(hardware)

        def _format_rec(recs, max_items=8):
            items = []
            for r in recs[:max_items]:
                items.append({
                    "model_name": r.model.name,
                    "family": r.model.family,
                    "params_b": r.model.params_b,
                    "quant": r.quant,
                    "vram_gb": round(r.vram_gb, 1),
                    "score": round(r.score, 1),
                    "reason": r.reason,
                    "tags": r.model.tags,
                    "description": r.model.description,
                })
            return items

        return {
            "hardware": {
                "total_ram_gb": hardware.total_ram_gb,
                "vram_gb": round(hardware.vram_gb, 1) if hardware.vram_gb else 0,
                "available_memory_gb": round(hardware.available_memory_gb, 1),
                "gpu_backends": hardware.gpu_backends,
                "cpu_cores": hardware.cpu_cores,
                "has_nvidia_gpu": hardware.has_nvidia_gpu,
                "has_amd_gpu": hardware.has_amd_gpu,
                "has_apple_silicon": hardware.has_apple_silicon,
            },
            "recommendations": _format_rec(default_recs, 8),
            "quant_focused_recommendations": _format_rec(quant_recs, 12),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/api/models/tags")
def list_model_tags():
    """列出所有模型标签"""
    try:
        from model.model_registry import get_all_models
        tags_set = set()
        for entry in get_all_models():
            for tag in entry.tags:
                tags_set.add(tag)
        return {"tags": sorted(list(tags_set))}
    except Exception as e:
        return {"tags": [], "error": str(e)}


@router.get("/api/models/families")
def list_model_families():
    """列出所有模型家族"""
    try:
        from model.model_registry import get_all_models
        families = {}
        for entry in get_all_models():
            if entry.family not in families:
                families[entry.family] = []
            families[entry.family].append({
                "name": entry.name,
                "params_b": entry.params_b,
                "tags": entry.tags,
            })
        return {"families": families}
    except Exception as e:
        return {"families": {}, "error": str(e)}


@router.delete("/api/models/installed")
def delete_installed_model(req: dict):
    """删除已安装的模型"""
    try:
        filename = req.get("filename", "")
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return {"status": "error", "message": "无效的文件名"}
        gguf_dir = get_external_path("gguf_models")
        fpath = os.path.join(gguf_dir, filename)
        abs_fpath = os.path.abspath(fpath)
        abs_gguf = os.path.abspath(gguf_dir)
        if not abs_fpath.startswith(abs_gguf + os.sep):
            return {"status": "error", "message": "路径超出允许范围"}
        if os.path.exists(abs_fpath) and abs_fpath.endswith(".gguf"):
            os.remove(abs_fpath)
            _download_tasks.pop(filename, None)
            return {"status": "ok", "message": f"已删除 {filename}"}
        return {"status": "error", "message": "文件不存在"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/models/delete")
def delete_model_by_path(req: dict):
    """删除指定路径的模型文件"""
    try:
        path = req.get("path", "")
        if not path or not os.path.isfile(path) or not path.lower().endswith(".gguf"):
            name = req.get("name", "")
            gguf_dir = get_external_path("gguf_models")
            path = os.path.join(gguf_dir, name) if name else path
        if os.path.isfile(path) and path.lower().endswith(".gguf"):
            os.remove(path)
            _download_tasks.pop(os.path.basename(path), None)
            return {"status": "ok", "message": f"已删除 {os.path.basename(path)}"}
        if os.path.isfile(path):
            os.remove(path)
            return {"status": "ok", "message": f"已删除 {os.path.basename(path)}"}
        return {"status": "error", "message": "文件不存在或路径无效"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/models/select")
def select_model(req: dict):
    """切换当前使用的模型"""
    try:
        model_path = req.get("path", "")
        if not model_path or not os.path.exists(model_path):
            return {"status": "error", "message": "模型路径无效"}
        settings_path = get_external_path("app_settings.json")
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

        # 根据路径特征和已有配置判断模型类型
        model_type = req.get("model_type", "")
        if not model_type:
            if model_path.lower().endswith('.gguf'):
                model_type = "gguf"
            elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.pt")):
                model_type = "self"
            else:
                model_type = settings.get("model_type", "huggingface")

        settings["model_type"] = model_type
        if model_type == "gguf":
            settings["gguf_path"] = model_path
            settings["model_name"] = model_path
        else:
            settings["model_name"] = model_path
            settings["gguf_path"] = ""
        settings["n_gpu_layers"] = settings.get("n_gpu_layers", -1)
        settings["n_ctx"] = settings.get("n_ctx", 2048)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return {"status": "ok", "message": "模型已切换，重启后生效"}
    except Exception as e:
        return {"status": "error", "message": str(e)}