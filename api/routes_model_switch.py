"""
模型热切换 API 路由
从 routes_system.py 拆分：模型切换、GGUF 下载、发布状态重置
"""
import json
import logging
import os
import sys
import threading

from fastapi import APIRouter, HTTPException

from core.app_state import app_state
from core.config import TrainingConfig, save_config
from core.memory_watchdog import MemoryWatchdog, force_memory_refresh
from core.utils import get_external_path

logger = logging.getLogger("ApiServer.ModelSwitch")
router = APIRouter()

SETTINGS_PATH = get_external_path("app_settings.json")

# 模型切换并发锁（防止多次点击重复切换）
_switch_lock = threading.Lock()
_switch_thread = None


@router.post("/api/system/reload_model")
def reload_model():
    """下载模型后立即热加载（兼容旧接口，内部委托 switch_model）"""
    settings = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    gguf_path = settings.get("gguf_path", "")
    model_name = settings.get("model_name", "")
    model_type = settings.get("model_type", "gguf" if gguf_path else "huggingface")
    return _do_switch_model(model_type, gguf_path, model_name)


@router.post("/api/system/switch_model")
def switch_model(req: dict):
    """统一模型热切换端点，支持 GGUF 和 HuggingFace 模型。
    
    改为异步执行：立即返回，在后台线程中加载模型。
    前端通过 GET /api/system/switch_status 轮询切换状态。
    """
    global _switch_thread

    model_type = req.get("model_type", "")
    gguf_path = req.get("gguf_path", "")
    model_name = req.get("model_name", "")

    if not model_type:
        return {"status": "error", "message": "缺少 model_type 参数（gguf、huggingface 或 self）"}

    # 并发保护：防止多次点击重复切换
    if not _switch_lock.acquire(blocking=False):
        current_status = app_state.get_switch_status()
        return {
            "status": "switching_in_progress",
            "message": f"正在切换模型中（{current_status.get('message', '加载中...')}），请勿重复操作",
        }

    try:
        current_status = app_state.get_switch_status()
        if current_status["status"] == "switching":
            _switch_lock.release()
            return {
                "status": "switching_in_progress",
                "message": f"正在切换模型中（{current_status.get('message', '加载中...')}），请勿重复操作",
            }

        model_display = os.path.basename(gguf_path or model_name) or model_type
        app_state.update_switch_status("switching", f"开始切换模型: {model_display}")

        def _do_switch_async():
            """后台线程执行的异步模型切换"""
            import traceback
            settings_path = SETTINGS_PATH
            try:
                # ── 0. 内存预检 ──
                app_state.update_switch_status("switching", "正在检查内存状态...")
                force_memory_refresh()
                wd = MemoryWatchdog()
                st = wd.status

                if st.avail_pct < 0.15:
                    app_state.update_switch_status(
                        "error", "",
                        f"内存不足（可用 {st.avail_gb:.1f}GB / {st.total_gb:.1f}GB），"
                        f"请关闭其他应用后重试"
                    )
                    return

                # ── 1. 保存新配置 ──
                app_state.update_switch_status("switching", "正在保存配置...")
                settings = {}
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = json.load(f)

                settings["model_type"] = model_type

                if model_type == "gguf":
                    if not gguf_path or not os.path.exists(gguf_path):
                        app_state.update_switch_status("error", "", f"GGUF 模型文件不存在: {gguf_path}")
                        return
                    settings["gguf_path"] = gguf_path
                    settings["model_name"] = gguf_path
                else:
                    if not model_name:
                        app_state.update_switch_status("error", "", "缺少 model_name 参数")
                        return
                    settings["model_name"] = model_name
                    settings["gguf_path"] = ""

                settings["n_gpu_layers"] = settings.get("n_gpu_layers", -1)
                settings["n_ctx"] = settings.get("n_ctx", 2048)

                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
                logger.info(f"📝 配置已保存: model_type={model_type}, path={gguf_path or model_name}")

                # ── 2. 卸载旧模型 ──
                app_state.update_switch_status("switching", "正在卸载旧模型...")
                app_state.unload_model()
                logger.info("🗑️ 旧模型已释放")

                # ── 2.5 释放后再次内存预检 ──
                import gc
                gc.collect()
                force_memory_refresh()
                st2 = MemoryWatchdog().status
                if st2.avail_pct < 0.25:
                    err_msg = (
                        f"释放旧模型后内存仍不足（可用 {st2.avail_gb:.1f}GB），"
                        f"请重启应用后重试"
                    )
                    app_state.update_switch_status("error", "", err_msg)
                    app_state.mark_startup_failed(err_msg)
                    return

                # ── 3. 加载新模型 ──
                app_state.update_switch_status("switching", "正在加载新模型（可能需要数分钟）...")
                from model.gguf_engine import is_gguf_model, find_gguf_file
                from model.model_setup import load_gguf_model, download_and_load_model
                from model.trainer import BaseInferenceEngine

                config = TrainingConfig()
                config.cache_dir = get_external_path("model_cache")
                config.model_type = model_type
                config.n_gpu_layers = settings.get("n_gpu_layers", -1)
                config.n_ctx = settings.get("n_ctx", 2048)

                if model_type == "gguf":
                    config.gguf_path = gguf_path
                    app_state.update_switch_status("switching", "正在加载 GGUF 模型...")
                    engine, tokenizer = load_gguf_model(config)
                    if engine is None or tokenizer is None:
                        app_state.update_switch_status("error", "", "GGUF 模型加载失败，请检查文件完整性")
                        app_state.mark_startup_failed("GGUF 模型加载失败")
                        return
                    app_state.update_model(
                        engine.model if hasattr(engine, 'model') else engine,
                        tokenizer, engine, gguf_path
                    )
                    logger.info(f"✅ GGUF 模型热切换成功: {os.path.basename(gguf_path)}")
                elif model_type == "self":
                    app_state.update_switch_status("switching", "正在加载 ModelSelf 原生模型...")
                    _load_self_model_switch(config, model_name or gguf_path)
                    logger.info(f"✅ ModelSelf 模型热切换成功: {model_name or gguf_path}")
                else:
                    config.model_name = model_name
                    if os.path.isdir(model_name):
                        config.resume_from_checkpoint = get_external_path("final_checkpoint.pt")
                    config.device = config.resolve_device()
                    app_state.update_switch_status("switching", "正在加载 HuggingFace 模型...")
                    model, tokenizer = download_and_load_model(config)
                    device = config.resolve_device()
                    trainer = BaseInferenceEngine(model, config, device)
                    app_state.update_model(model, tokenizer, trainer, model_name)
                    logger.info(f"✅ HF 模型热切换成功: {model_name}")

                # 模型加载成功后立即标记启动完成，清除可能残留的 startup_error
                app_state.mark_started()
                save_config(config, os.path.join(get_external_path("checkpoints"), "training_config.json"))

                app_state.update_switch_status("success", f"模型切换成功: {os.path.basename(gguf_path or model_name)}")
                logger.info(f"✅ 模型热切换成功: {model_display}")

            except Exception as e:
                logger.error(f"模型热切换失败: {traceback.format_exc()}")
                app_state.mark_startup_failed(str(e))
                app_state.update_switch_status("error", "", f"切换失败: {str(e)}")
            finally:
                _switch_lock.release()
                logger.info("🔓 模型切换锁已释放")

        _switch_thread = threading.Thread(target=_do_switch_async, daemon=True)
        _switch_thread.start()

        return {
            "status": "ok",
            "message": f"开始切换模型: {model_display}",
            "model_type": model_type,
        }
    except Exception as e:
        _switch_lock.release()
        return {"status": "error", "message": f"启动切换失败: {str(e)}"}


@router.get("/api/system/switch_status")
def get_switch_status():
    """获取当前模型切换进度状态（供前端轮询）"""
    switch_state = app_state.get_switch_status()
    return {
        "status": switch_state["status"],
        "message": switch_state["message"],
        "error": switch_state["error"],
    }


@router.post("/api/system/pub_reset")
def force_reset_publishing():
    """强制重置模型发布状态（修复死锁 API）"""
    result = app_state.force_reset_publishing()
    return {"status": "ok", **result}


def _do_switch_model(model_type: str, gguf_path: str, model_name: str):
    """同步执行模型切换（供 reload_model 等旧接口使用）"""
    import traceback

    settings_path = SETTINGS_PATH
    try:
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

        settings["model_type"] = model_type

        if model_type == "gguf":
            if not gguf_path or not os.path.exists(gguf_path):
                return {"status": "error", "message": f"GGUF 模型文件不存在: {gguf_path}"}
            settings["gguf_path"] = gguf_path
            settings["model_name"] = gguf_path
        else:
            if not model_name:
                return {"status": "error", "message": "缺少 model_name 参数"}
            settings["model_name"] = model_name
            settings["gguf_path"] = ""

        settings["n_gpu_layers"] = settings.get("n_gpu_layers", -1)
        settings["n_ctx"] = settings.get("n_ctx", 2048)

        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 配置已保存: model_type={model_type}, path={gguf_path or model_name}")

        app_state.unload_model()
        logger.info("🗑️ 旧模型已释放")

        from model.gguf_engine import is_gguf_model, find_gguf_file
        from model.model_setup import load_gguf_model, download_and_load_model
        from model.trainer import BaseInferenceEngine

        config = TrainingConfig()
        config.cache_dir = get_external_path("model_cache")
        config.model_type = model_type
        config.n_gpu_layers = settings.get("n_gpu_layers", -1)
        config.n_ctx = settings.get("n_ctx", 2048)

        if model_type == "gguf":
            config.gguf_path = gguf_path
            engine, tokenizer = load_gguf_model(config)
            if engine is None or tokenizer is None:
                app_state.mark_startup_failed("GGUF 模型加载失败")
                return {"status": "error", "message": "GGUF 模型加载失败，请检查文件完整性"}
            app_state.update_model(
                engine.model if hasattr(engine, 'model') else engine,
                tokenizer, engine, gguf_path
            )
            logger.info(f"✅ GGUF 模型热切换成功: {os.path.basename(gguf_path)}")
        elif model_type == "self":
            _load_self_model_switch(config, model_name or gguf_path)
            logger.info(f"✅ ModelSelf 模型热切换成功: {model_name or gguf_path}")
        else:
            config.model_name = model_name
            if os.path.isdir(model_name):
                config.resume_from_checkpoint = get_external_path("final_checkpoint.pt")
            config.device = config.resolve_device()

            model, tokenizer = download_and_load_model(config)
            device = config.resolve_device()
            trainer = BaseInferenceEngine(model, config, device)
            app_state.update_model(model, tokenizer, trainer, model_name)
            logger.info(f"✅ HF 模型热切换成功: {model_name}")

        app_state.mark_started()
        save_config(config, os.path.join(get_external_path("checkpoints"), "training_config.json"))

        return {
            "status": "ok",
            "message": f"模型切换成功: {os.path.basename(gguf_path or model_name)}",
            "model_type": model_type,
            "model_name": app_state._loaded_model_name,
        }

    except Exception as e:
        logger.error(f"模型热切换失败: {traceback.format_exc()}")
        app_state.mark_startup_failed(str(e))
        return {"status": "error", "message": f"切换失败: {str(e)}"}


def _load_self_model_switch(config: TrainingConfig, model_path: str):
    """ModelSelf 模型加载（态极已剥离，此功能不可用）"""
    app_state.update_switch_status("error", "", "ModelSelf 模型需要态极引擎支持，当前版本不包含态极模块")
