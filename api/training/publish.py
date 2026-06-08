"""
模型发布 & GGUF 导出 API 路由
POST /api/model/publish       — 合并 LoRA 权重并发布模型（SSE 流式）
GET  /api/model/published     — 列出所有已发布的模型
POST /api/model/export_gguf   — 导出 GGUF 量化格式（SSE 流式）
GET  /api/model/export_gguf/options — 返回可用的 GGUF 量化选项
"""
import asyncio
import datetime
import json as _json
import logging
import os
import threading
import queue
import time
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import TrainingConfig
from core.app_state import app_state
from core.utils import get_external_path

from model.model_setup import merge_and_save_lora_model, list_published_models

from .common import safe_put

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/model/publish")
async def publish_model():
    """合并 LoRA 权重并发布模型（SSE 流式进度）"""
    if not app_state.try_start_publishing():
        raise HTTPException(status_code=400, detail="正在发布中，请勿重复操作")
    if not app_state.startup_complete or app_state.model is None:
        app_state.finish_publishing()
        raise HTTPException(status_code=503, detail="模型未加载完成")

    async def event_generator():
        log_queue = queue.Queue(maxsize=64)
        publish_start_time = time.time()
        PUBLISH_TIMEOUT = 3600  # 1 小时超时保护
        lock_released_by_worker = False

        def _safe_put(msg, timeout=5.0):
            safe_put(log_queue, msg, timeout)

        def _progress_cb(desc: str, fraction: float):
            """merge_and_save_lora_model 进度回调"""
            _safe_put(_json.dumps({
                "type": "progress",
                "fraction": round(fraction, 4),
                "desc": desc,
            }, ensure_ascii=False))

        try:
            # 检测是否为态极模型
            is_taiji = False
            try:
                from taiji.architecture import ModelSelf
                is_taiji = isinstance(app_state.model, ModelSelf)
            except Exception:
                pass

            if is_taiji:
                yield f"data: {_json.dumps({'type':'progress','fraction':0.0,'desc':'🧬 发布态极模型...'},ensure_ascii=False)}\n\n"

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = get_external_path(f"published_models/taiji_{timestamp}")
                yield f"data: {_json.dumps({'type':'log','message':f'输出目录: {output_dir}'},ensure_ascii=False)}\n\n"

                def merge_worker():
                    nonlocal lock_released_by_worker
                    try:
                        from taiji.loader import save_model
                        import os
                        os.makedirs(output_dir, exist_ok=True)
                        save_model(app_state.model, app_state.tokenizer, output_dir)
                        if not app_state.stop_publishing_requested:
                            _safe_put(_json.dumps({
                                "type": "completed",
                                "message": f"✅ 态极模型发布成功！已保存到: {output_dir}",
                                "output_dir": output_dir,
                            }, ensure_ascii=False))
                    except Exception as e:
                        if not app_state.stop_publishing_requested:
                            logger.error(f"态极发布失败: {traceback.format_exc()}")
                            _safe_put(_json.dumps({
                                "type": "error",
                                "message": f"❌ 发布失败: {e}",
                            }, ensure_ascii=False))
                    finally:
                        app_state.finish_publishing()
                        lock_released_by_worker = True
                        _safe_put("[DONE]")
            else:
                yield f"data: {_json.dumps({'type':'progress','fraction':0.0,'desc':'开始合并 LoRA 权重并发布模型...'},ensure_ascii=False)}\n\n"

                config = TrainingConfig()
                config.model_name = app_state._loaded_model_name or config.model_name
                config.cache_dir = get_external_path("model_cache")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = get_external_path(f"published_models/omnicore_lora_{timestamp}")
                yield f"data: {_json.dumps({'type':'log','message':f'输出目录: {output_dir}'},ensure_ascii=False)}\n\n"

                def merge_worker():
                    nonlocal lock_released_by_worker
                    try:
                        merge_and_save_lora_model(config, output_dir, progress_callback=_progress_cb)
                        if not app_state.stop_publishing_requested:
                            _safe_put(_json.dumps({
                                "type": "completed",
                                "message": f"✅ 模型发布成功！已保存到: {output_dir}",
                                "output_dir": output_dir,
                            }, ensure_ascii=False))
                    except Exception as e:
                        if not app_state.stop_publishing_requested:
                            logger.error(f"发布失败: {traceback.format_exc()}")
                            _safe_put(_json.dumps({
                                "type": "error",
                                "message": f"❌ 发布失败: {e}",
                            }, ensure_ascii=False))
                    finally:
                        app_state.finish_publishing()
                        lock_released_by_worker = True
                        _safe_put("[DONE]")

            t = threading.Thread(target=merge_worker, daemon=True)
            t.start()
            app_state.register_background_task(t)

            heartbeat_counter = 0
            idle_seconds = 0
            while True:
                try:
                    has_message = False
                    while not log_queue.empty():
                        msg = log_queue.get_nowait()
                        if msg == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        yield f"data: {msg}\n\n"
                        has_message = True
                        heartbeat_counter = 0
                        idle_seconds = 0

                    if not has_message:
                        heartbeat_counter += 1
                        idle_seconds += 0.1
                        if heartbeat_counter >= 50:
                            yield ": heartbeat\n\n"
                            heartbeat_counter = 0
                        if idle_seconds >= PUBLISH_TIMEOUT:
                            logger.error("发布超时，强制终止")
                            yield f"data: {_json.dumps({'type':'error','message':'❌ 发布超时，已自动终止'},ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                    await asyncio.sleep(0.1)
                except (GeneratorExit, RuntimeError):
                    app_state.stop_publishing_requested = True
                    logger.info("发布客户端已断开连接，请求停止发布")
                    if not lock_released_by_worker:
                        app_state.finish_publishing()
                        lock_released_by_worker = True
                    return
        except Exception as e:
            if not isinstance(e, (GeneratorExit, RuntimeError)):
                logger.error(f"发布 SSE 生成器异常: {traceback.format_exc()}")
                yield f"data: {_json.dumps({'type':'error','message':f'发布过程异常: {e}'},ensure_ascii=False)}\n\n"
        finally:
            if not lock_released_by_worker:
                app_state.finish_publishing()
                lock_released_by_worker = True
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/model/published")
async def list_published():
    """列出所有已发布的模型"""
    pub_dir = get_external_path("published_models")
    return {"models": list_published_models(pub_dir)}


@router.post("/api/model/export_gguf")
async def export_gguf(request: dict = None):
    """
    将已发布的 HF 模型导出为 GGUF 量化格式（SSE 流式进度）
    请求体: {"published_dir": "...", "quant_type": "Q4_K_M"}
    """
    from model.gguf_exporter import export_published_to_gguf, list_quant_options

    published_dir = request.get("published_dir") if request else None
    quant_type = request.get("quant_type", "Q4_K_M") if request else "Q4_K_M"

    if not published_dir:
        pub_dir = get_external_path("published_models")
        models = list_published_models(pub_dir)
        if not models:
            raise HTTPException(status_code=404, detail="没有找到已发布的模型，请先发布模型")
        published_dir = models[0]["path"]
        logger.info(f"自动使用最新发布的模型: {published_dir}")

    if not os.path.isdir(published_dir):
        raise HTTPException(status_code=404, detail=f"模型目录不存在: {published_dir}")

    if quant_type not in list_quant_options():
        raise HTTPException(status_code=400, detail=f"不支持的量化类型: {quant_type}，可用: {', '.join(list_quant_options().keys())}")

    async def event_generator():
        log_queue = queue.Queue(maxsize=64)

        def _safe_put(msg, timeout=5.0):
            safe_put(log_queue, msg, timeout)

        def export_worker():
            try:
                _safe_put(_json.dumps({
                    "type": "progress",
                    "fraction": 0.0,
                    "desc": f"📤 开始导出 GGUF ({quant_type})...",
                }, ensure_ascii=False))

                def progress_cb(msg, fraction):
                    _safe_put(_json.dumps({
                        "type": "progress",
                        "fraction": round(fraction, 4),
                        "desc": msg,
                    }, ensure_ascii=False))

                result_path = export_published_to_gguf(
                    published_dir=published_dir,
                    quant_type=quant_type,
                    progress_callback=progress_cb,
                )

                file_size_gb = os.path.getsize(result_path) / (1024**3) if os.path.exists(result_path) else 0

                _safe_put(_json.dumps({
                    "type": "completed",
                    "message": f"✅ GGUF 导出完成！文件: {result_path} ({file_size_gb:.1f} GB)",
                    "file_path": result_path,
                    "file_size_gb": round(file_size_gb, 2),
                }, ensure_ascii=False))
            except Exception as e:
                logger.error(f"GGUF 导出失败: {e}")
                _safe_put(_json.dumps({
                    "type": "error",
                    "message": f"GGUF 导出失败: {e}",
                }, ensure_ascii=False))
            finally:
                _safe_put("[DONE]")

        t = threading.Thread(target=export_worker, daemon=True)
        t.start()
        app_state.register_background_task(t)

        heartbeat_counter = 0
        while True:
            try:
                has_message = False
                while not log_queue.empty():
                    msg = log_queue.get_nowait()
                    if msg == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {msg}\n\n"
                    has_message = True
                    heartbeat_counter = 0

                if not has_message:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 50:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                await asyncio.sleep(0.1)
            except (GeneratorExit, RuntimeError, Exception):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/model/export_gguf/options")
async def list_gguf_quant_options():
    """返回可用的 GGUF 量化选项"""
    from model.gguf_exporter import list_quant_options
    return {"options": list_quant_options()}
