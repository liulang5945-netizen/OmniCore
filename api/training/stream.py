"""
流式训练 API 路由
POST /api/train/stream — 后台多线程微调，SSE 实时返回 Loss 进度
"""
import asyncio
import json
import logging
import os
import sys as _sys
import threading
import queue
import traceback
import time

import torch

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import TrainingConfig
from core.app_state import app_state
from core.utils import get_external_path

from api.models import TrainRequest
from model.model_setup import setup_model_for_training, get_optimizer, get_scheduler
from model.data_loader import InstructionDataset, create_dataloader, split_dataset
from model.trainer import Trainer, BaseInferenceEngine
from tools.file_parser import parse_file_to_text, get_ocr_diagnostic_text

from .common import safe_put, collect_hardware_diag

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/train/stream")
async def train_stream(request: TrainRequest):
    """流式训练接口，后台多线程微调，SSE 实时返回 Loss 进度"""
    if not app_state.try_start_training():
        raise HTTPException(status_code=400, detail="当前已有训练任务在运行")
    if not app_state.startup_complete or app_state.model is None:
        app_state.finish_training()
        raise HTTPException(status_code=503, detail="模型未加载完成")

    # 检查是否支持训练
    from model.gguf_engine import BaseGGUFEngine
    _raw_model = app_state.model
    if isinstance(_raw_model, BaseGGUFEngine) or (
        hasattr(_raw_model, 'model') and isinstance(_raw_model.model, BaseGGUFEngine)
    ):
        app_state.finish_training()
        raise HTTPException(
            status_code=400,
            detail="GGUF 量化模型暂不支持微调训练。请加载 HuggingFace 格式的 Transformers 模型后再试。",
        )

    async def event_generator():
        # 限制队列大小（maxsize=256），避免训练线程产出远快于 SSE 消费者时
        # 队列无限膨胀导致内存耗尽 + 训练线程被 put() 阻塞
        log_queue = queue.Queue(maxsize=256)

        def _safe_put(msg, timeout=5.0):
            safe_put(log_queue, msg, timeout)

        def train_worker():
            _config = TrainingConfig()
            _config.lora_r = request.lora_r
            _config.lora_alpha = request.lora_alpha
            _config.num_epochs = request.epochs
            _config.learning_rate = request.learning_rate
            _config.batch_size = request.batch_size
            _config.use_lora = True
            _config.validation_split = 0.1
            _config.early_stopping_patience = 2
            _config.model_name = app_state._loaded_model_name or ""

            if request.datasets and len(request.datasets) > 0:
                dataset_files = [get_external_path(os.path.join("data", dp)) for dp in request.datasets]
                _config.train_file = dataset_files[0]
            elif request.dataset:
                dataset_files = [get_external_path(os.path.join("data", request.dataset))]
                _config.train_file = dataset_files[0]
            else:
                _safe_put(json.dumps({"type": "error", "message": "未指定数据集文件"}, ensure_ascii=False))
                _safe_put("[DONE]")
                return

            original_model = app_state.model
            original_trainer = app_state.trainer
            tokenizer = app_state.tokenizer
            device_str = "cpu"
            total_files = len(dataset_files)
            overall_stopped = False
            model = None  # 初始化，防止 finally 中 NameError

            try:
                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.0,
                    "desc": f"⏳ 正在初始化训练环境（共 {total_files} 个数据集）...",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                current_device = next(original_model.parameters()).device
                device_str = str(current_device)
                # 标准化设备名：PyTorch 返回 "cuda:0"，但训练循环用 "cuda" 做 == 判断
                if device_str.startswith("cuda"):
                    device_str = "cuda"
                elif device_str.startswith("mps"):
                    device_str = "mps"

                # ── 模型+硬件双感知自适应配置（传递已加载模型以准确识别参数量） ──
                _hw_diag = _config.auto_configure_for_hardware(loaded_model=original_model)
                _decisions = _hw_diag.get('decisions', [])
                _header = f"💻 训练设备: {_hw_diag['device_name']}  |  模型: ~{_hw_diag['model_params_b']:.1f}B"
                if _decisions:
                    _message = _header + "\n" + "\n".join(_decisions)
                else:
                    _message = _header
                _safe_put(json.dumps({
                    "type": "hardware_diag",
                    "device_type": _hw_diag["device"],
                    "device_name": _hw_diag["device_name"],
                    "ram_gb": _hw_diag["ram_gb_total"],
                    "gpu_name": _hw_diag["device_name"] if _hw_diag.get("vram_gb") else None,
                    "gpu_memory_gb": _hw_diag.get("vram_gb"),
                    "message": _message,
                    "auto_config": _hw_diag,
                }, ensure_ascii=False))

                # ── 训练线程优先级 + CPU 亲和性（纯压力比驱动，无硬编码） ──
                _apply_thread_priority(_hw_diag)

                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.01,
                    "desc": "🔧 应用 LoRA 配置（遍历模块树，大模型可能需要 1-3 分钟）...",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                model = setup_model_for_training(original_model, _config)

                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.03,
                    "desc": "⚙️ 创建优化器（扫描可训练参数）...",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                optimizer = get_optimizer(model, _config)

                # ==== 数据集解析阶段 ====
                file_dataset_infos, total_accum_steps = _parse_datasets(
                    dataset_files, tokenizer, _config, _safe_put, total_files
                )

                if not file_dataset_infos:
                    _safe_put(json.dumps({
                        "type": "error",
                        "message": "❌ 所有数据集文件均为空或无法解析，无法开始训练。请检查数据文件格式（支持 .jsonl / .json / .txt / .pdf）。",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                # 创建统一的 scheduler
                scheduler = get_scheduler(optimizer, _config, max(1, total_accum_steps))
                logger.info(f"📊 训练统计: {len(file_dataset_infos)} 个有效文件, 总更新步数={total_accum_steps}")

                _safe_put(json.dumps({
                    "type": "progress",
                    "fraction": round(0.05 + 0.10, 4),
                    "desc": "🎯 数据准备完成，即将开始训练！",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                for info_idx, (dataloader, val_dataloader, filename, dataset_size, file_steps, use_val) \
                        in enumerate(file_dataset_infos):
                    if overall_stopped:
                        break

                    file_progress_start = 0.16 + info_idx * (0.84 / len(file_dataset_infos))
                    file_progress_end = 0.16 + (info_idx + 1) * (0.84 / len(file_dataset_infos))

                    file_msg = f"📂 数据集 {info_idx+1}/{len(file_dataset_infos)}: {filename}" \
                        if len(file_dataset_infos) > 1 else f"📂 数据集: {filename}"

                    if use_val and val_dataloader is not None:
                        _safe_put(json.dumps({
                            "type": "progress", "fraction": round(file_progress_start + 0.015, 4),
                            "desc": f"📊 数据集分割: 样本 {dataset_size} 条 / 验证 {_config.validation_split*100:.0f}%",
                            "loss": None, "step": 0,
                        }, ensure_ascii=False))

                    _safe_put(json.dumps({
                        "type": "progress",
                        "fraction": round(file_progress_start + 0.02, 4),
                        "desc": f"🚀 {file_msg} | Epoch 1/{_config.num_epochs} | 样本数 {dataset_size}",
                        "loss": None, "step": 0,
                    }, ensure_ascii=False))

                    _trainer = Trainer(model, optimizer, scheduler, _config, device_str)
                    if val_dataloader is not None:
                        _trainer.val_dataloader = val_dataloader
                    app_state._trainer_ref = _trainer

                    for _fraction, desc, _loss_history, _metrics in _trainer.train(dataloader):
                        if app_state.stop_training_requested:
                            _trainer.is_stopped = True
                            overall_stopped = True
                            _safe_put('{"type":"stopped","message":"用户手动终止了训练"}')
                            break

                        global_fraction = file_progress_start + _fraction * (file_progress_end - file_progress_start)
                        event_data = _build_progress_event(
                            global_fraction, desc, _metrics,
                            info_idx, len(file_dataset_infos)
                        )
                        _safe_put(json.dumps(event_data, ensure_ascii=False))

                    if overall_stopped:
                        break

                    if len(file_dataset_infos) > 1:
                        _safe_put(json.dumps({
                            "type": "progress",
                            "fraction": round(file_progress_end - 0.001, 4),
                            "desc": f"✅ 文件 {info_idx+1}/{len(file_dataset_infos)} 训练完成: {filename}",
                            "loss": None, "step": 0,
                        }, ensure_ascii=False))

                if not overall_stopped:
                    if len(file_dataset_infos) > 1:
                        _safe_put(json.dumps({
                            "type": "completed",
                            "message": f"✅ 全部 {len(file_dataset_infos)} 个数据集训练完成！",
                        }, ensure_ascii=False))
                    else:
                        _safe_put('{"type":"completed","message":"✅ 训练完成！"}')
                _safe_put("[DONE]")

            except Exception as e:
                logger.error(f"训练异常: {traceback.format_exc()}")
                _safe_put(json.dumps({
                    "type": "error",
                    "message": f"训练出错: {e}",
                }, ensure_ascii=False))
                _safe_put("[DONE]")
            finally:
                _restore_model_after_training(
                    overall_stopped, model, original_model,
                    tokenizer, original_trainer, _config, device_str
                )

        t = threading.Thread(target=train_worker, daemon=True)
        t.start()
        app_state.register_background_task(t)

        # SSE 心跳循环
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

                # 心跳机制：每 5 秒无消息时发送心跳，防止前端/代理超时断开
                if not has_message:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 50:  # 0.1s * 50 = 5s
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0

                await asyncio.sleep(0.1)
            except (GeneratorExit, RuntimeError, Exception):
                if app_state.is_training:
                    logger.info("训练客户端已断开连接，请求停止训练")
                    app_state.stop_training_requested = True
                    if app_state._trainer_ref is not None:
                        app_state._trainer_ref.is_stopped = True
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ======================== 内部辅助函数 ========================

def _apply_thread_priority(hw_diag: dict):
    """根据内存压力设置训练线程优先级和 CPU 亲和性"""
    try:
        if _sys.platform == "win32":
            import ctypes
            _model_gb = hw_diag.get('model_gb', 0)
            _optim_gb = hw_diag.get('optimizer_gb', 0)
            _headroom_gb = hw_diag.get('headroom_gb', 0)
            _total_need = _model_gb + _optim_gb + _headroom_gb
            try:
                import psutil as _ps
                _ram_total = _ps.virtual_memory().total / (1024**3)
            except Exception:
                _ram_total = max(1.0, hw_diag.get('ram_gb_total', 16))
            _pressure = _total_need / max(0.1, _ram_total)
            _curr_thread = threading.current_thread()
            THREAD_SET_INFORMATION = 0x0020
            _handle = ctypes.windll.kernel32.OpenThread(THREAD_SET_INFORMATION, False, _curr_thread.native_id)
            if _handle:
                if _pressure < 0.3:
                    logger.info(f"训练线程保持 NORMAL（压力比 {_pressure:.2f} < 0.3，资源充裕）")
                elif _pressure < 0.6:
                    THREAD_PRIORITY_BELOW_NORMAL = -1
                    ctypes.windll.kernel32.SetThreadPriority(_handle, THREAD_PRIORITY_BELOW_NORMAL)
                    logger.info(f"训练线程降为 BELOW_NORMAL（压力比 {_pressure:.2f}，0.3~0.6）")
                else:
                    THREAD_PRIORITY_IDLE = -15
                    ctypes.windll.kernel32.SetThreadPriority(_handle, THREAD_PRIORITY_IDLE)
                    logger.info(f"训练线程降为 IDLE（压力比 {_pressure:.2f} > 0.6，高压力）")
                ctypes.windll.kernel32.CloseHandle(_handle)
            try:
                _cpu_phys = hw_diag.get('cpu_physical', os.cpu_count() or 4)
                _reserved = max(1, int(_cpu_phys * 0.15))
                _train_cores = _cpu_phys - _reserved
                if _train_cores > 0 and _train_cores < _cpu_phys:
                    _affinity_mask = sum(1 << c for c in range(_train_cores))
                    _h = ctypes.windll.kernel32.OpenThread(0x0020 | 0x0100, False, _curr_thread.native_id)
                    if _h:
                        ctypes.windll.kernel32.SetThreadAffinityMask(_h, _affinity_mask)
                        ctypes.windll.kernel32.CloseHandle(_h)
                        logger.info(f"🧵 CPU 亲和性: 训练线程限制在 {_train_cores}/{_cpu_phys} 核心（预留 {_reserved} 核心给 UI）")
            except Exception as _aff_e:
                logger.debug(f"设置 CPU 亲和性失败（非关键）: {_aff_e}")
    except Exception:
        pass


def _parse_datasets(dataset_files, tokenizer, config, safe_put_fn, total_files):
    """解析所有数据集文件，返回 (file_dataset_infos, total_accum_steps)"""
    file_dataset_infos = []
    total_accum_steps = 0
    dataset_phase_start_fraction = 0.05
    dataset_phase_range = 0.10  # 5% → 15% 为数据解析阶段

    for file_idx, dataset_path in enumerate(dataset_files):
        filename = os.path.basename(dataset_path)
        file_start_frac = dataset_phase_start_fraction + file_idx * (dataset_phase_range / max(1, total_files))

        safe_put_fn(json.dumps({
            "type": "progress",
            "fraction": round(file_start_frac, 4),
            "desc": f"📂 正在解析数据集 ({file_idx+1}/{total_files}): {filename}...",
            "loss": None, "step": 0,
        }, ensure_ascii=False))

        ext = os.path.splitext(filename)[1].lower()
        raw_text = None

        if ext == ".pdf":
            def make_pdf_callback(fidx, fname, fstart_frac, ftotal):
                def pdf_cb(current_page, total_pages, page_text):
                    if current_page == 0:
                        return
                    frac = fstart_frac + (current_page / max(1, total_pages)) * (
                        (dataset_phase_range / max(1, ftotal)) * 0.95
                    )
                    desc = f"📖 OCR 识别 [{fidx+1}/{ftotal}] {fname}: {current_page}/{total_pages} 页"
                    if page_text:
                        desc += f" ({len(page_text)} 字)"
                    safe_put_fn(json.dumps({
                        "type": "progress",
                        "fraction": round(min(frac, dataset_phase_start_fraction + dataset_phase_range), 4),
                        "desc": desc,
                        "loss": None, "step": 0,
                    }, ensure_ascii=False))
                return pdf_cb

            callback = make_pdf_callback(file_idx, filename, file_start_frac, total_files)
            raw_text = parse_file_to_text(dataset_path, progress_callback=callback)
            safe_put_fn(json.dumps({
                "type": "progress",
                "fraction": round(file_start_frac + dataset_phase_range / max(1, total_files) * 0.98, 4),
                "desc": f"✅ PDF OCR 完成: {filename} ({len(raw_text)} 字符)",
                "loss": None, "step": 0,
            }, ensure_ascii=False))
        else:
            raw_text = parse_file_to_text(dataset_path)

        # 空文件检查
        if not raw_text or not raw_text.strip():
            safe_put_fn(json.dumps({
                "type": "warning",
                "message": f"⚠️ 跳过空文件: {filename}",
            }, ensure_ascii=False))
            if ext == ".pdf":
                ocr_diag = get_ocr_diagnostic_text()
                safe_put_fn(json.dumps({
                    "type": "warning",
                    "message": ocr_diag,
                }, ensure_ascii=False))
            continue

        # 创建 InstructionDataset
        try:
            dataset = InstructionDataset(
                raw_text=raw_text, file_name=filename,
                tokenizer=tokenizer, max_length=config.max_length,
                pre_tokenize=False,
            )
        except Exception as e:
            safe_put_fn(json.dumps({
                "type": "warning",
                "message": f"⚠️ 加载数据集失败 ({filename}): {e}",
            }, ensure_ascii=False))
            continue

        if len(dataset) == 0:
            safe_put_fn(json.dumps({
                "type": "warning",
                "message": f"⚠️ 文件解析后无有效数据: {filename}",
            }, ensure_ascii=False))
            continue

        # 预编码数据集
        safe_put_fn(json.dumps({
            "type": "progress",
            "fraction": round(file_start_frac + (dataset_phase_range / max(1, total_files)) * 0.005, 4),
            "desc": f"🔧 预处理 [{file_idx+1}/{total_files}] {filename} ({len(dataset)} 条)...",
            "loss": None, "step": 0,
        }, ensure_ascii=False))

        def make_preload_callback(start_fraction, fname, fidx, ftotal):
            def callback(fraction, current, total):
                global_frac = start_fraction + fraction * 0.01
                safe_put_fn(json.dumps({
                    "type": "progress",
                    "fraction": round(global_frac, 4),
                    "desc": f"🔧 Tokenize [{fidx+1}/{ftotal}] {fname}: {current}/{total}",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))
            return callback

        preload_cb = make_preload_callback(
            file_start_frac + (dataset_phase_range / max(1, total_files)) * 0.005,
            filename, file_idx, total_files
        )
        dataset.pre_tokenize_with_progress(progress_callback=preload_cb)

        safe_put_fn(json.dumps({
            "type": "progress",
            "fraction": round(file_start_frac + (dataset_phase_range / max(1, total_files)) * 0.015, 4),
            "desc": f"✅ 预处理完成 [{file_idx+1}/{total_files}] {filename}",
            "loss": None, "step": 0,
        }, ensure_ascii=False))

        # 分割训练/验证集
        use_validation = config.validation_split > 0 and len(dataset) >= 10
        if use_validation:
            train_dataset, val_dataset = split_dataset(dataset, train_ratio=1.0 - config.validation_split)
            if val_dataset is None:
                val_dataloader = None
                dataloader = create_dataloader(dataset, batch_size=config.batch_size)
            else:
                dataloader = create_dataloader(train_dataset, batch_size=config.batch_size)
                val_dataloader = create_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
        else:
            dataloader = create_dataloader(dataset, batch_size=config.batch_size)
            val_dataloader = None

        file_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
        total_accum_steps += file_steps
        file_dataset_infos.append((dataloader, val_dataloader, filename, len(dataset), file_steps, use_validation))

    return file_dataset_infos, total_accum_steps


def _build_progress_event(fraction, desc, metrics, info_idx, total_files):
    """构建 SSE 进度事件数据"""
    event_data = {
        "type": "progress",
        "fraction": round(fraction, 4),
        "desc": f"[{info_idx+1}/{total_files}] {desc}" if total_files > 1 else desc,
    }
    if metrics:
        event_data.update({
            "elapsed": round(metrics.get("elapsed", 0), 1) if metrics.get("elapsed") is not None else None,
            "eta": round(metrics["eta"], 1) if metrics.get("eta") is not None else None,
            "lr": metrics.get("lr"),
            "epoch": metrics.get("epoch"),
            "total_epochs": metrics.get("total_epochs"),
            "grad_norm": round(metrics["grad_norm"], 4) if metrics.get("grad_norm") is not None else None,
            "samples_per_sec": round(metrics["samples_per_sec"], 1) if metrics.get("samples_per_sec") is not None else None,
            "total_steps": metrics.get("total_steps"),
            "loss": metrics.get("loss"),
            "step": metrics.get("step"),
            "memory_status": metrics.get("memory_status"),
            "device_type": metrics.get("device_type"),
            "device_name": metrics.get("device_name"),
            "gpu_name": metrics.get("gpu_name"),
            "gpu_memory_gb": metrics.get("gpu_memory_gb"),
            "ram_gb": metrics.get("ram_gb"),
        })
    return event_data


def _restore_model_after_training(stopped, model, original_model, tokenizer, original_trainer, config, device_str):
    """训练结束后恢复模型状态"""
    try:
        if not stopped and model is not None:
            app_state.update_model(
                model, tokenizer,
                BaseInferenceEngine(model, config, device_str),
                app_state._loaded_model_name or "",
            )
            logger.info("✅ 训练完成，已切换到微调后的 LoRA 模型用于推理")
        else:
            app_state.update_model(
                original_model, tokenizer,
                original_trainer or BaseInferenceEngine(original_model, config, device_str),
                app_state._loaded_model_name or "",
            )
            if stopped:
                logger.info("训练被终止，已恢复原始模型用于推理")
            else:
                logger.warning("微调模型不可用，已恢复原始模型用于推理")
    except Exception as restore_err:
        logger.warning(f"恢复模型实例失败（非关键）: {restore_err}")
    finally:
        app_state.finish_training()
