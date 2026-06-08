"""
断点续训 API 路由
POST /api/train/resume_checkpoint — 从检查点恢复训练（SSE 流式）

同时包含检查点管理:
GET  /api/train/checkpoints — 扫描检查点列表
"""
import asyncio
import glob
import json
import logging
import os
import threading
import queue
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import TrainingConfig
from core.app_state import app_state
from core.utils import get_external_path

from model.model_setup import setup_model_for_training, get_optimizer, get_scheduler, load_checkpoint
from model.data_loader import InstructionDataset, create_dataloader, split_dataset
from model.trainer import Trainer, BaseInferenceEngine
from tools.file_parser import parse_file_to_text

from .common import safe_put
from .stream import _parse_datasets, _build_progress_event

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.get("/api/train/checkpoints")
def list_checkpoints():
    import torch
    """扫描检查点目录，返回已有的检查点列表"""
    try:
        checkpoint_dir = get_external_path("checkpoints")
        if not os.path.isdir(checkpoint_dir):
            return {"checkpoints": []}

        pattern = os.path.join(checkpoint_dir, "checkpoint-e*.pt")
        ckpt_files = sorted(glob.glob(pattern))

        checkpoints = []
        for ckpt_path in ckpt_files:
            try:
                # 只读取元信息，不加载完整权重
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                basename = os.path.basename(ckpt_path)
                info = {
                    "filename": basename,
                    "path": ckpt_path,
                    "epoch": state.get("epoch", -1) + 1,  # 转换为 1-based
                    "step": state.get("step", 0),
                    "loss": state.get("loss", None),
                    "dataset_files": state.get("dataset_files", []),
                    "num_epochs": state.get("config", {}).get("num_epochs", "?"),
                }
                checkpoints.append(info)
            except Exception as e:
                logger.warning(f"读取检查点元信息失败 {ckpt_path}: {e}")
                checkpoints.append({
                    "filename": os.path.basename(ckpt_path),
                    "error": str(e),
                })

        # 按 step 降序排列（最新的在前）
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        return {"checkpoints": checkpoints, "count": len(checkpoints)}
    except Exception as e:
        logger.error(f"扫描检查点失败: {e}")
        return {"checkpoints": [], "error": str(e)}


@router.post("/api/train/resume_checkpoint")
async def resume_checkpoint(request: dict = None):
    """
    从检查点恢复训练（SSE 流式）
    接收 JSON body:
    {
        "checkpoint": "checkpoint-e2-s800.pt",     # 可选，默认为最新
        "datasets": ["file1.jsonl", ...]            # 可选，覆盖保存的数据集列表
    }
    """
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

    # 解析请求参数
    checkpoint_dir = get_external_path("checkpoints")
    checkpoint_filename = request.get("checkpoint") if request else None
    override_datasets = request.get("datasets") if request else None

    # 找到指定的检查点或最新的检查点
    if checkpoint_filename:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename)
        if not os.path.exists(ckpt_path):
            app_state.finish_training()
            raise HTTPException(status_code=404, detail=f"检查点文件不存在: {checkpoint_filename}")
    else:
        pattern = os.path.join(checkpoint_dir, "checkpoint-e*.pt")
        ckpt_files = sorted(glob.glob(pattern))
        if not ckpt_files:
            app_state.finish_training()
            raise HTTPException(status_code=404, detail="未找到任何检查点文件，无法恢复训练")
        ckpt_path = ckpt_files[-1]  # 最新的

    async def event_generator():
        log_queue = queue.Queue(maxsize=256)

        def _safe_put(msg, timeout=5.0):
            safe_put(log_queue, msg, timeout)

        def resume_worker():
            nonlocal checkpoint_filename, override_datasets
            import torch as _torch

            try:
                # 1. 读取检查点元信息
                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.0,
                    "desc": f"📂 读取检查点: {os.path.basename(ckpt_path)}...",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                ckpt_state = _torch.load(ckpt_path, map_location="cpu", weights_only=True)
                saved_epoch = ckpt_state.get("epoch", 0)          # 0-based
                saved_step = ckpt_state.get("step", 0)
                saved_loss = ckpt_state.get("loss", float("inf"))
                saved_config_dict = ckpt_state.get("config", {})
                saved_dataset_files = ckpt_state.get("dataset_files", [])

                logger.info(f"检查点信息: epoch={saved_epoch+1}, step={saved_step}, loss={saved_loss:.4f}, dataset_files={saved_dataset_files}")

                # 2. 创建配置（从检查点恢复配置参数）
                _config = TrainingConfig()
                if saved_config_dict:
                    for key, val in saved_config_dict.items():
                        if hasattr(_config, key) and key not in ("_hw_diag",):
                            try:
                                setattr(_config, key, val)
                            except Exception:
                                pass
                _config.use_lora = True
                _config.model_name = app_state._loaded_model_name or _config.model_name

                # 3. 确定数据集文件
                if override_datasets:
                    dataset_files = [get_external_path(os.path.join("data", dp)) for dp in override_datasets]
                elif saved_dataset_files:
                    dataset_files = saved_dataset_files
                else:
                    _safe_put(json.dumps({
                        "type": "error",
                        "message": "检查点中没有保存数据集文件列表，请在前端选择数据集后重试",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                dataset_files = [f for f in dataset_files if os.path.exists(f)]
                if not dataset_files:
                    _safe_put(json.dumps({
                        "type": "error",
                        "message": "数据集文件不存在，请重新上传数据集",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                total_files = len(dataset_files)
                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.01,
                    "desc": f"✅ 已读取检查点: Epoch {saved_epoch+1}, Step {saved_step}, "
                            f"Loss={saved_loss:.4f} | 从 {total_files} 个文件恢复训练",
                    "loss": None, "step": saved_step,
                }, ensure_ascii=False))

                original_model = app_state.model
                original_trainer = app_state.trainer
                tokenizer = app_state.tokenizer
                device_str = "cpu"
                overall_stopped = False
                model = None

                current_device = next(original_model.parameters()).device
                device_str = str(current_device)
                if device_str.startswith("cuda"):
                    device_str = "cuda"
                elif device_str.startswith("mps"):
                    device_str = "mps"

                # 4. 硬件自适应配置
                _hw_diag = _config.auto_configure_for_hardware(loaded_model=original_model)

                # 5. 应用 LoRA
                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.03,
                    "desc": "🔧 应用 LoRA 配置...",
                    "loss": None, "step": saved_step,
                }, ensure_ascii=False))

                model = setup_model_for_training(original_model, _config)

                # 6. 加载检查点权重
                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.05,
                    "desc": f"📥 加载检查点权重 (Step {saved_step})...",
                    "loss": None, "step": saved_step,
                }, ensure_ascii=False))

                optimizer = get_optimizer(model, _config)

                # 7. 重新解析数据集
                file_dataset_infos, total_remaining_steps = _parse_datasets_for_resume(
                    dataset_files, tokenizer, _config, _safe_put, total_files,
                    saved_epoch, saved_step
                )

                if not file_dataset_infos:
                    _safe_put(json.dumps({
                        "type": "error",
                        "message": "❌ 所有数据集文件均为空或无法解析。",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                # 检查是否还有剩余训练步数
                if total_remaining_steps == 0:
                    _safe_put(json.dumps({
                        "type": "warning",
                        "message": f"⚠️ 该检查点（Epoch {saved_epoch+1}/{_config.num_epochs}）已完成全部训练。"
                                   f"\n💡 如需继续训练，请在参数设置中增加 epoch 数后点击「开始训练」开启新训练。"
                                   f"\n💡 或选择更早的检查点进行恢复。",
                    }, ensure_ascii=False))
                    _safe_put(json.dumps({
                        "type": "completed",
                        "message": f"✅ 检查点训练已全部完成（共 {_config.num_epochs} 个 epoch），无需继续训练。"
                                   f"\n📋 如需发布模型，请使用「强制发布」功能。",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                # 8. 创建 scheduler 并加载检查点
                total_steps = total_remaining_steps + saved_step
                scheduler = get_scheduler(optimizer, _config, max(1, total_steps))

                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.20,
                    "desc": "🔄 恢复优化器和调度器状态...",
                    "loss": None, "step": saved_step,
                }, ensure_ascii=False))

                try:
                    load_checkpoint(model, optimizer, scheduler, ckpt_path, device_str)
                except Exception as e:
                    _safe_put(json.dumps({
                        "type": "warning",
                        "message": f"⚠️ 加载优化器/调度器状态失败: {e}（将继续使用新状态）",
                    }, ensure_ascii=False))

                logger.info(f"📊 断点续训统计: {len(file_dataset_infos)} 个有效文件, "
                            f"从 epoch {saved_epoch+1}/{_config.num_epochs} 继续, "
                            f"剩余步数={total_remaining_steps}")

                _safe_put(json.dumps({
                    "type": "progress", "fraction": 0.22,
                    "desc": f"🎯 准备就绪！从 Epoch {saved_epoch+1}/{_config.num_epochs} 继续训练",
                    "loss": None, "step": saved_step,
                }, ensure_ascii=False))

                # 9. 开始训练（从保存的 epoch 继续）
                for info_idx, (dataloader, val_dataloader, filename, dataset_size, file_steps, use_val) \
                        in enumerate(file_dataset_infos):
                    if overall_stopped:
                        break

                    file_progress_start = 0.24 + info_idx * (0.76 / len(file_dataset_infos))
                    file_progress_end = 0.24 + (info_idx + 1) * (0.76 / len(file_dataset_infos))

                    _trainer = Trainer(model, optimizer, scheduler, _config, device_str)
                    if val_dataloader is not None:
                        _trainer.val_dataloader = val_dataloader
                    app_state._trainer_ref = _trainer

                    start_epoch = saved_epoch + 1

                    for _fraction, desc, _loss_history, _metrics in _trainer.train(dataloader, start_epoch=start_epoch):
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

                if not overall_stopped:
                    _safe_put('{"type":"completed","message":"✅ 恢复训练完成！"}')
                _safe_put("[DONE]")

            except Exception as e:
                logger.error(f"恢复训练异常: {traceback.format_exc()}")
                _safe_put(json.dumps({
                    "type": "error",
                    "message": f"恢复训练出错: {e}",
                }, ensure_ascii=False))
                _safe_put("[DONE]")
            finally:
                try:
                    app_state.update_model(
                        original_model if overall_stopped else model,
                        tokenizer,
                        original_trainer or BaseInferenceEngine(original_model, _config, device_str),
                        app_state._loaded_model_name or "",
                    )
                except Exception as restore_err:
                    logger.warning(f"恢复模型实例失败: {restore_err}")
                finally:
                    app_state.finish_training()

        t = threading.Thread(target=resume_worker, daemon=True)
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

                if not has_message:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 50:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                await asyncio.sleep(0.1)
            except (GeneratorExit, RuntimeError, Exception):
                if app_state.is_training:
                    logger.info("恢复训练客户端已断开连接，请求停止训练")
                    app_state.stop_training_requested = True
                    if app_state._trainer_ref is not None:
                        app_state._trainer_ref.is_stopped = True
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _parse_datasets_for_resume(dataset_files, tokenizer, config, safe_put_fn, total_files, saved_epoch, saved_step):
    """解析数据集并计算剩余训练步数（用于断点续训）"""
    file_dataset_infos = []
    total_remaining_steps = 0
    dataset_phase_start_fraction = 0.08
    dataset_phase_range = 0.12

    for file_idx, dataset_path in enumerate(dataset_files):
        filename = os.path.basename(dataset_path)
        file_start_frac = dataset_phase_start_fraction + file_idx * (dataset_phase_range / max(1, total_files))

        safe_put_fn(json.dumps({
            "type": "progress",
            "fraction": round(file_start_frac, 4),
            "desc": f"📂 重新解析数据集 ({file_idx+1}/{total_files}): {filename}...",
            "loss": None, "step": saved_step,
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
                        "loss": None, "step": saved_step,
                    }, ensure_ascii=False))
                return pdf_cb

            callback = make_pdf_callback(file_idx, filename, file_start_frac, total_files)
            raw_text = parse_file_to_text(dataset_path, progress_callback=callback)
        else:
            raw_text = parse_file_to_text(dataset_path)

        if not raw_text or not raw_text.strip():
            safe_put_fn(json.dumps({
                "type": "warning",
                "message": f"⚠️ 跳过空文件: {filename}",
            }, ensure_ascii=False))
            continue

        dataset = InstructionDataset(
            raw_text=raw_text, file_name=filename,
            tokenizer=tokenizer, max_length=config.max_length,
            pre_tokenize=False,
        )

        if len(dataset) == 0:
            continue

        dataset.pre_tokenize_with_progress()

        # 分割验证集
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

        # 计算剩余步数（从当前 epoch 之后到结束）
        remaining_epochs = config.num_epochs - saved_epoch - 1
        remaining_epochs = max(0, remaining_epochs)
        steps_per_epoch = len(dataloader) // max(1, config.gradient_accumulation_steps)
        file_steps = steps_per_epoch * remaining_epochs
        total_remaining_steps += file_steps
        file_dataset_infos.append((dataloader, val_dataloader, filename, len(dataset), file_steps, use_validation))

    return file_dataset_infos, total_remaining_steps
