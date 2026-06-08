"""
自定义训练循环 —— 核心训练逻辑
完全手写 forward → loss → backward → grad_accum → clip → step → scheduler → log
支持暂停/停止/实时 Loss 报告

子模块拆分:
  - model/inference_engine.py → BaseInferenceEngine（推理引擎基类）
  - model/training_utils.py   → EarlyStoppingCriteria + CPU 线程优化常量
"""
import gc
import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import TrainingConfig
from core.memory_watchdog import MemoryWatchdog
from model.model_setup import save_checkpoint
from model.inference_engine import BaseInferenceEngine
from model.training_utils import INFERENCE_THREADS, TRAINING_THREADS

logger = logging.getLogger("Trainer")

# 初始化全局线程数为推理模式
torch.set_num_threads(INFERENCE_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(INFERENCE_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(INFERENCE_THREADS))


class Trainer(BaseInferenceEngine):
    """自定义训练器 —— 手写全部训练逻辑，继承推理引擎"""

    def __init__(self, model, optimizer, scheduler, config: TrainingConfig, device: str):
        super().__init__(model, config, device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = float("inf")
        self.global_step = 0
        self.is_paused = False
        self.is_stopped = False
        # 早停相关
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        # TensorBoard
        self.writer = None
        # 验证集 DataLoader
        self.val_dataloader = None

        # ── 硬件信息采集（用于前端展示 + 诊断）──
        self._hw_info = self._collect_hardware_info()

        # ── 运行时内存哨兵：委托给全局单例 MemoryWatchdog ──
        # 使用 self.config._hw_diag 中的自适应阈值初始化（若存在），否则使用默认值
        _wd_config = getattr(self.config, '_hw_diag', {}).get("memory_watchdog", {})
        self._watchdog = MemoryWatchdog(
            poll_interval=2.0,
            level0_pct=_wd_config.get("level0_pct", 0.35),
            level1_pct=_wd_config.get("level1_pct", 0.25),
            level2_pct=_wd_config.get("level2_pct", 0.15),
            level3_pct=_wd_config.get("level3_pct", 0.08),
            resume_pct=_wd_config.get("resume_pct", 0.30),
            trend_window=_wd_config.get("trend_window_size", 8),
        )
        # 训练专用哨兵状态（不与 MemoryWatchdog 单例共享，避免影响其他模块）
        self._mw_consecutive_warns = 0        # 连续警告计数
        self._mw_degraded_grad_accum_steps = None  # 梯度累积降级状态
        # 持久化吞吐量（在所有 yield 点共享上一个测量值，前端不闪回 --）
        self._last_samples_per_sec = 0.0
        self._mw_last_pause_yield_time = 0.0  # 上次暂停时发送 SSE 事件的时间戳

    # ── 向后兼容：_mw_* 属性代理到 MemoryWatchdog 单例 ──
    @property
    def _mw_level(self):
        return self._watchdog.status.level

    @property
    def _mw_avail_pct(self):
        return self._watchdog.status.avail_pct

    @staticmethod
    def _collect_hardware_info() -> dict:
        """采集训练设备的硬件信息，用于前端诊断展示"""
        import torch

        device_type = "cpu"
        device_name = "CPU"
        gpu_name = None
        gpu_memory_gb = None
        ram_gb = None

        # 检测设备
        if torch.cuda.is_available():
            device_type = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
                device_name = gpu_name
            except Exception:
                device_name = "CUDA GPU"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple MPS (Metal)"

        # 检测系统 RAM
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            ram_gb = round(TrainingConfig.get_total_ram_gb(), 1)

        return {
            "device_type": device_type,
            "device_name": device_name,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "ram_gb": ram_gb,
        }

    def _make_base_metrics(self, **overrides) -> dict:
        """生成包含硬件信息的基础 metrics dict，所有 yield 点复用"""
        m = {
            "device_type": self._hw_info["device_type"],
            "device_name": self._hw_info["device_name"],
            "gpu_name": self._hw_info.get("gpu_name"),
            "gpu_memory_gb": self._hw_info.get("gpu_memory_gb"),
            "ram_gb": self._hw_info.get("ram_gb"),
            # 持久化吞吐量字段（避免非主进度 yield 点显示 --）
            "samples_per_sec": self._last_samples_per_sec,
            "updates_per_epoch": 0,
        }
        m.update(overrides)
        return m

    # ======================== 主训练循环 ========================

    def train(self, dataloader: DataLoader, start_epoch: int = 0):
        """
        训练循环核心逻辑

        Yield:
            (fraction, description, loss_history, metrics_dict)
            metrics_dict 包含: elapsed, eta, lr, step, total_steps, epoch, total_epochs, loss, grad_norm
        """
        config = self.config
        model = self.model.to(self.device)
        model.train()

        total_batches = len(dataloader)
        if total_batches == 0:
            raise ValueError(
                "DataLoader 为空（batch 数为 0），无法开始训练。"
                "请检查：1) 数据集是否有数据 2) batch_size 是否大于数据集大小"
            )
        updates_per_epoch = max(1, total_batches // config.gradient_accumulation_steps)

        logger.info(f"训练配置: Epochs={config.num_epochs}, "
                    f"Batch={config.batch_size}, "
                    f"GradAccum={config.gradient_accumulation_steps}, "
                    f"EffectiveBatch={config.batch_size * config.gradient_accumulation_steps}, "
                    f"Updates/Epoch={updates_per_epoch}")

        # AMP 自动混合精度（仅 CUDA）
        scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None

        # 初始化 TensorBoard
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(config.output_dir, config.tensorboard_dir)
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_dir)
                logger.info(f"TensorBoard 日志目录: {tb_dir}")
                # 记录超参数
                self.writer.add_text("超参数/lora_r", str(config.lora_r))
                self.writer.add_text("超参数/lora_alpha", str(config.lora_alpha))
                self.writer.add_text("超参数/learning_rate", str(config.learning_rate))
                self.writer.add_text("超参数/batch_size", str(config.batch_size))
                self.writer.add_text("超参数/num_epochs", str(config.num_epochs))
            except Exception as e:
                logger.warning(f"TensorBoard 初始化失败（非关键）: {e}")
                self.writer = None

        # 总参数更新步数（跨所有 epoch）
        total_epochs_remaining = config.num_epochs - start_epoch
        total_train_steps = max(1, updates_per_epoch * total_epochs_remaining)
        self.loss_history = []

        # 记录训练开始时间，用于 ETA 计算
        self.train_start_time = time.time()
        self._last_report_time = self.train_start_time
        self._last_report_step = 0

        # 固定极小 fraction 发送准备消息（会被 routes_training 映射为 file_progress_start + epsilon）
        # 避免与 epoch_start_fraction(0.0) 产生回退，将准备信息合并到一条
        yield (0.0001, f"🔧 准备模型 → 🚀 开始训练 | Epoch 1/{config.num_epochs} | 总更新步数 {total_train_steps} | 批次 {total_batches} | 梯度累积 {config.gradient_accumulation_steps}", [], self._make_base_metrics())

        # 全局累计 update_idx（跨 epoch 不重置，确保 fraction 单调递增）
        global_update_idx = 0
        samples_processed = 0

        for epoch in range(start_epoch, config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            ep_loss = 0.0
            accum_loss = 0.0
            t0 = time.time()

            # Epoch 开始阶段通知（fraction 使用 epoch 起始位置，不再回退）
            epoch_start_fraction = (epoch - start_epoch) / total_epochs_remaining if total_epochs_remaining > 0 else 0.0
            yield (
                epoch_start_fraction,
                f"📖 Epoch {epoch + 1}/{config.num_epochs} 开始 | 正在读取数据...",
                self.loss_history,
                self._make_base_metrics(
                    elapsed=time.time() - self.train_start_time,
                    eta=None,
                    lr=self.scheduler.get_last_lr()[0] if self.scheduler else 0,
                    step=self.global_step,
                    total_steps=total_train_steps,
                    epoch=epoch + 1,
                    total_epochs=config.num_epochs,
                    loss=None,
                    grad_norm=None,
                ),
            )

            # 在迭代 DataLoader 前发出首条进度，消除静默期
            yield (
                epoch_start_fraction + 0.002,
                f"📖 Epoch {epoch + 1}/{config.num_epochs} | "
                f"准备首次前向传播（梯度累积 {config.gradient_accumulation_steps} 步/batch）...",
                self.loss_history,
                self._make_base_metrics(
                    elapsed=time.time() - self.train_start_time,
                    eta=None,
                    lr=self.scheduler.get_last_lr()[0] if self.scheduler else 0,
                    step=0,
                    total_steps=total_train_steps,
                    epoch=epoch + 1,
                    total_epochs=config.num_epochs,
                    loss=None,
                    grad_norm=None,
                ),
            )

            # ── 首次计算前 CPU 安全保护 ──
            # 1. 临时将 PyTorch 线程数降到最低，给 QtWebEngine 子进程喘息时间
            #    避免 WebEngine 因 CPU 被占满而崩溃重启新进程
            # 2. 插入主动 sleep，让 Qt 事件循环有机会处理 pending 的渲染/网络请求
            # 3. 执行一次 Python GC + CUDA cache clear，释放内存压力
            _saved_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 主动让出 CPU 调度，确保 QtWebEngineProcess 获得至少一个时间片
            time.sleep(0.5)

            pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Ep{epoch + 1}")

            updates_this_epoch = 0  # 本 epoch 内的参数更新次数
            first_batch_notified = False  # 首个批次完成时通知一次
            # 看门狗可能在首个 batch 完成前触发，初始化这些变量防止 NameError
            fraction = epoch / max(1, config.num_epochs)
            elapsed_total = 0.0
            eta_total = None
            lr = config.learning_rate
            # CUDA 异步传输加速
            _use_nonblocking = self.device == "cuda"
            # ════════════════════════════════════════════════════════════
            # 渐进式线程恢复策略
            # 目标：逐步恢复到 TRAINING_THREADS（物理核-1），为 QtWebEngine 预留 1 核。
            # INFERENCE_THREADS（全部物理核）仅在推理时使用。
            # ════════════════════════════════════════════════════════════
            if TRAINING_THREADS <= 2:
                _thread_recovery_phase = 2  # 核数少，直接标记完成
            else:
                _thread_recovery_phase = 0
            _thread_recovery_batches_remaining = 0

            for bidx, batch in pbar:
                # ── pre-batch 内存守卫：提前跳过，避免 forward 中 OOM ──
                if not self._pre_batch_memory_guard():
                    time.sleep(self._get_backoff_seconds())
                    continue

                # ── 渐进式线程恢复：分阶段提升到 TRAINING_THREADS ──
                if _thread_recovery_phase < 3:
                    if _thread_recovery_phase == 0:
                        if _thread_recovery_batches_remaining <= 0:
                            _thread_recovery_batches_remaining = 2
                        torch.set_num_threads(1)
                        _thread_recovery_batches_remaining -= 1
                        if _thread_recovery_batches_remaining <= 0:
                            _thread_recovery_phase = 1
                            _thread_recovery_batches_remaining = 3
                    elif _thread_recovery_phase == 1:
                        _half_threads = max(2, TRAINING_THREADS // 2)
                        torch.set_num_threads(_half_threads)
                        _thread_recovery_batches_remaining -= 1
                        if _thread_recovery_batches_remaining <= 0:
                            _thread_recovery_phase = 2
                            _thread_recovery_batches_remaining = 3
                    elif _thread_recovery_phase == 2:
                        _three_quarter = max(3, int(TRAINING_THREADS * 0.75))
                        torch.set_num_threads(_three_quarter)
                        _thread_recovery_batches_remaining -= 1
                        if _thread_recovery_batches_remaining <= 0:
                            _thread_recovery_phase = 3
                            torch.set_num_threads(TRAINING_THREADS)
                            os.environ["OMP_NUM_THREADS"] = str(TRAINING_THREADS)
                            os.environ["MKL_NUM_THREADS"] = str(TRAINING_THREADS)
                            time.sleep(0.15)

                # 暂停检测：sleep 轮询，不占CPU
                while self.is_paused and not self.is_stopped:
                    time.sleep(0.2)
                # 停止检测
                if self.is_stopped:
                    break

                # 异步数据搬运到设备（non_blocking 让 GPU 传输与计算重叠）
                input_ids = batch["input_ids"].to(self.device, non_blocking=_use_nonblocking)
                attn = batch["attention_mask"].to(self.device, non_blocking=_use_nonblocking)
                labels = batch["labels"].to(self.device, non_blocking=_use_nonblocking)

                # 混合精度前向传播
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=labels,
                        )
                        loss = outputs.loss / config.gradient_accumulation_steps
                    accum_loss += loss.item()
                    scaler.scale(loss).backward()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        labels=labels,
                    )
                    loss = outputs.loss / config.gradient_accumulation_steps
                    accum_loss += loss.item()
                    loss.backward()

                samples_processed += input_ids.shape[0]

                # 首个批次完成前向+反向传播后立即通知，消除「卡住」的疑虑
                if not first_batch_notified:
                    first_batch_notified = True
                    yield (
                        epoch_start_fraction + 0.005,
                        f"📖 Epoch {epoch + 1}/{config.num_epochs} | "
                        f"首个批次已完成（loss={loss.item() * config.gradient_accumulation_steps:.4f}），"
                        f"正在进行梯度累积...",
                        self.loss_history,
                        self._make_base_metrics(
                            elapsed=time.time() - self.train_start_time,
                            eta=None,
                            lr=self.scheduler.get_last_lr()[0] if self.scheduler else 0,
                            step=0,
                            total_steps=total_train_steps,
                            epoch=epoch + 1,
                            total_epochs=config.num_epochs,
                            loss=float(loss.item() * config.gradient_accumulation_steps),
                            grad_norm=None,
                        ),
                    )

                # 梯度累积 -> 更新参数
                if (bidx + 1) % config.gradient_accumulation_steps == 0 \
                        or (bidx + 1) == total_batches:

                    if scaler:
                        scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    lr = self.scheduler.get_last_lr()[0]

                    self.optimizer.zero_grad()

                    # 记录
                    self.global_step += 1
                    global_update_idx += 1
                    updates_this_epoch += 1
                    ep_loss += accum_loss
                    current_loss = float(accum_loss)  # 保存副本，用于后续 yield 和日志

                    # 日志
                    if self.global_step % config.logging_steps == 0:
                        self.loss_history.append({
                            "Step": self.global_step,
                            "Loss": current_loss,
                        })
                        logger.info(
                            f"[Step {self.global_step:6d}] "
                            f"loss={current_loss:.4f}  lr={lr:.2e}  gnorm={grad_norm:.2f}"
                        )
                        # TensorBoard 记录训练指标
                        if config.use_tensorboard and self.writer is not None:
                            self.writer.add_scalar("Loss/train", current_loss, self.global_step)
                            self.writer.add_scalar("LR", lr, self.global_step)
                            self.writer.add_scalar("GradNorm", grad_norm, self.global_step)

                    # 保存检查点（立即清理旧文件，避免 epoch 内堆积过多）
                    if self.global_step % config.save_steps == 0:
                        save_checkpoint(
                            model, self.optimizer, self.scheduler,
                            config, epoch, self.global_step, current_loss,
                            config.output_dir,
                            dataset_files=self._dataset_files if hasattr(self, '_dataset_files') else None,
                        )
                        self._cleanup_checkpoints()

                    pbar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": self.global_step,
                    })
                    accum_loss = 0.0  # 重置累积损失

                    # ── 运行时内存哨兵 v2.0（5 级渐进式 + 趋势检测）──
                    _mw_result = self._run_memory_watchdog()
                    if _mw_result == 4:
                        yield (
                            fraction,
                            f"🚨 内存告急！可用仅 {getattr(self, '_mw_avail_pct', 0)*100:.0f}%，紧急停止训练",
                            self.loss_history,
                            self._make_base_metrics(
                                elapsed=elapsed_total,
                                eta=eta_total,
                                lr=lr,
                                step=self.global_step,
                                total_steps=total_train_steps,
                                epoch=epoch + 1,
                                total_epochs=config.num_epochs,
                                loss=current_loss,
                                grad_norm=float(grad_norm) if grad_norm is not None else None,
                                memory_status=f"紧急停止: 可用 {getattr(self, '_mw_avail_pct', 0)*100:.0f}%",
                            ),
                        )
                        break
                    elif _mw_result == 3:
                        _now = time.time()
                        if _now - self._mw_last_pause_yield_time > 5.0:
                            self._mw_last_pause_yield_time = _now
                            yield (
                                fraction,
                                f"⏸️ {self._get_memory_status_msg()} → 暂停（退避 {self._get_backoff_seconds()}s）",
                                self.loss_history,
                                self._make_base_metrics(
                                    elapsed=elapsed_total, eta=eta_total, lr=lr,
                                    step=self.global_step, total_steps=total_train_steps,
                                    epoch=epoch + 1, total_epochs=config.num_epochs,
                                    loss=current_loss, grad_norm=None,
                                    memory_status=f"已暂停: {self._get_memory_status_msg()}",
                                ),
                            )
                        time.sleep(self._get_backoff_seconds())
                        continue
                    elif _mw_result == -1:
                        yield (
                            fraction,
                            f"▶️ {self._get_memory_status_msg()} → 自动恢复训练",
                            self.loss_history,
                            self._make_base_metrics(
                                elapsed=elapsed_total, eta=eta_total, lr=lr,
                                step=self.global_step, total_steps=total_train_steps,
                                epoch=epoch + 1, total_epochs=config.num_epochs,
                                loss=current_loss, grad_norm=None,
                                memory_status=f"已恢复: {self._get_memory_status_msg()}",
                            ),
                        )

                    # 进度报告：fraction 基于全局累计 update_idx，单调递增不回退
                    report_interval = max(1, min(total_train_steps // 100, 3))
                    if global_update_idx % report_interval == 0 or global_update_idx <= 5:
                        # 使用 epoch 偏移 + epoch 内进度，确保 fraction 平滑单调递增
                        fraction = min(1.0, (epoch - start_epoch + updates_this_epoch / max(1, updates_per_epoch)) / total_epochs_remaining)

                        # 计算 ETA
                        elapsed_total = time.time() - self.train_start_time
                        if fraction > 0.001 and global_update_idx > 0:
                            eta_total = elapsed_total / fraction - elapsed_total
                        else:
                            eta_total = None

                        # 计算吞吐量并持久化
                        elapsed_since_last = time.time() - self._last_report_time
                        steps_since_last = global_update_idx - self._last_report_step
                        steps_per_sec = steps_since_last / max(0.1, elapsed_since_last) if elapsed_since_last > 0 else 0
                        self._last_samples_per_sec = steps_per_sec * config.batch_size * config.gradient_accumulation_steps
                        self._last_report_time = time.time()
                        self._last_report_step = global_update_idx

                        yield (
                            fraction,
                            f"Epoch {epoch + 1}/{config.num_epochs} | "
                            f"进度 {fraction * 100:.1f}% | "
                            f"Step {self.global_step}/{total_train_steps}"
                            f"{' | Loss ' + f'{current_loss:.4f}' if current_loss > 0 else ''}",
                            self.loss_history,
                            self._make_base_metrics(
                                elapsed=elapsed_total,
                                eta=eta_total,
                                lr=lr,
                                step=self.global_step,
                                total_steps=total_train_steps,
                                epoch=epoch + 1,
                                total_epochs=config.num_epochs,
                                loss=current_loss,
                                grad_norm=float(grad_norm) if grad_norm is not None else None,
                                samples_per_sec=self._last_samples_per_sec,
                                updates_per_epoch=updates_per_epoch,
                            ),
                        )

            if self.is_stopped:
                logger.info("训练被手动终止！")
                break

            # Epoch 结束总结
            avg = ep_loss / max(1, updates_per_epoch)
            elapsed = time.time() - t0
            logger.info(f"Epoch {epoch + 1} 完成 | AvgLoss={avg:.4f} | 耗时={elapsed:.1f}s")

            save_checkpoint(
                model, self.optimizer, self.scheduler,
                config, epoch, self.global_step, avg,
                config.output_dir,
                dataset_files=self._dataset_files if hasattr(self, '_dataset_files') else None,
            )
            self._cleanup_checkpoints()

            if avg < self.best_loss:
                self.best_loss = avg
                logger.info(f">> 新的最佳 Loss: {self.best_loss:.4f}")

            # ===== 验证集评估 =====
            if config.validation_split > 0 and hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
                val_loss, val_ppl = self.evaluate(self.val_dataloader)
                logger.info(f"验证集: Loss={val_loss:.4f}  PPL={val_ppl:.2f}")

                # TensorBoard 记录验证指标
                if config.use_tensorboard and self.writer is not None:
                    self.writer.add_scalar("Loss/val", val_loss, self.global_step)
                    self.writer.add_scalar("Perplexity/val", val_ppl, self.global_step)

                # ===== Early Stopping =====
                if config.early_stopping_patience > 0:
                    if val_loss < self.best_val_loss - config.early_stopping_threshold:
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                        logger.info(f">> 新的最佳验证 Loss: {val_loss:.4f}（计数器重置）")
                    else:
                        self.early_stopping_counter += 1
                        logger.info(f"早停计数器: {self.early_stopping_counter}/{config.early_stopping_patience} "
                                    f"（当前 val_loss={val_loss:.4f} >= 最佳={self.best_val_loss:.4f}）")
                        if self.early_stopping_counter >= config.early_stopping_patience:
                            logger.info(f"🛑 触发早停！连续 {config.early_stopping_patience} 个 epoch 验证 Loss 未改善")
                            break
            else:
                # 无验证集时，使用训练 Loss 判断早停
                if config.early_stopping_patience > 0:
                    if avg < self.best_val_loss - config.early_stopping_threshold:
                        self.best_val_loss = avg
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= config.early_stopping_patience:
                            logger.info(f"🛑 触发早停（基于训练 Loss）！")
                            break

        # ── 训练结束，恢复推理线程数 ──
        torch.set_num_threads(INFERENCE_THREADS)
        os.environ["OMP_NUM_THREADS"] = str(INFERENCE_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(INFERENCE_THREADS)
        logger.info(f"训练结束，已恢复推理线程: {INFERENCE_THREADS} (全部物理核)")

        logger.info(f"训练完成！Best={self.best_loss:.4f} Steps={self.global_step}")

    # ======================== 评估 ========================

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        """验证集评估"""
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in tqdm(dataloader, desc="评估"):
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            total_loss += out.loss.item()
            n += 1

        avg_loss = total_loss / max(1, n)
        ppl = math.exp(avg_loss)
        logger.info(f"评估结果: Loss={avg_loss:.4f}  PPL={ppl:.2f}")
        self.model.train()
        return avg_loss, ppl

    # ==================== 运行时内存哨兵 v2.0（委托给单例 MemoryWatchdog）====================

    def _run_memory_watchdog(self, force_check: bool = False) -> int:
        """委托给单例 MemoryWatchdog，训练层附加梯度累积降级 & max_length 自适应"""
        wd = self._watchdog
        st = wd.status
        level = st.level
        trend = st.trend
        avail_pct = st.avail_pct

        if level == 4:
            logger.critical(f"🚨 内存仅 {avail_pct*100:.1f}%，紧急停止并保存检查点")
            from model.model_setup import save_checkpoint
            try:
                save_checkpoint(self.model, self.optimizer, self.scheduler,
                                self.config, -1, self.global_step, float('inf'),
                                self.config.output_dir)
            except Exception:
                pass
            self.is_stopped = True
            return 4

        if level == 3:
            if not wd._paused_by_watchdog:
                wd._paused_by_watchdog = True
                logger.warning(f"⏸️ 内存 {avail_pct*100:.1f}%（趋势: {trend}），退避 {wd.get_backoff_seconds()}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 3

        if level == 2:
            self._mw_consecutive_warns += 1
            gc.collect()
            if self._mw_degraded_grad_accum_steps is None:
                _orig = self.config.gradient_accumulation_steps
                _degraded = min(8, max(2, _orig + 2))
                if _degraded > _orig:
                    self._mw_degraded_grad_accum_steps = _orig
                    self.config.gradient_accumulation_steps = _degraded
                    logger.info(f"📉 grad_accum {_orig} → {_degraded}")
            if self._mw_consecutive_warns % 5 == 0:
                logger.info(f"⚠️ 连续 {self._mw_consecutive_warns} 次警告，内存 {avail_pct*100:.1f}%")
            return 2

        if level == 1:
            self._mw_consecutive_warns = 0
            gc.collect()
            if self.config.max_length > 128 and trend == "dropping_fast":
                old = self.config.max_length
                self.config.max_length = max(128, int(self.config.max_length * 0.9))
                if self.config.max_length < old:
                    logger.info(f"📏 max_length {old} → {self.config.max_length}")
            return 1

        # level 0 (充裕) 或恢复
        if wd._paused_by_watchdog and avail_pct >= wd.resume_pct:
            wd.reset_backoff()
            self._mw_consecutive_warns = 0
            if self._mw_degraded_grad_accum_steps is not None:
                self.config.gradient_accumulation_steps = self._mw_degraded_grad_accum_steps
                self._mw_degraded_grad_accum_steps = None
            logger.info(f"▶️ 内存恢复至 {avail_pct*100:.1f}%")
            return -1

        self._mw_consecutive_warns = 0
        wd.reset_backoff()
        if self._mw_degraded_grad_accum_steps is not None:
            self.config.gradient_accumulation_steps = self._mw_degraded_grad_accum_steps
            self._mw_degraded_grad_accum_steps = None
        return 0

    def _get_backoff_seconds(self) -> float:
        """委托给单例 MemoryWatchdog"""
        return self._watchdog.get_backoff_seconds()

    def _pre_batch_memory_guard(self) -> bool:
        """batch 前预检：可用内存是否足够处理下一个 batch"""
        st = self._watchdog.status
        per_sample_gb = getattr(self.config, 'per_sample_gb', 0)
        if per_sample_gb <= 0:
            return True
        batch_est_gb = self.config.batch_size * per_sample_gb * 1.5
        if st.avail_gb < batch_est_gb * 1.2:
            logger.warning(f"🛡️ pre-batch: 可用 {st.avail_gb:.1f}GB < 需要 {batch_est_gb*1.2:.1f}GB")
            return False
        return True

    def _get_memory_status_msg(self) -> str:
        """生成当前内存状态的描述文本（用于 SSE 通知前端）"""
        return self._watchdog.status.format_message()

    # ======================== 内部 ========================

    def _cleanup_checkpoints(self):
        """只保留最近的 N 个检查点"""
        import glob
        pattern = os.path.join(self.config.output_dir, "checkpoint-e*.pt")
        ckpts = sorted(glob.glob(pattern))
        for old in ckpts[: -self.config.save_total_limit]:
            try:
                os.remove(old)
                logger.info(f"已清理旧检查点: {old}")
            except OSError:
                pass
