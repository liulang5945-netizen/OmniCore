"""
独立可复用的运行时内存哨兵模块

核心设计：
1. 进程级单例 + 后台缓存轮询 —— 无论多少模块同时调用，整个进程只有 1 个线程
   每 2 秒做 1 次 psutil 调用，所有消费者零开销读取缓存值。
2. @memory_guarded 装饰器 —— 一行接入，自动识别同步/生成器函数。
3. guard() 上下文管理器 —— 代码块级别的内存保护。

使用方式：
    # 方式1: 装饰器（推荐，最简洁）
    @memory_guarded(min_avail_pct=0.10)
    def my_inference(prompt): ...

    # 方式2: 上下文管理器
    with MemoryWatchdog().guard("索引构建", min_avail_pct=0.15):
        heavy_computation()

    # 方式3: 主动查询（循环内自定义策略）
    wd = MemoryWatchdog()
    for batch in data:
        if wd.status.level >= 3:
            break
        process(batch)

5 级哨兵级别：
    Level 0 (≥35%): 充裕   Level 1 (≥25%): 注意
    Level 2 (≥15%): 警告   Level 3 (≥8%):  暂停
    Level 4 (<8%):  紧急停止
"""

import functools
import gc
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("MemoryWatchdog")


# ══════════════════════════════════════════════════════════════════
# MemoryStatus —— 不可变数据类
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MemoryStatus:
    """内存状态快照（不可变，线程安全）"""
    avail_pct: float          # 可用内存占比 (0.0 ~ 1.0)
    avail_gb: float           # 可用 GB
    total_gb: float           # 总 GB
    level: int                # 哨兵级别 0-4
    trend: str                # stable / dropping / dropping_fast / recovering
    timestamp: float          # 采集时间戳

    @classmethod
    def fallback(cls) -> "MemoryStatus":
        """psutil 不可用时的保守回退值（假设内存充裕，不阻塞任何操作）"""
        return cls(
            avail_pct=0.50, avail_gb=4.0, total_gb=8.0,
            level=0, trend="stable", timestamp=time.time()
        )

    def level_desc(self) -> str:
        """返回级别中文描述"""
        _map = {
            0: "充裕", 1: "注意", 2: "警告", 3: "告急", 4: "紧急"
        }
        return _map.get(self.level, "未知")

    def level_emoji(self) -> str:
        """返回级别对应 emoji"""
        _map = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "⚫"}
        return _map.get(self.level, "❓")

    def format_message(self) -> str:
        """生成人类可读的状态描述"""
        return (
            f"内存: {self.avail_pct*100:.1f}% 可用 "
            f"({self.avail_gb:.1f}/{self.total_gb:.1f} GB) "
            f"[{self.level_emoji()} {self.level_desc()}]"
        )


# ══════════════════════════════════════════════════════════════════
# GpuStatus —— GPU 显存状态快照
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GpuStatus:
    """GPU 显存状态快照（不可变，线程安全）"""
    available: bool           # 是否有可用 GPU
    name: str                 # GPU 名称
    total_gb: float           # 总显存 GB
    used_gb: float            # 已用显存 GB
    free_gb: float            # 空闲显存 GB
    utilization_pct: float    # 显存使用率 (0.0 ~ 1.0)
    level: int                # 哨兵级别 0-4（与 MemoryStatus 同标准）
    temperature: float        # GPU 温度 (°C)，不可用时为 0

    @classmethod
    def fallback(cls) -> "GpuStatus":
        """无 GPU 或 pynvml 不可用时的回退值"""
        return cls(
            available=False, name="N/A", total_gb=0.0,
            used_gb=0.0, free_gb=0.0, utilization_pct=0.0,
            level=0, temperature=0.0,
        )

    def level_desc(self) -> str:
        _map = {0: "充裕", 1: "注意", 2: "警告", 3: "告急", 4: "紧急"}
        return _map.get(self.level, "未知")

    def level_emoji(self) -> str:
        _map = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "⚫"}
        return _map.get(self.level, "❓")

    def format_message(self) -> str:
        if not self.available:
            return "GPU: 不可用"
        return (
            f"GPU [{self.name}]: {self.free_gb:.1f}/{self.total_gb:.1f} GB 空闲 "
            f"(使用 {self.utilization_pct*100:.0f}%) "
            f"[{self.level_emoji()} {self.level_desc()}]"
        )


# ══════════════════════════════════════════════════════════════════
# MemoryWatchdog —— 进程级单例 + 后台缓存轮询
# ══════════════════════════════════════════════════════════════════

class MemoryWatchdog:
    """
    进程级单例内存哨兵。

    后台线程每 poll_interval 秒采集一次 psutil 数据并写入缓存，
    所有消费者通过 .status 属性零开销读取。

    线程安全：读取缓存时使用细粒度锁，不阻塞其他读操作。
    """

    _instance: Optional["MemoryWatchdog"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(
        self,
        poll_interval: float = 2.0,
        level0_pct: float = 0.35,
        level1_pct: float = 0.25,
        level2_pct: float = 0.15,
        level3_pct: float = 0.08,
        resume_pct: float = 0.30,
        trend_window: int = 8,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # 哨兵阈值配置
        self.level0_pct = level0_pct
        self.level1_pct = level1_pct
        self.level2_pct = level2_pct
        self.level3_pct = level3_pct
        self.resume_pct = resume_pct

        # 轮询配置
        self._poll_interval = poll_interval

        # 缓存 & 线程安全
        self._cached_status: MemoryStatus = MemoryStatus.fallback()
        self._cached_gpu: GpuStatus = GpuStatus.fallback()
        self._cache_lock = threading.Lock()

        # 趋势检测
        self._trend_history: list = []  # [(timestamp, avail_pct), ...]
        self._trend_window = trend_window

        # 退避控制
        self._backoff_index = 0
        self._backoff_sequence = [2, 4, 8, 16, 30]
        self._paused_by_watchdog = False

        # 后台轮询线程
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, name="MemoryWatchdogPoller", daemon=True)
        self._poll_thread.start()
        logger.debug(
            f"MemoryWatchdog 单例已启动 (轮询间隔={poll_interval}s, "
            f"阈值: L0≥{level0_pct*100:.0f}% L1≥{level1_pct*100:.0f}% "
            f"L2≥{level2_pct*100:.0f}% L3≥{level3_pct*100:.0f}%)"
        )

    # ── 公开属性 ──

    @property
    def status(self) -> MemoryStatus:
        """零开销读取当前内存状态（返回缓存值）"""
        with self._cache_lock:
            return self._cached_status

    @property
    def gpu_status(self) -> GpuStatus:
        """零开销读取当前 GPU 显存状态（返回缓存值）"""
        with self._cache_lock:
            return self._cached_gpu

    # ── 后台轮询 ──

    def _poll_loop(self):
        """后台线程：每 poll_interval 秒采集一次系统内存 + GPU 显存"""
        while self._running:
            # ── 系统内存采集 ──
            try:
                import psutil as _ps
                vm = _ps.virtual_memory()
                avail_pct = vm.available / vm.total
                avail_gb = vm.available / (1024 ** 3)
                total_gb = vm.total / (1024 ** 3)

                # 更新趋势历史
                self._trend_history.append((time.time(), avail_pct))
                if len(self._trend_history) > self._trend_window:
                    self._trend_history.pop(0)

                # 计算趋势
                trend = self._detect_trend()

                # 计算哨兵级别
                level = self._calc_level(avail_pct, trend)

                # 写入缓存
                new_status = MemoryStatus(
                    avail_pct=round(avail_pct, 5),
                    avail_gb=round(avail_gb, 2),
                    total_gb=round(total_gb, 2),
                    level=level,
                    trend=trend,
                    timestamp=time.time(),
                )
                with self._cache_lock:
                    self._cached_status = new_status

            except ImportError:
                # psutil 不可用，保持回退值
                pass
            except Exception:
                logger.debug("MemoryWatchdog 内存轮询异常（非关键）", exc_info=True)

            # ── GPU 显存采集 ──
            try:
                new_gpu = self._poll_gpu()
                with self._cache_lock:
                    self._cached_gpu = new_gpu
            except Exception:
                logger.debug("MemoryWatchdog GPU 轮询异常（非关键）", exc_info=True)

            time.sleep(self._poll_interval)

    def _force_refresh(self):
        """强制立即刷新一次（用于关键决策点，如模型切换预检）"""
        try:
            import psutil as _ps
            vm = _ps.virtual_memory()
            avail_pct = vm.available / vm.total
            avail_gb = vm.available / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            self._trend_history.append((time.time(), avail_pct))
            if len(self._trend_history) > self._trend_window:
                self._trend_history.pop(0)
            trend = self._detect_trend()
            level = self._calc_level(avail_pct, trend)
            new_status = MemoryStatus(
                avail_pct=round(avail_pct, 5),
                avail_gb=round(avail_gb, 2),
                total_gb=round(total_gb, 2),
                level=level,
                trend=trend,
                timestamp=time.time(),
            )
            with self._cache_lock:
                self._cached_status = new_status
        except Exception:
            pass
        # 同步刷新 GPU
        try:
            new_gpu = self._poll_gpu()
            with self._cache_lock:
                self._cached_gpu = new_gpu
        except Exception:
            pass

    def _poll_gpu(self) -> GpuStatus:
        """
        采集 GPU 显存状态。

        优先使用 torch.cuda（零额外依赖），回退到 pynvml（NVIDIA 管理库）。
        两者均不可用时返回 fallback。
        """
        # 方式 1：通过 PyTorch（项目已依赖 torch，零额外安装成本）
        try:
            import torch
            if torch.cuda.is_available():
                device = 0  # 默认主 GPU
                total = torch.cuda.get_device_properties(device).total_mem / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                free = total - reserved  # reserved 包含 allocated + 缓存
                # 实际可用 = total - allocated（比 reserved 更准确）
                free_actual = total - allocated
                util_pct = allocated / total if total > 0 else 0.0
                name = torch.cuda.get_device_name(device)
                # 温度：torch 不直接提供，尝试 pynvml
                temp = 0.0
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                # 计算显存哨兵级别（与 RAM 同阈值）
                free_pct = free_actual / total if total > 0 else 1.0
                level = self._calc_level(free_pct, "stable")
                return GpuStatus(
                    available=True, name=name, total_gb=round(total, 2),
                    used_gb=round(allocated, 2), free_gb=round(free_actual, 2),
                    utilization_pct=round(util_pct, 3), level=level,
                    temperature=float(temp),
                )
        except ImportError:
            pass
        except Exception:
            logger.debug("torch.cuda GPU 查询失败", exc_info=True)

        # 方式 2：通过 pynvml（独立于 torch 的 NVIDIA 管理库）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            total = info.total / (1024 ** 3)
            used = info.used / (1024 ** 3)
            free = info.free / (1024 ** 3)
            util_pct = used / total if total > 0 else 0.0
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            free_pct = free / total if total > 0 else 1.0
            level = self._calc_level(free_pct, "stable")
            return GpuStatus(
                available=True, name=name, total_gb=round(total, 2),
                used_gb=round(used, 2), free_gb=round(free, 2),
                utilization_pct=round(util_pct, 3), level=level,
                temperature=float(temp),
            )
        except ImportError:
            pass
        except Exception:
            logger.debug("pynvml GPU 查询失败", exc_info=True)

        return GpuStatus.fallback()

    def stop(self):
        """停止后台轮询线程（程序退出时调用）"""
        self._running = False

    # ── 级别计算 ──

    def _calc_level(self, avail_pct: float, trend: str) -> int:
        """根据可用内存比例 + 趋势计算当前哨兵级别"""
        if avail_pct < self.level3_pct:
            return 4
        elif avail_pct < self.level2_pct:
            return 3 if self._paused_by_watchdog else (
                2 if trend != "dropping_fast" else 3
            )
        elif avail_pct < self.level1_pct:
            return 3 if (trend == "dropping_fast") else 2
        elif avail_pct < self.level0_pct:
            return 2 if trend == "dropping_fast" else 1
        else:
            return 0

    # ── 趋势检测（线性回归斜率） ──

    def _detect_trend(self) -> str:
        """线性回归分析最近 N 次采样的内存变化趋势"""
        if len(self._trend_history) < 4:
            return "stable"

        steps = [s for s, _ in self._trend_history]
        vals = [v for _, v in self._trend_history]
        n = len(steps)

        mean_step = sum(steps) / n
        mean_val = sum(vals) / n

        num = sum((steps[i] - mean_step) * (vals[i] - mean_val) for i in range(n))
        den = sum((s - mean_step) ** 2 for s in steps)

        if den == 0:
            return "stable"

        slope = num / den

        if slope < -0.02:
            return "dropping_fast"
        elif slope < -0.005:
            return "dropping"
        elif slope > 0.005:
            return "recovering"
        return "stable"

    # ── 指数退避 ──

    def get_backoff_seconds(self) -> float:
        """获取当前退避等待时间（指数增长：2→4→8→16→30s）"""
        idx = min(self._backoff_index, len(self._backoff_sequence) - 1)
        result = self._backoff_sequence[idx]
        self._backoff_index = min(self._backoff_index + 1, len(self._backoff_sequence) - 1)
        return float(result)

    def reset_backoff(self):
        """重置退避计数器（恢复后调用）"""
        self._backoff_index = 0
        self._paused_by_watchdog = False

    # ── 上下文管理器 ──

    @contextmanager
    def guard(self, operation_name: str = "操作", min_avail_pct: float = 0.10):
        """
        上下文管理器：进入时预检，退出时 gc.collect + (可选) torch cuda cache 清理。

        用法:
            with MemoryWatchdog().guard("RAG索引构建", min_avail_pct=0.15):
                rebuild_index()

        若进入时可用内存 < min_avail_pct，抛出 MemoryError。
        """
        st = self.status
        if st.avail_pct < min_avail_pct:
            raise MemoryError(
                f"[{operation_name}] 可用内存仅 {st.avail_pct*100:.1f}% "
                f"({st.avail_gb:.1f}/{st.total_gb:.1f} GB)，"
                f"低于安全阈值 {min_avail_pct*100:.0f}%，操作已中止"
            )
        try:
            yield
        finally:
            gc.collect()
            self._try_cuda_cleanup()

    # ── 预检方法 ──

    def can_proceed(self, min_avail_pct: float = 0.10) -> tuple:
        """
        操作前预检：是否足够内存执行操作。

        Returns:
            (can_proceed: bool, message: str)
        """
        st = self.status
        can = st.avail_pct >= min_avail_pct
        if can:
            return True, ""
        return False, (
            f"可用内存仅 {st.avail_pct*100:.1f}% ({st.avail_gb:.1f}/{st.total_gb:.1f} GB)，"
            f"低于安全阈值 {min_avail_pct*100:.0f}%"
        )

    @staticmethod
    def can_load_model(estimated_gb: float, safety_factor: float = 1.3) -> tuple:
        """
        检查是否有足够内存 + 显存加载模型。

        同时检查系统 RAM 和 GPU VRAM（如有），任一不足即拒绝。

        Args:
            estimated_gb: 预估模型内存占用（GB）
            safety_factor: 安全系数（默认 1.3，即需要 30% 余量）

        Returns:
            (can_load: bool, message: str)
        """
        wd = MemoryWatchdog()
        need_gb = estimated_gb * safety_factor
        errors = []

        # 检查系统 RAM
        ram = wd.status
        if ram.avail_gb < need_gb:
            errors.append(
                f"RAM 不足：需要 {need_gb:.1f}GB，可用 {ram.avail_gb:.1f}GB"
            )

        # 检查 GPU VRAM（如有 GPU）
        gpu = wd.gpu_status
        if gpu.available and gpu.free_gb > 0:
            if gpu.free_gb < need_gb:
                errors.append(
                    f"显存不足：需要 {need_gb:.1f}GB，{gpu.name} 空闲仅 {gpu.free_gb:.1f}GB"
                )

        if not errors:
            return True, ""
        return False, (
            f"加载模型约需 {estimated_gb:.1f}GB（含 {safety_factor}x 安全系数 = {need_gb:.1f}GB）。"
            + "；".join(errors) + "。请关闭其他应用后重试。"
        )

    @staticmethod
    def can_build_embeddings(chunk_count: int, embed_dim: int = 384) -> tuple:
        """
        检查是否有足够内存构建嵌入索引。

        嵌入内存 ≈ chunk_count * embed_dim * 4 bytes (float32)

        Returns:
            (can_build: bool, message: str)
        """
        wd = MemoryWatchdog()
        st = wd.status
        est_gb = chunk_count * embed_dim * 4 / (1024 ** 3)  # float32
        need_gb = est_gb * 2.0  # 2x 安全系数（编码过程有中间张量）
        can = st.avail_gb >= need_gb
        if can:
            return True, ""
        return False, (
            f"内存不足：构建嵌入索引约需 {need_gb:.1f}GB "
            f"({chunk_count} 段落 × {embed_dim} 维)，"
            f"当前可用仅 {st.avail_gb:.1f}GB。请减少文档数量后重试。"
        )

    @staticmethod
    def _try_cuda_cleanup():
        """尝试清理 CUDA 缓存（不抛异常）"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# @memory_guarded 装饰器
# ══════════════════════════════════════════════════════════════════

def memory_guarded(
    min_avail_pct: float = 0.10,
    on_critical: str = "raise",
    stream_check_every: int = 100,
):
    """
    装饰器：函数执行前自动预检内存，对生成器在流式产出中定期检查。

    Args:
        min_avail_pct: 最低可用内存比例阈值（0.0 ~ 1.0）
        on_critical: 内存不足时的行为
            'raise'        → 抛出 MemoryError（同步函数默认值）
            'return_none'  → 返回 None
            'yield_error'  → 用于生成器，yield 错误消息后 return
        stream_check_every: 生成器每产出 N 次后检查一次哨兵级别

    自动识别：
        - 同步函数 → 执行前预检，不足则按 on_critical 处理
        - 生成器函数 → 执行前预检 + 流式产出中每 stream_check_every 次检查

    用法：
        @memory_guarded(min_avail_pct=0.10)
        def my_inference(prompt): ...

        @memory_guarded(min_avail_pct=0.10, on_critical='yield_error')
        def my_stream(prompt): ...  # 生成器函数
    """
    def decorator(func):
        is_gen = inspect.isgeneratorfunction(func)

        if is_gen:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                wd = MemoryWatchdog()
                st = wd.status

                # 预检
                if st.avail_pct < min_avail_pct:
                    msg = (
                        f"[内存不足 ({st.avail_pct*100:.0f}%)，"
                        f"操作 '{func.__name__}' 已中止]"
                    )
                    logger.warning(msg)
                    if on_critical == "yield_error":
                        yield msg
                        return
                    elif on_critical == "return_none":
                        return
                    else:
                        raise MemoryError(msg)

                # 执行生成器，定期哨兵检查
                gen = func(*args, **kwargs)
                count = 0
                for item in gen:
                    count += 1
                    if count % stream_check_every == 0:
                        if wd.status.level >= 3:
                            msg = (
                                f"\n[内存告急 ({wd.status.avail_pct*100:.0f}%)，"
                                f"操作 '{func.__name__}' 已中止]"
                            )
                            logger.warning(msg)
                            yield msg
                            return
                    yield item
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                wd = MemoryWatchdog()
                st = wd.status

                if st.avail_pct < min_avail_pct:
                    msg = (
                        f"可用内存仅 {st.avail_pct*100:.1f}% "
                        f"({st.avail_gb:.1f}/{st.total_gb:.1f} GB)，"
                        f"低于阈值 {min_avail_pct*100:.0f}%，"
                        f"操作 '{func.__name__}' 已中止"
                    )
                    logger.warning(msg)
                    if on_critical == "return_none":
                        return None
                    else:
                        raise MemoryError(msg)

                return func(*args, **kwargs)
            return wrapper

    return decorator


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def get_memory_status_dict() -> dict:
    """
    获取当前内存 + GPU 状态的字典表示（供 API 端点使用）。

    Returns:
        可直接 JSON 序列化的 dict
    """
    wd = MemoryWatchdog()
    st = wd.status
    gpu = wd.gpu_status
    result = {
        "status": "ok",
        "total_gb": st.total_gb,
        "available_gb": st.avail_gb,
        "used_gb": round(st.total_gb - st.avail_gb, 1),
        "available_pct": round(st.avail_pct, 3),
        "level": st.level,
        "level_desc": st.level_desc(),
        "level_emoji": st.level_emoji(),
        "trend": st.trend,
        "message": st.format_message(),
        "timestamp": st.timestamp,
    }
    # GPU 信息（有 GPU 时附加）
    if gpu.available:
        result["gpu"] = {
            "name": gpu.name,
            "total_gb": gpu.total_gb,
            "used_gb": gpu.used_gb,
            "free_gb": gpu.free_gb,
            "utilization_pct": gpu.utilization_pct,
            "level": gpu.level,
            "level_desc": gpu.level_desc(),
            "level_emoji": gpu.level_emoji(),
            "temperature": gpu.temperature,
            "message": gpu.format_message(),
        }
    return result


def force_memory_refresh():
    """强制刷新内存状态缓存（用于关键决策点）"""
    MemoryWatchdog()._force_refresh()


def shutdown_watchdog():
    """优雅关闭后台轮询线程"""
    wd = MemoryWatchdog._instance
    if wd is not None:
        wd.stop()
