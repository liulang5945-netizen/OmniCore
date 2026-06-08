"""
M16: 性能分析器 — 推理/训练瓶颈自动识别

集成 PyTorch Profiler，自动识别:
  - 最慢的 CUDA kernel
  - 显存使用 waterfall
  - Python 层开销占比
  - kernel launch 频率

用法:
    from taiji.profiler import TaijiProfiler
    
    profiler = TaijiProfiler()
    with profiler.profile_inference(engine, prompt="你好") as report:
        report.print_summary()
        report.export_chrome_trace("trace.json")
"""

import torch
import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("Taiji.Profiler")


@dataclass
class TimingResult:
    """单次操作计时结果"""
    name: str
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0

    @property
    def avg_ms(self):
        return self.total_ms / self.count if self.count > 0 else 0.0


@dataclass
class ProfilerReport:
    """性能分析报告"""
    timings: Dict[str, TimingResult] = field(default_factory=dict)
    total_time_ms: float = 0.0
    num_tokens: int = 0
    tokens_per_second: float = 0.0

    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 70)
        print("  态极推理性能报告")
        print("=" * 70)
        print(f"  总时间:        {self.total_time_ms:.1f} ms")
        print(f"  生成 token 数:  {self.num_tokens}")
        print(f"  吞吐量:        {self.tokens_per_second:.1f} tokens/s")
        print(f"  每 token 延迟:  {self.total_time_ms / max(self.num_tokens, 1):.2f} ms")
        print("-" * 70)
        print(f"  {'操作':<30} {'总耗时(ms)':<12} {'次数':<8} {'平均(ms)':<10}")
        print("-" * 70)
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1].total_ms):
            print(f"  {name:<30} {t.total_ms:<12.2f} {t.count:<8} {t.avg_ms:<10.3f}")
        print("=" * 70 + "\n")


class TaijiProfiler:
    """
    态极性能分析器。
    
    用法:
        profiler = TaijiProfiler()
        
        # 方法 1: 简单计时
        profiler.start("forward")
        output = model(input_ids)
        profiler.stop("forward")
        
        # 方法 2: 上下文管理器
        with profiler.timer("attention"):
            attn_output = attention(q, k, v)
        
        # 方法 3: 完整推理分析
        report = profiler.profile_generate(engine, prompt="你好")
        report.print_summary()
    """

    def __init__(self):
        self._timings: Dict[str, TimingResult] = {}
        self._active_timers: Dict[str, float] = {}

    def start(self, name: str):
        """开始计时"""
        self._active_timers[name] = time.perf_counter() * 1000

    def stop(self, name: str):
        """停止计时"""
        if name not in self._active_timers:
            return
        elapsed = time.perf_counter() * 1000 - self._active_timers.pop(name)

        if name not in self._timings:
            self._timings[name] = TimingResult(name=name)
        t = self._timings[name]
        t.total_ms += elapsed
        t.count += 1
        t.min_ms = min(t.min_ms, elapsed)
        t.max_ms = max(t.max_ms, elapsed)

    @contextmanager
    def timer(self, name: str):
        """上下文管理器计时"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def profile_generate(self, engine, prompt: str, max_new_tokens: int = 64) -> ProfilerReport:
        """
        完整推理性能分析。
        
        Args:
            engine: 推理引擎 (NativeInferenceEngine 或 CudaInferenceEngine)
            prompt: 测试 prompt
            max_new_tokens: 生成 token 数
        
        Returns:
            ProfilerReport
        """
        self._timings.clear()

        total_start = time.perf_counter()

        with self.timer("tokenize"):
            inputs = engine.tokenizer(prompt, return_tensors="pt")

        with self.timer("generate"):
            output = engine.generate(prompt, max_new_tokens=max_new_tokens)

        total_ms = (time.perf_counter() - total_start) * 1000

        # 估算 token 数
        num_tokens = len(engine.tokenizer.encode(output)) if output else 0

        report = ProfilerReport(
            timings=self._timings.copy(),
            total_time_ms=total_ms,
            num_tokens=num_tokens,
            tokens_per_second=num_tokens / (total_ms / 1000) if total_ms > 0 else 0,
        )

        return report

    def get_report(self) -> ProfilerReport:
        """获取当前计时报告"""
        return ProfilerReport(timings=self._timings.copy())