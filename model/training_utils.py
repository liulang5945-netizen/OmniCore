"""
训练工具模块

从 model/trainer.py 中提取:
  - EarlyStoppingCriteria: 可中断的停止条件
  - CPU 线程优化常量 (INFERENCE_THREADS / TRAINING_THREADS)
"""
import logging
import os

from transformers import StoppingCriteria

logger = logging.getLogger("Trainer")

# ── 全局线程优化：推理时使用全部物理核心，训练时限制为物理核-1 ──
# 训练期间密集 CPU 计算会争夺 QtWebEngine 子进程的 CPU 时间片，
# 导致 WebEngine 被强杀重启，用户看到"二次打开客户端窗口"。
# 因此训练时预留 1 个物理核给 UI，推理时恢复全部核心以最大化 token/s。
_cpu_logical = os.cpu_count() or 4
try:
    import psutil
    _cpu_physical = psutil.cpu_count(logical=False) or max(1, _cpu_logical // 2)
except ImportError:
    _cpu_physical = max(1, _cpu_logical // 2)

# 推理时使用全部物理核心（训练期间由 Trainer.train() 临时降为物理核-1）
INFERENCE_THREADS = _cpu_physical
TRAINING_THREADS = max(1, _cpu_physical - 1)

logger.info(f"CPU 线程优化: 推理={INFERENCE_THREADS} (全部物理核) / 训练={TRAINING_THREADS} (预留1核给UI)"
            f" / 物理核={_cpu_physical}, 逻辑核={_cpu_logical}")


class EarlyStoppingCriteria(StoppingCriteria):
    """
    可中断的停止条件。
    当 stop_event 被 set() 后，模型生成循环会在下一步立即停止，
    而不是必须生成完 max_new_tokens 个 token 才结束。
    """
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event is not None and self.stop_event.is_set()
