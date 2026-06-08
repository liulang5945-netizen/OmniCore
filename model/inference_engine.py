"""
推理引擎模块

从 model/trainer.py 中提取:
  - BaseInferenceEngine: 推理引擎基类
    仅提供推理功能，不携带训练状态（optimizer/scheduler）
    用于 API 服务器中的推理场景
    
  优化:
    1. 自动检测并使用 C++ CUDA 引擎（如果可用）
    2. torch.inference_mode 替代 torch.no_grad（PyTorch 2.x 推荐）
    3. tokenizer 输入缓存（减少重复 tokenize）
"""
import logging
import threading

import torch
from transformers import TextIteratorStreamer, StoppingCriteriaList

from core.memory_watchdog import memory_guarded
from model.training_utils import EarlyStoppingCriteria

logger = logging.getLogger("Trainer")


def _try_import_cuda_engine():
    """懒加载并尝试导入 C++ CUDA 推理引擎"""
    try:
        from taiji.core.cuda_inference import CudaInferenceEngine, is_cuda_engine_available
        return CudaInferenceEngine, is_cuda_engine_available
    except ImportError:
        return None, None


class BaseInferenceEngine:
    """
    推理引擎基类
    仅提供推理功能，不携带训练状态（optimizer/scheduler）
    用于 API 服务器中的推理场景
    """

    def __init__(self, model, config, device: str):
        self.model = model
        self.config = config
        self.device = device
        
        # 尝试初始化 C++ CUDA 引擎
        self._cuda_engine = None
        self._cuda_engine_cls, self._cuda_available = _try_import_cuda_engine()
        if self._cuda_available and self._cuda_available():
            try:
                self._cuda_engine = self._cuda_engine_cls(model, None, device)
                self._cuda_engine.compile()
                logger.info("C++ CUDA 引擎已激活，推理将使用优化后的原生实现")
            except Exception as e:
                logger.warning(f"C++ CUDA 引擎初始化失败，将回退到 PyTorch: {e}")
                self._cuda_engine = None

    def _ensure_cuda_engine(self, tokenizer):
        """确保 C++ 引擎已准备好使用（设置 tokenizer）"""
        if self._cuda_engine is not None:
            self._cuda_engine.tokenizer = tokenizer
        return self._cuda_engine is not None

    @memory_guarded(min_avail_pct=0.10, on_critical='raise')
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """标准生成（完整结果），只返回新生成的 token"""
        # 优先使用 C++ CUDA 引擎（如果可用）
        if self._ensure_cuda_engine(tokenizer):
            with torch.inference_mode():
                return self._cuda_engine.generate(prompt, max_new_tokens, temperature, top_p)

        self.model.eval()
        inp = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inp.input_ids.shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 仅解码新生成的 token，避免重复输出 prompt 部分
        new_tokens = out[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

    @memory_guarded(min_avail_pct=0.10, on_critical='yield_error')
    def generate_stream(self, prompt: str, tokenizer, max_new_tokens: int = 512,
                        temperature: float = 0.7, top_p: float = 0.9,
                        stop_event=None):
        """流式生成（逐 token 输出），加入重复惩罚防循环。

        Args:
            stop_event: 可选的 threading.Event，当被 set() 时立即中断生成循环。
                        用于客户端断开连接时快速终止后台推理线程。
        """
        # 优先使用 C++ CUDA 引擎（如果可用）
        if self._ensure_cuda_engine(tokenizer):
            with torch.inference_mode():
                yield from self._cuda_engine.generate_stream(prompt, max_new_tokens, temperature, top_p)
            return

        self.model.eval()
        inp = tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inp,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 添加 EarlyStoppingCriteria，使 stop_event 能真正终止 model.generate
        if stop_event is not None:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                EarlyStoppingCriteria(stop_event)
            ])

        with torch.inference_mode():
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()

        for new_text in streamer:
            if stop_event and stop_event.is_set():
                break
            yield new_text
