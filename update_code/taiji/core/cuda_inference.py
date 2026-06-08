"""
M15: CUDA 原生推理引擎 — Python 包装层

自动将 PyTorch 模型权重传递给 C++ 引擎，整个推理循环在 C++ 中运行。
如果 C++ 引擎不可用，自动回退到 PyTorch Python。

用法:
    from taiji.cuda_inference import CudaInferenceEngine
    
    engine = CudaInferenceEngine(model, tokenizer, device="cuda")
    engine.compile()  # 将模型权重导出到 C++ 引擎
    
    output = engine.generate("你好", max_new_tokens=256)
    for chunk in engine.generate_stream("你好", max_new_tokens=512):
        print(chunk, end="")
"""

import logging
from typing import Optional, Generator, Dict, Any

import torch

logger = logging.getLogger("Taiji.CudaEngine")

_engine_module = None


def _load_engine_module():
    """懒加载 C++ 扩展"""
    global _engine_module
    if _engine_module is None:
        try:
            import taiji_cuda_engine
            _engine_module = taiji_cuda_engine
        except ImportError:
            raise RuntimeError(
                "CUDA 推理引擎未编译！请运行:\n"
                "  cd csrc && build.bat    (Windows)\n"
                "  或: cd csrc && pip install -e ."
            )
    return _engine_module


def is_cuda_engine_available() -> bool:
    """检查 CUDA Engine 是否可用"""
    try:
        _load_engine_module()
        return True
    except Exception:
        return False


class CudaInferenceEngine:
    """
    CUDA 原生推理引擎。
    
    将 PyTorch 模型的权重传入 C++ 侧，整个推理循环在 C++ 中运行，
    消除 Python GIL 开销 + 算子融合 + PagedAttention。
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._engine = None
        self._compiled = False

    def compile(self):
        """将模型权重导出到 C++ 引擎"""
        if self._compiled:
            return

        engine_mod = _load_engine_module()
        config = self.model.config

        # 从 PyTorch config 构建 C++ config
        cpp_config = engine_mod.ModelConfig()
        cpp_config.vocab_size = config.vocab_size
        cpp_config.hidden_size = config.hidden_size
        cpp_config.intermediate_size = config.intermediate_size
        cpp_config.num_hidden_layers = config.num_hidden_layers
        cpp_config.num_attention_heads = config.num_attention_heads
        cpp_config.num_key_value_heads = config.num_key_value_heads
        cpp_config.max_position_embeddings = config.max_position_embeddings
        cpp_config.rms_norm_eps = config.rms_norm_eps
        cpp_config.rope_theta = config.rope_theta

        # 创建 C++ 引擎
        self._engine = engine_mod.TaijiEngine(cpp_config, self.device)

        # 传递权重
        state_dict = self.model.state_dict()
        self._engine.load_state_dict(state_dict)

        self._compiled = True
        logger.info("✅ CUDA 推理引擎编译完成")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """文本生成"""
        if not self._compiled:
            self.compile()

        engine_mod = _load_engine_module()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()

        gen_config = engine_mod.GenerateConfig()
        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = temperature
        gen_config.top_p = top_p
        gen_config.eos_token_id = self.tokenizer.eos_token_id or -1

        output_ids = self._engine.generate(input_ids, gen_config)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_event=None,
    ) -> Generator[str, None, None]:
        """流式生成"""
        if not self._compiled:
            self.compile()

        engine_mod = _load_engine_module()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()

        gen_config = engine_mod.GenerateConfig()
        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = temperature
        gen_config.top_p = top_p
        gen_config.eos_token_id = self.tokenizer.eos_token_id or -1
        gen_config.tokens_per_yield = 8

        for token_batch in self._engine.generate_batched(input_ids, gen_config):
            if stop_event and stop_event.is_set():
                break
            text = self.tokenizer.decode(token_batch, skip_special_tokens=True)
            if text:
                yield text

    def forward(self, input_ids, use_cache=False):
        """完整前向 — 用于训练"""
        if not self._compiled:
            self.compile()
        return self._engine.forward(input_ids, use_cache)

    def reset_cache(self):
        """重置 KV Cache"""
        if self._engine:
            self._engine.reset_cache()

    def is_available(self) -> bool:
        return self._compiled


class TritonInferenceEngine:
    """
    Phase 2: Triton 加速推理引擎。
    
    使用 Triton 融合 kernel 替换 PyTorch ATen 的逐操作调用。
    仅在 CUDA 设备上可用。
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._engine = None

    def compile(self):
        """使用融合 TransformerBlock 替换模型层"""
        try:
            from csrc.cuda.fused_transformer_block import replace_with_fused_blocks
            self.model = replace_with_fused_blocks(self.model, self.model.config)
            self._engine = True
            logger.info("✅ Triton 引擎编译完成（融合 TransformerBlock 已替换）")
        except Exception as e:
            logger.warning(f"Triton 引擎不可用: {e}")

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9, **kwargs):
        if not self._engine:
            self.compile()
        from csrc.cuda.triton_engine import TritonEngine
        engine = TritonEngine(self.model, self.tokenizer, self.device)
        return engine.generate(prompt, max_new_tokens, temperature, top_p)

    def is_available(self):
        return self._engine is not None


class InferenceRouter:
    """
    推理路径自动路由器 — 三阶段升级路径。
    
    优先级:
      1. C++ Engine (Phase 1) — 最快，GIL 释放
      2. Triton Engine (Phase 2) — 融合 kernel，CUDA 加速
      3. PyTorch Native — 回退方案
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._backend = None
        self._engine = None

        # 优先级 1: C++ Engine
        if is_cuda_engine_available():
            try:
                engine = CudaInferenceEngine(model, tokenizer, device)
                engine.compile()
                self._engine = engine
                self._backend = "cuda_engine"
                logger.info("✅ 推理路由: C++ CUDA Engine (Phase 1)")
                return
            except Exception as e:
                logger.debug(f"C++ Engine 不可用: {e}")

        # 优先级 2: Triton Engine (仅 CUDA)
        if device.startswith("cuda"):
            try:
                engine = TritonInferenceEngine(model, tokenizer, device)
                engine.compile()
                self._engine = engine
                self._backend = "triton_engine"
                logger.info("✅ 推理路由: Triton Engine (Phase 2)")
                return
            except Exception as e:
                logger.debug(f"Triton Engine 不可用: {e}")

        # 优先级 3: PyTorch Native
        self._backend = "pytorch"
        logger.info("推理路由: PyTorch Native (回退)")

    def generate(self, prompt, task_type="chat", **kwargs):
        if self._backend == "cuda_engine":
            return self._engine.generate(prompt, **kwargs)
        elif self._backend == "triton_engine":
            return self._engine.generate(prompt, **kwargs)
        else:
            from taiji.inference import NativeInferenceEngine
            engine = NativeInferenceEngine(self.model, self.tokenizer, self.device)
            return engine.generate(prompt, **kwargs)

    def generate_stream(self, prompt, task_type="chat", **kwargs):
        if self._backend == "cuda_engine":
            yield from self._engine.generate_stream(prompt, **kwargs)
        else:
            from taiji.inference import NativeInferenceEngine
            engine = NativeInferenceEngine(self.model, self.tokenizer, self.device)
            yield from engine.generate_stream(prompt, **kwargs)

    def get_backend_info(self) -> dict:
        return {
            "backend": self._backend or "pytorch",
            "phase": {
                "cuda_engine": "Phase 1 (C++ libtorch)",
                "triton_engine": "Phase 2 (Triton fused kernels)",
                "pytorch": "PyTorch native",
            }.get(self._backend, "unknown"),
            "gill_free": self._backend in ("cuda_engine", "triton_engine"),
        }
