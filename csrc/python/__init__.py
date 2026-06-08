"""
Taiji CUDA Engine — Python 包入口

自动加载 C++ 扩展模块，提供简洁的 Python API。
如果 C++ 扩展未编译，提供清晰的错误信息和编译指引。

用法:
    from csrc.python import TaijiEngine, ModelConfig, GenerateConfig
    
    config = ModelConfig.size_125m()
    engine = TaijiEngine(config, device="cpu")
    engine.load_state_dict(model.state_dict())
    
    gen_config = GenerateConfig()
    gen_config.max_new_tokens = 256
    gen_config.temperature = 0.7
    
    output_ids = engine.generate([1, 2, 3], gen_config)
    print(output_ids)
"""

import os
import sys
import logging

logger = logging.getLogger("Taiji.CudaEngine")

# ── 尝试加载 C++ 扩展 ──
try:
    import taiji_cuda_engine
    TaijiEngine = taiji_cuda_engine.TaijiEngine
    ModelConfig = taiji_cuda_engine.ModelConfig
    GenerateConfig = taiji_cuda_engine.GenerateConfig
    __version__ = taiji_cuda_engine.__version__
    __phase__ = taiji_cuda_engine.__phase__
    CUDA_ENGINE_AVAILABLE = True
    logger.info(f"✅ CUDA Engine 已加载 (版本 {__version__}, {__phase__})")
except ImportError as e:
    CUDA_ENGINE_AVAILABLE = False
    TaijiEngine = None
    ModelConfig = None
    GenerateConfig = None
    __version__ = "0.0.0"
    __phase__ = "未编译"
    logger.warning(f"⚠️ CUDA Engine 不可用: {e}")


def is_available() -> bool:
    """检查 CUDA Engine 是否可用。"""
    return CUDA_ENGINE_AVAILABLE


def get_build_instructions() -> str:
    """返回编译指引。"""
    return """
═══════════════════════════════════════════════════════════
  Taiji CUDA Engine — 编译指引
═══════════════════════════════════════════════════════════

前提条件:
  - Python 3.9+
  - PyTorch 2.0+ (pip install torch)
  - pybind11 (pip install pybind11)
  - CMake 3.18+ (pip install cmake)
  - Visual Studio 2022 (Windows) 或 GCC 11+ (Linux)
  - CUDA Toolkit (可选，GPU 加速)

Windows 编译:
  cd csrc
  mkdir build && cd build
  cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build . --config Release
  copy Release\\taiji_cuda_engine*.pyd ..\\python\\

Linux 编译:
  cd csrc
  mkdir build && cd build
  cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build . --config Release
  cp taiji_cuda_engine*.so ../python/

快速编译 (pip):
  cd csrc
  pip install -e .

═══════════════════════════════════════════════════════════
"""


def check_or_raise():
    """如果 CUDA Engine 不可用，抛出异常并显示编译指引。"""
    if not CUDA_ENGINE_AVAILABLE:
        raise RuntimeError(
            "Taiji CUDA Engine 未编译！\n"
            + get_build_instructions()
        )