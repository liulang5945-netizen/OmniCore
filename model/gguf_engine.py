"""
GGUF 推理引擎
基于 llama-cpp-python，通过 Vulkan 利用 AMD/NVIDIA/Intel 显卡加速
专为 ROG ALLY（AMD Z1 Extreme RDNA3）等低内存设备优化

核心优势：
  1. 4-bit GGUF 量化内存仅 ~4-5GB（vs FP16 的 ~14GB）
  2. Vulkan 后端利用 AMD 集成显卡全速推理
  3. 支持 CPU+GPU 混合推理，显存不足时自动溢出到系统内存
"""

import os
import json
import logging
import threading
from typing import Optional, Generator
from dataclasses import dataclass

from core.memory_watchdog import memory_guarded

logger = logging.getLogger("GGUFEngine")


# ======================== 统一推理接口 ========================

class BaseGGUFEngine:
    """
    GGUF 模型推理引擎基类
    封装 llama-cpp-python，提供流式/非流式推理
    与 trainer.py 中的 BaseInferenceEngine 保持相同接口约定
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,      # -1 = 所有层放 GPU
        n_ctx: int = 2048,            # 上下文长度
        vocab_only: bool = False,
        verbose: bool = False,
        n_threads: int = None,        # CPU 线程数，None=自动
        n_batch: int = 2048,          # prompt处理批大小（越大越快，更耗内存）
        **kwargs,
    ):
        """
        初始化 GGUF 引擎

        Args:
            model_path: .gguf 模型文件路径
            n_gpu_layers: GPU 加速层数 (-1 = 全部, 0 = 纯 CPU)
            n_ctx: 上下文窗口大小
            vocab_only: 仅加载词表（用于测试）
            verbose: 是否打印详细日志
            n_threads: CPU 线程数，None=自动使用所有物理核心
            n_batch: prompt处理批大小，越大越快（默认2048）
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.n_threads = n_threads
        self.n_batch = n_batch
        self._llm = None
        self._lock = threading.Lock()
        self._loaded = False

    @memory_guarded(min_avail_pct=0.25, on_critical='raise')
    def load(self):
        """加载 GGUF 模型（懒加载）"""
        if self._loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GGUF 模型文件不存在: {self.model_path}\n"
                f"请先从 HuggingFace 下载量化模型，例如:\n"
                f"  https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF\n"
                f"  下载 DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
            )

        # 防崩溃：检查路径是否指向文件而非目录
        if os.path.isdir(self.model_path):
            raise ValueError(
                f"选择的路径是一个目录，不是 .gguf 模型文件: {self.model_path}\n"
                f"请选择一个以 .gguf 结尾的具体模型文件，例如 DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
            )

        # 防崩溃：检查文件扩展名
        if not self.model_path.lower().endswith(".gguf"):
            logger.warning(f"文件不是 .gguf 格式: {self.model_path}")
            raise ValueError(
                f"选择的文件不是 .gguf 格式: {self.model_path}\n"
                f"GGUF 模型文件必须以 .gguf 结尾。请选择正确的模型文件。"
            )

        try:
            from llama_cpp import Llama

            # 自动检测 CPU 核心数
            if self.n_threads is None:
                try:
                    # 使用物理核心数（不包含超线程虚拟核心的一半）
                    cpu_count = os.cpu_count() or 4
                    # 对于纯 CPU 推理，使用全部核心；GPU 推理时减少核心避免争用
                    if self.n_gpu_layers != 0:
                        self.n_threads = max(4, cpu_count // 2)  # GPU辅助CPU，一半核心
                    else:
                        self.n_threads = max(4, cpu_count)  # 纯CPU，全部核心
                    logger.info(f"自动检测 CPU 核心数: {cpu_count}, 设置 n_threads={self.n_threads}")
                except Exception:
                    self.n_threads = 4

            # 确定 GPU 后端实际可用性
            gpu_backend = "禁用"
            if self.n_gpu_layers == -1:
                gpu_layers_display = "全部"
            elif self.n_gpu_layers == 0:
                gpu_layers_display = "0 (纯CPU)"
            else:
                gpu_layers_display = str(self.n_gpu_layers)

            if self.n_gpu_layers != 0:
                # 检测 llama_cpp 编译时启用的后端
                try:
                    # 检查是否有 vulkan 或 cuda 可用
                    from llama_cpp import llama_supports_gpu_offload
                    if llama_supports_gpu_offload():
                        gpu_backend = "GPU加速"
                    else:
                        gpu_backend = "纯CPU(未检测到GPU后端)"
                        logger.warning(
                            "⚠️ 未检测到 GPU 加速后端！当前 llama-cpp-python 未编译 GPU 支持。\n"
                            "   推理运行在纯 CPU 模式，速度较慢。\n"
                            "   要启用 GPU 加速，请重新安装:\n"
                            "   pip uninstall llama-cpp-python -y\n"
                            "   set CMAKE_ARGS=\"-DLLAMA_VULKAN=on\" && pip install llama-cpp-python"
                        )
                except (ImportError, AttributeError):
                    # 无法检测，按默认假设 GPU 可用
                    gpu_backend = "GPU加速(假设)"
            else:
                gpu_layers_display = "0 (纯CPU)"

            logger.info(
                f"加载 GGUF 模型: {os.path.basename(self.model_path)}\n"
                f"  GPU 层数: {gpu_layers_display}\n"
                f"  上下文: {self.n_ctx}\n"
                f"  批大小: {self.n_batch}\n"
                f"  CPU 线程: {self.n_threads}\n"
                f"  GPU 后端: {gpu_backend}"
            )

            # 检测文件扩展名
            if not self.model_path.lower().endswith(".gguf"):
                logger.warning(f"文件不是 .gguf 格式: {self.model_path}")

            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )

            self._loaded = True
            logger.info(f"GGUF 模型加载成功！参数量等信息请参考模型文件")

        except ImportError as e:
            raise RuntimeError(
                f"需要安装 llama-cpp-python 才能使用 GGUF 模型！\n"
                f"安装命令 (ROG ALLY 需启用 Vulkan):\n"
                f"  set CMAKE_ARGS=\"-DLLAMA_VULKAN=on\"\n"
                f"  pip install llama-cpp-python\n\n"
                f"如果不需要 GPU 加速 (纯CPU):\n"
                f"  pip install llama-cpp-python\n\n"
                f"错误详情: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"加载 GGUF 模型失败: {e}") from e

    def unload(self):
        """卸载模型释放内存"""
        with self._lock:
            self._llm = None
            self._loaded = False
            import gc
            gc.collect()
            logger.info("GGUF 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ======================== 推理接口 ========================

    @memory_guarded(min_avail_pct=0.10, on_critical='yield_error')
    def generate_stream(
        self,
        prompt: str,
        tokenizer=None,
        max_new_tokens: int = 512,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成文本

        Args:
            prompt: 输入提示词
            tokenizer: 兼容参数（GGUF引擎不使用独立的tokenizer）
            max_new_tokens: 最大生成长度（兼容BaseInferenceEngine接口）
            max_tokens: 最大生成长度（兼容旧调用）
            temperature: 采样温度
            top_p: 核采样参数
            stop: 停止词列表

        Yields:
            逐步生成的文本片段
        """
        if max_tokens is None:
            max_tokens = max_new_tokens
        
        if not self._loaded:
            self.load()

        if stop is None:
            stop = ["</s>", "<|im_end|>", "\n###"]

        with self._lock:
            try:
                stream = self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.15,
                    stop=stop,
                    stream=True,
                    echo=False,
                )

                for output in stream:
                    chunk = output["choices"][0]["text"]
                    if chunk:
                        yield chunk

            except Exception as e:
                logger.error(f"GGUF 流式生成出错: {e}")
                yield f"\n\n[生成出错: {e}]"

    @memory_guarded(min_avail_pct=0.10, on_critical='raise')
    def generate(
        self,
        prompt: str,
        tokenizer=None,
        max_tokens: int = 512,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> str:
        """
        全量生成文本（非流式）

        Returns:
            完整生成文本
        """
        if not self._loaded:
            self.load()

        if max_new_tokens is not None:
            max_tokens = max_new_tokens

        if stop is None:
            stop = ["</s>", "<|im_end|>", "\n###"]

        with self._lock:
            try:
                output = self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.15,
                    stop=stop,
                    stream=False,
                    echo=False,
                )
                return output["choices"][0]["text"].strip()
            except Exception as e:
                logger.error(f"GGUF 全量生成出错: {e}")
                return f"[生成出错: {e}]"


# ======================== 模型下载辅助 ========================

GGUF_MODEL_REGISTRY = {
    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_gb": 4.5,
        "description": "DeepSeek 7B 4-bit 量化 — 推荐，仅需 ~5GB 内存",
    },
    "DeepSeek-R1-Distill-Qwen-7B-Q8_0": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
        "size_gb": 8.0,
        "description": "DeepSeek 7B 8-bit 量化 — 质量更高，需 ~9GB 内存",
    },
    "Qwen2.5-7B-Instruct-Q4_K_M": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.5,
        "description": "Qwen2.5 7B 4-bit 量化 — 优秀的中文支持",
    },
}


def list_available_gguf_models() -> dict:
    """列出可下载的 GGUF 模型列表"""
    return GGUF_MODEL_REGISTRY


def download_gguf_model(
    model_key: str,
    download_dir: str,
    progress_callback=None,
) -> str:
    """
    从 HuggingFace 下载 GGUF 模型文件

    Args:
        model_key: GGUF_MODEL_REGISTRY 中的键名
        download_dir: 下载保存目录
        progress_callback: 进度回调函数(下载字节数, 总字节数)

    Returns:
        下载完成的 .gguf 文件路径
    """
    if model_key not in GGUF_MODEL_REGISTRY:
        raise ValueError(f"未知的模型: {model_key}，可选: {list(GGUF_MODEL_REGISTRY.keys())}")

    import requests
    from tqdm import tqdm

    info = GGUF_MODEL_REGISTRY[model_key]
    repo = info["repo"]
    filename = info["filename"]

    # 构建 HuggingFace 下载 URL
    # 使用 HuggingFace Hub API 获取下载链接
    hf_url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, filename)

    # 如果文件已存在，跳过下载
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        logger.info(f"模型文件已存在: {local_path} ({file_size / 1e9:.1f} GB)")
        return local_path

    logger.info(f"开始下载 GGUF 模型: {filename}")
    logger.info(f"来源: {hf_url}")
    logger.info(f"保存到: {local_path}")

    try:
        # 流式下载
        response = requests.get(hf_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8 * 1024 * 1024  # 8MB 块

        with open(local_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"下载 {filename}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        if progress_callback:
                            progress_callback(pbar.n, total_size)

        logger.info(f"模型下载完成: {local_path}")
        return local_path

    except requests.RequestException as e:
        # 下载失败时清理未完成的文件
        if os.path.exists(local_path):
            os.remove(local_path)
        # 检查是否有本地备选文件
        fallback_path = _find_local_gguf_fallback(filename)
        if fallback_path:
            logger.warning(f"下载失败，使用本地备选: {fallback_path}")
            return fallback_path
        raise RuntimeError(f"下载 GGUF 模型失败: {e}\n"
                           f"你可以手动从 {hf_url} 下载，或检查本地 gguf_models 目录") from e
    except (ConnectionError, TimeoutError, OSError) as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        fallback_path = _find_local_gguf_fallback(filename)
        if fallback_path:
            logger.warning(f"网络错误，使用本地备选: {fallback_path}")
            return fallback_path
        raise RuntimeError(f"网络错误导致下载失败: {e}") from e


def _find_local_gguf_fallback(filename: str) -> Optional[str]:
    """在本地 gguf_models 目录中查找备选文件"""
    from core.utils import get_external_path
    local_dir = get_external_path("gguf_models")
    if not os.path.exists(local_dir):
        return None
    # 精确匹配
    exact = os.path.join(local_dir, filename)
    if os.path.exists(exact):
        return exact
    # 模糊匹配：查找所有 .gguf 文件
    for f in os.listdir(local_dir):
        if f.endswith(".gguf"):
            return os.path.join(local_dir, f)
    return None


# ======================== 快速检测函数 ========================

def is_gguf_model(path: str) -> bool:
    """检测文件或目录是否为 GGUF 模型"""
    if not path:
        return False
    path = str(path).lower()
    if path.endswith(".gguf"):
        return True
    # 也支持目录中包含 .gguf 文件的情况
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith(".gguf"):
                return True
    return False


def find_gguf_file(path: str, recursive: bool = True) -> Optional[str]:
    """
    在路径中查找 .gguf 文件，支持文件路径和目录。

    Args:
        path: 搜索路径
        recursive: 是否递归搜索子目录。True 用于用户主动选择 GGUF 目录场景；
                   False 仅搜索直接子文件（用于 HF 优先检测，防止 HuggingFace
                   仓库内嵌的 GGUF 变体被误识别）
    """
    if not path:
        return None
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return path
    if os.path.isdir(path):
        # 先检查直接子文件（顶层）
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if f.lower().endswith(".gguf") and os.path.isfile(fp):
                return fp
        if not recursive:
            return None
        # 递归搜索子目录（适配 HuggingFace 下载后的嵌套目录结构）
        # ⚠️ 仅当目录中无法找到 config.json 时才应启用递归，
        #    由调用方决定是否使用 recursive=True
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".gguf"):
                    return os.path.join(root, f)
    return None


# ======================== GPU 检测 ========================

def detect_available_gpu_backends() -> list:
    """
    检测可用的 GPU 后端
    Returns:
        列表: ["vulkan", "cuda", "metal", "cpu"]
    """
    backends = ["cpu"]  # CPU 永远可用

    try:
        import llama_cpp
        # 尝试检测 llama_cpp 编译时启用了哪些后端
        # 通过检查库文件中的符号或者直接测试
        try:
            # 创建一个极小的测试模型来检测 GPU 支持
            test = llama_cpp.Llama(
                model_path="",  # 空路径只检测编译选项
                n_gpu_layers=0,
                verbose=False,
                vocab_only=True,
            )
            # 如果没报错，说明编译成功
        except Exception:
            pass

        # 检查是否存在 Vulkan 的 DLL
        if os.name == "nt":  # Windows
            import ctypes
            try:
                ctypes.windll.LoadLibrary("vulkan-1.dll")
                backends.append("vulkan")
            except Exception:
                pass
            try:
                ctypes.windll.LoadLibrary("cublas64_12.dll")
                backends.append("cuda")
            except Exception:
                pass
    except ImportError:
        pass

    return backends
