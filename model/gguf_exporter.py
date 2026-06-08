"""
GGUF 模型导出器
将 HuggingFace 格式的模型（含 LoRA 合并后的微调模型）转换为 .gguf 量化格式

完整流水线:
  1. LoRA 训练 (trainer.py) → checkpoint-e*.pt
  2. Merge 合并 (model_setup.merge_and_save_lora_model) → HF 格式完整模型目录
  3. 本模块: HF → GGUF FP16 → Q4_K_M / Q8_0 等量化
  4. CPU 推理 (gguf_engine.py)  → 低内存设备本地推理

依赖:
  - llama.cpp convert_hf_to_gguf.py (自动下载)
  - llama-quantize 或 llama-cpp-python 内置量化功能

量化选项:
  Q2_K   - 极致压缩 (~2.5 bit, 质量损失较大)
  Q3_K_M - 高压缩 (~3.3 bit)
  Q4_K_M - 推荐平衡 (~4.5 bit, 质量损失很小)  ✅ 默认
  Q5_K_M - 高质量 (~5.5 bit)
  Q8_0   - 近乎无损 (~8 bit, 文件较大)
  F16    - 无量化 (16 bit float, 仅转换格式)
"""
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional

logger = logging.getLogger("GGUFExporter")

# ======================== 量化选项表 ========================

QUANT_OPTIONS = {
    "Q2_K":   {"desc": "Q2_K - 极致压缩 (~2.5 bit)", "size_ratio": 0.18},
    "Q3_K_M": {"desc": "Q3_K_M - 高压缩 (~3.3 bit)", "size_ratio": 0.22},
    "Q4_K_M": {"desc": "Q4_K_M - 推荐平衡 (~4.5 bit) ✅", "size_ratio": 0.28},
    "Q5_K_M": {"desc": "Q5_K_M - 高质量 (~5.5 bit)", "size_ratio": 0.35},
    "Q8_0":   {"desc": "Q8_0 - 近乎无损 (~8 bit)", "size_ratio": 0.55},
    "F16":    {"desc": "F16 - 仅转换格式（无量化）", "size_ratio": 1.0},
}


def _get_llama_cpp_dir() -> str:
    """获取或下载 llama.cpp 工具目录"""
    from core.utils import get_external_path
    tools_dir = get_external_path("llama_cpp_tools")
    os.makedirs(tools_dir, exist_ok=True)
    return tools_dir


def _ensure_convert_script(progress_cb: Callable = None) -> str:
    """
    确保 convert_hf_to_gguf.py 可用。
    如果不存在，从 llama.cpp GitHub 仓库下载转换脚本及依赖。
    """
    tools_dir = _get_llama_cpp_dir()
    convert_script = os.path.join(tools_dir, "convert_hf_to_gguf.py")
    gguf_py_dir = os.path.join(tools_dir, "gguf")

    # 检查脚本是否已存在
    if os.path.exists(convert_script) and os.path.isdir(gguf_py_dir):
        logger.info(f"转换脚本已就绪: {convert_script}")
        return convert_script

    if progress_cb:
        progress_cb("📥 下载 llama.cpp 转换工具...", 0.01)

    import urllib.request
    import ssl

    ctx = ssl.create_default_context()

    # llama.cpp 版本标签
    LLAMA_CPP_TAG = "b4684"
    BASE_URL = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{LLAMA_CPP_TAG}"

    # 需要下载的文件列表
    FILES_TO_DOWNLOAD = [
        "convert_hf_to_gguf.py",
        "gguf/__init__.py",
        "gguf/constants.py",
        "gguf/gguf_writer.py",
        "gguf/lazy.py",
        "gguf/tensor_mapping.py",
        "gguf/vocab.py",
        "gguf/utility.py",
    ]

    total = len(FILES_TO_DOWNLOAD)
    for i, fname in enumerate(FILES_TO_DOWNLOAD):
        url = f"{BASE_URL}/{fname}"
        local_path = os.path.join(tools_dir, fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if progress_cb:
            progress_cb(f"📥 下载 {fname}... ({i+1}/{total})", 0.01 + 0.04 * (i / total))

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                with open(local_path, "wb") as f:
                    f.write(resp.read())
            logger.info(f"已下载: {fname}")
        except Exception as e:
            # 如果下载失败且文件已存在，使用已有文件
            if os.path.exists(local_path):
                logger.warning(f"下载 {fname} 失败({e})，使用已有缓存")
            else:
                raise RuntimeError(
                    f"下载 llama.cpp 转换工具失败: {fname}\n"
                    f"URL: {url}\n"
                    f"错误: {e}\n"
                    f"请手动下载 llama.cpp 仓库到: {tools_dir}"
                ) from e

    if os.path.exists(convert_script):
        logger.info(f"llama.cpp 转换工具就绪: {convert_script}")
        return convert_script

    raise RuntimeError("下载后仍未找到 convert_hf_to_gguf.py，请检查网络连接或手动安装")


def _find_quantize_binary() -> Optional[str]:
    """
    查找 llama-quantize 二进制文件路径。

    搜索顺序:
    1. 项目外部工具目录中的可执行文件
    2. PATH 环境变量中的 llama-quantize
    3. llama-cpp-python 包自带的 quantize
    """
    tools_dir = _get_llama_cpp_dir()

    # 检查外部工具目录
    candidates = []
    if sys.platform == "win32":
        candidates = [
            os.path.join(tools_dir, "llama-quantize.exe"),
            os.path.join(tools_dir, "build", "bin", "Release", "llama-quantize.exe"),
        ]
    else:
        candidates = [
            os.path.join(tools_dir, "llama-quantize"),
            os.path.join(tools_dir, "build", "bin", "llama-quantize"),
        ]

    for c in candidates:
        if os.path.exists(c):
            logger.info(f"找到量化工具: {c}")
            return c

    # 检查 PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        exe_name = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
        exe_path = os.path.join(path_dir, exe_name)
        if os.path.exists(exe_path):
            return exe_path

    # 检查 llama-cpp-python 内置量化功能
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_model_quantize"):
            logger.info("使用 llama-cpp-python 内置量化功能")
            return "__builtin__"
    except ImportError:
        pass

    return None


def _quantize_with_llama_cpp(input_path: str, output_path: str, quant_type: str,
                              progress_cb: Callable = None) -> bool:
    """使用外部 llama-quantize 二进制文件进行量化"""
    quantize_bin = _find_quantize_binary()
    if not quantize_bin:
        raise RuntimeError("未找到 llama-quantize 工具，无法进行量化")

    if quantize_bin == "__builtin__":
        return _quantize_with_builtin(input_path, output_path, quant_type, progress_cb)

    cmd = [quantize_bin, input_path, output_path, quant_type]

    if progress_cb:
        progress_cb(f"🔧 量化: {quant_type} ...", 0.35)

    logger.info(f"执行量化: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                logger.debug(f"[quantize] {line}")
                if "[" in line and "]" in line and progress_cb:
                    # 尝试解析进度
                    progress_cb(f"🔧 量化中... {line[:80]}", 0.35)
        proc.wait(timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"量化进程退出码: {proc.returncode}")
        return os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("量化超时（10分钟）")
    except FileNotFoundError:
        raise RuntimeError(
            f"找不到量化工具: {quantize_bin}\n"
            f"请确保 llama.cpp 已正确安装。\n"
            f"下载地址: https://github.com/ggerganov/llama.cpp/releases"
        )


def _quantize_with_builtin(input_path: str, output_path: str, quant_type: str,
                            progress_cb: Callable = None) -> bool:
    """使用 llama-cpp-python 内置的量化功能"""
    try:
        import ctypes
        import llama_cpp

        # llama-cpp-python >= 0.2.0 支持此功能
        if not hasattr(llama_cpp, "llama_model_quantize"):
            raise RuntimeError("当前 llama-cpp-python 版本不支持内置量化，请升级或安装 llama-quantize 工具")

        if progress_cb:
            progress_cb(f"🔧 使用内置量化: {quant_type} ...", 0.35)

        # 调用底层量化函数
        llama_cpp.llama_model_quantize(
            input_path.encode("utf-8"),
            output_path.encode("utf-8"),
            ctypes.c_int(_quant_to_int(quant_type)),
        )
        return os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"内置量化失败: {e}，回退到外部工具")
        return _quantize_with_external(input_path, output_path, quant_type, progress_cb)


def _quantize_with_external(input_path: str, output_path: str, quant_type: str,
                             progress_cb: Callable = None) -> bool:
    """尝试使用外部 llama-quantize（最可靠的方式）"""
    quantize_bin = _find_quantize_binary()
    if not quantize_bin or quantize_bin == "__builtin__":
        raise RuntimeError(
            "未找到 llama-quantize 量化工具。\n"
            "请从 https://github.com/ggerganov/llama.cpp/releases 下载预编译版本，\n"
            "或将 llama-quantize 放入系统 PATH。"
        )
    return _quantize_with_llama_cpp(input_path, output_path, quant_type, progress_cb)


def _quant_to_int(quant_type: str) -> int:
    """量化类型字符串 → llama.cpp 内部枚举值"""
    mapping = {
        "Q2_K": 10, "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13,
        "Q4_K_S": 14, "Q4_K_M": 15, "Q5_K_S": 16, "Q5_K_M": 17,
        "Q6_K": 18, "Q8_0": 7, "Q8_1": 8, "F16": 1,
    }
    return mapping.get(quant_type, 15)  # 默认 Q4_K_M


def _estimate_output_size(input_dir: str, quant_type: str) -> float:
    """估算输出 GGUF 文件大小 (GB)"""
    total_size = 0
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                total_size += os.path.getsize(os.path.join(root, f))
    ratio = QUANT_OPTIONS.get(quant_type, {}).get("size_ratio", 0.28)
    return round(total_size * ratio / (1024**3), 2)


def export_to_gguf(
    hf_model_dir: str,
    output_dir: str,
    output_name: str = "model",
    quant_type: str = "Q4_K_M",
    progress_callback: Callable = None,
) -> str:
    """
    将 HuggingFace 格式模型导出为 GGUF 量化格式

    完整流水线: HF → GGUF FP16 → 量化 GGUF

    Args:
        hf_model_dir: HuggingFace 格式模型目录（含 config.json + safetensors）
        output_dir: 输出目录
        output_name: 输出文件名（不含扩展名，将自动添加 .gguf）
        quant_type: 量化类型（Q2_K ~ F16，默认 Q4_K_M）
        progress_callback: 进度回调 (message: str, fraction: float)

    Returns:
        生成的 .gguf 文件路径

    Raises:
        RuntimeError: 转换失败时
    """
    if quant_type not in QUANT_OPTIONS:
        raise ValueError(
            f"不支持的量化类型: {quant_type}\n"
            f"可用选项: {', '.join(QUANT_OPTIONS.keys())}"
        )

    if not os.path.isdir(hf_model_dir):
        raise FileNotFoundError(f"HF 模型目录不存在: {hf_model_dir}")
    if not os.path.exists(os.path.join(hf_model_dir, "config.json")):
        raise FileNotFoundError(f"目录中缺少 config.json: {hf_model_dir}")

    def _progress(msg: str, frac: float):
        logger.info(f"[GGUF导出 {frac*100:.0f}%] {msg}")
        if progress_callback:
            progress_callback(msg, frac)

    est_size = _estimate_output_size(hf_model_dir, quant_type)
    _progress(f"📊 预估输出大小: {est_size} GB ({quant_type})", 0.05)

    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 确保有转换脚本
    _progress("📦 准备转换工具...", 0.10)
    convert_script = _ensure_convert_script(
        lambda msg, f: _progress(msg, 0.10 + f * 0.05)
    )

    # 步骤2: HF → GGUF FP16
    fp16_path = os.path.join(output_dir, f"{output_name}_fp16.gguf")
    final_path = os.path.join(output_dir, f"{output_name}_{quant_type}.gguf")

    _progress(f"🔄 转换 HF → GGUF FP16 ...", 0.20)

    tools_dir = _get_llama_cpp_dir()
    env = os.environ.copy()
    env["PYTHONPATH"] = tools_dir + (os.pathsep + env.get("PYTHONPATH", "")
                                      if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable, convert_script,
        "--outfile", fp16_path,
        "--outtype", "f16",
        hf_model_dir,
    ]

    logger.info(f"转换命令: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=tools_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        last_progress = 0.20
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            logger.debug(f"[convert] {line}")

            # 解析 llama.cpp 输出中的进度
            if "Writing" in line or "layer" in line or "tensor" in line.lower():
                # 粗略进度：20% ~ 30%
                fraction = min(0.30, last_progress + 0.005)
                last_progress = fraction
                if "Writing" in line:
                    _progress(f"✍️ {line[:100]}", fraction)
            elif "Done" in line:
                _progress("✅ 转换完成", 0.30)
                break

        proc.wait(timeout=1200)  # 20分钟超时
        if proc.returncode != 0:
            raise RuntimeError(f"转换进程退出码: {proc.returncode}")

    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("HF → GGUF 转换超时（20分钟），模型可能过大")

    if not os.path.exists(fp16_path):
        raise RuntimeError(f"转换后未找到 GGUF FP16 文件: {fp16_path}")

    fp16_size_gb = os.path.getsize(fp16_path) / (1024**3)
    _progress(f"✅ GGUF FP16 生成完毕 ({fp16_size_gb:.1f} GB)", 0.30)

    # 步骤3: 量化（如果选择了量化类型）
    if quant_type == "F16":
        # 不需要量化，直接重命名
        shutil.move(fp16_path, final_path)
        _progress(f"✅ 导出完成: {final_path}", 1.0)
        return final_path

    _progress(f"🔧 量化 {quant_type} ...", 0.35)

    quant_success = _quantize_with_llama_cpp(
        fp16_path, final_path, quant_type,
        progress_callback=lambda msg, f: _progress(msg, 0.35 + f * 0.60),
    )

    if not quant_success:
        # 回退：保留 FP16 版本
        logger.warning("量化失败，保留 FP16 版本")
        shutil.move(fp16_path, final_path)
        _progress(f"⚠️ 量化失败，使用 FP16 格式: {final_path}", 1.0)
        return final_path

    # 清理 FP16 中间文件
    if os.path.exists(fp16_path):
        try:
            os.remove(fp16_path)
        except OSError:
            pass

    final_size_gb = os.path.getsize(final_path) / (1024**3)
    compression = (1 - final_size_gb / fp16_size_gb) * 100 if fp16_size_gb > 0 else 0
    _progress(
        f"✅ 导出完成！{os.path.basename(final_path)} "
        f"({final_size_gb:.1f} GB, 压缩率 {compression:.0f}%)",
        1.0,
    )

    return final_path


def export_published_to_gguf(
    published_dir: str,
    quant_type: str = "Q4_K_M",
    progress_callback: Callable = None,
) -> str:
    """
    一键将已发布的 LoRA 合并模型导出为 GGUF

    这是最便捷的入口：接收已 merge 的 HF 模型目录，直接输出 GGUF。

    Args:
        published_dir: 已发布的模型目录（含 config.json + model.safetensors）
        quant_type: 量化类型
        progress_callback: 进度回调

    Returns:
        .gguf 文件路径
    """
    if not os.path.isdir(published_dir):
        raise FileNotFoundError(f"发布目录不存在: {published_dir}")

    # 读取元数据获取模型名
    meta_path = os.path.join(published_dir, "publish_metadata.json")
    model_name = "omnicore_model"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_name = meta.get("base_model", "omnicore_model").split("/")[-1]
            model_name = f"{model_name}_lora"
        except Exception:
            pass

    # 输出到同级的 gguf 目录
    parent_dir = os.path.dirname(published_dir)
    gguf_dir = os.path.join(parent_dir, "gguf_exports")
    os.makedirs(gguf_dir, exist_ok=True)

    return export_to_gguf(
        hf_model_dir=published_dir,
        output_dir=gguf_dir,
        output_name=model_name,
        quant_type=quant_type,
        progress_callback=progress_callback,
    )


def list_quant_options() -> dict:
    """返回所有可用量化选项"""
    return QUANT_OPTIONS
