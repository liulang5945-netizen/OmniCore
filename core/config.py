"""
OmniCore 全局配置模块
集中管理所有配置，避免硬编码
"""
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict


# ======================== 镜像源配置 ========================

# HuggingFace 镜像源（可通过环境变量覆盖）
# 海外用户: 保持为空或设置为 https://huggingface.co
# 国内用户: 设置为 https://hf-mirror.com
HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT",
    os.environ.get("HF_MIRROR_URL", "https://hf-mirror.com")
)

# PyTorch 镜像源
TORCH_MIRROR = os.environ.get("TORCH_MIRROR", "")


def get_hf_endpoint() -> str:
    """获取 HuggingFace 端点 URL"""
    return HF_ENDPOINT


# ======================== 路径配置 ========================

def get_external_path(relative_path: str) -> str:
    """外部资源路径（用户数据目录）"""
    base_path = get_writable_base_dir()
    return os.path.join(base_path, relative_path)


def get_writable_base_dir() -> str:
    """
    检测并返回可写的数据根目录。
    
    策略：
    1. 如果安装目录可写，直接返回安装目录
    2. 如果安装目录不可写（如装到了 C:\\Program Files），
       自动降级到 %LOCALAPPDATA%\\OmniCore
    3. 如果 LocalAppData 也不可写（极罕见），回退到用户目录
    """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    
    # 测试安装目录是否可写
    test_path = os.path.join(base_path, '.write_test')
    try:
        with open(test_path, 'w') as f:
            f.write('test')
        os.remove(test_path)
        return base_path  # 可写，直接返回
    except (PermissionError, OSError):
        pass  # 不可写，降级到 LocalAppData
    
    # 降级到 %LOCALAPPDATA%\OmniCore
    fallback = os.path.join(
        os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
        'OmniCore'
    )
    try:
        os.makedirs(fallback, exist_ok=True)
        # 同样测试一下 fallback 是否可写
        test_fb = os.path.join(fallback, '.write_test')
        with open(test_fb, 'w') as f:
            f.write('test')
        os.remove(test_fb)
        return fallback
    except (PermissionError, OSError):
        # 终极回退到用户目录
        ultimate = os.path.join(os.path.expanduser('~'), 'OmniCore')
        os.makedirs(ultimate, exist_ok=True)
        return ultimate


def get_internal_path(relative_path: str) -> str:
    """内部资源路径（打包后只读资源）"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ======================== 服务配置 ========================

# FastAPI 后端配置
API_HOST = os.environ.get("OMNICORE_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("OMNICORE_PORT", "8000"))

# 日志级别
LOG_LEVEL = os.environ.get("OMNICORE_LOG_LEVEL", "INFO").upper()

# 模型加载超时（秒）
MODEL_LOAD_TIMEOUT = int(os.environ.get("OMNICORE_MODEL_TIMEOUT", "300"))


# ======================== 应用环境变量 ========================

def apply_env_overrides():
    """
    应用环境变量覆盖（在模型/框架加载前调用）
    仅当未设置时，才使用默认值
    """
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT


# ======================== 模型与训练配置 ========================

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = ""
    cache_dir: str = ""
    resume_from_checkpoint: str = ""
    device: str = "auto"
    model_type: str = "huggingface"  # "huggingface" | "gguf" | "self"
    gguf_path: str = ""              # .gguf 模型文件路径
    n_gpu_layers: int = -1           # -1 = 全部放 GPU, 0 = 纯 CPU, N = 前 N 层放 GPU
    n_ctx: int = 2048                # GGUF 模型的上下文窗口大小

    
    # 训练配置
    train_file: str = ""
    output_dir: str = "checkpoints"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 3
    validation_split: float = 0.0       # 验证集比例（0.0 = 不使用验证集）
    early_stopping_patience: int = 0    # 早停耐心值（0 = 不启用早停）
    early_stopping_threshold: float = 1e-4  # 早停最小改善阈值
    use_tensorboard: bool = False       # 是否启用 TensorBoard 日志
    tensorboard_dir: str = "tb_logs"    # TensorBoard 日志目录
    
    # LoRA 配置
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 量化加载配置
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # → 自适应诊断结果（由 auto_configure_for_hardware 填充，供前端展示）
    _hw_diag: dict = None

    # ======================== 系统内存检测 ========================

    @staticmethod
    def get_total_ram_gb() -> float:
        """获取系统总内存（GB）。
        
        psutil 是核心依赖（OmniCore 始终安装），
        直接使用 psutil 读取，不再保留 32 位时代的 ctypes 回退代码。
        """
        import psutil as _ps
        return _ps.virtual_memory().total / (1024 ** 3)

    @staticmethod
    def detect_low_memory_system() -> bool:
        """检测是否为低内存系统（64 位系统上以 24GB 为分界）"""
        ram_gb = TrainingConfig.get_total_ram_gb()
        return ram_gb <= 24.0

    # ======================== 设备与精度自动匹配 ========================

    def resolve_device(self) -> str:
        """自动判断最优的运算设备（委托给 core.hardware）"""
        from core.hardware import resolve_device as _resolve
        return _resolve(self)

    # ======================== 硬件自适应配置（模型 + 硬件双感知） ========================

    def _estimate_params_b(self, loaded_model=None) -> tuple:
        """四层递进估算模型参数量（委托给 core.hardware）"""
        from core.hardware import estimate_params_b as _estimate
        return _estimate(self, loaded_model)

    def auto_configure_for_hardware(self, loaded_model=None) -> dict:
        """纯公式驱动的硬件自适应配置（委托给 core.hardware）"""
        from core.hardware import auto_configure_for_hardware as _configure
        return _configure(self, loaded_model)

    def get_torch_dtype(self):
        """根据实际训练设备自动匹配最优计算精度（委托给 core.hardware）"""
        from core.hardware import get_torch_dtype as _get_dtype
        return _get_dtype(self)


def get_config(args=None) -> TrainingConfig:
    """生成或解析配置（支持命令行参数覆盖默认值）"""
    config = TrainingConfig()
    if args is None:
        return config
    for field_name in (
        "model_name", "cache_dir", "gguf_path",
        "batch_size", "num_epochs", "learning_rate",
        "load_in_4bit", "load_in_8bit", "use_lora",
        "lora_r", "lora_alpha",
        "output_dir", "dataset_path",
        "n_gpu_layers", "n_ctx",
    ):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)
    return config


def save_config(config: TrainingConfig, path: str):
    """将当前训练配置固化到 JSON 文件中（用于断点恢复）"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
