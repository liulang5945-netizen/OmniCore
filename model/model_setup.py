"""
模型加载与配置模块
支持 DeepSeek R1-7B 等大模型，提供量化加载、LoRA 配置和检查点管理
设计为可动态切换模型，未来升级只需修改 model_name
"""
import gc
import logging
import os
import warnings
from typing import Optional

# 屏蔽 HuggingFace 缺少 Token 的烦人警告，以及 transformers 版本更迭的弃用提示
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.config import TrainingConfig
from core.memory_watchdog import memory_guarded

# ── 全局线程优化：使用物理核心数，但为 QtWebEngine 预留 1 个核心 ──
# 防止训练/推理时的密集 CPU 计算导致 WebEngine 子进程无 CPU 可用而崩溃重启
_cpu_logical = os.cpu_count() or 4
try:
    import psutil
    _cpu_physical = psutil.cpu_count(logical=False) or max(1, _cpu_logical // 2)
except ImportError:
    _cpu_physical = max(1, _cpu_logical // 2)
# 预留 1 个物理核心给 QtWebEngine / UI 使用
_cpu_count = max(1, _cpu_physical - 1)
torch.set_num_threads(_cpu_count)
os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))

logger = logging.getLogger("ModelSetup")


def _find_target_modules(model) -> list:
    """
    自动检测模型中可用于 LoRA 的线性层模块名。
    涵盖主流 Transformer 架构：
      - LLaMA / Qwen / Gemma / Mistral: q_proj, k_proj, v_proj, o_proj
      - MPT / Phi: Wqkv, out_proj
      - GPT-2 / GPT-Neo / GPT-J: c_attn, c_proj
      - BERT / T5: query, key, value
      - Falcon: query_key_value
      - SwiGLU FFN: gate_proj, up_proj, down_proj
    """
    import torch.nn as nn

    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "Wqkv", "out_proj",
        "c_attn", "c_proj",
        "query", "key", "value",
        "query_key_value",
    }
    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)

    if found:
        logger.info(f"自动检测到 LoRA 目标模块: {sorted(found)}")
        return sorted(found)

    # 回退：找所有非 Embedding / lm_head 的 Linear 层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf not in ("lm_head",):
                found.add(leaf)

    if found:
        logger.warning(f"未找到标准注意力模块，回退到所有 Linear 层: {sorted(found)}")
        return sorted(found)

    # 最终兜底
    logger.warning("无法检测目标模块，使用默认 LLaMA 风格模块名")
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _resolve_model_path(path: str) -> str:
    """
    统一解析模型路径（合并原版与增强版逻辑）

    支持格式:
    1. 直接包含 config.json 的目录 → 直接使用
    2. HuggingFace 缓存目录 (含 snapshots/) → 自动定位到 snapshot 子目录
    3. .gguf 文件或目录 → 返回 .gguf 文件路径
    4. 宽松 os.walk 深度搜索 config.json

    Args:
        path: 用户选择的目录路径

    Returns:
        解析后的有效模型路径（目录含 config.json 或 .gguf 文件路径）
    """
    from model.gguf_engine import find_gguf_file

    path = os.path.abspath(path).replace("\\", "/")

    if os.path.isfile(path):
        raise RuntimeError(
            f"❌ 不能直接选择文件！请选择大模型所在的【文件夹】。\n"
            f"您选择的是文件: {path}"
        )

    # 情况1: 直接包含 config.json → 直接使用
    if os.path.exists(os.path.join(path, "config.json")):
        return path

    # 情况2: HuggingFace 缓存目录格式 (models--org--model/snapshots/<hash>/)
    snapshots_dir = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots_dir):
        commits = [d for d in os.listdir(snapshots_dir)
                   if os.path.isdir(os.path.join(snapshots_dir, d))]
        if commits:
            commits.sort(reverse=True)
            snapshot_path = os.path.join(snapshots_dir, commits[0]).replace("\\", "/")
            if os.path.exists(os.path.join(snapshot_path, "config.json")):
                logger.info(f"自动解析 HuggingFace 缓存目录 -> {snapshot_path}")
                return snapshot_path

    # 情况3: 再深一层找 snapshots（例如 models--org--model/ 内部）
    for entry in os.listdir(path):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub):
            sub_snapshots = os.path.join(sub, "snapshots")
            if os.path.isdir(sub_snapshots):
                commits = [d for d in os.listdir(sub_snapshots)
                           if os.path.isdir(os.path.join(sub_snapshots, d))]
                if commits:
                    commits.sort(reverse=True)
                    snapshot_path = os.path.join(sub_snapshots, commits[0]).replace("\\", "/")
                    if os.path.exists(os.path.join(snapshot_path, "config.json")):
                        logger.info(f"自动解析 HuggingFace 缓存子目录 -> {snapshot_path}")
                        return snapshot_path

    # 情况4: 宽松 os.walk 深度搜索 config.json（优先于 GGUF）
    for root, dirs, files in os.walk(path):
        if "config.json" in files:
            logger.info(f"宽松搜索找到模型目录: {root}")
            return root

    # 情况5: .gguf 文件或包含 .gguf 的目录（仅在无 config.json 时回退）
    gguf_file = find_gguf_file(path)
    if gguf_file:
        logger.info(f"检测到 GGUF 模型文件（无 config.json）: {gguf_file}")
        return gguf_file

    # 全都找不到
    raise RuntimeError(
        f"❌ 无法找到有效的模型文件！\n"
        f"路径: {path}\n\n"
        f"请检查所选目录是否包含:\n"
        f"  - config.json（HuggingFace 格式模型）\n"
        f"  - .gguf 文件（GGUF 量化模型）\n"
        f"提示: 如果是 HuggingFace 缓存目录，选择包含 snapshots/ 的文件夹即可"
    )


def _looks_like_local_path(path: str) -> bool:
    """
    判断字符串是否"看起来像"本地文件系统路径。
    这样即使路径不存在，也不会被误传到 HuggingFace Hub 当作远程仓库名校验。
    """
    import platform
    if platform.system() == "Windows":
        # Windows: 盘符 + 冒号（如 C:\）或 UNC 路径（\\server\share）
        return len(path) >= 2 and path[1] == ":" or path.startswith("\\\\")
    else:
        # Linux/macOS: 绝对路径以 / 开头，或相对路径包含 /
        return path.startswith("/") or path.startswith("./") or path.startswith("../")


def _check_enable_quant(config: TrainingConfig, device: str) -> bool:
    """
    判断是否应该启用 4-bit / 8-bit 量化
    放宽限制：即使不是 CUDA（如 ROG ALLY 的集成显卡、MPS 等），
    只要用户手动启用了量化配置，就允许使用。
    """
    if config.load_in_4bit or config.load_in_8bit:
        return True  # 用户明确要求量化
    if device == "cuda" or (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return True  # CUDA 设备默认启用
    return False


@memory_guarded(min_avail_pct=0.30, on_critical='raise')
def download_and_load_model(config: TrainingConfig):
    """
    下载并加载模型和 tokenizer
    支持 4-bit/8-bit 量化加载以适配 7B 等大模型
    新增：自动解析 HuggingFace 缓存目录，放宽量化启用条件（非CUDA也可手动启用）
    """
    model_name = config.model_name
    cache_dir = config.cache_dir
    device = config.resolve_device()
    use_quant = _check_enable_quant(config, device)

    logger.info(f"加载模型: {model_name}")
    logger.info(f"缓存目录: {cache_dir or '默认'}")
    logger.info(f"设备: {device}")

    # 修复本地路径读取问题：自动解析 HuggingFace 缓存目录格式
    if os.path.exists(model_name):
        model_name = _resolve_model_path(model_name)
        logger.info(f"解析后的模型路径: {model_name}")
    elif _looks_like_local_path(model_name):
        raise RuntimeError(
            f"❌ 模型路径不存在！\n"
            f"   配置的路径: {model_name}\n\n"
            f"   请检查:\n"
            f"   1. 路径是否正确（注意大小写和拼写）\n"
            f"   2. 模型文件是否已被移动或删除\n"
            f"   3. 如果是 HuggingFace 远程模型，请使用仓库名格式（如 'unsloth/DeepSeek-R1-Distill-Qwen-7B'）\n"
        )

    # ⭐ 安全检查：如果解析到 .gguf 文件，说明用户应使用 GGUF 模式加载
    if model_name.lower().endswith('.gguf'):
        import json as _json
        from core.utils import get_external_path
        error_msg = (
            f"❌ 该目录下只有 .gguf 量化模型文件，无法通过 HuggingFace Transformers 加载。\n"
            f"   GGUF 文件: {model_name}\n\n"
            f"🔧 系统已自动切换为 GGUF 推理引擎，请重启 OmniCore 使设置生效。"
        )
        try:
            settings_path = get_external_path("app_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    saved = _json.load(f)
                saved["model_type"] = "gguf"
                saved["gguf_path"] = model_name
                with open(settings_path, "w", encoding="utf-8") as f:
                    _json.dump(saved, f, ensure_ascii=False, indent=2)
                logger.info("✅ 已自动修复 app_settings.json，切换为 GGUF 模式")
        except Exception as _e:
            logger.warning(f"自动修复 settings 失败: {_e}")
        raise RuntimeError(error_msg)

    torch_dtype = config.get_torch_dtype()

    # 量化配置（非 CUDA 设备也可手动启用，例如 ROG ALLY 的集成显卡）
    bnb_config = None
    if use_quant:
        if config.load_in_4bit:
            logger.info("启用 4-bit 量化加载（内存占用降 75%，推荐 16GB 内存设备使用）")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif config.load_in_8bit:
            logger.info("启用 8-bit 量化加载（内存占用降 50%）")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        err_msg = str(e)
        if "sentencepiece" in err_msg or "tiktoken" in err_msg:
            raise RuntimeError(
                f"加载 tokenizer 失败: 缺少后端库。\n"
                f"请执行以下命令安装:\n"
                f"  pip install sentencepiece tiktoken\n"
                f"原始错误: {e}"
            ) from e
        raise RuntimeError(f"加载 tokenizer 失败: {e}") from e

    try:
        # 在非CUDA设备上尝试用 4-bit 量化时，需要处理 device_map
        # bitsandbytes 在 CPU 上也能工作，但 device_map="auto" 只在 CUDA 有效
        if device == "cuda":
            model_kwargs = {"device_map": "auto"}
        elif use_quant:
            # 非CUDA + 量化：让模型在CPU加载，然后model.to(device)处理
            model_kwargs = {}
        else:
            model_kwargs = {}

        # 尝试为安培(Ampere)及以上架构显卡(如 RTX 30/40 系)开启 Flash Attention 2 加速
        if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("⚡ 已启用 Flash Attention 2 加速训练与推理")
            except ImportError:
                logger.info("未安装 flash_attn，建议终端执行 `pip install flash-attn --no-build-isolation` 以获取 2-4 倍加速")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}") from e

    # 非CUDA设备需要手动移动到目标设备（但量化后的模型不能直接 .to()）
    if device not in ("cuda", "auto") and not use_quant:
        model = model.to(device)

    logger.info(f"模型加载成功！参数量: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer



def setup_model_for_training(model, config: TrainingConfig):
    """配置模型为训练模式（支持 LoRA）+ torch.compile 图优化加速"""
    device = config.resolve_device()

    if not config.use_lora:
        logger.info("使用全参数微调模式")
        return _maybe_compile(model, device)

    try:
        from peft import get_peft_model, LoraConfig, TaskType

        logger.info(
            f"使用 LoRA 微调模式 (rank={config.lora_r}, alpha={config.lora_alpha}, "
            f"dropout={config.lora_dropout})"
        )
        target_modules = _find_target_modules(model)

        logger.info(f"正在创建 LoRA 配置，目标模块: {target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        logger.info("正在应用 LoRA 适配器到模型（遍历模块树中，大模型可能需要 1-3 分钟）...")
        model = get_peft_model(model, lora_config)
        logger.info("LoRA 适配器已成功应用到模型！")

        model = _maybe_compile(model, device)

        logger.info("正在统计可训练参数量...")
        model.print_trainable_parameters()
        return model
    except ImportError:
        logger.warning("未安装 peft 库，回退到全参数微调。pip install peft 可启用 LoRA")
        return _maybe_compile(model, device)


def _maybe_compile(model, device: str):
    """
    安全启用 torch.compile() 图优化。必须在 LoRA 之后调用。

    设备策略:
    - CUDA: 跳过。GPU 已有 fp16+AMP+FlashAttn+cuBLAS，compile 额外创建计算图副本
      (≈1×模型权重)，7B 模型额外消耗 10-14GB 显存，极易 OOM 卡死。
    - CPU: 跳过。JIT 编译大型 Transformer（≥1B）需 20-40GB 临时内存，
      极易触发系统 OOM/swap 导致崩溃；且编译后的融合 kernel 在 CPU 上
      加速效果不稳定（0.3-1.5×），远不抵编译开销。
    - MPS: 跳过，兼容性不完整。
    """
    import torch as _torch

    if not hasattr(_torch, "compile"):
        return model

    # 所有设备均跳过 torch.compile：
    # - CUDA/MPS: 已知兼容性问题 + 额外显存开销
    # - CPU: JIT 编译大型模型会爆内存、卡死 UI 线程、编译后加速微乎其微
    if device in ("mps", "cuda", "cpu"):
        if device == "cpu":
            logger.info("CPU 训练已跳过 torch.compile（避免 JIT 编译爆内存/卡 UI）")
        return model

    if _has_quantized_weights(model):
        logger.info("检测到量化模型（bitsandbytes），跳过 torch.compile（不兼容）")
        return model

    try:
        model = _torch.compile(model, mode="default")
        logger.info(f"⚡ torch.compile(mode='default') 已启用（图优化加速）")
    except Exception as _compile_err:
        logger.warning(f"torch.compile() 启用失败（非致命）: {_compile_err}")

    return model


def _has_quantized_weights(model) -> bool:
    """检测模型是否使用 bitsandbytes 量化权重（与 torch.compile 不兼容）"""
    try:
        for _name, _mod in model.named_modules():
            _cls = type(_mod).__name__
            if "Linear4bit" in _cls or "Linear8bitLt" in _cls:
                return True
    except Exception:
        pass
    return False


def get_optimizer(model, config: TrainingConfig):
    """创建优化器（LoRA 模式下只优化可训练参数，大幅减少遍历开销）"""
    from torch.optim import AdamW

    logger.info(f"正在创建优化器，遍历模型参数...")
    no_decay = ["bias", "LayerNorm.weight"]

    # LoRA 或部分微调模式下，只遍历 requires_grad=True 的参数
    # 避免对 7B+ 全量参数做无效扫描（全量遍历可能需要数分钟）
    trainable_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]

    if not trainable_params:
        # 回退：如果所有参数都冻结，使用全量参数（理论上不应发生）
        logger.warning("未检测到 requires_grad=True 的参数，回退到全量参数遍历")
        trainable_params = list(model.named_parameters())

    decay_params = [
        p for n, p in trainable_params
        if not any(nd in n for nd in no_decay)
    ]
    no_decay_params = [
        p for n, p in trainable_params
        if any(nd in n for nd in no_decay)
    ]

    logger.info(
        f"优化器参数统计: 正则化衰减 {len(decay_params)} 组, "
        f"无衰减 {len(no_decay_params)} 组 "
        f"(共 {len(trainable_params)}/{sum(1 for _ in model.parameters())} 可训练)"
    )

    optimizer_grouped_parameters = []
    if decay_params:
        optimizer_grouped_parameters.append({
            "params": decay_params,
            "weight_decay": config.weight_decay,
        })
    if no_decay_params:
        optimizer_grouped_parameters.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
        })

    return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)


def get_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    """
    创建学习率调度器
    结合 Warmup 和 Cosine 衰减
    """
    from transformers import get_cosine_schedule_with_warmup

    warmup_steps = min(config.warmup_steps, total_steps // 10)
    logger.info(f"调度器: Cosine + Warmup={warmup_steps} / Total={total_steps}")

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def save_checkpoint(model, optimizer, scheduler, config: TrainingConfig,
                    epoch: int, step: int, loss: float, output_dir: str,
                    dataset_files: list = None):
    """保存检查点，新增 dataset_files 参数以便断点续训时恢复数据集列表"""
    import json

    os.makedirs(output_dir, exist_ok=True)
    filename = f"checkpoint-e{epoch+1}-s{step}.pt"
    ckpt_path = os.path.join(output_dir, filename)

    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.__dict__,
        "dataset_files": dataset_files or [],  # 保存数据集文件列表，用于断点续训
    }

    torch.save(state, ckpt_path)
    logger.info(f"检查点已保存: {ckpt_path}")

    # 保存配置副本
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device: str):
    """从检查点恢复模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    logger.info(f"从检查点恢复: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    missing_keys, unexpected_keys = model.load_state_dict(
        state["model_state_dict"], strict=False
    )
    if missing_keys:
        logger.warning(f"缺失的键: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"意外的键: {unexpected_keys[:5]}...")

    if optimizer and state.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"优化器状态加载失败（可能因模型结构变更）: {e}")

    if scheduler and state.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态加载失败: {e}")

    logger.info(f"恢复检查点: epoch={state.get('epoch', '?')}, "
                f"step={state.get('step', '?')}, loss={state.get('loss', '?'):.4f}")
    return state.get("epoch", 0), state.get("step", 0), state.get("loss", float("inf"))


def merge_and_save_lora_model(config: TrainingConfig, output_dir: str,
                               progress_callback: callable = None) -> str:
    """
    将 LoRA 微调后的模型权重合并到基座模型，并保存为完整模型
    支持 Publish（发布）流程

    Args:
        config: 训练配置
        output_dir: 合并后模型的保存目录
        progress_callback: 进度回调函数(desc: str, fraction: float)

    Returns:
        保存路径
    """
    import json

    def _report(desc, frac):
        if progress_callback:
            try:
                progress_callback(desc, frac)
            except Exception:
                pass
        logger.info(f"[{frac*100:.0f}%] {desc}")

    logger.info(f"开始合并 LoRA 权重，输出到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 找到最新的检查点
        _report("🔍 查找最新训练检查点...", 0.02)
        import glob
        checkpoint_dir = config.output_dir
        pattern = os.path.join(checkpoint_dir, "checkpoint-e*.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"未找到检查点文件: {pattern}")

        latest_ckpt = ckpts[-1]
        _report(f"📦 使用检查点: {os.path.basename(latest_ckpt)}", 0.05)

        # 预加载检查点，读取训练时使用的 LoRA 参数（必须与保存的权重匹配）
        _report("📥 加载检查点权重到内存...", 0.08)
        _ckpt_state_pre = torch.load(latest_ckpt, map_location="cpu", weights_only=True)
        _saved_config = _ckpt_state_pre.get("config", {})
        _actual_lora_r = _saved_config.get("lora_r", config.lora_r)
        _actual_lora_alpha = _saved_config.get("lora_alpha", config.lora_alpha)
        _actual_lora_dropout = _saved_config.get("lora_dropout", config.lora_dropout)
        _report(f"📐 LoRA 参数: r={_actual_lora_r}, alpha={_actual_lora_alpha}", 0.10)

        # 2. 重新加载基础模型（无量化，以便合并后可以保存为 torch.float16）
        _report("🔄 加载基础模型（这可能需要几分钟，请耐心等待）...", 0.12)
        from transformers import AutoModelForCausalLM
        device_str = config.resolve_device()
        if device_str == "cuda" and torch.cuda.is_available():
            model_kwargs = {"device_map": "auto"}
        else:
            model_kwargs = {}
        _merge_dtype = torch.float16 if (device_str == "cuda" and torch.cuda.is_available()) else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            torch_dtype=_merge_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )
        if device_str not in ("cuda", "auto") or not torch.cuda.is_available():
            base_model = base_model.to(device_str)
        _report("✅ 基础模型加载完成", 0.40)

        # 3. 重建 LoRA 模型并加载检查点权重（使用检查点中保存的参数，确保与权重形状匹配）
        _report("🔧 正在重建 LoRA 模型结构...", 0.42)
        from peft import get_peft_model, LoraConfig, TaskType
        # 优先从 checkpoint 配置中读取训练时实际使用的 target_modules
        # 回退到自动检测基础模型的模块名
        _saved_target_modules = _saved_config.get("lora_target_modules")
        if _saved_target_modules:
            _merge_target_modules = _saved_target_modules
            logger.info(f"从 checkpoint 恢复 target_modules: {_merge_target_modules}")
        else:
            _merge_target_modules = _find_target_modules(base_model)
            logger.info(f"自动检测 target_modules: {_merge_target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=_actual_lora_r,
            lora_alpha=_actual_lora_alpha,
            lora_dropout=_actual_lora_dropout,
            target_modules=_merge_target_modules,
            bias="none",
        )
        lora_model = get_peft_model(base_model, lora_config)
        
        # 加载训练后的 LoRA 权重
        _report("📥 加载训练后的 LoRA 权重...", 0.46)
        lora_model.load_state_dict(_ckpt_state_pre["model_state_dict"], strict=False)
        _report("✅ LoRA 权重加载完成", 0.50)

        # 4. 合并权重
        _report("🔄 正在合并 LoRA 权重到基础模型...", 0.52)
        merged_model = lora_model.merge_and_unload()
        _report("✅ LoRA 权重合并完成！", 0.70)

        # 5. 保存合并后的完整模型
        _report("💾 正在保存合并后的模型文件...", 0.72)
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        _report("✅ 模型文件保存完成", 0.85)

        # 6. 保存 tokenizer
        _report("💾 正在保存 tokenizer...", 0.87)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(output_dir)

        # 7. 写入发布元信息
        _report("📝 写入发布元信息...", 0.92)
        metadata = {
            "base_model": config.model_name,
            "lora_config": {
                "r": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
            },
            "training_params": {
                "epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            },
            "checkpoint_source": latest_ckpt,
            "publish_time": __import__("datetime").datetime.now().isoformat(),
        }
        meta_path = os.path.join(output_dir, "publish_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        _report("✅ 模型发布完成！", 1.0)
        logger.info(f"模型已发布到: {output_dir}")
        return output_dir

    except ImportError:
        raise RuntimeError("需要安装 peft 库才能合并 LoRA 权重: pip install peft")
    except Exception as e:
        logger.error(f"模型合并发布失败: {e}")
        raise


def list_published_models(base_dir: str) -> list:
    """列出所有已发布的模型"""
    import glob
    models = []
    if not os.path.exists(base_dir):
        return models

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        meta_path = os.path.join(entry_path, "publish_metadata.json")
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "config.json")):
            info = {
                "name": entry,
                "path": entry_path,
            }
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    info["metadata"] = meta
                except Exception:
                    info["metadata"] = {}
            models.append(info)

    return sorted(models, key=lambda x: x.get("metadata", {}).get("publish_time", ""), reverse=True)


# ======================== GGUF 模型加载 ========================

@memory_guarded(min_avail_pct=0.30, on_critical='raise')
def load_gguf_model(config: TrainingConfig):
    """
    加载 GGUF 量化模型（用于低内存设备的本地推理）
    通过 llama-cpp-python + Vulkan 实现 AMD/NVIDIA/Intel 显卡加速

    Args:
        config: 训练配置，需包含:
            - config.gguf_path: .gguf 文件路径
            - config.n_gpu_layers: GPU 加速层数 (-1=全部, 0=纯CPU)
            - config.n_ctx: 上下文窗口大小

    Returns:
        (gguf_engine, dummy_tokenizer) 元组
        - gguf_engine: BaseGGUFEngine 实例
        - dummy_tokenizer: 兼容 HuggingFace 接口的简易 tokenizer
    """
    from model.gguf_engine import BaseGGUFEngine, find_gguf_file, is_gguf_model

    if not config.gguf_path:
        raise ValueError("GGUF 模型路径未设置！请在设置中选择 .gguf 文件。")

    # 查找 .gguf 文件（支持选择文件本身或所在目录）
    gguf_file = find_gguf_file(config.gguf_path)
    if not gguf_file:
        raise FileNotFoundError(
            f"未找到 .gguf 文件！\n"
            f"选择的路径: {config.gguf_path}\n\n"
            f"请确保:\n"
            f"1. 文件以 .gguf 结尾\n"
            f"2. 文件路径中不包含特殊字符\n"
            f"3. 文件未被其他程序占用"
        )

    logger.info(f"发现 GGUF 模型文件: {gguf_file}")
    logger.info(f"GPU 层数: {config.n_gpu_layers} ({'全部' if config.n_gpu_layers == -1 else config.n_gpu_layers})")
    logger.info(f"上下文窗口: {config.n_ctx}")

    # 创建 GGUF 引擎
    engine = BaseGGUFEngine(
        model_path=gguf_file,
        n_gpu_layers=config.n_gpu_layers,
        n_ctx=config.n_ctx,
    )

    # 创建简易 tokenizer 兼容接口（方便 api_server.py 统一处理）
    class GGUFCompatibleTokenizer:
        """包装 GGUF 引擎，提供与 HuggingFace tokenizer 兼容的接口"""
        def __init__(self, gguf_engine):
            self.engine = gguf_engine
            self.pad_token_id = 0
            self.eos_token_id = 0
            self.bos_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [], "attention_mask": []}

        def decode(self, token_ids, **kwargs):
            return "" if not token_ids else str(token_ids)

        def apply_chat_template(self, messages, **kwargs):
            """将 OpenAI 格式消息转换为 GGUF 提示词"""
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            dummy_config = {"gguf_model": True, "engine": "llama-cpp-python"}
            import json
            with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
                json.dump(dummy_config, f)

    dummy_tokenizer = GGUFCompatibleTokenizer(engine)
    logger.info(f"GGUF 模型加载成功！引擎: llama-cpp-python, 文件: {os.path.basename(gguf_file)}")

    return engine, dummy_tokenizer


def clear_gpu_memory():
    """清理 GPU 显存（同时清理 CUDA 和 GGUF 缓存）"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU 显存已清理")
