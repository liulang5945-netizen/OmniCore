"""
模型加载器
从 api/app.py 提取：启动时模型加载、自动下载、GGUF/HF 识别、RAG 索引构建
"""
import json
import logging
import os
import sys

from core.config import TrainingConfig
from core.app_state import app_state
from core.utils import get_external_path

logger = logging.getLogger("ModelLoader")

# 启动下载进度（供前端轮询）
startup_download_progress = {
    "active": False, "percent": 0, "message": "", "total_mb": 0, "downloaded_mb": 0
}


def load_model_on_startup():
    """在后台线程中加载模型（原 api/app.py _async_load_model）"""
    import asyncio
    asyncio.run(_async_load_model())


async def _async_load_model():
    """异步加载模型"""
    app_state.mark_starting()
    try:
        config = TrainingConfig()
        config.cache_dir = get_external_path("model_cache")
        config.resume_from_checkpoint = get_external_path("final_checkpoint.pt")

        settings_path = get_external_path("app_settings.json")
        settings = _load_settings(settings_path)
        _apply_settings_to_config(config, settings)

        # 安装后首次自动下载模型
        if _should_auto_download(settings):
            _perform_auto_download(config, settings, settings_path)

        # 判断模型类型并加载
        _detect_and_load_model(config, settings_path)

        # 自动构建 RAG 索引
        _build_rag_index()

        app_state.mark_started()
    except Exception as e:
        import traceback
        error_msg = f"模型加载失败: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        app_state.mark_startup_failed(error_msg)


def _load_settings(settings_path: str) -> dict:
    """加载应用设置"""
    settings = {}
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    return settings


def _apply_settings_to_config(config: TrainingConfig, settings: dict):
    """将设置应用到训练配置"""
    config.device = settings.get("device", "auto")
    if "model_name" in settings and settings["model_name"].strip():
        config.model_name = settings["model_name"].strip()
    if "load_in_4bit" in settings:
        config.load_in_4bit = bool(settings["load_in_4bit"])
    if "load_in_8bit" in settings:
        config.load_in_8bit = bool(settings["load_in_8bit"])
    if "model_type" in settings:
        config.model_type = settings["model_type"]
    if "gguf_path" in settings:
        config.gguf_path = settings["gguf_path"]
    if "n_gpu_layers" in settings:
        config.n_gpu_layers = int(settings["n_gpu_layers"])
    if "n_ctx" in settings:
        config.n_ctx = int(settings["n_ctx"])


def _should_auto_download(settings: dict) -> bool:
    """检查是否需要自动下载模型"""
    return (settings.get("gguf_download_pending")
            and settings.get("gguf_model_key")
            and not settings.get("gguf_path"))


def _perform_auto_download(config: TrainingConfig, settings: dict, settings_path: str):
    """执行首次自动下载"""
    try:
        model_key = settings["gguf_model_key"]
        quant = settings.get("gguf_quant", "Q4_K_M")
        save_dir = settings.get("gguf_dir", get_external_path("gguf_models"))
        logger.info(f"📦 检测到安装时选择的模型，正在自动下载: {model_key} (量化: {quant})")

        from model.model_registry import get_model_download_info, get_all_models
        from model.model_downloader import ModelDownloader, DownloadProgress

        info = get_model_download_info(model_key, quant)
        if not info:
            info = _find_model_info_fallback(model_key, quant)
        if not info:
            logger.warning(f"⚠️ 无法解析模型下载信息: model_key={model_key}, quant={quant}")
            return

        os.makedirs(save_dir, exist_ok=True)
        startup_download_progress["active"] = True
        startup_download_progress["message"] = f"📥 正在下载 {os.path.basename(info['filename'])}..."

        def _startup_progress_cb(p: DownloadProgress):
            startup_download_progress["percent"] = round(p.percent, 1)
            startup_download_progress["total_mb"] = round(p.total_mb, 1)
            startup_download_progress["downloaded_mb"] = round(p.downloaded_mb, 1)
            startup_download_progress["message"] = (
                f"📥 下载模型 {p.filename} ... {p.percent:.0f}% "
                f"({p.downloaded_mb:.0f}/{p.total_mb:.0f} MB)"
            )

        downloader = ModelDownloader(save_dir=save_dir, mirror=True, verify_ssl=True)
        file_path = downloader.download_file(
            repo_id=info["repo"],
            filename=info["filename"],
            model_name=model_key,
            progress_callback=_startup_progress_cb,
        )
        startup_download_progress["active"] = False

        if file_path and os.path.exists(file_path):
            config.gguf_path = file_path
            config.model_type = "gguf"
            settings["gguf_path"] = file_path
            settings["model_type"] = "gguf"
            settings["n_gpu_layers"] = -1
            settings["n_ctx"] = 2048
            settings["gguf_download_pending"] = False
            settings["gguf_dir"] = save_dir
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 安装后自动下载完成: {file_path}")
        else:
            logger.warning("⚠️ 自动下载未获得有效文件，将以无模型状态启动")
    except Exception as auto_dl_err:
        logger.warning(f"⚠️ 安装后自动下载模型失败（以无模型状态启动）: {auto_dl_err}")


def _find_model_info_fallback(model_key: str, quant: str) -> dict:
    """回退查找模型下载信息"""
    from model.model_registry import get_all_models

    for entry in get_all_models():
        if entry.hf_repo == model_key:
            for v in entry.variants:
                if v.quant == quant:
                    return {"repo": entry.hf_repo, "filename": v.hf_filename}
            rec = entry.recommended_variant()
            if rec:
                return {"repo": entry.hf_repo, "filename": rec.hf_filename}
    return None


def _detect_and_load_model(config: TrainingConfig, settings_path: str):
    """检测模型类型并加载"""
    from model.gguf_engine import is_gguf_model, find_gguf_file
    from model.model_setup import load_gguf_model, download_and_load_model
    from model.trainer import BaseInferenceEngine

    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)

    # 检测路径类型
    model_path_is_taiji_dir = _check_taiji_dir(config.model_name)
    model_path_is_hf_dir = _check_hf_dir(config.model_name)
    model_path_is_gguf_dir = False

    # 态极模型自动识别：如果目录是态极格式但 model_type 未设置为 "self"，自动切换
    if model_path_is_taiji_dir and config.model_type != "self":
        logger.info(f"🧬 自动检测到态极 ModelSelf 模型目录，自动切换为态极模式")
        config.model_type = "self"
        _auto_fix_settings(settings_path, "self", "")

    if config.model_name and os.path.isdir(config.model_name) and not model_path_is_hf_dir:
        gguf_in_model_dir = find_gguf_file(config.model_name, recursive=False)
        if gguf_in_model_dir:
            logger.info(f"🔍 自动检测到 model_name 目录下仅含 .gguf 文件，自动切换为 GGUF 模式")
            config.gguf_path = gguf_in_model_dir
            config.model_type = "gguf"
            model_path_is_gguf_dir = True
            _auto_fix_settings(settings_path, "gguf", gguf_in_model_dir)

    gguf_path_valid = config.gguf_path and os.path.exists(config.gguf_path) and is_gguf_model(config.gguf_path)

    # 处理 model_type 不匹配
    if config.model_type == "gguf" and not gguf_path_valid:
        if model_path_is_hf_dir:
            logger.warning(f"⚠️ model_type 被设为 gguf，但路径是 HF 模型目录，自动降级为 huggingface")
            config.model_type = "huggingface"
            config.gguf_path = ""
            _auto_fix_settings(settings_path, "huggingface", "")
        else:
            logger.warning(f"⚠️ model_type=gguf 但 gguf_path 无效，且无可用 HF 模型，无模型启动")
            app_state.mark_started()
            return

    if config.model_type == "gguf" or gguf_path_valid:
        config.model_type = "gguf"

    # 加载模型
    if config.model_type == "gguf":
        _load_gguf_model(config)
    elif config.model_type == "self":
        _load_self_model(config)
    else:
        _load_hf_model(config)


def _check_hf_dir(model_name: str) -> bool:
    """检查路径是否为 HuggingFace 模型目录（排除态极 ModelSelf 目录）"""
    if not model_name or not os.path.isdir(model_name):
        return False
    config_path = os.path.join(model_name, "config.json")
    if os.path.exists(config_path):
        # 排除态极 ModelSelf 目录
        if _check_taiji_dir(model_name):
            return False
        return True
    # 浅层搜索前 2 层
    for _root, _dirs, _files in os.walk(model_name):
        if "config.json" in _files:
            cp = os.path.join(_root, "config.json")
            try:
                with open(cp, "r") as f:
                    cfg = json.load(f)
                # 态极专属字段存在则排除
                if "base_vocab_size" in cfg or "num_special_tokens" in cfg:
                    return False
            except Exception:
                pass
            return True
        _depth = _root[len(model_name):].count(os.sep)
        if _depth >= 2:
            _dirs.clear()
    return False


def _check_taiji_dir(model_path: str) -> bool:
    """
    检查目录是否为态极 ModelSelf 原生模型目录。
    判断依据：config.json 中包含态极专属字段 base_vocab_size 或 num_special_tokens。
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        # 态极专属字段
        return "base_vocab_size" in cfg or "num_special_tokens" in cfg
    except Exception:
        return False


def _auto_fix_settings(settings_path: str, model_type: str, gguf_path: str):
    """自动修复设置文件"""
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved["model_type"] = model_type
            if gguf_path:
                saved["gguf_path"] = gguf_path
            else:
                saved.pop("gguf_path", None)
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(saved, f, ensure_ascii=False, indent=2)
            logger.info("✅ 已自动修复 app_settings.json")
    except Exception:
        pass


def _load_gguf_model(config: TrainingConfig):
    """加载 GGUF 模型"""
    from model.model_setup import load_gguf_model

    if not config.gguf_path or not os.path.exists(config.gguf_path):
        logger.info("未检测到有效模型文件，将以【无模型状态】启动后台...")
        app_state.mark_started()
        return

    logger.info(f"使用 GGUF 引擎加载量化模型")
    logger.info(f"  GGUF 路径: {config.gguf_path}")
    logger.info(f"  GPU 层数: {'全部' if config.n_gpu_layers == -1 else config.n_gpu_layers}")
    logger.info(f"  上下文: {config.n_ctx}")

    model, tokenizer = load_gguf_model(config)
    trainer = model
    app_state.update_model(model, tokenizer, trainer, config.gguf_path or "GGUF模型")


def _load_hf_model(config: TrainingConfig):
    """加载 HuggingFace 模型"""
    from model.model_setup import download_and_load_model
    from model.trainer import BaseInferenceEngine

    if not getattr(config, "model_name", None) or config.model_name.lower() in ["none", ""]:
        logger.info("未检测到有效模型配置，将以【无模型状态】启动后台...")
        app_state.mark_started()
        return

    _hw_diag = config.auto_configure_for_hardware()
    device = config.resolve_device()

    # 内存安全检查
    model_path_str = config.model_name
    estimated_params_b = _hw_diag.get("model_params_b", 0) if _hw_diag else 0
    if os.path.isdir(model_path_str) and estimated_params_b > 0:
        total_ram = TrainingConfig.get_total_ram_gb()
        need_ram_fp16 = estimated_params_b * 2.2
        if not config.load_in_4bit and not config.load_in_8bit:
            if need_ram_fp16 > total_ram * 0.9:
                need_ram_int4 = estimated_params_b * 0.75
                error_msg = (
                    f"🛑 模型过大，无法安全加载！\n"
                    f"  模型: {os.path.basename(model_path_str)} (~{estimated_params_b:.1f}B 参数)\n"
                    f"  所需内存: ~{need_ram_fp16:.1f} GB (FP16)\n"
                    f"  系统总内存: {total_ram:.1f} GB\n\n"
                    f"🔧 解决方案:\n"
                    f"  1. 下载 GGUF 量化版，4-bit 仅需 ~{need_ram_int4:.1f} GB\n"
                    f"  2. 在设置中启用 4-bit 量化加载\n"
                    f"  3. 选择更小的模型（7B 及以下）"
                )
                logger.error(error_msg)
                app_state.mark_startup_failed(error_msg)
                return

    logger.info(f"开始加载模型: {config.model_name} 设备: {device}")
    model, tokenizer = download_and_load_model(config)
    trainer = BaseInferenceEngine(model, config, device)
    app_state.update_model(model, tokenizer, trainer, config.model_name)


def _load_self_model(config: TrainingConfig):
    """加载 ModelSelf 原生模型 + 态极多模态引擎"""
    try:
        from taiji import load_model, NativeInferenceEngine
    except ImportError:
        logger.warning("态极模块未安装，无法加载 ModelSelf 模型。请加载 HuggingFace 或 GGUF 格式的模型。")
        app_state.mark_started()
        return

    from agent.tool_registry import registry

    model_path = config.model_name
    if not model_path or not os.path.isdir(model_path):
        logger.info("未检测到 ModelSelf 模型目录，将以【无模型状态】启动后台...")
        app_state.mark_started()
        return

    device = config.resolve_device()
    logger.info(f"加载 ModelSelf 原生模型: {model_path} 设备: {device}")

    model, tokenizer = load_model(model_path, device=device)

    # 注册 OmniCore 的工具到分词器
    tool_names = [t.name for t in registry.list_tools(enabled_only=True)]
    for name in tool_names:
        tokenizer.register_tool(name)
    model.set_num_tools(len(tokenizer._tool_name_to_id))
    logger.info(f"已注册 {len(tool_names)} 个工具到模型")

    trainer = NativeInferenceEngine(model, tokenizer, device)
    app_state.update_model(model, tokenizer, trainer, model_path)

    # 将同一个 trainer 实例也注册为态极引擎（避免创建两个独立实例共享 model）
    try:
        app_state.set_taiji_engine(trainer)
        logger.info("态极推理引擎已注册到 app_state")
    except Exception as e:
        logger.warning(f"态极推理引擎注册失败: {e}")

    # 初始化态极生命系统（通过桥接层）
    try:
        from core.taiji_bridge import get_taiji_bridge
        bridge = get_taiji_bridge()
        bridge.initialize(model=model, tokenizer=tokenizer, device=str(device))
        bridge.start_life()
        logger.info("态极生命系统已启动")
    except Exception as e:
        logger.warning(f"态极生命系统初始化失败（基础推理仍可用）: {e}")


def refresh_taiji_tools():
    """MCP 服务器启动后刷新态极引擎的工具列表（让分词器认识新工具名）"""
    try:
        from core.taiji_bridge import get_taiji_bridge
        bridge = get_taiji_bridge()
        if not bridge.is_initialized:
            return

        taiji = bridge.taiji
        model = taiji.body.model
        tokenizer = taiji.body.tokenizer
        if not model or not tokenizer:
            return

        from agent.tool_registry import registry

        # 获取当前所有已注册工具名
        current_tools = set(tokenizer._tool_name_to_id.keys())
        new_tools = [t.name for t in registry.list_tools(enabled_only=True) if t.name not in current_tools]

        if not new_tools:
            return

        # 注册新工具到分词器
        for name in new_tools:
            tokenizer.register_tool(name)
        model.set_num_tools(len(tokenizer._tool_name_to_id))

        logger.info(f"态极工具列表已刷新: 新增 {len(new_tools)} 个工具 ({', '.join(new_tools[:5])}{'...' if len(new_tools) > 5 else ''})")
    except Exception as e:
        logger.debug(f"刷新态极工具列表失败: {e}")


def _build_rag_index():
    """自动构建 RAG 索引"""
    doc_dir = get_external_path("docs")
    if os.path.exists(doc_dir):
        for fname in os.listdir(doc_dir):
            fpath = os.path.join(doc_dir, fname)
            if os.path.isfile(fpath):
                try:
                    app_state.rag_kb.add_file(fpath)
                except Exception as e:
                    logger.warning(f"加载文档失败 {fname}: {e}")
        if app_state.rag_kb.documents:
            app_state.rag_kb.rebuild_index()