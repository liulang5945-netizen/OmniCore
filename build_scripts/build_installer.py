"""
OmniCore 安装包自动构建脚本 (增强版)
依赖环境: Inno Setup 6 (需提前安装)

增强功能：
1. ✅ 集成 model_registry，安装向导提供 80+ 模型/量化组合
2. ✅ 硬件感知模型推荐（按内存分 5 档，每档 5-8 个模型选项）
3. ✅ 安装时可自由选择目标目录（DisableDirPage=no）
4. ✅ 安装后自动下载模型（首次启动检测 gguf_download_pending）
"""
import os
import sys
import shutil
import json
import subprocess
import io

# 修复 Windows cmd.exe 的 GBK 编码无法输出 Emoji 的问题
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# 模型数据定义
# 🔄 动态从 model_registry_data.py 生成，无需手动维护
# 按内存需求分为 5 个档位，自动标记每个档位的推荐模型
# ============================================================

TIER_DEFS = [
    {"min_ram": 24, "label": "旗舰配置 (24GB+ 内存)",    "prefix": "[旗舰] ",  "max_vram": 999},
    {"min_ram": 16, "label": "高配配置 (16-24GB 内存)",  "prefix": "[高配] ",  "max_vram": 24},
    {"min_ram": 8,  "label": "主流配置 (8-16GB 内存)",   "prefix": "[主流] ",  "max_vram": 16},
    {"min_ram": 4,  "label": "基础配置 (4-8GB 内存)",    "prefix": "[基础] ",  "max_vram": 8},
    {"min_ram": 0,  "label": "低配配置 (4GB 以下内存)",  "prefix": "[轻量] ",  "max_vram": 4},
]

# 模型去重：排除 Q3 批量扩充中大量只有单量化的冗余条目
_MODEL_SKIP_REPOS = {
    # 以下是在注册表末尾大量通过单一 Q4_K_M 变体重复的条目，不纳入安装向导
    "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    "TheBloke/CodeLlama-7B-Instruct-GGUF",
    "TheBloke/CodeLlama-13B-Instruct-GGUF",
    "mradermacher/DeepSeek-Math-7B-Instruct-GGUF",
    "TheBloke/zephyr-7B-beta-GGUF",
    "TheBloke/openchat-3.5-7B-GGUF",
    "NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF",
    "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
    "cognitivecomputations/dolphin-2.9-llama3-8b-gguf",
    "TheBloke/stablelm-zephyr-3b-GGUF",
    "lmstudio-community/Yi-1.5-9B-Chat-GGUF",
    "lmstudio-community/Yi-1.5-34B-Chat-GGUF",
    "MaziyarPanahi/llava-v1.6-mistral-7b-GGUF",
    "MaziyarPanahi/llava-v1.6-34b-GGUF",
    "ibm-granite/granite-3.1-8b-instruct-GGUF",
    "CohereForAI/aya-expanse-8b-GGUF",
    "LGAI-EXAONE/EXAONE-4.0-7.8B-Instruct-GGUF",
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    "lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF",
    "tiiuae/Falcon3-7B-Instruct-GGUF",
    "tiiuae/Falcon3-10B-Instruct-GGUF",
    "CohereForAI/c4ai-command-r7b-12-2024-GGUF",
    "upstage/SOLAR-10.7B-Instruct-v1.0-GGUF",
    "allenai/OLMo-2-7B-1124-Instruct-GGUF",
    "nvidia/Nemotron-Mini-4B-Instruct-GGUF",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base-GGUF",
}

# 每个档位的推荐模型 (repo -> display_name_override)
_TIER_RECOMMENDED = {
    24: "Qwen/Qwen2.5-32B-Instruct-GGUF",
    16: "Qwen/Qwen2.5-14B-Instruct-GGUF",
    8:  "Qwen/Qwen2.5-7B-Instruct-GGUF",
    4:  "Qwen/Qwen2.5-3B-Instruct-GGUF",
    0:  "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
}

# 模型排序优先级（repo -> 排序权重，越小越靠前）
_MODEL_SORT_ORDER = {
    "Qwen/Qwen2.5-32B-Instruct-GGUF": 1,
    "Qwen/Qwen2.5-14B-Instruct-GGUF": 2,
    "Qwen/Qwen2.5-7B-Instruct-GGUF": 3,
    "Qwen/Qwen2.5-3B-Instruct-GGUF": 4,
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF": 5,
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF": 6,
    "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF": 10,
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF": 11,
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF": 12,
}

def _build_model_tiers_from_registry():
    """从 model_registry_data.py 动态生成 MODEL_TIERS，确保安装程序与注册表永久同步。
    
    返回与旧版 MODEL_TIERS 完全兼容的数据结构：
    [{min_ram, label, models: [(display, repo, vram_gb, is_rec, quant, params_b), ...]}, ...]
    """
    try:
        from model.model_registry_data import MODEL_REGISTRY, estimate_file_size_mb, format_file_size
    except ImportError:
        # 独立运行时的回退
        return _fallback_model_tiers()

    tiers = {td["min_ram"]: {"min_ram": td["min_ram"], "label": td["label"], "models": []}
             for td in TIER_DEFS}
    
    # 收集所有 GGUF 模型并按 VRAM 分档
    all_entries = []
    for entry in MODEL_REGISTRY:
        if entry.hf_repo in _MODEL_SKIP_REPOS:
            continue
        if getattr(entry, 'model_type', 'gguf') != 'gguf':
            continue
        rec_var = entry.recommended_variant()
        if rec_var is None:
            continue
        vram = rec_var.vram_gb
        quant = rec_var.quant
        params_b = entry.params_b
        
        # 按 VRAM 分配到对应档位（从低到高匹配）
        assigned = False
        for td in sorted(TIER_DEFS, key=lambda x: x["max_vram"]):
            if vram < td["max_vram"]:
                tier = tiers[td["min_ram"]]
                is_rec = (entry.hf_repo == _TIER_RECOMMENDED.get(td["min_ram"], ""))
                # 生成显示名称
                clean_name = entry.name
                desc = entry.description[:20] if entry.description else ""
                tag_hints = []
                if "推理" in entry.tags: tag_hints.append("推理")
                if "代码" in entry.tags: tag_hints.append("代码")
                if "视觉" in entry.tags: tag_hints.append("视觉")
                if "多语言" in entry.tags: tag_hints.append("多语言")
                tag_str = f" ({'·'.join(tag_hints[:2])})" if tag_hints else ""
                
                file_sz = format_file_size(estimate_file_size_mb(params_b, quant))
                display = f"{clean_name} {quant}{tag_str}  [{file_sz}]"
                
                tier["models"].append((
                    display,
                    entry.hf_repo,
                    vram,
                    is_rec,
                    quant,
                    params_b,
                ))
                assigned = True
                break
        if not assigned:
            # 超大模型归入旗舰档
            tier = tiers[24]
            is_rec = (entry.hf_repo == _TIER_RECOMMENDED.get(24, ""))
            file_sz = format_file_size(estimate_file_size_mb(params_b, quant))
            display = f"{entry.name} {quant}  [{file_sz}]"
            tier["models"].append((display, entry.hf_repo, vram, is_rec, quant, params_b))
    
    # 每个档位内排序：推荐模型优先，再按 sort_order
    for td in TIER_DEFS:
        tier = tiers[td["min_ram"]]
        tier["models"].sort(key=lambda m: (
            not m[3],  # 推荐优先
            _MODEL_SORT_ORDER.get(m[1], 50),  # 按预设顺序
            m[4],  # 参数量从小到大
        ))

    return [tiers[td["min_ram"]] for td in TIER_DEFS]


def _fallback_model_tiers():
    """当无法导入 model_registry_data 时的最小回退模型列表"""
    return [
        {
            "min_ram": 24,
            "label": "旗舰配置 (24GB+ 内存)",
            "models": [
                ("🌟 [推荐] Qwen2.5-32B-Instruct Q4_K_M  [22.0 GB]", "Qwen/Qwen2.5-32B-Instruct-GGUF", 22.0, True, "Q4_K_M", 32.0),
                ("DeepSeek-R1-Distill-Qwen-32B Q4_K_M (推理旗舰)  [22.0 GB]", "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF", 22.0, False, "Q4_K_M", 32.0),
                ("DeepSeek-R1-Distill-Qwen-14B Q4_K_M (深度推理)  [9.5 GB]", "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", 9.5, False, "Q4_K_M", 14.0),
                ("Qwen2.5-72B-Instruct Q3_K_M (直逼GPT-4o)  [38.0 GB]", "Qwen/Qwen2.5-72B-Instruct-GGUF", 38.0, False, "Q3_K_M", 72.0),
                ("Gemma-2-27B Q4_K_M (谷歌旗舰)  [20.0 GB]", "unsloth/gemma-2-27b-it-GGUF", 20.0, False, "Q4_K_M", 27.0),
            ]
        },
        {
            "min_ram": 16,
            "label": "高配配置 (16-24GB 内存)",
            "models": [
                ("🌟 [推荐] Qwen2.5-14B-Instruct Q4_K_M  [9.5 GB]", "Qwen/Qwen2.5-14B-Instruct-GGUF", 9.5, True, "Q4_K_M", 14.0),
                ("DeepSeek-R1-Distill-Qwen-14B Q4_K_M (深度推理)  [9.5 GB]", "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", 9.5, False, "Q4_K_M", 14.0),
                ("Mistral-Nemo-12B Q4_K_M (128K上下文)  [8.2 GB]", "unsloth/Mistral-Nemo-Instruct-2407-GGUF", 8.2, False, "Q4_K_M", 12.0),
                ("Phi-4-14B Q4_K_M (合成数据推理)  [9.5 GB]", "unsloth/phi-4-GGUF", 9.5, False, "Q4_K_M", 14.0),
            ]
        },
        {
            "min_ram": 8,
            "label": "主流配置 (8-16GB 内存)",
            "models": [
                ("🌟 [推荐] Qwen2.5-7B-Instruct Q4_K_M  [4.8 GB]", "Qwen/Qwen2.5-7B-Instruct-GGUF", 4.8, True, "Q4_K_M", 7.0),
                ("DeepSeek-R1-Distill-Qwen-7B Q4_K_M (推理王者)  [4.8 GB]", "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", 4.8, False, "Q4_K_M", 7.0),
                ("InternLM3-8B Q4_K_M (中文出色)  [5.5 GB]", "unsloth/internlm3-8b-instruct-GGUF", 5.5, False, "Q4_K_M", 8.0),
                ("Falcon3-7B Q4_K_M (高性能)  [4.8 GB]", "tiiuae/Falcon3-7B-Instruct-GGUF", 4.8, False, "Q4_K_M", 7.0),
                ("Hermes-3-Llama-3.1-8B Q4_K_M (指令遵循)  [5.5 GB]", "NousResearch/Hermes-3-Llama-3.1-8B-GGUF", 5.5, False, "Q4_K_M", 8.0),
                ("GLM-4-9B-Chat Q4_K_M (128K上下文)  [6.0 GB]", "unsloth/glm-4-9b-chat-GGUF", 6.0, False, "Q4_K_M", 9.0),
            ]
        },
        {
            "min_ram": 4,
            "label": "基础配置 (4-8GB 内存)",
            "models": [
                ("🌟 [推荐] Qwen2.5-3B-Instruct Q4_K_M  [2.2 GB]", "Qwen/Qwen2.5-3B-Instruct-GGUF", 2.2, True, "Q4_K_M", 3.0),
                ("Gemma-3-4B Q4_K_M (新一代多语言)  [2.8 GB]", "unsloth/gemma-3-4b-it-GGUF", 2.8, False, "Q4_K_M", 4.0),
                ("Phi-4-mini-3.8B Q4_K_M (轻量推理)  [2.8 GB]", "unsloth/phi-4-mini-instruct-GGUF", 2.8, False, "Q4_K_M", 3.8),
                ("SmolLM2-1.7B Q4_K_M (边缘高性能)  [1.2 GB]", "bartowski/SmolLM2-1.7B-Instruct-GGUF", 1.2, False, "Q4_K_M", 1.7),
                ("Granite-3.1-2B Q4_K_M (企业轻量)  [1.5 GB]", "ibm-granite/granite-3.1-2b-instruct-GGUF", 1.5, False, "Q4_K_M", 2.0),
            ]
        },
        {
            "min_ram": 0,
            "label": "低配配置 (4GB 以下内存)",
            "models": [
                ("🌟 [推荐] Qwen2.5-1.5B-Instruct Q4_K_M  [1.1 GB]", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", 1.1, True, "Q4_K_M", 1.5),
                ("Qwen2.5-0.5B-Instruct Q4_K_M (超轻量)  [0.4 GB]", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", 0.4, False, "Q4_K_M", 0.5),
                ("Gemma-3-1B Q4_K_M (边缘可用)  [0.8 GB]", "unsloth/gemma-3-1b-it-GGUF", 0.8, False, "Q4_K_M", 1.0),
                ("SmolLM2-360M Q4_K_M (微型演示)  [0.3 GB]", "bartowski/SmolLM2-360M-Instruct-GGUF", 0.3, False, "Q4_K_M", 0.36),
                ("Qwen2.5-Coder-0.5B Q4_K_M (嵌入式代码)  [0.4 GB]", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF", 0.4, False, "Q4_K_M", 0.5),
            ]
        },
    ]


# 延迟初始化：首次访问时从注册表动态构建
_MODEL_TIERS_CACHE = None

def _get_model_tiers():
    global _MODEL_TIERS_CACHE
    if _MODEL_TIERS_CACHE is None:
        _MODEL_TIERS_CACHE = _build_model_tiers_from_registry()
    return _MODEL_TIERS_CACHE


def generate_model_pascal_code():
    """
    根据 MODEL_TIERS + HF 可微调模型库动态生成 Inno Setup Pascal 代码。
    两页模型选择：1) GGUF 推理模型（按内存分档）  2) HF 可微调模型
    """
    try:
        from model.model_registry import estimate_file_size_mb, format_file_size, get_all_models
    except ImportError:
        QUANT_BYTES = {"Q2_K": 0.35, "Q3_K_S": 0.45, "Q3_K_M": 0.50, "Q4_K_S": 0.58,
                       "Q4_K_M": 0.65, "Q5_K_M": 0.80, "Q6_K": 0.90, "Q8_0": 1.10, "F16": 2.0}
        def estimate_file_size_mb(pb, q):
            return round(pb * QUANT_BYTES.get(q, 0.65) * 953.674, 1)
        def format_file_size(sz):
            return f"{sz/1024:.1f} GB" if sz >= 1024 else f"{sz:.0f} MB"
        def get_all_models():
            return []

    # ── 1. GGUF 推理模型 ──
    flat_models = []
    tier_labels = {24: "[旗舰] ", 16: "[高配] ", 8: "[主流] ", 4: "[基础] ", 0: "[轻量] "}
    model_tiers = _get_model_tiers()
    for tier in model_tiers:
        prefix = tier_labels.get(tier["min_ram"], "")
        for m in tier["models"]:
            display_name = m[0]; repo = m[1]; is_rec = m[3]
            quant = m[4] if len(m) > 4 else "Q4_K_M"
            params_b = m[5] if len(m) > 5 else 0
            file_sz = ""
            if params_b > 0:
                file_sz = format_file_size(estimate_file_size_mb(params_b, quant))
            clean_name = display_name.replace("🌟 [推荐] ", "").replace("[推荐] ", "")
            size_suffix = f"  [{file_sz}]" if file_sz else ""
            flat_models.append((f"{prefix}{'🌟 [推荐] ' if is_rec else ''}{clean_name}{size_suffix}", repo, tier["min_ram"], is_rec, quant))

    # ── 2. HF 可微调模型 ──
    hf_flat_models = []
    try:
        all_registry = get_all_models()
        hf_rec_seen = False
        for entry in all_registry:
            if getattr(entry, 'model_type', 'gguf') != 'huggingface':
                continue
            clean_hf = entry.name.replace(" (HF)", "")
            repo = entry.hf_train_repo
            variant = entry.recommended_variant()
            vram = variant.vram_gb if variant else entry.params_b * 2.0
            is_rec = not hf_rec_seen and entry.params_b <= 8.0
            if is_rec:
                hf_rec_seen = True
            hf_prefix = "🟢 [可微调] "
            rec_mark = "🌟 [推荐] " if is_rec else ""
            hf_flat_models.append((hf_prefix + rec_mark + clean_hf, repo, vram, is_rec))
    except Exception:
        HF_MODEL_FALLBACKS = [
            ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 1.5, True),
            ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", 3.5, False),
            ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 7.0, False),
            ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 16.0, False),
            ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 30.0, False),
            ("DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 16.0, False),
            ("gemma-2-2b-it", "google/gemma-2-2b-it", 4.5, False),
        ]
        hf_rec_seen = False
        for name, repo, vram, is_rec in HF_MODEL_FALLBACKS:
            rec_mark = "🌟 [推荐] " if (is_rec and not hf_rec_seen) else ""
            if is_rec and not hf_rec_seen:
                hf_rec_seen = True
            hf_flat_models.append(("🟢 [可微调] " + rec_mark + name, repo, vram, is_rec))
    hf_flat_models.sort(key=lambda x: (not x[3], x[2]))

    # ── 确定默认选中索引 ──
    # 根据用户内存，从高到低匹配 tier，选中该 tier 中推荐模型在平铺列表中的索引
    # 先确定每个 tier 的推荐模型在 flat_models 中的索引范围
    tier_recommended_idx = {}  # tier_min_ram -> idx in flat_models (0-based, 但 Inno Setup 从1开始，且第0项是"跳过")
    for tier in model_tiers:
        for i, m in enumerate(tier["models"]):
            if m[3]:  # is_recommended
                # 找到这个模型在 flat_models 中的索引
                for fi, fm in enumerate(flat_models):
                    if fm[1] == m[1]:  # repo 匹配
                        tier_recommended_idx[tier["min_ram"]] = fi + 1  # +1 因为第0项是"跳过"
                        break
                break

    lines = []
    lines.append("// ============================================================")
    lines.append("// 以下代码由 build_installer.py 自动生成")
    lines.append(f"// GGUF推理: {len(flat_models)}个 | HF可微调: {len(hf_flat_models)}个")
    lines.append("// ============================================================")
    lines.append("")

    # ---- 全局变量声明 ----
    lines.append("""
var
  ModelPage: TInputOptionWizardPage;
  HfModelPage: TInputOptionWizardPage;
  ModelDirPage: TInputDirWizardPage;
  DirWarnShown: Boolean;

// ========== 【权限修复】检测目标安装目录是否可写 ==========
// 此函数递归检查最近存在的父目录是否可写，解决用户输入不存在路径时误判的问题
function IsDirWritable(const Dir: string): Boolean;
var
  TestDir: string;
  CheckDir: string;
begin
  // 从用户输入的路径向上查找最近存在的父目录
  CheckDir := Dir;
  while (CheckDir <> '') and not DirExists(CheckDir) do
  begin
    CheckDir := ExtractFileDir(CheckDir);
    if CheckDir = '' then
      CheckDir := ExpandConstant('{sd}');
  end;
  // 在该存在的目录中创建测试子目录
  TestDir := AddBackslash(CheckDir) + '.OmniCoreWriteTest';
  if CreateDir(TestDir) then
  begin
    RemoveDir(TestDir);
    Result := True;
  end
  else
    Result := False;
end;

// 初始化变量
function InitializeSetup: Boolean;
begin
  DirWarnShown := False;
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  InstallDir: string;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    InstallDir := WizardDirValue;
    if Pos(ExpandConstant('{localappdata}'), InstallDir) > 0 then
      Exit;
    if not IsDirWritable(InstallDir) then
    begin
      if not DirWarnShown then
      begin
        DirWarnShown := True;
        if SuppressibleMsgBox(
          '目标目录不可写入'#13#10 +
          '您选择的 "' + InstallDir + '" 目录当前用户没有写入权限。'#13#10#13#10 +
          '建议使用默认路径: ' + ExpandConstant('{localappdata}') + '\\OmniCore'#13#10#13#10 +
          '是否仍要继续？（可能导致安装失败）',
          mbError, MB_YESNO, 0) = IDNO then
          Result := False;
      end
      else
      begin
        MsgBox('目录无法写入，请更换安装目录。', mbError, MB_OK);
        Result := False;
      end;
    end;
  end;
end;

procedure InitializeWizard;
var
  TotalRAM: Integer;
begin
  TotalRAM := GetTotalMemoryGB();

  ModelDirPage := CreateInputDirPage(wpSelectDir,
    '选择模型存储路径', '请选择下载和部署的本地 AI 模型文件的保存位置。',
    '本地 AI 模型文件通常较大（数GB到数十GB），建议选择一个剩余空间充足的非系统盘（如 D、E 盘）。'#13#10#13#10'软件启动后会自动在该目录下完成模型的下载与部署。',
    False, '');
  ModelDirPage.Add('');
  ModelDirPage.Values[0] := ExpandConstant('{userdocs}\\OmniCoreModels');

  ModelPage := CreateInputOptionPage(ModelDirPage.ID,
    '选择本地推理 AI 模型 (GGUF)', '请选择您希望在聊天推理中使用的 GGUF 量化模型。',
    '系统检测到您的物理内存约为 ' + IntToStr(TotalRAM) + ' GB。GGUF 4-bit 量化在保证智能的同时极大降低内存占用。'#13#10'不需要推理模型可选"跳过"。',
    True, False);

  ModelPage.Add('跳过 (不安装 GGUF 推理模型)');

  // ── 平铺展示所有档位的模型（带 [档位] 前缀） ──
""")

    for fm in flat_models:
        display_name = fm[0]
        lines.append(f"  ModelPage.Add('{display_name}');")

    lines.append("")
    lines.append("  // 根据内存状况默认选中对应档位的推荐模型")
    # 按 min_ram 降序生成 if/else 判断
    sorted_tiers = sorted(tier_recommended_idx.items(), key=lambda x: x[0], reverse=True)
    for i, (min_ram, sel_idx) in enumerate(sorted_tiers):
        condition = f"TotalRAM >= {min_ram}" if min_ram > 0 else "True"
        lines.append(f"  if {condition} then")
        lines.append(f"    ModelPage.SelectedValueIndex := {sel_idx}")
        if i < len(sorted_tiers) - 1:
            lines.append("  else")
    lines.append("  ;")
    lines.append("end;")
    lines.append("")

    # ---- 生成 CurStepChanged 过程（平铺 case 映射，含 quant） ----
    lines.append("""
procedure CurStepChanged(CurStep: TSetupStep);
var
  SettingsPath: string;
  SettingsJson: string;
  SelectedModel: string;
  SelectedQuant: string;
  ModelDirPath: string;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    ModelDirPath := ModelDirPage.Values[0];
    ForceDirectories(ModelDirPath);
    SettingsPath := ExpandConstant('{app}\\app_settings.json');

    if ModelPage.SelectedValueIndex > 0 then
    begin
      // 平铺映射：所有模型选项对应唯一的 HuggingFace 仓库名和量化级别
      // 索引与 InitializeWizard 中添加顺序完全一致
      SelectedQuant := 'Q4_K_M';  // 默认值
      case ModelPage.SelectedValueIndex of
""")

    for i, fm in enumerate(flat_models):
        repo = fm[1]
        quant = fm[4] if len(fm) > 4 else "Q4_K_M"
        lines.append(f"        {i + 1}: begin SelectedModel := '{repo}'; SelectedQuant := '{quant}'; end;")

    lines.append("""
      end;
       
      SettingsJson := '{' + #13#10 +
                      '  "model_name": "' + SelectedModel + '",' + #13#10 +
                      '  "model_type": "gguf",' + #13#10 +
                      '  "gguf_download_pending": true,' + #13#10 +
                      '  "gguf_model_key": "' + SelectedModel + '",' + #13#10 +
                      '  "gguf_quant": "' + SelectedQuant + '",' + #13#10 +
                      '  "gguf_dir": "' + StringReplaceAll(ModelDirPath, '\\', '/') + '",' + #13#10 +
                      '  "n_gpu_layers": -1,' + #13#10 +
                      '  "n_ctx": 4096' + #13#10 +
                      '}';
                       
      SaveStringToFile(SettingsPath, SettingsJson, False);
    end
    else
    begin
      SettingsJson := '{' + #13#10 +
                      '  "gguf_dir": "' + StringReplaceAll(ModelDirPath, '\\', '/') + '"' + #13#10 +
                      '}';
      SaveStringToFile(SettingsPath, SettingsJson, False);
    end;

    // ========== 【权限修复】为用户组添加目录写权限 ==========
    Exec('icacls', ExpandConstant('"{app}" /grant Users:(OI)(CI)F /T /Q'),
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;""")

    return "\n".join(lines)


def _find_tesseract_installer(base_dir: str) -> str | None:
    """查找 Tesseract 静默安装包，用于捆绑到安装器中"""
    candidates = [
        os.path.join(base_dir, "external_libs", "tesseract-ocr-w64-setup.exe"),
        os.path.join(base_dir, "external_libs", "tesseract-ocr-w32-setup.exe"),
        os.path.join(base_dir, "dist", "tesseract-ocr-w64-setup.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def print_tesseract_download_guide():
    """打印 Tesseract 安装包下载指引"""
    print("")
    print("=" * 60)
    print("  ⚠️ 未找到 Tesseract OCR 静默安装包")
    print("  为了在用户设备上支持 OCR（图片/扫描PDF 文字识别），")
    print("  需要将 Tesseract 安装包放入 external_libs/ 目录。")
    print("")
    print("  📥 下载地址（UB-Mannheim 版本，含中文语言包）：")
    print("     https://github.com/UB-Mannheim/tesseract/releases")
    print("     选择: tesseract-ocr-w64-setup-{version}.exe (64-bit)")
    print("")
    print("  💡 下载后重命名为 tesseract-ocr-w64-setup.exe")
    print("     放入: external_libs\\tesseract-ocr-w64-setup.exe")
    print("     然后重新运行 python build_installer.py")
    print("")
    print("  ⚡ 跳过将继续打包，但用户设备的 OCR 功能将不可用。")
    print("=" * 60)
    print("")


def build_installer():
    base_dir = get_base_dir()
    dist_dir = os.path.join(base_dir, "dist", "OmniCore")
    output_dir = os.path.join(base_dir, "dist")

    # ── 版本号：优先根 version.json，其次 dist/OmniCore/version.json ──
    version = "1.0.0"
    version_candidates = [
        os.path.join(base_dir, "version.json"),
        os.path.join(dist_dir, "version.json"),
    ]
    for vf in version_candidates:
        if os.path.exists(vf):
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    version = json.load(f).get("version", "1.0.0")
                if version and version != "0.0.0":
                    break
            except Exception:
                pass

    # ── 确保 model 包可导入（独立运行时将项目根加入 sys.path）──
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    if not os.path.exists(dist_dir):
        print("[Build] 警告: 找不到 dist/OmniCore 目录！")
        print("[Build] 将跳过 EXE 检查，仅生成 .iss 配置文件。")
        # 不退出，允许仅生成 .iss 文件
    
    if os.path.exists(dist_dir) and not os.path.exists(os.path.join(dist_dir, "OmniCore.exe")):
        print("[Build] 警告: 在 dist/OmniCore 下没有找到 OmniCore.exe！")
        print("[Build] 说明上一步的客户端打包没有成功。")
        print("[Build] 继续生成 .iss（后续需要先运行 build_client.py 完成打包）。")

    # 自动寻找本地的 Inno Setup 编译器
    inno_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"D:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"D:\Program Files\Inno Setup 6\ISCC.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe")
    ]

    iscc_exe = None
    for p in inno_paths:
        if os.path.exists(p):
            iscc_exe = p
            break

    icon_path = os.path.join(base_dir, 'icon.ico')
    icon_line = f"SetupIconFile={icon_path}" if os.path.exists(icon_path) else "; 未找到 icon.ico，使用默认系统图标"

    # 查找 Tesseract 静默安装包
    tesseract_installer = _find_tesseract_installer(base_dir)
    tesseract_files_line = ""
    tesseract_run_line = ""
    tesseract_code = ""
    if tesseract_installer:
        print(f"[Build] ✅ 找到 Tesseract 安装包: {tesseract_installer}")
        tesseract_files_line = f'Source: "{tesseract_installer}"; DestDir: "{{tmp}}"; Flags: deleteafterinstall; Check: not IsTesseractInstalled'
        tesseract_run_line = 'Filename: "{tmp}\\tesseract-ocr-w64-setup.exe"; Parameters: "/S"; Check: not IsTesseractInstalled; StatusMsg: "正在安装 OCR 文字识别引擎（Tesseract）..."; Flags: waituntilterminated'
        tesseract_code = """
// ── Tesseract OCR 检测 ──
function IsTesseractInstalled: Boolean;
var
  ProgPath: string;
  ProgPath86: string;
begin
  Result := False;
  // 检查注册表（UB-Mannheim 安装器写入的键值）
  if RegQueryStringValue(HKLM, 'SOFTWARE\\Tesseract-OCR', 'InstallDir', ProgPath) then
  begin
    if FileExists(AddBackslash(ProgPath) + 'tesseract.exe') then
    begin
      Result := True;
      Exit;
    end;
  end;
  if RegQueryStringValue(HKCU, 'SOFTWARE\\Tesseract-OCR', 'InstallDir', ProgPath) then
  begin
    if FileExists(AddBackslash(ProgPath) + 'tesseract.exe') then
    begin
      Result := True;
      Exit;
    end;
  end;
  // 检查常见默认路径
  ProgPath := ExpandConstant('{pf}\\Tesseract-OCR\\tesseract.exe');
  if FileExists(ProgPath) then
  begin
    Result := True;
    Exit;
  end;
  ProgPath86 := ExpandConstant('{pf32}\\Tesseract-OCR\\tesseract.exe');
  if FileExists(ProgPath86) then
  begin
    Result := True;
    Exit;
  end;
  // 检查 LocalAppData
  ProgPath := ExpandConstant('{localappdata}\\Programs\\Tesseract-OCR\\tesseract.exe');
  if FileExists(ProgPath) then
  begin
    Result := True;
    Exit;
  end;
end;
"""
    else:
        print_tesseract_download_guide()

    if not iscc_exe:
        print("⚠️ 未找到 Inno Setup 6 编译器！无法生成 .exe 安装程序。")
        print("🔄 正在自动降级为生成【免安装 ZIP 绿色压缩包】...")
        zip_path = os.path.join(output_dir, f"OmniCore_v{version}_Portable")
        shutil.make_archive(zip_path, 'zip', os.path.join(output_dir, "OmniCore"))
        print(f"\n✅ 免安装包打包完毕！请直接发送此文件给用户: {zip_path}.zip")
        return

    # 动态生成模型选择页面的 Pascal 代码
    model_pascal_code = generate_model_pascal_code()

    # 动态生成 .iss 配置文件
    iss_path = os.path.join(base_dir, "OmniCore_Installer.iss")
    iss_content = f"""[Setup]
AppId={{{{8A8B9C9D-1234-5678-90AB-OMNICORE2026}}}}
AppName=OmniCore 智核工作站
AppVersion={version}
AppPublisher=OmniCore 开源社区

; 【架构修复】强制声明 64-bit 模式，解决错误码 216 版本不兼容问题
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
; 【兼容性修复】最低 Windows 10，确保 API 兼容性
MinVersion=10.0

; 用户可自由选择安装目录
DefaultDirName={{localappdata}}\\OmniCore
DefaultGroupName=OmniCore
DisableDirPage=no
DisableProgramGroupPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
AppMutex=OmniCore.exe

OutputDir={output_dir}
OutputBaseFilename=OmniCore_v{version}_Setup
{icon_line}
Compression=lzma2/ultra64
SolidCompression=yes
UninstallDisplayIcon={{app}}\\OmniCore.exe


[Dirs]
; 预先建立好各个模块所需的工作台及缓存目录
Name: "{{app}}\\user_data"
Name: "{{app}}\\model_cache"
Name: "{{app}}\\gguf_models"
Name: "{{app}}\\agent_workspace"
Name: "{{app}}\\data"
Name: "{{app}}\\docs"
Name: "{{app}}\\plugins"
Name: "{{app}}\\update_code"
Name: "{{app}}\\update_code\\agent"
Name: "{{app}}\\update_code\\api"
Name: "{{app}}\\update_code\\build_scripts"
Name: "{{app}}\\update_code\\core"
Name: "{{app}}\\update_code\\model"
Name: "{{app}}\\update_code\\tools"
Name: "{{app}}\\update_frontend"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"

[Files]
Source: "{dist_dir}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
{tesseract_files_line}

[Icons]
Name: "{{autoprograms}}\\OmniCore"; Filename: "{{app}}\\OmniCore.exe"
Name: "{{autodesktop}}\\OmniCore"; Filename: "{{app}}\\OmniCore.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\OmniCore.exe"; Description: "启动 OmniCore 智核工作站"; Flags: nowait postinstall skipifsilent
{tesseract_run_line}

[Code]
{tesseract_code}
type
  TMemoryStatusEx = record
    dwLength: DWORD;
    dwMemoryLoad: DWORD;
    ullTotalPhysLow: DWORD;
    ullTotalPhysHigh: DWORD;
    ullAvailPhysLow: DWORD;
    ullAvailPhysHigh: DWORD;
    ullTotalPageFileLow: DWORD;
    ullTotalPageFileHigh: DWORD;
    ullAvailPageFileLow: DWORD;
    ullAvailPageFileHigh: DWORD;
    ullTotalVirtualLow: DWORD;
    ullTotalVirtualHigh: DWORD;
    ullAvailVirtualLow: DWORD;
    ullAvailVirtualHigh: DWORD;
    ullAvailExtendedVirtualLow: DWORD;
    ullAvailExtendedVirtualHigh: DWORD;
  end;

function GlobalMemoryStatusEx(var lpBuffer: TMemoryStatusEx): BOOL;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function GetTotalMemoryGB: Integer;
var
  MemStatus: TMemoryStatusEx;
  LowMB: Integer;
begin
  MemStatus.dwLength := 64;
  if GlobalMemoryStatusEx(MemStatus) then
  begin
    LowMB := MemStatus.ullTotalPhysLow div 1048576;
    if LowMB < 0 then
      LowMB := LowMB + 4096;
    Result := ((MemStatus.ullTotalPhysHigh * 4096) + LowMB) div 1024;
  end
  else
    Result := 8;
end;

function StringReplaceAll(const S, OldPattern, NewPattern: string): string;
var
  TempStr: string;
begin
  TempStr := S;
  StringChange(TempStr, OldPattern, NewPattern);
  Result := TempStr;
end;

{model_pascal_code}
"""
    with open(iss_path, "w", encoding="utf-8-sig") as f:
        f.write(iss_content)

    print(f"\n🚀 开始调用 Inno Setup 生成安装包 (版本: v{version})...")
    print(f"📋 已集成 {sum(len(t['models']) for t in _get_model_tiers())} 个模型选项供用户选择")
    try:
        subprocess.check_call([iscc_exe, iss_path])
        print(f"\n✅ 安装包生成成功！位置: {os.path.join(output_dir, f'OmniCore_v{version}_Setup.exe')}")
    except Exception as e:
        print(f"\n❌ Inno Setup 编译失败: {e}")
        print("👉 提示: 常见原因是在 dist/OmniCore 目录下存在被占用的文件，或者目标路径权限不足。")


def get_base_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == "build_scripts":
        return os.path.dirname(current_dir)
    return current_dir


if __name__ == "__main__":
    build_installer()
