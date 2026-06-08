"""
模型注册表与硬件感知推荐引擎
数据定义独立在 model_registry_data.py 中，此文件提供查询与推荐逻辑

量化内存计算公式：
    VRAM ≈ 参数量 × 每参数字节 × 1.05 (KV cache 开销)
"""
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger("ModelRegistry")

# ── 从数据文件导入所有定义 ──
from model.model_registry_data import (
    QuantInfo, QUANT_LEVELS,
    ModelVariant, ModelEntry,
    estimate_file_size_mb, format_file_size,
    MODEL_REGISTRY, MODEL_REGISTRY_HF,
)

# ======================== 推荐引擎 ========================

@dataclass
class HardwareProfile:
    """硬件配置画像"""
    total_ram_gb: float
    vram_gb: float = 0
    gpu_backends: List[str] = field(default_factory=lambda: ["cpu"])
    cpu_cores: int = 4
    has_nvidia_gpu: bool = False
    has_amd_gpu: bool = False
    has_apple_silicon: bool = False
    is_rog_ally: bool = False

    @property
    def available_memory_gb(self) -> float:
        if self.vram_gb > 0 and self.has_nvidia_gpu:
            return self.vram_gb - 1
        if self.vram_gb > 0 and self.has_amd_gpu:
            return min(self.total_ram_gb, self.vram_gb) - 2
        return self.total_ram_gb - 2

    @property
    def tier(self) -> str:
        has_gpu = self.has_nvidia_gpu or self.has_amd_gpu
        if self.total_ram_gb <= 16 and not has_gpu:
            return "low"
        if self.total_ram_gb <= 32 or not has_gpu:
            return "medium"
        return "high"

    def can_finetune_hf(self, params_b: float) -> tuple:
        hf_mem = params_b * 2.2
        avail = self.available_memory_gb
        if hf_mem <= avail * 0.8:
            return True, f"✅ 内存充裕，可流畅微调 (~{hf_mem:.1f}GB / {avail:.1f}GB 可用)"
        elif hf_mem <= avail * 0.95:
            return True, f"⚠️ 微调可行但内存偏紧 (~{hf_mem:.1f}GB / {avail:.1f}GB)"
        else:
            qlora_mem = params_b * 0.75
            if qlora_mem <= avail * 0.8:
                return False, f"❌ 完整模型需 {hf_mem:.1f}GB > {avail:.1f}GB。可启用 QLoRA 降至 ~{qlora_mem:.1f}GB"
            return False, f"❌ 模型过大 ({hf_mem:.1f}GB)，系统仅 {avail:.1f}GB"


@dataclass
class Recommendation:
    """推荐结果"""
    model: ModelEntry
    quant: str
    vram_gb: float
    score: float
    reason: str
    detail: str = ""
    category: str = "gguf"
    can_finetune: bool = False
    finetune_note: str = ""


def analyze_hardware() -> HardwareProfile:
    """自动检测硬件配置"""
    from core.config import TrainingConfig
    ram_gb = TrainingConfig.get_total_ram_gb()
    profile = HardwareProfile(total_ram_gb=ram_gb)
    try:
        import os
        import psutil
        profile.cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        try:
            import os
            profile.cpu_cores = os.cpu_count() or 4
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            profile.has_nvidia_gpu = True
            try:
                profile.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                profile.vram_gb = 8
            profile.gpu_backends.append("cuda")
    except Exception:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            profile.has_amd_gpu = True
            profile.gpu_backends.append("directml")
    except Exception:
        pass
    try:
        import platform
        if platform.processor() and "AMD" in platform.processor():
            profile.gpu_backends.append("vulkan")
    except Exception:
        pass
    return profile


def recommend_models(hardware: HardwareProfile, top_k: int = 5) -> List[Recommendation]:
    """智能推荐引擎 — 区分 GGUF 推理与 HF 微调"""
    available_mem = hardware.available_memory_gb
    hw_tier = hardware.tier
    recommendations = []

    for entry in (MODEL_REGISTRY + MODEL_REGISTRY_HF):
        is_hf = getattr(entry, 'model_type', 'gguf') == 'huggingface'
        if is_hf and hw_tier == "low":
            continue
        if is_hf:
            hf_mem = entry.params_b * 2.2
            if hf_mem > available_mem * 0.95:
                continue

        best_variant = None
        best_mem_diff = float('inf')
        for variant in entry.variants:
            variant_mem = variant.vram_gb
            mem_diff = variant_mem - available_mem
            if mem_diff <= 0:
                if abs(variant_mem - available_mem * 0.65) < best_mem_diff:
                    best_mem_diff = abs(variant_mem - available_mem * 0.65)
                    best_variant = variant
            elif abs(mem_diff) < 3.0 and best_variant is None:
                if best_variant is None or variant_mem < best_variant.vram_gb:
                    best_variant = variant
        if best_variant is None:
            continue

        score = 0.0
        mem_ratio = best_variant.vram_gb / max(available_mem, 1)
        if is_hf:
            if mem_ratio <= 0.2: score += 28
            elif mem_ratio <= 0.4: score += 25
            elif mem_ratio <= 0.6: score += 20
            elif mem_ratio <= 0.8: score += 12
            elif mem_ratio <= 0.95: score += 5
        else:
            if mem_ratio <= 0.3: score += 30
            elif mem_ratio <= 0.5: score += 25
            elif mem_ratio <= 0.7: score += 18
            elif mem_ratio <= 0.9: score += 8

        quant_info = QUANT_LEVELS.get(best_variant.quant)
        if quant_info:
            score += 30 if is_hf else quant_info.quality_score * 4

        if not is_hf:
            params_ratio = entry.params_b / max(available_mem, 1)
            if params_ratio < 0.5 and entry.params_b >= 3:
                score += min(entry.params_b, 10)
            elif entry.params_b >= 7 and params_ratio < 0.8:
                score += 5

        if "推荐" in entry.tags: score += 12
        if "旗舰" in entry.tags and hardware.total_ram_gb >= 24: score += 8
        if "中文" in entry.tags: score += 6
        if "代码" in entry.tags: score += 4
        if best_variant.is_recommended: score += 8

        can_ft = False
        ft_note = ""
        if is_hf:
            can_ft, ft_note = hardware.can_finetune_hf(entry.params_b)
            score += 15 if can_ft else -10
        elif hw_tier != "low" and entry.params_b <= 7:
            ft_note = "💡 可下载 HF 版本进行 4-bit QLoRA 微调"

        ram_gb_str = f"{best_variant.vram_gb:.1f}GB"
        if is_hf:
            reason = f"🔬 可微调 · {ram_gb_str} · BF16 完整模型"
            detail = f"{entry.description}\n参数量: {entry.params_b}B | 内存: ~{ram_gb_str}\n微调评估: {ft_note}\n适用场景: 针对特定任务微调，训练后导出 GGUF 部署"
            category = "huggingface"
        else:
            reason = f"⚡ GGUF · {ram_gb_str} · {best_variant.quant}"
            detail = f"{entry.description}\n参数量: {entry.params_b}B | 量化: {best_variant.quant} | 内存: ~{ram_gb_str}\n适用: 即开即用的本地推理"
            if ft_note: detail += f"\n{ft_note}"
            category = "gguf"

        recommendations.append(Recommendation(
            model=entry, quant=best_variant.quant, vram_gb=best_variant.vram_gb,
            score=min(score, 100), reason=reason, detail=detail,
            category=category, can_finetune=can_ft, finetune_note=ft_note,
        ))

    recommendations.sort(key=lambda r: r.score, reverse=True)
    gguf_recs = [r for r in recommendations if r.category == "gguf"]
    hf_recs = [r for r in recommendations if r.category == "huggingface"]

    result = []
    gguf_idx = hf_idx = 0
    while len(result) < top_k:
        for _ in range(min(3, top_k - len(result))):
            if gguf_idx < len(gguf_recs):
                result.append(gguf_recs[gguf_idx]); gguf_idx += 1
        if hf_idx < len(hf_recs) and len(result) < top_k:
            result.append(hf_recs[hf_idx]); hf_idx += 1
        if gguf_idx >= len(gguf_recs) and hf_idx >= len(hf_recs): break
        if gguf_idx >= len(gguf_recs):
            result.extend(hf_recs[hf_idx:hf_idx + (top_k - len(result))]); break
        if hf_idx >= len(hf_recs):
            result.extend(gguf_recs[gguf_idx:gguf_idx + (top_k - len(result))]); break
    return result[:top_k]


def recommend_for_quantization_focus(hardware: HardwareProfile) -> List[Recommendation]:
    """以量化为导向推荐"""
    available_mem = hardware.available_memory_gb
    results = []
    for entry in MODEL_REGISTRY:
        best_variant = None
        best_score = -1
        for variant in entry.variants:
            mem_ok = variant.vram_gb <= available_mem + 1
            if not mem_ok: continue
            quant_info = QUANT_LEVELS.get(variant.quant)
            q_score = quant_info.quality_score if quant_info else 0
            mem_comfort = max(0, available_mem - variant.vram_gb - 1)
            total = q_score * 8 + mem_comfort * 0.5 + (10 if variant.is_recommended else 0)
            if total > best_score:
                best_score = total; best_variant = variant
        if best_variant:
            results.append(Recommendation(
                model=entry, quant=best_variant.quant, vram_gb=best_variant.vram_gb,
                score=min(best_score, 100),
                reason=f"参数量 {entry.params_b}B · {best_variant.quant} · 需 {best_variant.vram_gb:.1f}GB"
            ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ======================== 查询函数 ========================

def _all_entries() -> List[ModelEntry]:
    return MODEL_REGISTRY + MODEL_REGISTRY_HF


def get_all_models() -> List[ModelEntry]:
    """获取所有模型（含 GGUF 和 HuggingFace 原版）"""
    return _all_entries()


def get_model_by_name(name: str) -> Optional[ModelEntry]:
    """按名称查找模型（含 GGUF 和 HF）"""
    for entry in _all_entries():
        if entry.name == name:
            return entry
    return None


def get_models_by_family(family: str) -> List[ModelEntry]:
    """按系列查找模型（含 GGUF 和 HF）"""
    return [e for e in _all_entries() if e.family == family]


def get_models_by_tag(tag: str) -> List[ModelEntry]:
    """按标签查找模型（含 GGUF 和 HF）"""
    return [e for e in _all_entries() if tag in e.tags]


def get_model_download_info(model_name: str, quant: str) -> Optional[dict]:
    """获取模型下载信息（含 GGUF 和 HF）"""
    entry = get_model_by_name(model_name)
    if not entry:
        return None
    for variant in entry.variants:
        if variant.quant == quant:
            return {
                "repo": entry.hf_repo,
                "filename": variant.hf_filename,
                "vram_gb": variant.vram_gb,
                "parameters_b": entry.params_b,
                "family": entry.family,
                "description": entry.description,
            }
    return None
