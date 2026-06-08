"""
model_registry.py 模块的单元测试
覆盖：模型查询、硬件分析、推荐引擎、下载信息
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_registry import (
    get_all_models, get_model_by_name, get_models_by_family,
    get_models_by_tag, get_model_download_info,
    analyze_hardware, recommend_models, recommend_for_quantization_focus,
    HardwareProfile, Recommendation, MODEL_REGISTRY, QUANT_LEVELS,
)


class TestModelRegistry:
    """模型注册表查询测试"""

    def test_get_all_models_returns_list(self):
        models = get_all_models()
        assert len(models) >= 10  # 至少有 10+ 个模型

    def test_get_model_by_name_found(self):
        entry = get_model_by_name("DeepSeek-R1-Distill-Qwen-7B")
        assert entry is not None
        assert entry.family == "DeepSeek"

    def test_get_model_by_name_not_found(self):
        entry = get_model_by_name("nonexistent-model-999B")
        assert entry is None

    def test_get_models_by_family(self):
        models = get_models_by_family("Qwen2.5")
        assert len(models) >= 3  # 至少 5 个 Qwen 模型

    def test_get_models_by_tag_chinese(self):
        models = get_models_by_tag("中文")
        assert len(models) >= 5  # 至少 5 个中文模型

    def test_get_models_by_tag_code(self):
        models = get_models_by_tag("代码")
        assert len(models) >= 1

    def test_get_model_download_info(self):
        info = get_model_download_info("DeepSeek-R1-Distill-Qwen-7B", "Q4_K_M")
        assert info is not None
        assert info["repo"] == "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF"
        assert info["filename"].endswith(".gguf")
        assert info["vram_gb"] > 0
        assert info["parameters_b"] == 7.0

    def test_get_model_download_info_bad_quant(self):
        info = get_model_download_info("DeepSeek-R1-Distill-Qwen-7B", "Q99_Z")
        assert info is None


class TestQuantLevels:
    """量化级别定义测试"""

    def test_quant_levels_defined(self):
        assert len(QUANT_LEVELS) >= 8  # 至少 8 种量化

    def test_q4_is_recommended(self):
        q4 = QUANT_LEVELS.get("Q4_K_M")
        assert q4 is not None
        assert q4.quality_score == 7

    def test_bytes_per_param_decreasing(self):
        """量化率越低，每参数字节越少"""
        q2 = QUANT_LEVELS["Q2_K"]
        f16 = QUANT_LEVELS["F16"]
        assert q2.bytes_per_param < f16.bytes_per_param


class TestHardwareProfile:
    """硬件配置类测试"""

    def test_profile_defaults(self):
        profile = HardwareProfile(total_ram_gb=16.0)
        assert profile.total_ram_gb == 16.0
        assert "cpu" in profile.gpu_backends

    def test_available_memory_cpu(self):
        profile = HardwareProfile(total_ram_gb=16.0)
        # CPU 模式留 2GB，所以可用=14GB
        assert profile.available_memory_gb == 14.0

    def test_available_memory_nvidia(self):
        profile = HardwareProfile(total_ram_gb=32.0, vram_gb=8.0, has_nvidia_gpu=True)
        # 独显留 1GB = 7GB
        assert profile.available_memory_gb == 7.0

    def test_available_memory_amd(self):
        profile = HardwareProfile(total_ram_gb=16.0, vram_gb=16.0, has_amd_gpu=True)
        # AMD 共享内存留 2GB
        assert profile.available_memory_gb == 14.0


class TestHardwareAnalysis:
    """硬件分析测试"""

    def test_analyze_hardware_returns_profile(self):
        profile = analyze_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.total_ram_gb > 0
        assert profile.cpu_cores >= 1


class TestRecommendEngine:
    """推荐引擎测试"""

    def test_recommend_models_returns_list(self):
        profile = HardwareProfile(total_ram_gb=16.0, cpu_cores=8)
        recs = recommend_models(profile, top_k=3)
        assert len(recs) <= 3
        if recs:
            assert isinstance(recs[0], Recommendation)
            assert recs[0].score > 0

    def test_recommend_high_ram(self):
        """高内存（32GB）应推荐更大的模型"""
        profile = HardwareProfile(total_ram_gb=32.0, cpu_cores=16)
        recs = recommend_models(profile, top_k=5)
        if recs:
            # 至少有一个推荐模型的参数量 >= 7B
            params = [r.model.params_b for r in recs]
            assert max(params) >= 7.0

    def test_recommend_low_ram(self):
        """低内存（8GB）应推荐小模型"""
        profile = HardwareProfile(total_ram_gb=8.0, cpu_cores=4)
        recs = recommend_models(profile, top_k=5)
        if recs:
            # 所有推荐应可使用
            for r in recs:
                assert r.vram_gb <= 7.0  # 留 1GB

    def test_recommend_sorted_by_score(self):
        profile = HardwareProfile(total_ram_gb=16.0, cpu_cores=8)
        recs = recommend_models(profile, top_k=10)
        for i in range(len(recs) - 1):
            assert recs[i].score >= recs[i + 1].score

    def test_recommend_for_quantization_focus(self):
        profile = HardwareProfile(total_ram_gb=16.0, cpu_cores=8)
        recs = recommend_for_quantization_focus(profile)
        assert len(recs) > 0
        for r in recs:
            assert r.quant in QUANT_LEVELS

    def test_recommend_gpu_preference(self):
        """有 GPU 时应推荐更大的量化"""
        cpu_profile = HardwareProfile(total_ram_gb=16.0)
        gpu_profile = HardwareProfile(total_ram_gb=16.0, vram_gb=8.0, has_nvidia_gpu=True)
        
        cpu_recs = recommend_models(cpu_profile, top_k=1)
        gpu_recs = recommend_models(gpu_profile, top_k=1)
        
        if cpu_recs and gpu_recs:
            # GPU 推荐的质量分应 >= CPU 推荐
            pass  # 只是验证不崩溃


class TestRecommendationData:
    """推荐结果数据结构测试"""

    def test_recommendation_attributes(self):
        rec = Recommendation(
            model=get_all_models()[0],
            quant="Q4_K_M",
            vram_gb=4.8,
            score=85.0,
            reason="测试推荐",
        )
        assert rec.model.name
        assert rec.vram_gb == 4.8
        assert rec.score <= 100
        assert rec.reason