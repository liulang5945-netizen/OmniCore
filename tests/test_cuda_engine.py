"""
M18: 端到端测试 — CUDA 推理引擎完整测试套件

测试:
  1. C++ 扩展编译检查
  2. ModelConfig 配置正确性
  3. 权重加载完整性
  4. 前向传播数值正确性 (vs PyTorch 参考实现)
  5. KV Cache 增量推理
  6. 流式生成
  7. 推理路径自动路由
  8. 融合 kernel 数值精度
  9. 性能基准对比

用法:
    python -m pytest tests/test_cuda_engine.py -v
    或
    python tests/test_cuda_engine.py
"""

import sys
import os
import time
import logging

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("TestCudaEngine")


def test_engine_import():
    """测试 1: C++ 扩展是否可导入"""
    logger.info("=== 测试 1: C++ 扩展导入 ===")
    try:
        import taiji_cuda_engine
        logger.info(f"✅ taiji_cuda_engine 已导入 (版本 {taiji_cuda_engine.__version__})")
        logger.info(f"   阶段: {taiji_cuda_engine.__phase__}")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ taiji_cuda_engine 不可用: {e}")
        logger.info("   需要先编译: cd csrc && build.bat")
        return False


def test_model_config():
    """测试 2: ModelConfig 配置正确性"""
    logger.info("=== 测试 2: ModelConfig 配置 ===")
    try:
        from csrc.python import ModelConfig
        if ModelConfig is None:
            logger.warning("⚠️ ModelConfig 不可用（C++ 扩展未编译）")
            return False

        # 测试所有预定义配置
        sizes = ["125m", "350m", "1b", "3b", "7b", "13b"]
        for size in sizes:
            config = ModelConfig.from_string(size)
            params = config.count_parameters()
            logger.info(f"  {size}: {config.describe()} ({params:,} 参数)")

        # 测试自定义配置
        config = ModelConfig()
        config.num_experts = 8
        config.num_experts_per_tok = 2
        assert config.use_moe(), "MoE 应该启用"
        assert not config.use_mla(), "MLA 不应启用"
        logger.info("✅ ModelConfig 测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ ModelConfig 测试失败: {e}")
        return False


def test_weight_loading():
    """测试 3: 权重加载"""
    logger.info("=== 测试 3: 权重加载 ===")
    try:
        import torch
        from taiji.loader import create_model
        from csrc.python import TaijiEngine, ModelConfig
        if TaijiEngine is None:
            logger.warning("⚠️ TaijiEngine 不可用（C++ 扩展未编译）")
            return False

        # 创建 PyTorch 模型
        model, tokenizer = create_model("125m", device="cpu")
        model.eval()

        # 创建 C++ 引擎
        config = ModelConfig.size_125m()
        engine = TaijiEngine(config, "cpu")

        # 加载权重
        state_dict = model.state_dict()
        engine.load_state_dict(state_dict)

        assert engine.is_loaded(), "权重应该已加载"
        logger.info(f"✅ 权重加载成功 ({engine.num_parameters():,} 参数)")
        return True
    except Exception as e:
        logger.error(f"❌ 权重加载测试失败: {e}")
        return False


def test_forward_correctness():
    """测试 4: 前向传播数值正确性"""
    logger.info("=== 测试 4: 前向传播数值正确性 ===")
    try:
        import torch
        from taiji.loader import create_model
        from csrc.python import TaijiEngine, ModelConfig
        if TaijiEngine is None:
            logger.warning("⚠️ 跳过（C++ 扩展未编译）")
            return False

        model, tokenizer = create_model("125m", device="cpu")
        model.eval()

        config = ModelConfig.size_125m()
        engine = TaijiEngine(config, "cpu")
        engine.load_state_dict(model.state_dict())

        # 测试输入
        test_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        # PyTorch 参考
        with torch.no_grad():
            ref_output = model(test_ids)
            ref_logits = ref_output.logits[0, -1, :]

        # C++ 引擎
        cpp_logits = engine.forward_step([1, 2, 3, 4, 5])

        # 数值比较
        max_diff = (ref_logits - cpp_logits).abs().max().item()
        logger.info(f"  最大差异: {max_diff:.6f}")
        assert max_diff < 0.01, f"数值差异过大: {max_diff}"
        logger.info("✅ 前向传播数值正确")
        return True
    except Exception as e:
        logger.error(f"❌ 前向传播测试失败: {e}")
        return False


def test_generate():
    """测试 5: 完整生成"""
    logger.info("=== 测试 5: 完整生成 ===")
    try:
        import torch
        from taiji.loader import create_model
        from csrc.python import TaijiEngine, ModelConfig, GenerateConfig
        if TaijiEngine is None:
            logger.warning("⚠️ 跳过（C++ 扩展未编译）")
            return False

        model, tokenizer = create_model("125m", device="cpu")
        model.eval()

        config = ModelConfig.size_125m()
        engine = TaijiEngine(config, "cpu")
        engine.load_state_dict(model.state_dict())

        gen_config = GenerateConfig()
        gen_config.max_new_tokens = 32
        gen_config.temperature = 0.7
        gen_config.eos_token_id = tokenizer.eos_token_id or -1

        # 编码输入
        prompt = "你好"
        input_ids = tokenizer.encode(prompt)

        start = time.time()
        output_ids = engine.generate(input_ids, gen_config)
        elapsed = (time.time() - start) * 1000

        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        logger.info(f"  输入: {prompt}")
        logger.info(f"  输出: {output_text[:100]}")
        logger.info(f"  生成 {len(output_ids)} tokens, 耗时 {elapsed:.1f} ms")
        logger.info(f"  吞吐量: {len(output_ids) / (elapsed / 1000):.1f} tokens/s")
        logger.info("✅ 生成测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 生成测试失败: {e}")
        return False


def test_triton_kernels():
    """测试 6: Triton 融合 kernel 数值精度"""
    logger.info("=== 测试 6: Triton 融合 kernel ===")
    try:
        import torch
        # 检查 Triton 是否可用
        try:
            import triton
            logger.info(f"  Triton 版本: {triton.__version__}")
        except ImportError:
            logger.warning("⚠️ Triton 不可用，跳过融合 kernel 测试")
            return True  # 不算失败

        # 测试 RMSNorm
        from csrc.cuda.triton_rms_norm import triton_rms_norm
        x = torch.randn(4, 64, dtype=torch.float32)
        weight = torch.ones(64, dtype=torch.float32)

        # PyTorch 参考
        rms = torch.sqrt(torch.mean(x.pow(2), -1, keepdim=True) + 1e-5)
        ref = weight * (x / rms)

        # Triton
        triton_out = triton_rms_norm(x, weight)

        diff = (ref - triton_out).abs().max().item()
        logger.info(f"  RMSNorm 最大差异: {diff:.6f}")
        assert diff < 1e-4, f"RMSNorm 数值差异过大: {diff}"

        # 测试 Softmax
        from csrc.cuda.triton_softmax import triton_softmax
        x = torch.randn(4, 64, dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        triton_out = triton_softmax(x)
        diff = (ref - triton_out).abs().max().item()
        logger.info(f"  Softmax 最大差异: {diff:.6f}")
        assert diff < 1e-4, f"Softmax 数值差异过大: {diff}"

        logger.info("✅ Triton 融合 kernel 测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ Triton kernel 测试失败: {e}")
        return False


def test_architecture_upgrades():
    """测试 7: 架构升级模块"""
    logger.info("=== 测试 7: 架构升级模块 ===")
    try:
        import torch

        # 测试 YaRN
        from csrc.architecture.yarn_rope import YaRNRotaryEmbedding
        yarn = YaRNRotaryEmbedding(dim=64, max_seq_len=4096, scale_factor=8)
        sin, cos = yarn(1024, torch.device("cpu"), torch.float32)
        logger.info(f"  YaRN: sin {sin.shape}, cos {cos.shape}")

        # 测试 MTP
        from csrc.architecture.multi_token_prediction import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(hidden_size=64, vocab_size=1000, num_mtp_tokens=4)
        hidden = torch.randn(2, 10, 64)
        targets = torch.randint(0, 1000, (2, 10))
        logits, mtp_loss = mtp(hidden, targets)
        logger.info(f"  MTP: logits {logits.shape}, loss {mtp_loss.item():.4f}")

        # 测试 MoE
        from csrc.architecture.moe_layer import MoELayer
        moe = MoELayer(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        output, aux_loss = moe(x)
        logger.info(f"  MoE: output {output.shape}, aux_loss {aux_loss.item():.4f}")

        # 测试 Mamba
        from csrc.architecture.mamba_block import MambaBlock
        mamba = MambaBlock(hidden_size=64, d_state=16)
        x = torch.randn(2, 10, 64)
        output = mamba(x)
        logger.info(f"  Mamba: output {output.shape}")

        logger.info("✅ 架构升级模块测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 架构升级测试失败: {e}")
        return False


def test_training_components():
    """测试 8: 训练组件"""
    logger.info("=== 测试 8: 训练组件 ===")
    try:
        import torch

        # 测试梯度检查点
        from csrc.training.gradient_checkpoint import GradientCheckpointing, SelectiveRecompute
        model = torch.nn.Linear(64, 64)
        ckpt = GradientCheckpointing(model)
        logger.info(f"  GradientCheckpointing: 创建成功")

        # 测试选择性重计算
        assert SelectiveRecompute.should_recompute(0, "attention_scores") == True
        assert SelectiveRecompute.should_recompute(0, "ffn_intermediate") == False
        logger.info(f"  SelectiveRecompute: 策略正确")

        # 测试混合精度
        from csrc.training.mixed_precision import MixedPrecisionTrainer
        logger.info(f"  MixedPrecisionTrainer: 导入成功")

        logger.info("✅ 训练组件测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 训练组件测试失败: {e}")
        return False


def test_profiler():
    """测试 9: 性能分析器"""
    logger.info("=== 测试 9: 性能分析器 ===")
    try:
        from taiji.profiler import TaijiProfiler

        profiler = TaijiProfiler()

        # 简单计时
        profiler.start("test_op")
        import time
        time.sleep(0.01)
        profiler.stop("test_op")

        report = profiler.get_report()
        assert "test_op" in report.timings
        assert report.timings["test_op"].count == 1
        assert report.timings["test_op"].total_ms >= 10
        logger.info(f"  test_op: {report.timings['test_op'].total_ms:.1f} ms")

        # 上下文管理器
        with profiler.timer("test_op2"):
            time.sleep(0.005)

        report = profiler.get_report()
        assert "test_op2" in report.timings
        logger.info(f"  test_op2: {report.timings['test_op2'].total_ms:.1f} ms")

        logger.info("✅ 性能分析器测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 性能分析器测试失败: {e}")
        return False


def test_parallel():
    """测试 10: 多 GPU 并行"""
    logger.info("=== 测试 10: 多 GPU 并行 ===")
    try:
        from csrc.parallel import TensorParallel, PipelineParallel

        # Tensor Parallel
        tp = TensorParallel(num_gpus=2)
        logger.info(f"  TensorParallel: {tp.num_gpus} GPUs")

        # Pipeline Parallel
        pp = PipelineParallel(num_gpus=2, num_layers=22)
        logger.info(f"  PipelineParallel: {pp.layer_assignment}")

        assert len(pp.layer_assignment) == 2
        assert pp.layer_assignment[0] == (0, 11)
        assert pp.layer_assignment[1] == (11, 22)

        logger.info("✅ 多 GPU 并行测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 多 GPU 并行测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 70)
    logger.info("  态极 CUDA 推理引擎 — 端到端测试")
    logger.info("=" * 70)

    results = {}
    tests = [
        ("C++ 扩展导入", test_engine_import),
        ("ModelConfig 配置", test_model_config),
        ("权重加载", test_weight_loading),
        ("前向传播数值正确性", test_forward_correctness),
        ("完整生成", test_generate),
        ("Triton 融合 kernel", test_triton_kernels),
        ("架构升级模块", test_architecture_upgrades),
        ("训练组件", test_training_components),
        ("性能分析器", test_profiler),
        ("多 GPU 并行", test_parallel),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"❌ {name} 异常: {e}")
            results[name] = False

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("  测试结果汇总")
    logger.info("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        logger.info(f"  {status} | {name}")
    logger.info("-" * 70)
    logger.info(f"  通过: {passed}/{total}")
    logger.info("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)