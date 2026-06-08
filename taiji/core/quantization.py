"""
态极模型量化模块
================

支持三种量化方式：
1. 动态量化 (Dynamic Quantization) — 最简单，推理时量化权重
2. 静态量化 (Static Quantization) — 需要校准数据，精度更高
3. INT4/NF4 量化 — 用于内存极度受限的场景

对于 125M 模型，动态量化通常就够了。
"""
import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple

logger = logging.getLogger("Taiji.Quantization")


class TaijiQuantizer:
    """态极模型量化器"""

    # 支持的量化精度
    PRECISIONS = {
        "dynamic_int8": "动态 INT8 量化（推荐，速度快，精度损失小）",
        "static_int8": "静态 INT8 量化（需要校准数据，精度更高）",
        "int4_nf4": "NF4 4-bit 量化（极致压缩，精度损失较大）",
        "none": "不量化",
    }

    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype=torch.qint8) -> nn.Module:
        """
        动态量化 — 最简单的量化方式。

        权重在推理时动态量化为 INT8，激活值保持 FP32。
        适合 CPU 推理，GPU 上效果有限。

        Args:
            model: 态极模型
            dtype: 量化精度 (qint8, quint8)

        Returns:
            量化后的模型
        """
        # 只量化 Linear 层（Transformer 的主要计算量）
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # 量化目标层类型
            dtype=dtype,
        )
        logger.info(f"动态量化完成 (dtype={dtype})")
        return quantized

    @staticmethod
    def quantize_int4_nf4(model: nn.Module) -> nn.Module:
        """
        INT4/NF4 量化 — 极致压缩。

        使用 bitsandbytes 库进行 4-bit 量化。
        需要安装 bitsandbytes: pip install bitsandbytes

        Args:
            model: 态极模型

        Returns:
            量化后的模型
        """
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear4bit
        except ImportError:
            logger.warning("bitsandbytes 未安装，无法进行 INT4 量化。请运行: pip install bitsandbytes")
            return model

        # 替换 Linear 层为 4-bit 版本
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # 创建 4-bit 线性层
                new_module = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4",
                )
                # 复制权重
                with torch.no_grad():
                    new_module.weight = bnb.nn.Params4bit(
                        module.weight.data,
                        requires_grad=False,
                        quant_type="nf4",
                    )
                    if module.bias is not None:
                        new_module.bias = module.bias
                setattr(model, name, new_module)
            else:
                # 递归处理子模块
                TaijiQuantizer.quantize_int4_nf4(module)

        logger.info("INT4/NF4 量化完成")
        return model

    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: list,
        device: str = "cpu",
    ) -> nn.Module:
        """
        静态量化 — 使用校准数据确定量化参数。

        比动态量化精度更高，但需要校准数据。

        Args:
            model: 态极模型
            calibration_data: 校准数据（输入 tensor 列表）
            device: 设备

        Returns:
            量化后的模型
        """
        model.eval()

        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm" if device == "cpu" else "qnnpack")

        # 准备量化
        prepared = torch.quantization.prepare(model)

        # 校准
        logger.info(f"开始校准（{len(calibration_data)} 个样本）...")
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, torch.Tensor):
                    prepared(data.to(device))
                elif isinstance(data, dict):
                    prepared(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()})

        # 转换为量化模型
        quantized = torch.quantization.convert(prepared)
        logger.info("静态量化完成")
        return quantized

    @staticmethod
    def get_model_size(model: nn.Module) -> dict:
        """获取模型大小信息"""
        param_size = 0
        buffer_size = 0
        param_count = 0

        for param in model.parameters():
            param_count += param.numel()
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            "param_count": param_count,
            "param_size_mb": param_size / 1024 / 1024,
            "buffer_size_mb": buffer_size / 1024 / 1024,
            "total_size_mb": total_size / 1024 / 1024,
            "param_dtype": str(next(model.parameters()).dtype) if param_count > 0 else "N/A",
        }

    @staticmethod
    def benchmark(model: nn.Module, input_ids: torch.Tensor, num_runs: int = 10) -> dict:
        """基准测试推理速度"""
        import time

        model.eval()
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        # 预热
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)

        # 计时
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                model(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        return {
            "avg_ms": avg_time * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "num_runs": num_runs,
        }


def quantize_model(
    model: nn.Module,
    precision: str = "dynamic_int8",
    calibration_data: list = None,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    量化态极模型的便捷函数。

    Args:
        model: 态极模型
        precision: 量化精度 ("dynamic_int8", "static_int8", "int4_nf4", "none")
        calibration_data: 校准数据（静态量化需要）
        device: 设备

    Returns:
        (量化后的模型, 量化信息)
    """
    # 记录量化前的大小
    before = TaijiQuantizer.get_model_size(model)

    if precision == "none":
        return model, {"precision": "none", "before": before, "after": before}

    if precision == "dynamic_int8":
        quantized = TaijiQuantizer.quantize_dynamic(model)
    elif precision == "static_int8":
        if not calibration_data:
            logger.warning("静态量化需要校准数据，回退到动态量化")
            quantized = TaijiQuantizer.quantize_dynamic(model)
            precision = "dynamic_int8 (fallback)"
        else:
            quantized = TaijiQuantizer.quantize_static(model, calibration_data, device)
    elif precision == "int4_nf4":
        quantized = TaijiQuantizer.quantize_int4_nf4(model)
    else:
        logger.warning(f"未知量化精度: {precision}，不量化")
        return model, {"precision": "none", "before": before, "after": before}

    # 记录量化后的大小
    after = TaijiQuantizer.get_model_size(quantized)

    info = {
        "precision": precision,
        "before": before,
        "after": after,
        "compression_ratio": before["total_size_mb"] / max(after["total_size_mb"], 0.001),
    }

    logger.info(
        f"量化完成: {before['total_size_mb']:.1f}MB → {after['total_size_mb']:.1f}MB "
        f"(压缩 {info['compression_ratio']:.1f}x)"
    )

    return quantized, info
