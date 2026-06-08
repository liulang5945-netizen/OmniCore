/**
 * Taiji CUDA Engine — pybind11 绑定
 *
 * 将 C++ TaijiEngine 暴露给 Python。
 * 所有推理方法使用 py::call_guard<py::gil_scoped_release>() 释放 GIL。
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

#include "core/engine.h"
#include "core/model_config.h"
#include "core/weight_loader.h"

namespace py = pybind11;
using namespace taiji;

PYBIND11_MODULE(taiji_cuda_engine, m) {
    m.doc() = R"pbdoc(
        Taiji CUDA Inference Engine
        ==========================
        
        C++/CUDA 原生推理引擎，消除 Python GIL 开销。
        支持态极全部 5 个 Head（语言/工具/记忆/规划/感知）。
        
        Phase 1: 基础推理循环 (libtorch ATen)
        Phase 2: Triton 融合 kernels
        Phase 3: 融合交叉熵 + 采样
        Phase 4: CUDA Graphs
        Phase 5: PagedAttention
    )pbdoc";

    // ═══════════════════════════════════════════════════════════
    // ModelConfig
    // ═══════════════════════════════════════════════════════════

    py::class_<ModelConfig>(m, "ModelConfig", "模型架构配置")
        .def(py::init<>())
        // 基础参数
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        .def_readwrite("intermediate_size", &ModelConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &ModelConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &ModelConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &ModelConfig::num_key_value_heads)
        .def_readwrite("max_position_embeddings", &ModelConfig::max_position_embeddings)
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        // 多头参数
        .def_readwrite("num_tools", &ModelConfig::num_tools)
        .def_readwrite("max_tools", &ModelConfig::max_tools)
        .def_readwrite("num_plan_actions", &ModelConfig::num_plan_actions)
        .def_readwrite("num_env_tokens", &ModelConfig::num_env_tokens)
        .def_readwrite("num_memory_slots", &ModelConfig::num_memory_slots)
        .def_readwrite("memory_dim", &ModelConfig::memory_dim)
        // MoE / MTP / MLA
        .def_readwrite("num_experts", &ModelConfig::num_experts)
        .def_readwrite("num_experts_per_tok", &ModelConfig::num_experts_per_tok)
        .def_readwrite("num_mtp_heads", &ModelConfig::num_mtp_heads)
        .def_readwrite("latent_dim", &ModelConfig::latent_dim)
        .def_readwrite("rope_dim", &ModelConfig::rope_dim)
        // 计算属性
        .def_property_readonly("head_dim", &ModelConfig::head_dim)
        .def_property_readonly("num_queries_per_kv", &ModelConfig::num_queries_per_kv)
        .def("count_parameters", &ModelConfig::count_parameters)
        .def("describe", &ModelConfig::describe)
        .def("use_gqa", &ModelConfig::use_gqa)
        .def("use_moe", &ModelConfig::use_moe)
        .def("use_mla", &ModelConfig::use_mla)
        .def("use_mtp", &ModelConfig::use_mtp)
        // 预定义配置
        .def_static("size_125m", &ModelConfig::size_125m)
        .def_static("size_350m", &ModelConfig::size_350m)
        .def_static("size_1b", &ModelConfig::size_1b)
        .def_static("size_3b", &ModelConfig::size_3b)
        .def_static("size_7b", &ModelConfig::size_7b)
        .def_static("size_13b", &ModelConfig::size_13b)
        .def_static("from_string", &ModelConfig::from_string)
        .def("__repr__", &ModelConfig::describe);

    // ═══════════════════════════════════════════════════════════
    // GenerateConfig
    // ═══════════════════════════════════════════════════════════

    py::class_<GenerateConfig>(m, "GenerateConfig", "推理配置")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &GenerateConfig::max_new_tokens)
        .def_readwrite("temperature", &GenerateConfig::temperature)
        .def_readwrite("top_p", &GenerateConfig::top_p)
        .def_readwrite("repetition_penalty", &GenerateConfig::repetition_penalty)
        .def_readwrite("eos_token_id", &GenerateConfig::eos_token_id)
        .def_readwrite("stop_on_token", &GenerateConfig::stop_on_token)
        .def_readwrite("batch_size", &GenerateConfig::batch_size)
        .def_readwrite("tokens_per_yield", &GenerateConfig::tokens_per_yield);

    // ═══════════════════════════════════════════════════════════
    // TaijiEngine (核心)
    // ═══════════════════════════════════════════════════════════

    py::class_<TaijiEngine>(m, "TaijiEngine", "态极 C++ 推理引擎")
        .def(py::init<const ModelConfig&, const std::string&>(),
             py::arg("config"),
             py::arg("device") = "cpu",
             R"pbdoc(
                创建推理引擎实例。
                
                Args:
                    config: 模型配置 (ModelConfig)
                    device: 目标设备 ("cpu" 或 "cuda:0")
             )pbdoc")

        // 权重管理
        .def("load_state_dict",
             &TaijiEngine::load_state_dict,
             py::arg("state_dict"),
             "从 Python dict 加载模型权重（与 PyTorch model.state_dict() 兼容）")
        .def("is_loaded", &TaijiEngine::is_loaded, "权重是否已加载")
        .def("num_parameters", &TaijiEngine::num_parameters, "统计参数量")
        .def("set_num_tools", &TaijiEngine::set_num_tools, py::arg("n"), "设置活跃工具数量")
        .def("config", &TaijiEngine::config, py::return_value_policy::reference_internal, "获取模型配置")
        .def("device", &TaijiEngine::device, py::return_value_policy::reference_internal, "获取设备")

        // 推理 — 释放 GIL
        .def("generate",
             &TaijiEngine::generate,
             py::arg("input_ids"),
             py::arg("gen_config"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                自回归生成 — 整个循环在 C++ 侧运行，释放 Python GIL。
                
                Args:
                    input_ids: 输入 token ID 列表 (List[int])
                    gen_config: 生成配置 (GenerateConfig)
                
                Returns:
                    生成的 token ID 列表 (不含输入)
             )pbdoc")

        .def("generate_batched",
             &TaijiEngine::generate_batched,
             py::arg("input_ids"),
             py::arg("gen_config"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                流式生成 — 按批次返回 token IDs。
                
                Args:
                    input_ids: 输入 token ID 列表
                    gen_config: 生成配置 (tokens_per_yield 控制每批大小)
                
                Returns:
                    每批 token ID 列表的向量 (List[List[int]])
             )pbdoc")

        .def("forward_ids",
             &TaijiEngine::forward_ids,
             py::arg("input_ids"),
             py::arg("use_cache") = false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                完整前向 — 返回所有位置的 logits。
                
                Args:
                    input_ids: 输入 token ID 列表 (List[int])
                    use_cache: 是否使用 KV Cache
                
                Returns:
                    logits Tensor [batch, seq_len, vocab_size]
             )pbdoc")

        // KV Cache 管理
        .def("reset_cache", &TaijiEngine::reset_cache, "重置 KV Cache")
        .def("cache_seq_len", &TaijiEngine::cache_seq_len, "当前 KV Cache 序列长度");

    // ═══════════════════════════════════════════════════════════
    // 版本信息
    // ═══════════════════════════════════════════════════════════

    m.attr("__version__") = "0.1.0";
    m.attr("__phase__") = "Phase 1: 基础推理循环 (libtorch ATen)";
    m.attr("PHASE_PLAN") = R"pbdoc(
Phase 1: 基础推理循环 (libtorch ATen)          ← 当前
Phase 2: Triton 融合前向 kernels
Phase 3: 融合交叉熵 + 采样
Phase 4: CUDA Graphs + 全 5 个 Head
Phase 5: PagedAttention
    )pbdoc";
}