cmake_minimum_required(VERSION 3.18)
project(taiji_cuda_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── 查找依赖 ──
find_package(Torch REQUIRED)

# CUDA 可选：有则启用 GPU 加速，无则纯 CPU
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CUDA toolkit not found — building CPU-only engine")
endif()

# pybind11: 优先用 find_package，fallback 到 torch 自带的
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # 尝试从 Python 环境获取
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    find_package(pybind11 QUIET)
endif()
if(NOT pybind11_FOUND)
    # 最终 fallback: 手动 include
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Using pybind11 include: ${PYBIND11_INCLUDE}")
endif()

# ── CUDA 架构配置 ──
# 自动检测 GPU 架构，或手动指定
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")  # Volta → Hopper
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Torch version: ${TORCH_VERSION}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")

# ── CUDA 源文件 ──
set(CUDA_SOURCES
    # 初始阶段暂无 .cu 文件，后续添加:
    # cuda/rms_norm.cu
    # cuda/swiglu.cu
    # cuda/paged_attention.cu
    # cuda/top_p_sampling.cu
)

# ── C++ 源文件 ──
set(CPP_SOURCES
    core/engine.cpp
    core/weight_loader.cpp
    bindings.cpp
)

# ── 构建共享库 ──
if(pybind11_FOUND)
    # 使用 pybind11 的 add_pybind11_module（自动处理 Python 绑定）
    pybind11_add_module(taiji_cuda_engine
        ${CPP_SOURCES}
        ${CUDA_SOURCES}
    )
else()
    # Fallback: 手动构建
    add_library(taiji_cuda_engine SHARED
        ${CPP_SOURCES}
        ${CUDA_SOURCES}
    )
    target_include_directories(taiji_cuda_engine PRIVATE ${PYBIND11_INCLUDE})
    set_target_properties(taiji_cuda_engine PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
    )
endif()

# torch_python 提供 pybind11::type_caster<at::Tensor> 等 Python 绑定符号
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIB)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIB}")
    target_link_libraries(taiji_cuda_engine PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIB}")
else()
    message(WARNING "torch_python not found — Tensor type binding may fail")
    target_link_libraries(taiji_cuda_engine PRIVATE "${TORCH_LIBRARIES}")
endif()
target_include_directories(taiji_cuda_engine PRIVATE ${CMAKE_SOURCE_DIR})

# 解决 UTF-8 源文件在 MSVC (code page 936) 下的编译错误
if(MSVC)
    target_compile_options(taiji_cuda_engine PRIVATE /utf-8)
endif()

# 添加 Python include 路径（pybind11 需要 Python.h）
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_INCLUDE_DIR)
    target_include_directories(taiji_cuda_engine PRIVATE "${PYTHON_INCLUDE_DIR}")
endif()

# 添加 Python lib 路径（链接 python310.lib 等）
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys,os; print(os.path.join(sys.base_prefix,'libs'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_LIB_DIR AND EXISTS "${PYTHON_LIB_DIR}")
    target_link_directories(taiji_cuda_engine PRIVATE "${PYTHON_LIB_DIR}")
    message(STATUS "Python lib dir: ${PYTHON_LIB_DIR}")
endif()

# Windows 特殊处理
if(WIN32)
    # 复制 torch DLL 到输出目录
    set(TORCH_DLL_DIR "${TORCH_INSTALL_PREFIX}/lib")
    add_custom_command(TARGET taiji_cuda_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_DLL_DIR}/c10.dll"
            "${TORCH_DLL_DIR}/torch_cpu.dll"
            "$<TARGET_FILE_DIR:taiji_cuda_engine>"
        COMMENT "Copying Torch DLLs..."
    )
    if(EXISTS "${TORCH_DLL_DIR}/torch_cuda.dll")
        add_custom_command(TARGET taiji_cuda_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TORCH_DLL_DIR}/torch_cuda.dll"
                "$<TARGET_FILE_DIR:taiji_cuda_engine>"
        )
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output: ${CMAKE_BINARY_DIR}")