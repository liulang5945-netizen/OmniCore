"""
Taiji CUDA Engine — 一键编译脚本

用法:
    python csrc/setup.py build_ext --inplace
    或
    cd csrc && pip install -e .
"""

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # 确保输出目录存在
        os.makedirs(extdir, exist_ok=True)
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_PREFIX_PATH={self._get_torch_cmake_path()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        
        build_dir = os.path.join(ext.sourcedir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # CMake configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_dir,
        )
        
        # CMake build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir,
        )
    
    def _get_torch_cmake_path(self):
        try:
            import torch
            return torch.utils.cmake_prefix_path
        except ImportError:
            return ""

setup(
    name="taiji-cuda-engine",
    version="0.1.0",
    author="OmniCore",
    description="Taiji C++/CUDA Inference Engine",
    long_description="态极原生 C++/CUDA 推理引擎，消除 Python GIL 开销",
    ext_modules=[CMakeExtension("taiji_cuda_engine", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "pybind11>=2.10.0"],
)