"""
PyInstaller 运行时钩子 — 修复 numpy 2.x 在打包环境下的路径解析
===============================================================

问题背景：
numpy 2.x 将 C 扩展（.pyd）移到了 numpy/_core/ 子目录。PyInstaller 内置的
hook-numpy.py 虽然尝试使用 collect_dynamic_libs 收集这些文件，但由于这些 .pyd
文件位于包的子目录而非顶层，collect_dynamic_libs 返回空列表，导致打包后的程序
无法找到 _multiarray_umath 等关键扩展模块。

此钩子在 PyInstaller 解包后的运行时环境中运行，在 numpy 被导入前确保：
1. sys._MEIPASS 正确指向解包目录
2. numpy 的 _core 目录中的 .pyd 文件可以被正确加载
"""

import os
import sys


def _fix_numpy_path():
    """修复打包环境下 numpy 的路径解析问题。"""
    if not getattr(sys, 'frozen', False):
        return  # 非打包模式，无需修复

    meipass = getattr(sys, '_MEIPASS', None)
    if not meipass or not os.path.isdir(meipass):
        return

    # ---------------------------------------------------------------
    # 方案1：确保 numpy/_core 目录存在于 sys._MEIPASS 下
    #        PyInstaller 的 contents_directory='_internal' 会将文件
    #        放在 _internal 下，但老的 numpy 代码可能直接基于 __file__
    #        路径在 sys._MEIPASS 下查找。
    # ---------------------------------------------------------------
    numpy_core_dest = os.path.join(meipass, 'numpy', '_core')
    if not os.path.isdir(numpy_core_dest):
        # 尝试在 _internal 中查找
        internal_dir = os.path.join(meipass, '_internal')
        if os.path.isdir(internal_dir):
            alt_numpy_core = os.path.join(internal_dir, 'numpy', '_core')
            if os.path.isdir(alt_numpy_core):
                # 创建从 meipass/numpy/_core 到 _internal/numpy/_core 的路径映射
                # 方法：将 _internal 添加到 sys.path 或创建符号链接
                # 最简单可靠的方式：将 _internal 添加到 numpy 的 __path__
                pass  # 交给方案2处理

    # ---------------------------------------------------------------
    # 方案2：确保 numpy._core 包的 __path__ 包含正确的 .pyd 目录
    #        使用延迟导入，在 numpy 加载后立即修复
    # ---------------------------------------------------------------
    # 注册一个导入后回调（通过修改 sys.meta_path）
    class _NumpyPathFixer:
        """导入后修复器：在 numpy 加载后立即修复其路径"""
        
        def __init__(self):
            self._fixed = False
        
        def find_spec(self, fullname, path, target=None):
            if self._fixed:
                return None
            if fullname == 'numpy._core' or fullname == 'numpy._core.multiarray':
                self._fix_numpy_paths()
            return None
        
        def _fix_numpy_paths(self):
            if self._fixed:
                return
            self._fixed = True
            try:
                import numpy._core
                import numpy._core._multiarray_umath
                # 如果导入成功，说明路径已修复，无需额外操作
            except (ImportError, ModuleNotFoundError):
                # 尝试通过修改 sys.path 来修复
                meipass = getattr(sys, '_MEIPASS', '')
                internal = os.path.join(meipass, '_internal')
                if os.path.isdir(internal):
                    numpy_internal = os.path.join(internal, 'numpy')
                    if os.path.isdir(numpy_internal):
                        # 将 _internal/numpy 放入 sys.path 的最前面
                        if numpy_internal not in sys.path:
                            sys.path.insert(0, numpy_internal)
                        # 同时添加 _internal 本身
                        if internal not in sys.path:
                            sys.path.insert(0, internal)

    # 注册修复器
    _fixer = _NumpyPathFixer()
    sys.meta_path.insert(0, _fixer)


def _preload_numpy_cext():
    """在应用程序导入 numpy 前，预先加载 C 扩展模块。"""
    if not getattr(sys, 'frozen', False):
        return

    meipass = getattr(sys, '_MEIPASS', None)
    if not meipass:
        return

    # 确定 .pyd 文件的位置（可能在 _internal 下）
    pyd_search_dirs = [
        os.path.join(meipass, '_internal', 'numpy', '_core'),
        os.path.join(meipass, 'numpy', '_core'),
    ]

    for pyd_dir in pyd_search_dirs:
        if os.path.isdir(pyd_dir):
            # 将该目录加入 os.add_dll_directory 搜索路径（Windows）
            if sys.platform == 'win32':
                try:
                    os.add_dll_directory(pyd_dir)
                except (AttributeError, Exception):
                    pass
            # 同时也加入到 PATH 环境变量中（对某些旧版依赖有帮助）
            current_path = os.environ.get('PATH', '')
            if pyd_dir not in current_path:
                os.environ['PATH'] = pyd_dir + os.pathsep + current_path


# ==========================================
# 执行修复
# ==========================================
_fix_numpy_path()
_preload_numpy_cext()
