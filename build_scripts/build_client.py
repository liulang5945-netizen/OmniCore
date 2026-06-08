"""
OmniCore 桌面客户端打包脚本 (PyInstaller)
"""
import os, sys, json, shutil, subprocess, io
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

def get_base_dir():
    d = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(d) if os.path.basename(d) == "build_scripts" else d

def find_venv_python(base_dir):
    for c in [os.path.join(base_dir, "venv", "Scripts", "python.exe"),
              os.path.join(base_dir, ".venv", "Scripts", "python.exe")]:
        if os.path.exists(c): return c
    return None

def switch_to_venv_python(base_dir):
    vp = find_venv_python(base_dir)
    if vp is None: print("[Build] ⚠️ 未找到 venv"); return False
    if sys.executable.lower().replace("/","\\") == vp.lower().replace("/","\\"): return True
    print(f"[Build] ➜ 切换到 venv: {vp}")
    r = subprocess.run([vp, __file__] + sys.argv[1:], cwd=base_dir)
    sys.exit(r.returncode)

REQUIRED = ["PyInstaller","fastapi","torch","transformers","peft",
    "langchain","langchain_community","langchain_openai","langchain_core",
    "langchain_experimental","sentence_transformers","PyQt6","uvicorn",
    "pydantic","bitsandbytes","datasets","numpy","scipy","pandas",
    "requests","PyPDF2","python-docx","pdfminer.six","jieba","accelerate"]

def check_venv_deps(pkgs):
    m = [p for p in pkgs if not _try_import(p)]
    if m:
        print(f"[Build] ❌ 缺少: {', '.join(m)}，安装中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + m)
        print("[Build] ✅ 安装完成")
    return True

def _try_import(p): 
    try: __import__(p); return True
    except ImportError: return False

def check_deps():
    try:
        import PyInstaller as pi
        print(f"[Build] ✅ PyInstaller {pi.__version__}")
    except ImportError:
        print("[Build] 安装 PyInstaller...")
        subprocess.check_call([sys.executable,"-m","pip","install","pyinstaller"])

def check_arch():
    import struct
    b = struct.calcsize("P")*8
    print(f"[Build] 🖥️ Python: {b}-bit")
    if b==32:
        if input("[Build] 32-bit 继续?(y/N): ").strip().lower()!='y': sys.exit(0)
    else: print("[Build] ✅ 64-bit")

# ── 强制清理：takeown + icacls /reset + rd ──
def _sanitize_path(path):
    """规范化路径，防止路径注入和特殊字符问题"""
    return os.path.normpath(path)

def _force_delete(path):
    """最激进的删除：先 takeown 夺取所有权 → icacls 重置 ACL → rd 删除
    返回 True 仅当目录/文件确实被成功删除。
    """
    path = _sanitize_path(path)
    # 先验证路径是否存在
    if not os.path.exists(path):
        return True  # 已经不存在，视为成功
    # CMD 方式：使用列表参数避免 shell 注入
    try:
        result = subprocess.run(
            ['cmd', '/c', 'takeown', '/f', path, '/r', '/d', 'y'],
            timeout=60, capture_output=True)
        subprocess.run(
            ['cmd', '/c', 'icacls', path, '/reset', '/t', '/q'],
            timeout=60, capture_output=True)
        result = subprocess.run(
            ['cmd', '/c', 'rd', '/s', '/q', path],
            timeout=60, capture_output=True)
        if not os.path.exists(path):
            return True
    except Exception as e:
        print(f"[Build] ⚠️ CMD 删除失败: {e}")
    # PowerShell 作为备选：转义单引号防止路径中断
    ps_path = path.replace("'", "''")
    try:
        subprocess.check_call(
            ['powershell', '-NoProfile', '-Command',
             f"if (Test-Path '{ps_path}') {{ Remove-Item -Path '{ps_path}' -Recurse -Force -ErrorAction Stop }}"],
            timeout=30, capture_output=True)
        if not os.path.exists(path):
            return True
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='replace').strip() if e.stderr else "未知错误"
        print(f"[Build] ⚠️ PowerShell 删除失败: {stderr_msg}")
    except Exception as e:
        print(f"[Build] ⚠️ PowerShell 删除异常: {e}")
    return False

def _force_clean(path, label):
    if not os.path.exists(path): return
    print(f"[Build] 清理: {path}")
    if _force_delete(path):
        print(f"[Build] ✅ 已清理: {path}")
    else:
        print(f"[Build] ⚠️ 无法清理 {label}，跳过")

# ── .spec monkey-patch：替换 PyInstaller 的文件操作 ──
PYI_PATCH = r"""
import PyInstaller.building.utils as _pibu
import PyInstaller.building.build_main as _pyi_bm
import os as _os, subprocess as _sp, shutil as _shutil

# ── 修补 find_binary_dependencies（PyInstaller 6.x 新增的激进二进制依赖扫描）──
# 该函数会在隔离子进程中尝试 import 所有已收集的包，用于发现动态链接库依赖。
# 但 sentence_transformers/torch 等超大包会导致子进程因内存不足而崩溃。
# 修补策略：过滤掉导致崩溃的包，仅对安全的包执行扫描。
_ORIG_FIND_BINARY_DEPS = _pyi_bm.find_binary_dependencies
_PACKAGES_TO_SKIP = frozenset({
    'sentence_transformers', 'torch', 'transformers',
    'sklearn', 'scipy', 'numpy', 'pandas', 'matplotlib',
    'bitsandbytes', 'accelerate', 'peft', 'datasets',
    'langchain', 'langchain_community', 'langchain_openai',
    'langchain_core', 'langchain_experimental',
    'huggingface_hub', 'tokenizers', 'safetensors',
    'sympy', 'networkx', 'filelock', 'fsspec',
    'cv2', 'imageio', 'librosa', 'moviepy',
    'grpc', 'alembic', 'optuna', 'sqlalchemy',
    'openai', 'httpx', 'httpcore', 'anyio',
})
_skipped_count = [0]

def _should_skip(pkg_name):
    # 检查包名是否应跳过二进制依赖扫描。仅匹配包名本身和 Python 子包（.分隔符）。
    if pkg_name in _PACKAGES_TO_SKIP:
        return True
    for skip in _PACKAGES_TO_SKIP:
        if pkg_name.startswith(skip + '.'):
            return True
    return False

def _patched_find_binary_deps(binaries, collected_packages, *args, **kwargs):
    safe_packages = [p for p in collected_packages if not _should_skip(p)]
    skipped = len(collected_packages) - len(safe_packages)
    if skipped > 0:
        _skipped_count[0] += skipped
        print(f"[PYI-Patch] Skipped {skipped} heavy packages from binary dep scan")
    if safe_packages:
        return _ORIG_FIND_BINARY_DEPS(binaries, safe_packages, *args, **kwargs)
    return []
_pyi_bm.find_binary_dependencies = _patched_find_binary_deps


def _force_delete(p):
    # 强制删除路径，返回 True 仅当确实被成功删除。
    p = _os.path.normpath(p)
    if not _os.path.exists(p):
        return True
    # CMD 方式：使用列表参数避免 shell 注入
    try:
        _sp.run(['cmd', '/c', 'takeown', '/f', p, '/r', '/d', 'y'],
                timeout=60, capture_output=True)
        _sp.run(['cmd', '/c', 'icacls', p, '/reset', '/t', '/q'],
                timeout=60, capture_output=True)
        _sp.run(['cmd', '/c', 'rd', '/s', '/q', p],
                timeout=60, capture_output=True)
        if not _os.path.exists(p):
            return True
    except Exception:
        pass
    # PowerShell 备选：转义单引号
    ps_p = p.replace("'", "''")
    try:
        _sp.check_call(
            ['powershell', '-NoProfile', '-Command',
             "if (Test-Path '" + ps_p + "') { Remove-Item -Path '" + ps_p + "' -Recurse -Force -ErrorAction Stop }"],
            timeout=30, capture_output=True)
        return not _os.path.exists(p)
    except Exception:
        return False

# patch _rmtree
_orig_rmtree = _pibu._rmtree
def _patched_rmtree(path):
    if _os.path.isdir(path):
        if _force_delete(path): return
    _orig_rmtree(path)
_pibu._rmtree = _patched_rmtree

# patch _make_clean_directory
_orig_mcd = _pibu._make_clean_directory
def _patched_mcd(path):
    if _os.path.exists(path):
        if _force_delete(path):
            _os.makedirs(path, exist_ok=True)
            return
    _orig_mcd(path)
_pibu._make_clean_directory = _patched_mcd

# patch shutil.copyfile
_orig_copyfile = _shutil.copyfile
def _patched_copyfile(src, dst):
    if _os.path.exists(dst): _force_delete(dst)
    _orig_copyfile(src, dst)
_shutil.copyfile = _patched_copyfile

"""

def build(is_public=False):
    check_arch()
    base_dir = get_base_dir()
    dist_dir = os.path.join(base_dir, "dist", "OmniCore")
    mode_label = "公开版 (不含 ModelSelf)" if is_public else "私有版 (含 ModelSelf)"
    print(f"{'='*60}\n  OmniCore 打包 [{mode_label}]\n  Python: {sys.executable}\n  版本: {sys.version.split()[0]}\n{'='*60}\n")

    for d in ["build","dist"]:
        _force_clean(os.path.join(base_dir, d), d)

    for cd in ["api/model_cache","model_cache"]:
        cp = os.path.join(base_dir, cd)
        if os.path.exists(cp):
            for item in os.listdir(cp):
                ip = os.path.join(cp, item)
                try:
                    if os.path.isdir(ip): shutil.rmtree(ip, ignore_errors=True)
                    else: os.remove(ip)
                except Exception: pass

    for d in ["update_code","update_frontend","external_libs"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    for s in ["agent","api","build_scripts","core","model","taiji","tools"]:
        os.makedirs(os.path.join(base_dir,"update_code",s), exist_ok=True)

    vf = os.path.join(base_dir,"version.json")
    bv = "1.0.0"
    if os.path.exists(vf):
        try:
            with open(vf,"r",encoding="utf-8") as f: bv = json.load(f).get("version","1.0.0")
        except Exception: pass
    print(f"[Build] 📋 版本: v{bv}")

    entry = os.path.join(base_dir,"api","run_app.py").replace('\\','/')
    bdu = base_dir.replace('\\','/')
    if not os.path.exists(entry): print(f"[Build] ❌ 找不到 {entry}"); sys.exit(1)

    idata = "('icon.ico', '.')," if os.path.exists(os.path.join(base_dir,'icon.ico')) else ""
    iexe = "icon='icon.ico'," if os.path.exists(os.path.join(base_dir,'icon.ico')) else ""
    fdata = "('frontend/dist', 'frontend/dist')," if os.path.exists(os.path.join(base_dir,'frontend','dist')) else ""
    rhook = f"'{bdu}/build_scripts/runtime_hook_numpy.py'" if os.path.exists(os.path.join(base_dir,'build_scripts','runtime_hook_numpy.py')) else ""

    # 根据公开/私有版决定是否包含 ModelSelf
    if is_public:
        taiji_datas = ""
        taiji_imports = ""
        print("[Build] 📋 公开版: 不包含 ModelSelf")
    else:
        taiji_datas = "('taiji','taiji'),"
        taiji_imports = (
            "'taiji','taiji.architecture','taiji.layers','taiji.config',"
            "'taiji.tokenizer','taiji.inference','taiji.loader','taiji.trainer',"
        )
        print("[Build] 📋 私有版: 包含 ModelSelf")

    # Note: Do NOT exclude transformers.models.* here.
    # The auto modules dynamically import model sub-modules (e.g. tokenization_auto -> encoder_decoder),
    # so excluding them causes runtime ModuleNotFoundError.
    _all_excludes = [
        'torchvision','torchaudio','tensorflow','tensorboard',
        'notebook','jupyter','IPython','bokeh','plotly','altair',
        'tkinter','PyQt5','PySide2','PySide6','wx','pytest','sphinx',
        'nose','h5py','openpyxl','pyarrow','fastparquet','cython','numba','curses',
    ]
    _excludes_str = repr(_all_excludes)

    spec = PYI_PATCH + f"""
import os, sys, ctypes.util, glob
sys.setrecursionlimit(10000)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
import PyQt6

# ── 收集 ctypes 依赖的 libffi DLL ──
_wb = []
_ffi_found = False
for _ffi_name in ['libffi-8.dll', 'libffi-7.dll']:
    # 1. 通过 ctypes.util 查找
    _ffi_path = ctypes.util.find_library(_ffi_name)
    if _ffi_path and os.path.isfile(_ffi_path):
        _wb.append((_ffi_path, '.'))
        _ffi_found = True
        break
    # 2. 在当前 Python (venv) 的 DLLs 目录中搜索
    _py_dlls = os.path.join(os.path.dirname(sys.executable), 'DLLs')
    _ffi_path = os.path.join(_py_dlls, _ffi_name)
    if os.path.isfile(_ffi_path):
        _wb.append((_ffi_path, '.'))
        _ffi_found = True
        break
    # 3. 在基础 Python (base_prefix) 的 DLLs 目录中搜索（venv 场景）
    _base_dlls = os.path.join(sys.base_prefix, 'DLLs')
    _ffi_path = os.path.join(_base_dlls, _ffi_name)
    if os.path.isfile(_ffi_path):
        _wb.append((_ffi_path, '.'))
        _ffi_found = True
        break
    # 4. 在 PATH 中搜索
    for _p in os.environ.get('PATH', '').split(os.pathsep):
        _ffi_path = os.path.join(_p, _ffi_name)
        if os.path.isfile(_ffi_path):
            _wb.append((_ffi_path, '.'))
            _ffi_found = True
            break
    if _ffi_found:
        break
if not _ffi_found:
    print("[Build] ⚠️ 未找到 libffi DLL，_ctypes 可能无法正常工作")

_q6b = os.path.join(os.path.dirname(PyQt6.__path__[0]), 'PyQt6', 'Qt6')
_wd = []
_wep = os.path.join(_q6b, 'bin', 'QtWebEngineProcess.exe')
if os.path.exists(_wep): _wb.append((_wep, 'PyQt6/Qt6/bin'))
_qt = os.path.join(_q6b, 'translations')
if os.path.exists(_qt):
    _ld = os.path.join(_qt, 'qtwebengine_locales')
    if os.path.exists(_ld): _wd.append((_ld, 'PyQt6/Qt6/translations/qtwebengine_locales'))
    import glob as _g
    for _qm in _g.glob(os.path.join(_qt, 'qtwebengine_*.qm')): _wd.append((_qm, 'PyQt6/Qt6/translations'))
_qr = os.path.join(_q6b, 'resources')
if os.path.exists(_qr): _wd.append((_qr, 'PyQt6/Qt6/resources'))

a = Analysis(['{entry}'], pathex=['{bdu}'], binaries=_wb,
    datas=[{idata}('update_code','update_code'),('external_libs','external_libs'),
        ('agent','agent'),('api','api'),('build_scripts','build_scripts'),
        ('core','core'),('model','model'),{taiji_datas}('tools','tools'),{fdata}]+_wd,
    hiddenimports=[
        'PyQt6','PyQt6.QtCore','PyQt6.QtGui','PyQt6.QtWidgets','PyQt6.QtWebEngineWidgets','PyQt6.QtWebEngineCore',
        'fastapi','uvicorn','starlette','pydantic','multipart','pydantic_core','pydantic_core._pydantic_core',
        'transformers','transformers.models.auto','transformers.models.auto.tokenization_auto',
        'transformers.models.auto.configuration_auto','transformers.models.auto.modeling_auto',
        'transformers.models.auto.processing_auto','transformers.generation','tokenizers',
        'transformers.models.llama','transformers.models.qwen2','transformers.models.qwen3',
        'transformers.models.gemma','transformers.models.bert','transformers.models.gpt2',
        'transformers.models.llama4','transformers.models.qwen2_5_vl','transformers.models.qwen2_5_omni',
        'transformers.models.code_llama',
        'torch','torch.nn','torch.optim','torch_directml','peft','peft.tuners.lora',
        {taiji_imports}'langchain','langchain.agents','langchain.tools','langchain_community','langchain_openai','langchain_core','langchain_experimental',
        'sentence_transformers','sentence_transformers.models',
        'sklearn','sklearn.feature_extraction.text','sklearn.metrics','scipy.sparse','scipy._lib','scipy._lib.array_api_compat','numpy.testing',
        'PyPDF2','docx','pdfminer','jsonlines','pandas','numpy','tqdm','matplotlib',
        'requests','bs4','duckduckgo_search','llama_cpp','bitsandbytes',
    ], hookspath=[], hooksconfig={{}}, runtime_hooks=[{rhook}],
    excludes={_excludes_str},
    noarchive=False)

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='OmniCore', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=False, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None, {iexe})
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, upx_exclude=[], name='OmniCore')
"""

    # 根据是否公开版选择 spec 文件名
    spec_name = "OmniCore_public.spec" if is_public else "OmniCore.spec"
    sp = os.path.join(base_dir, spec_name)
    with open(sp,"w",encoding="utf-8") as f: f.write(spec)
    print(f"[Build] ✅ spec: {sp}")

    print(f"\n[Build] 🚀 开始打包...\n")
    try:
        subprocess.check_call([sys.executable,"-m","PyInstaller","--clean","--noconfirm",sp], cwd=base_dir)
    except subprocess.CalledProcessError as e:
        print(f"\n[Build] ❌ 打包失败！错误码: {e.returncode}")
        sys.exit(1)

    print(f"\n[Build] 📦 打包完成！\n[Build]   输出: {dist_dir}")
    rp = os.path.join(base_dir,"requirements.txt")
    if os.path.exists(rp): shutil.copy(rp, os.path.join(dist_dir,"requirements.txt"))
    vjp = os.path.join(dist_dir,"version.json")
    with open(vjp,"w",encoding="utf-8") as f:
        json.dump({"version":bv,"build_date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_url":"","changelog":"","notes":f"OmniCore v{bv}"}, f, ensure_ascii=False, indent=2)
    vbs = os.path.join(dist_dir,"启动OmniCore.vbs")
    with open(vbs,"w",encoding="utf-8") as f:
        f.write('Set ws = CreateObject("WScript.Shell")\n')
        f.write('currentDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)\n')
        f.write('ws.Run chr(34) & currentDir & "\\OmniCore.exe" & chr(34), 0, False\n')
    print(f"[Build] ✅ 版本清单已生成\n{'='*60}\n  ✅ 打包成功!\n{'='*60}\n")
    return dist_dir

if __name__ == "__main__":
    base_dir = get_base_dir()
    print("[Build] 🔍 检查 Python 环境...")
    switch_to_venv_python(base_dir)
    print("[Build] 🔍 验证核心依赖...")
    check_venv_deps(REQUIRED)
    print("[Build] ✅ 所有核心依赖已就绪")
    check_deps()
    is_public = "--public" in sys.argv
    build(is_public=is_public)
