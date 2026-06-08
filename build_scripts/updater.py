"""
OmniCore 热更新核心模块（安全强化版）
======================================
提供版本管理、模块热重载、补丁安装、远程更新检查等能力。
无需重新打包 .exe，即可实现 Python 代码和前端 UI 的热更新。

使用流程:
    1. 开发者运行 build_update.py 生成更新包 (.zip)
    2. 用户端自动检测更新（GitHub Release 或手动上传）
    3. updater 将补丁解压到 update_code/ 和 update_frontend/
    4. Python 模块通过 ModuleHotReloader 热重载
    5. 前端通过 update_frontend/ 目录自动生效（无需重启）
"""
import os
import sys
import json
import shutil
import hashlib
import logging
import importlib
import threading
import tempfile
import zipfile
import urllib.request
import urllib.error
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, asdict
from core.config import get_external_path

logger = logging.getLogger("Updater")

# ======================== 版本管理 ========================

@dataclass
class VersionInfo:
    """版本信息"""
    version: str = "0.0.0"
    build_date: str = ""
    update_url: str = ""
    changelog: str = ""
    notes: str = ""

    def __str__(self):
        return f"v{self.version} ({self.build_date})"


class VersionManager:
    """版本管理器：读取/写入 version.json"""

    def __init__(self):
        self._path = get_external_path("version.json")
        self._lock = threading.Lock()
        self._info = self._load()

    def _load(self) -> VersionInfo:
        """从磁盘加载版本信息"""
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return VersionInfo(**data)
        except Exception as e:
            logger.warning(f"读取版本信息失败: {e}")
        return VersionInfo()

    def save(self, info: VersionInfo):
        """保存版本信息到磁盘"""
        with self._lock:
            self._info = info
            try:
                with open(self._path, "w", encoding="utf-8") as f:
                    json.dump(asdict(info), f, ensure_ascii=False, indent=2)
                logger.info(f"版本信息已保存: {info}")
            except Exception as e:
                logger.error(f"保存版本信息失败: {e}")

    @property
    def current(self) -> VersionInfo:
        return self._info

    def set_version(self, version: str, build_date: str = "", changelog: str = ""):
        """设置当前版本"""
        info = self._info
        info.version = version
        if build_date:
            info.build_date = build_date
        if changelog:
            info.changelog = changelog
        self.save(info)

    def set_update_url(self, url: str):
        """设置更新检查 URL（GitHub Release API 或自定义）"""
        info = self._info
        info.update_url = url
        self.save(info)


# ======================== 模块热重载 ========================

class ModuleHotReloader:
    """
    Python 模块热重载器（支持子目录包结构）

    功能：
    1. 重载已导入的模块（更新函数、类等）
    2. 从 update_code/ 目录加载新模块（支持子目录结构）
    3. 支持模块间依赖重载
    4. 安全的回滚机制

    路径映射：
    - update_code/xxx.py            -> xxx（扁平模块，兼容旧版）
    - update_code/api/routes_chat.py -> api.routes_chat（子目录包结构）
    """

    def __init__(self, update_dir: str = None):
        self.update_dir = update_dir or get_external_path("update_code")
        os.makedirs(self.update_dir, exist_ok=True)
        if self.update_dir not in sys.path:
            sys.path.insert(0, self.update_dir)

    def _find_patch_path(self, module_name: str) -> str:
        """定位补丁文件路径（支持扁平和子目录两种结构）"""
        # 路径 1：扁平模式
        flat_path = os.path.join(self.update_dir, f"{module_name}.py")
        if os.path.exists(flat_path):
            return flat_path
        # 路径 2：子目录模式
        rel_path = module_name.replace(".", os.sep) + ".py"
        subdir_path = os.path.join(self.update_dir, rel_path)
        if os.path.exists(subdir_path):
            return subdir_path
        return ""

    def get_available_patches(self) -> List[str]:
        """列出 update_code 目录下可用的补丁模块（递归扫描子目录）"""
        patches = []
        if not os.path.exists(self.update_dir):
            return patches
        for root, dirs, files in os.walk(self.update_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    fpath = os.path.join(root, f)
                    rel_path = os.path.relpath(fpath, self.update_dir)
                    mod_name = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]
                    patches.append(mod_name)
        return sorted(patches)

    def reload_module(self, module_name: str) -> bool:
        """
        热重载指定模块（支持子目录结构）

        从 update_code/ 查找该模块的最新版本并重载。
        如果模块尚未导入，则从 update_code/ 导入。
        如果模块已导入，则执行完整重载。

        Args:
            module_name: 模块名（如 'api.routes_chat' 或 'config'）

        Returns:
            是否重载成功
        """
        try:
            patch_path = self._find_patch_path(module_name)
            if not patch_path:
                logger.warning(f"补丁文件不存在: {module_name}")
                return False

            # 确保 update_dir 在导入路径中
            if self.update_dir not in sys.path:
                sys.path.insert(0, self.update_dir)

            old_module = None
            if module_name in sys.modules:
                old_module = sys.modules[module_name]
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, patch_path
                    )
                    if spec is None:
                        logger.error(f"无法创建模块 spec: {module_name}")
                        return False

                    new_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = new_module
                    spec.loader.exec_module(new_module)
                    logger.info(f"OK 模块热重载成功: {module_name}")
                    return True
                except Exception as e:
                    # 重载失败，恢复旧模块
                    if old_module is not None:
                        sys.modules[module_name] = old_module
                    logger.error(f"X 模块重载失败 {module_name}: {e}")
                    return False
            else:
                # 模块尚未导入，直接从补丁文件加载
                spec = importlib.util.spec_from_file_location(
                    module_name, patch_path
                )
                if spec is None:
                    logger.error(f"无法创建模块 spec: {module_name}")
                    return False

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"OK 模块从补丁加载: {module_name}")
                return True

        except Exception as e:
            logger.error(f"X 模块重载异常 {module_name}: {e}")
            return False

    def reload_all_patches(self) -> Dict[str, bool]:
        """
        重载 update_code/ 目录下所有可用的补丁模块（递归扫描子目录）

        Returns:
            {模块名: 是否成功}
        """
        results = {}
        patches = self.get_available_patches()
        for module_name in patches:
            results[module_name] = self.reload_module(module_name)
        return results

    def reload_with_deps(self, module_name: str, depth: int = 0) -> Dict[str, bool]:
        """
        重载模块及其依赖（实验性功能）

        注意：由于 Python 的动态特性，自动追踪依赖可能有遗漏。
        推荐手动指定需要重载的模块列表。
        """
        results = {}
        if depth > 5:  # 防止循环依赖
            return results

        results[module_name] = self.reload_module(module_name)
        return results

    def get_module_source(self, module_name: str) -> Optional[str]:
        """获取补丁模块的源代码"""
        patch_path = self._find_patch_path(module_name)
        if patch_path and os.path.exists(patch_path):
            try:
                with open(patch_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"读取补丁源码失败 {module_name}: {e}")
        return None


# ======================== 远程更新检查 ========================

class UpdateChecker:
    """
    远程更新检查器
    
    支持两种更新源:
    1. GitHub Releases API
    2. 自定义 HTTP 服务器（返回 JSON 版本信息）
    """

    def __init__(self, version_manager: VersionManager, timeout: int = 10):
        self.version_manager = version_manager
        self.timeout = timeout
        self._latest_info: Optional[VersionInfo] = None

    def check_github_release(self, repo: str) -> Optional[VersionInfo]:
        """
        检查 GitHub Release 更新
        
        Args:
            repo: "owner/repo" 格式的仓库名
            
        Returns:
            如果有更新，返回最新版本信息；否则返回 None
        """
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        try:
            req = urllib.request.Request(
                api_url,
                headers={
                    "User-Agent": "OmniCore-Updater/1.0",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            tag = data.get("tag_name", "").lstrip("v")
            body = data.get("body", "")
            # 查找更新包下载 URL
            update_asset_url = ""
            for asset in data.get("assets", []):
                if asset["name"].endswith("-update.zip"):
                    update_asset_url = asset["browser_download_url"]
                    break
            
            info = VersionInfo(
                version=tag,
                build_date=data.get("published_at", "")[:10],
                update_url=update_asset_url,
                changelog=body,
                notes=data.get("name", ""),
            )
            
            self._latest_info = info
            return info

        except urllib.error.HTTPError as e:
            logger.warning(f"GitHub API 请求失败 (HTTP {e.code}): {e.reason}")
            if e.code == 403:
                logger.warning("GitHub API 速率限制，请稍后再试")
            return None
        except Exception as e:
            logger.warning(f"检查 GitHub Release 失败: {e}")
            return None

    def check_custom_url(self, url: str) -> Optional[VersionInfo]:
        """
        从自定义 URL 检查更新
        
        URL 应返回 JSON：
        {
            "version": "1.0.1",
            "build_date": "2026-05-08",
            "update_url": "https://example.com/update.zip",
            "changelog": "修复了xxx",
            "notes": "v1.0.1"
        }
        """
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-Updater/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            info = VersionInfo(**data)
            self._latest_info = info
            return info

        except Exception as e:
            logger.warning(f"检查自定义更新源失败: {e}")
            return None

    def has_update(self, latest: VersionInfo) -> bool:
        """判断是否有新版本"""
        current = self.version_manager.current.version
        latest_ver = latest.version
        try:
            cur_parts = [int(x) for x in current.split(".")]
            lat_parts = [int(x) for x in latest_ver.split(".")]
            # 补齐长度
            max_len = max(len(cur_parts), len(lat_parts))
            cur_parts += [0] * (max_len - len(cur_parts))
            lat_parts += [0] * (max_len - len(lat_parts))
            return lat_parts > cur_parts
        except ValueError:
            # 非数字版本号，按字符串比较
            return latest_ver != current

    @property
    def latest_info(self) -> Optional[VersionInfo]:
        return self._latest_info


# ======================== 更新包安装器 ========================

class UpdatePackageInstaller:
    """
    更新包安装器
    
    处理三种更新内容:
    1. Python 代码补丁 -> update_code/
    2. 前端 UI 补丁 -> update_frontend/
    3. 版本信息 -> version.json
    
    更新包格式 (.zip):
    ├── version.json              # 新版本信息
    ├── manifest.json             # 文件清单（含校验和）
    └── update_code/
    │   ├── module1.py
    │   └── module2.py
    └── update_frontend/
        ├── index.html
        ├── assets/
        └── ...
    """

    def __init__(self, callback: Callable = None):
        self.callback = callback  # 进度回调 callback(percent, message)
        self._lock = threading.Lock()

    def _report(self, percent: float, message: str):
        """报告进度"""
        if self.callback:
            try:
                self.callback(percent, message)
            except Exception:
                pass
        logger.info(f"[{percent:.0f}%] {message}")

    def install_from_url(self, url: str, verify_ssl: bool = True) -> bool:
        """
        从 URL 下载并安装更新包
        
        Args:
            url: 更新包下载 URL
            verify_ssl: 是否验证 SSL 证书
            
        Returns:
            安装是否成功
        """
        self._report(0, "正在下载更新包...")
        try:
            # 下载到临时文件
            tmp_dir = tempfile.mkdtemp(prefix="omnicore_update_")
            zip_path = os.path.join(tmp_dir, "update_package.zip")

            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-Updater/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = min(30, (downloaded / total_size) * 30)
                            self._report(pct, f"下载中... {downloaded // 1024}KB / {total_size // 1024}KB")

            self._report(30, "下载完成，正在安装...")
            result = self.install_from_zip(zip_path)
            
            # 清理临时文件
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
            if result:
                self._report(100, "✅ 更新安装完成！")
            return result

        except Exception as e:
            logger.error(f"下载更新包失败: {e}")
            self._report(0, f"❌ 下载失败: {e}")
            return False

    def install_from_zip(self, zip_path: str) -> bool:
        """
        从本地 ZIP 文件安装更新
        
        Args:
            zip_path: ZIP 文件路径
            
        Returns:
            安装是否成功
        """
        with self._lock:
            return self._do_install(zip_path)

    def _do_install(self, zip_path: str) -> bool:
        """内部安装逻辑（已加锁）"""
        tmp_extract = None
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"更新包文件不存在: {zip_path}")

            # 解压到临时目录
            tmp_extract = tempfile.mkdtemp(prefix="omnicore_extract_")
            with zipfile.ZipFile(zip_path, "r") as zf:
                # 验证压缩包完整性
                bad_file = zf.testzip()
                if bad_file:
                    raise zipfile.BadZipFile(f"压缩包损坏: {bad_file}")
                zf.extractall(tmp_extract)

            self._report(40, "验证更新包...")

            # 验证 manifest.json
            manifest_path = os.path.join(tmp_extract, "manifest.json")
            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

            # 验证文件校验和
            for file_entry in manifest.get("files", []):
                rel_path = file_entry.get("path", "")
                expected_hash = file_entry.get("sha256", "")
                if not rel_path or not expected_hash:
                    continue
                full_path = os.path.join(tmp_extract, rel_path)
                if os.path.exists(full_path):
                    actual_hash = self._sha256_file(full_path)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"文件校验失败: {rel_path}\n"
                            f"  期望: {expected_hash}\n"
                            f"  实际: {actual_hash}"
                        )

            self._report(50, "验证通过，正在安装...")

            # --- 1. 安装 Python 补丁（支持子目录结构） ---
            src_code = os.path.join(tmp_extract, "update_code")
            dst_code = get_external_path("update_code")
            if os.path.exists(src_code):
                self._report(55, "安装 Python 补丁...")
                os.makedirs(dst_code, exist_ok=True)
                patch_count = 0
                for root, dirs, files in os.walk(src_code):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for fname in files:
                        if fname.endswith(".py"):
                            src_file = os.path.join(root, fname)
                            rel_path = os.path.relpath(src_file, src_code)
                            dst_file = os.path.join(dst_code, rel_path)
                            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                            shutil.copy2(src_file, dst_file)
                            patch_count += 1
                            logger.info(f"  安装 Python 补丁: {rel_path}")
                logger.info(f"  共安装 {patch_count} 个 Python 补丁（含子目录）")

            # --- 2. 安装前端 UI 补丁 ---
            src_ui = os.path.join(tmp_extract, "update_frontend")
            dst_ui = get_external_path("update_frontend")
            if os.path.exists(src_ui):
                self._report(70, "安装前端 UI 补丁...")
                # 使用原子性复制（复用 hot_update.py 的逻辑）
                self._atomic_copy(src_ui, dst_ui)
                logger.info("  前端 UI 补丁已安装")

            # --- 3. 更新版本信息 ---
            src_version = os.path.join(tmp_extract, "version.json")
            if os.path.exists(src_version):
                self._report(85, "更新版本信息...")
                dst_version = get_external_path("version.json")
                shutil.copy2(src_version, dst_version)
                logger.info(f"  版本信息已更新: {src_version}")

            # 清理临时目录
            shutil.rmtree(tmp_extract, ignore_errors=True)
            tmp_extract = None

            self._report(90, "应用代码补丁...")
            
            # --- 4. 热重载 Python 补丁 ---
            reloader = ModuleHotReloader()
            results = reloader.reload_all_patches()
            success_count = sum(1 for v in results.values() if v)
            fail_count = sum(1 for v in results.values() if not v)
            
            if fail_count > 0:
                logger.warning(f"部分补丁重载失败: {fail_count} 个失败")
            if success_count > 0:
                logger.info(f"✅ {success_count} 个 Python 模块已热重载")

            return True

        except Exception as e:
            logger.error(f"安装更新包失败: {e}")
            self._report(0, f"❌ 安装失败: {e}")
            return False

        finally:
            # 确保清理临时文件
            if tmp_extract and os.path.exists(tmp_extract):
                shutil.rmtree(tmp_extract, ignore_errors=True)

    def _atomic_copy(self, src_dir: str, dst_dir: str, max_retries: int = 3):
        """原子性复制目录（带备份和回滚）"""
        parent = os.path.dirname(dst_dir)
        os.makedirs(parent, exist_ok=True)
        
        tmp_dir = os.path.join(parent, f"_update_tmp_{int(time.time())}")
        backup_dir = None
        
        try:
            shutil.copytree(src_dir, tmp_dir)
            
            if os.path.exists(dst_dir):
                backup_dir = os.path.join(parent, f"_update_bak_{int(time.time())}")
                for attempt in range(max_retries):
                    try:
                        os.rename(dst_dir, backup_dir)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            raise
            
            for attempt in range(max_retries):
                try:
                    os.rename(tmp_dir, dst_dir)
                    tmp_dir = None
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        raise
            
            if backup_dir and os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
                
        except Exception as e:
            # 回滚
            if backup_dir and os.path.exists(backup_dir) and not os.path.exists(dst_dir):
                os.rename(backup_dir, dst_dir)
            raise e
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _sha256_file(filepath: str) -> str:
        """计算文件的 SHA256 哈希"""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# ======================== 便捷函数 ========================

def get_version() -> VersionInfo:
    """获取当前版本信息的快捷函数"""
    return VersionManager().current


def hot_reload_all() -> Dict[str, bool]:
    """热重载所有补丁的快捷函数"""
    return ModuleHotReloader().reload_all_patches()


def check_for_updates(repo: str = "") -> Optional[VersionInfo]:
    """
    检查更新的快捷函数
    
    Args:
        repo: GitHub "owner/repo" 或自定义 URL
        
    Returns:
        如果有更新返回版本信息，否则返回 None
    """
    vm = VersionManager()
    checker = UpdateChecker(vm)
    
    if repo.startswith("http"):
        return checker.check_custom_url(repo)
    elif "/" in repo:
        return checker.check_github_release(repo)
    else:
        # 使用配置中的 update_url
        url = vm.current.update_url
        if url.startswith("http"):
            return checker.check_custom_url(url)
        return None


def install_update(zip_path: str) -> bool:
    """
    安装更新包的快捷函数
    
    Args:
        zip_path: 本地 ZIP 文件路径或 URL
        
    Returns:
        安装是否成功
    """
    installer = UpdatePackageInstaller()
    if zip_path.startswith("http"):
        return installer.install_from_url(zip_path)
    else:
        return installer.install_from_zip(zip_path)


if __name__ == "__main__":
    # 命令行模式
    import argparse
    parser = argparse.ArgumentParser(description="OmniCore 更新工具")
    parser.add_argument("--check", help="检查更新 (GitHub repo 或 URL)", default="")
    parser.add_argument("--install", help="安装更新包 (本地路径或 URL)", default="")
    parser.add_argument("--reload", help="热重载指定模块", default="")
    parser.add_argument("--reload-all", action="store_true", help="热重载所有补丁")
    parser.add_argument("--version", action="store_true", help="显示当前版本")
    
    args = parser.parse_args()
    
    if args.version:
        v = get_version()
        print(f"当前版本: {v}")
        print(f"更新地址: {v.update_url or '(未设置)'}")
        
    if args.check:
        print(f"正在检查更新: {args.check}...")
        info = check_for_updates(args.check)
        if info:
            print(f"最新版本: v{info.version}")
            print(f"发布日期: {info.build_date}")
            print(f"更新日志:\n{info.changelog}")
        else:
            print("当前已是最新版本或检查失败")
            
    if args.install:
        print(f"正在安装更新: {args.install}...")
        ok = install_update(args.install)
        print("✅ 安装完成!" if ok else "❌ 安装失败!")
        
    if args.reload:
        reloader = ModuleHotReloader()
        ok = reloader.reload_module(args.reload)
        print(f"✅ 重载成功: {args.reload}" if ok else f"❌ 重载失败: {args.reload}")
        
    if args.reload_all:
        reloader = ModuleHotReloader()
        results = reloader.reload_all_patches()
        for name, ok in results.items():
            print(f"  {'✅' if ok else '❌'} {name}")
