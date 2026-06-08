"""
OmniCore 前端热更新脚本（安全强化版）
=====================================
提供两种模式：
  1. `hot_update.py`          — 编译 Vue 前端并原子性部署到 update_frontend/
  2. `hot_update.py --package` — 编译前端 + 打包 Python 补丁为增量更新包 (.zip)

增量更新包结构 (.zip):
  ├── version.json       # 版本信息（version, build_date, changelog）
  ├── manifest.json      # 文件清单及 SHA256 校验值
  ├── update_frontend/   # 编译后的前端静态文件（可选）
  └── update_code/       # Python 补丁模块（可选）

安全特性：
  1. 原子性目录复制（临时目录 + 重命名 + 回滚）
  2. 更新包文件完整性校验（SHA256）
  3. 文件占用重试机制
"""
import os
import shutil
import subprocess
import sys
import json
import hashlib
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Optional

TEMP_PREFIX = "_hotupdate_tmp_"


def get_external_path(relative_path):
    """与 run_app.py 和 api_server.py 保持一致的外部路径解析"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)


def _sha256_file(filepath: str) -> str:
    """计算文件的 SHA256 哈希值"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(dirpath: str) -> dict:
    """递归计算目录下所有文件的 SHA256，返回 {相对路径: sha256}"""
    result = {}
    for root, dirs, files in os.walk(dirpath):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, dirpath)
            result[rel] = _sha256_file(fpath)
    return result


def _atomic_copy(src_dir: str, dst_dir: str, max_retries: int = 3):
    """
    原子性复制目录：
    1. 先复制到临时目录
    2. 如果目标存在，先重命名为备份
    3. 原子性重命名临时目录到目标
    4. 若失败，自动回滚
    """
    parent = os.path.dirname(dst_dir)
    os.makedirs(parent, exist_ok=True)
    
    # 创建临时目录
    tmp_dir = os.path.join(parent, f"{TEMP_PREFIX}{int(time.time())}")
    
    try:
        # 复制到临时目录
        shutil.copytree(src_dir, tmp_dir)
        
        # 如果已有目标，备份
        backup_dir = None
        if os.path.exists(dst_dir):
            backup_dir = os.path.join(parent, f"{TEMP_PREFIX}backup_{int(time.time())}")
            for attempt in range(max_retries):
                try:
                    os.rename(dst_dir, backup_dir)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        raise
        
        # 原子性重命名临时目录到目标
        for attempt in range(max_retries):
            try:
                os.rename(tmp_dir, dst_dir)
                tmp_dir = None  # 标记已成功
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    raise
        
        # 清理备份
        if backup_dir and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
            
        return True
        
    except Exception as e:
        print(f"[HotUpdate] 复制失败: {e}")
        # 回滚
        if backup_dir and os.path.exists(backup_dir) and not os.path.exists(dst_dir):
            os.rename(backup_dir, dst_dir)
        raise
    finally:
        # 清理残留临时目录
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def create_update_package(
    version: str = "1.0.0",
    changelog: str = "",
    include_frontend: bool = True,
    include_patches: bool = True,
    output_path: Optional[str] = None,
) -> str:
    """
    生成增量更新包 (.zip)
    
    参数:
        version:      版本号 (如 "1.0.1")
        changelog:    更新日志文本
        include_frontend: 是否编译并包含前端
        include_patches:  是否包含 update_code/ 中的 Python 补丁
        output_path:  输出路径（默认: ./dist/OmniCore-{version}.zip）
    
    返回:
        生成的 ZIP 文件路径
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # ========== 编译前端 ==========
    frontend_zip_dir = None  # 待加入 ZIP 的前端目录
    if include_frontend:
        frontend_dir = os.path.join(base_dir, "frontend")
        if not os.path.exists(frontend_dir):
            print(f"[HotUpdate] ⚠️ 前端目录不存在: {frontend_dir}，跳过前端打包")
        else:
            # 确保依赖已安装
            if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
                print("[HotUpdate] 安装前端依赖...")
                subprocess.check_call(["npm", "install"], cwd=frontend_dir, shell=True)
            
            print(f"[HotUpdate] 🏗️ 编译 Vue 前端 v{version}...")
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=frontend_dir,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode != 0:
                print(f"[HotUpdate] ❌ 编译失败!\n{result.stderr}")
                sys.exit(1)
            
            dist_dir = os.path.join(frontend_dir, "dist")
            if not os.path.exists(dist_dir):
                print(f"[HotUpdate] ❌ 编译产物不存在: {dist_dir}")
                sys.exit(1)
            
            # 将 dist 复制到临时目录命名为 update_frontend
            frontend_zip_dir = tempfile.mkdtemp(prefix=TEMP_PREFIX)
            shutil.copytree(dist_dir, os.path.join(frontend_zip_dir, "update_frontend"))
            print(f"[HotUpdate] ✅ 前端编译完成")
    else:
        # 尝试从现有 update_frontend/ 获取
        existing = get_external_path("update_frontend")
        if os.path.exists(existing):
            frontend_zip_dir = tempfile.mkdtemp(prefix=TEMP_PREFIX)
            shutil.copytree(existing, os.path.join(frontend_zip_dir, "update_frontend"))
            print(f"[HotUpdate] 📂 使用现有 update_frontend/")
    
    # ========== 收集 Python 补丁 ==========
    patches_dir = None
    if include_patches:
        code_dir = os.path.join(base_dir, "update_code")
        if os.path.exists(code_dir) and os.listdir(code_dir):
            patch_files = [f for f in os.listdir(code_dir) if f.endswith(".py")]
            if patch_files:
                patches_dir = tempfile.mkdtemp(prefix=TEMP_PREFIX)
                code_dst = os.path.join(patches_dir, "update_code")
                os.makedirs(code_dst)
                for fname in patch_files:
                    shutil.copy2(os.path.join(code_dir, fname), os.path.join(code_dst, fname))
                print(f"[HotUpdate] 📦 包含 {len(patch_files)} 个 Python 补丁: {', '.join(patch_files)}")
    
    # ========== 生成 version.json ==========
    version_data = {
        "version": version,
        "build_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "changelog": changelog,
        "notes": f"OmniCore v{version} incremental update package",
    }
    
    # ========== 创建 ZIP ==========
    if output_path is None:
        os.makedirs(os.path.join(base_dir, "dist"), exist_ok=True)
        output_path = os.path.join(base_dir, "dist", f"OmniCore-{version}.zip")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"[HotUpdate] 📦 打包增量更新包: {output_path}")
    
    manifest_entries = {}
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 添加 version.json
        version_json = json.dumps(version_data, ensure_ascii=False, indent=2)
        zf.writestr("version.json", version_json)
        print(f"[HotUpdate]   ✅ version.json")
        
        # 添加前端文件
        if frontend_zip_dir:
            ui_dir = os.path.join(frontend_zip_dir, "update_frontend")
            for root, dirs, files in os.walk(ui_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.join("update_frontend", os.path.relpath(fpath, ui_dir))
                    zf.write(fpath, arcname)
                    manifest_entries[arcname] = _sha256_file(fpath)
            print(f"[HotUpdate]   ✅ 前端文件 ({len(manifest_entries)} 个)")
        
        # 添加 Python 补丁
        if patches_dir:
            code_dst = os.path.join(patches_dir, "update_code")
            for root, dirs, files in os.walk(code_dst):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.join("update_code", os.path.relpath(fpath, code_dst))
                    zf.write(fpath, arcname)
                    manifest_entries[arcname] = _sha256_file(fpath)
            print(f"[HotUpdate]   ✅ Python 补丁 ({os.listdir(code_dst)})")
        
        # 生成 manifest.json
        manifest = {
            "package_version": version,
            "created_at": datetime.now().isoformat(),
            "files": manifest_entries,
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"[HotUpdate]   ✅ manifest.json ({len(manifest_entries)} 项)")
    
    # ========== 清理临时目录 ==========
    if frontend_zip_dir:
        shutil.rmtree(frontend_zip_dir, ignore_errors=True)
    if patches_dir:
        shutil.rmtree(patches_dir, ignore_errors=True)
    
    # ========== 同时部署到 update_frontend ==========
    if include_frontend and frontend_zip_dir is None:
        pass  # 前端已编译但未单独存在，不需要额外部署
    elif include_frontend:
        dist_dir = os.path.join(frontend_dir, "dist")
        if os.path.exists(dist_dir):
            target_dir = get_external_path("update_frontend")
            print(f"[HotUpdate] 🚀 原子性部署前端到: {target_dir}")
            _atomic_copy(dist_dir, target_dir)
    
    file_size = os.path.getsize(output_path)
    print(f"[HotUpdate] ✅ 增量更新包创建成功!")
    print(f"[HotUpdate]    路径: {output_path}")
    print(f"[HotUpdate]    大小: {file_size / 1024:.1f} KB")
    print(f"[HotUpdate]    版本: v{version}")
    print(f"[HotUpdate]    内容: {'前端' if include_frontend else ''}{' + ' if include_frontend and include_patches else ''}{'Python补丁' if include_patches else ''}")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniCore 热更新工具")
    parser.add_argument("--package", "-p", action="store_true",
                        help="生成增量更新包 (.zip)")
    parser.add_argument("--version", "-v", default="1.0.0",
                        help="版本号 (默认: 1.0.0)")
    parser.add_argument("--changelog", "-c", default="",
                        help="更新日志")
    parser.add_argument("--no-frontend", action="store_true",
                        help="不包含前端")
    parser.add_argument("--no-patches", action="store_true",
                        help="不包含 Python 补丁")
    parser.add_argument("--output", "-o", default=None,
                        help="输出路径 (默认: ./dist/OmniCore-{version}.zip)")
    
    args = parser.parse_args()
    
    if args.package:
        # 包模式：生成增量更新 ZIP
        create_update_package(
            version=args.version,
            changelog=args.changelog,
            include_frontend=not args.no_frontend,
            include_patches=not args.no_patches,
            output_path=args.output,
        )
    else:
        # 传统模式：仅编译前端并部署到 update_frontend
        print("热更新: 开始编译...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        frontend_dir = os.path.join(base_dir, "frontend")
        
        if not os.path.exists(frontend_dir):
            print(f"[HotUpdate] 错误: 找不到前端目录 {frontend_dir}")
            sys.exit(1)
        
        if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
            print("[HotUpdate] node_modules 不存在，正在安装依赖...")
            subprocess.check_call(["npm", "install"], cwd=frontend_dir, shell=True)
        
        print("编译 Vue 前端 (npm run build)...")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        
        if result.returncode != 0:
            print(f"[HotUpdate] 编译失败!\n{result.stderr}")
            sys.exit(1)
        
        dist_dir = os.path.join(frontend_dir, "dist")
        if not os.path.exists(dist_dir):
            print(f"[HotUpdate] 错误: 编译产物 {dist_dir} 不存在")
            sys.exit(1)
        
        target_dir = get_external_path("update_frontend")
        
        print(f"正在原子性部署到: {target_dir}")
        _atomic_copy(dist_dir, target_dir)
        
        print("热更新部署成功！")
        print("请在 OmniCore 软件中按 F5 或 Ctrl+R 刷新页面。")
        print(f"    部署路径: {target_dir}")


if __name__ == "__main__":
    main()
