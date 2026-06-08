"""
模型下载功能验证脚本
测试网络连通性、镜像可达性及实际下载流程
"""
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_downloader import (
    ModelDownloader,
    DownloadProgress,
    diagnose_network,
    _check_url_reachable,
)
from model.model_registry import MODEL_REGISTRY, get_model_download_info


def progress_callback(p: DownloadProgress):
    """打印下载进度"""
    if p.status == "downloading":
        bar_len = 30
        filled = int(bar_len * p.percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {p.percent:5.1f}%  "
            f"{p.downloaded_mb:6.1f}/{p.total_mb:.1f}MB  "
            f"{p.speed_mbps:.1f}MB/s  "
            f"ETA: {p.eta_seconds:.0f}s  "
            f"源: {p.current_mirror}",
            end="",
        )
    elif p.status == "verifying":
        print(f"\r  正在校验 SHA256...")
    elif p.status == "error":
        print(f"\n  ❌ 错误: {p.error_message}")
    elif p.status == "idle" and p.error_message:
        print(f"\r  ⏳ {p.error_message}", end="")


def test_network_diagnosis():
    """测试1：网络诊断"""
    print("=" * 60)
    print("  测试 1: 网络诊断")
    print("=" * 60)
    
    results = diagnose_network()
    
    print(f"\n  整体状态: {results['overall_status']}")
    print(f"  建议: {results['recommendation']}\n")
    
    for m in results["mirrors"]:
        icon = "✅" if m["status"] == "reachable" else "❌"
        print(f"  {icon} {m['name']}: {m['status']}")
    
    return results["overall_status"] != "all_down"


def test_small_file_url():
    """测试2：检查一个小文件的 HEAD 请求"""
    print("\n" + "=" * 60)
    print("  测试 2: 检查模型文件 URL 可达性")
    print("=" * 60)
    
    # 使用有实际文件的仓库进行 URL 测试（HuggingFaceTB 只有 LFS 指针）
    test_urls = [
        ("hf-mirror.com", "https://hf-mirror.com/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"),
        ("huggingface.co", "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"),
    ]
    
    all_ok = True
    for name, url in test_urls:
        ok, err = _check_url_reachable(url, timeout=15)
        icon = "✅" if ok else "❌"
        if ok:
            print(f"  {icon} {name} - URL 可达")
        else:
            print(f"  {icon} {name} - 不可达: {err[:100]}")
            all_ok = False
    
    return all_ok


def test_download_tiny_model():
    """测试3：实际下载一个超小模型（SmolLM2-360M，约 0.3GB）"""
    print("\n" + "=" * 60)
    print("  测试 3: 实际下载测试 (SmolLM2-360M ~0.3GB)")
    print("  注意：测试完成后会自动清理下载的文件")
    print("=" * 60)
    
    model_name = "SmolLM2-360M"
    quant = "Q4_K_M"
    # 直接使用有实际文件的 bartowski 仓库（HuggingFaceTB 仓库只有 LFS 指针文件）
    repo_id = "bartowski/SmolLM2-360M-Instruct-GGUF"
    filename = "SmolLM2-360M-Instruct-Q4_K_M.gguf"
    
    print(f"  模型: SmolLM2-360M-Instruct")
    print(f"  仓库: {repo_id}")
    print(f"  文件: {filename}")
    print(f"  预计大小: 约 258MB\n")
    
    # 创建下载器
    downloader = ModelDownloader(
        save_dir=os.path.join(os.path.dirname(__file__), "_test_downloads"),
        mirror=True,
        verify_ssl=True,
    )
    
    try:
        print("  开始下载...")
        file_path = downloader.download_file(
            repo_id=repo_id,
            filename=filename,
            model_name=model_name,
            progress_callback=progress_callback,
        )
        print(f"\n  ✅ 下载成功!")
        print(f"  文件路径: {file_path}")
        print(f"  文件大小: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
        
        # 清理测试文件
        print("\n  正在清理测试文件...")
        import shutil
        test_dir = os.path.dirname(file_path)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        print("  ✅ 清理完成")
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ 下载失败: {e}")
        
        # 清理可能的残留文件
        import shutil
        test_dir = downloader.save_dir
        try:
            safe_name = model_name.replace("/", "_").replace("\\", "_")
            model_dir = os.path.join(test_dir, safe_name)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        except Exception:
            pass
        
        return False


def test_registry_integrity():
    """测试4：注册表完整性检查"""
    print("\n" + "=" * 60)
    print("  测试 4: 注册表完整性检查")
    print("=" * 60)
    
    issues = []
    for entry in MODEL_REGISTRY:
        if not entry.hf_repo:
            issues.append(f"  ⚠️  {entry.name}: 缺少 hf_repo")
        if not entry.variants:
            issues.append(f"  ⚠️  {entry.name}: 没有量化变体")
        for v in entry.variants:
            if not v.hf_filename:
                issues.append(f"  ⚠️  {entry.name}/{v.quant}: 缺少文件名")
    
    total_models = len(MODEL_REGISTRY)
    total_variants = sum(len(e.variants) for e in MODEL_REGISTRY)
    
    print(f"  模型总数: {total_models}")
    print(f"  量化变体总数: {total_variants}")
    
    if issues:
        print(f"\n  发现 {len(issues)} 个问题:")
        for issue in issues[:10]:
            print(issue)
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
        return False
    else:
        print("  ✅ 注册表完整，无问题")
        return True


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "   OmniCore 模型下载功能验证".center(50) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # 测试1: 网络诊断
    results["network"] = test_network_diagnosis()
    
    # 测试4: 注册表完整性
    results["registry"] = test_registry_integrity()
    
    # 测试2: URL 可达性
    results["url_reachable"] = test_small_file_url()
    
    # 测试3: 实际下载（只有当网络可达时才进行）
    if results["network"] or results["url_reachable"]:
        user_input = input("\n是否进行实际下载测试？(约 0.3GB，测试后自动清理) [y/N]: ").strip().lower()
        if user_input == "y":
            results["download"] = test_download_tiny_model()
        else:
            print("  跳过下载测试")
            results["download"] = None
    else:
        print("\n  ⚠️  网络不可达，跳过下载测试")
        results["download"] = None
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("  验证结果汇总")
    print("=" * 60)
    
    status_map = {True: "✅ 通过", False: "❌ 失败", None: "⏭️ 跳过"}
    
    print(f"  网络诊断:     {status_map[results['network']]}")
    print(f"  注册表完整性: {status_map[results['registry']]}")
    print(f"  URL 可达性:   {status_map[results['url_reachable']]}")
    print(f"  实际下载:     {status_map.get(results.get('download'), '⏭️ 跳过')}")
    
    all_required = results["network"] and results["registry"] and results["url_reachable"]
    if all_required:
        print("\n  ✅ 基本功能验证通过！模型下载功能正常可用。")
    else:
        print("\n  ❌ 部分验证未通过，请检查网络连接或代理设置。")
        if not results["network"]:
            print("     - 网络不可达，请检查是否可以访问 huggingface.co 或 hf-mirror.com")
            print("     - 可尝试设置代理或 VPN")
    
    print()
    return 0 if all_required else 1


if __name__ == "__main__":
    sys.exit(main())