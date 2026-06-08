"""
taiji 目录整理脚本
==================

将 66 个平铺文件整理为 8 个子目录。
在原位置保留重导出 stub，确保所有现有 import 不断。

用法：python reorganize_taiji.py
"""
import os
import shutil
import sys

TAIJI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taiji")

# 目录结构定义
SUBDIRS = {
    "core": {
        "files": [
            "inference.py",
            "cuda_inference.py",
            "hybrid_engine.py",
            "quantization.py",
            "native_agent.py",
        ],
        "description": "核心推理引擎",
    },
    "life": {
        "files": [
            "life_scheduler.py",
            "feed_engine.py",
            "sleep_engine.py",
            "play_engine.py",
            "evolution_engine.py",
            "body.py",
        ],
        "description": "生命系统",
    },
    "train": {
        "files": [
            "trainer.py",
            "train_pipeline.py",
            "dpo_trainer.py",
            "multimodal_trainer.py",
            "train_tokenizer.py",
            "data_cleaner.py",
        ],
        "description": "训练系统",
    },
    "multimodal": {
        "files": [
            "multimodal_engine.py",
            "vision_encoder.py",
            "audio_encoder.py",
            "image_generator.py",
            "voice_generator.py",
            "video_engine.py",
            "screen_reader.py",
            "voice_interface.py",
            "taiji_multimodal.py",
        ],
        "description": "多模态",
    },
    "agent": {
        "files": [
            "reflector.py",
            "planner.py",
            "perception.py",
            "memory.py",
            "working_memory.py",
        ],
        "description": "Agent 能力",
    },
    "safety": {
        "files": [
            "safety.py",
            "security_guard.py",
            "constitutional_ai.py",
        ],
        "description": "安全系统",
    },
    "infra": {
        "files": [
            "events.py",
            "actions.py",
            "profiler.py",
            "embryo.py",
            "user_profile.py",
            "code_understander.py",
            "self_evaluator.py",
            "auto_upgrade.py",
            "taiji_tokenizer.py",
        ],
        "description": "基础设施",
    },
    "data": {
        "files": [
            "seed_data.py",
            "data_generator.py",
            "agent_data.py",
            "knowledge_distiller.py",
            "knowledge_to_intelligence.py",
            "multimodal_data_generator.py",
            "taiji_graduation_data.py",
            "taiji_graduation_data_v2.py",
            "taiji_knowledge_data.py",
            "taiji_ultimate_training_data.py",
        ],
        "description": "训练数据生成",
    },
    "tests": {
        "files": [
            "test_basic.py",
            "test_new_abilities.py",
            "test_soul.py",
            "test_tokenizer_real.py",
            "test_trainer.py",
        ],
        "description": "测试",
    },
}

# 不移动的文件（被太多 relative import，风险太高）
KEEP_FILES = {
    "__init__.py",
    "config.py",
    "layers.py",
    "architecture.py",
    "tokenizer.py",
    "loader.py",
    "README.md",
    "tokenizer_corpus.txt",
}


def main():
    print("=" * 50)
    print("taiji 目录整理")
    print("=" * 50)

    # Step 1: 创建子目录
    print("\n[1/4] 创建子目录...")
    for subdir, info in SUBDIRS.items():
        subdir_path = os.path.join(TAIJI_DIR, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        # 创建 __init__.py（重导出所有模块）
        init_path = os.path.join(subdir_path, "__init__.py")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(f'"""\ntaiji.{subdir} — {info["description"]}\n"""\n')

        print(f"  创建 {subdir}/")

    # Step 2: 移动文件并创建 stub
    print("\n[2/4] 移动文件...")
    moved = 0
    stubs = 0

    for subdir, info in SUBDIRS.items():
        subdir_path = os.path.join(TAIJI_DIR, subdir)

        for filename in info["files"]:
            src = os.path.join(TAIJI_DIR, filename)
            dst = os.path.join(subdir_path, filename)

            if not os.path.exists(src):
                print(f"  跳过 {filename}（不存在）")
                continue

            # 移动文件
            shutil.move(src, dst)
            moved += 1

            # 在原位置创建 stub
            module_name = filename[:-3]  # 去掉 .py
            stub_content = f'"""向后兼容 — 实际模块已移至 taiji.{subdir}.{module_name}"""\nfrom taiji.{subdir}.{module_name} import *\n'
            with open(src, "w", encoding="utf-8") as f:
                f.write(stub_content)
            stubs += 1

        print(f"  {subdir}/: 移动了 {len(info['files'])} 个文件")

    # Step 3: 更新子目录内的 import
    print("\n[3/4] 更新子目录内 import...")
    import_fixes = 0

    for subdir, info in SUBDIRS.items():
        subdir_path = os.path.join(TAIJI_DIR, subdir)

        for filename in info["files"]:
            filepath = os.path.join(subdir_path, filename)
            if not os.path.exists(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original = content

            # 修复 relative import：from .xxx → from taiji.xxx（如果 xxx 已被移动）
            # 这些文件现在在子目录里，relative import 会找不到
            for other_subdir, other_info in SUBDIRS.items():
                for other_file in other_info["files"]:
                    other_module = other_file[:-3]
                    # from .xxx import → from taiji.xxx_subdir.xxx import
                    old_import = f"from .{other_module} import"
                    new_import = f"from taiji.{other_subdir}.{other_module} import"
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        import_fixes += 1

            # 修复对保留在根目录的模块的 relative import
            for keep_file in KEEP_FILES:
                if keep_file.endswith(".py") and keep_file != "__init__.py":
                    keep_module = keep_file[:-3]
                    # from .xxx import（保留在根目录的模块）→ 不需要改，因为 __init__.py 会重导出
                    # 但实际上 relative import 在子目录里找不到根目录的模块
                    # 需要改为 from taiji.xxx import
                    old_import = f"from .{keep_module} import"
                    new_import = f"from taiji.{keep_module} import"
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        import_fixes += 1

            if content != original:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

    print(f"  修复了 {import_fixes} 处 import")

    # Step 4: 更新 __init__.py 的 import 路径
    print("\n[4/4] 更新 __init__.py...")
    init_path = os.path.join(TAIJI_DIR, "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 更新 __init__.py 中的 import 路径
    # from .xxx import → from taiji.subdir.xxx import（对于已移动的模块）
    import_mapping = {}
    for subdir, info in SUBDIRS.items():
        for filename in info["files"]:
            module_name = filename[:-3]
            import_mapping[module_name] = subdir

    for module_name, subdir in import_mapping.items():
        # 处理 from .xxx import yyy
        old_pattern = f"from .{module_name} import"
        new_pattern = f"from taiji.{subdir}.{module_name} import"
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)

        # 处理 from taiji.xxx import（lazy imports in methods）
        old_pattern = f"from taiji.{module_name} import"
        new_pattern = f"from taiji.{subdir}.{module_name} import"
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)

    if content != original:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  __init__.py 已更新")
    else:
        print("  __init__.py 无需更新")

    # 汇总
    print("\n" + "=" * 50)
    print("整理完成！")
    print(f"  移动文件: {moved}")
    print(f"  创建 stub: {stubs}")
    print(f"  修复 import: {import_fixes}")
    print(f"  子目录: {len(SUBDIRS)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
