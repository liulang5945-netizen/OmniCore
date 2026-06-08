"""
ModelSelf 训练器测试
验证: 数据集构建、训练循环、检查点保存
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dataset():
    """测试数据集构建"""
    from taiji.tokenizer import ModelSelfTokenizer
    from taiji.trainer import build_dataset

    tokenizer = ModelSelfTokenizer()

    # 注册一些工具
    for name in ["read_local_file", "write_file", "execute_python", "search",
                  "list_directory", "edit_file", "create_project", "analyze_code",
                  "install_dependency", "learn_knowledge", "query_knowledge",
                  "run_command", "read_webpage"]:
        tokenizer.register_tool(name)

    dataset = build_dataset(tokenizer, max_length=256)
    print(f"Dataset size: {len(dataset)} samples")

    # 检查一个样本
    sample = dataset[0]
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Tool target: {sample['tool_target']}")

    print("[OK] Dataset test passed\n")
    return dataset, tokenizer


def test_training():
    """测试训练循环"""
    from taiji.architecture import ModelSelf
    from taiji.config import ModelConfig
    from taiji.tokenizer import ModelSelfTokenizer
    from taiji.trainer import ModelSelfTrainer, build_dataset

    # 创建小模型
    config = ModelConfig.size_125m()
    config.num_hidden_layers = 4  # 更小的模型用于测试
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    model = ModelSelf(config)
    tokenizer = ModelSelfTokenizer()

    for name in ["read_local_file", "write_file", "execute_python", "search",
                  "list_directory", "learn_knowledge", "query_knowledge"]:
        tokenizer.register_tool(name)
    model.set_num_tools(len(tokenizer._tool_name_to_id))

    # 构建数据集
    dataset = build_dataset(tokenizer, max_length=128)

    # 创建训练器
    trainer = ModelSelfTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=1e-3,
        warmup_steps=5,
        gradient_accumulation_steps=1,
    )

    # 训练 (只跑 1 epoch，快速验证)
    save_dir = "./test_taiji_checkpoints"
    loss_history = []

    for fraction, desc, losses, metrics in trainer.train(
        dataset=dataset,
        num_epochs=1,
        batch_size=2,
        save_dir=save_dir,
        save_steps=100,  # 不保存中间检查点
        log_steps=2,
        device="cpu",
    ):
        loss_history = losses
        print(f"  {desc}")

    print(f"\nFinal loss: {loss_history[-1]:.4f}" if loss_history else "No steps completed")

    # 清理
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print("[OK] Training test passed\n")


def test_full_pipeline():
    """完整流程测试: 创建 → 训练 → 保存 → 加载 → 推理"""
    from taiji import create_model, save_model, load_model, NativeInferenceEngine
    from taiji.trainer import ModelSelfTrainer, build_dataset
    from taiji.config import ModelConfig

    # 1. 创建小模型
    print("1. Creating model...")
    config = ModelConfig.size_125m()
    config.num_hidden_layers = 4
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    from taiji.architecture import ModelSelf
    from taiji.tokenizer import ModelSelfTokenizer

    model = ModelSelf(config)
    tokenizer = ModelSelfTokenizer()

    for name in ["read_local_file", "write_file", "execute_python", "search",
                  "list_directory", "learn_knowledge", "query_knowledge"]:
        tokenizer.register_tool(name)
    model.set_num_tools(len(tokenizer._tool_name_to_id))

    # 2. 训练
    print("2. Training...")
    dataset = build_dataset(tokenizer, max_length=128)
    trainer = ModelSelfTrainer(model, tokenizer, learning_rate=1e-3, warmup_steps=5)

    for _, desc, _, _ in trainer.train(dataset, num_epochs=1, batch_size=2,
                                       save_dir="./test_pipeline_ckpt",
                                       save_steps=100, log_steps=5, device="cpu"):
        pass

    # 3. 保存
    print("3. Saving...")
    save_path = "./test_pipeline_model"
    save_model(model, tokenizer, save_path)

    # 4. 加载
    print("4. Loading...")
    model2, tokenizer2 = load_model(save_path, device="cpu")

    # 5. 推理
    print("5. Inference...")
    engine = NativeInferenceEngine(model2, tokenizer2, "cpu")
    engine.register_tools(["search", "read_file", "execute_python"])

    step = engine.generate_react_step("请搜索 Python 教程", max_new_tokens=50)
    print(f"ReAct step: {step}")

    # 清理
    import shutil
    for d in [save_path, "./test_pipeline_ckpt"]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print("[OK] Full pipeline test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ModelSelf Trainer Tests")
    print("=" * 60)
    print()

    test_dataset()
    test_training()
    test_full_pipeline()

    print("=" * 60)
    print("All trainer tests passed!")
    print("=" * 60)
