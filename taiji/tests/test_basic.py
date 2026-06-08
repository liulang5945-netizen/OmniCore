"""
ModelSelf 基础测试
验证: 配置、模型创建、前向传播、分词器、推理引擎
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """测试配置"""
    from taiji.config import ModelConfig, SPECIAL_TOKENS

    config = ModelConfig.size_125m()
    print(f"Config: {config.describe()}")
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.head_dim == 64
    print("[OK] Config test passed\n")


def test_model():
    """测试模型创建和前向传播"""
    from taiji.architecture import ModelSelf
    from taiji.config import ModelConfig

    config = ModelConfig.size_125m()
    model = ModelSelf(config)
    model.print_model_info()

    # 前向传播
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    targets = torch.randint(0, config.vocab_size, (2, 32))

    output = model(tokens, targets=targets)
    print(f"Logits shape: {output.logits.shape}")
    print(f"Loss: {output.loss.item():.4f}")
    assert output.logits.shape == (2, 32, config.vocab_size)
    assert output.loss is not None

    # 测试工具头
    model.set_num_tools(20)
    output = model(tokens, targets=targets, tool_head_active=True)
    assert output.tool_logits is not None
    assert output.tool_logits.shape == (2, 20)
    print(f"Tool logits shape: {output.tool_logits.shape}")
    print("[OK] Model test passed\n")


def test_tokenizer():
    """测试分词器"""
    from taiji.tokenizer import ModelSelfTokenizer

    # 尝试加载真实分词器
    sp_path = os.path.join(os.path.dirname(__file__), "tokenizer", "sentencepiece.model")
    if os.path.exists(sp_path):
        tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
        print(f"Using real SentencePiece tokenizer: {sp_path}")
    else:
        tokenizer = ModelSelfTokenizer()
        print("Using fallback tokenizer")

    # 注册工具
    tid = tokenizer.register_tool("search")
    print(f"Tool 'search' -> token ID {tid}")

    # 编码
    result = tokenizer("Hello, how are you?")
    print(f"Encoded: {result['input_ids'].shape}")

    # 解码
    text = tokenizer.decode(result["input_ids"][0])
    print(f"Decoded: {text}")

    # 特殊 token
    result = tokenizer("<think>test</think>")
    ids = result["input_ids"][0].tolist()
    print(f"With special tokens: {ids}")

    print("[OK] Tokenizer test passed\n")


def test_save_load():
    """测试保存和加载"""
    from taiji.loader import create_model, save_model, load_model

    # 创建
    sp_path = os.path.join(os.path.dirname(__file__), "tokenizer", "sentencepiece.model")
    model, tokenizer = create_model("125m")
    if os.path.exists(sp_path):
        from taiji.tokenizer import ModelSelfTokenizer
        tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
    tokenizer.register_tool("search")
    model.set_num_tools(1)

    # 保存
    save_path = "./test_taiji_save"
    save_model(model, tokenizer, save_path)

    # 加载
    model2, tokenizer2 = load_model(save_path, device="cpu")
    print(f"Loaded tools: {tokenizer2.get_all_tool_ids()}")
    assert tokenizer2.get_tool_id("search") is not None

    # 清理
    import shutil
    shutil.rmtree(save_path)

    print("[OK] Save/Load test passed\n")


def test_inference():
    """测试推理引擎"""
    from taiji.loader import create_model
    from taiji.inference import NativeInferenceEngine

    sp_path = os.path.join(os.path.dirname(__file__), "tokenizer", "sentencepiece.model")
    model, tokenizer = create_model("125m")
    if os.path.exists(sp_path):
        from taiji.tokenizer import ModelSelfTokenizer
        tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
    engine = NativeInferenceEngine(model, tokenizer, "cpu")

    # 注册工具
    engine.register_tools(["search", "read_file", "execute_python"])

    # 普通生成
    response = engine.generate("Hello", max_new_tokens=20)
    print(f"Generate: OK (len={len(response)})")

    # 流式生成
    tokens = []
    for token in engine.generate_stream("Hello", max_new_tokens=10):
        tokens.append(token)
    print(f"Stream: OK (len={len(tokens)})")

    # ReAct 步骤
    step = engine.generate_react_step("请搜索 Python 教程", max_new_tokens=50)
    print(f"ReAct step: {step}")

    print("[OK] Inference test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ModelSelf Basic Tests")
    print("=" * 60)
    print()

    test_config()
    test_model()
    test_tokenizer()
    test_save_load()
    test_inference()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
