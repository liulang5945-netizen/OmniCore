"""
OmniCore 智核工作站 - 主入口
声明：本文档开源共享，遵循 MIT 协议

统一入口点：
  python main.py                    # 启动 API 服务（模型+前端）
  python main.py --no-ui            # 仅加载模型（命令行模式）
  python main.py --train            # 训练模式
"""
import argparse
import os
import sys

base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from core.config import TrainingConfig, get_config, save_config
import uvicorn


def main():
    """主入口"""
    # 第一层：主入口参数
    parser = argparse.ArgumentParser(description="OmniCore 智核工作站")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称或路径")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="缓存目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径")
    parser.add_argument("--no-ui", action="store_true",
                        help="仅加载模型，不启动 UI")
    parser.add_argument("--train", action="store_true",
                        help="启动训练模式")
    
    # 解析主入口参数并移除已解析的，避免传递给 get_config
    args, remaining = parser.parse_known_args()

    # 使用剩余参数（如果有）或 None 来获取训练配置
    if remaining:
        config = get_config(args=remaining)
    else:
        config = get_config(args=[])

    if args.model_name:
        config.model_name = args.model_name
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.checkpoint:
        config.resume_from_checkpoint = args.checkpoint

    print(f"🧠 OmniCore 智核工作站")
    print(f"   模型: {config.model_name}")
    print(f"   设备: {config.resolve_device()}")

    if args.no_ui:
        print("ℹ️ no-ui 模式：仅模型加载")
        return

    if args.train:
        print("🚀 训练模式启动...")
        from model.data_loader import InstructionDataset, create_dataloader
        from model.model_setup import download_and_load_model, setup_model_for_training, get_optimizer, get_scheduler
        from model.trainer import Trainer
        
        model, tokenizer = download_and_load_model(config)
        model = setup_model_for_training(model, config)
        optimizer = get_optimizer(model, config)
        
        dataset = InstructionDataset(config.train_file, tokenizer, max_length=config.max_length)
        dataloader = create_dataloader(dataset, batch_size=config.batch_size)
        device = config.resolve_device()
        model = model.to(device)
        
        total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
        scheduler = get_scheduler(optimizer, config, total_steps)
        
        trainer = Trainer(model, optimizer, scheduler, config, device)
        for fraction, desc, loss_history in trainer.train(dataloader):
            print(f"\r{desc}", end="", flush=True)
        print(f"\n✅ 训练完成！")
        return

    print("🚀 正在通过命令行启动后台 API 服务...")
    uvicorn.run("api.api_server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
