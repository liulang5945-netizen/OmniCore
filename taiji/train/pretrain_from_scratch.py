"""
态极 350M 从头预训练脚本
========================
用大规模数据从头训练态极模型。

用法:
  python taiji/train/pretrain_from_scratch.py

训练会持续数天（CPU），支持中断恢复。
"""
import os
import sys
import json
import time
import logging
import torch
import torch.nn.functional as F

# 确保项目根目录在 Python 路径中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Pretrain")

# ======================== 配置 ========================

CONFIG = {
    "model_size": "350m",
    "data_file": "taiji/training_data/pretrain_all.jsonl",
    "output_dir": "taiji/evolution_data/upgraded_models/pretrained",
    "max_seq_len": 512,
    "batch_size": 2,                     # 小 batch 减少单步时间
    "gradient_accumulation_steps": 16,   # 有效 batch = 2 * 16 = 32
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "max_steps": 100000,
    "save_every": 100,                     # 每 100 步保存，避免丢失
    "log_every": 10,                     # 每 10 步输出日志
    "eval_every": 500,
    "resume_from": None,

    # ======================== 早停配置 ========================
    "target_loss": 2.0,                    # loss 低于此值时提前结束训练
    "early_stopping_patience": 5,          # loss 连续 N 次检查无改善则停止（每次检查间隔 = save_every）
    "early_stopping_min_delta": 0.005,     # loss 改善幅度小于此值视为无改善
}


# ======================== 数据加载 ========================

class PretrainDataset(torch.utils.data.Dataset):
    """流式加载 JSONL 预训练数据"""

    def __init__(self, data_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"加载数据: {data_file}")
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 提取文本：对话格式取最后一条 assistant，否则取 text
                    if "messages" in item:
                        texts = [m["content"] for m in item["messages"] if m.get("role") != "system"]
                        text = " ".join(texts)
                    elif "text" in item:
                        text = item["text"]
                    else:
                        continue
                    if len(text) < 50:
                        continue
                    self.samples.append(text)
                except json.JSONDecodeError:
                    continue

        logger.info(f"加载完成: {len(self.samples):,} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text)
        # 截断到 max_len
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        # 自回归：input = ids[:-1], target = ids[1:]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    """动态填充到批次内最长序列"""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_labels = []
    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


# ======================== 训练主循环 ========================

def main():
    from taiji.config import ModelConfig
    from taiji.architecture import ModelSelf
    from taiji.tokenizer import ModelSelfTokenizer
    from taiji.loader import save_model

    cfg = CONFIG

    # 创建模型
    logger.info(f"创建模型: {cfg['model_size']}")
    model_config = ModelConfig.size_350m()
    model = ModelSelf(model_config)

    # 加载 tokenizer
    sp_path = os.path.join("taiji", "tokenizer", "sentencepiece.model")
    if os.path.exists(sp_path):
        tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
    else:
        tokenizer = ModelSelfTokenizer()

    device = torch.device("cpu")
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量: {param_count:,} ({param_count/1e6:.1f}M)")

    # 加载数据
    dataset = PretrainDataset(cfg["data_file"], tokenizer, max_len=cfg["max_seq_len"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    # 学习率调度
    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
        return max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 恢复训练：自动查找最新 checkpoint
    global_step = 0
    best_loss = float("inf")
    resume_path = cfg["resume_from"]
    if not resume_path and os.path.exists(cfg["output_dir"]):
        # 自动找最新的 step_* 目录
        step_dirs = []
        for d in os.listdir(cfg["output_dir"]):
            if d.startswith("step_"):
                try:
                    step_num = int(d.split("_")[1])
                    step_dirs.append((step_num, os.path.join(cfg["output_dir"], d)))
                except ValueError:
                    pass
        if step_dirs:
            step_dirs.sort(reverse=True)
            resume_path = step_dirs[0][1]
            global_step = step_dirs[0][0]

    if resume_path and os.path.exists(os.path.join(resume_path, "model.pt")):
        ckpt = torch.load(os.path.join(resume_path, "model.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict({k: v for k, v in ckpt.items() if not k.startswith("_")}, strict=False)
        logger.info(f"从 checkpoint 恢复: {resume_path} (step {global_step})")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 中断时自动保存
    import signal
    _interrupted = [False]
    def _signal_handler(sig, frame):
        logger.info("收到中断信号，正在保存 checkpoint...")
        _interrupted[0] = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 训练
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    step_count = 0
    start_time = time.time()

    # 早停状态
    _patience_counter = 0  # 连续未改善次数
    _early_stopped = False

    logger.info(f"开始训练: max_steps={cfg['max_steps']}, batch={cfg['batch_size']}, "
                f"grad_accum={cfg['gradient_accumulation_steps']}, lr={cfg['learning_rate']}")
    logger.info(f"数据: {len(dataset):,} 条, 每 epoch 约 {len(dataloader)} 步")
    logger.info("")

    epoch = 0
    while global_step < cfg["max_steps"] and not _interrupted[0] and not _early_stopped:
        epoch += 1
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if global_step >= cfg["max_steps"] or _interrupted[0] or _early_stopped:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向
            output = model(input_ids, targets=labels)
            loss = output.loss / cfg["gradient_accumulation_steps"]

            # 反向
            loss.backward()
            total_loss += loss.item()
            step_count += 1

            # 梯度累积
            if step_count % cfg["gradient_accumulation_steps"] == 0:
                step_start = time.time()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_time = time.time() - step_start

                # 日志
                if global_step % cfg["log_every"] == 0:
                    avg_loss = total_loss / step_count * cfg["gradient_accumulation_steps"]
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {global_step:>6}/{cfg['max_steps']} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Step: {step_time:.1f}s | "
                        f"Elapsed: {elapsed/60:.1f}min"
                    )

                # 保存 checkpoint
                if global_step % cfg["save_every"] == 0:
                    avg_loss = total_loss / step_count * cfg["gradient_accumulation_steps"]
                    ckpt_dir = os.path.join(cfg["output_dir"], f"step_{global_step}")
                    save_model(model, tokenizer, ckpt_dir)
                    logger.info(f"Checkpoint 保存: {ckpt_dir}")

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_dir = os.path.join(cfg["output_dir"], "best")
                        save_model(model, tokenizer, best_dir)
                        logger.info(f"Best 模型更新: loss={best_loss:.4f}")

                    # 保存训练状态
                    state_path = os.path.join(cfg["output_dir"], "training_state.json")
                    with open(state_path, "w") as f:
                        json.dump({
                            "global_step": global_step,
                            "best_loss": best_loss,
                            "epoch": epoch,
                            "elapsed_hours": (time.time() - start_time) / 3600,
                            "patience_counter": _patience_counter,
                        }, f, indent=2)

                    # ===== 早停检查 =====
                    _target = cfg.get("target_loss")
                    _patience = cfg.get("early_stopping_patience", 5)
                    _min_delta = cfg.get("early_stopping_min_delta", 0.005)

                    # 1) 目标 loss 达成 → 立即停止
                    if _target is not None and avg_loss <= _target:
                        logger.info(f"🎯 目标 loss 达成！当前 {avg_loss:.4f} ≤ 目标 {_target}，提前结束训练")
                        _early_stopped = True
                        break

                    # 2) patience 检查：loss 是否有实质改善
                    if avg_loss < best_loss - _min_delta:
                        _patience_counter = 0  # 有改善，重置计数
                    else:
                        _patience_counter += 1
                        if _patience_counter >= _patience:
                            logger.info(
                                f"⚠️ 早停触发：连续 {_patience} 次检查 loss 无实质改善 "
                                f"(best={best_loss:.4f}, current={avg_loss:.4f}, delta={_min_delta})"
                            )
                            _early_stopped = True
                            break

    # 训练完成
    total_time = time.time() - start_time
    final_loss = total_loss / step_count * cfg['gradient_accumulation_steps'] if step_count > 0 else float('inf')
    logger.info("")
    logger.info("=" * 50)
    if _early_stopped:
        logger.info(f"训练提前结束（早停）！")
    else:
        logger.info(f"训练完成！")
    logger.info(f"  总步数: {global_step}")
    logger.info(f"  最终 loss: {final_loss:.4f}")
    logger.info(f"  最佳 loss: {best_loss:.4f}")
    logger.info(f"  总耗时: {total_time/3600:.1f} 小时")
    logger.info("=" * 50)

    # 保存最终模型
    final_dir = os.path.join(cfg["output_dir"], "final")
    save_model(model, tokenizer, final_dir)
    logger.info(f"最终模型保存: {final_dir}")

    if _interrupted[0]:
        logger.info("训练被中断，checkpoint 已保存，下次启动会自动恢复")


if __name__ == "__main__":
    main()
