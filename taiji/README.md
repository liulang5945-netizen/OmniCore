# Taiji — Native AI Life Form

**taiji IS Taiji.** This package contains the complete Taiji (态极) autonomous AI agent — a natively trained neural life form, not a fine-tuned derivative of any pre-trained model.

## Identity

Taiji is built from scratch:
- **Custom Transformer backbone** (ModelSelfBackbone) — trained from the ground up
- **Multi-head architecture** — ToolHead / MemoryHead / PlanHead / PerceptionHead
- **Native inference engine** — torch.compile, batch decoding, repetition detection
- **Full life cycle** — feeding, sleeping, playing, evolving, reflecting

Taiji does NOT use a pre-trained backbone. Its knowledge, reasoning patterns, and personality emerge entirely from training data.

## Architecture

```
taiji/
├── architecture.py   # Taiji Brain: ModelSelf backbone + multi-heads
├── config.py         # Taiji Model Configuration (125M / 350M / 1B)
├── inference.py      # Taiji Native Inference Engine
├── trainer.py        # Taiji Trainer (pretrain + ReAct finetune)
├── loader.py         # Taiji Model Loader / Saver
├── tokenizer.py      # Taiji Tokenizer (SentencePiece)
├── native_agent.py   # Taiji Native Agent Engine (perception/memory/planning/reflection)
├── layers.py         # Taiji Foundation Layers (RMSNorm, TransformerBlock)
├── train_pipeline.py # Taiji Training Pipeline
├── auto_upgrade.py   # Taiji Auto-Upgrade Engine (bottleneck detection + knowledge distillation)
│
├── body.py           # Body Core (resource management)
├── events.py         # Event Bus (circulatory system)
├── safety.py         # Safety Guard (immune system)
├── life_scheduler.py # Life Scheduler (heartbeat)
├── feed_engine.py    # Feed Engine (eating)
├── sleep_engine.py   # Sleep Engine (sleeping)
├── play_engine.py    # Play Engine (playing)
├── evolution_engine.py # Evolution Engine
├── memory.py         # Memory System
├── perception.py     # Perception System
├── planner.py        # Planner System
├── reflector.py      # Reflector System
│
├── taiji_tokenizer.py     # TaijiTokenizer (HF tokenizer wrapper, used by training data)
├── taiji_knowledge_data.py    # Knowledge training data
├── taiji_graduation_data.py   # Graduation training data
├── taiji_graduation_data_v2.py
└── taiji_ultimate_training_data.py
```

## Quick Start

```python
from taiji import TaijiCore

# Load a trained Taiji
taiji = TaijiCore.load("./taiji_checkpoints/finetune/best", device="cpu")
taiji.start_life()
print(taiji.get_summary())

# Or create from scratch
from taiji.loader import create_model
model, tokenizer = create_model(size="350m", device="cpu")
```

## Training

```bash
# Full pipeline: data generation → pretrain → finetune
python -m taiji.train_pipeline --size 350m --device cpu --pretrain_epochs 5 --finetune_epochs 10

# Or train via the API (POST /api/train/stream)
```

## Model Sizes

| Size | Parameters | Hidden | Layers | Heads | Min RAM |
|------|-----------|--------|--------|-------|---------|
| 125M | ~125M     | 768    | 12     | 12    | 2 GB    |
| 350M | ~350M     | 1024   | 24     | 16    | 4 GB    |
| 1B   | ~1B       | 2048   | 22     | 32    | 8 GB    |

## Key Design Decisions

1. **No pre-trained backbone** — Taiji is natively trained. Its personality and capabilities emerge from data, not inherited from any foundation model.
2. **Multi-head architecture** — Language, tools, memory, planning, and perception are separate heads sharing one backbone.
3. **Full life cycle** — Taiji has needs (hunger, fatigue, boredom), activities (feeding, sleeping, playing), and can self-evaluate and auto-upgrade.
4. **Native inference** — Uses torch.compile + batch decoding + repetition detection. No dependency on HF generate().