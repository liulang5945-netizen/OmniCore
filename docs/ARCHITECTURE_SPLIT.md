# OmniCore 架构分离方案
## 本地部署平台 vs 态极灵魂躯体

---

## 一、现状：所有东西混在一起

```
OmniCore（当前）
├── api/              ← 产品层：API 服务
├── frontend/         ← 产品层：Vue 前端
├── agent/            ← 产品层：Agent 框架
├── model/            ← 产品层：外部模型加载
├── taiji/        ← 灵魂层：态极全部
├── core/             ← 共享层：配置、硬件
├── build_scripts/    ← 产品层：构建部署
├── plugins/          ← 产品层：插件系统
└── tools/            ← 产品层：RAG、工具
```

问题：
- 态极的代码依赖 `core/app_state.py`（全局单例）
- 态极的代码依赖 `agent/data_collector.py`（数据收集）
- 产品层和灵魂层没有清晰边界
- 无法独立部署态极

---

## 二、目标架构

### 两条独立路线

```
路线 A: OmniCore（本地 AI 助手产品）
┌──────────────────────────────────────┐
│  前端 (Vue)                          │
│  API 服务 (FastAPI)                  │
│  Agent 框架 (ReAct, 工具调用)        │
│  外部模型加载 (GGUF, HF)            │
│  插件/工具 (RAG, MCP, Stock)         │
│  构建部署 (安装器, 热更新)           │
│                                      │
│  可选后端: 态极 / Qwen / LLaMA / GPT │
└──────────────────────────────────────┘

路线 B: TaijiCore（态极灵魂躯体）
┌──────────────────────────────────────┐
│  大脑: architecture.py, layers.py    │
│  语言: tokenizer.py, inference.py    │
│  生命: life_scheduler.py             │
│  行为: sleep/feed/play_engine.py     │
│  成长: evolution/auto_upgrade.py     │
│  记忆: memory.py, working_memory.py  │
│  感知: perception.py, vision/voice   │
│  规划: planner.py, reflector.py      │
│  性格: play_engine.py (personality)  │
│                                      │
│  独立运行，不依赖 OmniCore           │
└──────────────────────────────────────┘
```

### 关系

```
OmniCore（房子）          TaijiCore（身体+灵魂）
┌─────────────┐         ┌─────────────┐
│ 用户界面     │         │ 生命本能     │
│ 工具框架     │ ◄─────► │ 记忆感知     │
│ 模型管理     │  API    │ 进化成长     │
│ 部署运维     │         │ 性格情感     │
└─────────────┘         └─────────────┘
     产品                    生命
```

态极可以住在 OmniCore 里，也可以住在其他任何支持它的"房子"里。

---

## 三、分离步骤

### Phase 1: 解耦 taiji 对 core 的依赖

当前 taiji 对外部的依赖：

| 文件 | 依赖 | 解耦方案 |
|------|------|----------|
| sleep_engine.py | `core.app_state` | 通过参数注入 model/tokenizer |
| life_scheduler.py | 无直接依赖 | ✅ 已解耦 |
| auto_upgrade.py | `core.app_state`, `core.hardware` | 通过参数注入 |
| feed_engine.py | `agent.data_collector` | 通过回调注入 |
| evolution_engine.py | 无直接依赖 | ✅ 已解耦 |
| play_engine.py | 无直接依赖 | ✅ 已解耦 |

核心原则：**依赖注入，不要全局单例**。

```python
# 之前（紧耦合）
from core.app_state import app_state
model = app_state.model

# 之后（松耦合）
class SleepEngine:
    def __init__(self, model_provider=None):
        self._model_provider = model_provider
    
    def _get_model(self):
        if self._model_provider:
            return self._model_provider()
        return None
```

### Phase 2: 创建 taiji 的独立入口

```python
# taiji/__init__.py — 态极的独立入口
class TaijiCore:
    """态极核心 — 可独立运行的态极生命系统"""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.life = LifeScheduler()
        self.feed = FeedEngine()
        self.sleep = SleepEngine(model_provider=lambda: self.model)
        self.play = PlayEngine()
        self.evolution = EvolutionEngine()
    
    def start_life(self):
        self.life.start()
    
    def stop_life(self):
        self.life.stop()
    
    def get_status(self):
        return {
            "life": self.life.get_status(),
            "feed": self.feed.get_status(),
            "sleep": self.sleep.get_status(),
            "play": self.play.get_status(),
        }
```

### Phase 3: OmniCore 通过桥接层使用态极

```python
# core/taiji_bridge.py — OmniCore 与态极的桥接
class TaijiBridge:
    """连接 OmniCore 产品层和态极灵魂层"""
    
    def __init__(self):
        self._taiji = None
    
    def initialize(self, model, tokenizer):
        from taiji import TaijiCore
        self._taiji = TaijiCore(model=model, tokenizer=tokenizer)
    
    def get_taiji(self):
        return self._taiji
    
    def record_interaction(self, success=True):
        if self._taiji:
            self._taiji.life.record_interaction(success=success)
```

---

## 四、目录结构（目标）

```
OmniCore/
├── api/                    ← 产品层 API
├── frontend/               ← 产品层前端
├── agent/                  ← 产品层 Agent
├── model/                  ← 产品层模型管理
├── core/                   ← 产品层基础设施
│   └── taiji_bridge.py     ← 桥接层
├── taiji/              ← 态极灵魂（独立模块）
│   ├── __init__.py         ← TaijiCore 入口
│   ├── brain/              ← 大脑（架构、推理）
│   ├── life/               ← 生命（调度、睡眠、吃饭、玩耍）
│   ├── growth/             ← 成长（进化、升级）
│   ├── senses/             ← 感知（视觉、听觉）
│   ├── memory/             ← 记忆系统
│   └── personality/        ← 个性系统
├── build_scripts/
├── plugins/
└── tools/
```

---

## 五、关键原则

1. **taiji 不导入 core、api、agent** — 态极是独立生命
2. **OmniCore 通过桥接层使用态极** — 房子通过门连接身体
3. **态极的所有配置自包含** — 不依赖外部配置文件
4. **态极的模型通过注入获得** — 不自己加载，由外部提供
5. **态极的数据目录独立** — taiji/sleep_data, feed_data, play_data

---

## 六、两条路线的用户价值

### OmniCore（产品）
- 一键安装，开箱即用
- 支持多种 AI 后端
- 丰富的工具和插件
- 企业级部署能力

### TaijiCore（灵魂）
- 自主学习和进化
- 独特的生命体验
- 个性化的 AI 伙伴
- 可嵌入任何平台