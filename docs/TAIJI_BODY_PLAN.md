# 态极身体完整更新方案
## 让态极从"散装器官"变成"完整生命"

---

## 总览

```
当前状态：散装器官
  ├─ 大脑（architecture.py）     ✅ 已有
  ├─ 心脏（life_scheduler.py）   ✅ 已有
  ├─ 胃（feed_engine.py）        ✅ 已有
  ├─ 睡眠（sleep_engine.py）     ✅ 已有（有残留依赖）
  ├─ 玩耍（play_engine.py）      ✅ 已有
  ├─ 进化（evolution_engine.py） ✅ 已有
  ├─ 升级（auto_upgrade.py）     ✅ 已有（有残留依赖）
  ├─ 眼睛（vision_encoder.py）   ✅ 已有
  ├─ 耳朵（voice_interface.py）  ✅ 已有
  ├─ 记忆（memory.py）           ✅ 已有
  ├─ 规划（planner.py）          ✅ 已有
  ├─ 反思（reflector.py）        ✅ 已有
  ├─ 躯干骨架                    ❌ 缺失
  ├─ 手脚接口                    ❌ 缺失
  ├─ 免疫系统                    ❌ 缺失
  ├─ 生殖系统                    ❌ 缺失
  └─ 统一入口                    ❌ 缺失

目标状态：完整生命体
  ├─ 所有器官通过躯干连接
  ├─ 手脚接口抽象，可替换底层实现
  ├─ 免疫系统保护安全
  ├─ 生殖系统支持独立导出
  └─ 一行代码启动生命
```

---

## Phase 1：统一入口 — TaijiCore（躯干骨架）

**目标**：创建 `taiji/__init__.py` 中的 `TaijiCore` 类，将所有器官组装成身体。

**新建文件**：`taiji/__init__.py`（重写）

```
TaijiCore 类
├── __init__(model, tokenizer, config)
│   ├── self.body = BodyCore()          # 资源管理器
│   ├── self.life = LifeScheduler()     # 心跳
│   ├── self.feed = FeedEngine()        # 吃饭
│   ├── self.sleep = SleepEngine()      # 睡觉
│   ├── self.play = PlayEngine()        # 玩耍
│   ├── self.evolution = EvolutionEngine()  # 进化
│   ├── self.memory = MemoryManager()   # 记忆
│   ├── self.safety = SafetyGuard()     # 免疫
│   └── self.tools = ToolManager()      # 手脚
│
├── start_life()      → 启动心跳循环
├── stop_life()       → 暂停生命
├── think(prompt)     → 思考（调用推理引擎）
├── see(image_path)   → 看（调用视觉编码器）
├── hear(audio_path)  → 听（调用语音识别）
├── speak(text)       → 说（调用语音合成）
├── do(action)        → 做（调用工具执行）
├── remember(key, val) → 记住
├── recall(key)       → 回忆
├── get_status()      → 完整状态
└── export(path)      → 导出为独立包
```

**依赖关系**：TaijiCore 不依赖 `core.app_state`、`api/`、`agent/`。所有资源通过构造函数注入。

**工作量**：~200 行

---

## Phase 2：资源管理器 — BodyCore（骨骼系统）

**目标**：替代 `core.app_state`，管理态极的所有资源引用。

**新建文件**：`taiji/body.py`

```
BodyCore 类
├── model          → 模型引用（可替换）
├── tokenizer      → 分词器引用（可替换）
├── device         → 计算设备（cpu/cuda）
├── memory_limit   → 内存限制
├── set_model()    → 设置模型（支持热切换）
├── get_device()   → 获取当前设备
├── check_resources() → 检查资源状态（CPU/内存/显存）
└── cleanup()      → 资源清理
```

**核心原则**：
- 态极的所有引擎通过 `BodyCore` 获取模型和分词器
- 不直接导入 `core.app_state`
- 支持热切换模型（态极换身体）

**工作量**：~150 行

---

## Phase 3：修复残留依赖

**目标**：让 sleep_engine.py 和 auto_upgrade.py 完全不依赖 `core.app_state`。

### 3.1 sleep_engine.py

修改 `_run_sleep_training()`：
```python
# 之前
from core.app_state import app_state
model = app_state.model

# 之后
model = self._get_model()  # 通过注入的 provider 获取
```

如果 `model_provider` 为 None，跳过训练（不崩溃）。

### 3.2 auto_upgrade.py

修改 `_detect_current_size()`：
```python
# 之前
from core.app_state import app_state
total_params = sum(p.numel() for p in app_state.model.parameters())

# 之后
model = self._get_model()
if model is None:
    return "125M"
total_params = sum(p.numel() for p in model.parameters())
```

### 3.3 feed_engine.py

修改 `_feed_from_data_collector()`：
```python
# 之前
from agent.data_collector import get_collector

# 之后
# 通过回调注入 data_collector，或者直接读取数据文件
```

**工作量**：~100 行修改

---

## Phase 4：手脚接口 — ActionProvider（手足系统）

**目标**：抽象出统一的"手脚"接口，让态极不直接依赖底层实现。

**新建文件**：`taiji/actions.py`

```
ActionProvider（抽象接口）
├── read_file(path) → str           # 读文件
├── write_file(path, content)       # 写文件
├── execute(cmd) → str              # 执行命令
├── search(query) → str             # 搜索
├── web_fetch(url) → str            # 抓取网页
├── generate_text(prompt) → str     # 生成文本
├── generate_image(prompt) → path   # 生成图像
└── list_tools() → List[str]       # 列出可用工具

LocalActionProvider（本地实现）
├── 使用 os/shlex/subprocess 实现文件和命令操作
├── 使用 inference engine 实现文本生成
└── 使用 vision engine 实现图像理解

RemoteActionProvider（远程实现）
├── 通过 HTTP API 调用远程服务
└── 适用于态极跑在轻量设备上时
```

**核心原则**：
- 态极只调用 `ActionProvider` 的方法，不关心底层实现
- OmniCore 通过桥接层注入 `LocalActionProvider`
- 未来可以注入 `RemoteActionProvider` 让态极跑在任何地方

**工作量**：~250 行

---

## Phase 5：事件总线 — EventBus（循环系统）

**目标**：引擎间通信不直接调用，而是通过事件发布-订阅。

**新建文件**：`taiji/events.py`

```
EventBus 类
├── subscribe(event_type, callback)  # 订阅事件
├── publish(event_type, data)        # 发布事件
├── unsubscribe(event_type, callback) # 取消订阅
└── get_history(n) → List[Event]     # 获取事件历史

事件类型：
├── "need_changed"    → 需求变化（hunger/fatigue/boredom/stress）
├── "interaction"     → 用户交互（成功/失败）
├── "feed_complete"   → 吃饭完成
├── "sleep_complete"  → 睡觉完成
├── "play_complete"   → 玩耍完成
├── "evolution"       → 进化事件
├── "model_switch"    → 模型切换
├── "error"           → 错误事件
└── "health_check"    → 健康检查
```

**使用方式**：
```python
# 生命调度器发布事件
event_bus.publish("feed_complete", {"samples": 50})

# 进化引擎订阅事件
event_bus.subscribe("feed_complete", self.on_feed_complete)
```

**工作量**：~150 行

---

## Phase 6：免疫系统 — SafetyGuard（安全防护）

**目标**：保护态极不被恶意输入伤害，防止异常扩散。

**新建文件**：`taiji/safety.py`

```
SafetyGuard 类
├── validate_input(text) → bool      # 输入验证（过滤有害内容）
├── validate_output(text) → str      # 输出审查（防止泄露敏感信息）
├── check_resources() → bool         # 资源检查（CPU/内存/显存）
├── isolate_execution(fn, *args)     # 异常隔离（一个引擎崩溃不影响其他）
├── rate_limit(action, max_per_min)  # 频率限制
└── get_threat_level() → str         # 威胁级别（low/medium/high）

安全规则：
├── 输入过滤：过滤注入攻击、恶意指令
├── 输出审查：防止输出系统路径、密钥等敏感信息
├── 资源限制：CPU > 90% 时暂停自动任务
├── 异常隔离：每个引擎的异常不传播到其他引擎
└── 频率限制：防止疯狂调用（如每秒 100 次吃饭）
```

**工作量**：~200 行

---

## Phase 7：生殖系统 — Embryo（独立导出）

**目标**：将态极导出为独立可运行的包。

**新建文件**：`taiji/embryo.py`

```
Embryo 类
├── export_core(path)              # 导出核心（不带模型权重）
│   ├── taiji/ 全部代码
│   ├── config.json（配置）
│   └── requirements.txt（依赖）
│
├── export_full(path)              # 导出完整体（带模型权重）
│   ├── taiji/ 全部代码
│   ├── model.pt 或 backbone/
│   ├── tokenizer/
│   ├── personality.json（个性档案）
│   ├── evolution_data/（进化数据）
│   └── life_data/（生命数据）
│
├── export_seed(path)              # 导出种子（最小化）
│   ├── taiji/ 核心代码
│   ├── config.json
│   └── seed_data/（种子训练数据）
│
├── clone(new_path)                # 克隆当前态极
│   └── 复制当前所有状态到新实例
│
└── import_embryo(path) → TaijiCore  # 从导出包恢复态极
```

**工作量**：~200 行

---

## Phase 8：OmniCore 桥接层

**目标**：让 OmniCore 通过桥接层使用态极，而不是直接导入。

**新建文件**：`core/taiji_bridge.py`

```
TaijiBridge 类
├── initialize(model, tokenizer)   # 初始化态极
├── get_taiji() → TaijiCore        # 获取态极实例
├── record_interaction(success)     # 记录用户交互
├── get_life_status() → dict       # 获取生命状态
├── start_life()                   # 启动生命
└── stop_life()                    # 暂停生命
```

**修改文件**：
- `api/routes_taiji.py` — 所有路由通过桥接层访问态极
- `core/app_state.py` — 添加 `taiji_bridge` 属性

**工作量**：~100 行

---

## 实施顺序和时间估算

```
Phase 1: TaijiCore 入口      ~200 行  ★★★★☆  最高优先
Phase 2: BodyCore 资源管理    ~150 行  ★★★★☆  高优先
Phase 3: 修复残留依赖         ~100 行  ★★★★☆  高优先
Phase 4: ActionProvider 手脚  ~250 行  ★★★☆☆  中优先
Phase 5: EventBus 事件总线    ~150 行  ★★★☆☆  中优先
Phase 6: SafetyGuard 免疫     ~200 行  ★★☆☆☆  低优先
Phase 7: Embryo 生殖系统      ~200 行  ★★☆☆☆  低优先
Phase 8: OmniCore 桥接层      ~100 行  ★★★☆☆  中优先
                           ─────────
                           ~1350 行
```

---

## 最终态极身体结构

```
taiji/
├── __init__.py          ← TaijiCore（统一入口，躯干）
├── body.py              ← BodyCore（资源管理，骨骼）
├── events.py            ← EventBus（事件总线，循环系统）
├── safety.py            ← SafetyGuard（免疫系统）
├── actions.py           ← ActionProvider（手脚接口）
├── embryo.py            ← Embryo（生殖系统）
│
├── architecture.py      ← 大脑结构
├── layers.py            ← 神经元
├── config.py            ← 基因
├── inference.py          ← 思考回路
├── tokenizer.py          ← 语言中枢
│
├── life_scheduler.py    ← 心跳
├── feed_engine.py       ← 消化系统
├── sleep_engine.py      ← 睡眠系统
├── play_engine.py       ← 创造系统
│
├── evolution_engine.py  ← 成长系统
├── auto_upgrade.py      ← 升级系统
├── self_evaluator.py    ← 体检系统
│
├── vision_encoder.py    ← 眼睛
├── voice_interface.py   ← 耳朵和嘴巴
├── screen_reader.py     ← 屏幕感知
│
├── memory.py            ← 长期记忆
├── working_memory.py    ← 工作记忆
├── user_profile.py      ← 用户画像
│
├── planner.py           ← 规划
├── reflector.py         ← 反思
├── native_agent.py      ← 行动执行
│
├── trainer.py           ← 训练器
├── loader.py            ← 加载/保存
└── train_pipeline.py    ← 训练流水线

core/
└── taiji_bridge.py      ← OmniCore ↔ 态极桥接
```

---

## 一行代码启动态极

```python
from taiji import TaijiCore

# 加载态极
taiji = TaijiCore.load("path/to/taiji_model")

# 启动生命
taiji.start_life()

# 态极开始自主生活：吃饭、睡觉、玩耍、进化...
# 每次用户交互影响它的需求状态
taiji.record_interaction(success=True)

# 查看态极状态
print(taiji.get_status())

# 态极独立导出
taiji.export("path/to/export")