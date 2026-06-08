# OmniCore 智核工作站

> 本地部署的全功能 AI 助手 — 一个可以装进口袋的 AI 工作站

OmniCore 是一款**开箱即用的本地 AI 桌面应用**，支持多种大语言模型的加载、对话、微调训练和 Agent 工具调用，所有数据完全留在本地，无需联网。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🖥️ **桌面应用** | PyQt6 原生 GUI + Vue 3 Web 前端，系统托盘常驻 |
| 🤖 **多模型支持** | GGUF 量化模型（llama.cpp）+ HuggingFace Transformers 模型 |
| 🔧 **Agent 模式** | ReAct 推理 + 工具调用（代码执行、文件操作、网页搜索） |
| 📚 **RAG 知识库** | 本地文档向量检索，支持 PDF/DOCX/TXT/代码文件 |
| 🎯 **微调训练** | LoRA 微调，支持 JSONL/JSON/TXT/PDF 数据集 |
| 🧠 **记忆系统** | 情景记忆 + 语义记忆 + 压缩记忆，三层架构 |
| 🔌 **插件系统** | 可扩展的插件架构，自定义工具和功能 |
| 🔄 **热更新** | 支持代码和前端的增量热更新，无需重启 |
| 🛡️ **隐私优先** | 所有数据本地存储，不上传任何信息到云端 |

## 📦 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 64位 | Windows 11 |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 10 GB 可用空间 | 50 GB+（模型文件较大） |
| Python | 3.10 | 3.10 |
| Node.js | 18+ | 20+（前端开发需要） |

## 🚀 快速开始

### 方式一：安装包（推荐）

1. 从 [Releases](../../releases) 下载最新安装包 `OmniCore_v1.5.0_Setup.exe`
2. 运行安装程序，按向导完成安装
3. 启动 OmniCore，首次运行会自动检测环境并安装依赖

### 方式二：开发模式

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/OmniCore.git
cd OmniCore

# 2. 创建 Python 虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 安装前端依赖
cd frontend
npm install
cd ..

# 5. 启动开发模式（后端 + 前端热重载）
dev.bat
```

启动后访问 `http://localhost:5173` 即可使用 Web 前端，或通过系统托盘图标打开桌面应用。

## 🏗️ 项目结构

```
OmniCore/
├── api/                    # FastAPI 后端服务
│   ├── app.py              # FastAPI 应用入口
│   ├── routes_chat.py      # 对话 API
│   ├── routes_agent.py     # Agent API
│   ├── routes_training.py  # 训练 API
│   ├── routes_rag.py       # RAG API
│   └── training/           # 训练子模块
├── agent/                  # Agent 引擎
│   ├── agent.py            # 主 Agent 逻辑
│   ├── agent_tools.py      # 工具注册与执行
│   ├── react_engine.py     # ReAct 推理引擎
│   └── multi_agent.py      # 多 Agent 协作
├── core/                   # 核心模块
│   ├── app_state.py        # 全局状态管理
│   ├── model_loader.py     # 模型加载器
│   ├── hardware.py         # 硬件检测
│   ├── security.py         # 安全模块
│   └── config.py           # 配置管理
├── model/                  # 模型相关
│   ├── trainer.py          # 训练器
│   ├── data_loader.py      # 数据加载
│   └── model_setup.py      # 模型配置
├── tools/                  # 工具集
│   ├── rag.py              # RAG 知识库
│   ├── file_parser.py      # 文件解析
│   └── code_executor.py    # 代码执行
├── frontend/               # Vue 3 Web 前端
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── build_scripts/          # 构建与打包
│   ├── build_client.py     # PyInstaller 打包
│   ├── build_installer.py  # Inno Setup 安装程序
│   ├── hot_update.py       # 热更新包生成
│   └── updater.py          # 运行时更新器
├── docs/                   # 文档
├── tests/                  # 测试
├── requirements.txt        # Python 依赖
├── dev.bat                 # 开发模式启动脚本
├── OmniCore.spec           # PyInstaller 配置（自动生成）
└── OmniCore_Installer.iss  # Inno Setup 配置（自动生成）
```

## 🔧 构建与打包

```bash
# 构建 PyInstaller 客户端
python build_scripts/build_client.py

# 生成安装程序
python build_scripts/build_installer.py

# 生成热更新包
python build_scripts/hot_update.py --package
```

详细构建文档见 [docs/BUILD.md](docs/BUILD.md)。

## 📖 使用说明

### 加载模型

1. 启动应用后，进入「模型管理」页面
2. 选择本地模型文件（GGUF 格式或 HuggingFace 目录）
3. 点击「加载模型」，等待模型初始化完成

### 对话

1. 在对话页面输入问题
2. 支持 Markdown 渲染、代码高亮
3. 可开启 Agent 模式获得工具调用能力

### 微调训练

1. 准备训练数据（JSONL 格式）
2. 上传到数据集管理页面
3. 配置训练参数（LoRA rank、学习率等）
4. 开始训练，实时查看 Loss 曲线

### RAG 知识库

1. 上传文档到知识库（PDF/DOCX/TXT）
2. 系统自动进行向量化索引
3. 对话时自动检索相关知识

## 📋 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.5.0 | 2026-06 | 最终版封存，稳定性优化 |
| v1.4.0 | 2026-05 | 插件系统 + CI/CD |
| v1.3.0 | 2026-04 | 视觉引擎 + 工作流 DAG 编排 |
| v1.2.0 | 2026-03 | 记忆系统 v2（情景+语义+压缩） |
| v1.1.0 | 2026-02 | Agent 模式 + RAG 知识库 |
| v1.0.0 | 2026-01 | 首次发布 |

完整变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) — 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) — 现代 Python Web 框架
- [Vue.js](https://vuejs.org/) — 渐进式 JavaScript 框架
- [Element Plus](https://element-plus.org/) — Vue 3 UI 组件库
- [LangChain](https://www.langchain.com/) — LLM 应用开发框架
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 推理引擎
