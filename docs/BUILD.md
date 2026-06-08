# OmniCore 构建指南

## 环境准备

### 必需环境

| 工具 | 版本 | 用途 |
|------|------|------|
| Python | 3.10.x | 后端运行时 |
| Node.js | 18+ / 20+ | 前端构建 |
| npm | 9+ | 前端包管理 |
| Git | 2.30+ | 版本控制 |

### 可选环境

| 工具 | 版本 | 用途 |
|------|------|------|
| Inno Setup 6 | 6.0+ | 生成 Windows 安装程序 |
| Tesseract OCR | 5.0+ | PDF 文字识别 |
| Visual Studio Build Tools | 2022 | torch.compile 优化（可选） |

## 开发模式

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/OmniCore.git
cd OmniCore

# 2. 创建 Python 虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 安装可选依赖（GPU 加速、OCR 等）
pip install -r requirements-optional.txt

# 5. 安装前端依赖
cd frontend
npm install
cd ..

# 6. 启动开发模式
dev.bat
```

`dev.bat` 会同时启动：
- 后端：`uvicorn` 在 `http://localhost:8000`
- 前端：`vite` 在 `http://localhost:5173`（带热重载）

## 构建安装包

### 第一步：PyInstaller 打包

```bash
python build_scripts/build_client.py
```

这会：
1. 动态生成 `OmniCore.spec` 配置
2. 调用 PyInstaller 打包为单目录应用
3. 输出到 `dist/OmniCore/`

可选参数：
- `--public` — 生成不含态极（ModelSelf）的公开版

### 第二步：生成安装程序

```bash
python build_scripts/build_installer.py
```

这会：
1. 自动检测 Inno Setup 6 编译器
2. 动态生成 `.iss` 配置（含 61 个模型选项）
3. 输出 `OmniCore_v1.5.0_Setup.exe`

如果未安装 Inno Setup，会降级生成 ZIP 绿色包。

### 第三步：生成热更新包（可选）

```bash
python build_scripts/hot_update.py --package
```

生成增量更新 ZIP，包含：
- 变更的 Python 模块
- 编译后的前端资源
- `version.json` + `manifest.json` + SHA256 校验

## CI/CD

项目使用 GitHub Actions 自动化：

- `.github/workflows/test.yml` — Windows 环境下的单元测试 + 集成测试
- `.github/workflows/ci.yml` — 代码质量检查（flake8 + pytest）

## 常见问题

### PyInstaller 打包时 OOM

PyInstaller 扫描 torch/transformers 等大包的二进制依赖时可能内存不足。`build_client.py` 已内置 monkey-patch 跳过这些包的二进制扫描。

### Windows 文件锁定错误

打包过程中如果遇到文件锁定错误，`build_client.py` 已 monkey-patch PyInstaller 的文件操作函数来处理此问题。

### torch.compile 找不到 cl.exe

Windows 上 `torch.compile()` 的 Inductor 后端需要 Visual Studio Build Tools 中的 `cl.exe`。如果未安装，推理引擎会自动跳过编译，对小模型（<1B）性能影响很小。
