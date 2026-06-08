"""
Agent 文件系统与代码开发工具集
包含：文件操作、项目脚手架、代码分析、依赖管理
"""
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from core.utils import get_external_path

logger = logging.getLogger("AgentTools")

_WORKSPACE_DIR = None

def _get_workspace() -> str:
    global _WORKSPACE_DIR
    if _WORKSPACE_DIR is None:
        _WORKSPACE_DIR = get_external_path("agent_workspace")
        os.makedirs(_WORKSPACE_DIR, exist_ok=True)
    return _WORKSPACE_DIR


def _resolve_safe_path(target: str) -> Optional[str]:
    """安全解析路径，防止目录穿越"""
    ws = os.path.abspath(_get_workspace())
    result = os.path.abspath(os.path.join(ws, target))
    if result == ws or result.startswith(ws + os.sep):
        return result
    return None


# ======================== 文件系统操作 ========================

def read_local_file(input_str: str) -> str:
    """通用本地文件提取（分页支持）"""
    # 安全检查：防止目录穿越
    file_path = input_str.split(",")[0].strip() if input_str else ""
    if file_path and not _resolve_safe_path(file_path):
        return "Blocked: 路径不在允许范围内"
    try:
        from tools.file_parser import parse_file_to_text
        input_str = input_str.strip()
        file_path = input_str
        page = 1

        if not os.path.exists(file_path):
            parts = input_str.rsplit(",", 1)
            if len(parts) == 2:
                try:
                    possible_page = int(parts[1].strip())
                    possible_path = parts[0].strip()
                    if os.path.exists(possible_path):
                        file_path = possible_path
                        page = possible_page
                except ValueError:
                    pass

        if not os.path.exists(file_path):
            ws_path = os.path.join(_get_workspace(), file_path)
            if os.path.exists(ws_path):
                file_path = ws_path

        if not os.path.exists(file_path):
            return f"错误: 找不到文件 '{file_path}'。当前工作台目录: {_get_workspace()}"

        text = parse_file_to_text(file_path)
        if not text:
            return "文件为空"

        page_size = 4000
        total_pages = max(1, (len(text) + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        content = text[start_idx:end_idx]

        return f"--- 第 {page}/{total_pages} 页 ---\n{content}\n--- (若需阅读下一页，请调用工具并传入 `{file_path}, {page + 1}`) ---"
    except Exception as e:
        return f"读取文件失败: {e}"


def write_file(input_str: str) -> str:
    """将内容写入文件（在工作台内）。输入格式: 文件路径 | 文件内容"""
    try:
        parts = input_str.split("|", 1)
        if len(parts) < 2:
            return "错误: 输入格式必须为 `文件路径 | 文件内容`"
        file_path = parts[0].strip()
        content = parts[1]
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 文件已写入: {rel_path} (共 {len(content)} 字符)"
    except Exception as e:
        return f"❌ 写入文件失败: {e}"


def edit_file(input_str: str) -> str:
    """编辑文件中的特定内容（精确替换）。输入格式: 文件路径 | 旧文本 | 新文本"""
    try:
        parts = input_str.split("|", 2)
        if len(parts) < 3:
            return "错误: 输入格式必须为 `文件路径 | 要搜索的旧文本 | 替换后的新文本`"
        file_path = parts[0].strip()
        old_text = parts[1]
        new_text = parts[2]
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        if not os.path.exists(resolved):
            return f"错误: 文件不存在 '{resolved}'"
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            stripped_old = old_text.strip()
            if stripped_old in content:
                old_text = stripped_old
            else:
                return f"错误: 在文件中未找到匹配的文本。请确认内容完全一致（包括缩进）。"
        new_content = content.replace(old_text, new_text, 1)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 文件已编辑: {rel_path} (替换了 1 处匹配)"
    except Exception as e:
        return f"❌ 编辑文件失败: {e}"


def delete_file(input_str: str) -> str:
    """删除工作台中的文件或空目录。"""
    try:
        file_path = input_str.strip()
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        if not os.path.exists(resolved):
            return f"错误: 路径不存在 '{resolved}'"
        if os.path.isfile(resolved):
            os.remove(resolved)
            rel_path = os.path.relpath(resolved, _get_workspace())
            return f"✅ 文件已删除: {rel_path}"
        elif os.path.isdir(resolved):
            try:
                os.rmdir(resolved)
                rel_path = os.path.relpath(resolved, _get_workspace())
                return f"✅ 空目录已删除: {rel_path}"
            except OSError:
                return f"错误: 目录非空，无法删除。如需删除整个目录请使用 `rmtree` 命令。"
        return f"错误: 未知路径类型 '{resolved}'"
    except Exception as e:
        return f"❌ 删除失败: {e}"


def list_directory(input_str: str = "") -> str:
    """列出工作台目录内容。"""
    try:
        target_dir = input_str.strip() if input_str.strip() else ""
        if target_dir:
            resolved = _resolve_safe_path(target_dir)
            if resolved is None:
                return f"❌ 路径不安全或超出工作台目录: {target_dir}"
        else:
            resolved = _get_workspace()
        if not os.path.exists(resolved):
            return f"错误: 目录不存在 '{resolved}'"
        if not os.path.isdir(resolved):
            return f"错误: '{resolved}' 不是目录"
        entries = os.listdir(resolved)
        if not entries:
            return f"📁 目录为空: {os.path.relpath(resolved, _get_workspace()) or '.'}"
        dirs = []
        files = []
        for entry in sorted(entries):
            entry_path = os.path.join(resolved, entry)
            if os.path.isdir(entry_path):
                dirs.append(f"  📁 {entry}/")
            else:
                try:
                    size = os.path.getsize(entry_path)
                    size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}KB"
                except Exception:
                    size_str = "?"
                files.append(f"  📄 {entry} ({size_str})")
        rel = os.path.relpath(resolved, _get_workspace())
        header = f"📁 工作台目录: {'/'.join(rel.split(os.sep)) if rel != '.' else '.'}"
        result = [header]
        if dirs:
            result.append(f"\n📂 子目录 ({len(dirs)}):")
            result.extend(dirs)
        if files:
            result.append(f"\n📃 文件 ({len(files)}):")
            result.extend(files)
        result.append(f"\n共 {len(dirs)} 个目录，{len(files)} 个文件")
        return "\n".join(result)
    except Exception as e:
        return f"❌ 列出目录失败: {e}"


def create_directory(input_str: str) -> str:
    """在工作台中创建目录（支持多级）。"""
    try:
        dir_path = input_str.strip()
        resolved = _resolve_safe_path(dir_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {dir_path}"
        os.makedirs(resolved, exist_ok=True)
        rel_path = os.path.relpath(resolved, _get_workspace())
        return f"✅ 目录已创建: {rel_path}/"
    except Exception as e:
        return f"❌ 创建目录失败: {e}"


# ======================== 项目脚手架 ========================

_TEMPLATES = {
    "python-script": {
        "main.py": lambda n: f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"{n} - 由 OmniCore Agent 自动创建\"\"\"
import sys

def main():
    print(f"Hello from {n}!")
    print(f"Python {{sys.version}}")

if __name__ == "__main__":
    main()
""",
        "README.md": lambda n: f"# {n}\n\n由 OmniCore Agent 创建的 Python 脚本项目。\n",
    },
    "web-app": {
        "index.html": lambda n: f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>{n}</title><link rel="stylesheet" href="style.css"></head>
<body>
    <div id="app"><h1>{n}</h1><p>由 OmniCore Agent 创建</p><div id="content"></div></div>
    <script src="app.js"></script>
</body>
</html>""",
        "style.css": lambda n: f"/* {n} 样式 */\n* {{ margin: 0; padding: 0; box-sizing: border-box; }}\nbody {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}\nh1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}\n",
        "app.js": lambda n: f"// {n} - 主逻辑\nconsole.log('{n} loaded!');\ndocument.addEventListener('DOMContentLoaded', () => {{\n    document.getElementById('content').innerHTML = '<p>欢迎使用 {n}！</p>';\n}});\n",
        "README.md": lambda n: f"# {n}\n\n由 OmniCore Agent 创建的 Web 应用项目。\n",
    },
    "vue-app": {
        "package.json": lambda n: json.dumps({
            "name": n.lower().replace(" ", "-"),
            "version": "1.0.0",
            "private": True,
            "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
            "dependencies": {"vue": "^3.4.0"},
            "devDependencies": {"@vitejs/plugin-vue": "^5.0.0", "vite": "^5.0.0"}
        }, indent=2),
        "vite.config.js": lambda n: """import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig({ plugins: [vue()] })
""",
        "index.html": lambda n: f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>{n}</title></head>
<body><div id="app"></div><script type="module" src="/src/main.js"></script></body>
</html>""",
        "src/main.js": lambda n: """import { createApp } from 'vue'
import App from './App.vue'
createApp(App).mount('#app')
""",
        "src/App.vue": lambda n: f"""<template>
  <div class="app">
    <h1>{{{{ title }}}}</h1>
    <p>由 OmniCore Agent 创建的 Vue 3 应用</p>
    <button @click="count++">点击次数: {{{{ count }}}}</button>
  </div>
</template>
<script setup>
import {{ ref }} from 'vue'
const title = '{n}'
const count = ref(0)
</script>
<style scoped>
.app {{ text-align: center; padding: 40px; font-family: sans-serif; }}
button {{ padding: 8px 16px; margin-top: 16px; cursor: pointer; border-radius: 6px; border: 1px solid #ddd; }}
</style>
""",
        "README.md": lambda n: f"# {n}\n\n由 OmniCore Agent 创建的 Vue 3 应用。\n\n```bash\nnpm install\nnpm run dev\n```\n",
    },
    "express-api": {
        "package.json": lambda n: json.dumps({
            "name": n.lower().replace(" ", "-"),
            "version": "1.0.0",
            "main": "index.js",
            "scripts": {"start": "node index.js", "dev": "node --watch index.js"},
            "dependencies": {"express": "^4.18.0"}
        }, indent=2),
        "index.js": lambda n: f"""const express = require('express');
const app = express();
app.use(express.json());

app.get('/', (req, res) => res.json({{ message: '{n} API is running' }}));
app.get('/health', (req, res) => res.json({{ status: 'ok', uptime: process.uptime() }}));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${{PORT}}`));
""",
        "README.md": lambda n: f"# {n}\n\nExpress API 项目。\n\n```bash\nnpm install\nnpm start\n```\n",
    },
    "fastapi": {
        "main.py": lambda n: f"""from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="{n}")

@app.get("/")
def root():
    return {{"message": "{n} API is running"}}

@app.get("/health")
def health():
    return {{"status": "ok"}}

# pip install fastapi uvicorn
# 运行: uvicorn main:app --reload
""",
        "requirements.txt": lambda n: "fastapi>=0.100.0\nuvicorn>=0.23.0\n",
        "README.md": lambda n: f"# {n}\n\nFastAPI 项目。\n\n```bash\npip install -r requirements.txt\nuvicorn main:app --reload\n```\n",
    },
}


def create_project(input_str: str) -> str:
    """创建项目脚手架。输入格式: 项目类型 | 项目名"""
    try:
        parts = [p.strip() for p in input_str.split("|")]
        project_type = parts[0].lower() if len(parts) > 0 else "empty"
        project_name = parts[1] if len(parts) > 1 else f"project_{uuid.uuid4().hex[:6]}"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        if not safe_name:
            safe_name = f"project_{uuid.uuid4().hex[:6]}"

        ws = _get_workspace()
        project_dir = os.path.join(ws, safe_name)
        if os.path.exists(project_dir):
            return f"错误: 项目目录 '{safe_name}' 已存在。请使用其他名称。"

        os.makedirs(project_dir, exist_ok=True)
        
        # 默认模板
        templates = {
            "empty": {"README.md": f"# {safe_name}\n\n由 OmniCore Agent 创建。\n"},
        }
        templates.update(_TEMPLATES)
        
        if project_type not in templates:
            types_list = ", ".join(templates.keys())
            return f"错误: 不支持的项目类型 '{project_type}'。可选: {types_list}"

        files_to_create = templates[project_type]
        created = []
        for fname, content_fn in files_to_create.items():
            fpath = os.path.join(project_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            content = content_fn(safe_name) if callable(content_fn) else content_fn
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            created.append(fname)

        return (
            f"✅ 项目 '{safe_name}' 创建成功！({project_type})\n"
            f"📁 位置: {safe_name}/\n"
            f"📄 文件 ({len(created)}):\n"
            + "\n".join(f"   📄 {f}" for f in created)
        )
    except Exception as e:
        return f"❌ 创建项目失败: {e}"


# ======================== 代码分析 ========================

def analyze_code(input_str: str) -> str:
    """分析工作台中的代码文件语法。"""
    try:
        file_path = input_str.strip()
        resolved = _resolve_safe_path(file_path)
        if resolved is None:
            return f"❌ 路径不安全或超出工作台目录: {file_path}"
        if not os.path.exists(resolved):
            return f"错误: 文件不存在 '{resolved}'"

        ext = os.path.splitext(resolved)[1].lower()
        with open(resolved, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if ext == ".py":
            source = "".join(lines)
            try:
                compile(source, resolved, "exec")
                code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
                imports = [l for l in lines if l.strip().startswith(("import ", "from "))]
                funcs = [l for l in lines if l.strip().startswith(("def ", "async def "))]
                classes = [l for l in lines if l.strip().startswith("class ")]
                return (f"✅ Python 语法检查通过！\n📊 统计: 总行数 {len(lines)}, "
                        f"代码 {len(code_lines)} 行, 导入 {len(imports)}, "
                        f"函数 {len(funcs)}, 类 {len(classes)}")
            except SyntaxError as e:
                return f"❌ 语法错误: 行 {e.lineno}: {e.msg}"
        elif ext == ".json":
            try:
                json.loads("".join(lines))
                return f"✅ JSON 格式正确。"
            except json.JSONDecodeError as e:
                return f"❌ JSON 格式错误: {e}"
        return f"📄 文件分析: {file_path}, {len(lines)} 行"
    except Exception as e:
        return f"❌ 分析失败: {e}"


def install_dependency(input_str: str) -> str:
    """安装 Python 依赖包。输入: 包名（必须是合法的 PyPI 包名+可选版本约束）"""
    import re as _re
    try:
        package = input_str.strip()
        if not package:
            return "错误: 请输入要安装的包名"
        if any(c in package for c in ["|", ";", "&", "`", "$", ">", "<", "(", ")", "[", "]"]):
            return "⛔ 安全拦截: 包名包含危险字符"
        # 验证包名格式：合法的 PyPI 包名 + 可选版本约束
        # 允许: requests, torch>=2.0.0,<3.0.0, package[extra], git+https://...
        _SAFE_PKG_PATTERN = _re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?'  # 包名
            r'(\[[a-zA-Z0-9,_ -]+\])?'                       # extras
            r'(([><=!~]+[a-zA-Z0-9.*-]+)(,\s*[><=!~]+[a-zA-Z0-9.*-]+)*)?'  # 版本约束
            r'$'
        )
        # 逐个包验证（支持 "pkg1 pkg2" 多包安装）
        for pkg in package.split():
            if not _SAFE_PKG_PATTERN.match(pkg):
                return f"⛔ 安全拦截: '{pkg}' 不是合法的包名格式"
        logger.info(f"安装依赖: {package}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        output = (result.stdout + result.stderr)[:2000]
        if result.returncode == 0:
            return f"✅ 依赖安装成功: {package}\n{output[:500]}"
        return f"❌ 安装失败:\n{output[:1000]}"
    except subprocess.TimeoutExpired:
        return "⏰ 安装超时（超过 120 秒）"
    except Exception as e:
        return f"❌ 安装依赖失败: {e}"