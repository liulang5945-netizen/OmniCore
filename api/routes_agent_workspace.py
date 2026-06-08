"""
工作台/工作空间管理 API 路由
从 routes_agent.py 拆分：工作台文件操作、代码执行、项目脚手架、插件上传、网络诊断
"""
import json
import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File

from core.utils import get_external_path

from .models import CodeRunRequest, CreateProjectRequest, FileSaveRequest

logger = logging.getLogger("ApiServer.Agent.Workspace")
router = APIRouter()


def _get_workspace_dir() -> str:
    """获取工作台目录（优先自定义路径）"""
    settings_path = get_external_path("app_settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            custom_path = settings.get("workspace_path", "")
            if custom_path and os.path.isdir(custom_path):
                return os.path.abspath(custom_path)
        except Exception:
            pass
    return get_external_path("agent_workspace")


@router.get("/api/workspace/path")
def get_workspace_path():
    """获取当前工作台路径"""
    # 优先读取自定义路径
    settings_path = get_external_path("app_settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            custom_path = settings.get("workspace_path", "")
            if custom_path and os.path.isdir(custom_path):
                return {"status": "ok", "path": os.path.abspath(custom_path)}
        except Exception:
            pass
    ws_dir = get_external_path("agent_workspace")
    return {"status": "ok", "path": os.path.abspath(ws_dir)}


@router.post("/api/workspace/path")
def set_workspace_path(req: dict):
    """设置工作台路径"""
    new_path = req.get("path", "").strip()
    if not new_path:
        raise HTTPException(status_code=400, detail="路径不能为空")
    if not os.path.isdir(new_path):
        raise HTTPException(status_code=400, detail=f"路径不存在或不是目录: {new_path}")
    
    settings_path = get_external_path("app_settings.json")
    settings = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            pass
    settings["workspace_path"] = os.path.abspath(new_path)
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    
    return {"status": "ok", "path": os.path.abspath(new_path)}


@router.get("/api/workspace/files")
def list_workspace_files():
    """列出工作台根目录下的文件"""
    ws_dir = _get_workspace_dir()
    os.makedirs(ws_dir, exist_ok=True)
    files = [f for f in os.listdir(ws_dir) if os.path.isfile(os.path.join(ws_dir, f))]
    return {"files": files}


@router.get("/api/workspace/tree")
def list_workspace_tree():
    """递归列出工作台完整目录树"""
    ws_dir = _get_workspace_dir()
    os.makedirs(ws_dir, exist_ok=True)

    def build_tree(dir_path: str) -> list:
        entries = []
        try:
            items = sorted(os.listdir(dir_path))
            for name in items:
                item_path = os.path.join(dir_path, name)
                rel_path = os.path.relpath(item_path, ws_dir)
                if os.path.isdir(item_path):
                    entries.append({
                        "name": name,
                        "path": rel_path,
                        "type": "directory",
                        "children": build_tree(item_path),
                    })
                else:
                    try:
                        size = os.path.getsize(item_path)
                    except Exception:
                        size = 0
                    entries.append({
                        "name": name,
                        "path": rel_path,
                        "type": "file",
                        "size": size,
                    })
        except Exception:
            pass
        return entries

    return {"tree": build_tree(ws_dir)}


@router.get("/api/workspace/file")
def get_workspace_file(name: str):
    """读取工作台中的文件内容"""
    ws_dir = _get_workspace_dir()
    file_path = os.path.abspath(os.path.join(ws_dir, name))
    if not (file_path == ws_dir or file_path.startswith(ws_dir + os.sep)):
        return {"content": "", "error": "路径不安全"}
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return {"content": f.read(), "size": os.path.getsize(file_path)}
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    return {"content": f.read(), "size": os.path.getsize(file_path)}
            except Exception:
                return {"content": "(二进制文件)", "size": os.path.getsize(file_path)}
        except Exception as e:
            return {"content": "", "error": str(e)}
    return {"content": "", "error": "文件不存在"}


@router.post("/api/workspace/file")
def save_workspace_file(req: FileSaveRequest):
    """写入工作台文件"""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=422, detail="文件名不能为空")
    ws_dir = _get_workspace_dir()
    os.makedirs(ws_dir, exist_ok=True)
    safe_path = os.path.abspath(os.path.join(ws_dir, req.name))
    if not (safe_path == ws_dir or safe_path.startswith(ws_dir + os.sep)):
        return {"status": "error", "message": "路径不安全"}
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(req.content)
    except IsADirectoryError:
        raise HTTPException(status_code=422, detail=f"'{req.name}' 是目录不是文件")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"无权限写入: {e}")
    return {"status": "ok", "path": os.path.relpath(safe_path, ws_dir)}


@router.post("/api/workspace/run")
def run_workspace_code(req: CodeRunRequest):
    """运行前端工作台中的 Python 代码"""
    try:
        from agent.sandbox_executor import execute_python_with_files
        result = execute_python_with_files(req.code)
        return {
            "output": result.get("output", ""),
            "files_created": result.get("files_created", []),
            "success": result.get("success", False),
            "error": result.get("error", ""),
        }
    except Exception as e:
        return {"output": f"运行失败: {str(e)}", "success": False, "error": str(e)}


@router.post("/api/workspace/create_project")
async def create_project(req: CreateProjectRequest):
    """创建项目脚手架"""
    from agent.agent import create_project as agent_create_project
    try:
        ws_dir = get_external_path("agent_workspace")
        os.makedirs(ws_dir, exist_ok=True)
        result = agent_create_project(f"{req.type} | project_{req.type}")
        return {"status": "ok", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/workspace/delete/{name:path}")
def delete_workspace_item(name: str):
    """删除工作台文件或文件夹"""
    try:
        ws_dir = _get_workspace_dir()
        item_path = os.path.abspath(os.path.join(ws_dir, name))
        if item_path == ws_dir:
            raise HTTPException(status_code=403, detail="禁止删除工作台根目录")
        if not item_path.startswith(ws_dir + os.sep):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================== Agent 增强 API ========================

@router.post("/api/agent/analyze_code")
async def analyze_code_api(req: CodeRunRequest):
    """分析工作台中的代码文件"""
    from agent.agent import analyze_code
    try:
        result = analyze_code(req.code)
        return {"result": result}
    except Exception as e:
        return {"result": f"分析失败: {str(e)}"}


@router.post("/api/agent/install_dependency")
async def install_dependency_api(req: CodeRunRequest):
    """安装 Python 依赖包"""
    from agent.agent import install_dependency
    try:
        result = install_dependency(req.code)
        return {"result": result}
    except Exception as e:
        return {"result": f"安装失败: {str(e)}"}


@router.get("/api/agent/plans")
def list_plans_api():
    """获取所有任务计划"""
    from agent.agent import list_plans
    try:
        result = list_plans("")
        return {"plans": result}
    except Exception as e:
        return {"plans": f"获取失败: {str(e)}"}


@router.get("/api/agent/context")
def load_context_api(key: str = ""):
    """读取开发上下文"""
    from agent.agent import load_context
    try:
        result = load_context(key)
        return {"context": result}
    except Exception as e:
        return {"context": f"读取失败: {str(e)}"}


@router.post("/api/agent/save_context")
async def save_context_api(req: CodeRunRequest):
    """保存开发上下文"""
    from agent.agent import save_context
    try:
        result = save_context(req.code)
        return {"result": result}
    except Exception as e:
        return {"result": f"保存失败: {str(e)}"}


# ======================== 插件管理 ========================

@router.post("/api/plugins/upload")
async def upload_plugin(file: UploadFile = File(...)):
    """接收拖拽的智能体插件并热加载"""
    try:
        plugins_dir = get_external_path("plugins")
        os.makedirs(plugins_dir, exist_ok=True)
        file_path = os.path.join(plugins_dir, os.path.basename(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "message": f"插件 `{file.filename}` 已成功热安装！下一条对话将直接生效。"
        }
    except Exception as e:
        logger.error(f"插件安装失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================== 常用路径 ========================

@router.get("/api/workspace/quick_paths")
def get_quick_paths():
    """获取常用路径列表（桌面、文档、用户主目录等）"""
    import platform
    home = os.path.expanduser("~")
    paths = []

    if platform.system() == "Windows":
        desktop = os.path.join(home, "Desktop")
        documents = os.path.join(home, "Documents")
        downloads = os.path.join(home, "Downloads")
        if os.path.isdir(desktop):
            paths.append({"label": "桌面", "path": desktop})
        if os.path.isdir(documents):
            paths.append({"label": "文档", "path": documents})
        if os.path.isdir(downloads):
            paths.append({"label": "下载", "path": downloads})
        # 常见盘符
        for letter in "CDEFG":
            drive = f"{letter}:\\"
            if os.path.isdir(drive):
                paths.append({"label": f"{letter}: 盘", "path": drive})
    else:
        desktop = os.path.join(home, "Desktop")
        documents = os.path.join(home, "Documents")
        downloads = os.path.join(home, "Downloads")
        if os.path.isdir(desktop):
            paths.append({"label": "桌面", "path": desktop})
        if os.path.isdir(documents):
            paths.append({"label": "文档", "path": documents})
        if os.path.isdir(downloads):
            paths.append({"label": "下载", "path": downloads})

    paths.append({"label": "用户主目录", "path": home})
    return {"paths": paths}


# ======================== 网络诊断 ========================

@router.get("/api/network/diagnose")
def network_diagnose():
    """网络诊断：检测 HuggingFace 镜像和官方源的连通性"""
    try:
        from model.model_downloader import diagnose_network
        result = diagnose_network()
        return {"status": "ok", "diagnosis": result}
    except Exception as e:
        logger.error(f"网络诊断失败: {e}")
        return {"status": "error", "message": str(e)}