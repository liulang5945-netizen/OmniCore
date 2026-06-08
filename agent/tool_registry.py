"""
工具注册表 (Tool Registry)
==========================
统一管理所有 Agent 工具的注册、查找、执行。
支持本地工具和 MCP 远程工具。

使用方式:
    from agent.tool_registry import registry
    
    # 注册本地工具
    registry.register(ToolDef(
        name="my_tool",
        description="工具描述",
        parameters={"type": "object", "properties": {...}},
        func=my_function,
    ))
    
    # 执行工具
    result = registry.execute("my_tool", {"arg1": "value1"})
    
    # 获取所有工具的 JSON Schema（用于 LLM function calling）
    schemas = registry.get_tool_schemas()
"""
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ToolRegistry")


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    func: Optional[Callable] = None
    source: str = "local"          # "local" | "mcp" | "plugin"
    source_id: str = ""            # 来源标识，如 MCP 服务器名
    enabled: bool = True
    category: str = "通用"

    def to_schema(self) -> dict:
        """转换为 OpenAI function calling 格式的 JSON Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_info(self) -> dict:
        """转换为前端展示用的信息字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "source_id": self.source_id,
            "enabled": self.enabled,
            "category": self.category,
        }


class ToolRegistry:
    """工具注册表：管理所有工具的注册、查找、执行"""

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    # ======================== 注册 ========================

    def register(self, tool: ToolDef):
        """注册一个工具"""
        if not tool.name:
            logger.warning("尝试注册无名工具，已忽略")
            return
        if tool.name in self._tools:
            logger.info(f"工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool
        logger.debug(f"已注册工具: {tool.name} (来源: {tool.source})")

    def register_many(self, tools: List[ToolDef]):
        """批量注册工具"""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> bool:
        """注销一个工具"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"已注销工具: {name}")
            return True
        return False

    def unregister_by_source(self, source_id: str):
        """注销来自特定来源的所有工具"""
        to_remove = [name for name, t in self._tools.items() if t.source_id == source_id]
        for name in to_remove:
            del self._tools[name]
        logger.info(f"已注销来源 '{source_id}' 的 {len(to_remove)} 个工具")

    # ======================== 查询 ========================

    def get(self, name: str) -> Optional[ToolDef]:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self, source: str = None, enabled_only: bool = True) -> List[ToolDef]:
        """列出工具"""
        tools = list(self._tools.values())
        if source:
            tools = [t for t in tools if t.source == source]
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    def list_names(self, enabled_only: bool = True) -> List[str]:
        """列出所有工具名"""
        return [t.name for t in self.list_tools(enabled_only=enabled_only)]

    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    def count(self) -> int:
        """工具总数"""
        return len(self._tools)

    # ======================== 启用/禁用 ========================

    def enable(self, name: str):
        """启用工具"""
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str):
        """禁用工具"""
        if name in self._tools:
            self._tools[name].enabled = False

    def enable_source(self, source_id: str):
        """启用来自特定来源的所有工具"""
        for t in self._tools.values():
            if t.source_id == source_id:
                t.enabled = True

    def disable_source(self, source_id: str):
        """禁用来自特定来源的所有工具"""
        for t in self._tools.values():
            if t.source_id == source_id:
                t.enabled = False

    # ======================== 执行 ========================

    def execute(self, name: str, args: dict) -> str:
        """执行指定工具"""
        tool = self._tools.get(name)
        if not tool:
            return f"❌ 工具 '{name}' 不存在。可用工具: {', '.join(self.list_names())}"
        if not tool.enabled:
            return f"❌ 工具 '{name}' 已禁用"
        if not tool.func:
            return f"❌ 工具 '{name}' 没有可执行的函数"

        try:
            # 支持两种参数传递方式：
            # 1. 如果工具参数只有一个 "input" 属性，直接传字符串
            # 2. 否则传整个字典
            params = tool.parameters.get("properties", {})
            if len(params) == 1 and "input" in params:
                # 单参数 "input" 模式
                if isinstance(args, dict):
                    value = args.get("input", args.get("value", json.dumps(args, ensure_ascii=False)))
                else:
                    value = str(args)
                result = tool.func(str(value))
            elif len(params) == 1:
                # 单参数模式（参数名不是 "input"）
                param_name = list(params.keys())[0]
                if isinstance(args, dict):
                    value = args.get(param_name, json.dumps(args, ensure_ascii=False))
                else:
                    value = str(args)
                result = tool.func(str(value))
            else:
                # 多参数模式
                if isinstance(args, dict):
                    result = tool.func(**args)
                else:
                    result = tool.func(str(args))

            return str(result) if result is not None else "✅ 工具执行完成（无返回值）"
        except TypeError as e:
            # 参数类型不匹配，尝试用 input 字符串重试
            try:
                if isinstance(args, dict):
                    result = tool.func(json.dumps(args, ensure_ascii=False))
                else:
                    result = tool.func(str(args))
                return str(result) if result is not None else "✅ 工具执行完成"
            except Exception as retry_e:
                return f"❌ 工具 '{name}' 参数错误: {retry_e}"
        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {traceback.format_exc()}")
            return f"❌ 工具 '{name}' 执行失败: {e}"

    # ======================== Schema 导出 ========================

    def get_tool_schemas(self, enabled_only: bool = True) -> list:
        """获取所有工具的 JSON Schema（用于 LLM function calling）"""
        tools = self.list_tools(enabled_only=enabled_only)
        return [t.to_schema() for t in tools]

    def get_tool_descriptions(self, enabled_only: bool = True) -> str:
        """获取所有工具的文本描述（用于 prompt）"""
        tools = self.list_tools(enabled_only=enabled_only)
        if not tools:
            return "暂无可用工具。"

        lines = ["可用工具:"]
        for t in tools:
            params = t.parameters.get("properties", {})
            param_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
            if param_desc:
                lines.append(f"- **{t.name}**({param_desc}): {t.description}")
            else:
                lines.append(f"- **{t.name}**: {t.description}")
        return "\n".join(lines)

    def get_all_info(self, enabled_only: bool = True) -> list:
        """获取所有工具的详细信息（用于前端展示）"""
        return [t.to_info() for t in self.list_tools(enabled_only=enabled_only)]

    def clear(self):
        """清空所有工具"""
        self._tools.clear()

    def __repr__(self):
        return f"ToolRegistry(tools={len(self._tools)})"


# ======================== 全局单例 ========================

registry = ToolRegistry()


# ── 自修改工具注册（让态极能自主发现和安装新工具） ──
def _register_self_modification_tools():
    """注册自修改工具到全局注册表"""
    try:
        from agent.self_modification import get_self_modification_engine
        _sm_engine = get_self_modification_engine()

        def _discover_tools(input_str: str) -> str:
            """搜索可用工具。输入: 能力描述关键词"""
            keyword = input_str.strip()
            if not keyword:
                return "请输入要搜索的能力关键词，如：浏览器、翻译、数据库"
            matches = _sm_engine._discovery.find_matching_tools(keyword, registry)
            if not matches:
                return f"未找到与 '{keyword}' 匹配的工具"
            lines = [f"找到 {len(matches)} 个匹配工具:"]
            for m in matches:
                lines.append(f"  [{m['source']}] {m['name']} - {m['description'][:60]} (匹配度: {m['match_score']:.1f})")
            return "\n".join(lines)

        def _install_tool(input_str: str) -> str:
            """安装一个新工具。输入: 工具ID（MCP服务器ID或插件ID）"""
            tool_id = input_str.strip()
            if not tool_id:
                return "请输入要安装的工具ID"
            result = _sm_engine._try_install({"source": "mcp", "id": tool_id}, registry)
            if result.get("success"):
                return f"✅ {result['message']}"
            result = _sm_engine._try_install({"source": "plugin", "id": tool_id}, registry)
            if result.get("success"):
                return f"✅ {result['message']}"
            return f"❌ 安装失败: {result.get('message', '未知错误')}"

        def _my_capabilities(input_str: str) -> str:
            """查看当前已具备的能力"""
            tools = registry.list_tools(enabled_only=True)
            lines = [f"当前已注册 {len(tools)} 个工具:"]
            for t in tools:
                lines.append(f"  - {t.name}: {t.description[:50]}")
            return "\n".join(lines)

        registry.register(ToolDef(
            name="discover_tools",
            description="搜索可用的新工具，根据能力关键词在MCP市场和插件目录中查找",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "能力关键词，如：浏览器、翻译、数据库"}}},
            func=_discover_tools,
            source="self_modification",
            category="自修改",
        ))
        registry.register(ToolDef(
            name="install_tool",
            description="安装一个新的MCP服务器或插件工具",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "工具ID"}}},
            func=_install_tool,
            source="self_modification",
            category="自修改",
        ))
        def _evolve(input_str: str) -> str:
            """自主进化：遇到不会的能力时，自动搜索或编写工具补齐。输入: 缺失的能力描述"""
            ability = input_str.strip()
            if not ability:
                return "请描述你需要但不具备的能力"
            result = _sm_engine.evolve(ability, registry)
            if result.get("evolved"):
                return f"✅ 进化成功! {result.get('message', '')}"
            return f"❌ 进化失败: {result.get('message', '未知原因')}"

        registry.register(ToolDef(
            name="my_capabilities",
            description="查看当前已具备的所有工具能力列表",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "留空即可"}}},
            func=_my_capabilities,
            source="self_modification",
            category="自修改",
        ))
        registry.register(ToolDef(
            name="evolve",
            description="自主进化：遇到不具备的能力时，自动搜索市场或自己编写代码生成工具补齐能力",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "缺失的能力描述，如：翻译、数据库操作、图表生成"}}},
            func=_evolve,
            source="self_modification",
            category="自修改",
        ))
        logger.info("自修改工具已注册: discover_tools, install_tool, my_capabilities, evolve")
    except Exception as e:
        logger.debug(f"自修改工具注册失败: {e}")


def register_local_tools():
    """注册所有本地内置工具到注册表"""
    from agent.agent_tools import (
        read_local_file, write_file, edit_file, delete_file,
        list_directory, create_directory, create_project,
        install_dependency, analyze_code,
    )
    from agent.agent_planner import (
        create_plan, update_plan, get_plan, list_plans,
        save_context, load_context,
    )

    local_tools = [
        ToolDef(
            name="read_local_file",
            description="读取工作台文件内容，支持分页。输入文件路径（可选逗号后跟页码）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "文件路径，如 data.txt, 2 表示第2页"}
            }, "required": ["input"]},
            func=read_local_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="write_file",
            description="在工作台中创建或覆盖写入文件。输入格式: 文件路径 | 文件内容",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 文件路径 | 文件内容"}
            }, "required": ["input"]},
            func=write_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="edit_file",
            description="精确编辑工作台中的文件内容。输入格式: 文件路径 | 旧文本 | 新文本",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 文件路径 | 旧文本 | 新文本"}
            }, "required": ["input"]},
            func=edit_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="delete_file",
            description="删除工作台中的文件或空目录。输入文件路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "要删除的文件或目录路径"}
            }, "required": ["input"]},
            func=delete_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="list_directory",
            description="列出工作台目录内容。输入目录路径（留空则列出根目录）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "目录路径（可选）"}
            }},
            func=list_directory,
            source="local", category="文件",
        ),
        ToolDef(
            name="create_directory",
            description="在工作台中创建目录。输入目录路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "要创建的目录路径"}
            }, "required": ["input"]},
            func=create_directory,
            source="local", category="文件",
        ),
        ToolDef(
            name="create_project",
            description="创建完整项目脚手架。输入格式: 项目类型 | 项目名。支持: python-script, web-app, vue-app 等。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 项目类型 | 项目名"}
            }, "required": ["input"]},
            func=create_project,
            source="local", category="开发",
        ),
        ToolDef(
            name="install_dependency",
            description="安装 Python 依赖包。输入包名。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "包名，如 requests"}
            }, "required": ["input"]},
            func=install_dependency,
            source="local", category="开发",
        ),
        ToolDef(
            name="analyze_code",
            description="分析代码文件语法。支持 .py, .js, .json。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "文件路径"}
            }, "required": ["input"]},
            func=analyze_code,
            source="local", category="开发",
        ),
        ToolDef(
            name="create_plan",
            description="为复杂开发任务创建执行计划。输入任务描述。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "任务描述"}
            }, "required": ["input"]},
            func=create_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="update_plan",
            description="更新计划步骤状态。输入格式: 计划ID | 步骤序号 | 状态(done/failed/skip)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 计划ID | 步骤序号 | 状态"}
            }, "required": ["input"]},
            func=update_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="get_plan",
            description="获取任务计划进度。输入计划ID或'all'查看全部。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "计划ID或'all'"}
            }, "required": ["input"]},
            func=get_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="save_context",
            description="保存开发上下文信息。输入格式: key | value",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: key | value"}
            }, "required": ["input"]},
            func=save_context,
            source="local", category="规划",
        ),
        ToolDef(
            name="load_context",
            description="读取已保存的上下文信息。输入key。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "上下文key"}
            }, "required": ["input"]},
            func=load_context,
            source="local", category="规划",
        ),
    ]

    # Python 代码执行
    try:
        from agent.sandbox_executor import execute_python_code_safe
        local_tools.append(ToolDef(
            name="execute_python",
            description="在安全沙箱中执行 Python 代码。输入必须是纯 Python 代码。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "Python 代码"}
            }, "required": ["input"]},
            func=execute_python_code_safe,
            source="local", category="开发",
        ))
    except ImportError:
        pass

    # 搜索引擎
    try:
        from agent.agent import _create_robust_search
        import json as _json
        # 读取用户搜索配置
        _search_engine = "智能多核"
        _search_key = ""
        _ui_settings = {}
        try:
            from core.config import get_external_path
            _sp = get_external_path("app_settings.json")
            import os as _os
            if _os.path.exists(_sp):
                with open(_sp, "r", encoding="utf-8") as _f:
                    _ui_settings = _json.load(_f)
                _search_engine = _ui_settings.get("search_engine", "智能多核")
                _search_key = _ui_settings.get("search_key", "")
        except Exception:
            pass
        search_func = _create_robust_search(_search_engine, _ui_settings, _search_key, "")
        local_tools.append(ToolDef(
            name="search",
            description="在互联网上搜索最新信息。输入简短的搜索关键词。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "搜索关键词"}
            }, "required": ["input"]},
            func=search_func,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 网页阅读
    try:
        from agent.agent import read_webpage
        local_tools.append(ToolDef(
            name="read_webpage",
            description="深入阅读指定网址的网页正文。输入完整 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=read_webpage,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 高级浏览器访问（优先 Playwright MCP，降级到 read_webpage）
    try:
        def _browse_web(url: str) -> str:
            """浏览器访问网页，支持 JS 渲染。优先用 Playwright MCP，降级到 requests。"""
            # 优先尝试 Playwright MCP
            pw_tools = [n for n in registry.list_names() if "playwright" in n and ("get_content" in n or "navigate" in n)]
            if pw_tools:
                try:
                    # 先导航
                    nav_tool = [n for n in pw_tools if "navigate" in n]
                    if nav_tool:
                        registry.execute(nav_tool[0], {"url": url})
                    # 再获取内容
                    content_tool = [n for n in pw_tools if "get_content" in n or "get_text" in n or "page_content" in n]
                    if content_tool:
                        result = registry.execute(content_tool[0], {})
                        if result and len(str(result).strip()) > 50:
                            return str(result)[:8000]
                except Exception as e:
                    logger.debug(f"Playwright MCP 浏览失败，降级: {e}")

            # 降级到 read_webpage
            try:
                from agent.agent import read_webpage
                return read_webpage(url)
            except Exception:
                return f"无法访问: {url}"

        local_tools.append(ToolDef(
            name="browse_web",
            description="用浏览器访问网页（支持 JavaScript 渲染的动态页面）。输入完整 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=_browse_web,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 智能网页抓取（优先 MCP fetch，降级到 requests）
    try:
        def _smart_fetch(url: str) -> str:
            """智能抓取网页，返回 Markdown 格式正文。优先用 MCP fetch，降级到 requests。"""
            # 优先尝试 MCP fetch
            fetch_tools = [n for n in registry.list_names() if "fetch" in n and "markdown" in n]
            if fetch_tools:
                try:
                    result = registry.execute(fetch_tools[0], {"url": url})
                    if result and len(str(result).strip()) > 50:
                        return str(result)[:10000]
                except Exception as e:
                    logger.debug(f"MCP fetch 失败，降级: {e}")

            # 降级到 requests + BeautifulSoup
            try:
                import requests
                from bs4 import BeautifulSoup
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                resp.encoding = resp.apparent_encoding
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)[:10000]
            except Exception as e:
                return f"抓取失败: {e}"

        local_tools.append(ToolDef(
            name="smart_fetch",
            description="智能抓取网页正文，返回 Markdown 格式。适合文章、文档、博客等。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=_smart_fetch,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 命令行执行
    try:
        from agent.agent import run_command
        local_tools.append(ToolDef(
            name="run_command",
            description="运行本地命令行命令（安全白名单）。输入命令字符串。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "命令字符串"}
            }, "required": ["input"]},
            func=run_command,
            source="local", category="系统",
        ))
    except Exception:
        pass

    # B 站字幕
    try:
        from tools.bilibili_subtitle import read_bilibili_subtitle
        local_tools.append(ToolDef(
            name="read_bilibili_subtitle",
            description="读取 B 站视频官方 CC 字幕。输入 B 站视频 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "B 站视频 URL"}
            }, "required": ["input"]},
            func=read_bilibili_subtitle,
            source="local", category="媒体",
        ))
    except ImportError:
        pass

    # 通用知识自学习工具
    try:
        from agent.knowledge_learner import get_knowledge_learner
        _learner = get_knowledge_learner()

        def _learn_knowledge(input_str: str) -> str:
            """启动某领域的知识学习。输入格式: 领域名 | 来源URL1,URL2,... | 深度(shallow/medium/deep)"""
            try:
                parts = [p.strip() for p in input_str.split("|")]
                domain = parts[0]
                sources = None
                depth = "medium"
                if len(parts) > 1 and parts[1]:
                    sources = [s.strip() for s in parts[1].split(",") if s.strip()]
                if len(parts) > 2 and parts[2]:
                    depth = parts[2].strip()
                session = _learner.start_learning(domain, sources=sources, depth=depth)
                return (
                    f"✅ 学习完成 [{domain}]\n"
                    f"状态: {session.status}\n"
                    f"采集: {session.entries_collected} 条\n"
                    f"新增: {session.entries_new} | 更新: {session.entries_updated} | 跳过: {session.entries_skipped}\n"
                    f"验证得分: {session.verify_score:.0%}\n"
                    + "\n".join(session.log[-5:])
                )
            except Exception as e:
                return f"❌ 学习失败: {e}"

        def _query_knowledge(input_str: str) -> str:
            """查询已学习的知识。输入: 查询问题（可选 | 领域名 限定范围）"""
            try:
                parts = [p.strip() for p in input_str.split("|")]
                question = parts[0]
                domain = parts[1] if len(parts) > 1 else ""
                return _learner.query(question, domain=domain)
            except Exception as e:
                return f"❌ 查询失败: {e}"

        def _learning_report(input_str: str) -> str:
            """查看学习进度报告。输入: 领域名（留空查看全部）"""
            try:
                return _learner.get_learning_report(domain=input_str.strip())
            except Exception as e:
                return f"❌ 获取报告失败: {e}"

        local_tools.extend([
            ToolDef(
                name="learn_knowledge",
                description="启动某领域的知识自学习（自动采集、结构化、存储、验证）。输入格式: 领域名 | 来源URL(逗号分隔,可选) | 深度(shallow/medium/deep)",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "格式: 领域名 | 来源URL | 深度"}
                }, "required": ["input"]},
                func=_learn_knowledge,
                source="local", category="学习",
            ),
            ToolDef(
                name="query_knowledge",
                description="查询已学习的知识库。输入问题（可选 | 领域名）。",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "查询问题（可选|领域名）"}
                }, "required": ["input"]},
                func=_query_knowledge,
                source="local", category="学习",
            ),
            ToolDef(
                name="learning_report",
                description="查看知识学习进度报告。输入领域名（留空查看全部）。",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "领域名（可选）"}
                }},
                func=_learning_report,
                source="local", category="学习",
            ),
        ])
    except Exception as e:
        logger.warning(f"知识学习工具注册失败: {e}")

    registry.register_many(local_tools)
    logger.info(f"已注册 {len(local_tools)} 个本地工具")


# 模块加载时自动注册本地工具
try:
    register_local_tools()
except Exception as e:
    logger.warning(f"本地工具注册失败（可能尚未初始化）: {e}")

# 注册自修改工具（在本地工具之后，确保 registry 已有工具列表）
try:
    _register_self_modification_tools()
except Exception as e:
    logger.debug(f"自修改工具注册失败: {e}")
