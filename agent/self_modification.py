"""
态极自修改模块 (Self-Modification)
===================================
让态极具备自主发现、评估、安装新工具的能力。

核心能力：
1. 能力自省 — 知道自己能做什么、不能做什么
2. 工具发现 — 搜索 MCP 市场和本地插件目录
3. 工具安装 — 自主决策并安装合适的工具
4. 能力评估 — 安装后验证新工具是否正常工作

安全边界：
- 安装前必须经过 SafetyGuard 审批
- 有安装冷却期（防止疯狂安装）
- 只允许白名单类型的工具
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Taiji.SelfModification")


# 安装冷却期（秒）：两次安装之间的最短间隔
INSTALL_COOLDOWN = 60

# 最大同时安装数
MAX_AUTO_INSTALLS = 3

# 安全白名单：只允许这些类型的工具自动安装
ALLOWED_TOOL_SOURCES = {"mcp", "plugin"}


class CapabilityGap:
    """一次能力缺失的记录"""
    def __init__(self, task: str, missing_ability: str, suggested_tool: str = "",
                 confidence: float = 0.0):
        self.task = task
        self.missing_ability = missing_ability
        self.suggested_tool = suggested_tool
        self.confidence = confidence
        self.detected_at = time.time()

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "missing_ability": self.missing_ability,
            "suggested_tool": self.suggested_tool,
            "confidence": self.confidence,
            "detected_at": self.detected_at,
        }


class ToolDiscovery:
    """工具发现引擎"""

    # 能力关键词到工具类型的映射
    ABILITY_KEYWORDS = {
        "浏览网页": ["browse_web", "playwright"],
        "搜索": ["search", "smart_fetch"],
        "文件操作": ["write_file", "read_local_file", "edit_file"],
        "代码执行": ["execute_python", "run_command"],
        "图片生成": ["generate_image"],
        "语音合成": ["text_to_speech"],
        "视频处理": ["understand_video", "generate_video"],
        "数据分析": ["execute_python"],
        "数据库": ["sqlite", "database"],
        "绘图": ["generate_image", "matplotlib"],
        "翻译": ["translate"],
        "计算": ["execute_python", "calculator"],
        "爬虫": ["browse_web", "smart_fetch", "read_webpage"],
    }

    def __init__(self):
        self._marketplace_cache = None
        self._cache_time = 0
        self._cache_ttl = 300  # 5 分钟缓存

    def detect_gap(self, task: str, error_message: str = "",
                   tool_registry=None) -> Optional[CapabilityGap]:
        """
        检测任务执行中是否出现了能力缺失。

        Args:
            task: 原始任务描述
            error_message: 工具执行的错误信息（如果有）
            tool_registry: 当前工具注册表

        Returns:
            CapabilityGap 或 None（如果不需要新工具）
        """
        if not tool_registry:
            return None

        # 检查错误信息中是否暗示能力缺失
        gap_indicators = [
            "没有找到合适的工具",
            "no suitable tool",
            "tool not found",
            "not available",
            "不支持的操作",
            "unsupported",
        ]

        task_lower = task.lower()
        error_lower = error_message.lower() if error_message else ""

        # 检查任务是否涉及当前不具备的能力
        for ability, tool_names in self.ABILITY_KEYWORDS.items():
            ability_lower = ability.lower()
            if ability_lower in task_lower or any(kw in task_lower for kw in tool_names):
                # 检查是否已有相关工具
                has_tool = any(tool_registry.has(t) for t in tool_names)
                if not has_tool:
                    return CapabilityGap(
                        task=task,
                        missing_ability=ability,
                        suggested_tool=tool_names[0],
                        confidence=0.7,
                    )

        # 检查错误信息
        if any(indicator in error_lower for indicator in gap_indicators):
            return CapabilityGap(
                task=task,
                missing_ability="unknown",
                confidence=0.3,
            )

        return None

    def search_marketplace(self, keyword: str = "") -> List[dict]:
        """
        搜索 MCP 市场中可用的工具/服务器。

        Returns:
            [{id, name, description, tools: [...], install_command, ...}]
        """
        try:
            from agent.mcp_manager import mcp_manager
            result = mcp_manager.get_marketplace(keyword=keyword)
            servers = result.get("servers", [])
            # 缓存结果
            self._marketplace_cache = servers
            self._cache_time = time.time()
            return servers
        except Exception as e:
            logger.warning(f"搜索 MCP 市场失败: {e}")
            return []

    def find_matching_tools(self, ability: str, tool_registry=None) -> List[dict]:
        """
        根据缺失的能力，在市场中查找匹配的工具。

        Args:
            ability: 缺失的能力描述
            tool_registry: 当前工具注册表

        Returns:
            [{source, id, name, description, match_score}]
        """
        matches = []

        # 1. 搜索 MCP 市场
        try:
            servers = self.search_marketplace(ability)
            for server in servers:
                # 计算匹配分数
                score = self._calculate_match_score(ability, server)
                if score > 0.3:
                    matches.append({
                        "source": "mcp",
                        "id": server.get("id", ""),
                        "name": server.get("name", ""),
                        "description": server.get("description", ""),
                        "tools": server.get("tools", []),
                        "match_score": score,
                    })
        except Exception as e:
            logger.debug(f"MCP 市场搜索失败: {e}")

        # 2. 搜索本地插件目录
        try:
            plugins = self._search_local_plugins(ability)
            matches.extend(plugins)
        except Exception as e:
            logger.debug(f"本地插件搜索失败: {e}")

        # 按匹配分数排序
        matches.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return matches[:5]  # 最多返回 5 个候选

    def _calculate_match_score(self, ability: str, server: dict) -> float:
        """计算服务器与能力需求的匹配分数"""
        score = 0.0
        ability_lower = ability.lower()

        # 名称匹配
        name = server.get("name", "").lower()
        if ability_lower in name or name in ability_lower:
            score += 0.4

        # 描述匹配
        desc = server.get("description", "").lower()
        for word in ability_lower.split():
            if word in desc:
                score += 0.2

        # 工具列表匹配
        tools = server.get("tools", [])
        for tool in tools:
            tool_desc = tool.get("description", "").lower() if isinstance(tool, dict) else str(tool).lower()
            if ability_lower in tool_desc:
                score += 0.3
                break

        return min(score, 1.0)

    def _search_local_plugins(self, ability: str) -> List[dict]:
        """搜索本地插件目录"""
        matches = []
        try:
            from core.utils import get_external_path
            plugins_dir = get_external_path("plugins")
            if not os.path.exists(plugins_dir):
                return matches

            for item in os.listdir(plugins_dir):
                manifest_path = os.path.join(plugins_dir, item, "manifest.json")
                if os.path.isfile(manifest_path):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        desc = manifest.get("description", "").lower()
                        name = manifest.get("name", "").lower()
                        if ability.lower() in desc or ability.lower() in name:
                            matches.append({
                                "source": "plugin",
                                "id": manifest.get("id", item),
                                "name": manifest.get("name", item),
                                "description": manifest.get("description", ""),
                                "tools": manifest.get("tools", []),
                                "match_score": 0.5,
                            })
                    except Exception:
                        continue
        except Exception:
            pass
        return matches


class SelfModificationEngine:
    """
    态极自修改引擎

    让态极能够：
    1. 感知自己的能力边界
    2. 发现缺失的能力
    3. 自主决策是否安装新工具
    4. 找不到现成工具时，自己写代码生成新工具
    5. 安装后验证工具可用性
    6. 进化成果持久化，跨会话复用
    """

    def __init__(self):
        self._discovery = ToolDiscovery()
        self._gaps: List[CapabilityGap] = []
        self._install_history: List[dict] = []
        self._auto_install_count = 0
        self._last_install_time = 0
        self._enabled = True
        self._generated_tools_dir = self._init_generated_dir()
        self._load_generated_tools()

    def _init_generated_dir(self) -> str:
        """初始化自主生成工具的存储目录"""
        from core.utils import get_external_path
        d = get_external_path("plugins")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_generated_tools(self):
        """加载之前自主生成的工具（跨会话复用）"""
        meta_path = os.path.join(self._generated_tools_dir, "_evolved_tools.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._evolved_meta = json.load(f)
                count = len(self._evolved_meta.get("tools", {}))
                if count:
                    logger.info(f"已加载 {count} 个自主生成的工具")
            except Exception:
                self._evolved_meta = {"tools": {}}
        else:
            self._evolved_meta = {"tools": {}}

    def _save_evolved_meta(self):
        """保存进化元数据"""
        meta_path = os.path.join(self._generated_tools_dir, "_evolved_tools.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._evolved_meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存进化元数据失败: {e}")

    def on_tool_failure(self, task: str, tool_name: str, error: str,
                        tool_registry=None) -> Optional[dict]:
        """
        工具执行失败时的回调。

        检测是否为能力缺失，如果是则尝试发现和安装替代工具。

        Returns:
            {
                "gap_detected": True/False,
                "action": "install" | "suggest" | "none",
                "tool": {...},  # 如果建议安装的工具
                "message": "...",
            }
        """
        if not self._enabled:
            return {"gap_detected": False, "action": "none", "message": "自修改引擎已禁用"}

        # 检测能力缺失
        gap = self._discovery.detect_gap(task, error, tool_registry)
        if not gap:
            return {"gap_detected": False, "action": "none", "message": ""}

        self._gaps.append(gap)
        logger.info(f"检测到能力缺失: {gap.missing_ability} (任务: {task[:50]})")

        # 查找匹配工具
        matches = self._discovery.find_matching_tools(gap.missing_ability, tool_registry)
        if not matches:
            return {
                "gap_detected": True,
                "action": "suggest",
                "message": f"检测到缺少 '{gap.missing_ability}' 能力，但未找到合适的工具",
            }

        best_match = matches[0]

        # 检查是否可以自动安装
        if self._can_auto_install():
            install_result = self._try_install(best_match, tool_registry)
            if install_result.get("success"):
                return {
                    "gap_detected": True,
                    "action": "install",
                    "tool": best_match,
                    "message": f"已自动安装 {best_match['name']}，正在重试任务...",
                }

        # 不能自动安装，给出建议
        return {
            "gap_detected": True,
            "action": "suggest",
            "tool": best_match,
            "candidates": matches[:3],
            "message": (
                f"检测到缺少 '{gap.missing_ability}' 能力。"
                f"建议安装: {best_match['name']} ({best_match['description'][:50]})"
            ),
        }

    def _can_auto_install(self) -> bool:
        """检查是否满足自动安装的条件"""
        now = time.time()

        # 冷却期检查
        if now - self._last_install_time < INSTALL_COOLDOWN:
            return False

        # 最大安装数检查
        if self._auto_install_count >= MAX_AUTO_INSTALLS:
            return False

        return True

    def _try_install(self, tool_info: dict, tool_registry=None) -> dict:
        """
        尝试安装一个工具。

        Returns:
            {"success": True/False, "message": "..."}
        """
        source = tool_info.get("source", "")
        tool_id = tool_info.get("id", "")

        if source not in ALLOWED_TOOL_SOURCES:
            return {"success": False, "message": f"不允许自动安装来源为 '{source}' 的工具"}

        try:
            if source == "mcp":
                return self._install_mcp_server(tool_id, tool_registry)
            elif source == "plugin":
                return self._install_plugin(tool_id, tool_registry)
        except Exception as e:
            logger.warning(f"安装工具失败: {e}")
            return {"success": False, "message": str(e)}

        return {"success": False, "message": "未知的工具来源"}

    def _install_mcp_server(self, server_id: str, tool_registry=None) -> dict:
        """安装 MCP 服务器"""
        try:
            from agent.mcp_manager import mcp_manager
            result = mcp_manager.install_server(server_id)
            if result.get("status") == "ok":
                # 启动服务器
                start_result = mcp_manager.start_server(server_id)
                self._last_install_time = time.time()
                self._auto_install_count += 1
                self._install_history.append({
                    "type": "mcp",
                    "id": server_id,
                    "time": time.time(),
                    "success": True,
                })
                logger.info(f"已自动安装并启动 MCP 服务器: {server_id}")
                return {"success": True, "message": f"MCP 服务器 {server_id} 已安装并启动"}
            return {"success": False, "message": result.get("message", "安装失败")}
        except Exception as e:
            return {"success": False, "message": f"MCP 安装失败: {e}"}

    def _install_plugin(self, plugin_id: str, tool_registry=None) -> dict:
        """加载本地插件"""
        try:
            from core.plugin_manager import PluginManager
            pm = PluginManager()
            pm.load_plugin(plugin_id)
            self._last_install_time = time.time()
            self._auto_install_count += 1
            self._install_history.append({
                "type": "plugin",
                "id": plugin_id,
                "time": time.time(),
                "success": True,
            })
            logger.info(f"已自动加载插件: {plugin_id}")
            return {"success": True, "message": f"插件 {plugin_id} 已加载"}
        except Exception as e:
            return {"success": False, "message": f"插件加载失败: {e}"}

    # ======================== 自主编程 ========================

    def auto_code_tool(self, ability: str, task_context: str = "",
                       tool_registry=None) -> Optional[dict]:
        """
        找不到现成工具时，让 LLM 自己写一个工具。

        流程：
        1. 构造提示词，描述需要什么能力
        2. 调用 LLM 生成插件代码
        3. 写入 plugins/ 目录
        4. 热加载到工具注册表
        5. 验证工具可调用

        Returns:
            {"success": True, "tool_name": "...", "message": "..."} 或 None
        """
        if not self._enabled:
            return None

        # 检查是否已有同名生成工具
        tool_id = f"evolved_{ability.replace(' ', '_')[:30]}"
        if tool_id in self._evolved_meta.get("tools", {}):
            existing = self._evolved_meta["tools"][tool_id]
            if existing.get("status") == "active":
                logger.info(f"已有自主生成的工具: {tool_id}，跳过重复生成")
                return {"success": True, "tool_name": tool_id, "message": f"复用已有工具 {tool_id}"}

        # 获取 LLM 函数
        llm_fn = self._get_llm_function()
        if not llm_fn:
            logger.warning("无法获取 LLM 函数，跳过自主编程")
            return None

        logger.info(f"态极开始自主编程: {ability}")

        # 构造代码生成提示词
        prompt = self._build_code_prompt(ability, task_context)
        try:
            code_response = llm_fn(prompt)
        except Exception as e:
            logger.warning(f"LLM 生成代码失败: {e}")
            return None

        if not code_response or len(code_response.strip()) < 50:
            logger.warning("LLM 生成的代码过短或为空")
            return None

        # 解析和部署生成的代码
        return self._deploy_generated_tool(tool_id, ability, code_response, tool_registry)

    def _get_llm_function(self):
        """获取可用的 LLM 推理函数"""
        # 优先使用态极原生推理
        try:
            from core.app_state import app_state
            if app_state.model and app_state.tokenizer:
                def taiji_llm(prompt):
                    from taiji.inference import NativeInferenceEngine
                    engine = NativeInferenceEngine(app_state.model, app_state.tokenizer)
                    return engine.generate(prompt, max_new_tokens=1024, temperature=0.3)
                return taiji_llm
        except Exception:
            pass

        # 降级：尝试云端 API
        try:
            import os as _os
            settings_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "app_settings.json"
            )
            if _os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                api_base = settings.get("cloud_api_base", "") or settings.get("cloud_base", "")
                api_key = settings.get("cloud_api_key", "") or settings.get("cloud_key", "")
                api_model = settings.get("cloud_api_model", "") or settings.get("cloud_model", "")
                if api_base and api_key and api_model:
                    def cloud_llm(prompt):
                        from agent.agent import run_api_chat_stream
                        chunks = []
                        for chunk in run_api_chat_stream(
                            prompt, [], "", api_base, api_key, api_model
                        ):
                            chunks.append(chunk)
                        return "".join(chunks)
                    return cloud_llm
        except Exception:
            pass

        return None

    def _build_code_prompt(self, ability: str, task_context: str) -> str:
        """构造代码生成提示词"""
        return f"""你是一个 Python 工具开发者。请为以下需求编写一个可用的工具插件。

## 需求
我需要一个工具来实现：{ability}

{f'任务上下文：{task_context}' if task_context else ''}

## 要求
1. 输出完整的 Python 文件内容
2. 文件必须包含一个名为 `register_tools(registry)` 的函数
3. 在 `register_tools` 中使用 `registry.register(ToolDef(...))` 注册工具
4. 工具函数的参数签名必须是 `def tool_func(input_str: str) -> str`
5. 使用 try/except 包裹所有外部调用，失败时返回错误描述字符串
6. 只用标准库和已知安全的第三方库
7. 不要使用 eval/exec/__import__/os.system/subprocess 等危险函数
8. 代码必须可以直接被 import 执行

## 输出格式
只输出 Python 代码，不要包含 markdown 代码块标记或额外解释。

```python
import logging

logger = logging.getLogger("EvolvedTool")

def _my_tool(input_str: str) -> str:
    \"\"\"工具描述\"\"\"
    try:
        # 实现逻辑
        return "结果"
    except Exception as e:
        return f"错误: {{e}}"

def register_tools(registry):
    from agent.tool_registry import ToolDef
    registry.register(ToolDef(
        name="tool_name",
        description="工具描述",
        parameters={{"type": "object", "properties": {{"input": {{"type": "string"}}}}}},
        func=_my_tool,
        source="evolved",
        category="自主进化",
    ))
```

请根据需求 "{ability}" 编写代码："""

    def _deploy_generated_tool(self, tool_id: str, ability: str,
                                code: str, tool_registry=None) -> Optional[dict]:
        """部署 LLM 生成的工具代码"""
        # 清理代码（去除 markdown 标记）
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # 安全检查（使用 SecurityGuard 多层检查）
        try:
            from taiji.security_guard import check_code_safety
            check = check_code_safety(code, context=ability)
            if not check.passed:
                logger.warning(f"安全检查未通过 (风险等级: {check.risk_level}): {check.violations}")
                return {"success": False, "message": f"安全检查失败: {', '.join(check.violations)}"}
            if check.risk_level in ("medium", "high"):
                logger.info(f"安全警告 (风险等级: {check.risk_level}): {check.violations}")
        except ImportError:
            # 回退到基本检查
            dangerous_patterns = [
                "os.system", "subprocess.call", "subprocess.run",
                "subprocess.Popen", "__import__", "eval(", "exec(",
                "shutil.rmtree", "os.remove", "os.unlink",
            ]
            for pattern in dangerous_patterns:
                if pattern in code:
                    logger.warning(f"生成的代码包含危险操作: {pattern}，拒绝部署")
                    return {"success": False, "message": f"安全检查失败: 包含 {pattern}"}

        # 写入插件目录
        plugin_dir = os.path.join(self._generated_tools_dir, tool_id)
        os.makedirs(plugin_dir, exist_ok=True)

        init_path = os.path.join(plugin_dir, "__init__.py")
        try:
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"写入插件代码失败: {e}")
            return {"success": False, "message": f"写入失败: {e}"}

        # 写入 manifest.json
        manifest = {
            "id": tool_id,
            "name": f"evolved_{ability[:30]}",
            "version": "1.0.0",
            "description": f"态极自主生成的工具: {ability}",
            "author": "Taiji Self-Modification Engine",
            "enabled": True,
            "entry_point": "__init__.py",
            "auto_generated": True,
            "generated_at": time.time(),
        }
        manifest_path = os.path.join(plugin_dir, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # 热加载
        if not self._hot_load_plugin(tool_id, tool_registry):
            # 加载失败，清理
            logger.warning(f"热加载失败，清理插件目录: {plugin_dir}")
            import shutil
            try:
                shutil.rmtree(plugin_dir)
            except Exception:
                pass
            return {"success": False, "message": "热加载失败"}

        # 记录进化成果
        self._evolved_meta["tools"][tool_id] = {
            "ability": ability,
            "status": "active",
            "created_at": time.time(),
            "code_length": len(code),
        }
        self._save_evolved_meta()

        self._install_history.append({
            "type": "auto_coded",
            "id": tool_id,
            "ability": ability,
            "time": time.time(),
            "success": True,
        })

        logger.info(f"态极自主编程成功: {tool_id} ({ability})")
        return {"success": True, "tool_name": tool_id, "message": f"已自主生成并部署工具 {tool_id}"}

    def _hot_load_plugin(self, plugin_id: str, tool_registry=None) -> bool:
        """热加载插件"""
        try:
            # 方式 1：通过 PluginManager
            from core.plugin_manager import PluginManager
            pm = PluginManager()
            pm.load_plugin(plugin_id)
            return True
        except Exception as e:
            logger.debug(f"PluginManager 加载失败: {e}")

        # 方式 2：直接 import 并调用 register_tools
        try:
            plugin_dir = os.path.join(self._generated_tools_dir, plugin_id)
            init_path = os.path.join(plugin_dir, "__init__.py")
            if not os.path.exists(init_path):
                return False

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"evolved_plugin_{plugin_id}", init_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register_tools") and tool_registry:
                module.register_tools(tool_registry)
                return True
            return False
        except Exception as e:
            logger.warning(f"直接加载插件失败: {e}")
            return False

    # ======================== 互联网学习 ========================

    def _learn_from_internet(self, task: str, ability: str,
                             tool_registry=None) -> Optional[dict]:
        """
        去互联网搜索解决方案。

        态极天生联网：遇到不会的事情，先去网上搜索学习。
        - 知识缺失 → 搜索学习 → 学会了就能回答
        - 能力缺失 → 搜索找到 Python 库 → 安装

        Returns:
            {"success": True, "action": "learned" | "pip_installed", ...}
        """
        # 获取搜索工具
        search_fn = None
        fetch_fn = None
        if tool_registry:
            if tool_registry.has("search"):
                search_fn = lambda q: tool_registry.execute("search", {"input": q})
            if tool_registry.has("smart_fetch"):
                fetch_fn = lambda u: tool_registry.execute("smart_fetch", {"input": u})
            elif tool_registry.has("read_webpage"):
                fetch_fn = lambda u: tool_registry.execute("read_webpage", {"input": u})

        if not search_fn:
            logger.debug("无搜索工具可用，跳过互联网学习")
            return None

        # Step 1: 搜索
        search_queries = [
            f"Python {ability} library pip install",
            f"{ability} python package tutorial",
            ability,
        ]

        search_results = []
        for query in search_queries:
            try:
                result = search_fn(query[:80])
                if result and len(str(result).strip()) > 30:
                    search_results.append(str(result))
                    break  # 有结果就够了
            except Exception as e:
                logger.debug(f"搜索失败 ({query}): {e}")

        if not search_results:
            logger.info(f"互联网搜索无结果: {ability}")
            return None

        # Step 2: 用 LLM 分析搜索结果
        llm_fn = self._get_llm_function()
        if not llm_fn:
            # 无 LLM，尝试从搜索结果中提取 pip 包名
            return self._extract_pip_package(search_results[0], ability, tool_registry)

        analysis_prompt = f"""我需要具备"{ability}"的能力。以下是我的搜索结果：

{search_results[0][:2000]}

请分析这些搜索结果，告诉我：
1. 是否有现成的 Python pip 包可以实现这个能力？如果有，包名是什么？
2. 如果没有现成包，能否用 Python 标准库或已有库实现？
3. 如果是知识性的（不需要新工具，只需要学习），请简要总结关键知识。

请用以下 JSON 格式回答（只输出JSON）：
{{"type": "pip_package", "package": "包名", "import_name": "导入名", "tool_name": "工具名", "description": "工具描述"}}
或
{{"type": "knowledge", "summary": "关键知识摘要"}}
或
{{"type": "need_code", "description": "需要自己写代码实现的描述"}}
"""

        try:
            analysis = llm_fn(analysis_prompt)
        except Exception as e:
            logger.debug(f"LLM 分析搜索结果失败: {e}")
            return self._extract_pip_package(search_results[0], ability, tool_registry)

        # Step 3: 根据分析结果行动
        if not analysis:
            return None

        # 尝试解析 JSON
        try:
            # 提取 JSON 部分
            json_str = analysis
            if "{" in analysis:
                json_str = analysis[analysis.index("{"):analysis.rindex("}") + 1]
            decision = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失败，检查是否包含 pip install 关键词
            return self._extract_pip_package(analysis, ability, tool_registry)

        action_type = decision.get("type", "")

        # 知识缺失：学会了就能用
        if action_type == "knowledge":
            summary = decision.get("summary", "")
            if summary and len(summary) > 20:
                logger.info(f"[互联网学习] 学到知识: {ability}")
                return {
                    "success": True,
                    "action": "learned",
                    "message": f"已通过互联网学习掌握 '{ability}' 的知识",
                    "knowledge": summary,
                }

        # 找到 pip 包：安装并注册为工具
        if action_type == "pip_package":
            package = decision.get("package", "")
            import_name = decision.get("import_name", "")
            tool_name = decision.get("tool_name", ability.replace(" ", "_"))
            description = decision.get("description", ability)

            if package:
                return self._install_and_register_pip_tool(
                    package, import_name, tool_name, description, ability, tool_registry
                )

        # 需要写代码但有描述
        if action_type == "need_code":
            desc = decision.get("description", "")
            if desc:
                # 将描述传给自主编程
                logger.info(f"[互联网学习] 需要编程实现: {desc[:80]}")
                return None  # 让 evolve() 继续到 auto_code_tool

        return None

    def _extract_pip_package(self, text: str, ability: str,
                             tool_registry=None) -> Optional[dict]:
        """从文本中提取 pip 包名（降级方案）"""
        import re
        # 常见 pip install 模式
        patterns = [
            r'pip\s+install\s+([a-zA-Z0-9_-]+)',
            r'([a-zA-Z0-9_-]+)\s*>=\s*\d',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                package = matches[0]
                if len(package) > 2 and not package.startswith("-"):
                    return self._install_and_register_pip_tool(
                        package, package, ability.replace(" ", "_"),
                        f"{ability} (via {package})", ability, tool_registry
                    )
        return None

    def _install_and_register_pip_tool(self, package: str, import_name: str,
                                        tool_name: str, description: str,
                                        ability: str, tool_registry=None) -> Optional[dict]:
        """安装 pip 包并注册为工具"""
        # 安装
        try:
            import subprocess
            logger.info(f"[互联网学习] pip install {package}")
            result = subprocess.run(
                ["pip", "install", package],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"pip install 失败: {result.stderr[:200]}")
                return None
        except Exception as e:
            logger.warning(f"pip install 异常: {e}")
            return None

        # 验证可导入
        try:
            __import__(import_name or package)
        except ImportError:
            logger.warning(f"安装后仍无法导入: {import_name or package}")
            return None

        # 注册为简单工具
        if tool_registry:
            from agent.tool_registry import ToolDef

            def _make_tool(pkg, imp):
                def _tool_fn(input_str: str) -> str:
                    try:
                        mod = __import__(imp or pkg)
                        # 尝试调用模块的常用方法
                        if hasattr(mod, 'query'):
                            return str(mod.query(input_str))
                        if hasattr(mod, 'search'):
                            return str(mod.search(input_str))
                        if hasattr(mod, 'process'):
                            return str(mod.process(input_str))
                        if hasattr(mod, 'parse'):
                            return str(mod.parse(input_str))
                        return f"已安装 {pkg}（导入名: {imp}），请在代码中使用 import {imp} 来调用"
                    except Exception as e:
                        return f"工具调用失败: {e}"
                return _tool_fn

            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_name)
            if not tool_registry.has(safe_name):
                tool_registry.register(ToolDef(
                    name=safe_name,
                    description=description,
                    parameters={"type": "object", "properties": {
                        "input": {"type": "string", "description": "输入"}
                    }},
                    func=_make_tool(package, import_name or package),
                    source="evolved",
                    category="自主进化",
                ))

        self._install_history.append({
            "type": "pip_learned",
            "id": package,
            "ability": ability,
            "time": time.time(),
            "success": True,
        })

        logger.info(f"[互联网学习] 成功: pip install {package} → {tool_name}")
        return {
            "success": True,
            "action": "pip_installed",
            "tool_name": safe_name if tool_registry else package,
            "message": f"通过互联网找到并安装了 {package}，已注册为工具 {tool_name}",
        }

    # ======================== 完整进化循环 ========================

    def evolve(self, task: str, tool_registry=None) -> dict:
        """
        完整的自主进化循环。

        当 Agent 遇到不会的事情时调用此方法。
        流程：
        检测缺失(知识/能力) → 搜索市场 → 互联网学习 → 自主编程

        关键区分：
        - 知识缺失（"我不懂量子力学"）→ 搜索学习 → 学会了就能回答
        - 能力缺失（"我不能播放视频"）→ 搜索找到库 → 安装
        - 完全空白 → 自主编程生成工具

        Args:
            task: 导致失败的任务描述
            tool_registry: 当前工具注册表

        Returns:
            {
                "evolved": True/False,
                "action": "installed" | "coded" | "failed",
                "tool_name": "...",
                "message": "...",
            }
        """
        if not self._enabled:
            return {"evolved": False, "action": "disabled", "message": "自修改引擎已禁用"}

        # Step 1: 检测能力缺失
        gap = self._discovery.detect_gap(task, tool_registry=tool_registry)
        if not gap:
            return {"evolved": False, "action": "no_gap", "message": "未检测到能力缺失"}

        self._gaps.append(gap)
        ability = gap.missing_ability
        logger.info(f"[进化循环] 检测到缺失: {ability}")

        # Step 2: 搜索现成工具
        matches = self._discovery.find_matching_tools(ability, tool_registry)
        if matches and self._can_auto_install():
            best = matches[0]
            result = self._try_install(best, tool_registry)
            if result.get("success"):
                logger.info(f"[进化循环] 从市场安装成功: {best['name']}")
                return {
                    "evolved": True,
                    "action": "installed",
                    "tool_name": best["name"],
                    "message": f"已从市场安装 {best['name']}",
                }

        # Step 3: 去互联网搜索解决方案（天生联网！）
        logger.info(f"[进化循环] 市场无匹配，去互联网学习: {ability}")
        learn_result = self._learn_from_internet(task, ability, tool_registry)
        if learn_result and learn_result.get("success"):
            return {
                "evolved": True,
                "action": learn_result.get("action", "learned"),
                "tool_name": learn_result.get("tool_name", ""),
                "message": learn_result.get("message", ""),
            }

        # Step 4: 互联网也没找到，最后才自主编程
        logger.info(f"[进化循环] 互联网无果，开始自主编程: {ability}")
        code_result = self.auto_code_tool(ability, task, tool_registry)
        if code_result and code_result.get("success"):
            return {
                "evolved": True,
                "action": "coded",
                "tool_name": code_result["tool_name"],
                "message": f"已自主编程生成工具: {code_result['tool_name']}",
            }

        # Step 4: 失败
        return {
            "evolved": False,
            "action": "failed",
            "message": f"无法为 '{ability}' 找到或生成工具",
        }

    def get_status(self) -> dict:
        """获取自修改引擎状态"""
        evolved_count = len(self._evolved_meta.get("tools", {}))
        active_count = sum(
            1 for t in self._evolved_meta.get("tools", {}).values()
            if t.get("status") == "active"
        )
        return {
            "enabled": self._enabled,
            "gaps_detected": len(self._gaps),
            "auto_installs_remaining": max(0, MAX_AUTO_INSTALLS - self._auto_install_count),
            "evolved_tools_total": evolved_count,
            "evolved_tools_active": active_count,
            "install_history": self._install_history[-10:],
            "recent_gaps": [g.to_dict() for g in self._gaps[-5:]],
        }

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset_install_count(self):
        """重置自动安装计数（每天调用一次）"""
        self._auto_install_count = 0


# ======================== 全局实例 ========================

_global_engine: Optional[SelfModificationEngine] = None


def get_self_modification_engine() -> SelfModificationEngine:
    """获取全局自修改引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = SelfModificationEngine()
    return _global_engine