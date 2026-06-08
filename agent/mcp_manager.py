"""
MCP 服务器管理器 (MCP Manager)
==============================
管理 MCP 服务器的生命周期（安装/启动/停止/卸载），
提供市场浏览功能，并将 MCP 工具注册到统一的工具注册表。

核心职责：
1. 从配置文件加载已安装的 MCP 服务器
2. 管理服务器进程的启动和停止
3. 将 MCP 工具桥接到 Agent 工具注册表
4. 提供市场数据浏览和一键安装
5. 支持从 Cline 配置导入
"""
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from agent.mcp_client import MCPClient, MCPServerConfig, MCPToolInfo, create_client_from_market
from agent.tool_registry import registry, ToolDef

logger = logging.getLogger("MCPManager")


class MCPManager:
    """MCP 服务器管理器"""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._marketplace_data: dict = {}
        self._installed_servers: Dict[str, dict] = {}  # id -> config dict
        self._lock = threading.Lock()
        self._config_path: str = ""
        self._marketplace_path: str = ""

        self._init_paths()
        self._load_marketplace()
        self._load_installed_config()
        self._auto_start_core_servers()

    def _init_paths(self):
        """初始化配置文件路径"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._marketplace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_marketplace.json")
        self._config_path = os.path.join(base_dir, "mcp_servers_config.json")

    # ======================== 自动启动核心 MCP ========================

    # 后台自动启动的 MCP 服务器（提供浏览器自动化和网页抓取能力）
    CORE_MCP_SERVERS = ["playwright", "fetch"]

    def _auto_start_core_servers(self):
        """后台自动安装并启动核心 MCP 服务器（不阻塞主程序）"""
        import threading
        import shutil

        # 检查 npx 是否可用（打包环境中可能不存在）
        if not shutil.which("npx"):
            logger.info("npx 不可用，跳过 MCP 服务器自动启动")
            return

        def _do_start():
            for server_id in self.CORE_MCP_SERVERS:
                try:
                    # 未安装则自动安装
                    if server_id not in self._installed_servers:
                        result = self.install_server(server_id)
                        if result.get("status") not in ("ok", "already_installed"):
                            logger.warning(f"自动安装 MCP 服务器 '{server_id}' 失败: {result.get('message')}")
                            continue
                        logger.info(f"已自动安装 MCP 服务器: {server_id}")

                    # 启动服务器（带超时保护）
                    result = self.start_server(server_id)
                    if result.get("status") in ("ok", "already_running"):
                        tools_count = result.get("tools_count", 0)
                        logger.info(f"已自动启动 MCP 服务器: {server_id} ({tools_count} 个工具)")
                    else:
                        logger.warning(f"自动启动 MCP 服务器 '{server_id}' 失败: {result.get('message')}")
                except Exception as e:
                    logger.warning(f"MCP 服务器 '{server_id}' 自动启动异常: {e}")

        t = threading.Thread(target=_do_start, daemon=True, name="mcp-auto-start")
        t.start()

    def _load_marketplace(self):
        """加载市场数据（本地 + 远程缓存）"""
        local_data = {"version": "1.0.0", "servers": [], "categories": []}
        try:
            if os.path.exists(self._marketplace_path):
                with open(self._marketplace_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
        except Exception as e:
            logger.error(f"加载本地市场数据失败: {e}")

        # 加载远程缓存（如果有）
        remote_cache_path = os.path.join(os.path.dirname(self._marketplace_path), "mcp_remote_cache.json")
        remote_data = {"servers": []}
        try:
            if os.path.exists(remote_cache_path):
                with open(remote_cache_path, "r", encoding="utf-8") as f:
                    remote_data = json.load(f)
        except Exception:
            pass

        # 合并：本地 + 远程（去重，本地优先）
        local_ids = {s["id"] for s in local_data.get("servers", [])}
        merged_servers = list(local_data.get("servers", []))
        for s in remote_data.get("servers", []):
            if s.get("id") not in local_ids:
                merged_servers.append(s)
                local_ids.add(s["id"])

        self._marketplace_data = {
            "version": local_data.get("version", "1.0.0"),
            "categories": local_data.get("categories", ["核心", "开发", "网络", "浏览器", "AI推理", "媒体", "系统", "数据库", "云服务", "效率"]),
            "servers": merged_servers,
            "last_remote_update": remote_data.get("fetch_time", ""),
        }
        logger.info(f"已加载 MCP 市场数据: {len(merged_servers)} 个服务器 (本地 {len(local_data.get('servers', []))}, 远程缓存 {len(remote_data.get('servers', []))})")

    def refresh_marketplace(self) -> dict:
        """从远程源刷新 MCP 市场数据"""
        import threading

        def _do_refresh():
            try:
                servers = self._fetch_remote_mcp_servers()
                if servers:
                    cache_path = os.path.join(os.path.dirname(self._marketplace_path), "mcp_remote_cache.json")
                    import datetime
                    cache_data = {
                        "servers": servers,
                        "fetch_time": datetime.datetime.now().isoformat(),
                        "source": "npm + github",
                        "count": len(servers),
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    # 重新加载合并数据
                    self._load_marketplace()
                    logger.info(f"远程 MCP 市场刷新完成: {len(servers)} 个服务器")
                else:
                    logger.warning("远程 MCP 市场未返回数据")
            except Exception as e:
                logger.error(f"刷新远程 MCP 市场失败: {e}")

        t = threading.Thread(target=_do_refresh, daemon=True, name="mcp-market-refresh")
        t.start()
        return {"status": "refreshing", "message": "正在从远程源刷新 MCP 市场..."}

    def _fetch_remote_mcp_servers(self) -> list:
        """从多个远程源获取 MCP 服务器列表"""
        import urllib.request
        import urllib.parse
        all_servers = []
        seen_packages = set()

        # 源 1: npm 搜索 @modelcontextprotocol 相关包
        try:
            url = "https://registry.npmjs.org/-/v1/search?text=mcp+server+model+context+protocol&size=50"
            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-MCP/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                for obj in data.get("objects", []):
                    pkg = obj.get("package", {})
                    name = pkg.get("name", "")
                    if not name or name in seen_packages:
                        continue
                    if "mcp" not in name.lower() and "model-context" not in name.lower():
                        continue
                    seen_packages.add(name)
                    desc = pkg.get("description", "")
                    version = pkg.get("version", "")
                    author = pkg.get("publisher", {}).get("username", "")
                    links = pkg.get("links", {})
                    all_servers.append({
                        "id": name.replace("/", "-").replace("@", ""),
                        "name": desc[:40] if desc else name,
                        "icon": "📦",
                        "category": self._guess_category(name, desc),
                        "npm_package": name,
                        "description": desc,
                        "command": "cmd",
                        "args": ["/c", "npx", "-y", name],
                        "args_template": [],
                        "tools_count": 0,
                        "rating": 3,
                        "author": author,
                        "version": version,
                        "npm_url": links.get("npm", f"https://www.npmjs.com/package/{name}"),
                        "github_url": links.get("repository", ""),
                        "source": "npm_search",
                        "tags": self._extract_tags(name, desc),
                    })
            logger.info(f"npm 搜索获取 {len(all_servers)} 个 MCP 包")
        except Exception as e:
            logger.warning(f"npm 搜索失败: {e}")

        # 源 2: 搜索 @anthropic、@executeautomation、@upstash 等知名作者的 MCP 包
        known_prefixes = [
            "@modelcontextprotocol/server-",
            "@anthropic/",
            "@executeautomation/",
            "@upstash/",
            "@agentdeskai/",
            "@benborla29/",
            "mcp-",
        ]
        try:
            url = "https://registry.npmjs.org/-/v1/search?text=mcp-server&size=100"
            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-MCP/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                for obj in data.get("objects", []):
                    pkg = obj.get("package", {})
                    name = pkg.get("name", "")
                    if name in seen_packages:
                        continue
                    # 只保留看起来像 MCP 服务器的包
                    is_mcp = any(name.startswith(p) for p in known_prefixes) or "mcp-server" in name or "mcp_server" in name
                    if not is_mcp:
                        continue
                    seen_packages.add(name)
                    desc = pkg.get("description", "")
                    all_servers.append({
                        "id": name.replace("/", "-").replace("@", ""),
                        "name": desc[:40] if desc else name,
                        "icon": "📦",
                        "category": self._guess_category(name, desc),
                        "npm_package": name,
                        "description": desc,
                        "command": "cmd",
                        "args": ["/c", "npx", "-y", name],
                        "args_template": [],
                        "tools_count": 0,
                        "rating": 3,
                        "author": pkg.get("publisher", {}).get("username", ""),
                        "version": pkg.get("version", ""),
                        "npm_url": f"https://www.npmjs.com/package/{name}",
                        "source": "npm_search",
                        "tags": self._extract_tags(name, desc),
                    })
            logger.info(f"npm 补充搜索获取 {len(all_servers) - len(seen_packages)} 个额外 MCP 包")
        except Exception as e:
            logger.warning(f"npm 补充搜索失败: {e}")

        # 源 3: 从 GitHub awesome-mcp-servers 获取
        try:
            url = "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
            req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-MCP/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
                import re
                # 解析 npm 包链接
                npm_links = re.findall(r'https?://(?:www\.)?npmjs\.com/package/([@\w/-]+)', content)
                for pkg_name in npm_links:
                    if pkg_name not in seen_packages and ("mcp" in pkg_name.lower() or "server" in pkg_name.lower()):
                        seen_packages.add(pkg_name)
                        all_servers.append({
                            "id": pkg_name.replace("/", "-").replace("@", ""),
                            "name": pkg_name.split("/")[-1].replace("server-", "").replace("mcp-", "").replace("-", " ").title(),
                            "icon": "⭐",
                            "category": self._guess_category(pkg_name, ""),
                            "npm_package": pkg_name,
                            "description": f"来自 awesome-mcp-servers 社区推荐",
                            "command": "cmd",
                            "args": ["/c", "npx", "-y", pkg_name],
                            "args_template": [],
                            "tools_count": 0,
                            "rating": 4,
                            "author": "community",
                            "source": "awesome_list",
                            "tags": ["社区推荐"],
                        })
            logger.info(f"awesome-mcp-servers 获取 {len(npm_links)} 个包链接")
        except Exception as e:
            logger.warning(f"GitHub awesome-mcp-servers 获取失败: {e}")

        return all_servers

    @staticmethod
    def _guess_category(name: str, desc: str) -> str:
        """根据包名和描述猜测分类"""
        text = (name + " " + desc).lower()
        if any(k in text for k in ["database", "sql", "postgres", "mysql", "redis", "mongo", "sqlite"]):
            return "数据库"
        if any(k in text for k in ["browser", "playwright", "puppeteer", "web", "fetch", "scrape"]):
            return "浏览器"
        if any(k in text for k in ["search", "brave", "google", "wikipedia"]):
            return "网络"
        if any(k in text for k in ["git", "github", "gitlab", "ci", "deploy", "docker", "kubernetes"]):
            return "开发"
        if any(k in text for k in ["memory", "think", "reason", "agent", "chain"]):
            return "AI推理"
        if any(k in text for k in ["file", "fs", "directory", "path"]):
            return "核心"
        if any(k in text for k in ["slack", "discord", "email", "calendar", "notion"]):
            return "效率"
        if any(k in text for k in ["image", "video", "audio", "media", "tts"]):
            return "媒体"
        if any(k in text for k in ["system", "process", "monitor", "log"]):
            return "系统"
        if any(k in text for k in ["aws", "azure", "gcloud", "cloud", "s3"]):
            return "云服务"
        return "开发"

    @staticmethod
    def _extract_tags(name: str, desc: str) -> list:
        """从包名和描述中提取标签"""
        tags = set()
        text = (name + " " + desc).lower()
        tag_keywords = {
            "数据库": ["database", "sql", "db"],
            "搜索": ["search", "query"],
            "文件": ["file", "filesystem"],
            "浏览器": ["browser", "web"],
            "Git": ["git", "version"],
            "API": ["api", "rest", "http"],
            "AI": ["ai", "llm", "agent"],
            "云": ["cloud", "aws", "azure"],
        }
        for tag, keywords in tag_keywords.items():
            if any(k in text for k in keywords):
                tags.add(tag)
        return list(tags)[:5]

    def _load_installed_config(self):
        """加载已安装的 MCP 服务器配置"""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self._installed_servers = config.get("servers", {})
                logger.info(f"已加载 MCP 配置: {len(self._installed_servers)} 个已安装服务器")
        except Exception as e:
            logger.error(f"加载 MCP 配置失败: {e}")
            self._installed_servers = {}

    def _save_installed_config(self):
        """保存已安装的 MCP 服务器配置"""
        try:
            config = {
                "version": "1.0.0",
                "servers": self._installed_servers,
            }
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"已保存 MCP 配置: {len(self._installed_servers)} 个服务器")
        except Exception as e:
            logger.error(f"保存 MCP 配置失败: {e}")

    # ======================== 市场功能 ========================

    def get_marketplace(self, category: str = "", keyword: str = "") -> dict:
        """浏览 MCP 服务器市场"""
        servers = self._marketplace_data.get("servers", [])
        categories = self._marketplace_data.get("categories", [])

        # 按分类筛选
        if category:
            servers = [s for s in servers if s.get("category", "") == category]

        # 按关键词搜索
        if keyword:
            keyword_lower = keyword.lower()
            servers = [s for s in servers if
                       keyword_lower in s.get("name", "").lower() or
                       keyword_lower in s.get("description", "").lower() or
                       keyword_lower in s.get("npm_package", "").lower() or
                       any(keyword_lower in tag for tag in s.get("tags", []))]

        # 标记已安装状态
        result = []
        for s in servers:
            server_info = dict(s)
            server_info["installed"] = s["id"] in self._installed_servers
            server_info["running"] = s["id"] in self._clients and self._clients[s["id"]].is_running()
            # 如果已安装，合并安装时的配置
            if s["id"] in self._installed_servers:
                installed = self._installed_servers[s["id"]]
                server_info["enabled"] = installed.get("enabled", True)
            result.append(server_info)

        return {
            "servers": result,
            "categories": categories,
            "total": len(result),
        }

    def get_server_detail(self, server_id: str) -> Optional[dict]:
        """获取服务器详情"""
        # 从市场查找
        for s in self._marketplace_data.get("servers", []):
            if s["id"] == server_id:
                detail = dict(s)
                detail["installed"] = server_id in self._installed_servers
                detail["running"] = server_id in self._clients and self._clients[server_id].is_running()
                if server_id in self._clients:
                    detail["runtime_info"] = self._clients[server_id].get_server_info()
                return detail
        return None

    # ======================== 安装/卸载 ========================

    def install_server(self, server_id: str, custom_config: dict = None) -> dict:
        """
        安装 MCP 服务器
        从市场数据或自定义配置中安装
        """
        if server_id in self._installed_servers:
            return {"status": "already_installed", "message": f"服务器 '{server_id}' 已安装"}

        config = custom_config or {}

        # 如果没有自定义配置，从市场查找
        if not config:
            market_entry = self._find_market_entry(server_id)
            if not market_entry:
                return {"status": "error", "message": f"未在市场中找到服务器 '{server_id}'"}
            config = market_entry

        # 保存安装配置
        install_record = {
            "id": server_id,
            "name": config.get("name", server_id),
            "command": config.get("command", "cmd"),
            "args": config.get("args", []),
            "args_template": config.get("args_template", []),
            "env": config.get("env", {}),
            "npm_package": config.get("npm_package", ""),
            "description": config.get("description", ""),
            "icon": config.get("icon", "🧩"),
            "category": config.get("category", "通用"),
            "enabled": True,
            "auto_start": False,
        }

        self._installed_servers[server_id] = install_record
        self._save_installed_config()

        logger.info(f"已安装 MCP 服务器: {server_id}")
        return {"status": "ok", "message": f"服务器 '{config.get('name', server_id)}' 安装成功", "config": install_record}

    def uninstall_server(self, server_id: str) -> dict:
        """卸载 MCP 服务器"""
        if server_id not in self._installed_servers:
            return {"status": "error", "message": f"服务器 '{server_id}' 未安装"}

        # 先停止运行
        if server_id in self._clients:
            self.stop_server(server_id)

        # 移除配置
        del self._installed_servers[server_id]
        self._save_installed_config()

        logger.info(f"已卸载 MCP 服务器: {server_id}")
        return {"status": "ok", "message": f"服务器 '{server_id}' 已卸载"}

    def _find_market_entry(self, server_id: str) -> Optional[dict]:
        """从市场数据中查找服务器"""
        for s in self._marketplace_data.get("servers", []):
            if s["id"] == server_id:
                return s
        return None

    # ======================== 启动/停止 ========================

    def start_server(self, server_id: str, workspace_path: str = "") -> dict:
        """启动已安装的 MCP 服务器"""
        if server_id not in self._installed_servers:
            return {"status": "error", "message": f"服务器 '{server_id}' 未安装，请先安装"}

        if server_id in self._clients and self._clients[server_id].is_running():
            return {"status": "already_running", "message": f"服务器 '{server_id}' 已在运行"}

        install_config = self._installed_servers[server_id]

        # 构建启动参数
        args = list(install_config.get("args", []))
        args_template = install_config.get("args_template", [])
        for tmpl in args_template:
            if tmpl == "{workspace_path}":
                args.append(workspace_path or os.getcwd())
            else:
                args.append(tmpl)

        config = MCPServerConfig(
            id=server_id,
            name=install_config.get("name", server_id),
            command=install_config.get("command", "cmd"),
            args=args,
            env=install_config.get("env", {}),
            npm_package=install_config.get("npm_package", ""),
            description=install_config.get("description", ""),
            icon=install_config.get("icon", "🧩"),
            category=install_config.get("category", "通用"),
        )

        client = MCPClient(config)

        with self._lock:
            # 如果有旧的客户端，先清理
            if server_id in self._clients:
                self._clients[server_id].stop()
            self._clients[server_id] = client

        success = client.start()

        if success:
            # 将 MCP 工具注册到工具注册表
            self._register_mcp_tools(server_id, client)
            return {
                "status": "ok",
                "message": f"服务器 '{install_config.get('name', server_id)}' 启动成功",
                "tools": client.get_tool_names(),
                "tools_count": len(client.get_tools()),
            }
        else:
            with self._lock:
                self._clients.pop(server_id, None)
            return {"status": "error", "message": f"服务器 '{server_id}' 启动失败，请检查 npm 是否已安装"}

    def stop_server(self, server_id: str) -> dict:
        """停止 MCP 服务器"""
        if server_id not in self._clients:
            return {"status": "error", "message": f"服务器 '{server_id}' 未在运行"}

        client = self._clients[server_id]
        client.stop()

        # 注销 MCP 工具
        registry.unregister_by_source(server_id)

        with self._lock:
            del self._clients[server_id]

        logger.info(f"已停止 MCP 服务器: {server_id}")
        return {"status": "ok", "message": f"服务器 '{server_id}' 已停止"}

    def restart_server(self, server_id: str, workspace_path: str = "") -> dict:
        """重启 MCP 服务器"""
        if server_id in self._clients:
            self.stop_server(server_id)
        return self.start_server(server_id, workspace_path)

    # ======================== 工具桥接 ========================

    def _register_mcp_tools(self, server_id: str, client: MCPClient):
        """将 MCP 工具注册到 Agent 工具注册表"""
        mcp_tools = client.get_tools()
        registered = 0

        for mcp_tool in mcp_tools:
            # 为每个 MCP 工具创建一个闭包函数
            tool_func = self._create_tool_proxy(client, mcp_tool.name)

            # 转换 MCP input_schema 为标准参数格式
            parameters = mcp_tool.input_schema
            if not parameters:
                parameters = {"type": "object", "properties": {}}

            tool_def = ToolDef(
                name=f"mcp_{server_id}_{mcp_tool.name}",
                description=f"[MCP:{server_id}] {mcp_tool.description}",
                parameters=parameters,
                func=tool_func,
                source="mcp",
                source_id=server_id,
                enabled=True,
                category=client.config.category,
            )

            registry.register(tool_def)
            registered += 1

        logger.info(f"已注册 {registered} 个 MCP 工具 (来源: {server_id})")

    def _create_tool_proxy(self, client: MCPClient, tool_name: str):
        """创建 MCP 工具的代理函数"""
        def proxy(*args, **kwargs):
            # 将参数统一转为字典
            if kwargs:
                arguments = kwargs
            elif args:
                if len(args) == 1 and isinstance(args[0], dict):
                    arguments = args[0]
                elif len(args) == 1:
                    arguments = {"input": str(args[0])}
                else:
                    arguments = {"input": " ".join(str(a) for a in args)}
            else:
                arguments = {}

            return client.call_tool(tool_name, arguments)

        proxy.__name__ = f"mcp_{tool_name}"
        proxy.__doc__ = f"MCP 工具: {tool_name}"
        return proxy

    # ======================== Cline 配置导入 ========================

    def import_from_cline(self, cline_config_path: str = "") -> dict:
        """从 Cline MCP 配置文件导入服务器"""
        if not cline_config_path:
            # 默认 Cline 配置路径
            cline_config_path = os.path.join(
                os.path.expanduser("~"),
                "AppData", "Roaming", "Code", "User", "globalStorage",
                "saoudrizwan.claude-dev", "settings", "cline_mcp_settings.json"
            )

        if not os.path.exists(cline_config_path):
            return {"status": "error", "message": f"Cline 配置文件不存在: {cline_config_path}"}

        try:
            with open(cline_config_path, "r", encoding="utf-8") as f:
                cline_config = json.load(f)

            mcp_servers = cline_config.get("mcpServers", {})
            imported = 0

            for server_id, server_config in mcp_servers.items():
                if server_config.get("disabled", False):
                    continue

                # 检查是否已安装
                if server_id in self._installed_servers:
                    continue

                # 从市场查找匹配信息
                market_entry = self._find_market_entry_by_npm(
                    self._extract_npm_package(server_config.get("args", []))
                )

                install_config = {
                    "id": server_id,
                    "name": market_entry["name"] if market_entry else server_id.split("/")[-1],
                    "command": server_config.get("command", "cmd"),
                    "args": server_config.get("args", []),
                    "args_template": [],
                    "env": server_config.get("env", {}),
                    "npm_package": self._extract_npm_package(server_config.get("args", [])),
                    "description": market_entry["description"] if market_entry else f"从 Cline 导入: {server_id}",
                    "icon": market_entry["icon"] if market_entry else "🧩",
                    "category": market_entry["category"] if market_entry else "导入",
                    "enabled": True,
                    "auto_start": False,
                }

                self._installed_servers[server_id] = install_config
                imported += 1

            self._save_installed_config()

            return {
                "status": "ok",
                "message": f"从 Cline 导入了 {imported} 个 MCP 服务器",
                "imported": imported,
                "total_in_cline": len(mcp_servers),
            }

        except Exception as e:
            return {"status": "error", "message": f"导入失败: {e}"}

    def _extract_npm_package(self, args: list) -> str:
        """从命令参数中提取 npm 包名"""
        for arg in args:
            if arg.startswith("@") or (not arg.startswith("-") and not arg.startswith("/") and not arg.startswith("npx")):
                if "/" in arg or arg.startswith("@"):
                    # 去掉版本号
                    return arg.split("@")[0] + ("@" + arg.split("@")[1] if "@" in arg[1:] else "")
        # 找 -y 后面的参数
        for i, arg in enumerate(args):
            if arg == "-y" and i + 1 < len(args):
                return args[i + 1]
        return ""

    def _find_market_entry_by_npm(self, npm_package: str) -> Optional[dict]:
        """通过 npm 包名查找市场条目"""
        if not npm_package:
            return None
        for s in self._marketplace_data.get("servers", []):
            if s.get("npm_package", "") == npm_package:
                return s
        return None

    # ======================== 状态查询 ========================

    def get_installed_servers(self) -> list:
        """获取所有已安装的服务器"""
        result = []
        for server_id, config in self._installed_servers.items():
            info = dict(config)
            info["installed"] = True
            info["running"] = server_id in self._clients and self._clients[server_id].is_running()
            if server_id in self._clients:
                info["runtime_info"] = self._clients[server_id].get_server_info()
            result.append(info)
        return result

    def get_running_servers(self) -> list:
        """获取所有运行中的服务器"""
        result = []
        for server_id, client in self._clients.items():
            if client.is_running():
                result.append(client.get_server_info())
        return result

    def get_all_mcp_tools(self) -> list:
        """获取所有 MCP 工具"""
        tools = registry.list_tools(source="mcp")
        return [t.to_info() for t in tools]

    def get_status(self) -> dict:
        """获取 MCP 管理器整体状态"""
        return {
            "marketplace_count": len(self._marketplace_data.get("servers", [])),
            "installed_count": len(self._installed_servers),
            "running_count": sum(1 for c in self._clients.values() if c.is_running()),
            "mcp_tools_count": len(registry.list_tools(source="mcp")),
            "installed_servers": self.get_installed_servers(),
        }

    # ======================== 自定义服务器 ========================

    def add_custom_server(self, server_id: str, name: str, command: str, args: list,
                          npm_package: str = "", description: str = "", env: dict = None) -> dict:
        """添加自定义 MCP 服务器"""
        config = {
            "command": command,
            "args": args,
            "name": name,
            "npm_package": npm_package,
            "description": description,
            "icon": "➕",
            "category": "自定义",
            "env": env or {},
        }
        return self.install_server(server_id, config)

    # ======================== 清理 ========================

    def shutdown_all(self):
        """停止所有运行中的 MCP 服务器"""
        for server_id in list(self._clients.keys()):
            try:
                self.stop_server(server_id)
            except Exception as e:
                logger.error(f"停止服务器 {server_id} 失败: {e}")
        logger.info("所有 MCP 服务器已停止")

    # ======================== Agent 插件市场 ========================

    def get_plugin_marketplace(self, category: str = "", keyword: str = "") -> dict:
        """浏览 Agent 插件市场"""
        # 加载本地插件市场数据
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin_marketplace.json")
        local_data = {"plugins": [], "categories": []}
        try:
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
        except Exception:
            pass

        # 加载远程缓存
        remote_path = os.path.join(os.path.dirname(self._marketplace_path), "plugin_remote_cache.json")
        remote_data = {"plugins": []}
        try:
            if os.path.exists(remote_path):
                with open(remote_path, "r", encoding="utf-8") as f:
                    remote_data = json.load(f)
        except Exception:
            pass

        # 合并去重
        local_ids = {p["id"] for p in local_data.get("plugins", [])}
        merged = list(local_data.get("plugins", []))
        for p in remote_data.get("plugins", []):
            if p.get("id") not in local_ids:
                merged.append(p)
                local_ids.add(p["id"])

        # 筛选
        if category:
            merged = [p for p in merged if p.get("category") == category]
        if keyword:
            kw = keyword.lower()
            merged = [p for p in merged if kw in p.get("name", "").lower() or kw in p.get("description", "").lower()]

        return {
            "plugins": merged,
            "categories": local_data.get("categories", []),
            "total": len(merged),
            "last_remote_update": remote_data.get("fetch_time", ""),
        }

    def refresh_plugin_marketplace(self) -> dict:
        """从远程源刷新 Agent 插件市场"""
        import threading

        def _do_refresh():
            try:
                plugins = self._fetch_remote_plugins()
                if plugins:
                    cache_path = os.path.join(os.path.dirname(self._marketplace_path), "plugin_remote_cache.json")
                    import datetime
                    cache_data = {
                        "plugins": plugins,
                        "fetch_time": datetime.datetime.now().isoformat(),
                        "count": len(plugins),
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"远程插件市场刷新完成: {len(plugins)} 个插件")

            except Exception as e:
                logger.error(f"刷新远程插件市场失败: {e}")

        t = threading.Thread(target=_do_refresh, daemon=True, name="plugin-market-refresh")
        t.start()
        return {"status": "refreshing", "message": "正在从远程源刷新插件市场..."}

    def _fetch_remote_plugins(self) -> list:
        """从远程源获取 Agent 插件列表"""
        import urllib.request
        all_plugins = []
        seen = set()

        # 源 1: 从 GitHub awesome-claude-plugins 等获取
        sources = [
            ("https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md", "awesome_mcp"),
        ]

        for url, source_name in sources:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "OmniCore-Agent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode("utf-8")
                    import re
                    # 提取 GitHub 仓库链接
                    repos = re.findall(r'https://github\.com/([\w-]+/[\w-]+)', content)
                    for repo in repos[:30]:  # 限制数量
                        if repo not in seen and any(k in repo.lower() for k in ["plugin", "tool", "agent", "mcp"]):
                            seen.add(repo)
                            all_plugins.append({
                                "id": repo.replace("/", "-"),
                                "name": repo.split("/")[-1].replace("-", " ").title(),
                                "icon": "🔗",
                                "category": "开发工具",
                                "description": f"来自 GitHub: {repo}",
                                "github": f"https://github.com/{repo}",
                                "tools": [],
                                "author": repo.split("/")[0],
                                "source": source_name,
                            })
                logger.info(f"从 {source_name} 获取 {len(all_plugins)} 个插件")
            except Exception as e:
                logger.warning(f"获取 {source_name} 失败: {e}")

        return all_plugins

    def __del__(self):
        try:
            self.shutdown_all()
        except Exception:
            pass


# ======================== 全局单例 ========================

mcp_manager = MCPManager()