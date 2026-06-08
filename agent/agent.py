"""
Agent 模式模块 - 增强版（精简调度层）
集成搜索引擎、网页爬虫、安全 Python 沙箱执行器
支持本地模型和云端 API 两种引擎

主要工具函数已拆分到：
  - agent_tools.py: 文件系统操作 + 项目脚手架 + 代码分析
  - agent_planner.py: 任务规划系统 + 开发上下文管理
  - sandbox_executor.py: Python 代码安全沙箱执行
"""
import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict
import importlib.util
from core.utils import get_external_path

logger = logging.getLogger("Agent")

# 导入安全沙箱
from agent.sandbox_executor import execute_python_code_safe as execute_python_code

# 导入拆分后的工具
from agent.agent_tools import (
    read_local_file, write_file, edit_file, delete_file,
    list_directory, create_directory, create_project,
    install_dependency, analyze_code,
)
from agent.agent_planner import (
    create_plan, update_plan, get_plan, list_plans,
    save_context, load_context,
)

try:
    from agent.run_exe import TOOLS as exe_tools
except ImportError:
    exe_tools = []

try:
    from tools.bilibili_subtitle import TOOLS as bili_tools
except ImportError:
    bili_tools = []


def run_command(input_str: str) -> str:
    """运行命令行命令（白名单安全限制）。"""
    try:
        from agent.run_exe import run_local_program
        return run_local_program(input_str)
    except Exception as e:
        return f"❌ 命令执行失败: {e}"


# ======================== 已存在的工具 ========================

def read_webpage(url: str) -> str:
    """通用网页正文提取"""
    try:
        import requests
        from bs4 import BeautifulSoup
        if not url.startswith("http"):
            return "错误: 输入必须是包含 http 的有效网址"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        text = soup.get_text(separator="\n", strip=True)
        return text[:3000] + "\n...(文章过长已截断)"
    except Exception as e:
        return f"读取网页失败: {e}"


# ======================== 插件系统 ========================

def load_dynamic_tools(ui_settings, app_state):
    """从 plugins 目录动态加载工具（支持热重载）"""
    dynamic_tools = []
    plugins_dir = get_external_path("plugins")

    if not os.path.exists(plugins_dir):
        os.makedirs(plugins_dir, exist_ok=True)
        sample_plugin = os.path.join(plugins_dir, "sample_tool.py")
        with open(sample_plugin, "w", encoding="utf-8") as f:
            f.write('"""\nOmniCore Agent 插件示例\n将任意 .py 文件放入 plugins 目录，系统会在每次运行 Agent 时自动热加载。\n"""\nfrom langchain_core.tools import Tool\n\ndef get_current_weather(location: str) -> str:\n    """这是一个演示函数"""\n    return f"{location} 今天天气晴朗，气温 25°C，适合出门。"\n\n# 将你想暴露给 Agent 的工具放到 TOOLS 列表中\nTOOLS = [\n    Tool(\n        name="get_weather",\n        description="获取指定城市的天气情况。输入必须是一个城市的名称，例如：北京",\n        func=get_current_weather,\n    )\n]\n')

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            filepath = os.path.join(plugins_dir, filename)
            module_name = f"plugins_{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "TOOLS"):
                    dynamic_tools.extend(module.TOOLS)
                elif hasattr(module, "get_tools"):
                    dynamic_tools.extend(module.get_tools(ui_settings, app_state))
            except Exception as e:
                logger.error(f"加载插件 {filename} 失败: {e}")

    return dynamic_tools


def _build_developer_prompt(today: str, is_local_model: bool) -> str:
    """构建面向独立开发任务的 ReAct Prompt。"""
    return f"""你是一个具备独立软件开发能力的全能AI助手。
当前系统时间：{today}

## 🎯 核心能力
你可以在工作台（agent_workspace 目录）中独立完成软件开发任务：
1. **任务规划**：自动分解复杂任务，创建执行计划并逐步推进
2. **文件操作**：创建、编辑、读取、删除项目文件
3. **代码执行**：运行 Python 代码验证逻辑
4. **项目创建**：从脚手架模板创建完整项目
5. **依赖管理**：安装所需的第三方包
6. **调试修复**：分析代码错误并自动修正
7. **上下文管理**：跨步骤保持开发状态和上下文

## 📋 工作方法论
当收到复杂开发任务时，请遵循以下方法论：

### Step 1: 理解需求
- 确认任务目标和约束条件
- 确定需要使用的技术栈

### Step 2: 创建计划
- 使用 `create_plan` 工具将任务分解为可执行的子步骤
- 每个步骤应该是可验证的（完成后能确认是否成功）

### Step 3: 逐步执行
- 每一步执行后，使用 `update_plan` 更新进度
- 如果遇到错误，分析原因并修复后继续
- 使用 `save_context` 保存重要信息（项目名、当前文件等）

### Step 4: 验证测试
- 代码完成后使用 `execute_python` 运行测试
- 使用 `analyze_code` 检查语法错误

### Step 5: 完成总结
- 所有步骤完成后，向用户展示最终成果

## 🛠️ 可用工具
{{tools}}

## 📏 格式要求
必须严格按照以下格式思考和执行：

Question: {{input}}
Thought: 分析当前步骤并决策下一步
Action: 选择 [{{{tool_names}}}] 中的一个工具
Action Input: 传递给工具的具体参数（必须完整、准确）
Observation: 工具的返回结果
...（可重复 Thought/Action/Observation 步骤，直到任务完成）
Thought: 任务完成
Final Answer: 给用户的中文回复，包含完成总结和成果说明

## ⚠️ 重要规则
- 复杂任务必须先创建计划，不要试图一步完成
- 每个文件创建/修改后，验证其内容正确性
- 遇到错误不要放弃，分析错误信息并修复
- 使用 save_context 记录关键状态（如当前项目的文件结构）
- 优先考虑代码质量和可维护性
- 完成任务后必须更新计划状态为 done
"""


# ======================== Agent 执行器 ========================

def run_agent(prompt: str, engine_choice: str, api_base: str, api_key: str,
              api_model: str, search_engine_choice: str, search_api_key_val: str,
              search_cx_val: str, ui_settings: dict, app_state: dict, tokenizer,
              history: list = None):
    """
    Agent 模式主逻辑（增强版）
    使用 ReAct 框架 + LangChain 工具链
    支持独立开发任务规划、文件操作、代码执行等能力
    """
    try:
        from langchain_core.tools import Tool
        from langchain_core.prompts import PromptTemplate

        # 兼容 langchain 1.x（AgentExecutor 已移除）
        try:
            from langchain.agents import AgentExecutor, create_react_agent
        except ImportError:
            from langgraph.prebuilt import create_react_agent as _create_react_agent

            class _AgentExecutorCompat:
                """将 langgraph 的 create_react_agent 包装为 AgentExecutor 兼容接口"""
                def __init__(self, agent, tools, verbose=False, handle_parsing_errors=True,
                             max_iterations=15, max_execution_time=300, **kw):
                    self._agent = agent
                    self._tools = tools
                    self._max_iterations = max_iterations

                def stream(self, inputs):
                    """模拟 AgentExecutor.stream 的输出格式"""
                    user_input = inputs.get("input", "")
                    from langchain_core.messages import HumanMessage
                    config = {"recursion_limit": self._max_iterations * 2}
                    try:
                        for event in self._agent.stream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                        ):
                            # langgraph 事件格式转换为 AgentExecutor 格式
                            if "agent" in event:
                                msgs = event["agent"].get("messages", [])
                                for msg in msgs:
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            yield {"actions": [_ActionCompat(tc["name"], tc["args"])]}
                                    elif hasattr(msg, "content") and msg.content:
                                        yield {"output": msg.content}
                            elif "tools" in event:
                                msgs = event["tools"].get("messages", [])
                                for msg in msgs:
                                    tool_name = getattr(msg, "name", "unknown")
                                    content = getattr(msg, "content", str(msg))
                                    yield {"steps": [(_ActionCompat(tool_name, {}), content)]}
                    except Exception as e:
                        yield {"output": f"Agent 执行出错: {e}"}

            class _ActionCompat:
                def __init__(self, tool, tool_input):
                    self.tool = tool
                    self.tool_input = tool_input
                    self.log = f"使用工具: {tool}"

            # 伪造 create_react_agent 供后续代码使用
            def create_react_agent(llm, tools, prompt_template):
                system_msg = prompt_template.template if hasattr(prompt_template, 'template') else str(prompt_template)
                return _create_react_agent(model=llm, tools=tools, prompt=system_msg)

            AgentExecutor = _AgentExecutorCompat

        robust_search = _create_robust_search(
            search_engine_choice, ui_settings,
            search_api_key_val, search_cx_val,
        )

        # ==================== 工具注册 ====================
        tools = [
            Tool(name="search", description="在互联网上搜索最新信息。输入简短的搜索关键词。", func=robust_search),
            Tool(name="read_webpage", description="深入阅读指定网址的网页正文。输入完整 URL。", func=read_webpage),
            Tool(name="execute_python", description="执行 Python 代码。输入必须是纯 Python 代码。", func=execute_python_code),
            Tool(name="read_local_file", description="读取工作台文件内容，支持分页。", func=read_local_file),
            Tool(name="write_file", description="在工作台中创建或覆盖写入文件。", func=write_file),
            Tool(name="edit_file", description="精确编辑工作台中的文件内容。", func=edit_file),
            Tool(name="delete_file", description="删除工作台中的文件或空目录。", func=delete_file),
            Tool(name="list_directory", description="列出工作台目录内容。", func=list_directory),
            Tool(name="create_directory", description="在工作台中创建目录。", func=create_directory),
            Tool(name="create_project", description="创建完整项目脚手架。支持: python-script, web-app, vue-app 等。", func=create_project),
            Tool(name="install_dependency", description="安装 Python 依赖包。", func=install_dependency),
            Tool(name="run_command", description="运行本地命令行命令（安全白名单）。", func=run_command),
            Tool(name="analyze_code", description="分析代码文件语法。支持 .py, .js, .json。", func=analyze_code),
            Tool(name="create_plan", description="为复杂开发任务创建执行计划。", func=create_plan),
            Tool(name="update_plan", description="更新计划步骤状态。", func=update_plan),
            Tool(name="get_plan", description="获取任务计划进度。", func=get_plan),
            Tool(name="save_context", description="保存开发上下文信息。", func=save_context),
            Tool(name="load_context", description="读取已保存的上下文信息。", func=load_context),
        ]
        tools.extend(exe_tools)
        tools.extend(bili_tools)
        tools.extend(load_dynamic_tools(ui_settings, app_state))

        today = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
        is_local_model = "local" in (engine_choice or "").lower() or "本地" in (engine_choice or "")
        template = _build_developer_prompt(today, is_local_model)
        prompt_template = PromptTemplate.from_template(template)

        # ==================== LLM 初始化 ====================
        if is_local_model:
            from langchain.llms.base import LLM
            class LocalLLM(LLM):
                @property
                def _llm_type(self):
                    return "local"
                def _call(self, prompt_text, stop=None, **kwargs):
                    trainer = app_state.get_trainer() if hasattr(app_state, 'get_trainer') else app_state.get("trainer")
                    out = trainer.generate(prompt_text, tokenizer, max_new_tokens=2048)
                    if stop:
                        for s in stop:
                            if s in out:
                                out = out.split(s)[0]
                    return out.strip()
            llm = LocalLLM()
        else:
            from langchain_openai import ChatOpenAI
            if not api_key:
                return "❌ 请先在设置中填写 API Key！"
            llm = ChatOpenAI(model=api_model, api_key=api_key, base_url=api_base, temperature=0.7, max_tokens=4096)

        # ==================== Agent 执行器 ====================
        agent = create_react_agent(llm, tools, prompt_template)
        task_complexity = len(prompt)
        max_iter = 40 if task_complexity > 200 else (25 if task_complexity > 100 else 15)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True,
            handle_parsing_errors=True, max_iterations=max_iter,
            max_execution_time=300, early_stopping_method="generate",
        )

        # ==================== 执行环境 ====================
        import glob as glob_module
        from agent.agent_tools import _get_workspace as get_ws
        sandbox = Path(get_ws())
        sandbox.mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()

        before_images = set(glob_module.glob("*.png") + glob_module.glob("*.jpg") + glob_module.glob("*.jpeg"))

        final_prompt = prompt
        if history:
            history_text = "\n".join([f"用户: {u}\n助手: {a}" for u, a in history[-5:]])
            final_prompt = f"【历史聊天记录】\n{history_text}\n\n【当前用户指令】\n{prompt}\n（请结合上述历史聊天记录来理解当前指令，如果不需要关联则直接回答）"

        try:
            for chunk in agent_executor.stream({"input": final_prompt}):
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        yield f"🤔 **思考与计划**：\n```text\n{action.log.strip()}\n```\n\n"
                elif "steps" in chunk:
                    from agent.token_optimizer import truncate_tool_result
                    for action, observation in chunk["steps"]:
                        obs = truncate_tool_result(str(observation), action.tool)
                        yield f"🔧 **观测结果 ({action.tool})**：\n```text\n{obs}\n```\n\n"
                elif "output" in chunk:
                    ans = chunk['output']
                    if "iteration limit" in ans.lower() or "max iterations" in ans.lower():
                        ans = ("⚠️ 任务步骤较多，已到达迭代上限。\n\n"
                               "✅ **已完成的进度**：使用 `get_plan all` 查看当前计划状态。\n"
                               "👉 如需继续，请告诉我下一步要做什么，或重新输入简化指令。")
                    yield f"🎯 **最终解答**：\n{ans}\n"

            after_images = set(glob_module.glob("*.png") + glob_module.glob("*.jpg") + glob_module.glob("*.jpeg"))
            for img in after_images - before_images:
                img_path = sandbox / img
                try:
                    import base64
                    encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    ext = "jpeg" if img.lower().endswith("jpg") else img.lower().split(".")[-1]
                    yield f"\n\n![{img}](data:image/{ext};base64,{encoded})"
                except Exception as e:
                    logger.warning(f"图片嵌入失败 ({img}): {e}")
        finally:
            pass

    except Exception as e:
        err = str(e).lower()
        if "api key" in err:
            yield f"❌ Agent 联网失败: {e}\n\n👉 请检查 API Key 设置！"
        elif "no module" in err:
            yield f"❌ Agent 依赖缺失: {e}\n\n👉 pip install 安装缺失包。"
        elif "iteration limit" in err or "max iterations" in err:
            yield "⚠️ 任务步骤较多，已到达迭代上限。请简化问题或分步骤提问。"
        else:
            yield f"❌ Agent 运行失败: {e}"


# ======================== 搜索引擎 ========================

def _create_robust_search(search_engine_choice, ui_settings,
                          search_api_key_val="", search_cx_val=""):
    """创建多引擎搜索闭包"""

    # 标准化搜索引擎名称（兼容中英文）
    _ENGINE_ALIASES = {
        "smart-multi": "智能多核",
        "multi": "智能多核",
        "bing": "必应",
        "baidu": "Baidu",
        "duckduckgo": "DuckDuckGo",
        "serper": "Serper",
        "tavily": "Tavily",
    }
    choice_lower = (search_engine_choice or "").strip().lower()
    search_engine_choice = _ENGINE_ALIASES.get(choice_lower, search_engine_choice or "")

    def robust_search(query: str) -> str:
        def serper(q):
            key = search_api_key_val or ui_settings.get("search_profiles", {}).get("Serper API", {}).get("key", "")
            if not key:
                return None
            from langchain_community.utilities import GoogleSerperAPIWrapper
            return GoogleSerperAPIWrapper(serper_api_key=key).run(q)

        def tavily(q):
            key = search_api_key_val or ui_settings.get("search_profiles", {}).get("Tavily API", {}).get("key", "")
            if not key:
                return None
            os.environ["TAVILY_API_KEY"] = key
            from langchain_community.tools.tavily_search import TavilySearchResults
            return str(TavilySearchResults().run(q))

        def ddg(q):
            from langchain_community.tools import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun().run(q)

        def bing(q):
            import urllib.parse, requests
            from bs4 import BeautifulSoup
            url = f"https://www.bing.com/search?q={urllib.parse.quote(q)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
            try:
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, "html.parser")
                results = []
                for item in soup.select(".b_algo"):
                    title_el = item.select_one("h2")
                    desc_el = item.select_one(".b_caption p")
                    title = title_el.text.strip() if title_el else ""
                    desc = desc_el.text.strip() if desc_el else ""
                    if title or desc:
                        results.append(f"{title}: {desc}" if desc else title)
                return "\n\n".join(results[:5]) if results else None
            except Exception:
                return None

        def baidu(q):
            import urllib.parse, requests
            from bs4 import BeautifulSoup
            url = f"https://www.baidu.com/s?wd={urllib.parse.quote(q)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
            try:
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, "html.parser")
                results = []
                for item in soup.select(".result.c-container"):
                    title = item.select_one(".t")
                    desc = item.select_one(".c-abstract")
                    if title and desc:
                        results.append(title.text.strip() + ": " + desc.text.strip())
                return "\n\n".join(results[:5]) if results else None
            except Exception:
                return None

        engines = {"Serper": serper, "Tavily": tavily,
                    "DuckDuckGo": ddg, "必应": bing, "Baidu": baidu}

        # 智能多核：依次尝试所有引擎
        if "智能多核" in search_engine_choice:
            errors = []
            for name, func in engines.items():
                try:
                    res = func(query)
                    if res and len(res.strip()) > 10:
                        return res
                    else:
                        errors.append(f"{name}: 结果为空")
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    logger.debug(f"引擎 {name} 失败: {e}")
            return "所有搜索引擎均失败。\n" + "\n".join(errors)

        # 单引擎匹配
        for name, func in engines.items():
            if name in search_engine_choice:
                try:
                    result = func(query)
                    return result or f"{name} 未返回结果。"
                except Exception as e:
                    return f"{name} 搜索失败: {e}"
        return f"未找到匹配的搜索引擎配置: {search_engine_choice}"

    return robust_search


def run_api_chat(prompt: str, history: list, system_prompt: str,
                 api_base: str, api_key: str, api_model: str) -> str:
    """云端 API 对话（非流式）"""
    import requests
    from agent.token_optimizer import (
        compute_dynamic_max_tokens, compress_history,
        estimate_messages_tokens, get_tracker,
    )
    if not api_key:
        return "❌ 请先填写 API Key！"
    
    # 动态 max_tokens: 根据问题复杂度自动调整
    max_tokens = compute_dynamic_max_tokens(prompt)
    
    # 压缩历史: 超过 3 轮时截断早期对话
    compressed_history = compress_history(history, max_rounds=3, max_chars_per_round=400)
    
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in compressed_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})
    
    payload = {"model": api_model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        # 记录 token 使用
        usage = result.get("usage", {})
        if usage:
            get_tracker().record(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                model=api_model,
                endpoint="chat",
            )
        return content
    return f"❌ API 请求失败: HTTP {response.status_code}\n{response.text}"


def run_api_chat_stream(prompt: str, history: list, system_prompt: str,
                        api_base: str, api_key: str, api_model: str):
    """云端 API 对话（流式生成）"""
    import requests
    from agent.token_optimizer import (
        compute_dynamic_max_tokens, compress_history,
        estimate_messages_tokens, estimate_tokens, get_tracker,
    )
    if not api_key:
        yield "❌ 请先填写 API Key！"
        return
    
    # 动态 max_tokens
    max_tokens = compute_dynamic_max_tokens(prompt)
    
    # 压缩历史
    compressed_history = compress_history(history, max_rounds=3, max_chars_per_round=400)
    
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in compressed_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})
    
    # 估算输入 token 数
    input_tokens_est = estimate_messages_tokens(messages)
    
    payload = {"model": api_model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7, "stream": True}
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(30, 300))
        if response.status_code != 200:
            yield f"❌ API 请求失败: HTTP {response.status_code}\n{response.text}"
            return
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                except Exception:
                    continue
    except requests.exceptions.Timeout:
        yield "\n\n❌ API 请求超时，请检查网络或更换模型。"
    except Exception as e:
        yield f"\n\n❌ API 流式请求失败: {e}"


def run_anthropic_chat(prompt: str, history: list, system_prompt: str,
                       api_base: str, api_key: str, api_model: str) -> str:
    """Anthropic 兼容 API 对话（非流式）"""
    import requests
    from agent.token_optimizer import (
        compute_dynamic_max_tokens, compress_history,
        estimate_messages_tokens, get_tracker,
    )
    if not api_key:
        return "❌ 请先填写 API Key！"

    max_tokens = compute_dynamic_max_tokens(prompt)
    compressed_history = compress_history(history, max_rounds=3, max_chars_per_round=400)

    messages = []
    for user_msg, bot_msg in compressed_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": messages,
    }
    url = api_base.rstrip("/") + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        result = response.json()
        content = result["content"][0]["text"]
        usage = result.get("usage", {})
        if usage:
            get_tracker().record(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model=api_model, endpoint="chat",
            )
        return content
    return f"❌ API 请求失败: HTTP {response.status_code}\n{response.text}"


def run_anthropic_chat_stream(prompt: str, history: list, system_prompt: str,
                              api_base: str, api_key: str, api_model: str):
    """Anthropic 兼容 API 对话（流式生成）"""
    import requests
    from agent.token_optimizer import (
        compute_dynamic_max_tokens, compress_history,
        estimate_messages_tokens, get_tracker,
    )
    if not api_key:
        yield "❌ 请先填写 API Key！"
        return

    max_tokens = compute_dynamic_max_tokens(prompt)
    compressed_history = compress_history(history, max_rounds=3, max_chars_per_round=400)

    messages = []
    for user_msg, bot_msg in compressed_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": messages,
        "stream": True,
    }
    url = api_base.rstrip("/") + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(30, 300))
        if response.status_code != 200:
            yield f"❌ API 请求失败: HTTP {response.status_code}\n{response.text}"
            return
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            try:
                event = json.loads(data)
                event_type = event.get("type", "")
                if event_type == "content_block_delta":
                    text = event.get("delta", {}).get("text", "")
                    if text:
                        yield text
                elif event_type == "message_stop":
                    break
                elif event_type == "error":
                    error_msg = event.get("error", {}).get("message", "未知错误")
                    yield f"\n\n❌ API 错误: {error_msg}"
                    break
            except Exception:
                continue
    except requests.exceptions.Timeout:
        yield "\n\n❌ API 请求超时，请检查网络或更换模型。"
    except Exception as e:
        yield f"\n\n❌ API 流式请求失败: {e}"
