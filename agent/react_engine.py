"""
Agent ReAct 推理引擎
====================
实现 Thought -> Action -> Observation 自主推理循环。
支持本地模型和云端 API 双模式。
"""
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger("ReActEngine")


@dataclass
class AgentStep:
    step_number: int
    thought: str = ""
    action: str = ""
    action_args: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class AgentResult:
    task: str
    final_answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0
    status: str = "completed"


class ToolCallParser:
    """多策略工具调用解析器

    解析策略优先级:
    1. 原生 tool_calls（云端 API 返回）
    2. JSON 代码块 → ```json {...} ```
    3. JSON 行内 → {"tool": "...", "args": {...}}
    4. <tool_call> 标签
    5. Action 格式 → Action: tool_name(args)
    6. 尝试修复不完整 JSON
    """

    def parse(self, content: str, tool_calls_raw: list = None,
              available_tools: list = None) -> list:
        """多策略解析工具调用"""
        # 策略 1: 原生 tool_calls
        if tool_calls_raw:
            return self._parse_native_calls(tool_calls_raw)

        if not content:
            return []

        # 策略 2: JSON 代码块
        results = self._extract_json_blocks(content)
        if results:
            return self._validate_all(results, available_tools)

        # 策略 3: <tool_call> 标签
        results = self._extract_xml_tool_calls(content)
        if results:
            return self._validate_all(results, available_tools)

        # 策略 4: 行内 JSON
        results = self._extract_inline_json(content)
        if results:
            return self._validate_all(results, available_tools)

        # 策略 5: Action 格式
        results = self._extract_action_format(content)
        if results:
            return self._validate_all(results, available_tools)

        # 策略 6: 尝试修复不完整 JSON
        results = self._try_repair_json(content)
        if results:
            return self._validate_all(results, available_tools)

        return []

    def _parse_native_calls(self, tool_calls_raw: list) -> list:
        """解析原生 tool_calls"""
        results = []
        for tc in tool_calls_raw:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if name:
                results.append({"name": name, "arguments": args})
        return results

    def _extract_json_blocks(self, text: str) -> list:
        """从 ```json ... ``` 代码块和行内 JSON 中提取"""
        import re
        results = []
        # 代码块
        for match in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, list):
                    results.extend(self._normalize(data))
                elif isinstance(data, dict):
                    results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        if results:
            return results
        # 单个 JSON 对象
        for match in re.finditer(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL):
            try:
                data = json.loads(match.group(0))
                results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        return results

    def _extract_xml_tool_calls(self, text: str) -> list:
        """从 <tool_call>...</tool_call> 标签中提取"""
        import re
        results = []
        for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        return results

    def _extract_inline_json(self, text: str) -> list:
        """提取行内 JSON 工具调用"""
        import re
        results = []
        pattern = r'\{[^{}]*?"tool"\s*:\s*"([^"]+)"[^{}]*?"args"\s*:\s*(\{[^}]*?\})[^{}]*?\}'
        for tool_name, args_str in re.findall(pattern, text, re.DOTALL):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"input": args_str}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _extract_action_format(self, text: str) -> list:
        """提取 Action: tool_name(args) 格式"""
        import re
        results = []
        for tool_name, args_str in re.findall(r'Action\s*:\s*(\w+)\(([^)]*)\)', text):
            args_str = args_str.strip().strip('"\'')
            if args_str:
                # 尝试解析为 JSON
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                args = {}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _try_repair_json(self, text: str) -> list:
        """尝试修复不完整的 JSON"""
        import re
        results = []
        # 查找类似 {"tool": "xxx", "args": { 开头但不完整的内容
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"([^"]+)".*?"args"\s*:\s*\{([^}]*)', text, re.DOTALL):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            if args_str:
                # 尝试给 args 加上闭合括号
                try:
                    args = json.loads("{" + args_str + "}")
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                args = {}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _normalize(self, items: list) -> list:
        """标准化各种 JSON 格式到统一格式"""
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("tool") or item.get("name") or item.get("function") or ""
            args = item.get("args") or item.get("arguments") or item.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if name:
                results.append({"name": name, "arguments": args})
        return results

    def _validate_all(self, results: list, available_tools: list = None) -> list:
        """验证工具名是否在可用列表中"""
        if not available_tools:
            return results
        valid = []
        for tc in results:
            if tc["name"] in available_tools:
                valid.append(tc)
            else:
                # 模糊匹配：忽略大小写
                for t in available_tools:
                    if t.lower() == tc["name"].lower():
                        tc["name"] = t
                        valid.append(tc)
                        break
        return valid


class FewShotGenerator:
    """根据可用工具动态生成 few-shot 示例"""

    def generate(self, tool_schemas: list, max_examples: int = 3) -> str:
        """为工具列表生成标准格式的 few-shot 示例"""
        if not tool_schemas:
            return ""

        examples = []
        for i, schema in enumerate(tool_schemas[:max_examples]):
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            if not name:
                continue

            # 构造示例参数
            example_args = {}
            for pname, pinfo in list(params.items())[:2]:
                ptype = pinfo.get("type", "string")
                if ptype == "string":
                    example_args[pname] = "示例值"
                elif ptype == "integer":
                    example_args[pname] = 5
                elif ptype == "boolean":
                    example_args[pname] = True
                elif ptype == "number":
                    example_args[pname] = 1.0

            examples.append(
                f"用户: 请帮我完成某个任务\n"
                f"助手: 我来思考一下...\n"
                f"```json\n{json.dumps({'tool': name, 'args': example_args}, ensure_ascii=False, indent=2)}\n```"
            )

        return "\n\n".join(examples) if examples else ""


class ReActEngine:
    """ReAct 推理引擎：Thought -> Action -> Observation 循环"""

    def __init__(self, max_steps: int = 15, stream_callback: Callable = None):
        self.max_steps = max_steps
        self.stream_callback = stream_callback
        self._cancelled = False
        self._tool_parser = ToolCallParser()

    def _is_native_model(self) -> bool:
        """检查当前模型是否为 ModelSelf 原生模型"""
        from core.app_state import app_state
        trainer = app_state.trainer
        return trainer is not None and hasattr(trainer, 'generate_react_step')

    def _run_native(self, task: str, result: AgentResult,
                    start_time: float, registry) -> AgentResult:
        """
        ModelSelf 原生快速路径。

        直接使用模型的 generate_react_step() 输出结构化工具调用，
        不需要经过文本生成 + ToolCallParser 解析。
        """
        from core.app_state import app_state
        from agent.data_collector import get_collector

        trainer = app_state.trainer
        collector = get_collector()

        # 构建 prompt
        tool_desc = registry.get_tool_descriptions()
        prompt = f"[系统] 你是 OmniCore AI 助手。可用工具:\n{tool_desc}\n[用户] {task}\n[助手] "

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                result.status = "stopped"
                break

            step = AgentStep(step_number=step_num)
            step_start = time.time()

            try:
                self._emit("step_start", {"step": step_num})

                # 原生推理: 直接输出结构化结果
                # prompt 格式现在与训练时完全一致：
                # [系统] 你是 OmniCore AI 助手。可用工具:\n{tool_desc}\n[用户] {task}\n[助手] 
                react_result = trainer.generate_react_step(
                    prompt, max_new_tokens=256, temperature=0.3,
                )

                step.thought = react_result.get("thought", "")

                # 最终回答
                if "final_answer" in react_result:
                    step.is_final = True
                    step.observation = react_result["final_answer"]
                    result.steps.append(step)
                    result.final_answer = react_result["final_answer"]
                    result.status = "completed"

                    # 收集训练数据
                    collector.collect_react_step(
                        task=task, thought=step.thought,
                        action=None, action_args=None,
                        observation=None, is_final=True,
                        final_answer=react_result["final_answer"],
                    )
                    self._emit("final_answer", {"answer": step.observation, "step": step_num})
                    break

                # 工具调用
                action = react_result.get("action", "")
                action_args = react_result.get("action_args", {})

                if action:
                    step.action = action
                    step.action_args = action_args
                    self._emit("tool_call", {"step": step_num, "tool": action, "args": action_args})

                    observation = registry.execute(action, action_args)
                    step.observation = observation

                    # 收集训练数据
                    collector.collect_react_step(
                        task=task, thought=step.thought,
                        action=action, action_args=action_args,
                        observation=observation, is_final=False,
                    )

                    self._emit("observation", {"step": step_num, "tool": action, "result": observation[:500]})

                    # 更新 prompt 加入观察结果
                    prompt += f"<think>{step.thought}</think><tool_call>{action} {json.dumps(action_args)}\n<tool_result>{observation}</tool_result>\n"
                else:
                    # 没有工具调用也没有最终回答，视为最终回答
                    step.is_final = True
                    result.steps.append(step)
                    result.final_answer = step.thought
                    result.status = "completed"
                    break

            except Exception as e:
                step.error = str(e)
                logger.error(f"Native ReAct 步骤 {step_num} 异常: {traceback.format_exc()}")

            step.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step)
            self._emit("step_end", {"step": step_num, "thought": step.thought[:200],
                                    "action": step.action, "duration_ms": step.duration_ms})

        else:
            result.status = "max_steps"
            result.final_answer = f"达到最大推理步数 ({self.max_steps})"

        # 定期刷新收集的数据
        collector.flush()

        result.total_duration_ms = (time.time() - start_time) * 1000
        self._emit("complete", {"status": result.status, "steps": len(result.steps),
                                "duration_ms": result.total_duration_ms})
        return result

    def cancel(self):
        self._cancelled = True

    def _emit(self, event_type: str, data: dict):
        if self.stream_callback:
            try:
                self.stream_callback(event_type, data)
            except Exception:
                pass

    def run(self, task: str, system_prompt: str = "", history: list = None,
            skill_tools: list = None) -> AgentResult:
        """执行 ReAct 推理循环"""
        from agent.tool_registry import registry
        from agent.skill_manager import skill_manager

        self._cancelled = False
        result = AgentResult(task=task)
        start_time = time.time()

        # ── ModelSelf 原生快速路径 ──
        if self._is_native_model():
            return self._run_native(task, result, start_time, registry)

        # 获取工具 Schema（如果有技能过滤则只返回技能工具）
        if skill_tools:
            tool_schemas = [t.to_schema() for t in 
                           [registry.get(n) for n in skill_tools if registry.get(n)]]
        else:
            tool_schemas = registry.get_tool_schemas()

        # 构建系统提示
        if not system_prompt:
            tool_desc = registry.get_tool_descriptions()
            if skill_tools:
                # 只列出技能工具的描述
                lines = ["可用工具:"]
                for name in skill_tools:
                    t = registry.get(name)
                    if t:
                        lines.append(f"- **{t.name}**: {t.description}")
                tool_desc = "\n".join(lines)
            system_prompt = self._build_system_prompt(tool_desc)

        # 检查是否有技能系统提示
        skill_prompt = skill_manager.get_skill_system_prompt()
        if skill_prompt:
            system_prompt = skill_prompt + "\n\n" + system_prompt

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        self._emit("start", {"task": task, "max_steps": self.max_steps})

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                result.status = "stopped"
                break

            step = AgentStep(step_number=step_num)
            step_start = time.time()

            try:
                self._emit("step_start", {"step": step_num})
                llm_response = self._call_llm(system_prompt, messages, tool_schemas)

                if llm_response.get("error"):
                    step.error = llm_response["error"]
                    result.steps.append(step)
                    result.status = "failed"
                    break

                content = llm_response.get("content", "")
                tool_calls = llm_response.get("tool_calls", [])

                if content:
                    step.thought = content

                # 检查最终答案
                if self._is_final_answer(content, tool_calls, step_num):
                    step.is_final = True
                    step.observation = content
                    result.steps.append(step)
                    result.final_answer = content
                    result.status = "completed"
                    self._emit("final_answer", {"answer": content, "step": step_num})
                    break

                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                tool_args = {"input": tool_args}

                        # 检查是否在技能工具范围内
                        if skill_tools and tool_name not in skill_tools:
                            step.observation = f"工具 {tool_name} 不在当前技能范围内"
                            messages.append({"role": "assistant", "content": content or ""})
                            messages.append({"role": "user", "content": step.observation + "。请使用可用的工具。"})
                            continue

                        step.action = tool_name
                        step.action_args = tool_args

                        self._emit("tool_call", {"step": step_num, "tool": tool_name, "args": tool_args})
                        observation = registry.execute(tool_name, tool_args)
                        step.observation = observation

                        self._emit("observation", {"step": step_num, "tool": tool_name, "result": observation[:500]})

                        messages.append({
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": [{"id": f"call_{step_num}", "type": "function",
                                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]
                        })
                        messages.append({"role": "tool", "tool_call_id": f"call_{step_num}", "content": observation})
                else:
                    step.thought = content
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "请继续执行任务。如果已完成请给出最终答案。"})

            except Exception as e:
                step.error = str(e)
                logger.error(f"ReAct 步骤 {step_num} 异常: {traceback.format_exc()}")

            step.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step)
            self._emit("step_end", {"step": step_num, "thought": step.thought[:200], "action": step.action, "duration_ms": step.duration_ms})

        else:
            result.status = "max_steps"
            result.final_answer = f"达到最大推理步数 ({self.max_steps})"

        result.total_duration_ms = (time.time() - start_time) * 1000
        self._emit("complete", {"status": result.status, "steps": len(result.steps), "duration_ms": result.total_duration_ms})
        return result

    def run_stream(self, task: str, system_prompt: str = "", history: list = None,
                   skill_tools: list = None):
        """流式执行 ReAct 推理"""
        from agent.tool_registry import registry
        from agent.skill_manager import skill_manager

        self._cancelled = False

        if skill_tools:
            tool_schemas = [t.to_schema() for t in [registry.get(n) for n in skill_tools if registry.get(n)]]
        else:
            tool_schemas = registry.get_tool_schemas()

        if not system_prompt:
            tool_desc = registry.get_tool_descriptions()
            if skill_tools:
                lines = ["可用工具:"]
                for name in skill_tools:
                    t = registry.get(name)
                    if t:
                        lines.append(f"- **{t.name}**: {t.description}")
                tool_desc = "\n".join(lines)
            system_prompt = self._build_system_prompt(tool_desc)

        skill_prompt = skill_manager.get_skill_system_prompt()
        if skill_prompt:
            system_prompt = skill_prompt + "\n\n" + system_prompt

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        yield {"type": "start", "data": {"task": task}}

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                yield {"type": "cancelled", "data": {}}
                break

            try:
                llm_response = self._call_llm(system_prompt, messages, tool_schemas)
                if llm_response.get("error"):
                    yield {"type": "error", "data": {"error": llm_response["error"]}}
                    break

                content = llm_response.get("content", "")
                tool_calls = llm_response.get("tool_calls", [])

                if content:
                    yield {"type": "thought", "data": {"step": step_num, "content": content}}

                if self._is_final_answer(content, tool_calls, step_num):
                    yield {"type": "final", "data": {"answer": content, "step": step_num}}
                    break

                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                tool_args = {"input": tool_args}

                        yield {"type": "action", "data": {"step": step_num, "tool": tool_name, "args": tool_args}}
                        observation = registry.execute(tool_name, tool_args)
                        yield {"type": "observation", "data": {"step": step_num, "tool": tool_name, "result": observation[:1000]}}

                        messages.append({"role": "assistant", "content": content or "",
                                         "tool_calls": [{"id": f"call_{step_num}", "type": "function",
                                                         "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]})
                        messages.append({"role": "tool", "tool_call_id": f"call_{step_num}", "content": observation})
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "请继续执行任务。如果已完成请给出最终答案。"})

            except Exception as e:
                yield {"type": "error", "data": {"step": step_num, "error": str(e)}}
                break

        yield {"type": "done", "data": {}}

    def _call_llm(self, system_prompt: str, messages: list, tools: list) -> dict:
        """调用 LLM（云端优先，本地回退）"""
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # 尝试云端 API
        try:
            import os
            settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                api_base = settings.get("cloud_api_base", "")
                api_key = settings.get("cloud_api_key", "")
                api_model = settings.get("cloud_api_model", "")
                if api_base and api_key and api_model:
                    return self._call_openai_compatible(api_base, api_key, api_model, full_messages, tools)
        except Exception as e:
            logger.warning(f"云端 API 调用失败: {e}")

        return self._call_local_model(full_messages)

    def _call_openai_compatible(self, api_base: str, api_key: str, model: str, messages: list, tools: list) -> dict:
        import urllib.request
        url = api_base.rstrip("/") + "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            result = {"content": message.get("content", ""), "tool_calls": []}
            for tc in message.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"input": args}
                result["tool_calls"].append({"id": tc.get("id", ""), "name": func.get("name", ""), "arguments": args})
            return result
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return {"error": str(e)}

    def _call_local_model(self, messages: list, available_tools: list = None) -> dict:
        from core.app_state import app_state
        try:
            trainer = app_state.trainer
            if trainer and hasattr(trainer, 'generate'):
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"[系统] {content}")
                    elif role == "user":
                        prompt_parts.append(f"[用户] {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"[助手] {content}")
                    elif role == "tool":
                        prompt_parts.append(f"[工具结果] {content}")
                prompt = "\n".join(prompt_parts) + "\n[助手]"
                response = trainer.generate(prompt, max_new_tokens=1000, temperature=0.3, stop_sequences=["[用户]", "[系统]"])
                content = response.strip()
                # 使用多策略解析器
                tool_names = [s.get("function", {}).get("name", "") for s in (available_tools or [])]
                tool_calls = self._tool_parser.parse(content, available_tools=tool_names)
                if tool_calls:
                    return {"content": "", "tool_calls": tool_calls}
                # 自修复：第一次解析失败时重试一次
                if content and len(content) > 10:
                    repair_result = self._self_repair(content, messages, available_tools)
                    if repair_result:
                        return repair_result
                return {"content": content, "tool_calls": []}
        except Exception as e:
            logger.warning(f"本地模型调用失败: {e}")
        return {"error": "无可用的 LLM 模型。请在设置中配置云端 API 或加载本地模型。"}

    def _self_repair(self, raw_output: str, messages: list, available_tools: list = None) -> dict:
        """自修复机制：当工具调用解析失败时，请求 LLM 修正格式"""
        try:
            from core.app_state import app_state
            trainer = app_state.trainer
            if not trainer or not hasattr(trainer, 'generate'):
                return None

            tool_names = [s.get("function", {}).get("name", "") for s in (available_tools or [])]
            tool_list = ", ".join(tool_names[:10]) if tool_names else "无"

            repair_prompt = (
                f"[系统] 你的上一条回复无法被解析为有效的工具调用。\n"
                f"原始回复: {raw_output[:300]}\n\n"
                f"可用工具: {tool_list}\n\n"
                f"请严格按照以下 JSON 格式重新输出工具调用（只输出 JSON，不要其他内容）:\n"
                f'{{"tool": "工具名", "args": {{"参数": "值"}}}}\n'
                f"[助手]"
            )
            response = trainer.generate(repair_prompt, max_new_tokens=300, temperature=0.1,
                                         stop_sequences=["[用户]", "[系统]"])
            content = response.strip()
            tool_calls = self._tool_parser.parse(content, available_tools=tool_names)
            if tool_calls:
                logger.info("自修复成功：LLM 已修正工具调用格式")
                return {"content": "", "tool_calls": tool_calls}
        except Exception as e:
            logger.debug(f"自修复失败: {e}")
        return None

    def _build_system_prompt(self, tool_descriptions: str, tool_schemas: list = None) -> str:
        """构建增强版系统提示（含 few-shot 示例和严格格式约束）"""
        few_shot = ""
        if tool_schemas:
            generator = FewShotGenerator()
            few_shot = generator.generate(tool_schemas, max_examples=2)
            if few_shot:
                few_shot = f"\n## 示例\n{few_shot}\n"

        return f"""你是一个自主推理的 AI Agent。通过思考、选择工具、执行、观察结果来完成任务。

## 工具列表
{tool_descriptions}

## 输出格式（严格遵守）
调用工具时，必须输出以下 JSON 格式（不要输出其他内容）：
```json
{{"tool": "工具名", "args": {{"参数名": "值"}}}}
```
{few_shot}
## 规则
- 每次只调用一个工具
- 只能使用上述列出的工具
- 如果任务已完成，直接用自然语言给出最终答案（不要包含 JSON）
- 保持简洁"""

    def _is_final_answer(self, content: str, tool_calls: list, step_num: int) -> bool:
        if tool_calls:
            return False
        if not content:
            return False
        final_markers = ["最终答案", "Final Answer", "任务完成", "已完成", "总结"]
        for marker in final_markers:
            if marker in content:
                return True
        # 至少经过 2 步推理才允许直接作为最终答案
        return step_num >= 2 and not tool_calls and len(content) > 20


class AgentController:
    """Agent 主控制器"""

    def __init__(self, max_steps: int = 15):
        self.engine = ReActEngine(max_steps=max_steps)
        self._task_history: list = []

    def run_task(self, task: str, stream_callback: Callable = None, skill_id: str = None) -> AgentResult:
        from agent.skill_manager import skill_manager
        if stream_callback:
            self.engine.stream_callback = stream_callback

        skill_tools = None
        if skill_id:
            skill = skill_manager.activate_skill(skill_id)
            if skill:
                skill_tools = skill.tools

        result = self.engine.run(task, history=self._task_history[-10:], skill_tools=skill_tools)

        self._task_history.append({"role": "user", "content": task})
        if result.final_answer:
            self._task_history.append({"role": "assistant", "content": result.final_answer})

        # 从成功的任务中学习
        if result.status == "completed" and len(result.steps) >= 2:
            skill_manager.learn_from_task(task, 
                [{"thought": s.thought, "action": s.action} for s in result.steps],
                result.final_answer)

        return result

    def run_task_stream(self, task: str, skill_id: str = None):
        from agent.skill_manager import skill_manager
        skill_tools = None
        if skill_id:
            skill = skill_manager.activate_skill(skill_id)
            if skill:
                skill_tools = skill.tools
        return self.engine.run_stream(task, history=self._task_history[-10:], skill_tools=skill_tools)

    def cancel(self):
        self.engine.cancel()

    def clear_history(self):
        self._task_history.clear()

    def get_history(self) -> list:
        return self._task_history.copy()