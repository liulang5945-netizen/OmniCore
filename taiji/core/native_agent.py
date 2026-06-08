"""
Taiji Native Agent Engine
Nervous System — All organs working together as a complete living being.

Replaces standalone inference, integrates perception/memory/planning/reflection subsystems.
Taiji is not just a model — it's a complete autonomous Agent entity.
"""
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
from threading import Event

from taiji.core.inference import NativeInferenceEngine
from taiji.agent.perception import PerceptionSystem
from taiji.agent.memory import MemorySystem
from taiji.agent.planner import PlannerSystem, PlanAction
from taiji.agent.reflector import ReflectorSystem
from taiji.architecture import ModelSelf
from taiji.tokenizer import ModelSelfTokenizer

logger = logging.getLogger("ModelSelf.NativeAgent")


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
    reflection: Optional[str] = None


@dataclass
class AgentResult:
    task: str
    final_answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0
    status: str = "completed"


class NativeAgentEngine:
    """
    原生 Agent 引擎 — 模型即 Agent

    集成所有子系统:
    - 感知: 编码工作台状态
    - 记忆: 短期/长期记忆读写
    - 规划: 任务分解与跟踪
    - 反思: 错误检测与纠正
    - 推理: 原生 ReAct 步骤

    与 NativeInferenceEngine 的区别:
    - 后者只做文本生成 + 工具分类
    - 本引擎驱动完整的 Agent 循环
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_steps: int = 15,
        stream_callback: Optional[Callable] = None,
        workspace_path: Optional[str] = None,
        memory_save_path: Optional[str] = None,
    ):
        """
        Args:
            model: ModelSelf（态极）
            tokenizer: ModelSelfTokenizer
        """
        self.inference = NativeInferenceEngine(model, tokenizer, device)
        self.perception = PerceptionSystem(tokenizer)
        self.memory = MemorySystem()
        self.planner = PlannerSystem()
        self.reflector = ReflectorSystem()

        # 天生联网：自动搜索上下文注入
        self.web_context_provider = None  # func(task, step_num) -> Optional[str]
        self._web_cache = {}  # 缓存搜索结果，避免重复搜索
        self._web_cache_ttl = 300  # 缓存有效期（秒）
        self._web_last_search_time = 0

        self.max_steps = max_steps
        self.stream_callback = stream_callback
        self.workspace_path = workspace_path
        self.memory_save_path = memory_save_path
        self._cancelled = False

        # 自修改引擎：自主发现和安装新工具
        self._self_mod = None
        try:
            from agent.self_modification import get_self_modification_engine
            self._self_mod = get_self_modification_engine()
            logger.info("自修改引擎已连接")
        except Exception:
            pass

        # 加载持久化的长期记忆
        if memory_save_path:
            self.memory.load(memory_save_path)

    def generate(self, prompt, tokenizer=None, max_new_tokens=256,
                 temperature=0.7, top_p=0.9):
        """兼容 BaseInferenceEngine 接口"""
        return self.inference.generate(prompt, tokenizer, max_new_tokens, temperature, top_p)

    def generate_stream(self, prompt, tokenizer=None, max_new_tokens=512,
                        temperature=0.7, top_p=0.9, stop_event=None):
        """兼容 BaseInferenceEngine 流式接口"""
        return self.inference.generate_stream(
            prompt, tokenizer, max_new_tokens, temperature, top_p, stop_event
        )

    def run(self, task: str, tool_registry=None) -> AgentResult:
        """
        执行完整的 Agent 任务循环。

        流程:
        1. 感知: 编码工作台状态
        2. 记忆: 注入相关记忆
        3. 推理: 生成 ReAct 步骤
        4. 执行: 调用工具
        5. 反思: 评估结果
        6. 规划: 更新进度
        7. 记忆: 写入新记忆
        """
        self._cancelled = False
        self.reflector.reset()
        result = AgentResult(task=task)
        start_time = time.time()

        self._emit("start", {"task": task, "max_steps": self.max_steps})

        # 感知: 编码工作台
        if self.workspace_path:
            env_tokens = self.perception.encode_workspace(self.workspace_path)

        # 记忆: 获取上下文
        mem_context = self.memory.get_context_tokens(self.inference.tokenizer)

        # 规划: 创建初步计划
        tool_desc = ""
        if tool_registry:
            tool_desc = tool_registry.get_tool_descriptions()

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                result.status = "stopped"
                break

            step = AgentStep(step_number=step_num)
            step_start = time.time()

            try:
                self._emit("step_start", {"step": step_num})

                # 构建 prompt
                prompt_parts = []
                if mem_context:
                    prompt_parts.append(f"<mem_read>{self.inference.tokenizer.decode(mem_context, skip_special_tokens=False)}</mem_read>")
                if self.planner.current_plan:
                    plan_text = self.planner.current_plan.to_token_text()
                    prompt_parts.append(plan_text)

                # 天生联网：自动注入搜索上下文
                if self.web_context_provider:
                    try:
                        web_ctx = self.web_context_provider(task, step_num)
                        if web_ctx:
                            prompt_parts.append(f"<web_search>\n{web_ctx}\n</web_search>")
                    except Exception:
                        pass

                # 添加历史步骤
                for prev_step in result.steps:
                    if prev_step.thought:
                        prompt_parts.append(f"<think>{prev_step.thought}</think>")
                    if prev_step.action:
                        args_str = json.dumps(prev_step.action_args, ensure_ascii=False)
                        prompt_parts.append(f"<tool_call>{prev_step.action} {args_str}")
                    if prev_step.observation:
                        prompt_parts.append(f"<tool_result>{prev_step.observation[:300]}</tool_result>")

                prompt_parts.append(f"[系统] 你是 OmniCore AI 助手。可用工具:\n{tool_desc}")
                prompt_parts.append(f"[用户] {task}")
                prompt_parts.append("[助手] ")
                full_prompt = "\n".join(prompt_parts)

                # 原生推理（如果 prompt 太长，去掉 web context 重试）
                react_result = self.inference.generate_react_step(
                    full_prompt, max_new_tokens=256, temperature=0.3,
                )

                # 如果输出为空且有 web context，去掉 web context 重试
                if not react_result.get("thought") and not react_result.get("final_answer") and not react_result.get("action"):
                    if "<web_search>" in full_prompt:
                        stripped_prompt = re.sub(r'<web_search>.*?</web_search>\n?', '', full_prompt, flags=re.DOTALL)
                        logger.info("模型输出为空，去掉联网上下文重试")
                        react_result = self.inference.generate_react_step(
                            stripped_prompt, max_new_tokens=256, temperature=0.3,
                        )

                step.thought = react_result.get("thought", "")

                # 如果连续 3 步都没有有效输出，强制结束
                if step_num >= 3 and not react_result.get("action") and not react_result.get("final_answer"):
                    fallback = step.thought or "抱歉，我暂时无法回答这个问题。"
                    react_result["final_answer"] = fallback

                if not self.planner.current_plan and step_num == 1:
                    self.planner.create_plan(task, [f"执行: {task[:50]}"])

                # 最终回答
                if "final_answer" in react_result:
                    step.is_final = True
                    step.observation = react_result["final_answer"]
                    result.steps.append(step)
                    result.final_answer = react_result["final_answer"]
                    result.status = "completed"
                    self.memory.auto_write(f"完成任务: {task[:100]}", importance=0.8)
                    self.planner.complete_current_step("任务完成")
                    self.reflector.evaluate_result("任务", "任务完成")
                    self._emit("final_answer", {"answer": step.observation, "step": step_num})
                    break

                # 工具调用
                action = react_result.get("action", "")
                action_args = react_result.get("action_args", {})

                if action:
                    step.action = action
                    step.action_args = action_args

                    # ── v2: 执行前自验证 ──
                    confidence, warnings = self.reflector.verify_before_act(
                        action, action_args, context=task,
                    )
                    if warnings:
                        self._emit("verify", {
                            "step": step_num, "tool": action,
                            "confidence": confidence, "warnings": warnings,
                        })
                    # 低置信度时，生成备选方案并注入记忆
                    if confidence < 0.5 and tool_registry:
                        try:
                            alt_tools = list(getattr(tool_registry, '_tools', {}).keys())
                            alternatives = self.reflector.suggest_alternatives(
                                action, action_args, alt_tools,
                            )
                            if alternatives:
                                alt_text = "; ".join(
                                    f"{a['tool']}({a['reason']})" for a in alternatives[:2]
                                )
                                self.memory.auto_write(
                                    f"低置信度: {action} → 备选: {alt_text}",
                                    importance=0.7,
                                )
                                self._emit("alternatives", {
                                    "step": step_num, "alternatives": alternatives,
                                })
                        except Exception:
                            pass

                    self._emit("tool_call", {"step": step_num, "tool": action, "args": action_args})

                    observation = tool_registry.execute(action, action_args)
                    step.observation = observation

                    # 反思
                    reflection = self.reflector.evaluate_result(action, observation)
                    if reflection.type.value == "detect" and reflection.should_retry:
                        self.memory.auto_write(f"错误: {action} -> {reflection.message}", importance=0.6)
                    elif reflection.type.value == "confirm":
                        self.memory.auto_write(f"{action}: {observation[:200]}", importance=0.5)

                    self.planner.complete_current_step(f"{action} -> {observation[:100]}")

                    extra_actions = react_result.get("extra_actions", [])
                    for extra in extra_actions:
                        extra_result = tool_registry.execute(extra["action"], extra["action_args"])
                        self.memory.auto_write(f"{extra['action']}: {extra_result[:200]}", importance=0.5)

                    self._emit("observation", {"step": step_num, "tool": action, "result": observation[:500]})
                else:
                    step.is_final = True
                    result.steps.append(step)
                    result.final_answer = step.thought or "(无输出)"
                    result.status = "completed"
                    break

                self.memory.consolidate()

            except Exception as e:
                step.error = str(e)
                self.reflector.evaluate_result("系统", str(e))

                # ── 自主进化：工具失败时尝试补齐能力 ──
                if self._self_mod and step.action:
                    try:
                        evolution = self._self_mod.evolve(task, tool_registry)
                        if evolution.get("evolved"):
                            evolved_msg = evolution.get("message", "")
                            logger.info(f"态极自主进化成功: {evolved_msg}")
                            self.memory.auto_write(
                                f"自主进化: {evolved_msg}",
                                importance=0.9,
                            )
                            self._emit("evolution", {
                                "step": step_num,
                                "action": evolution.get("action", ""),
                                "tool_name": evolution.get("tool_name", ""),
                                "message": evolved_msg,
                            })
                            # 进化成功，不计入连续错误，继续下一步重试
                            step.reflection = f"已进化: {evolved_msg}"
                            continue
                    except Exception as evo_err:
                        logger.debug(f"自主进化失败: {evo_err}")

                if self.reflector.should_abort():
                    result.status = "error"
                    result.final_answer = f"连续错误过多，任务终止: {e}"
                    break

            step.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step)
            self._emit("step_end", {"step": step_num, "thought": step.thought[:200],
                                    "action": step.action, "duration_ms": step.duration_ms})
        else:
            result.status = "max_steps"
            result.final_answer = f"达到最大推理步数 ({self.max_steps})"

        result.total_duration_ms = (time.time() - start_time) * 1000
        if self.memory_save_path:
            self.memory.save(self.memory_save_path)
        self._emit("complete", {"status": result.status, "steps": len(result.steps),
                                "duration_ms": result.total_duration_ms})
        return result

    def register_tools(self, tool_names: list):
        """注册工具名到分词器和模型"""
        for name in tool_names:
            self.inference.tokenizer.register_tool(name)
        self.inference.model.set_num_tools(len(self.inference.tokenizer._tool_name_to_id))
        # 连接知识学习器
        self._connect_knowledge_learner()
        # 连接天生联网
        self._connect_web_context()

    def _connect_knowledge_learner(self):
        """连接知识学习器，注入搜索、LLM 和浏览器能力"""
        try:
            from agent.knowledge_learner import get_knowledge_learner
            from agent.tool_registry import registry
            learner = get_knowledge_learner()
            # 注入LLM函数
            if hasattr(self, 'inference') and hasattr(self.inference, 'generate'):
                learner.set_llm(lambda p: self.inference.generate(p, max_new_tokens=512, temperature=0.3))
            # 注入搜索函数
            if registry.has("search"):
                learner.set_search(lambda q: registry.execute("search", {"input": q}))
            # 注入网页阅读器（降级用）
            if registry.has("read_webpage"):
                learner.set_web_reader(lambda u: registry.execute("read_webpage", {"input": u}))
            # 注入智能抓取（优先用 smart_fetch，即 MCP fetch）
            if registry.has("smart_fetch"):
                learner.set_fetch_fn(lambda u: registry.execute("smart_fetch", {"input": u}))
                logger.info("知识学习器已连接智能抓取 (smart_fetch)")
            elif registry.has("browse_web"):
                learner.set_fetch_fn(lambda u: registry.execute("browse_web", {"input": u}))
                logger.info("知识学习器已连接浏览器 (browse_web)")
            logger.info("知识学习器已连接: LLM+搜索+浏览器就绪")
        except Exception as e:
            logger.warning(f"知识学习器连接失败: {e}")

    def _connect_web_context(self):
        """天生联网 + 边学边聊：自动搜索、注入上下文、后台沉淀知识"""
        try:
            from agent.tool_registry import registry
            if not registry.has("search"):
                return

            # 边学边聊：后台沉淀知识的计数器（避免每次搜索都写入）
            self._learn_counter = 0
            self._learn_threshold = 3  # 每 3 次有价值的搜索自动沉淀一次

            def _auto_search_context(task: str, step_num: int) -> Optional[str]:
                """自动搜索 + 深入读取 + 后台沉淀知识"""
                import time as _time
                import re
                now = _time.time()

                # 简单问题不需要联网（节省上下文空间）
                simple_patterns = ['几月', '几号', '日期', '时间', '几点', '天气',
                                   '你好', '谢谢', '你是谁', 'hello', 'hi', '今天']
                task_lower = task.lower()
                if any(p in task_lower for p in simple_patterns) and len(task) < 30:
                    return None

                # 缓存检查
                cache_key = task[:100]
                if cache_key in self._web_cache:
                    cached_time, cached_result = self._web_cache[cache_key]
                    if now - cached_time < self._web_cache_ttl:
                        return cached_result if step_num <= 2 else None

                # 只在第一步和每 5 步搜索一次
                if step_num > 1 and step_num % 5 != 0:
                    return None

                query = task[:80].replace("\n", " ")
                if len(query) < 5:
                    return None

                try:
                    # 第一层：搜索引擎摘要
                    result = registry.execute("search", {"input": query})
                    if not result or len(str(result).strip()) < 20:
                        return None

                    full_context = f"【搜索摘要】\n{str(result)[:500]}"
                    collected_pages = []  # 收集到的网页内容（用于后台沉淀）

                    # 第二层：深入读取前 2 个网页
                    urls = re.findall(r'https?://[^\s\)\"\']+', str(result))
                    urls = [u for u in urls if not any(x in u for x in
                            ['google.com/search', 'baidu.com/s', 'bing.com/search', 'duckduckgo.com'])]

                    for url in urls[:2]:
                        try:
                            if registry.has("smart_fetch"):
                                page = registry.execute("smart_fetch", {"input": url})
                            elif registry.has("browse_web"):
                                page = registry.execute("browse_web", {"input": url})
                            elif registry.has("read_webpage"):
                                page = registry.execute("read_webpage", {"input": url})
                            else:
                                break

                            if page and len(str(page).strip()) > 100:
                                full_context += f"\n\n【网页正文: {url[:60]}】\n{str(page)[:600]}"
                                collected_pages.append({"url": url, "content": str(page)})
                        except Exception:
                            pass

                    # 边学边聊：后台沉淀有价值的知识
                    if collected_pages:
                        self._learn_counter += 1
                        if self._learn_counter >= self._learn_threshold:
                            self._learn_counter = 0
                            self._background_learn(query, task, collected_pages)

                    # 控制长度：避免撑爆模型上下文窗口
                    truncated = full_context[:600]
                    self._web_cache[cache_key] = (now, truncated)
                    return truncated

                except Exception as e:
                    logger.debug(f"自动联网搜索失败: {e}")
                return None

            self.web_context_provider = _auto_search_context
            logger.info("态极天生联网已激活: 搜索 + 深入读取 + 边学边聊")
        except Exception as e:
            logger.debug(f"天生联网连接失败: {e}")

    def _background_learn(self, query: str, task: str, pages: list):  # noqa: task保留供未来扩展
        """边学边聊：后台将搜索到的有价值内容沉淀到知识库"""
        import threading

        def _do_learn():
            try:
                from agent.knowledge_learner import get_knowledge_learner
                learner = get_knowledge_learner()
                if not learner:
                    return

                # 从任务中提取领域名
                domain = query[:30].replace("是什么", "").replace("怎么", "").replace("如何", "").strip()
                if len(domain) < 2:
                    domain = query[:20]

                # 直接将已采集的内容写入知识库（不重复抓取）
                for page in pages:
                    content = page.get("content", "")
                    url = page.get("url", "")
                    if content and len(content) > 200:
                        # 用 LLM 结构化提取（如果可用）
                        if learner._llm:
                            try:
                                prompt = (
                                    f"从以下网页内容中提取关键知识点，输出 JSON 数组格式。\n"
                                    f"每个知识点包含: concept(概念名), content(核心内容,50-150字), tags(关键词数组)\n"
                                    f"领域: {domain}\n"
                                    f"内容:\n{content[:2000]}\n\n"
                                    f"输出JSON:"
                                )
                                llm_result = learner._llm(prompt)
                                if llm_result and "[" in llm_result:
                                    import json
                                    # 提取 JSON 部分
                                    json_str = llm_result[llm_result.index("["):llm_result.rindex("]")+1]
                                    entries = json.loads(json_str)
                                    from agent.knowledge_learner import KnowledgeEntry
                                    for entry_data in entries[:3]:  # 每页最多 3 条
                                        if isinstance(entry_data, dict) and entry_data.get("concept"):
                                            entry = KnowledgeEntry(
                                                id="",
                                                domain=domain,
                                                concept=entry_data["concept"],
                                                content=entry_data.get("content", ""),
                                                tags=entry_data.get("tags", []),
                                                source=url,
                                                confidence=0.6,  # 自动学习的置信度较低
                                            )
                                            learner.store.add(entry)
                                    logger.debug(f"边学边聊: 已从 {url[:50]} 沉淀 {len(entries)} 条知识")
                            except Exception as e:
                                logger.debug(f"边学边聊结构化失败: {e}")
                        else:
                            # 无 LLM 时，按段落简单提取
                            from agent.knowledge_learner import KnowledgeEntry
                            paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 50]
                            for para in paragraphs[:2]:
                                entry = KnowledgeEntry(
                                    id="",
                                    domain=domain,
                                    concept=para[:30],
                                    content=para[:200],
                                    tags=[domain],
                                    source=url,
                                    confidence=0.4,
                                )
                                learner.store.add(entry)
                            logger.debug(f"边学边聊: 已从 {url[:50]} 简单沉淀知识")

                # 保存知识库
                learner.store.save()

            except Exception as e:
                logger.debug(f"边学边聊后台沉淀失败: {e}")

        # 后台线程执行，不阻塞推理
        t = threading.Thread(target=_do_learn, daemon=True, name="background-learn")
        t.start()

    def get_status(self) -> dict:
        return {
            "type": "native_agent",
            "memory": self.memory.get_stats(),
            "plan": self.planner.get_status(),
            "reflection": self.reflector.get_stats(),
            "registered_tools": len(self.inference.tokenizer._tool_name_to_id),
        }

    def cancel(self):
        self._cancelled = True

    def _emit(self, event_type: str, data: dict):
        if self.stream_callback:
            try:
                self.stream_callback(event_type, data)
            except Exception:
                pass