"""
多 Agent 协作系统 (Multi-Agent Orchestrator)
============================================
基于角色的任务分解和协作执行。
支持多角色 Agent 之间的消息传递和任务编排。
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("MultiAgent")


@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str
    display_name: str
    description: str
    system_prompt: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    max_steps: int = 10


@dataclass
class SubTask:
    """子任务"""
    id: str
    description: str
    assigned_role: str = ""
    status: str = "pending"  # pending/running/done/failed
    result: str = ""
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CollabTask:
    """协作任务"""
    id: str
    original_task: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: str = "pending"
    final_result: str = ""
    created_at: float = 0


class MessageBus:
    """Agent 间通信消息总线"""

    def __init__(self):
        self._messages: List[dict] = []

    def publish(self, topic: str, sender: str, content: str, metadata: dict = None):
        self._messages.append({
            "id": str(uuid.uuid4())[:8],
            "topic": topic,
            "sender": sender,
            "content": content[:500],
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def get_messages(self, topic: str = None, limit: int = 50) -> list:
        msgs = self._messages
        if topic:
            msgs = [m for m in msgs if m.get("topic", "") == topic]
        return msgs[-limit:]

    def clear(self):
        self._messages.clear()


class Orchestrator:
    """多 Agent 编排器"""

    def __init__(self):
        self.roles: Dict[str, AgentRole] = {}
        self._tasks: Dict[str, CollabTask] = {}
        self.message_bus = MessageBus()
        self._init_default_roles()

    def _init_default_roles(self):
        """初始化默认角色"""
        self.roles = {
            "coder": AgentRole(
                name="coder",
                display_name="👨‍💻 程序员",
                description="负责代码编写、调试和优化",
                system_prompt="你是一个专业的程序员，擅长编写高质量代码。",
                allowed_tools=["write_file", "edit_file", "execute_python", "analyze_code", "install_dependency"],
                max_steps=15,
            ),
            "researcher": AgentRole(
                name="researcher",
                display_name="🔍 研究员",
                description="负责信息搜索、文档查阅和技术调研",
                system_prompt="你是一个技术研究员，擅长搜索和整理信息。",
                allowed_tools=["search", "read_webpage", "read_local_file"],
                max_steps=10,
            ),
            "planner": AgentRole(
                name="planner",
                display_name="📋 规划师",
                description="负责任务分解、计划制定和进度管理",
                system_prompt="你是一个项目规划师，擅长任务分解和管理。",
                allowed_tools=["create_plan", "update_plan", "get_plan", "save_context"],
                max_steps=8,
            ),
            "reviewer": AgentRole(
                name="reviewer",
                display_name="🔎 审查员",
                description="负责代码审查、质量检查和测试验证",
                system_prompt="你是一个代码审查员，擅长发现问题和提出改进建议。",
                allowed_tools=["read_local_file", "analyze_code", "execute_python"],
                max_steps=8,
            ),
        }

    def decompose_task(self, task: str) -> CollabTask:
        """将任务分解为子任务（简单规则分解）"""
        collab_task = CollabTask(
            id=str(uuid.uuid4())[:8],
            original_task=task,
            created_at=time.time(),
        )

        # 简单的任务分解逻辑
        task_lower = task.lower()
        subtasks = []

        # 需要搜索的关键词
        search_keywords = ["搜索", "查找", "了解", "调研", "search", "find", "查询"]
        needs_search = any(kw in task_lower for kw in search_keywords)

        # 需要编码的关键词
        code_keywords = ["编写", "创建", "开发", "实现", "代码", "程序", "build", "create", "code", "implement"]
        needs_coding = any(kw in task_lower for kw in code_keywords)

        step_num = 0

        if needs_search:
            step_num += 1
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description=f"调研和收集关于 '{task}' 的相关信息",
                assigned_role="researcher",
                priority=1,
            ))

        if needs_coding:
            step_num += 1
            plan_deps = []
            if needs_search:
                plan_deps = ["sub_1"]
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description=f"制定 '{task}' 的开发计划",
                assigned_role="planner",
                priority=2,
                dependencies=plan_deps,
            ))

            step_num += 1
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description=f"实现 '{task}' 的核心代码",
                assigned_role="coder",
                priority=3,
                dependencies=[f"sub_{step_num - 1}"] if subtasks else [],
            ))

            step_num += 1
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description="审查和测试代码质量",
                assigned_role="reviewer",
                priority=4,
                dependencies=[f"sub_{step_num - 1}"],
            ))

        # 如果没有匹配到特定模式，使用通用分解
        if not subtasks:
            step_num += 1
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description=f"分析和理解任务: {task}",
                assigned_role="planner",
                priority=1,
            ))
            step_num += 1
            subtasks.append(SubTask(
                id=f"sub_{step_num}",
                description=f"执行任务: {task}",
                assigned_role="coder",
                priority=2,
                dependencies=["sub_1"],
            ))

        collab_task.subtasks = subtasks
        self._tasks[collab_task.id] = collab_task

        self.message_bus.publish("task_decomposed", "orchestrator",
                                f"任务已分解为 {len(subtasks)} 个子任务",
                                {"task_id": collab_task.id})

        return collab_task

    def execute_task(self, task_id: str) -> dict:
        """
        执行协作任务 — 真正调用 ReAct 引擎执行每个子任务。

        每个子任务根据其角色（coder/researcher/planner/reviewer）
        使用对应的 system_prompt 和 allowed_tools 调用 ReAct 引擎。
        """
        task = self._tasks.get(task_id)
        if not task:
            return {"status": "error", "message": f"任务 {task_id} 不存在"}

        task.status = "running"
        results = []

        # 获取 ReAct 引擎
        react_engine = self._get_react_engine()

        for subtask in task.subtasks:
            # 检查依赖
            deps_met = all(
                any(st.id == dep and st.status == "done" for st in task.subtasks)
                for dep in subtask.dependencies
            ) if subtask.dependencies else True

            if not deps_met:
                subtask.status = "failed"
                subtask.result = "依赖未满足"
                results.append({"subtask_id": subtask.id, "status": "failed", "result": "依赖未满足"})
                continue

            subtask.status = "running"
            role = self.roles.get(subtask.assigned_role)
            self.message_bus.publish("subtask_start", subtask.assigned_role,
                                    f"开始执行: {subtask.description[:100]}")

            # 真正执行子任务
            try:
                if react_engine:
                    # 用 ReAct 引擎执行
                    system_prompt = role.system_prompt if role else "你是一个AI助手。"
                    full_prompt = f"{system_prompt}\n\n任务: {subtask.description}"

                    result = react_engine.run(
                        task=subtask.description,
                        system_prompt=system_prompt,
                        max_steps=role.max_steps if role else 10,
                    )
                    subtask.result = result.final_answer if result.final_answer else str(result)
                    subtask.status = "done" if not result.error else "failed"
                else:
                    # 没有 ReAct 引擎，用简单执行
                    subtask.result = self._simple_execute(subtask, role)
                    subtask.status = "done"

            except Exception as e:
                logger.error(f"子任务执行失败: {e}")
                subtask.status = "failed"
                subtask.result = f"执行失败: {str(e)}"

            self.message_bus.publish(
                "subtask_done" if subtask.status == "done" else "subtask_failed",
                subtask.assigned_role,
                subtask.result[:200],
            )

            results.append({
                "subtask_id": subtask.id,
                "role": subtask.assigned_role,
                "status": subtask.status,
                "result": subtask.result,
            })

        task.status = "completed" if all(r["status"] == "done" for r in results) else "partial"
        task.final_result = "\n".join(
            f"[{r.get('role', '?')}] {r.get('result', '')}" for r in results
        )

        return {
            "status": "ok",
            "task_id": task_id,
            "subtasks_results": results,
            "final_result": task.final_result,
        }

    def _get_react_engine(self):
        """获取 ReAct 引擎（延迟加载，避免循环依赖）"""
        try:
            from core.app_state import app_state
            if app_state.trainer:
                return app_state.trainer
            # 尝试创建一个简单的 ReAct 引擎
            from agent.react_engine import ReActEngine
            return ReActEngine()
        except Exception as e:
            logger.warning(f"无法获取 ReAct 引擎: {e}")
            return None

    def _simple_execute(self, subtask: SubTask, role: AgentRole) -> str:
        """简单执行（无 ReAct 引擎时的回退方案）"""
        # 根据角色类型给出基本响应
        role_name = subtask.assigned_role
        if role_name == "coder":
            return f"[程序员] 已分析编码任务: {subtask.description[:100]}。请使用具体的代码执行工具完成。"
        elif role_name == "researcher":
            return f"[研究员] 已分析调研任务: {subtask.description[:100]}。请使用搜索工具获取信息。"
        elif role_name == "reviewer":
            return f"[审查员] 已分析审查任务: {subtask.description[:100]}。请使用代码分析工具检查。"
        else:
            return f"[{role_name}] 任务已接收: {subtask.description[:100]}"

    def list_tasks(self) -> list:
        return [
            {
                "id": t.id,
                "original_task": t.original_task[:100],
                "status": t.status,
                "subtasks_count": len(t.subtasks),
                "subtasks": [
                    {"id": st.id, "description": st.description[:80], "role": st.assigned_role, "status": st.status}
                    for st in t.subtasks
                ],
            }
            for t in self._tasks.values()
        ]


# 全局单例
orchestrator = Orchestrator()
message_bus = orchestrator.message_bus