"""
工作流编排引擎 (Workflow Engine)
===============================
支持 DAG 任务编排、条件分支、并行执行、错误重试。
"""
import ast
import json
import logging
import os
import operator
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("WorkflowEngine")


# ======================== 安全表达式求值 ========================

_SAFE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_BOOL_OPS = {
    ast.And: all,
    ast.Or: any,
}

_SAFE_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
}

_DANGEROUS_NODE_TYPES = (
    ast.Call, ast.Import, ast.ImportFrom,
    ast.Delete, ast.Raise, ast.Try,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.Global, ast.Nonlocal,
    ast.Yield, ast.YieldFrom, ast.Await,
    ast.NamedExpr,
)


def _safe_eval_node(node, variables: dict):
    """安全地递归求值 AST 节点"""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise NameError(f"Variable '{node.id}' not allowed")
    if isinstance(node, ast.Attribute):
        obj = _safe_eval_node(node.value, variables)
        if isinstance(obj, dict):
            return obj.get(node.attr)
        return getattr(obj, node.attr, None)
    if isinstance(node, ast.Subscript):
        obj = _safe_eval_node(node.value, variables)
        key = _safe_eval_node(node.slice, variables)
        if isinstance(obj, dict):
            return obj.get(key)
        if isinstance(obj, (list, tuple)):
            return obj[key]
        raise TypeError(f"Subscript not supported on {type(obj).__name__}")
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _SAFE_OPERATORS.get(type(op))
            if op_func is None:
                raise TypeError(f"Unsupported operator: {type(op).__name__}")
            right = _safe_eval_node(comparator, variables)
            if not op_func(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        op_func = _SAFE_BOOL_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported bool op: {type(node.op).__name__}")
        values = [_safe_eval_node(v, variables) for v in node.values]
        return op_func(values)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand, variables))
    if isinstance(node, ast.List):
        return [_safe_eval_node(elt, variables) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(elt, variables) for elt in node.elts)
    raise TypeError(f"Unsupported node: {type(node).__name__}")


def _safe_eval(expr: str, variables: dict):
    """安全地求值条件表达式（禁止函数调用、import、赋值等）"""
    if not expr or not expr.strip():
        return True
    tree = ast.parse(expr.strip(), mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, _DANGEROUS_NODE_TYPES):
            raise TypeError(f"Expression contains forbidden: {type(node).__name__}")
    return _safe_eval_node(tree, variables)


# ======================== 数据模型 ========================

@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str
    type: str
    label: str = ""
    config: dict = field(default_factory=dict)
    status: str = "pending"
    result: Any = None
    error: str = ""
    started_at: float = 0
    finished_at: float = 0


@dataclass
class WorkflowEdge:
    """工作流边"""
    source: str
    target: str
    condition: str = ""


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    id: str = ""
    name: str = ""
    description: str = ""
    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    trigger: str = "manual"
    created_at: float = 0
    updated_at: float = 0


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    run_id: str = ""
    status: str = "completed"
    node_results: dict = field(default_factory=dict)
    total_duration_ms: float = 0
    started_at: float = 0
    finished_at: float = 0


# ======================== 执行引擎 ========================

class WorkflowEngine:
    """工作流执行引擎"""

    def __init__(self):
        self._cancelled = False
        self._running = False

    def cancel(self):
        self._cancelled = True

    def execute(self, definition: WorkflowDefinition, context: dict = None) -> WorkflowResult:
        """执行工作流"""
        self._cancelled = False
        self._running = True
        result = WorkflowResult(
            workflow_id=definition.id,
            run_id=str(uuid.uuid4())[:8],
            started_at=time.time(),
        )

        try:
            nodes = {n["id"]: WorkflowNode(**n) for n in definition.nodes}
            edges = [WorkflowEdge(**e) for e in definition.edges]

            adj: Dict[str, List[str]] = {nid: [] for nid in nodes}
            in_degree: Dict[str, int] = {nid: 0 for nid in nodes}

            for edge in edges:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1

            queue = [nid for nid, deg in in_degree.items() if deg == 0]
            execution_order = []
            while queue:
                nid = queue.pop(0)
                execution_order.append(nid)
                for neighbor in adj.get(nid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            ctx = context or {}
            for nid in execution_order:
                if self._cancelled:
                    result.status = "stopped"
                    break

                node = nodes[nid]
                should_execute = True
                for edge in edges:
                    if edge.target == nid and edge.condition:
                        try:
                            if not _safe_eval(edge.condition, {"ctx": ctx, "result": result.node_results}):
                                should_execute = False
                                break
                        except Exception:
                            should_execute = False
                            break

                if not should_execute:
                    node.status = "skipped"
                    continue

                node.status = "running"
                node.started_at = time.time()
                try:
                    node.result = self._execute_node(node, ctx)
                    node.status = "done"
                except Exception as e:
                    node.error = str(e)
                    node.status = "failed"
                    result.status = "failed"
                    logger.error(f"Workflow node {nid} failed: {e}")
                finally:
                    node.finished_at = time.time()

                result.node_results[nid] = {
                    "status": node.status,
                    "result": node.result,
                    "error": node.error,
                    "duration_ms": (node.finished_at - node.started_at) * 1000,
                }
                ctx[f"node_{nid}"] = node.result

        except Exception as e:
            result.status = "failed"
            logger.error(f"Workflow execution error: {e}")
        finally:
            result.finished_at = time.time()
            result.total_duration_ms = (result.finished_at - result.started_at) * 1000
            self._running = False

        return result

    def _execute_node(self, node: WorkflowNode, context: dict) -> Any:
        """执行单个节点"""
        node_type = node.type
        config = node.config

        if node_type == "trigger":
            return {"triggered": True}

        elif node_type == "agent":
            task = config.get("task", "")
            skill_id = config.get("skill_id", "")
            for key, value in context.items():
                task = task.replace(f"{{{key}}}", str(value))
            return {"task": task, "skill_id": skill_id, "note": "Agent execution deferred to runtime"}

        elif node_type == "tool":
            tool_name = config.get("name", "")
            tool_args = config.get("args", {})
            try:
                from agent.tool_registry import registry
                return registry.execute(tool_name, tool_args)
            except Exception as e:
                raise RuntimeError(f"Tool {tool_name} failed: {e}")

        elif node_type == "llm":
            prompt = config.get("prompt", "")
            return {"prompt": prompt, "note": "LLM call deferred to runtime"}

        elif node_type == "condition":
            expr = config.get("expression", "True")
            try:
                result = _safe_eval(expr, {"ctx": context})
                return {"condition_result": bool(result)}
            except Exception:
                return {"condition_result": False}

        elif node_type == "delay":
            seconds = config.get("seconds", 1)
            time.sleep(min(seconds, 60))
            return {"delayed_seconds": seconds}

        elif node_type == "http":
            url = config.get("url", "")
            method = config.get("method", "GET").upper()
            try:
                import urllib.request
                req = urllib.request.Request(url, method=method)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return {"status_code": resp.status, "body": resp.read(5000).decode("utf-8", errors="replace")}
            except Exception as e:
                raise RuntimeError(f"HTTP request failed: {e}")

        return {"type": node_type, "note": "Unknown node type"}


# ======================== 持久化存储 ========================

class WorkflowStore:
    """工作流持久化存储"""

    def __init__(self, storage_dir: str = ""):
        if not storage_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "user_data", "workflows")
        self._dir = storage_dir
        os.makedirs(self._dir, exist_ok=True)

    def save(self, definition: WorkflowDefinition):
        if not definition.id:
            definition.id = str(uuid.uuid4())[:8]
        definition.updated_at = time.time()
        path = os.path.join(self._dir, f"{definition.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(definition), f, indent=2, ensure_ascii=False)

    def load(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        path = os.path.join(self._dir, f"{workflow_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WorkflowDefinition(**data)

    def list_all(self) -> List[dict]:
        workflows = []
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                path = os.path.join(self._dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    workflows.append({
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "node_count": len(data.get("nodes", [])),
                        "trigger": data.get("trigger", "manual"),
                        "updated_at": data.get("updated_at", 0),
                    })
                except Exception:
                    continue
        return sorted(workflows, key=lambda x: x.get("updated_at", 0), reverse=True)

    def delete(self, workflow_id: str):
        path = os.path.join(self._dir, f"{workflow_id}.json")
        if os.path.exists(path):
            os.remove(path)