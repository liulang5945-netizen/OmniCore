"""
Agent 任务规划与上下文管理系统
包含：任务计划 CRUD、开发上下文保存/加载
"""
import datetime
import json
import logging
import os
import threading
import uuid
from typing import Dict, Optional

from core.utils import get_external_path

logger = logging.getLogger("AgentPlanner")

_WORKSPACE_DIR = None

def _get_workspace() -> str:
    global _WORKSPACE_DIR
    if _WORKSPACE_DIR is None:
        _WORKSPACE_DIR = get_external_path("agent_workspace")
        os.makedirs(_WORKSPACE_DIR, exist_ok=True)
    return _WORKSPACE_DIR


# ======================== 任务规划系统 ========================

_plans: Dict[str, dict] = {}
_plans_lock = threading.Lock()


def _generate_plan_id() -> str:
    return f"plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def _save_plans():
    """持久化保存计划到文件（线程安全）"""
    try:
        plans_file = os.path.join(_get_workspace(), "_agent_plans.json")
        with _plans_lock:
            plans_copy = dict(_plans)
        with open(plans_file, "w", encoding="utf-8") as f:
            json.dump(plans_copy, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存计划失败: {e}")


def _load_plans():
    """从文件加载计划（线程安全）"""
    global _plans
    try:
        plans_file = os.path.join(_get_workspace(), "_agent_plans.json")
        if os.path.exists(plans_file):
            with open(plans_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            with _plans_lock:
                _plans = loaded
    except Exception as e:
        logger.warning(f"加载计划失败: {e}")
        with _plans_lock:
            _plans = {}


def create_plan(task_description: str, steps_str: str) -> str:
    """为复杂任务创建执行计划。"""
    try:
        _load_plans()
        parts = task_description.split("|", 1)
        description = parts[0].strip()
        steps_text = parts[1].strip() if len(parts) > 1 else steps_str.strip()

        steps = []
        for line in steps_text.split("\n"):
            line = line.strip()
            if line:
                step_text = line
                for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.",
                               "- ", "* ", "• ", "# ", "## ", "### "]:
                    if step_text.startswith(prefix):
                        step_text = step_text[len(prefix):].strip()
                        break
                steps.append({"text": step_text, "status": "pending", "detail": ""})

        if not steps:
            steps = [{"text": steps_text, "status": "pending", "detail": ""}]

        plan_id = _generate_plan_id()
        _plans[plan_id] = {
            "id": plan_id,
            "description": description,
            "steps": steps,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "in_progress",
            "current_step": 0,
            "notes": "",
        }
        _save_plans()

        steps_summary = "\n".join(
            f"  {i+1}. {'[ ]' if s['status']=='pending' else '[x]'} {s['text']}"
            for i, s in enumerate(steps)
        )
        return (
            f"✅ 任务计划已创建 (ID: {plan_id})\n\n"
            f"📋 任务: {description}\n"
            f"📝 步骤 ({len(steps)} 步):\n{steps_summary}\n\n"
            f"开始执行第 1 步..."
        )
    except Exception as e:
        return f"❌ 创建计划失败: {e}"


def update_plan(input_str: str) -> str:
    """更新计划中某一步的状态。"""
    try:
        _load_plans()
        parts = [p.strip() for p in input_str.split("|")]
        if len(parts) < 3:
            return "错误: 输入格式必须为 `plan_id | 步骤序号 | 新状态 | 备注(可选)`"

        plan_id = parts[0]
        if plan_id not in _plans:
            return f"错误: 找不到计划 '{plan_id}'"

        plan = _plans[plan_id]
        target = parts[1]
        new_status = parts[2].lower()
        note = parts[3] if len(parts) > 3 else ""

        valid_statuses = {"done", "pending", "failed", "skip", "in_progress"}
        if new_status not in valid_statuses:
            return f"错误: 无效状态 '{new_status}'，可选: {', '.join(valid_statuses)}"

        if target == "status":
            plan["status"] = new_status
            plan["notes"] = note or plan.get("notes", "")
            _save_plans()
            return f"✅ 任务计划 '{plan_id}' 状态已更新为 '{new_status}'"

        step_idx = int(target) - 1
        if step_idx < 0 or step_idx >= len(plan["steps"]):
            return f"错误: 步骤序号 {target} 超出范围 (1-{len(plan['steps'])})"

        plan["steps"][step_idx]["status"] = new_status
        if note:
            plan["steps"][step_idx]["detail"] = note
        plan["current_step"] = step_idx + 1

        all_done = all(s["status"] == "done" for s in plan["steps"])
        any_failed = any(s["status"] == "failed" for s in plan["steps"])

        if all_done:
            plan["status"] = "completed"
            result = f"✅ 所有步骤已完成！任务计划 '{plan_id}' 执行完毕！"
        elif any_failed:
            plan["status"] = "failed"
            result = f"⚠️ 部分步骤失败，任务计划 '{plan_id}' 中止。"
        else:
            plan["status"] = "in_progress"
            next_step = step_idx + 2
            while next_step <= len(plan["steps"]):
                if plan["steps"][next_step - 1]["status"] == "pending":
                    break
                next_step += 1
            if next_step <= len(plan["steps"]):
                result = f"✅ 步骤 {target} 标记为 '{new_status}'。继续执行第 {next_step} 步..."
            else:
                result = f"✅ 步骤 {target} 标记为 '{new_status}'。"

        _save_plans()
        return result
    except ValueError:
        return "错误: 步骤序号必须是数字"
    except Exception as e:
        return f"❌ 更新计划失败: {e}"


def get_plan(plan_id: str = "") -> str:
    """获取计划详情。"""
    try:
        _load_plans()
        if not _plans:
            return "当前没有活跃的任务计划。"

        if plan_id and plan_id != "all":
            if plan_id not in _plans:
                return f"错误: 找不到计划 '{plan_id}'"
            plans_to_show = {plan_id: _plans[plan_id]}
        else:
            plans_to_show = _plans

        results = []
        for pid, plan in plans_to_show.items():
            status_icon = {"in_progress": "🔄", "completed": "✅", "failed": "❌", "pending": "⏳"}
            icon = status_icon.get(plan["status"], "📋")
            steps_text = "\n".join(
                f"  {i+1}. {'✅' if s['status']=='done' else '❌' if s['status']=='failed' else '⏭️' if s['status']=='skip' else '⬜' if s['status']=='pending' else '🔄'} {s['text']}"
                + (f" - {s['detail']}" if s.get("detail") else "")
                for i, s in enumerate(plan["steps"])
            )
            notes = f"\n📌 备注: {plan['notes']}" if plan.get("notes") else ""
            results.append(
                f"{icon} 计划: {plan['description']} (ID: {pid})\n"
                f"状态: {plan['status']} | 创建: {plan['created_at']}\n"
                f"{steps_text}{notes}"
            )

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"❌ 获取计划失败: {e}"


def list_plans(_: str = "") -> str:
    """列出所有任务计划。"""
    return get_plan("all")


# ======================== 开发上下文管理 ========================

_context_store: Dict[str, str] = {}

def _save_context_store():
    try:
        ctx_file = os.path.join(_get_workspace(), "_agent_context.json")
        with open(ctx_file, "w", encoding="utf-8") as f:
            json.dump(_context_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_context_store():
    global _context_store
    try:
        ctx_file = os.path.join(_get_workspace(), "_agent_context.json")
        if os.path.exists(ctx_file):
            with open(ctx_file, "r", encoding="utf-8") as f:
                _context_store = json.load(f)
    except Exception:
        _context_store = {}


def save_context(input_str: str) -> str:
    """保存开发上下文信息（跨步骤传递数据）。"""
    try:
        _load_context_store()
        parts = [p.strip() for p in input_str.split("|", 1)]
        if len(parts) < 2:
            return "错误: 输入格式必须为 `key | value`"
        key, value = parts
        _context_store[key] = value
        _save_context_store()
        return f"✅ 已保存上下文: {key} = {value[:100]}{'...' if len(value) > 100 else ''}"
    except Exception as e:
        return f"❌ 保存上下文失败: {e}"


def load_context(input_str: str = "") -> str:
    """读取所有已保存的上下文信息。"""
    try:
        _load_context_store()
        if not _context_store:
            return "当前没有保存的开发上下文。"

        if input_str and input_str.strip():
            key = input_str.strip()
            if key in _context_store:
                return f"{key} = {_context_store[key]}"
            return f"未找到上下文 key: '{key}'"

        lines = []
        for k, v in _context_store.items():
            v_str = v[:200] + "..." if len(v) > 200 else v
            lines.append(f"  {k} = {v_str}")
        return "📋 开发上下文:\n" + "\n".join(lines)
    except Exception as e:
        return f"❌ 读取上下文失败: {e}"