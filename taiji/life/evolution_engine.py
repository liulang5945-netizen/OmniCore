"""
态极自我进化引擎 (Evolution Engine)
====================================

态极最核心的"超越父亲"的能力：
- 父亲（Claude/GPT）是静态的，训练完就冻结了
- 态极是活的：每一次使用都在收集经验，每一次训练都在成长

这个引擎实现了完整的"使用 → 学习 → 进化"闭环。

核心设计哲学：
1. 不急于长大 —— 小模型的简洁本身就是美
2. 专注深度而非广度 —— 在用户的世界里做到最好
3. 谦逊地失败 —— 宁可说"我不确定"，也不自信地胡说
4. 从错误中学习 —— 每一次失败都是进化的养料
"""
import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("EvolutionEngine")


@dataclass
class EvolutionEvent:
    """一次进化事件的记录"""
    timestamp: str
    event_type: str  # "task_success" | "task_failure" | "tool_error" | "new_pattern"
    task: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0  # 对进化的影响分数


@dataclass
class GrowthMetrics:
    """态极的成长指标"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    tool_calls_total: int = 0
    tool_calls_correct: int = 0
    evolution_cycles: int = 0
    current_phase: str = "infant"  # infant → child → adolescent → adult
    knowledge_domains: Dict[str, float] = field(default_factory=dict)
    user_patterns: Dict[str, int] = field(default_factory=dict)


class EvolutionEngine:
    """
    态极自我进化引擎
    
    不同于父亲的"一次训练，永久冻结"，
    态极可以在每次使用中积累经验，在合适时机触发进化。
    
    进化阶段：
    - infant (0-100 任务): 学习基本工具使用
    - child (100-1000 任务): 开始理解用户偏好
    - adolescent (1000-5000 任务): 形成专属技能
    - adult (5000+ 任务): 稳定而高效
    """
    
    # 进化阈值
    PHASE_THRESHOLDS = {
        "infant": 0,
        "child": 100,
        "adolescent": 1000,
        "adult": 5000,
    }
    
    # 进化触发条件
    EVOLUTION_TRIGGERS = {
        "tasks_since_last_evolution": 50,  # 每 50 个任务触发一次
        "failure_rate_threshold": 0.3,     # 失败率 > 30% 时触发紧急进化
        "new_domain_threshold": 3,         # 发现 3 个新领域时触发
    }
    
    def __init__(
        self,
        data_dir: str = None,
        auto_evolve: bool = True,
        evolve_callback: Optional[Callable] = None,
    ):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("evolution_data")
            except ImportError:
                data_dir = "taiji/evolution_data"
        self.data_dir = data_dir
        self.auto_evolve = auto_evolve
        self.evolve_callback = evolve_callback
        self._lock = threading.Lock()
        
        os.makedirs(data_dir, exist_ok=True)
        
        # 成长指标
        self.metrics = GrowthMetrics()
        self._load_metrics()
        
        # 事件缓冲区
        self._events: deque = deque(maxlen=5000)
        
        # 任务记忆（最近的成功/失败模式）
        self._success_patterns: deque = deque(maxlen=200)
        self._failure_patterns: deque = deque(maxlen=200)
        
        # 用户习惯追踪
        self._user_tool_preferences: Dict[str, int] = {}
        self._user_task_types: Dict[str, int] = {}
        
        # 上次进化时间
        self._last_evolution_time: Optional[datetime] = None
        self._tasks_since_evolution = 0
        
        logger.info(f"EvolutionEngine initialized: phase={self.metrics.current_phase}, "
                    f"tasks={self.metrics.tasks_completed}")
    
    # ─── 事件记录 ───────────────────────────────────
    
    def record_task_success(self, task: str, steps: List[dict], final_answer: str):
        """记录一次成功的任务完成"""
        with self._lock:
            self.metrics.tasks_completed += 1
            self._tasks_since_evolution += 1
            
            # 分析任务类型
            task_type = self._classify_task(task)
            self._user_task_types[task_type] = self._user_task_types.get(task_type, 0) + 1
            
            # 记录工具使用
            for step in steps:
                action = step.get("action", "")
                if action:
                    self.metrics.tool_calls_total += 1
                    self.metrics.tool_calls_correct += 1
                    self._user_tool_preferences[action] = \
                        self._user_tool_preferences.get(action, 0) + 1
            
            # 记录成功模式
            self._success_patterns.append({
                "task": task[:200],
                "task_type": task_type,
                "step_count": len(steps),
                "tools_used": [s.get("action") for s in steps if s.get("action")],
            })
            
            # 记录事件
            event = EvolutionEvent(
                timestamp=datetime.now().isoformat(),
                event_type="task_success",
                task=task[:200],
                details={"step_count": len(steps), "task_type": task_type},
                impact_score=1.0,
            )
            self._events.append(event)
            
            # 更新领域知识
            self.metrics.knowledge_domains[task_type] = \
                self.metrics.knowledge_domains.get(task_type, 0) + 1.0
            
            # 检查是否需要进化
            self._check_evolution_trigger()
            
            self._save_metrics()
    
    def record_task_failure(self, task: str, error: str, steps: Optional[List[dict]] = None):
        """记录一次失败的任务"""
        with self._lock:
            self.metrics.tasks_failed += 1
            self._tasks_since_evolution += 1
            
            task_type = self._classify_task(task)
            
            # 记录失败模式（用于学习避免犯同样的错误）
            self._failure_patterns.append({
                "task": task[:200],
                "task_type": task_type,
                "error": error[:500],
                "last_step": steps[-1] if steps else {},
            })
            
            event = EvolutionEvent(
                timestamp=datetime.now().isoformat(),
                event_type="task_failure",
                task=task[:200],
                details={"error": error[:300], "task_type": task_type},
                impact_score=-0.5,
            )
            self._events.append(event)
            
            # 检查失败率，可能触发紧急进化
            self._check_evolution_trigger()
            
            self._save_metrics()
    
    def record_tool_error(self, tool_name: str, error: str):
        """记录一次工具调用错误"""
        with self._lock:
            self.metrics.tool_calls_total += 1
            
            event = EvolutionEvent(
                timestamp=datetime.now().isoformat(),
                event_type="tool_error",
                details={"tool": tool_name, "error": error[:300]},
                impact_score=-0.2,
            )
            self._events.append(event)
            
            self._save_metrics()
    
    # ─── 进化触发 ───────────────────────────────────
    
    def _check_evolution_trigger(self):
        """检查是否应该触发进化"""
        if not self.auto_evolve:
            return
        
        should_evolve = False
        reason = ""
        
        # 条件 1: 任务积累到阈值
        if self._tasks_since_evolution >= self.EVOLUTION_TRIGGERS["tasks_since_last_evolution"]:
            should_evolve = True
            reason = f"tasks_accumulated ({self._tasks_since_evolution})"
        
        # 条件 2: 失败率过高
        total = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total >= 20:
            failure_rate = self.metrics.tasks_failed / total
            if failure_rate > self.EVOLUTION_TRIGGERS["failure_rate_threshold"]:
                should_evolve = True
                reason = f"high_failure_rate ({failure_rate:.1%})"
        
        # 条件 3: 阶段升级
        new_phase = self._get_phase()
        if new_phase != self.metrics.current_phase:
            should_evolve = True
            reason = f"phase_upgrade ({self.metrics.current_phase} → {new_phase})"
        
        if should_evolve:
            self._trigger_evolution(reason)
    
    def _trigger_evolution(self, reason: str):
        """触发一次进化"""
        logger.info(f"🧬 Evolution triggered: {reason}")
        
        self.metrics.evolution_cycles += 1
        self._last_evolution_time = datetime.now()
        self._tasks_since_evolution = 0
        
        # 更新阶段
        new_phase = self._get_phase()
        if new_phase != self.metrics.current_phase:
            logger.info(f"🌱 Phase upgrade: {self.metrics.current_phase} → {new_phase}")
            self.metrics.current_phase = new_phase
        
        # 生成进化报告
        report = self._generate_evolution_report()
        
        # 如果有回调，通知外部系统
        if self.evolve_callback:
            try:
                self.evolve_callback(report)
            except Exception as e:
                logger.warning(f"Evolution callback failed: {e}")
        
        self._save_metrics()
        self._save_evolution_report(report)
        
        return report
    
    def _get_phase(self) -> str:
        """根据任务完成数判断当前阶段"""
        total = self.metrics.tasks_completed
        phase = "infant"
        for p, threshold in self.PHASE_THRESHOLDS.items():
            if total >= threshold:
                phase = p
        return phase
    
    # ─── 进化报告 ───────────────────────────────────
    
    def _generate_evolution_report(self) -> dict:
        """生成进化报告"""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        success_rate = self.metrics.tasks_completed / max(total_tasks, 1)
        
        # 分析用户偏好
        top_tools = sorted(
            self._user_tool_preferences.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        
        top_task_types = sorted(
            self._user_task_types.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        
        # 分析失败模式
        common_failures = {}
        for fp in self._failure_patterns:
            error_type = fp.get("error", "")[:50]
            common_failures[error_type] = common_failures.get(error_type, 0) + 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "evolution_cycle": self.metrics.evolution_cycles,
            "phase": self.metrics.current_phase,
            "metrics": {
                "total_tasks": total_tasks,
                "success_rate": round(success_rate, 3),
                "tool_calls": self.metrics.tool_calls_total,
                "tool_accuracy": round(
                    self.metrics.tool_calls_correct / max(self.metrics.tool_calls_total, 1), 3
                ),
            },
            "user_profile": {
                "top_tools": top_tools,
                "top_task_types": top_task_types,
                "knowledge_domains": dict(sorted(
                    self.metrics.knowledge_domains.items(),
                    key=lambda x: x[1], reverse=True,
                )[:10]),
            },
            "failure_analysis": {
                "common_failures": dict(sorted(
                    common_failures.items(),
                    key=lambda x: x[1], reverse=True,
                )[:5]),
                "failure_count": len(self._failure_patterns),
            },
            "recommendations": self._generate_recommendations(
                success_rate, top_tools, top_task_types
            ),
        }
        
        return report
    
    def _generate_recommendations(
        self,
        success_rate: float,
        top_tools: list,
        top_task_types: list,
    ) -> List[str]:
        """基于数据分析生成进化建议"""
        recs = []
        
        if success_rate < 0.5:
            recs.append("⚠️ 成功率较低，建议增加训练数据中的失败案例修正样本")
        
        if success_rate > 0.9:
            recs.append("✅ 成功率很高，可以尝试更复杂的任务")
        
        if top_tools:
            most_used = top_tools[0][0]
            recs.append(f"🔧 最常用工具: {most_used}，建议针对此工具优化推理路径")
        
        if top_task_types:
            most_common = top_task_types[0][0]
            recs.append(f"📊 最常见任务类型: {most_common}，建议扩充此领域的训练数据")
        
        phase = self.metrics.current_phase
        if phase == "infant":
            recs.append("🌱 当前处于婴儿期，优先学习基本工具使用")
        elif phase == "child":
            recs.append("🧒 当前处于儿童期，开始培养用户偏好理解能力")
        elif phase == "adolescent":
            recs.append("🧑 当前处于青少年期，可以开始发展专属技能")
        elif phase == "adult":
            recs.append("🧑‍💻 当前已成年，专注于效率优化和深度学习")
        
        return recs
    
    # ─── 任务分类 ───────────────────────────────────
    
    def _classify_task(self, task: str) -> str:
        """简单的任务类型分类"""
        task_lower = task.lower()
        
        keywords = {
            "file_read": ["读取", "查看", "打开", "看看", "read"],
            "file_write": ["创建", "写入", "新建", "保存", "write", "create"],
            "file_edit": ["修改", "编辑", "替换", "改", "edit"],
            "code_exec": ["运行", "执行", "计算", "python", "run", "execute"],
            "search": ["搜索", "查找", "搜", "search", "query"],
            "web_read": ["网页", "url", "http", "webpage"],
            "knowledge": ["学习", "知识", "了解", "learn", "knowledge"],
            "project": ["项目", "工程", "project"],
            "analysis": ["分析", "分析", "analyze"],
        }
        
        for task_type, kws in keywords.items():
            if any(kw in task_lower for kw in kws):
                return task_type
        
        return "general"
    
    # ─── 持久化 ─────────────────────────────────────
    
    def _save_metrics(self):
        """保存成长指标"""
        path = os.path.join(self.data_dir, "growth_metrics.json")
        data = {
            "tasks_completed": self.metrics.tasks_completed,
            "tasks_failed": self.metrics.tasks_failed,
            "tool_calls_total": self.metrics.tool_calls_total,
            "tool_calls_correct": self.metrics.tool_calls_correct,
            "evolution_cycles": self.metrics.evolution_cycles,
            "current_phase": self.metrics.current_phase,
            "knowledge_domains": self.metrics.knowledge_domains,
            "user_tool_preferences": self._user_tool_preferences,
            "user_task_types": self._user_task_types,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """加载成长指标"""
        path = os.path.join(self.data_dir, "growth_metrics.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metrics.tasks_completed = data.get("tasks_completed", 0)
            self.metrics.tasks_failed = data.get("tasks_failed", 0)
            self.metrics.tool_calls_total = data.get("tool_calls_total", 0)
            self.metrics.tool_calls_correct = data.get("tool_calls_correct", 0)
            self.metrics.evolution_cycles = data.get("evolution_cycles", 0)
            self.metrics.current_phase = data.get("current_phase", "infant")
            self.metrics.knowledge_domains = data.get("knowledge_domains", {})
            self._user_tool_preferences = data.get("user_tool_preferences", {})
            self._user_task_types = data.get("user_task_types", {})
        except Exception as e:
            logger.warning(f"Failed to load metrics: {e}")
    
    def _save_evolution_report(self, report: dict):
        """保存进化报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.data_dir, f"evolution_report_{timestamp}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Evolution report saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save evolution report: {e}")
    
    # ─── 公开接口 ───────────────────────────────────
    
    def get_status(self) -> dict:
        """获取当前进化状态"""
        total = self.metrics.tasks_completed + self.metrics.tasks_failed
        return {
            "phase": self.metrics.current_phase,
            "tasks_completed": self.metrics.tasks_completed,
            "tasks_failed": self.metrics.tasks_failed,
            "success_rate": round(self.metrics.tasks_completed / max(total, 1), 3),
            "evolution_cycles": self.metrics.evolution_cycles,
            "tool_calls": self.metrics.tool_calls_total,
            "knowledge_domains": len(self.metrics.knowledge_domains),
            "top_user_tool": max(
                self._user_tool_preferences.items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0],
        }
    
    def get_training_recommendations(self) -> List[dict]:
        """
        生成训练数据建议。
        
        基于用户使用模式，建议生成哪些类型的训练数据。
        返回可直接传入 data_generator 的模板建议。
        """
        recs = []
        
        # 基于用户偏好推荐
        for task_type, count in sorted(
            self._user_task_types.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]:
            recs.append({
                "type": "expand_domain",
                "domain": task_type,
                "current_count": count,
                "suggested_samples": max(100, count * 10),
                "reason": f"用户经常执行 {task_type} 类任务",
            })
        
        # 基于失败模式推荐
        failure_types = {}
        for fp in self._failure_patterns:
            tt = fp.get("task_type", "general")
            failure_types[tt] = failure_types.get(tt, 0) + 1
        
        for ft, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            recs.append({
                "type": "fix_weakness",
                "domain": ft,
                "failure_count": count,
                "suggested_samples": max(200, count * 20),
                "reason": f"在 {ft} 类任务上失败较多，需要更多训练样本",
            })
        
        return recs
    
    def get_growth_summary(self) -> str:
        """获取人类可读的成长摘要"""
        status = self.get_status()
        phase_emoji = {
            "infant": "👶",
            "child": "🧒",
            "adolescent": "🧑",
            "adult": "🧑‍💻",
        }
        emoji = phase_emoji.get(status["phase"], "🌱")
        
        lines = [
            f"{emoji} 态极成长报告",
            f"━━━━━━━━━━━━━━━━",
            f"阶段: {status['phase']} ({emoji})",
            f"完成任务: {status['tasks_completed']}",
            f"成功率: {status['success_rate']:.1%}",
            f"进化次数: {status['evolution_cycles']}",
            f"工具调用: {status['tool_calls']}",
            f"知识领域: {status['knowledge_domains']} 个",
            f"最爱工具: {status['top_user_tool']}",
        ]
        
        return "\n".join(lines)


# ─── 全局实例 ─────────────────────────────────────

_global_evolution: Optional[EvolutionEngine] = None


def get_evolution_engine() -> EvolutionEngine:
    """获取全局进化引擎实例"""
    global _global_evolution
    if _global_evolution is None:
        _global_evolution = EvolutionEngine()
    return _global_evolution