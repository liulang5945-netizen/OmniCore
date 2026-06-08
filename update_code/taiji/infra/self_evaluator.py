"""
态极自我评估能力 (Self Evaluator)
===================================

态极的新能力 #3：评估自己的表现。

每次任务完成后自动评估质量，分析推理链，
生成改进建议，让进化更有方向。
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("SelfEvaluator")


@dataclass
class StepEvaluation:
    """单步评估"""
    step_num: int
    action: str
    thought_quality: float  # 0-1
    action_quality: float   # 0-1
    was_effective: bool
    improvement_hint: str = ""


@dataclass
class TaskEvaluation:
    """任务整体评估"""
    task: str
    timestamp: str
    overall_score: float        # 0-1
    efficiency_score: float     # 0-1 (步骤数越少越好)
    correctness_score: float    # 0-1
    step_evaluations: List[StepEvaluation] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SelfEvaluator:
    """
    态极的自我评估引擎
    
    在每次任务完成后：
    1. 评估每个步骤的质量
    2. 分析整体表现
    3. 识别优势和弱点
    4. 生成改进建议
    5. 将结果反馈给进化引擎
    """
    
    def __init__(self):
        self._evaluation_history: List[TaskEvaluation] = []
    
    def evaluate_task(
        self,
        task: str,
        steps: List[dict],
        final_answer: str,
        was_successful: bool,
        max_steps: int = 10,
    ) -> TaskEvaluation:
        """
        评估一个任务的完成质量。
        
        Args:
            task: 用户任务
            steps: 执行步骤列表
            final_answer: 最终回答
            was_successful: 是否成功
            max_steps: 最大允许步骤数
            
        Returns:
            TaskEvaluation 评估结果
        """
        step_evals = []
        
        for i, step in enumerate(steps):
            step_eval = self._evaluate_step(step, i, len(steps), was_successful)
            step_evals.append(step_eval)
        
        # 计算各项分数
        efficiency = self._calc_efficiency(len(steps), max_steps)
        correctness = 1.0 if was_successful else 0.3
        
        # 如果有步骤评估，取平均
        if step_evals:
            thought_avg = sum(s.thought_quality for s in step_evals) / len(step_evals)
            action_avg = sum(s.action_quality for s in step_evals) / len(step_evals)
            overall = (thought_avg * 0.3 + action_avg * 0.3 + efficiency * 0.2 + correctness * 0.2)
        else:
            overall = correctness
        
        # 识别优势和弱点
        strengths, weaknesses = self._identify_patterns(step_evals, was_successful)
        
        # 生成改进建议
        recommendations = self._generate_recommendations(
            step_evals, efficiency, correctness, was_successful
        )
        
        evaluation = TaskEvaluation(
            task=task[:200],
            timestamp=datetime.now().isoformat(),
            overall_score=round(overall, 3),
            efficiency_score=round(efficiency, 3),
            correctness_score=round(correctness, 3),
            step_evaluations=step_evals,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )
        
        self._evaluation_history.append(evaluation)
        
        # 只保留最近 100 次评估
        if len(self._evaluation_history) > 100:
            self._evaluation_history = self._evaluation_history[-100:]
        
        return evaluation
    
    def _evaluate_step(self, step: dict, index: int, total: int, success: bool) -> StepEvaluation:
        """评估单个步骤（多维度信号）"""
        action = step.get("action", "")
        thought = step.get("thought", "")
        observation = step.get("observation", "")

        # 思考质量：结构化程度 + 信息量
        thought_quality = 0.3  # 基线
        if thought:
            thought_len = len(thought)
            # 长度维度
            if thought_len > 20:
                thought_quality = 0.5
            if thought_len > 50:
                thought_quality = 0.6
            if thought_len > 100:
                thought_quality = 0.7
            # 结构化关键词（中英文）
            structure_keywords = [
                "分析", "因为", "所以", "首先", "然后", "总结", "因此",
                "需要", "应该", "可以", "问题", "方案",
                "analysis", "because", "therefore", "first", "then",
            ]
            kw_count = sum(1 for kw in structure_keywords if kw in thought.lower())
            thought_quality = min(thought_quality + kw_count * 0.05, 1.0)

        # 行动质量：工具选择合理性 + 参数完整性
        action_quality = 0.3  # 无行动基线
        if action:
            action_quality = 0.6  # 有工具调用
            # 有参数比没有好
            args = step.get("action_args", {})
            if args and isinstance(args, dict) and len(args) > 0:
                action_quality = 0.75
            # 首步/末步特殊处理
            if index == total - 1 and action in ("answer", "respond"):
                action_quality = 0.85  # 最后一步用 answer 收尾是好的

        # 有效性：综合任务结果和观察内容
        was_effective = False
        if success:
            was_effective = True
        elif observation:
            # 观察中没有错误标记 → 可能有效
            error_markers = ["error", "Error", "错误", "失败", "failed", "Traceback"]
            if not any(m in observation for m in error_markers):
                was_effective = bool(action)

        # 改进建议
        hint = ""
        if not thought:
            hint = "思考过程缺失，应该先分析再行动"
        elif not action and index < total - 1:
            hint = "中间步骤没有工具调用，可能遗漏了执行"
        elif action and not thought:
            hint = "直接行动缺少推理，容易出错"
        elif observation and "error" in observation.lower() and "retry" not in thought.lower():
            hint = "遇到错误但未考虑重试或替代方案"

        return StepEvaluation(
            step_num=index + 1,
            action=action or "(无工具调用)",
            thought_quality=thought_quality,
            action_quality=action_quality,
            was_effective=was_effective,
            improvement_hint=hint,
        )
    
    def _calc_efficiency(self, actual_steps: int, max_steps: int) -> float:
        """计算效率分数"""
        if actual_steps <= 1:
            return 1.0
        ratio = actual_steps / max_steps
        return max(1.0 - ratio, 0.0)
    
    def _identify_patterns(
        self, step_evals: List[StepEvaluation], success: bool
    ) -> tuple:
        """识别优势和弱点"""
        strengths = []
        weaknesses = []
        
        if not step_evals:
            return strengths, weaknesses
        
        avg_thought = sum(s.thought_quality for s in step_evals) / len(step_evals)
        
        if avg_thought > 0.7:
            strengths.append("思考过程清晰详细")
        else:
            weaknesses.append("思考过程不够深入")
        
        if len(step_evals) <= 3 and success:
            strengths.append("高效完成任务，步骤精简")
        elif len(step_evals) > 5:
            weaknesses.append("步骤过多，可能有冗余操作")
        
        if all(s.was_effective for s in step_evals if s.action != "(无工具调用)"):
            strengths.append("工具选择准确")
        
        no_action_steps = sum(1 for s in step_evals if s.action == "(无工具调用)")
        if no_action_steps > len(step_evals) * 0.3:
            weaknesses.append("过多步骤没有工具调用")
        
        return strengths, weaknesses
    
    def _generate_recommendations(
        self, step_evals, efficiency, correctness, success
    ) -> List[str]:
        """生成改进建议"""
        recs = []
        
        if not success:
            recs.append("任务失败，建议先分析错误原因再重试")
        
        if efficiency < 0.5:
            recs.append("效率较低，尝试减少不必要的步骤")
        
        if step_evals:
            avg_thought = sum(s.thought_quality for s in step_evals) / len(step_evals)
            if avg_thought < 0.6:
                recs.append("建议在行动前更详细地分析问题")
        
        return recs
    
    def get_stats(self) -> dict:
        """获取评估统计"""
        if not self._evaluation_history:
            return {"total_evaluations": 0}
        
        scores = [e.overall_score for e in self._evaluation_history]
        return {
            "total_evaluations": len(self._evaluation_history),
            "avg_score": round(sum(scores) / len(scores), 3),
            "best_score": round(max(scores), 3),
            "worst_score": round(min(scores), 3),
            "success_rate": round(
                sum(1 for e in self._evaluation_history if e.correctness_score > 0.5)
                / len(self._evaluation_history), 3
            ),
        }
    
    def get_improvement_trends(self) -> List[str]:
        """分析改进趋势"""
        if len(self._evaluation_history) < 5:
            return ["数据不足，至少需要 5 次评估"]
        
        recent = self._evaluation_history[-10:]
        older = self._evaluation_history[-20:-10] if len(self._evaluation_history) >= 20 else self._evaluation_history[:10]
        
        recent_avg = sum(e.overall_score for e in recent) / len(recent)
        older_avg = sum(e.overall_score for e in older) / len(older)
        
        trends = []
        if recent_avg > older_avg + 0.05:
            trends.append(f"✅ 表现提升中 ({older_avg:.2f} → {recent_avg:.2f})")
        elif recent_avg < older_avg - 0.05:
            trends.append(f"⚠️ 表现下降 ({older_avg:.2f} → {recent_avg:.2f})")
        else:
            trends.append(f"➡️ 表现稳定 ({recent_avg:.2f})")
        
        return trends


# ─── 全局实例 ─────────────────────────────────────

_global_evaluator: Optional[SelfEvaluator] = None


def get_self_evaluator() -> SelfEvaluator:
    """获取全局自我评估器实例"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = SelfEvaluator()
    return _global_evaluator