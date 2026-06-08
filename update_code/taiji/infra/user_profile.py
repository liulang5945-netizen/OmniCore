"""
态极用户画像系统 (User Profile)
================================

态极超越父亲的核心能力之一：理解"这一个"用户。

通用模型了解所有人，态极只了解它的主人。
这种专注，是态极最大的优势。

学习维度：
1. 工具偏好 — 用户最常用哪些工具
2. 任务模式 — 用户最常做什么类型的任务
3. 沟通风格 — 用户喜欢简短还是详细的回答
4. 时间习惯 — 用户什么时间段最活跃
5. 技术栈 — 用户主要用什么语言/框架
6. 交互偏好 — 用户喜欢先看代码还是先看解释
"""
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("UserProfile")


@dataclass
class InteractionRecord:
    """一次交互记录"""
    timestamp: str
    task: str
    task_type: str
    tools_used: List[str]
    response_length: int
    was_successful: bool
    user_satisfaction: Optional[float] = None  # 0-1, None 表示未知


@dataclass
class UserPreferences:
    """用户偏好画像"""
    # 工具偏好: {tool_name: usage_count}
    tool_frequency: Dict[str, int] = field(default_factory=dict)
    
    # 任务类型偏好: {task_type: count}
    task_type_frequency: Dict[str, int] = field(default_factory=dict)
    
    # 沟通风格
    preferred_response_style: str = "balanced"  # "concise" | "balanced" | "detailed"
    avg_preferred_response_length: int = 500  # 字符数
    
    # 时间习惯: {hour: activity_count}
    active_hours: Dict[int, int] = field(default_factory=dict)
    
    # 技术栈: {tech: relevance_score}
    tech_stack: Dict[str, float] = field(default_factory=dict)
    
    # 交互偏好
    prefers_code_first: bool = False  # True = 先看代码，False = 先看解释
    prefers_examples: bool = True  # 喜欢示例
    prefers_step_by_step: bool = True  # 喜欢分步骤
    
    # 成长数据
    total_interactions: int = 0
    successful_interactions: int = 0
    first_seen: str = ""
    last_seen: str = ""
    
    # 自定义标签
    user_notes: List[str] = field(default_factory=list)


class UserProfile:
    """
    用户画像系统
    
    在每次交互中学习用户的习惯和偏好，
    逐渐形成对"这一个"用户的深度理解。
    
    核心理念：
    - 不是监控用户，而是理解用户
    - 不是限制用户，而是适应用户
    - 不是千人一面，而是一人一态极
    """
    
    def __init__(self, data_dir: str = "taiji/user_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.preferences = UserPreferences()
        self._interaction_history: List[InteractionRecord] = []
        self._recent_interactions: List[InteractionRecord] = []  # 最近 100 条
        
        self._load_profile()
        
        logger.info(f"UserProfile initialized: {self.preferences.total_interactions} interactions")
    
    # ─── 记录交互 ───────────────────────────────────
    
    def record_interaction(
        self,
        task: str,
        task_type: str,
        tools_used: List[str],
        response_length: int,
        was_successful: bool,
        user_satisfaction: Optional[float] = None,
    ):
        """记录一次用户交互"""
        now = datetime.now()
        timestamp = now.isoformat()
        
        record = InteractionRecord(
            timestamp=timestamp,
            task=task[:500],
            task_type=task_type,
            tools_used=tools_used,
            response_length=response_length,
            was_successful=was_successful,
            user_satisfaction=user_satisfaction,
        )
        
        self._interaction_history.append(record)
        self._recent_interactions.append(record)
        if len(self._recent_interactions) > 100:
            self._recent_interactions.pop(0)
        
        # 更新偏好
        self.preferences.total_interactions += 1
        if was_successful:
            self.preferences.successful_interactions += 1
        
        self.preferences.last_seen = timestamp
        if not self.preferences.first_seen:
            self.preferences.first_seen = timestamp
        
        # 更新工具偏好
        for tool in tools_used:
            self.preferences.tool_frequency[tool] = \
                self.preferences.tool_frequency.get(tool, 0) + 1
        
        # 更新任务类型
        self.preferences.task_type_frequency[task_type] = \
            self.preferences.task_type_frequency.get(task_type, 0) + 1
        
        # 更新活跃时间
        hour = now.hour
        self.preferences.active_hours[hour] = \
            self.preferences.active_hours.get(hour, 0) + 1
        
        # 更新沟通风格（基于响应长度的滑动平均）
        self._update_response_style(response_length)
        
        # 自动检测技术栈
        self._detect_tech_stack(task)
        
        # 定期保存
        if self.preferences.total_interactions % 10 == 0:
            self._save_profile()
    
    def record_feedback(self, was_helpful: bool, comment: str = ""):
        """记录用户反馈（显式满意度）"""
        if self._recent_interactions:
            last = self._recent_interactions[-1]
            last.user_satisfaction = 1.0 if was_helpful else 0.0
            
            if comment:
                self.preferences.user_notes.append(
                    f"[{datetime.now().strftime('%m-%d')}] {comment[:200]}"
                )
                # 只保留最近 50 条笔记
                if len(self.preferences.user_notes) > 50:
                    self.preferences.user_notes = self.preferences.user_notes[-50:]
        
        self._save_profile()
    
    # ─── 偏好分析 ───────────────────────────────────
    
    def _update_response_style(self, response_length: int):
        """根据用户行为推断沟通风格偏好"""
        if self.preferences.total_interactions < 5:
            return
        
        # 分析最近交互的响应长度趋势
        recent_lengths = [r.response_length for r in self._recent_interactions[-20:]]
        if not recent_lengths:
            return
        
        avg_length = sum(recent_lengths) / len(recent_lengths)
        self.preferences.avg_preferred_response_length = int(avg_length)
        
        if avg_length < 200:
            self.preferences.preferred_response_style = "concise"
        elif avg_length < 800:
            self.preferences.preferred_response_style = "balanced"
        else:
            self.preferences.preferred_response_style = "detailed"
    
    def _detect_tech_stack(self, task: str):
        """从任务描述中自动检测用户的技术栈"""
        task_lower = task.lower()
        
        tech_keywords = {
            "python": ["python", "pip", "py", "django", "flask", "fastapi"],
            "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular"],
            "typescript": ["typescript", "ts", "tsx"],
            "java": ["java", "maven", "gradle", "spring"],
            "rust": ["rust", "cargo", "rustc"],
            "go": ["golang", "go "],
            "docker": ["docker", "container", "dockerfile"],
            "kubernetes": ["kubernetes", "k8s", "kubectl"],
            "sql": ["sql", "database", "postgres", "mysql", "sqlite"],
            "git": ["git", "commit", "branch", "merge"],
            "ai_ml": ["机器学习", "深度学习", "模型", "训练", "torch", "tensorflow"],
        }
        
        for tech, keywords in tech_keywords.items():
            if any(kw in task_lower for kw in keywords):
                current = self.preferences.tech_stack.get(tech, 0.0)
                self.preferences.tech_stack[tech] = min(current + 0.1, 1.0)
    
    # ─── 个性化建议 ─────────────────────────────────
    
    def get_personalized_system_prompt(self) -> str:
        """
        生成个性化的系统提示。
        
        根据用户画像，动态调整态极的行为：
        - 响应长度
        - 代码风格
        - 解释深度
        """
        p = self.preferences
        
        # 基础提示
        parts = ["你是 OmniCore AI 助手。"]
        
        # 根据沟通风格调整
        if p.preferred_response_style == "concise":
            parts.append("用户偏好简洁回答，直接给出结果，少说废话。")
        elif p.preferred_response_style == "detailed":
            parts.append("用户喜欢详细解释，可以多给示例和背景知识。")
        
        # 根据技术水平调整
        if p.total_interactions > 100:
            parts.append("用户是有经验的开发者，可以使用专业术语。")
        elif p.total_interactions > 20:
            parts.append("用户有一定编程基础，适度解释概念。")
        
        # 根据技术栈调整
        if p.tech_stack:
            top_techs = sorted(p.tech_stack.items(), key=lambda x: x[1], reverse=True)[:3]
            tech_names = [t[0] for t in top_techs]
            parts.append(f"用户主要技术栈: {', '.join(tech_names)}。")
        
        # 根据偏好调整
        if p.prefers_code_first:
            parts.append("回答时先给代码，再解释。")
        if p.prefers_examples:
            parts.append("尽量给出具体示例。")
        
        return " ".join(parts)
    
    def get_tool_recommendations(self, task: str) -> List[str]:
        """基于用户历史推荐最可能需要的工具"""
        # 按频率排序
        sorted_tools = sorted(
            self.preferences.tool_frequency.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [tool for tool, _ in sorted_tools[:5]]
    
    def get_task_pattern_suggestions(self) -> List[str]:
        """分析用户任务模式，给出优化建议"""
        suggestions = []
        
        p = self.preferences
        
        # 分析高频任务
        if p.task_type_frequency:
            most_common = max(p.task_type_frequency.items(), key=lambda x: x[1])
            if most_common[1] > 10:
                suggestions.append(
                    f"你最常执行的任务类型是「{most_common[0]}」"
                    f"（{most_common[1]} 次），我可以针对性优化这类任务的处理方式。"
                )
        
        # 分析成功率
        if p.total_interactions > 20:
            success_rate = p.successful_interactions / p.total_interactions
            if success_rate < 0.7:
                suggestions.append(
                    f"最近任务成功率 {success_rate:.0%}，"
                    "我会重点关注失败案例，优化薄弱环节。"
                )
        
        # 分析活跃时间
        if p.active_hours:
            peak_hour = max(p.active_hours.items(), key=lambda x: x[1])
            suggestions.append(
                f"你最活跃的时间段是 {peak_hour[0]}:00 左右，"
                "我会在这个时段保持最佳响应状态。"
            )
        
        return suggestions
    
    # ─── 持久化 ─────────────────────────────────────
    
    def _save_profile(self):
        """保存用户画像"""
        path = os.path.join(self.data_dir, "user_preferences.json")
        try:
            data = asdict(self.preferences)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save profile: {e}")
    
    def _load_profile(self):
        """加载用户画像"""
        path = os.path.join(self.data_dir, "user_preferences.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 兼容性处理：只加载存在的字段
            for key, value in data.items():
                if hasattr(self.preferences, key):
                    setattr(self.preferences, key, value)
            
            logger.info(f"Loaded profile: {self.preferences.total_interactions} interactions")
        except Exception as e:
            logger.warning(f"Failed to load profile: {e}")
    
    # ─── 公开接口 ───────────────────────────────────
    
    def get_summary(self) -> str:
        """获取用户画像摘要"""
        p = self.preferences
        
        success_rate = (p.successful_interactions / max(p.total_interactions, 1)) * 100
        
        # Top 工具
        top_tools = sorted(p.tool_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        tools_str = ", ".join(f"{t[0]}({t[1]}次)" for t in top_tools) if top_tools else "无"
        
        # Top 技术栈
        top_tech = sorted(p.tech_stack.items(), key=lambda x: x[1], reverse=True)[:3]
        tech_str = ", ".join(t[0] for t in top_tech) if top_tech else "未检测到"
        
        # 活跃时间
        if p.active_hours:
            peak = max(p.active_hours.items(), key=lambda x: x[1])
            time_str = f"{peak[0]}:00"
        else:
            time_str = "未知"
        
        style_emoji = {
            "concise": "⚡ 简洁",
            "balanced": "⚖️ 平衡",
            "detailed": "📖 详细",
        }
        
        lines = [
            "👤 用户画像",
            "━━━━━━━━━━━━━━━━",
            f"总交互: {p.total_interactions} 次",
            f"成功率: {success_rate:.0f}%",
            f"沟通风格: {style_emoji.get(p.preferred_response_style, '未知')}",
            f"最爱工具: {tools_str}",
            f"技术栈: {tech_str}",
            f"活跃时间: {time_str}",
            f"偏好: {'先代码后解释' if p.prefers_code_first else '先解释后代码'}",
        ]
        
        if p.user_notes:
            lines.append(f"\n📝 用户笔记 ({len(p.user_notes)} 条):")
            for note in p.user_notes[-3:]:
                lines.append(f"  • {note}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """获取统计数据"""
        p = self.preferences
        return {
            "total_interactions": p.total_interactions,
            "success_rate": round(p.successful_interactions / max(p.total_interactions, 1), 3),
            "preferred_style": p.preferred_response_style,
            "top_tool": max(p.tool_frequency.items(), key=lambda x: x[1])[0] if p.tool_frequency else None,
            "top_tech": max(p.tech_stack.items(), key=lambda x: x[1])[0] if p.tech_stack else None,
            "peak_hour": max(p.active_hours.items(), key=lambda x: x[1])[0] if p.active_hours else None,
        }


# ─── 全局实例 ─────────────────────────────────────

_global_profile: Optional[UserProfile] = None


def get_user_profile() -> UserProfile:
    """获取全局用户画像实例"""
    global _global_profile
    if _global_profile is None:
        _global_profile = UserProfile()
    return _global_profile