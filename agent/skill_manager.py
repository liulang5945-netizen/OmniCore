"""
技能管理器 (Skill Manager)
==========================
管理 Agent 技能（Skill）：一组工具 + 系统提示的组合。
支持从任务执行中学习和积累技能。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("SkillManager")


@dataclass
class Skill:
    """技能定义"""
    id: str
    name: str
    description: str = ""
    system_prompt: str = ""        # 技能专用的系统提示
    tools: List[str] = field(default_factory=list)  # 允许使用的工具名列表
    examples: List[dict] = field(default_factory=list)  # 学习到的示例
    category: str = "通用"
    created_from: str = ""         # 从哪个任务学习的
    use_count: int = 0


class SkillManager:
    """技能管理器"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._active_skill: Optional[Skill] = None
        self._config_path: str = ""
        self._init_paths()
        self._load_skills()

    def _init_paths(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._config_path = os.path.join(base_dir, "agent_skills.json")

    def _load_skills(self):
        """从配置文件加载技能"""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = Skill(**skill_data)
                    self._skills[skill.id] = skill
                logger.info(f"已加载 {len(self._skills)} 个技能")
        except Exception as e:
            logger.warning(f"加载技能配置失败: {e}")

    def _save_skills(self):
        """保存技能到配置文件"""
        try:
            data = {
                "skills": [
                    {
                        "id": s.id, "name": s.name, "description": s.description,
                        "system_prompt": s.system_prompt, "tools": s.tools,
                        "examples": s.examples[-5:],  # 只保留最近5个示例
                        "category": s.category, "created_from": s.created_from,
                        "use_count": s.use_count,
                    }
                    for s in self._skills.values()
                ]
            }
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存技能配置失败: {e}")

    def activate_skill(self, skill_id: str) -> Optional[Skill]:
        """激活一个技能"""
        skill = self._skills.get(skill_id)
        if skill:
            skill.use_count += 1
            self._active_skill = skill
            self._save_skills()
            logger.info(f"激活技能: {skill.name}")
        return skill

    def deactivate_skill(self):
        """停用当前技能"""
        self._active_skill = None

    def get_skill_system_prompt(self) -> str:
        """获取当前激活技能的系统提示"""
        if self._active_skill and self._active_skill.system_prompt:
            return self._active_skill.system_prompt
        return ""

    def learn_from_task(self, task: str, steps: list, final_answer: str):
        """从成功的任务执行中学习技能"""
        # 简单实现：将任务模式记录为示例
        for skill in self._skills.values():
            if self._is_relevant(skill, task):
                skill.examples.append({
                    "task": task[:200],
                    "steps_summary": len(steps),
                    "answer_preview": final_answer[:100],
                })
                self._save_skills()
                return

    def _is_relevant(self, skill: Skill, task: str) -> bool:
        """判断任务是否与技能相关"""
        keywords = skill.name.lower().split()
        task_lower = task.lower()
        return any(kw in task_lower for kw in keywords if len(kw) > 2)

    def list_skills(self) -> list:
        """列出所有技能"""
        return [
            {
                "id": s.id, "name": s.name, "description": s.description,
                "tools": s.tools, "category": s.category, "use_count": s.use_count,
                "examples_count": len(s.examples),
                "active": self._active_skill and self._active_skill.id == s.id,
            }
            for s in self._skills.values()
        ]

    def create_skill(self, skill_id: str, name: str, description: str = "",
                     system_prompt: str = "", tools: list = None, category: str = "自定义") -> Skill:
        """创建新技能"""
        skill = Skill(
            id=skill_id, name=name, description=description,
            system_prompt=system_prompt, tools=tools or [], category=category,
        )
        self._skills[skill_id] = skill
        self._save_skills()
        return skill


# 全局单例
skill_manager = SkillManager()