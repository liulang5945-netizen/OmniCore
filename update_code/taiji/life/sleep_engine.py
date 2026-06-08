"""
态极睡眠引擎 (Sleep Engine)
============================

态极最独特的能力：睡觉。

就像人脑在睡眠中巩固记忆、修剪突触、整合经验，
态极在用户不活跃时自动进入"睡眠"状态，
整理收集的数据、微调模型、更新用户画像。

睡眠周期：
Phase 1 (浅睡眠): 记忆整理 — 清理 WorkingMemory
Phase 2 (深睡眠): 模型训练 — 用收集的数据在线微调
Phase 3 (REM): 知识整合 — 进化引擎 + 用户画像更新
Phase 4 (清醒): 自我评估 — 检查模型健康状态
"""
import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("SleepEngine")


@dataclass
class SleepReport:
    """一次睡眠的报告"""
    timestamp: str
    duration_seconds: float
    phases_completed: List[str] = field(default_factory=list)
    memory_entries_cleared: int = 0
    training_samples_used: int = 0
    training_loss: Optional[float] = None
    evolution_events: int = 0
    user_patterns_updated: int = 0
    health_status: str = "unknown"
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SleepConfig:
    """睡眠配置"""
    auto_sleep_enabled: bool = True
    sleep_interval_hours: float = 4.0       # 每 4 小时自动睡眠一次
    min_idle_minutes: int = 30               # 空闲 30 分钟后才触发
    max_cpu_percent: float = 80.0            # CPU < 80% 才睡眠
    max_memory_percent: float = 90.0         # 内存 < 90% 才睡眠
    training_enabled: bool = True            # 睡眠时是否训练
    max_training_steps: int = 50             # 睡眠时最大训练步数
    save_checkpoints: bool = True            # 睡眠时保存 checkpoint


class SleepEngine:
    """
    态极的睡眠引擎
    
    核心理念：
    - 睡眠不是浪费时间，而是成长的关键
    - 就像人脑在睡眠中巩固记忆、整合经验
    - 态极在用户休息时自动整理、学习、进化
    
    睡眠触发条件：
    1. 定时触发（每 N 小时）
    2. 空闲触发（用户超过 M 分钟没有交互）
    3. 手动触发（用户/系统主动调用）
    """
    
    def __init__(self, config: Optional[SleepConfig] = None, data_dir: str = "taiji/sleep_data",
                 model_provider=None, tokenizer_provider=None):
        """
        Args:
            config: 睡眠配置
            data_dir: 数据目录
            model_provider: 模型获取回调（解耦 core.app_state）
            tokenizer_provider: 分词器获取回调
        """
        self.config = config or SleepConfig()
        self.data_dir = data_dir
        self._model_provider = model_provider
        self._tokenizer_provider = tokenizer_provider
        self._last_sleep_time: Optional[datetime] = None
        self._last_activity_time: Optional[datetime] = None
        self._sleep_history: List[SleepReport] = []
        self._is_sleeping = False
        self._auto_sleep_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        os.makedirs(data_dir, exist_ok=True)
        self._load_history()
        
        logger.info(f"SleepEngine initialized: auto={self.config.auto_sleep_enabled}, interval={self.config.sleep_interval_hours}h")
    
    # ─── 公开接口 ───────────────────────────────────
    
    def sleep(self, reason: str = "manual") -> SleepReport:
        """
        让态极进入睡眠。
        
        Args:
            reason: 睡眠原因（"manual", "auto", "scheduled"）
            
        Returns:
            SleepReport 睡眠报告
        """
        if self._is_sleeping:
            logger.warning("Already sleeping, skipping")
            return SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0)
        
        self._is_sleeping = True
        start_time = time.time()
        
        logger.info(f"💤 Taiji is going to sleep... (reason: {reason})")
        
        report = SleepReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
        )
        
        # Phase 1: 浅睡眠 — 记忆整理
        try:
            self._sleep_phase_memory_consolidation(report)
            report.phases_completed.append("memory_consolidation")
            logger.info("  Phase 1: Memory consolidation ✅")
        except Exception as e:
            logger.warning(f"  Phase 1 failed: {e}")
        
        # Phase 2: 深睡眠 — 模型训练
        if self.config.training_enabled:
            try:
                self._sleep_phase_model_training(report)
                report.phases_completed.append("model_training")
                logger.info("  Phase 2: Model training ✅")
            except Exception as e:
                logger.warning(f"  Phase 2 failed: {e}")
        
        # Phase 3: REM — 知识整合
        try:
            self._sleep_phase_knowledge_integration(report)
            report.phases_completed.append("knowledge_integration")
            logger.info("  Phase 3: Knowledge integration ✅")
        except Exception as e:
            logger.warning(f"  Phase 3 failed: {e}")
        
        # Phase 4: 清醒准备 — 自我评估
        try:
            health = self._sleep_phase_evaluation(report)
            report.health_status = health.get("status", "unknown")
            report.phases_completed.append("evaluation")
            logger.info("  Phase 4: Evaluation ✅")
        except Exception as e:
            logger.warning(f"  Phase 4 failed: {e}")
        
        # 计算睡眠时长
        report.duration_seconds = round(time.time() - start_time, 1)
        self._last_sleep_time = datetime.now()
        self._is_sleeping = False
        
        # 保存报告
        self._sleep_history.append(report)
        self._save_history()
        
        logger.info(f"⏰ Taiji woke up! Duration: {report.duration_seconds}s, Phases: {len(report.phases_completed)}")
        
        return report
    
    def wake(self):
        """唤醒态极"""
        self._is_sleeping = False
        logger.info("☀️ Taiji is awake!")
    
    def record_activity(self):
        """记录用户活动（用于判断是否空闲）"""
        self._last_activity_time = datetime.now()
    
    def start_auto_sleep(self):
        """启动自动睡眠线程"""
        if not self.config.auto_sleep_enabled:
            return
        
        if self._auto_sleep_thread and self._auto_sleep_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._auto_sleep_thread = threading.Thread(target=self._auto_sleep_loop, daemon=True)
        self._auto_sleep_thread.start()
        logger.info("Auto-sleep thread started")
    
    def stop_auto_sleep(self):
        """停止自动睡眠"""
        self._stop_event.set()
        if self._auto_sleep_thread:
            self._auto_sleep_thread.join(timeout=5)
        logger.info("Auto-sleep thread stopped")
    
    def _auto_sleep_loop(self):
        """自动睡眠循环"""
        while not self._stop_event.is_set():
            time.sleep(60)  # 每分钟检查一次
            
            if self._should_auto_sleep():
                self.sleep(reason="auto")
    
    def _should_auto_sleep(self) -> bool:
        """检查是否应该自动睡眠"""
        if self._is_sleeping:
            return False
        
        # 检查距上次睡眠的时间
        if self._last_sleep_time:
            hours_since_last = (datetime.now() - self._last_sleep_time).total_seconds() / 3600
            if hours_since_last < self.config.sleep_interval_hours:
                return False
        
        # 检查空闲时间
        if self._last_activity_time:
            idle_minutes = (datetime.now() - self._last_activity_time).total_seconds() / 60
            if idle_minutes < self.config.min_idle_minutes:
                return False
        
        return True
    
    # ─── 睡眠阶段实现 ──────────────────────────────
    
    def _sleep_phase_memory_consolidation(self, report: SleepReport):
        """Phase 1: 记忆整理 — 清理 WorkingMemory"""
        try:
            from taiji.working_memory import get_working_memory
            wm = get_working_memory()
            
            modified = wm.get_modified_keys()
            report.memory_entries_cleared = len(modified)
            
            if modified:
                logger.info(f"  Consolidating {len(modified)} modified memory entries")
            
            # 导出修改过的内容（可以保存到长期存储）
            for key in modified:
                content = wm.export(key)
                if content:
                    # 保存到文件
                    safe_name = key.replace("/", "_").replace("\\", "_")
                    save_path = os.path.join(self.data_dir, f"memory_{safe_name}.txt")
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content)
            
            # 清理工作记忆
            wm.clear()
            logger.info("  Working memory cleared")
            
        except ImportError:
            logger.info("  WorkingMemory not available, skipping")
    
    def _sleep_phase_model_training(self, report: SleepReport):
        """Phase 2: 模型训练 — 用收集的数据在线微调态极模型"""
        try:
            from agent.data_collector import get_collector
            from taiji.data_generator import generate_bulk_react_data, generate_bulk_conversation_data
            
            collector = get_collector()
            react_data, conv_data = collector.load_as_training_data()
            
            report.training_samples_used = len(react_data) + len(conv_data)
            
            if not react_data and not conv_data:
                logger.info("  No new training data, using generated data")
                react_data = generate_bulk_react_data(50)
                conv_data = generate_bulk_conversation_data(20)
                report.training_samples_used = len(react_data) + len(conv_data)
            
            logger.info(f"  Training with {report.training_samples_used} samples")
            
            # 保存训练数据供后续使用
            train_data_path = os.path.join(self.data_dir, "sleep_training_data.jsonl")
            with open(train_data_path, "w", encoding="utf-8") as f:
                for item in react_data:
                    f.write(json.dumps({"type": "react", **item}, ensure_ascii=False) + "\n")
                for item in conv_data:
                    f.write(json.dumps({"type": "conversation", **item}, ensure_ascii=False) + "\n")
            
            # 从喂养引擎获取待消化的训练样本
            try:
                from taiji.feed_engine import get_feed_engine
                feed_engine = get_feed_engine()
                pending_samples = feed_engine.get_pending_samples()
                if pending_samples:
                    logger.info(f"  Got {len(pending_samples)} pending samples from feed engine")
                    # 将喂养引擎的样本也加入训练
                    for sample in pending_samples:
                        if sample.get("type") == "react":
                            react_data.append(sample)
                        elif sample.get("type") == "conversation":
                            conv_data.append(sample)
                    report.training_samples_used += len(pending_samples)
            except Exception as e:
                logger.debug(f"  Feed engine integration skipped: {e}")

            # 实际调用态极训练器进行在线微调
            training_loss = self._run_sleep_training(react_data, conv_data)
            if training_loss is not None:
                report.training_loss = training_loss
                logger.info(f"  Sleep training completed, loss={training_loss:.4f}")
            
            # 标记数据已使用
            collector.flush()

            # 清除喂养引擎已消化的样本
            try:
                from taiji.feed_engine import get_feed_engine
                feed_engine = get_feed_engine()
                feed_engine.clear_pending_samples()
                logger.info("  Feed engine pending samples cleared")
            except Exception:
                pass
            
        except ImportError:
            logger.info("  DataCollector not available, skipping")
    
    def _run_sleep_training(self, react_data: list, conv_data: list):
        """
        执行睡眠训练：在线微调态极 ModelSelf 模型

        直接使用 PyTorch 进行轻量级微调，限制步数以控制睡眠时长。
        通过注入的 model_provider/tokenizer_provider 获取模型，不依赖 core.app_state。
        """
        import torch
        import torch.nn.functional as F
        try:
            # 优先使用注入的 provider（解耦 core.app_state）
            model = None
            tokenizer = None
            if self._model_provider:
                model = self._model_provider()
            if self._tokenizer_provider:
                tokenizer = self._tokenizer_provider()

            # 回退：尝试从 core.app_state 获取（向后兼容）
            if model is None or tokenizer is None:
                try:
                    from core.app_state import app_state
                    if model is None:
                        model = app_state.model
                    if tokenizer is None:
                        tokenizer = app_state.tokenizer
                except ImportError:
                    pass

            if model is None or tokenizer is None:
                logger.info("  No model available, skipping training")
                return None
            
            # 判断是否为态极模型
            from taiji.architecture import ModelSelf
            if not isinstance(model, ModelSelf):
                logger.info("  Current model is not ModelSelf, skipping sleep training")
                return None
            
            # 合并训练数据
            all_texts = []
            for item in react_data:
                task = item.get("task", "")
                if task:
                    all_texts.append(task)
            for item in conv_data:
                if isinstance(item, dict) and "messages" in item:
                    text = " ".join(m.get("content", "") for m in item["messages"] if m.get("role") != "system")
                    if text.strip():
                        all_texts.append(text)
            
            if not all_texts:
                logger.info("  No valid training texts, skipping")
                return None
            
            # 限制训练步数（睡眠时轻量训练）
            max_steps = min(self.config.max_training_steps, len(all_texts))
            device = next(model.parameters()).device
            
            # 创建优化器（睡眠时用较小学习率，避免灾难性遗忘）
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            model.train()
            
            logger.info(f"  Starting sleep training: {max_steps} steps on {device}")
            
            final_loss = None
            step_count = 0
            
            for i in range(max_steps):
                text = all_texts[i % len(all_texts)]
                
                # 编码
                ids = tokenizer.encode(text)
                if len(ids) < 5:
                    continue
                ids = ids[:512]  # 截断
                
                # 构建 input_ids 和 labels（自回归）
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                labels = input_ids.clone()
                
                # 前向
                output = model(input_ids, targets=labels)
                loss = output.loss
                
                if loss is None or loss.item() == 0:
                    continue
                
                # 反向
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                final_loss = loss.item()
                step_count += 1
                
                if step_count % 10 == 0:
                    logger.info(f"  Sleep training step {step_count}/{max_steps}, loss={final_loss:.4f}")
            
            model.eval()
            
            # 训练后保存 checkpoint（保存到 model_name 的父目录，与微调路径一致）
            if self.config.save_checkpoints and final_loss is not None:
                try:
                    from taiji.loader import save_model
                    # 优先保存到当前模型所在目录，确保重启后自动加载
                    checkpoint_dir = None
                    try:
                        from core.app_state import app_state
                        model_path = getattr(app_state, "_loaded_model_name", "") or ""
                        if model_path and os.path.isdir(model_path):
                            checkpoint_dir = model_path  # 直接覆盖当前 best
                    except Exception:
                        pass
                    if not checkpoint_dir:
                        checkpoint_dir = os.path.join(self.data_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_model(model, tokenizer, checkpoint_dir)
                    logger.info(f"  Checkpoint saved to {checkpoint_dir}")
                except Exception as e:
                    logger.warning(f"  Checkpoint save failed: {e}")
            
            return final_loss
            
        except Exception as e:
            logger.warning(f"  Sleep training failed: {e}")
            return None
    
    def _sleep_phase_knowledge_integration(self, report: SleepReport):
        """Phase 3: 知识整合 — 进化引擎 + 用户画像"""
        # 进化引擎
        try:
            from taiji.evolution_engine import get_evolution_engine
            engine = get_evolution_engine()
            
            if engine.metrics.tasks_completed > 0:
                report.evolution_events = engine.metrics.evolution_cycles
                
                # 检查是否需要触发进化
                total = engine.metrics.tasks_completed + engine.metrics.tasks_failed
                if total > 0 and total % 50 == 0:
                    engine._trigger_evolution("sleep_cycle")
                    report.evolution_events += 1
                    
        except ImportError:
            logger.info("  EvolutionEngine not available, skipping")
        
        # 用户画像
        try:
            from taiji.user_profile import get_user_profile
            profile = get_user_profile()
            
            suggestions = profile.get_task_pattern_suggestions()
            report.user_patterns_updated = len(suggestions)
            
            if suggestions:
                report.recommendations.extend(suggestions)
                
        except ImportError:
            logger.info("  UserProfile not available, skipping")
    
    def _sleep_phase_evaluation(self, report: SleepReport) -> dict:
        """Phase 4: 自我评估"""
        try:
            from taiji.self_evaluator import get_self_evaluator
            from taiji.evolution_engine import get_evolution_engine
            
            evaluator = get_self_evaluator()
            engine = get_evolution_engine()
            
            stats = evaluator.get_stats()
            trends = evaluator.get_improvement_trends()
            
            health = {
                "phase": engine.metrics.current_phase,
                "tasks_completed": engine.metrics.tasks_completed,
                "evaluation_count": stats.get("total_evaluations", 0),
                "avg_score": stats.get("avg_score", 0),
                "trends": trends,
                "status": "healthy",
            }
            
            # 保存健康报告
            health_path = os.path.join(self.data_dir, "health_report.json")
            with open(health_path, "w", encoding="utf-8") as f:
                json.dump(health, f, indent=2, ensure_ascii=False)
            
            return health
            
        except ImportError:
            return {"status": "unknown"}
    
    # ─── 持久化 ─────────────────────────────────────
    
    def _save_history(self):
        """保存睡眠历史"""
        path = os.path.join(self.data_dir, "sleep_history.json")
        try:
            data = []
            for report in self._sleep_history[-50:]:  # 只保留最近 50 次
                data.append({
                    "timestamp": report.timestamp,
                    "duration_seconds": report.duration_seconds,
                    "phases_completed": report.phases_completed,
                    "memory_entries_cleared": report.memory_entries_cleared,
                    "training_samples_used": report.training_samples_used,
                    "evolution_events": report.evolution_events,
                    "health_status": report.health_status,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save sleep history: {e}")
    
    def _load_history(self):
        """加载睡眠历史"""
        path = os.path.join(self.data_dir, "sleep_history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self._sleep_history.append(SleepReport(**item))
        except Exception as e:
            logger.warning(f"Failed to load sleep history: {e}")
    
    # ─── 状态查询 ───────────────────────────────────
    
    def get_status(self) -> dict:
        """获取睡眠引擎状态"""
        return {
            "is_sleeping": self._is_sleeping,
            "last_sleep": self._last_sleep_time.isoformat() if self._last_sleep_time else None,
            "last_activity": self._last_activity_time.isoformat() if self._last_activity_time else None,
            "total_sleeps": len(self._sleep_history),
            "auto_sleep_enabled": self.config.auto_sleep_enabled,
        }
    
    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        
        sleeping = "💤 睡眠中" if status["is_sleeping"] else "☀️ 清醒"
        last_sleep = status["last_sleep"] or "从未睡眠"
        
        lines = [
            "💤 睡眠引擎状态",
            "━━━━━━━━━━━━━━━━",
            f"当前状态: {sleeping}",
            f"上次睡眠: {last_sleep}",
            f"总睡眠次数: {status['total_sleeps']}",
            f"自动睡眠: {'✅ 开启' if status['auto_sleep_enabled'] else '❌ 关闭'}",
        ]
        
        if self._sleep_history:
            last = self._sleep_history[-1]
            lines.append(f"\n最近一次睡眠报告:")
            lines.append(f"  时长: {last.duration_seconds}s")
            lines.append(f"  阶段: {', '.join(last.phases_completed)}")
            lines.append(f"  健康状态: {last.health_status}")
        
        return "\n".join(lines)
    
    def get_sleep_trends(self) -> List[str]:
        """分析睡眠趋势"""
        if len(self._sleep_history) < 3:
            return ["数据不足，至少需要 3 次睡眠记录"]
        
        recent = self._sleep_history[-5:]
        avg_duration = sum(r.duration_seconds for r in recent) / len(recent)
        avg_phases = sum(len(r.phases_completed) for r in recent) / len(recent)
        
        trends = [
            f"最近 {len(recent)} 次睡眠平均时长: {avg_duration:.1f}s",
            f"平均完成阶段数: {avg_phases:.1f}/4",
        ]
        
        # 检查训练效果
        recent_training = [r.training_samples_used for r in recent if r.training_samples_used > 0]
        if recent_training:
            avg_samples = sum(recent_training) / len(recent_training)
            trends.append(f"平均训练样本数: {avg_samples:.0f}")
        
        return trends


# ─── 全局实例 ─────────────────────────────────────

_global_sleep: Optional[SleepEngine] = None


def get_sleep_engine(config: Optional[SleepConfig] = None) -> SleepEngine:
    """获取全局睡眠引擎实例"""
    global _global_sleep
    if _global_sleep is None:
        _global_sleep = SleepEngine(config)
    return _global_sleep