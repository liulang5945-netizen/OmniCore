"""
态极混合推理引擎
让态极借用外部成熟模型进行对话，同时学习成长。

核心机制：
1. 态极先自己"思考"，产生初步理解
2. 将消息发给外部模型，获得高质量回答
3. 态极将自己的思考与外部回答对比学习
4. 将 (用户消息, 外部回答) 存入训练数据池
5. 后台：积累足够数据后自动微调态极自身
"""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Generator, Dict, Any

logger = logging.getLogger("Taiji.HybridEngine")


# ══════════════════════════════════════════════════════════════════
# PII 脱敏器 — 训练数据写入前必须经过脱敏
# ══════════════════════════════════════════════════════════════════

# 常见 PII 模式（顺序敏感：先匹配长模式，避免子串误匹配）
_PII_PATTERNS = [
    # API Keys / Tokens（常见格式：sk-xxx, ghp_xxx, key-xxx, Bearer xxx）
    (re.compile(r'\b(sk-[A-Za-z0-9]{20,})', re.IGNORECASE), '[API_KEY]'),
    (re.compile(r'\b(ghp_[A-Za-z0-9]{36,})', re.IGNORECASE), '[GITHUB_TOKEN]'),
    (re.compile(r'\b(key-[A-Za-z0-9]{20,})', re.IGNORECASE), '[API_KEY]'),
    (re.compile(r'(Bearer\s+[A-Za-z0-9._\-]{20,})', re.IGNORECASE), '[BEARER_TOKEN]'),
    (re.compile(r'(Authorization:\s*[A-Za-z0-9._\-+/=]{20,})', re.IGNORECASE), '[AUTH_HEADER]'),
    # AWS 密钥
    (re.compile(r'\b(AKIA[0-9A-Z]{16})\b'), '[AWS_ACCESS_KEY]'),
    (re.compile(r'\b([A-Za-z0-9/+=]{40})\b(?=.*(?:aws|secret))', re.IGNORECASE), '[AWS_SECRET]'),
    # 私钥块
    (re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----'), '[PRIVATE_KEY]'),
    # 邮箱地址
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    # 手机号（中国大陆 11 位）
    (re.compile(r'\b1[3-9]\d{9}\b'), '[PHONE]'),
    # IP 地址（IPv4）
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP_ADDR]'),
    # 密码赋值（password=xxx, passwd=xxx, pwd=xxx）
    (re.compile(r'((?:password|passwd|pwd|secret|token)\s*[=:]\s*)\S+', re.IGNORECASE), r'\1[PASSWORD]'),
    # 信用卡号（16 位数字，可带空格/横线分隔）
    (re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'), '[CARD_NUM]'),
    # 身份证号（中国大陆 18 位）
    (re.compile(r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'), '[ID_CARD]'),
]


def sanitize_pii(text: str) -> str:
    """
    对文本进行 PII（个人身份信息）脱敏。

    在写入训练数据前调用，防止用户的 API Key、密码、邮箱、手机号等
    敏感信息泄露到微调数据集中。

    Args:
        text: 原始文本

    Returns:
        脱敏后的文本（PII 替换为 [REDACTED] 标记）
    """
    if not text:
        return text
    result = text
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class HybridEngine:
    """
    混合推理引擎 — 态极的"学习式对话"核心
    
    态极借用外部模型进行对话，同时：
    - 自己尝试理解问题
    - 记录外部模型的高质量回答
    - 积累训练数据，为自身成长做准备
    """
    
    def __init__(
        self,
        taiji_engine,       # 态极多模态引擎
        data_collector,     # 数据收集器
        save_path: str = None,  # 训练数据保存路径
    ):
        self.taiji_engine = taiji_engine
        self.data_collector = data_collector
        self.save_path = save_path
        
        # 能力评估表：各类问题的自评分数 (0~1)
        self.capability_scores = {
            "general": 0.1,    # 通用对话
            "code": 0.1,       # 代码编写
            "knowledge": 0.1,  # 知识问答
            "creative": 0.1,   # 创意写作
            "math": 0.1,       # 数学推理
        }
        # 毕业阈值：达到此分数后态极可以独立回答该类问题
        self.graduate_threshold = 0.8
        
        # 对话历史缓存（用于训练数据收集）
        self._conversation_buffer = []
        self._save_interval = 50  # 每 50 条对话保存一次
        
        # 统计
        self.stats = {
            "total_conversations": 0,
            "taiji_handled": 0,      # 态极独立处理的对话数
            "external_handled": 0,   # 外部模型处理的对话数
            "training_samples": 0,   # 累计收集的训练样本数
        }
        
        logger.info("混合推理引擎已初始化")
    
    def classify_question(self, prompt: str) -> str:
        """
        简单分类用户问题类型
        用于判断态极是否有能力独立回答
        """
        prompt_lower = prompt.lower()
        
        # 代码相关
        code_keywords = ["代码", "编程", "python", "javascript", "函数", "bug",
                        "code", "program", "function", "script", "实现", "写一个"]
        if any(kw in prompt_lower for kw in code_keywords):
            return "code"
        
        # 数学相关
        math_keywords = ["计算", "数学", "方程", "证明", "求解", "公式",
                        "calculate", "math", "equation", "prove", "公式"]
        if any(kw in prompt_lower for kw in math_keywords):
            return "math"
        
        # 知识问答
        knowledge_keywords = ["什么是", "为什么", "如何", "解释", "原理",
                            "what is", "why", "how", "explain", "概念", "定义"]
        if any(kw in prompt_lower for kw in knowledge_keywords):
            return "knowledge"
        
        # 创意写作
        creative_keywords = ["写", "创作", "故事", "诗", "文章", "设计",
                           "write", "create", "story", "poem", "article"]
        if any(kw in prompt_lower for kw in creative_keywords):
            return "creative"
        
        return "general"
    
    def should_handle_by_taiji(self, prompt: str) -> bool:
        """
        判断态极是否有能力独立处理这个问题
        
        基于能力评估分数，只有当该类问题的分数达到毕业阈值时
        才让态极独立处理
        """
        category = self.classify_question(prompt)
        score = self.capability_scores.get(category, 0.0)
        return score >= self.graduate_threshold
    
    def generate_hybrid_stream(
        self,
        prompt: str,
        external_generate_fn,  # 外部模型的生成流 (prompt) -> Generator
        system_prompt: str = "",
        history: list = None,
        max_new_tokens: int = 512,
        stop_event=None,
    ) -> Generator[str, None, None]:
        """
        混合流式生成

        关键优化：态极思考与外部模型请求 **并行** 执行。
        旧实现是同步阻塞 —— 必须等态极慢速推理完毕才去调外部模型，
        导致首字响应延迟（TTFT）被态极拖累。
        现在两者同时启动，外部模型结果先到就先输出。
        """
        self.stats["total_conversations"] += 1
        category = self.classify_question(prompt)

        # ── 步骤1：态极思考与外部模型并行 ──
        taiji_thought = ""
        full_response = ""
        self.stats["external_handled"] += 1

        # 构建完整 prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # 用线程池并发执行态极思考（不阻塞主线程）
        _taiji_future = None
        _executor = None
        if self.taiji_engine and hasattr(self.taiji_engine, 'text_engine'):
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TaijiThink")
            def _do_taiji_think():
                try:
                    react_result = self.taiji_engine.text_engine.inference.generate_react_step(
                        f"[用户] {prompt}\n[助手]",
                        max_new_tokens=128,
                        temperature=0.3,
                    )
                    thought = react_result.get("thought", "")
                    if "final_answer" in react_result:
                        thought = react_result["final_answer"]
                    return thought
                except Exception as e:
                    logger.debug(f"态极思考失败（正常，继续使用外部模型）: {e}")
                    return ""
            _taiji_future = _executor.submit(_do_taiji_think)

        try:
            # 步骤2：调用外部模型生成回答（与态极思考并行）
            response = external_generate_fn(full_prompt)

            if response:
                full_response = response
                for char in response:
                    if stop_event and stop_event.is_set():
                        break
                    yield char
            else:
                # 外部模型无结果，等待态极思考完成作为降级
                if _taiji_future is not None:
                    try:
                        taiji_thought = _taiji_future.result(timeout=30)
                    except Exception:
                        pass
                if taiji_thought:
                    full_response = taiji_thought
                    for char in taiji_thought:
                        if stop_event and stop_event.is_set():
                            break
                        yield char
                else:
                    yield "抱歉，我暂时无法回答这个问题。请稍后重试。"

        except Exception as e:
            logger.error(f"外部模型调用失败: {e}")
            # 降级使用态极
            if _taiji_future is not None:
                try:
                    taiji_thought = _taiji_future.result(timeout=30)
                except Exception:
                    pass
            if taiji_thought:
                full_response = taiji_thought
                for char in taiji_thought:
                    yield char
            else:
                yield f"抱歉，外部模型调用失败: {e}"
        finally:
            # 收集态极思考结果（如果还没拿到）
            if _taiji_future is not None and not taiji_thought:
                try:
                    taiji_thought = _taiji_future.result(timeout=0)
                except Exception:
                    pass
            if _executor is not None:
                _executor.shutdown(wait=False)

        # 步骤3：收集训练数据（PII 脱敏后写入）
        if full_response:
            self._collect_training_data(
                prompt=prompt,
                response=full_response,
                taiji_thought=taiji_thought,
                category=category,
            )
    
    def generate_hybrid(
        self,
        prompt: str,
        external_generate_fn,
        system_prompt: str = "",
        history: list = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        混合非流式生成（态极思考与外部模型并行）
        """
        self.stats["total_conversations"] += 1
        category = self.classify_question(prompt)

        # 态极思考与外部模型并行
        taiji_thought = ""
        full_response = ""
        self.stats["external_handled"] += 1

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        _taiji_future = None
        _executor = None
        if self.taiji_engine and hasattr(self.taiji_engine, 'text_engine'):
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TaijiThink")
            def _do_taiji_think():
                try:
                    react_result = self.taiji_engine.text_engine.inference.generate_react_step(
                        f"[用户] {prompt}\n[助手]",
                        max_new_tokens=128,
                        temperature=0.3,
                    )
                    thought = react_result.get("thought", "")
                    if "final_answer" in react_result:
                        thought = react_result["final_answer"]
                    return thought
                except Exception:
                    return ""
            _taiji_future = _executor.submit(_do_taiji_think)

        try:
            response = external_generate_fn(full_prompt)
            full_response = response if response else ""
        except Exception as e:
            logger.error(f"外部模型调用失败: {e}")

        # 收集态极思考结果
        if _taiji_future is not None:
            try:
                taiji_thought = _taiji_future.result(timeout=30)
            except Exception:
                pass
        if _executor is not None:
            _executor.shutdown(wait=False)

        # 降级：外部模型无结果时使用态极
        if not full_response:
            full_response = taiji_thought or "抱歉，无法回答。"

        # 收集训练数据（PII 脱敏后写入）
        if full_response:
            self._collect_training_data(
                prompt=prompt,
                response=full_response,
                taiji_thought=taiji_thought,
                category=category,
            )

        return full_response
    
    def _collect_training_data(
        self,
        prompt: str,
        response: str,
        taiji_thought: str,
        category: str,
    ):
        """
        收集训练数据，供态极后台学习。

        ⚠️ 安全要求：所有文本在写入前必须经过 PII 脱敏，
        防止用户的 API Key、密码、邮箱、手机号等泄露到微调数据集。
        """
        # PII 脱敏：清除用户提问和模型回答中的敏感信息
        safe_prompt = sanitize_pii(prompt)
        safe_response = sanitize_pii(response)
        safe_thought = sanitize_pii(taiji_thought) if taiji_thought else ""

        # 构建训练样本（仅使用脱敏后的内容）
        sample = {
            "messages": [
                {"role": "system", "content": "你是态极 AI 助手，一个智能的本地 AI 助手。"},
                {"role": "user", "content": safe_prompt},
                {"role": "assistant", "content": safe_response},
            ],
            "metadata": {
                "category": category,
                "taiji_thought": safe_thought,
                "source": "hybrid_engine",
                "timestamp": time.time(),
                "pii_sanitized": True,  # 标记已脱敏
            },
        }
        
        self._conversation_buffer.append(sample)
        self.stats["training_samples"] += 1
        
        # 定期保存
        if len(self._conversation_buffer) >= self._save_interval:
            self._flush_training_data()
        
        # 同时交给 data_collector（使用脱敏后的数据）
        if self.data_collector:
            try:
                self.data_collector.collect_conversation(safe_prompt, safe_response)
            except Exception:
                pass
    
    def _flush_training_data(self):
        """将缓冲区的训练数据保存到文件"""
        if not self._conversation_buffer:
            return
        
        if not self.save_path:
            logger.warning("未设置训练数据保存路径，跳过保存")
            return
        
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "a", encoding="utf-8") as f:
                for sample in self._conversation_buffer:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            logger.info(f"已保存 {len(self._conversation_buffer)} 条训练数据到 {self.save_path}")
            self._conversation_buffer = []
        except Exception as e:
            logger.error(f"保存训练数据失败: {e}")
    
    def update_capability(self, category: str, delta: float):
        """
        更新某类问题的能力分数
        由 self_evaluator 在评估后调用
        """
        if category in self.capability_scores:
            old_score = self.capability_scores[category]
            # 使用指数移动平均，避免分数剧烈波动
            self.capability_scores[category] = min(1.0, old_score * 0.8 + delta * 0.2)
            new_score = self.capability_scores[category]
            
            # 检查是否"毕业"
            if old_score < self.graduate_threshold and new_score >= self.graduate_threshold:
                logger.info(f"🎓 态极在 [{category}] 领域毕业！分数: {new_score:.2f}")
    
    def get_status(self) -> dict:
        """获取混合引擎状态"""
        # 计算整体能力
        avg_capability = sum(self.capability_scores.values()) / len(self.capability_scores)
        
        return {
            "type": "hybrid_engine",
            "stats": self.stats.copy(),
            "capability_scores": self.capability_scores.copy(),
            "average_capability": round(avg_capability, 3),
            "graduate_threshold": self.graduate_threshold,
            "buffered_samples": len(self._conversation_buffer),
            "graduated_categories": [
                cat for cat, score in self.capability_scores.items()
                if score >= self.graduate_threshold
            ],
        }
    
    def flush(self):
        """手动触发保存"""
        self._flush_training_data()