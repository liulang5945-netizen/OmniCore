"""
Token 优化器

智能管理 token 使用，节约云端 API 额度：
- 动态计算 max_tokens
- 对话历史压缩
- token 计数估算
- 工具结果智能截断
"""
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("TokenOptimizer")


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。
    
    中文: 平均 1 个汉字 ≈ 1.5-2 tokens
    英文: 平均 1 个单词 ≈ 1.3 tokens
    代码: 平均 1 个字符 ≈ 0.4 tokens
    
    使用混合策略估算，误差约 ±15%，足够用于优化决策。
    """
    if not text:
        return 0
    
    # 统计字符类型
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other_chars = len(text) - chinese_chars - ascii_chars
    
    # 估算
    tokens = int(chinese_chars * 1.8 + ascii_chars * 0.35 + other_chars * 1.0)
    return max(tokens, 1)


def estimate_messages_tokens(messages: list) -> int:
    """估算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content) + 4  # 每条消息有约 4 tokens 的格式开销
    return total


def compute_dynamic_max_tokens(prompt: str, max_cap: int = 4096, min_tokens: int = 128) -> int:
    """
    根据输入问题的复杂度，动态计算合理的 max_tokens。
    
    简单问题 → 小 max_tokens（省额度）
    复杂问题 → 大 max_tokens（给足空间）
    
    Args:
        prompt: 用户问题
        max_cap: 最大上限
        min_tokens: 最小值
    
    Returns:
        建议的 max_tokens 值
    """
    input_tokens = estimate_tokens(prompt)
    
    # 短问题（< 50 tokens）: 问候、日期、简单查询
    if input_tokens < 50:
        return min_tokens  # 128
    
    # 中等问题（50-200 tokens）: 解释概念、写代码片段
    if input_tokens < 200:
        return 512
    
    # 较长问题（200-500 tokens）: 复杂分析、多步骤任务
    if input_tokens < 500:
        return 1024
    
    # 长问题（500+ tokens）: 详细需求、长文处理
    return min(max_cap, 2048)


def compress_history(history: list, max_rounds: int = 3, max_chars_per_round: int = 500) -> list:
    """
    压缩对话历史，保留关键信息同时减少 token 消耗。
    
    策略:
    1. 保留最近 max_rounds 轮的完整对话
    2. 更早的对话只保留摘要（截断长回复）
    3. 超长回复截断到 max_chars_per_round
    
    Args:
        history: [(user_msg, assistant_msg), ...] 格式的历史
        max_rounds: 保留完整内容的轮数
        max_chars_per_round: 早期对话每轮最大字符数
    
    Returns:
        压缩后的历史（同样格式）
    """
    if not history or len(history) <= max_rounds:
        return _truncate_rounds(history, max_chars_per_round)
    
    # 早期对话：截断保留
    early = history[:-max_rounds]
    recent = history[-max_rounds:]
    
    compressed_early = []
    for user_msg, bot_msg in early:
        # 用户消息保留完整（通常较短）
        # 助手回复截断
        if len(bot_msg) > max_chars_per_round:
            bot_msg = bot_msg[:max_chars_per_round] + "...(已省略)"
        compressed_early.append((user_msg, bot_msg))
    
    # 近期对话：截断超长回复但保留更多
    compressed_recent = _truncate_rounds(recent, max_chars_per_round * 2)
    
    return compressed_early + compressed_recent


def _truncate_rounds(history: list, max_chars: int) -> list:
    """截断每轮对话中过长的回复"""
    if not history:
        return []
    result = []
    for user_msg, bot_msg in history:
        if len(bot_msg) > max_chars:
            bot_msg = bot_msg[:max_chars] + "...(已省略)"
        result.append((user_msg, bot_msg))
    return result


def truncate_tool_result(result: str, tool_name: str = "") -> str:
    """
    根据工具类型智能截断结果。
    
    不同工具的结果合理长度不同：
    - search: 300 字符足够
    - read_file: 800 字符
    - execute_python: 500 字符
    - 其他: 600 字符
    """
    limits = {
        "search": 400,
        "read_webpage": 300,
        "read_local_file": 1000,
        "execute_python": 600,
        "list_directory": 400,
        "run_command": 400,
        "analyze_code": 600,
        "install_dependency": 200,
        "create_project": 200,
    }
    
    limit = limits.get(tool_name, 600)
    
    if len(result) <= limit:
        return result
    return result[:limit] + f"\n...(结果过长，已截断到 {limit} 字符)"


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "") -> float:
    """
    估算 API 调用费用（美元）。
    
    基于主流 API 价格估算，实际价格因提供商而异。
    """
    # 粗略价格（每 1M tokens，美元）
    prices = {
        "gpt-4o": (2.5, 10),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-3.5-turbo": (0.5, 1.5),
        "deepseek": (0.14, 0.28),
        "claude": (3, 15),
    }
    
    model_lower = model.lower()
    input_price, output_price = (0.5, 1.5)  # 默认价格
    
    for key, (ip, op) in prices.items():
        if key in model_lower:
            input_price, output_price = ip, op
            break
    
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 6)


class TokenTracker:
    """
    Token 使用追踪器（单例）。
    
    记录每次 API 调用的 token 消耗，提供统计报告。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._records = []
            cls._instance._total_input = 0
            cls._instance._total_output = 0
        return cls._instance
    
    def record(self, input_tokens: int, output_tokens: int, model: str = "", 
               endpoint: str = ""):
        """记录一次 API 调用"""
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._records.append({
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "model": model,
            "endpoint": endpoint,
        })
        # 只保留最近 1000 条记录
        if len(self._records) > 1000:
            self._records = self._records[-500:]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self._records:
            return {"total_input": 0, "total_output": 0, "total": 0, "calls": 0}
        
        return {
            "total_input": self._total_input,
            "total_output": self._total_output,
            "total": self._total_input + self._total_output,
            "calls": len(self._records),
            "avg_input": self._total_input // len(self._records),
            "avg_output": self._total_output // len(self._records),
            "last_10_calls": self._records[-10:],
        }
    
    def reset(self):
        """重置统计"""
        self._records.clear()
        self._total_input = 0
        self._total_output = 0


def get_tracker() -> TokenTracker:
    """获取全局 token 追踪器"""
    return TokenTracker()