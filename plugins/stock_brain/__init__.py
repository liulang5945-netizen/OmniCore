"""
Stock Brain 插件入口
====================
自学习量化投资系统

功能概述：
- 历史数据采集 + 政策事件编码
- 技术指标特征工程
- LSTM 时序预测模型
- 自学习循环（训练 → 回测 → 调参 → 迭代）
- 每日实战（盘前预测 → 收盘复盘 → 算法调整）
- 资本适配（不同本金不同策略）
- 风险偏好控制（保守/稳健/积极/激进）
"""
import logging

logger = logging.getLogger("StockBrain")


def register_tools():
    """插件加载时调用：注册所有工具到 ToolRegistry"""
    from plugins.stock_brain.tools import register_tools as _reg
    _reg()
    logger.info("Stock Brain 插件已加载，8 个工具已注册")


def unregister_tools():
    """插件卸载时调用：注销所有工具"""
    from plugins.stock_brain.tools import unregister_tools as _unreg
    _unreg()
    logger.info("Stock Brain 插件已卸载")