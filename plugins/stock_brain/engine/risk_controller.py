"""
风险控制器
==========
根据风险偏好和资本适配策略，检查交易信号是否合规。
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("StockBrain.RiskController")


def check_risk(config, predictions: List[dict]) -> dict:
    """
    风控检查

    Args:
        config: StockBrainConfig
        predictions: 预测结果列表 [{symbol, signal, confidence, ...}]

    Returns:
        {passed: bool, warnings: [], violations: [], adjustments: []}
    """
    risk = config.risk
    capital = config.capital
    warnings = []
    violations = []
    adjustments = []

    # 1. 检查仓位集中度
    bullish_count = sum(1 for p in predictions if p.get("signal", "") in ("看多", "强烈看多"))
    if bullish_count > risk.max_positions:
        warnings.append(
            f"看多信号数({bullish_count})超过最大持仓数({risk.max_positions})，"
            f"建议仅选择信心最高的 {risk.max_positions} 只"
        )

    # 2. 检查单股仓位
    if risk.max_single_position < 1.0:
        max_pos_pct = f"{risk.max_single_position:.0%}"
        adjustments.append(f"单股最大仓位: {max_pos_pct}")

    # 3. 检查现金保留
    if risk.cash_reserve > 0:
        adjustments.append(f"现金保留: {risk.cash_reserve:.0%}")

    # 4. 低信心信号告警
    for pred in predictions:
        if pred.get("confidence", 0) < 0.3:
            warnings.append(
                f"{pred.get('symbol', '')} 信心偏低({pred.get('confidence', 0):.0%})，建议观望"
            )

    # 5. 检查止损线
    adjustments.append(f"止损线: {risk.stop_loss:.0%}")
    adjustments.append(f"最大回撤: {risk.max_drawdown:.0%}")

    # 6. 资本适配检查
    strategy = capital.strategy_mode
    adjustments.append(f"策略模式: {strategy} (本金: ¥{capital.current_capital:,.0f})")

    # 7. 极端市场检测（所有股票强烈看空）
    all_bearish = all(p.get("signal", "") in ("看空", "强烈看空") for p in predictions)
    if all_bearish:
        violations.append("⚠ 全部看空！建议清仓观望，保留现金")

    return {
        "passed": len(violations) == 0,
        "warnings": warnings,
        "violations": violations,
        "adjustments": adjustments,
        "risk_level": risk.risk_tolerance,
        "strategy_mode": strategy,
    }


def calculate_position_size(capital: float, risk_tolerance: str,
                            num_stocks: int, stop_loss: float) -> dict:
    """
    计算仓位大小

    Returns:
        {per_stock_amount, per_stock_ratio, total_invested, cash_remaining}
    """
    from plugins.stock_brain.config import RISK_PROFILES
    profile = RISK_PROFILES.get(risk_tolerance, RISK_PROFILES["稳健"])

    # 可投资金额 = 总资金 * (1 - 现金保留比例)
    investable = capital * (1 - profile["cash_reserve"])

    # 每只股票的金额
    actual_stocks = min(num_stocks, profile["max_positions"])
    per_stock = investable / max(actual_stocks, 1)

    # 每只股票不超过最大仓位
    max_per_stock = capital * profile["max_single_position"]
    per_stock = min(per_stock, max_per_stock)

    total_invested = per_stock * actual_stocks
    cash_remaining = capital - total_invested

    # 根据止损线计算最大可亏损
    max_loss_per_stock = per_stock * stop_loss
    total_max_loss = max_loss_per_stock * actual_stocks

    return {
        "capital": capital,
        "investable": investable,
        "num_stocks": actual_stocks,
        "per_stock_amount": round(per_stock, 2),
        "per_stock_ratio": round(per_stock / capital, 4),
        "total_invested": round(total_invested, 2),
        "cash_remaining": round(cash_remaining, 2),
        "cash_ratio": round(cash_remaining / capital, 4),
        "max_loss_per_stock": round(max_loss_per_stock, 2),
        "total_max_loss": round(total_max_loss, 2),
    }


def dynamic_risk_adjust(config, current_return: float, days_elapsed: int) -> dict:
    """
    动态风险调整

    根据当前收益与目标的差距，动态调整风险参数。

    Args:
        current_return: 当前年化收益率
        days_elapsed: 已过天数

    Returns:
        调整建议
    """
    risk = config.risk
    target = risk.target_annual_return
    year_progress = days_elapsed / 365.0

    adjustments = []

    if year_progress > 0.1:
        # 预期收益 = 当前收益 / 年进度
        projected = current_return / max(year_progress, 0.01)
        gap = target - projected

        if gap > 0.1:
            # 收益落后较多
            adjustments.append(f"⚠ 收益落后目标 {gap:.1%}，可适度放宽风险（在上限内）")
            adjustments.append("建议: 可将止损线放宽 1-2%，或增加仓位比例")
        elif gap < -0.05:
            # 超额完成
            adjustments.append(f"✅ 超额完成目标 {abs(gap):.1%}，建议收紧风险锁定利润")
            adjustments.append("建议: 收紧止损线，降低仓位，转为防守模式")
        else:
            adjustments.append(f"📊 收益进度正常，目标差距 {gap:.1%}，维持当前策略")

    return {
        "current_return": current_return,
        "target_return": target,
        "year_progress": year_progress,
        "adjustments": adjustments,
    }