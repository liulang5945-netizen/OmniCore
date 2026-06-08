"""
回测引擎
========
用历史数据模拟策略表现，计算绩效指标。
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("StockBrain.Backtester")


class BacktestResult:
    """回测结果"""

    def __init__(self):
        self.total_return = 0.0
        self.annual_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_loss_ratio = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.avg_holding_days = 0.0
        self.equity_curve = []
        self.trade_log = []

    def to_dict(self) -> dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "annual_return": f"{self.annual_return:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_loss_ratio": round(self.profit_loss_ratio, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": round(self.avg_holding_days, 1),
        }

    def summary(self) -> str:
        return (
            f"📊 回测结果\n"
            f"{'='*35}\n"
            f"总收益: {self.total_return:.2%}\n"
            f"年化收益: {self.annual_return:.2%}\n"
            f"夏普比率: {self.sharpe_ratio:.3f}\n"
            f"最大回撤: {self.max_drawdown:.2%}\n"
            f"胜率: {self.win_rate:.2%}\n"
            f"盈亏比: {self.profit_loss_ratio:.2f}\n"
            f"总交易次数: {self.total_trades}\n"
            f"盈利/亏损: {self.winning_trades}/{self.losing_trades}\n"
            f"平均持仓天数: {self.avg_holding_days:.1f}\n"
        )


def run_backtest(predictions: np.ndarray, actual_returns: np.ndarray,
                 dates: Optional[np.ndarray] = None,
                 threshold: float = 0.01, initial_capital: float = 100000,
                 stop_loss: float = 0.05, take_profit: float = 0.10,
                 position_ratio: float = 0.3) -> BacktestResult:
    """
    执行回测

    Args:
        predictions: 模型预测的涨跌幅数组
        actual_returns: 实际涨跌幅数组
        dates: 日期数组（可选）
        threshold: 开仓阈值（预测涨跌幅超过此值才交易）
        initial_capital: 初始资金
        stop_loss: 止损比例
        take_profit: 止盈比例
        position_ratio: 每次交易的仓位比例

    Returns:
        BacktestResult
    """
    result = BacktestResult()
    n = min(len(predictions), len(actual_returns))

    if n == 0:
        return result

    capital = initial_capital
    equity_curve = [capital]
    trades = []
    in_position = False
    entry_price = 0
    entry_idx = 0

    for i in range(n):
        pred = predictions[i]
        actual = actual_returns[i]

        if in_position:
            # 持仓中：检查止损止盈
            cumulative_return = actual
            if stop_loss > 0 and cumulative_return <= -stop_loss:
                # 止损
                pnl = capital * position_ratio * (-stop_loss)
                capital += pnl
                trades.append({
                    "type": "stop_loss",
                    "return": -stop_loss,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                })
                in_position = False
            elif take_profit > 0 and cumulative_return >= take_profit:
                # 止盈
                pnl = capital * position_ratio * take_profit
                capital += pnl
                trades.append({
                    "type": "take_profit",
                    "return": take_profit,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                })
                in_position = False
            elif i - entry_idx >= 5:
                # 持仓到期（5天）
                pnl = capital * position_ratio * actual
                capital += pnl
                trades.append({
                    "type": "hold_expire",
                    "return": actual,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                })
                in_position = False
        else:
            # 空仓：检查是否开仓
            if pred > threshold:
                # 做多信号
                in_position = True
                entry_price = 1.0
                entry_idx = i
            elif pred < -threshold:
                # 看空信号（空仓观望，不融券做空）
                pass

        equity_curve.append(capital)

    # 计算绩效指标
    result.equity_curve = equity_curve
    result.trade_log = trades
    result.total_trades = len(trades)

    if trades:
        profits = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
        result.winning_trades = len(profits)
        result.losing_trades = len(losses)
        result.win_rate = len(profits) / len(trades) if trades else 0

        avg_profit = np.mean(profits) if profits else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        result.profit_loss_ratio = avg_profit / (avg_loss + 1e-10)

        holding_days = [t.get("holding_days", 0) for t in trades]
        result.avg_holding_days = np.mean(holding_days) if holding_days else 0

    # 总收益
    result.total_return = (capital - initial_capital) / initial_capital

    # 年化收益（假设250个交易日）
    if n > 0:
        years = n / 250
        result.annual_return = (1 + result.total_return) ** (1 / max(years, 0.01)) - 1

    # 最大回撤
    peak = equity_curve[0]
    max_dd = 0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + 1e-10)
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown = max_dd

    # 夏普比率（简化版）
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-10)
        if np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(250)

    return result


def evaluate_prediction_accuracy(predictions: np.ndarray, actual: np.ndarray,
                                 threshold: float = 0.0) -> dict:
    """
    评估预测准确率

    Args:
        predictions: 预测涨跌幅
        actual: 实际涨跌幅
        threshold: 方向判断阈值

    Returns:
        {direction_accuracy, mae, rmse, corr}
    """
    n = min(len(predictions), len(actual))
    if n == 0:
        return {"direction_accuracy": 0, "mae": 0, "rmse": 0, "corr": 0}

    pred = predictions[:n]
    act = actual[:n]

    # 方向准确率
    pred_dir = (pred > threshold).astype(int)
    act_dir = (act > threshold).astype(int)
    direction_accuracy = np.mean(pred_dir == act_dir)

    # 误差
    mae = np.mean(np.abs(pred - act))
    rmse = np.sqrt(np.mean((pred - act) ** 2))

    # 相关系数
    corr = 0.0
    if np.std(pred) > 0 and np.std(act) > 0:
        corr = np.corrcoef(pred, act)[0, 1]

    return {
        "direction_accuracy": float(direction_accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": float(corr),
        "samples": n,
    }