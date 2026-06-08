"""
复盘分析器
==========
比对盘前预测 vs 实际结果，分析差异原因，生成参数调整建议。
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("StockBrain.Postmortem")


def analyze_predictions(predictions: List[dict], actual_results: List[dict]) -> dict:
    """
    分析预测 vs 实际结果

    Args:
        predictions: 盘前预测 [{symbol, predicted_return, predicted_direction, signal}]
        actual_results: 实际结果 [{symbol, actual_pct, actual_direction}]

    Returns:
        复盘分析结果
    """
    # 按 symbol 匹配
    actual_map = {a["symbol"]: a for a in actual_results}
    pred_map = {p["symbol"]: p for p in predictions}

    common_symbols = set(actual_map.keys()) & set(pred_map.keys())
    if not common_symbols:
        return {"error": "无匹配的预测/实际数据"}

    direction_correct = 0
    total = 0
    errors = []
    stock_details = []

    for symbol in common_symbols:
        pred = pred_map[symbol]
        actual = actual_map[symbol]

        pred_return = pred.get("predicted_return", 0)
        actual_pct = actual.get("actual_pct", 0)

        # 方向判断
        pred_direction = "上涨" if pred_return > 0 else "下跌"
        actual_direction = actual.get("actual_direction", "上涨" if actual_pct > 0 else "下跌")
        is_correct = pred_direction == actual_direction

        if is_correct:
            direction_correct += 1
        total += 1

        # 幅度误差
        amplitude_error = abs(pred_return - actual_pct)

        detail = {
            "symbol": symbol,
            "name": pred.get("name", ""),
            "predicted_return": pred_return,
            "predicted_pct": f"{pred_return * 100:.2f}%",
            "actual_pct": actual_pct,
            "actual_pct_str": f"{actual_pct * 100:.2f}%",
            "direction_correct": is_correct,
            "amplitude_error": amplitude_error,
            "signal": pred.get("signal", ""),
        }
        stock_details.append(detail)

        if not is_correct:
            errors.append(
                f"{symbol}({pred.get('name', '')}): "
                f"预测{pred_direction} {pred_return*100:.2f}% → 实际{actual_direction} {actual_pct*100:.2f}%"
            )

    direction_accuracy = direction_correct / total if total > 0 else 0
    amplitude_errors = [d["amplitude_error"] for d in stock_details]
    avg_amplitude_error = np.mean(amplitude_errors) if amplitude_errors else 0

    # 检测系统性偏差
    pred_returns = [pred_map[s]["predicted_return"] for s in common_symbols]
    actual_returns = [actual_map[s]["actual_pct"] for s in common_symbols]
    bias = np.mean(pred_returns) - np.mean(actual_returns)
    bias_type = "高估" if bias > 0.005 else ("低估" if bias < -0.005 else "无偏")

    return {
        "direction_accuracy": direction_accuracy,
        "direction_correct": direction_correct,
        "direction_total": total,
        "avg_amplitude_error": float(avg_amplitude_error),
        "bias": float(bias),
        "bias_type": bias_type,
        "errors": errors,
        "stock_details": stock_details,
    }


def suggest_adjustments(postmortem: dict, config) -> List[str]:
    """
    基于复盘结果生成参数调整建议

    Args:
        postmortem: analyze_predictions 的输出
        config: StockBrainConfig

    Returns:
        建议列表
    """
    suggestions = []

    accuracy = postmortem.get("direction_accuracy", 0.5)
    bias_type = postmortem.get("bias_type", "无偏")
    bias = postmortem.get("bias", 0)
    errors = postmortem.get("errors", [])

    # 1. 方向准确率建议
    if accuracy < 0.4:
        suggestions.append(
            "🔴 方向准确率过低(< 40%)，建议重新训练模型，增加训练数据或调整特征"
        )
    elif accuracy < 0.5:
        suggestions.append(
            "🟡 方向准确率偏低(< 50%)，建议增加事件特征权重或调整回看天数"
        )
    elif accuracy > 0.7:
        suggestions.append(
            "🟢 方向准确率良好(> 70%)，当前模型表现稳定"
        )

    # 2. 系统性偏差修正
    if bias_type == "高估":
        suggestions.append(
            f"📊 模型系统性高估收益(偏差 {bias*100:.2f}%)，建议："
            f"降低模型预测的缩放系数，或提高开仓阈值"
        )
    elif bias_type == "低估":
        suggestions.append(
            f"📊 模型系统性低估收益(偏差 {bias*100:.2f}%)，建议："
            f"适当提高模型预测的缩放系数"
        )

    # 3. 个股特定错误分析
    stock_details = postmortem.get("stock_details", [])
    for detail in stock_details:
        if not detail.get("direction_correct") and detail.get("amplitude_error", 0) > 0.03:
            suggestions.append(
                f"⚠ {detail['symbol']}({detail.get('name', '')}): "
                f"方向错误且幅度误差大({detail['amplitude_error']*100:.1f}%)，"
                f"建议检查该股是否有特殊事件影响"
            )

    # 4. 风险偏好适配
    risk = config.risk
    if accuracy < 0.5 and risk.risk_tolerance in ("积极", "激进"):
        suggestions.append(
            "⚠ 当前准确率下不应使用高风险策略，建议降级为'稳健'"
        )

    # 5. 连续错误
    if len(errors) >= 3:
        suggestions.append(
            f"⚠ 今日 {len(errors)} 只股票方向判断错误，建议减少持仓数量，降低仓位"
        )

    return suggestions


def generate_weekly_summary(reports: List[dict]) -> str:
    """生成周度总结"""
    if not reports:
        return "本周暂无复盘数据"

    total_days = len(reports)
    accuracies = [r.get("postmortem", {}).get("direction_accuracy", 0) for r in reports]
    avg_accuracy = np.mean(accuracies) if accuracies else 0

    all_errors = []
    for r in reports:
        all_errors.extend(r.get("postmortem", {}).get("errors", []))

    lines = [
        f"📊 周度复盘总结",
        f"{'='*35}",
        f"交易天数: {total_days}",
        f"平均方向准确率: {avg_accuracy:.0%}",
        f"最佳日: {max(accuracies):.0%}" if accuracies else "",
        f"最差日: {min(accuracies):.0%}" if accuracies else "",
        f"总错误数: {len(all_errors)}",
    ]

    if avg_accuracy >= 0.6:
        lines.append("\n✅ 本周表现良好，继续保持当前策略")
    elif avg_accuracy >= 0.5:
        lines.append("\n📊 本周表现一般，建议微调参数")
    else:
        lines.append("\n⚠ 本周表现不佳，建议暂停交易并重新训练模型")

    return "\n".join(filter(None, lines))