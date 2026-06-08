"""
每日实战循环
============
盘前预测 → 盘中监控 → 收盘复盘 → 夜间自省
"""
import datetime
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("StockBrain.DailyCycle")

_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "daily_reports")


class DailyPrediction:
    """单只股票的每日预测"""
    def __init__(self, symbol: str, name: str = ""):
        self.symbol = symbol
        self.name = name
        self.predicted_return = 0.0
        self.predicted_direction = ""
        self.signal = ""
        self.confidence = 0.0
        self.suggested_action = ""
        self.support_price = 0.0
        self.resistance_price = 0.0

    def to_dict(self):
        return vars(self)


class DailyReport:
    """每日报告"""
    def __init__(self, date: str, phase: str):
        self.date = date
        self.phase = phase  # pre_market / intraday / post_market
        self.market_prediction = {}
        self.stock_predictions: List[dict] = []
        self.risk_check = {}
        self.postmortem = {}
        self.recommendations = []
        self.created_at = datetime.datetime.now().isoformat()

    def to_dict(self):
        return vars(self)

    def summary(self) -> str:
        lines = [f"📋 {self.date} {'盘前预测' if self.phase == 'pre_market' else '收盘复盘'}"]
        lines.append("=" * 40)
        if self.market_prediction:
            mp = self.market_prediction
            lines.append(f"大盘预测: {mp.get('direction', '?')} {mp.get('predicted_pct', '')}")
        if self.stock_predictions:
            lines.append(f"\n持仓预测:")
            for sp in self.stock_predictions:
                lines.append(f"  {sp.get('symbol', '')} {sp.get('name', '')}: "
                             f"{sp.get('predicted_direction', '?')} {sp.get('predicted_pct', '')} "
                             f"[{sp.get('signal', '')}]")
                if sp.get('suggested_action'):
                    lines.append(f"    建议: {sp['suggested_action']}")
        if self.postmortem:
            pm = self.postmortem
            lines.append(f"\n复盘:")
            lines.append(f"  方向准确率: {pm.get('direction_accuracy', 0):.0%}")
            if pm.get('errors'):
                for err in pm['errors'][:3]:
                    lines.append(f"  ⚠ {err}")
        if self.recommendations:
            lines.append(f"\n建议:")
            for r in self.recommendations:
                lines.append(f"  · {r}")
        return "\n".join(lines)


class DailyCycleEngine:
    """每日实战循环引擎"""

    def __init__(self, config):
        self.config = config
        os.makedirs(_REPORTS_DIR, exist_ok=True)
        self._consecutive_errors = 0
        self._daily_history: List[dict] = []

    def run_pre_market(self, symbols: List[str] = None) -> DailyReport:
        """盘前预测"""
        symbols = symbols or self.config.watch_list
        today = datetime.date.today().isoformat()
        report = DailyReport(today, "pre_market")

        try:
            from plugins.stock_brain.data.market_data import get_realtime_quote
            from plugins.stock_brain.model.stock_predictor import StockPredictor
            from plugins.stock_brain.data.feature_engineer import build_feature_matrix

            # 加载模型
            try:
                predictor = StockPredictor.load("best")
            except Exception:
                try:
                    predictor = StockPredictor.load("latest")
                except Exception:
                    report.recommendations.append("⚠ 未找到已训练模型，请先执行 brain_learn")
                    return report

            # 获取最新数据并预测
            from plugins.stock_brain.data.market_data import get_stock_history
            stock_preds = []
            for symbol in symbols:
                try:
                    df = get_stock_history(symbol, start_date=(
                        datetime.date.today() - datetime.timedelta(days=120)
                    ).strftime("%Y%m%d"))
                    if df is None or len(df) < 60:
                        continue

                    feat_df = build_feature_matrix(df, use_time=self.config.learning.use_time_features)
                    if len(feat_df) < predictor.input_size:
                        continue

                    # 取最后 lookback 天的数据
                    lookback = self.config.learning.lookback_days
                    window = feat_df.iloc[-lookback:]
                    feature_cols = predictor.feature_cols or [
                        c for c in feat_df.columns
                        if c not in {"date", "symbol", "name"} and np.issubdtype(feat_df[c].dtype, np.number)
                    ]
                    X = window[feature_cols].values.astype(np.float32)

                    result = predictor.predict_single(X)
                    quote = get_realtime_quote(symbol)

                    pred = {
                        "symbol": symbol,
                        "name": quote.get("name", "") if quote else "",
                        "current_price": quote.get("price", 0) if quote else 0,
                        "predicted_return": result["predicted_return"],
                        "predicted_pct": result["predicted_pct"],
                        "predicted_direction": result["direction"],
                        "signal": result["signal"],
                        "confidence": result["confidence"],
                    }

                    # 生成建议
                    if result["signal"] in ("强烈看多", "看多"):
                        pred["suggested_action"] = "持有/加仓"
                    elif result["signal"] in ("强烈看空", "看空"):
                        pred["suggested_action"] = "减仓/观望"
                    else:
                        pred["suggested_action"] = "持有不动"

                    stock_preds.append(pred)
                except Exception as e:
                    logger.warning(f"预测 {symbol} 失败: {e}")

            report.stock_predictions = stock_preds

            # 风控检查
            from plugins.stock_brain.engine.risk_controller import check_risk
            report.risk_check = check_risk(self.config, stock_preds)

            # 大盘预测
            if stock_preds:
                avg_pred = np.mean([p["predicted_return"] for p in stock_preds])
                report.market_prediction = {
                    "predicted_return": float(avg_pred),
                    "predicted_pct": f"{avg_pred * 100:.2f}%",
                    "direction": "上涨" if avg_pred > 0 else "下跌",
                }

            self._save_report(report)
            return report

        except Exception as e:
            logger.error(f"盘前预测失败: {e}", exc_info=True)
            report.recommendations.append(f"❌ 预测失败: {e}")
            return report

    def run_post_market(self, symbols: List[str] = None,
                        pre_market_report: DailyReport = None) -> DailyReport:
        """收盘复盘"""
        symbols = symbols or self.config.watch_list
        today = datetime.date.today().isoformat()
        report = DailyReport(today, "post_market")

        try:
            from plugins.stock_brain.data.market_data import get_realtime_quote

            # 获取实际收盘数据
            actual_results = []
            for symbol in symbols:
                quote = get_realtime_quote(symbol)
                if quote:
                    actual_results.append({
                        "symbol": symbol,
                        "name": quote.get("name", ""),
                        "actual_pct": quote.get("pct_change", 0) / 100,
                        "actual_direction": "上涨" if quote.get("pct_change", 0) > 0 else "下跌",
                        "close_price": quote.get("price", 0),
                    })

            report.stock_predictions = actual_results

            # 复盘分析
            if pre_market_report and pre_market_report.stock_predictions:
                from plugins.stock_brain.engine.postmortem import analyze_predictions
                report.postmortem = analyze_predictions(
                    pre_market_report.stock_predictions,
                    actual_results,
                )

                # 连续错误追踪
                if report.postmortem.get("direction_accuracy", 1) < 0.5:
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 0

                # 连续错误告警
                threshold = self.config.daily.consecutive_error_threshold
                if self._consecutive_errors >= threshold:
                    report.recommendations.append(
                        f"⚠ 连续 {self._consecutive_errors} 天胜率低于50%，建议重新训练模型"
                    )

            # 动态调整建议
            if report.postmortem and self.config.daily.auto_adjust:
                from plugins.stock_brain.engine.postmortem import suggest_adjustments
                adjustments = suggest_adjustments(report.postmortem, self.config)
                report.recommendations.extend(adjustments)

            self._save_report(report)
            self._daily_history.append(report.to_dict())
            return report

        except Exception as e:
            logger.error(f"收盘复盘失败: {e}", exc_info=True)
            report.recommendations.append(f"❌ 复盘失败: {e}")
            return report

    def _save_report(self, report: DailyReport):
        """保存报告"""
        try:
            fname = f"{report.date}_{report.phase}.json"
            path = os.path.join(_REPORTS_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"保存报告失败: {e}")

    def load_pre_market_report(self, date: str = None) -> Optional[DailyReport]:
        """加载盘前报告（用于收盘复盘对比）"""
        date = date or datetime.date.today().isoformat()
        path = os.path.join(_REPORTS_DIR, f"{date}_pre_market.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report = DailyReport(data["date"], data["phase"])
                report.stock_predictions = data.get("stock_predictions", [])
                report.market_prediction = data.get("market_prediction", {})
                return report
            except Exception:
                return None
        return None

    def get_win_rate_stats(self, days: int = 30) -> dict:
        """获取胜率统计"""
        reports = []
        for fname in sorted(os.listdir(_REPORTS_DIR), reverse=True):
            if fname.endswith("_post_market.json"):
                try:
                    with open(os.path.join(_REPORTS_DIR, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("postmortem", {}).get("direction_accuracy") is not None:
                        reports.append(data)
                except Exception:
                    continue
                if len(reports) >= days:
                    break

        if not reports:
            return {"message": "暂无复盘数据"}

        accuracies = [r["postmortem"]["direction_accuracy"] for r in reports]
        return {
            "total_days": len(reports),
            "avg_accuracy": np.mean(accuracies),
            "recent_7d": np.mean(accuracies[:7]) if len(accuracies) >= 7 else np.mean(accuracies),
            "recent_30d": np.mean(accuracies),
            "best_day": max(accuracies),
            "worst_day": min(accuracies),
        }