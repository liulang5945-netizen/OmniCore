"""
stock_brain 插件工具注册
========================
注册 8 个 Agent 工具到 ToolRegistry
"""
import logging

logger = logging.getLogger("StockBrain.Tools")


def _get_config():
    from plugins.stock_brain.config import load_config
    return load_config()


def _brain_learn(input_str: str) -> str:
    """启动历史自学习循环"""
    try:
        config = _get_config()
        symbols = None
        if input_str.strip():
            symbols = [s.strip() for s in input_str.split(",") if s.strip()]

        from plugins.stock_brain.engine.self_learner import SelfLearningEngine
        engine = SelfLearningEngine(config)
        result = engine.run(symbols=symbols)

        if result.get("status") == "completed":
            bm = result.get("best_metrics", {})
            return (
                f"✅ 自学习完成\n"
                f"迭代轮数: {result.get('iterations', 0)}\n"
                f"最佳夏普比率: {bm.get('sharpe', 0):.3f}\n"
                f"最佳年化收益: {bm.get('annual_return', 0):.2%}\n"
                f"最佳胜率: {bm.get('win_rate', 0):.2%}\n"
                f"最大回撤: {bm.get('max_drawdown', 0):.2%}\n"
                f"预测准确率: {bm.get('direction_accuracy', 0):.2%}\n"
                f"\n模型已保存为 'best'，可用于 brain_predict / brain_daily"
            )
        else:
            return f"❌ 自学习失败: {result.get('error', '未知错误')}"
    except Exception as e:
        return f"❌ 自学习失败: {e}"


def _brain_daily(input_str: str) -> str:
    """每日实战循环"""
    try:
        config = _get_config()
        phase = input_str.strip().lower() if input_str.strip() else "pre_market"

        from plugins.stock_brain.engine.daily_cycle import DailyCycleEngine
        engine = DailyCycleEngine(config)

        if phase in ("pre_market", "pre", "盘前"):
            report = engine.run_pre_market()
        elif phase in ("post_market", "post", "收盘", "复盘"):
            pre_report = engine.load_pre_market_report()
            report = engine.run_post_market(pre_market_report=pre_report)
        else:
            report = engine.run_pre_market()

        return report.summary()
    except Exception as e:
        return f"❌ 每日循环失败: {e}"


def _brain_predict(input_str: str) -> str:
    """预测指定股票"""
    try:
        config = _get_config()
        symbols = [s.strip() for s in input_str.split(",") if s.strip()]
        if not symbols:
            symbols = config.watch_list

        from plugins.stock_brain.model.stock_predictor import StockPredictor
        from plugins.stock_brain.data.market_data import get_stock_history, get_realtime_quote
        from plugins.stock_brain.data.feature_engineer import build_feature_matrix
        import datetime
        import numpy as np

        try:
            predictor = StockPredictor.load("best")
        except Exception:
            predictor = StockPredictor.load("latest")

        results = []
        for symbol in symbols:
            try:
                df = get_stock_history(symbol, start_date=(
                    datetime.date.today() - datetime.timedelta(days=120)
                ).strftime("%Y%m%d"))
                if df is None or len(df) < 60:
                    results.append(f"{symbol}: 数据不足")
                    continue

                feat_df = build_feature_matrix(df, use_time=config.learning.use_time_features)
                lookback = config.learning.lookback_days
                window = feat_df.iloc[-lookback:]
                feature_cols = predictor.feature_cols or [
                    c for c in feat_df.columns
                    if c not in {"date", "symbol", "name"} and np.issubdtype(feat_df[c].dtype, np.number)
                ]
                X = window[feature_cols].values.astype(np.float32)
                pred = predictor.predict_single(X)
                quote = get_realtime_quote(symbol)

                name = quote.get("name", "") if quote else ""
                price = quote.get("price", 0) if quote else 0
                results.append(
                    f"📈 {symbol} {name} (现价: ¥{price})\n"
                    f"   预测涨跌幅: {pred['predicted_pct']}\n"
                    f"   方向: {pred['direction']} | 信号: {pred['signal']}\n"
                    f"   信心: {pred['confidence']:.0%}"
                )
            except Exception as e:
                results.append(f"{symbol}: 预测失败 - {e}")

        return "\n\n".join(results)
    except Exception as e:
        return f"❌ 预测失败: {e}"


def _brain_backtest(input_str: str) -> str:
    """回测策略"""
    try:
        config = _get_config()
        symbols = [s.strip() for s in input_str.split(",") if s.strip()] or config.watch_list

        from plugins.stock_brain.data.market_data import batch_get_history
        from plugins.stock_brain.data.feature_engineer import build_feature_matrix, create_sliding_windows
        from plugins.stock_brain.model.stock_predictor import StockPredictor
        from plugins.stock_brain.engine.backtester import run_backtest, evaluate_prediction_accuracy
        import numpy as np

        try:
            predictor = StockPredictor.load("best")
        except Exception:
            predictor = StockPredictor.load("latest")

        market_data = batch_get_history(symbols, years=config.learning.train_years)
        all_pred, all_actual = [], []

        for symbol, df in market_data.items():
            feat_df = build_feature_matrix(df, use_time=config.learning.use_time_features)
            X, y, _ = create_sliding_windows(
                feat_df, lookback=config.learning.lookback_days,
                forecast=config.learning.forecast_days,
            )
            if len(X) == 0:
                continue
            predictions = predictor.predict(X)
            all_pred.append(predictions)
            all_actual.append(y)

        if not all_pred:
            return "❌ 无有效数据进行回测"

        pred_all = np.concatenate(all_pred)
        actual_all = np.concatenate(all_actual)

        bt = run_backtest(
            pred_all, actual_all,
            threshold=0.005,
            stop_loss=config.risk.stop_loss,
            initial_capital=config.capital.current_capital,
        )
        eval_result = evaluate_prediction_accuracy(pred_all, actual_all)

        return (
            f"{bt.summary()}\n"
            f"预测评估:\n"
            f"  方向准确率: {eval_result['direction_accuracy']:.2%}\n"
            f"  平均绝对误差: {eval_result['mae']:.4f}\n"
            f"  相关系数: {eval_result['corr']:.3f}"
        )
    except Exception as e:
        return f"❌ 回测失败: {e}"


def _brain_report(input_str: str) -> str:
    """查看报告"""
    try:
        config = _get_config()
        report_type = input_str.strip()

        if report_type in ("daily", "今日", ""):
            from plugins.stock_brain.engine.daily_cycle import DailyCycleEngine
            engine = DailyCycleEngine(config)
            pre = engine.load_pre_market_report()
            if pre:
                return pre.summary()
            return "今日暂无盘前预测报告"

        elif report_type in ("winrate", "胜率"):
            from plugins.stock_brain.engine.daily_cycle import DailyCycleEngine
            engine = DailyCycleEngine(config)
            stats = engine.get_win_rate_stats()
            if "message" in stats:
                return stats["message"]
            return (
                f"📊 胜率统计\n"
                f"{'='*30}\n"
                f"统计天数: {stats['total_days']}\n"
                f"平均准确率: {stats['avg_accuracy']:.0%}\n"
                f"近7天: {stats['recent_7d']:.0%}\n"
                f"近30天: {stats['recent_30d']:.0%}\n"
                f"最佳日: {stats['best_day']:.0%}\n"
                f"最差日: {stats['worst_day']:.0%}"
            )

        elif report_type in ("learn", "学习"):
            from plugins.stock_brain.engine.self_learner import SelfLearningEngine
            engine = SelfLearningEngine(config)
            progress = engine.get_progress()
            if progress.get("best_metrics"):
                bm = progress["best_metrics"]
                return (
                    f"🧠 学习进度\n"
                    f"{'='*30}\n"
                    f"状态: {progress['status']}\n"
                    f"迭代轮数: {progress['iterations_done']}/{progress['iterations_max']}\n"
                    f"最佳夏普: {bm.get('sharpe', 0):.3f}\n"
                    f"最佳年化: {bm.get('annual_return', 0):.2%}\n"
                    f"最佳胜率: {bm.get('win_rate', 0):.2%}"
                )
            return "尚未开始学习，请执行 brain_learn"

        return f"未知报告类型: {report_type}。可选: daily / winrate / learn"
    except Exception as e:
        return f"❌ 获取报告失败: {e}"


def _brain_scan(input_str: str) -> str:
    """扫描选股"""
    try:
        config = _get_config()
        from plugins.stock_brain.data.market_data import get_stock_history, get_realtime_quote
        from plugins.stock_brain.data.feature_engineer import build_feature_matrix
        from plugins.stock_brain.model.stock_predictor import StockPredictor
        import datetime
        import numpy as np

        try:
            predictor = StockPredictor.load("best")
        except Exception:
            predictor = StockPredictor.load("latest")

        # 扫描 watch_list 中的股票
        symbols = config.watch_list
        if input_str.strip():
            symbols = [s.strip() for s in input_str.split(",") if s.strip()]

        bullish = []
        bearish = []

        for symbol in symbols:
            try:
                df = get_stock_history(symbol, start_date=(
                    datetime.date.today() - datetime.timedelta(days=120)
                ).strftime("%Y%m%d"))
                if df is None or len(df) < 60:
                    continue

                feat_df = build_feature_matrix(df, use_time=config.learning.use_time_features)
                lookback = config.learning.lookback_days
                window = feat_df.iloc[-lookback:]
                feature_cols = predictor.feature_cols or [
                    c for c in feat_df.columns
                    if c not in {"date", "symbol", "name"} and np.issubdtype(feat_df[c].dtype, np.number)
                ]
                X = window[feature_cols].values.astype(np.float32)
                pred = predictor.predict_single(X)
                quote = get_realtime_quote(symbol)

                entry = {
                    "symbol": symbol,
                    "name": quote.get("name", "") if quote else "",
                    "signal": pred["signal"],
                    "predicted_pct": pred["predicted_pct"],
                    "confidence": pred["confidence"],
                }

                if pred["signal"] in ("看多", "强烈看多"):
                    bullish.append(entry)
                elif pred["signal"] in ("看空", "强烈看空"):
                    bearish.append(entry)
            except Exception:
                continue

        # 按信心排序
        bullish.sort(key=lambda x: x["confidence"], reverse=True)
        bearish.sort(key=lambda x: x["confidence"], reverse=True)

        lines = [f"🔍 市场扫描结果 ({datetime.date.today()})", "=" * 40]
        if bullish:
            lines.append(f"\n🟢 看多 ({len(bullish)} 只):")
            for b in bullish:
                lines.append(f"  {b['symbol']} {b['name']}: {b['signal']} {b['predicted_pct']} (信心:{b['confidence']:.0%})")
        if bearish:
            lines.append(f"\n🔴 看空 ({len(bearish)} 只):")
            for b in bearish:
                lines.append(f"  {b['symbol']} {b['name']}: {b['signal']} {b['predicted_pct']} (信心:{b['confidence']:.0%})")
        if not bullish and not bearish:
            lines.append("\n⚪ 无明确信号，建议观望")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 扫描失败: {e}"


def _brain_risk(input_str: str) -> str:
    """设置/查看风险偏好"""
    try:
        config = _get_config()

        if not input_str.strip():
            # 查看当前配置
            r = config.risk
            c = config.capital
            return (
                f"🛡️ 当前风险配置\n"
                f"{'='*35}\n"
                f"风险偏好: {r.risk_tolerance}\n"
                f"目标年化: {r.target_annual_return:.0%}\n"
                f"最大回撤: {r.max_drawdown:.0%}\n"
                f"止损线: {r.stop_loss:.0%}\n"
                f"单股最大仓位: {r.max_single_position:.0%}\n"
                f"最大持仓数: {r.max_positions}\n"
                f"现金保留: {r.cash_reserve:.0%}\n"
                f"选股偏好: {r.stock_preference}\n"
                f"持有周期: {r.preferred_hold_period}\n"
                f"本金: ¥{c.current_capital:,.0f}\n"
                f"策略模式: {c.strategy_mode}"
            )

        # 设置新风险偏好
        from plugins.stock_brain.config import apply_risk_profile, save_config
        level = input_str.strip()
        if level not in ("保守", "稳健", "积极", "激进"):
            return "❌ 无效风险等级。可选: 保守 / 稳健 / 积极 / 激进"

        config = apply_risk_profile(config, level)
        save_config(config)
        return f"✅ 风险偏好已设置为: {level}\n" + _brain_risk("")
    except Exception as e:
        return f"❌ 操作失败: {e}"


def _brain_capital(input_str: str) -> str:
    """设置/查看本金与资本适配"""
    try:
        config = _get_config()

        if not input_str.strip():
            c = config.capital
            from plugins.stock_brain.config import CAPITAL_STRATEGIES
            strategy = CAPITAL_STRATEGIES.get(c.strategy_mode, {})
            return (
                f"💰 资本配置\n"
                f"{'='*35}\n"
                f"初始本金: ¥{c.initial_capital:,.0f}\n"
                f"当前本金: ¥{c.current_capital:,.0f}\n"
                f"策略模式: {c.strategy_mode}\n"
                f"策略描述: {strategy.get('description', '')}\n"
                f"最大持仓: {strategy.get('max_positions', '?')} 只\n"
                f"止损线: {strategy.get('stop_loss', 0):.0%}"
            )

        # 设置新本金
        amount = float(input_str.strip().replace(",", "").replace("¥", ""))
        from plugins.stock_brain.config import apply_capital_strategy, save_config
        config = apply_capital_strategy(config, amount)
        config.capital.initial_capital = amount
        save_config(config)
        return f"✅ 本金已设置为: ¥{amount:,.0f}\n" + _brain_capital("")
    except ValueError:
        return "❌ 请输入有效金额，如: 300000"
    except Exception as e:
        return f"❌ 操作失败: {e}"


def register_tools():
    """注册所有 stock_brain 工具到 ToolRegistry"""
    from agent.tool_registry import registry, ToolDef

    tools = [
        ToolDef(
            name="brain_learn",
            description="启动自学习循环：采集历史数据 → 特征工程 → 训练模型 → 回测 → 参数优化 → 迭代。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选,留空使用默认股票池)"}
            }, "required": ["input"]},
            func=_brain_learn,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_daily",
            description="每日实战循环。输入: pre_market(盘前预测) 或 post_market(收盘复盘)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "pre_market 或 post_market"}
            }, "required": ["input"]},
            func=_brain_daily,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_predict",
            description="预测指定股票走势。输入股票代码(逗号分隔)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码,如 600519,300750"}
            }, "required": ["input"]},
            func=_brain_predict,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_backtest",
            description="回测策略在历史数据上的表现。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选)"}
            }, "required": ["input"]},
            func=_brain_backtest,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_report",
            description="查看报告。输入: daily(今日) / winrate(胜率) / learn(学习进度)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "报告类型: daily/winrate/learn"}
            }, "required": ["input"]},
            func=_brain_report,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_scan",
            description="扫描市场选股。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选)"}
            }, "required": ["input"]},
            func=_brain_scan,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_risk",
            description="设置/查看风险偏好。输入: 保守/稳健/积极/激进 或留空查看当前配置。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "风险等级(可选)"}
            }, "required": ["input"]},
            func=_brain_risk,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_capital",
            description="设置/查看本金与资本适配策略。输入金额或留空查看。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "金额(可选,如 300000)"}
            }, "required": ["input"]},
            func=_brain_capital,
            source="plugin", source_id="stock_brain", category="量化",
        ),
    ]

    registry.register_many(tools)
    logger.info(f"stock_brain: 已注册 {len(tools)} 个工具")


# ======================== 交易工具 ========================

_broker_instance = None


def _get_broker(broker_type: str = "sim"):
    """获取或创建券商实例（单例）"""
    global _broker_instance
    if _broker_instance is not None:
        return _broker_instance
    try:
        # 触发注册
        from plugins.stock_brain.trading import sim_broker
        from plugins.stock_brain.trading import ths_broker
        from plugins.stock_brain.trading import em_broker
        from plugins.stock_brain.trading.broker_adapter import create_broker
        _broker_instance = create_broker(broker_type)
        _broker_instance.connect()
        return _broker_instance
    except Exception as e:
        raise RuntimeError(f"连接券商失败: {e}")


def _brain_trade(input_str: str) -> str:
    """
    交易执行（半自动模式）
    输入格式: 操作 | 股票代码 | 数量 | 价格 | 券商类型(可选)
    示例:
      buy | 600519 | 100 | 1800
      sell | 300750 | 200 | 250
      account
      positions
      orders
      reset | 500000
    """
    try:
        parts = [p.strip() for p in input_str.split("|")]
        action = parts[0].lower() if parts else ""

        broker_type = parts[4] if len(parts) > 4 else "sim"
        broker = _get_broker(broker_type)

        if action in ("account", "账户", ""):
            acct = broker.get_account()
            lines = [
                f"💰 账户信息 ({broker.name})",
                f"{'='*35}",
                f"总资产: ¥{acct.total_assets:,.2f}",
                f"可用资金: ¥{acct.available_cash:,.2f}",
                f"持仓市值: ¥{acct.market_value:,.2f}",
                f"总盈亏: ¥{acct.total_profit:,.2f}",
            ]
            if acct.positions:
                lines.append(f"\n📊 持仓 ({len(acct.positions)} 只):")
                for p in acct.positions:
                    pnl_icon = "🟢" if p.profit >= 0 else "🔴"
                    lines.append(
                        f"  {pnl_icon} {p.symbol} {p.name}: "
                        f"{p.quantity}股 | 成本¥{p.cost_price:.2f} | "
                        f"现价¥{p.current_price:.2f} | "
                        f"盈亏 ¥{p.profit:,.2f} ({p.profit_pct:.2%})"
                    )
            return "\n".join(lines)

        elif action in ("positions", "持仓"):
            positions = broker.get_positions()
            if not positions:
                return "📊 当前无持仓"
            lines = [f"📊 当前持仓 ({len(positions)} 只):"]
            for p in positions:
                pnl_icon = "🟢" if p.profit >= 0 else "🔴"
                lines.append(
                    f"  {pnl_icon} {p.symbol} {p.name}: "
                    f"{p.quantity}股(可卖{p.available_quantity}) | "
                    f"成本¥{p.cost_price:.2f} → 现价¥{p.current_price:.2f} | "
                    f"盈亏 ¥{p.profit:,.2f} ({p.profit_pct:.2%})"
                )
            return "\n".join(lines)

        elif action in ("orders", "委托"):
            orders = broker.get_today_orders()
            if not orders:
                return "📋 今日无委托"
            lines = [f"📋 今日委托 ({len(orders)} 条):"]
            for o in orders:
                lines.append(
                    f"  {o.order_id} | {o.symbol} | {o.side} | "
                    f"{o.quantity}股 @ ¥{o.price:.2f} | {o.status} | {o.note}"
                )
            return "\n".join(lines)

        elif action in ("reset", "重置"):
            from plugins.stock_brain.trading.sim_broker import SimBroker
            if isinstance(broker, SimBroker):
                new_cash = float(parts[1].strip()) if len(parts) > 1 else 100000
                broker.reset(new_cash)
                return f"✅ 模拟账户已重置: ¥{new_cash:,.0f}"
            return "❌ 重置仅支持模拟账户"

        elif action in ("buy", "买入", "sell", "卖出"):
            if len(parts) < 4:
                return "❌ 格式: buy | 股票代码 | 数量 | 价格"
            symbol = parts[1].strip()
            quantity = int(parts[2].strip())
            price = float(parts[3].strip()) if parts[3].strip() else 0

            # 风控检查
            from plugins.stock_brain.engine.risk_controller import check_risk
            config = _get_config()
            risk_check = check_risk(config, [{"symbol": symbol, "signal": "看多" if action in ("buy", "买入") else "看空", "confidence": 0.5}])
            if risk_check.get("violations"):
                return f"🛡️ 风控拦截:\n" + "\n".join(risk_check["violations"])

            side = "buy" if action in ("buy", "买入") else "sell"
            order = broker.place_order(symbol, side, quantity, price)

            result = f"{'✅' if order.status == 'filled' else '⚠' if order.status == 'submitted' else '❌'} {order.note}\n"
            result += f"订单状态: {order.status} | 订单号: {order.order_id}"
            if order.commission > 0:
                result += f"\n手续费: ¥{order.commission:.2f}"

            # 提示半自动模式
            if broker_type == "sim":
                result += "\n💡 当前为模拟交易模式，切换到实盘请指定: ths(同花顺) 或 em(东方财富)"
            return result

        else:
            return (
                "❌ 未知操作。可选:\n"
                "  account — 查看账户\n"
                "  positions — 查看持仓\n"
                "  orders — 查看今日委托\n"
                "  buy | 代码 | 数量 | 价格 — 买入\n"
                "  sell | 代码 | 数量 | 价格 — 卖出\n"
                "  reset | 金额 — 重置模拟账户"
            )
    except Exception as e:
        return f"❌ 交易操作失败: {e}"


def _brain_portfolio(input_str: str) -> str:
    """查看完整持仓报告（含盈亏分析）"""
    try:
        broker_type = input_str.strip() or "sim"
        broker = _get_broker(broker_type)
        acct = broker.get_account()

        lines = [
            f"📊 完整持仓报告 ({broker.name})",
            f"{'='*40}",
            f"总资产: ¥{acct.total_assets:,.2f}",
            f"可用资金: ¥{acct.available_cash:,.2f}",
            f"持仓市值: ¥{acct.market_value:,.2f}",
            f"总盈亏: ¥{acct.total_profit:,.2f}",
            f"仓位比例: {acct.market_value / (acct.total_assets + 1e-10):.1%}",
        ]

        if acct.positions:
            total_profit_pct = acct.total_profit / (acct.total_assets - acct.total_profit + 1e-10)
            lines.append(f"总收益率: {total_profit_pct:.2%}")
            lines.append(f"\n{'─'*40}")

            for p in sorted(acct.positions, key=lambda x: x.profit, reverse=True):
                pnl_icon = "🟢" if p.profit >= 0 else "🔴"
                lines.extend([
                    f"\n{pnl_icon} {p.symbol} {p.name}",
                    f"  持股: {p.quantity}股 (可卖{p.available_quantity}股)",
                    f"  成本: ¥{p.cost_price:.2f} × {p.quantity} = ¥{p.cost_price * p.quantity:,.2f}",
                    f"  现价: ¥{p.current_price:.2f} × {p.quantity} = ¥{p.market_value:,.2f}",
                    f"  盈亏: ¥{p.profit:,.2f} ({p.profit_pct:.2%})",
                ])

            # 集中度分析
            if acct.positions:
                max_pos = max(acct.positions, key=lambda x: x.market_value)
                concentration = max_pos.market_value / (acct.market_value + 1e-10)
                if concentration > 0.5:
                    lines.append(f"\n⚠ 持仓集中度警告: {max_pos.name}占比{concentration:.1%}，建议分散")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 获取持仓报告失败: {e}"


def register_tools():
    """注册所有 stock_brain 工具到 ToolRegistry"""
    from agent.tool_registry import registry, ToolDef

    tools = [
        ToolDef(
            name="brain_learn",
            description="启动自学习循环：采集历史数据 → 特征工程 → 训练模型 → 回测 → 参数优化 → 迭代。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选,留空使用默认股票池)"}
            }, "required": ["input"]},
            func=_brain_learn,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_daily",
            description="每日实战循环。输入: pre_market(盘前预测) 或 post_market(收盘复盘)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "pre_market 或 post_market"}
            }, "required": ["input"]},
            func=_brain_daily,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_predict",
            description="预测指定股票走势。输入股票代码(逗号分隔)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码,如 600519,300750"}
            }, "required": ["input"]},
            func=_brain_predict,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_backtest",
            description="回测策略在历史数据上的表现。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选)"}
            }, "required": ["input"]},
            func=_brain_backtest,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_report",
            description="查看报告。输入: daily(今日) / winrate(胜率) / learn(学习进度)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "报告类型: daily/winrate/learn"}
            }, "required": ["input"]},
            func=_brain_report,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_scan",
            description="扫描市场选股。输入股票代码(逗号分隔,可选)。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "股票代码(逗号分隔,可选)"}
            }, "required": ["input"]},
            func=_brain_scan,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_risk",
            description="设置/查看风险偏好。输入: 保守/稳健/积极/激进 或留空查看当前配置。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "风险等级(可选)"}
            }, "required": ["input"]},
            func=_brain_risk,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        ToolDef(
            name="brain_capital",
            description="设置/查看本金与资本适配策略。输入金额或留空查看。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "金额(可选,如 300000)"}
            }, "required": ["input"]},
            func=_brain_capital,
            source="plugin", source_id="stock_brain", category="量化",
        ),
        # 交易工具
        ToolDef(
            name="brain_trade",
            description="交易执行（支持模拟/同花顺/东方财富）。格式: buy|代码|数量|价格 或 sell|代码|数量|价格 或 account/positions/orders",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "操作|股票代码|数量|价格|券商(可选sim/ths/em)"}
            }, "required": ["input"]},
            func=_brain_trade,
            source="plugin", source_id="stock_brain", category="交易",
        ),
        ToolDef(
            name="brain_portfolio",
            description="查看完整持仓报告（含盈亏分析和集中度检查）。输入券商类型(可选: sim/ths/em)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "券商类型(可选: sim/ths/em)"}
            }, "required": ["input"]},
            func=_brain_portfolio,
            source="plugin", source_id="stock_brain", category="交易",
        ),
    ]

    registry.register_many(tools)
    logger.info(f"stock_brain: 已注册 {len(tools)} 个工具")


def unregister_tools():
    """注销所有 stock_brain 工具"""
    from agent.tool_registry import registry
    registry.unregister_by_source("stock_brain")
