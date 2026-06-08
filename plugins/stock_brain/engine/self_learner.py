"""
自学习循环引擎
==============
核心流程：采集历史数据 → 特征工程 → 训练模型 → 回测 → 参数优化 → 迭代
"""
import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional, Callable

import numpy as np

logger = logging.getLogger("StockBrain.SelfLearner")

_ITERATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "iterations")


class LearningIteration:
    """一轮学习迭代的记录"""

    def __init__(self, iteration_id: int):
        self.iteration_id = iteration_id
        self.started_at = ""
        self.finished_at = ""
        self.train_result = {}
        self.backtest_result = {}
        self.evaluate_result = {}
        self.params_snapshot = {}
        self.improvement = 0.0
        self.status = "pending"
        self.log = []

    def to_dict(self) -> dict:
        return {
            "iteration_id": self.iteration_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "train_result": self.train_result,
            "backtest_result": self.backtest_result,
            "evaluate_result": self.evaluate_result,
            "params_snapshot": self.params_snapshot,
            "improvement": self.improvement,
            "status": self.status,
            "log": self.log[-20:],
        }


class SelfLearningEngine:
    """
    自学习循环引擎

    将整个学习过程自动化：
    数据采集 → 特征工程 → 训练 → 回测 → 评估 → 调参 → 再训练
    """

    def __init__(self, config, progress_callback: Optional[Callable] = None):
        """
        Args:
            config: StockBrainConfig 实例
            progress_callback: 进度回调 func(message: str)
        """
        self.config = config
        self._progress_cb = progress_callback
        self.iterations: List[LearningIteration] = []
        self.best_metrics = {}
        self.status = "idle"
        self._stop_flag = False

    def _log(self, msg: str):
        logger.info(msg)
        if self._progress_cb:
            self._progress_cb(msg)

    def stop(self):
        self._stop_flag = True

    def run(self, symbols: List[str] = None) -> dict:
        """
        执行完整的自学习循环

        Args:
            symbols: 股票代码列表（默认使用配置中的 watch_list）

        Returns:
            最终学习结果
        """
        symbols = symbols or self.config.watch_list
        self.status = "running"
        self._stop_flag = False
        os.makedirs(_ITERATIONS_DIR, exist_ok=True)

        self._log(f"🚀 启动自学习循环 | 股票池: {symbols} | 最大迭代: {self.config.learning.max_iterations}")

        try:
            # Phase 1: 采集数据
            self._log("=" * 50)
            self._log("📥 Phase 1: 采集历史数据")
            from plugins.stock_brain.data.market_data import batch_get_history
            market_data = batch_get_history(symbols, years=self.config.learning.train_years)
            if not market_data:
                return {"status": "failed", "error": "数据采集失败"}
            self._log(f"  采集完成: {len(market_data)} 只股票")

            # Phase 2: 特征工程
            self._log("=" * 50)
            self._log("🔧 Phase 2: 特征工程")
            from plugins.stock_brain.data.feature_engineer import build_feature_matrix, create_sliding_windows

            all_X, all_y = [], []
            feature_cols = None
            for symbol, df in market_data.items():
                feat_df = build_feature_matrix(
                    df,
                    use_time=self.config.learning.use_time_features,
                )
                X, y, cols = create_sliding_windows(
                    feat_df,
                    lookback=self.config.learning.lookback_days,
                    forecast=self.config.learning.forecast_days,
                )
                all_X.append(X)
                all_y.append(y)
                if feature_cols is None:
                    feature_cols = cols
                self._log(f"  {symbol}: {X.shape[0]} 个样本, {X.shape[2]} 个特征")

            X_all = np.concatenate(all_X, axis=0)
            y_all = np.concatenate(all_y, axis=0)
            self._log(f"  总数据集: {X_all.shape[0]} 个样本")

            # 划分训练/验证/测试集
            n = len(X_all)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)

            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
            X_test, y_test = X_all[val_end:], y_all[val_end:]

            self._log(f"  训练:{len(X_train)} 验证:{len(X_val)} 测试:{len(X_test)}")

            # Phase 3-5: 迭代训练
            best_sharpe = -999
            patience_counter = 0
            max_iterations = self.config.learning.max_iterations
            patience = self.config.learning.early_stop_patience

            for iteration in range(1, max_iterations + 1):
                if self._stop_flag:
                    self._log("⏹ 用户停止自学习")
                    break

                self._log(f"\n{'='*50}")
                self._log(f"🔄 迭代 {iteration}/{max_iterations}")

                iter_record = LearningIteration(iteration)
                iter_record.started_at = datetime.datetime.now().isoformat()

                try:
                    # 3a. 训练模型
                    self._log(f"  🏋️ 训练模型...")
                    from plugins.stock_brain.model.stock_predictor import StockPredictor

                    predictor = StockPredictor(
                        input_size=X_train.shape[2],
                        forecast_days=self.config.learning.forecast_days,
                    )
                    predictor.feature_cols = feature_cols or []

                    train_result = predictor.train(
                        X_train, y_train, X_val, y_val,
                        epochs=50, batch_size=32,
                    )
                    iter_record.train_result = train_result
                    self._log(f"  训练完成: loss={train_result.get('best_val_loss', 0):.6f}")

                    # 3b. 回测
                    self._log(f"  📊 回测验证...")
                    from plugins.stock_brain.engine.backtester import run_backtest, evaluate_prediction_accuracy

                    predictions = predictor.predict(X_test)
                    bt_result = run_backtest(
                        predictions, y_test,
                        threshold=0.005,
                        stop_loss=self.config.risk.stop_loss,
                        initial_capital=self.config.capital.current_capital,
                    )
                    iter_record.backtest_result = bt_result.to_dict()
                    self._log(f"  回测: 年化{bt_result.annual_return:.2%} 夏普{bt_result.sharpe_ratio:.3f} 胜率{bt_result.win_rate:.2%}")

                    # 3c. 评估
                    eval_result = evaluate_prediction_accuracy(predictions, y_test)
                    iter_record.evaluate_result = eval_result
                    self._log(f"  预测准确率: {eval_result['direction_accuracy']:.2%} 相关性: {eval_result['corr']:.3f}")

                    # 3d. 判断是否改善
                    current_sharpe = bt_result.sharpe_ratio
                    improvement = current_sharpe - best_sharpe
                    iter_record.improvement = improvement

                    if improvement > 0:
                        best_sharpe = current_sharpe
                        patience_counter = 0
                        predictor.save("best")
                        self._log(f"  ✨ 新最佳模型! 夏普: {current_sharpe:.3f}")
                        self.best_metrics = {
                            "sharpe": current_sharpe,
                            "annual_return": bt_result.annual_return,
                            "win_rate": bt_result.win_rate,
                            "max_drawdown": bt_result.max_drawdown,
                            "direction_accuracy": eval_result["direction_accuracy"],
                            "iteration": iteration,
                        }
                    else:
                        patience_counter += 1
                        self._log(f"  📉 未改善 (耐心 {patience_counter}/{patience})")

                    # 保存最新模型
                    predictor.save("latest")
                    iter_record.status = "completed"
                    iter_record.finished_at = datetime.datetime.now().isoformat()

                except Exception as e:
                    iter_record.status = "failed"
                    iter_record.log.append(f"错误: {e}")
                    self._log(f"  ❌ 迭代失败: {e}")

                self.iterations.append(iter_record)
                self._save_iteration(iter_record)

                # 早停检查
                if patience_counter >= patience:
                    self._log(f"\n⏹ 连续 {patience} 轮未改善，停止迭代")
                    break

                # 目标达成检查
                target = self.config.learning
                if (bt_result.sharpe_ratio >= target.target_sharpe and
                    bt_result.win_rate >= target.target_win_rate):
                    self._log(f"\n🎯 目标达成! 夏普={bt_result.sharpe_ratio:.3f} 胜率={bt_result.win_rate:.2%}")
                    break

            self.status = "completed"
            self._log(f"\n{'='*50}")
            self._log(f"🏁 自学习完成!")
            self._log(f"  总迭代: {len(self.iterations)} 轮")
            if self.best_metrics:
                self._log(f"  最佳夏普: {self.best_metrics.get('sharpe', 0):.3f}")
                self._log(f"  最佳年化: {self.best_metrics.get('annual_return', 0):.2%}")
                self._log(f"  最佳胜率: {self.best_metrics.get('win_rate', 0):.2%}")

            return {
                "status": "completed",
                "iterations": len(self.iterations),
                "best_metrics": self.best_metrics,
            }

        except Exception as e:
            self.status = "failed"
            self._log(f"❌ 自学习失败: {e}")
            logger.error(f"自学习失败: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _save_iteration(self, iteration: LearningIteration):
        """保存迭代记录"""
        try:
            path = os.path.join(_ITERATIONS_DIR, f"iter_{iteration.iteration_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(iteration.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"保存迭代记录失败: {e}")

    def get_progress(self) -> dict:
        """获取当前进度"""
        return {
            "status": self.status,
            "iterations_done": len(self.iterations),
            "iterations_max": self.config.learning.max_iterations,
            "best_metrics": self.best_metrics,
            "latest_iteration": self.iterations[-1].to_dict() if self.iterations else None,
        }

    def get_history(self) -> List[dict]:
        """获取所有迭代历史"""
        return [it.to_dict() for it in self.iterations]