"""
时序预测模型
============
基于 LSTM 的股价走势预测模型。
预测未来 N 天的涨跌幅。
支持训练、预测、保存/加载。
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("StockBrain.Predictor")

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "models")


def _ensure_model_dir():
    os.makedirs(_MODEL_DIR, exist_ok=True)


class StockPredictor:
    """
    LSTM 时序预测模型
    预测未来涨跌幅（回归任务）
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 forecast_days: int = 5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_days = forecast_days
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.train_history = []
        self._build_model()

    def _build_model(self):
        """构建 LSTM 模型"""
        try:
            import torch
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True,
                    )
                    self.norm = nn.LayerNorm(hidden_size)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, output_size),
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_hidden = lstm_out[:, -1, :]
                    out = self.norm(last_hidden)
                    return self.fc(out).squeeze(-1)

            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            logger.info(f"构建 LSTM 模型: input={self.input_size}, hidden={self.hidden_size}, layers={self.num_layers}")

        except ImportError:
            logger.warning("PyTorch 未安装，将使用线性回归降级方案")
            self.model = "sklearn_fallback"

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, lr: float = 0.001,
              early_stopping_patience: int = 5) -> dict:
        """
        训练模型

        Returns:
            训练结果字典: {epochs, best_val_loss, train_loss, history}
        """
        # 标准化特征
        X_train, self.scaler = self._normalize(X_train)
        if X_val is not None:
            X_val = self._apply_normalize(X_val)

        if self.model == "sklearn_fallback":
            return self._train_sklearn(X_train, y_train, X_val, y_val)

        return self._train_pytorch(X_train, y_train, X_val, y_val,
                                   epochs, batch_size, lr, early_stopping_patience)

    def _train_pytorch(self, X_train, y_train, X_val, y_val,
                       epochs, batch_size, lr, patience) -> dict:
        """PyTorch 训练"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            val_loss = avg_loss

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(torch.FloatTensor(X_val).to(device))
                    val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(device)).item()
                self.model.train()

            scheduler.step(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} | Train: {avg_loss:.6f} | Val: {val_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"  早停于 Epoch {epoch+1}")
                break

        # 恢复最佳模型
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        self.train_history = history
        result = {
            "epochs": len(history),
            "best_val_loss": best_val_loss,
            "final_train_loss": history[-1]["train_loss"],
            "history": history,
        }
        logger.info(f"训练完成: {result['epochs']} 轮, 最佳验证损失: {best_val_loss:.6f}")
        return result

    def _train_sklearn(self, X_train, y_train, X_val, y_val) -> dict:
        """sklearn 降级训练"""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error

        # 将 3D 滑动窗口展平为 2D
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
            random_state=42,
        )
        self.model.fit(X_flat, y_train)

        train_pred = self.model.predict(X_flat)
        train_loss = mean_squared_error(y_train, train_pred)

        val_loss = train_loss
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_pred = self.model.predict(X_val_flat)
            val_loss = mean_squared_error(y_val, val_pred)

        self.train_history = [{"epoch": 1, "train_loss": train_loss, "val_loss": val_loss}]
        return {
            "epochs": 1,
            "best_val_loss": val_loss,
            "final_train_loss": train_loss,
            "history": self.train_history,
            "model_type": "GradientBoosting",
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: shape (N, lookback, features) 或 (lookback, features)

        Returns:
            预测涨跌幅数组
        """
        if X.ndim == 2:
            X = X[np.newaxis, :]
        X = self._apply_normalize(X)

        if self.model == "sklearn_fallback":
            X_flat = X.reshape(X.shape[0], -1)
            return self.model.predict(X_flat)

        import torch
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            pred = self.model(X_t).cpu().numpy()
        return pred

    def predict_single(self, features_window: np.ndarray) -> dict:
        """
        单次预测（用于每日实战）

        Args:
            features_window: shape (lookback, features)

        Returns:
            {predicted_return, direction, confidence, signal}
        """
        pred = self.predict(features_window)
        predicted_return = float(pred[0]) if len(pred) > 0 else 0.0

        direction = "上涨" if predicted_return > 0 else "下跌"
        confidence = min(abs(predicted_return) * 10, 1.0)  # 简单置信度

        # 信号生成
        if predicted_return > 0.03:
            signal = "强烈看多"
        elif predicted_return > 0.01:
            signal = "看多"
        elif predicted_return < -0.03:
            signal = "强烈看空"
        elif predicted_return < -0.01:
            signal = "看空"
        else:
            signal = "观望"

        return {
            "predicted_return": predicted_return,
            "predicted_pct": f"{predicted_return * 100:.2f}%",
            "direction": direction,
            "confidence": confidence,
            "signal": signal,
        }

    # ======================== 保存/加载 ========================

    def save(self, name: str = "latest"):
        """保存模型"""
        _ensure_model_dir()
        model_path = os.path.join(_MODEL_DIR, f"{name}_model.pt")
        meta_path = os.path.join(_MODEL_DIR, f"{name}_meta.json")

        if self.model == "sklearn_fallback":
            import joblib
            joblib.dump(self.model, os.path.join(_MODEL_DIR, f"{name}_sklearn.joblib"))
        else:
            import torch
            torch.save(self.model.state_dict(), model_path)

        meta = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "forecast_days": self.forecast_days,
            "feature_cols": self.feature_cols,
            "train_history": self.train_history[-5:],
            "model_type": "pytorch" if self.model != "sklearn_fallback" else "sklearn",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 保存 scaler
        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, os.path.join(_MODEL_DIR, f"{name}_scaler.joblib"))

        logger.info(f"模型已保存: {name}")

    @classmethod
    def load(cls, name: str = "latest") -> "StockPredictor":
        """加载模型"""
        _ensure_model_dir()
        meta_path = os.path.join(_MODEL_DIR, f"{name}_meta.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        predictor = cls(
            input_size=meta["input_size"],
            hidden_size=meta.get("hidden_size", 128),
            num_layers=meta.get("num_layers", 2),
            dropout=meta.get("dropout", 0.2),
            forecast_days=meta.get("forecast_days", 5),
        )
        predictor.feature_cols = meta.get("feature_cols", [])
        predictor.train_history = meta.get("train_history", [])

        if meta.get("model_type") == "sklearn":
            import joblib
            sklearn_path = os.path.join(_MODEL_DIR, f"{name}_sklearn.joblib")
            if os.path.exists(sklearn_path):
                predictor.model = joblib.load(sklearn_path)
            else:
                # 兼容旧版 pickle 格式
                import pickle
                with open(os.path.join(_MODEL_DIR, f"{name}_sklearn.pkl"), "rb") as f:
                    predictor.model = pickle.load(f)
        else:
            import torch
            model_path = os.path.join(_MODEL_DIR, f"{name}_model.pt")
            predictor.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # 加载 scaler
        scaler_path = os.path.join(_MODEL_DIR, f"{name}_scaler.joblib")
        if os.path.exists(scaler_path):
            import joblib
            predictor.scaler = joblib.load(scaler_path)
        else:
            # 兼容旧版 pickle 格式
            scaler_pkl = os.path.join(_MODEL_DIR, f"{name}_scaler.pkl")
            if os.path.exists(scaler_pkl):
                import pickle
                with open(scaler_pkl, "rb") as f:
                    predictor.scaler = pickle.load(f)

        logger.info(f"模型已加载: {name}")
        return predictor

    # ======================== 辅助 ========================

    def _normalize(self, X: np.ndarray) -> tuple:
        """标准化特征"""
        from sklearn.preprocessing import StandardScaler
        original_shape = X.shape
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            X_scaled = self.scaler.transform(X_flat)

        return X_scaled.reshape(original_shape), self.scaler

    def _apply_normalize(self, X: np.ndarray) -> np.ndarray:
        """应用已有的标准化"""
        if self.scaler is None:
            return X
        original_shape = X.shape
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(original_shape)