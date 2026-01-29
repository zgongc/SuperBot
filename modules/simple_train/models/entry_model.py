#!/usr/bin/env python3
"""
modules/simple_train/models/entry_model.py
SuperBot - Entry Filter Model
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Entry model class - Filters strategy signals.

Mode:
- filter: The strategy signal arrives -> The model says "open" or "open_long"
- direction: Suggests a direction independent of the strategy (LONG/SHORT/HOLD)
- both: Both of them

Desteklenen model tipleri:
- XGBoost: Gradient boosting (fast, good for tabular data)
- LightGBM: Light gradient boosting (faster, for large datasets)
- LSTM: Sequence model (for temporal patterns)

Usage:
    from modules.simple_train.models import EntryModel

    model = EntryModel(config_path="modules/simple_train/configs/training.yaml")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

Dependencies:
    - python>=3.10
    - xgboost>=2.0.0
    - lightgbm>=4.0.0
    - torch>=2.0.0 (for LSTM)
    - scikit-learn>=1.3.0
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# PATH SETUP
# =============================================================================

SIMPLE_TRAIN_ROOT = Path(__file__).parent.parent
SUPERBOT_ROOT = SIMPLE_TRAIN_ROOT.parent.parent

if str(SUPERBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERBOT_ROOT))

# =============================================================================
# LOGGER SETUP
# =============================================================================

try:
    from core.logger_engine import get_logger
    logger = get_logger("modules.simple_train.models.entry_model")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.models.entry_model")


# =============================================================================
# ENTRY MODEL
# =============================================================================

class EntryModel:
    """
    Entry filter model class.

    Filters strategy signals:
    - Signal received -> Model: "Open" (1) or "Close" (0)

    Attributes:
        config: Model config
        model_type: "xgboost", "lightgbm", "lstm"
        mode: "filter", "direction", "both"
        model: Trained model instance
        is_fitted: Is the model trained?
    """

    SUPPORTED_TYPES = ["xgboost", "lightgbm", "lstm"]
    SUPPORTED_MODES = ["filter", "direction", "both"]

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None
    ):
        """
        Initializes the EntryModel.

        Args:
            config: Entry model config dictionary (priority)
            config_path: Path to the training.yaml file
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Default config path
            default_path = SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
            self.config = self._load_config(default_path)

        # Model parameters
        self.model_type: str = self.config.get("type", "xgboost")
        self.mode: str = self.config.get("mode", "filter")
        self.filter_config: dict = self.config.get("filter", {})
        self.direction_config: dict = self.config.get("direction", {})

        # Model instance
        self.model: Any = None
        self.is_fitted: bool = False

        # Sequence length for LSTM
        self.sequence_length: int = self.config.get("lstm", {}).get("sequence_length", 100)

        # Create model
        self._create_model()

        logger.info(f"🤖 EntryModel initialized: type={self.model_type}, mode={self.mode}")

    def _load_config(self, config_path: str | Path) -> dict:
        """Loads the configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Configuration not found: {config_path}")
            return self._default_config()

        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        return full_config.get("entry_model", self._default_config())

    def _default_config(self) -> dict:
        """Returns the default configuration."""
        return {
            "enabled": True,
            "type": "xgboost",
            "mode": "filter",
            "filter": {
                "enabled": True,
                "threshold": 0.5,
                "require_match": True
            },
            "direction": {
                "enabled": False
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8
            },
            "lstm": {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
                "sequence_length": 100
            }
        }

    def _create_model(self) -> None:
        """Creates a model instance."""
        if self.model_type == "xgboost":
            self._create_xgboost()
        elif self.model_type == "lightgbm":
            self._create_lightgbm()
        elif self.model_type == "lstm":
            self._create_lstm()
        else:
            logger.warning(f"⚠️ Unknown model type: {self.model_type}, using xgboost")
            self.model_type = "xgboost"
            self._create_xgboost()

    def _create_xgboost(self) -> None:
        """Creates an XGBoost model."""
        try:
            import xgboost as xgb

            params = self.config.get("xgboost", {})
            self.model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
            logger.info("   📊 XGBoost model created")

        except ImportError:
            logger.error("❌ XGBoost is not installed: pip install xgboost")
            raise

    def _create_lightgbm(self) -> None:
        """Creates a LightGBM model."""
        try:
            import lightgbm as lgb

            params = self.config.get("lightgbm", {})
            self.model = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                num_leaves=params.get("num_leaves", 31),
                subsample=params.get("subsample", 0.8),
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            logger.info("   📊 LightGBM model created")

        except ImportError:
            logger.error("❌ LightGBM is not installed: pip install lightgbm")
            raise

    def enable_class_imbalance(self, y: np.ndarray) -> None:
        """
        Enable class imbalance handling to balance WIN/LOSE weights.

        XGBoost: scale_pos_weight = n_negative / n_positive
        LightGBM: is_unbalance = True

        Args:
            y: Label array (0/1)
        """
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if n_pos == 0:
            logger.warning("⚠️ CIH: No positive samples found, skipping")
            return

        ratio = n_neg / n_pos

        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=ratio)
            logger.info(f"⚖️ CIH: XGBoost scale_pos_weight={ratio:.2f} (neg={n_neg}, pos={n_pos})")
        elif self.model_type == "lightgbm":
            self.model.set_params(is_unbalance=True)
            logger.info(f"⚖️ CIH: LightGBM is_unbalance=True (neg={n_neg}, pos={n_pos}, ratio={ratio:.2f})")
        else:
            logger.info(f"ℹ️ CIH: Class imbalance handling not supported for {self.model_type}")

    def _create_lstm(self) -> None:
        """Creates an LSTM model."""
        try:
            import torch
            import torch.nn as nn

            params = self.config.get("lstm", {})
            self.lstm_params = {
                "hidden_size": params.get("hidden_size", 128),
                "num_layers": params.get("num_layers", 2),
                "dropout": params.get("dropout", 0.2),
                "bidirectional": params.get("bidirectional", False)
            }

            # Model placeholder - input_size is determined during fitting
            self.model = None
            self._lstm_initialized = False

            logger.info("   📊 LSTM model placeholder created")

        except ImportError:
            logger.error("❌ PyTorch is not installed: pip install torch")
            raise

    def _build_lstm_model(self, input_size: int) -> None:
        """Creates the LSTM model with the actual input size."""
        import torch
        import torch.nn as nn

        class LSTMClassifier(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                         dropout: float, bidirectional: bool):
                super().__init__()

                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                self.num_directions = 2 if bidirectional else 1

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True
                )

                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * self.num_directions, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                lstm_out, _ = self.lstm(x)
                # Get the last timestep
                last_output = lstm_out[:, -1, :]
                return self.fc(last_output)

        self.model = LSTMClassifier(
            input_size=input_size,
            **self.lstm_params
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self._lstm_initialized = True
        logger.info(f"   📊 LSTM model created: input_size={input_size}, device={self.device}")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> "EntryModel":
        """
        Trains the model.

        Args:
            X: Feature matrix (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Labels (0: off, 1: on)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs for LSTM
            batch_size: Batch size for LSTM
            early_stopping_patience: Early stopping patience

        Returns:
            self: For the fluent interface.
        """
        # DataFrame -> numpy (feature isimlerini sakla)
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]
        if isinstance(y, pd.Series):
            y = y.values

        # Dtype check and conversion (multi-output fix)
        if X.dtype == object or X.dtype == 'O':
            logger.warning(f"⚠️ X dtype={X.dtype}, being converted to float64")
            try:
                X = X.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Could not convert X to float: {e}")
                logger.error(f"   First line: {X[0]}")
                raise ValueError(f"There are non-numeric values in the X array: {e}")

        # NaN check
        if np.any(np.isnan(X)):
            logger.warning("⚠️ There are NaN values in X, replacing them with 0")
            X = np.nan_to_num(X, nan=0.0)

        if np.any(np.isnan(y)):
            logger.warning("⚠️ There are NaN values in y, replacing them with 0")
            y = np.nan_to_num(y, nan=0.0)

        logger.info(f"🎓 Training starting: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"   📊 Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

        if self.model_type in ["xgboost", "lightgbm"]:
            self._fit_tree_model(X, y, X_val, y_val)
        elif self.model_type == "lstm":
            self._fit_lstm(X, y, X_val, y_val, epochs, batch_size, early_stopping_patience)

        self.is_fitted = True
        logger.info("✅ Training completed")

        return self

    def _fit_tree_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None
    ) -> None:
        """XGBoost/LightGBM training."""
        # Make sure it's 2D
        if X.ndim == 3:
            # (n_samples, seq_len, n_features) -> (n_samples, seq_len * n_features)
            X = X.reshape(X.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)

        if X_val is not None and y_val is not None:
            # LightGBM does not support the verbose parameter, it uses callbacks.
            if self.model_type == "lightgbm":
                import lightgbm as lgb
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)]  # 0 = sessiz mod
                )
            else:
                # XGBoost
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
        else:
            self.model.fit(X, y)

    def _fit_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        epochs: int,
        batch_size: int,
        patience: int
    ) -> None:
        """LSTM training."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Make sure it's 3D
        if X.ndim == 2:
            # (n_samples, n_features) -> (n_samples, 1, n_features)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # Create model
        input_size = X.shape[2]
        if not self._lstm_initialized:
            self._build_lstm_model(input_size)

        # Tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        # DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Validation
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                self.model.train()

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"   📈 Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"   📈 Epoch {epoch+1}: train_loss={avg_loss:.4f}")

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Makes a prediction.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Binary predictions (0: closed, 1: open)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet! Call fit() first.")

        # DataFrame -> numpy
        if isinstance(X, pd.DataFrame):
            X = X.values

        # NaN check
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)

        proba = self.predict_proba(X)
        threshold = self.filter_config.get("threshold", 0.5)

        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Performs probability estimation.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Probability scores (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet! Call fit() first.")

        # DataFrame -> numpy (feature isimlerini sakla)
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values

        # NaN check
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)

        if self.model_type in ["xgboost", "lightgbm"]:
            return self._predict_proba_tree(X, feature_names)
        elif self.model_type == "lstm":
            return self._predict_proba_lstm(X)

        return np.zeros(len(X))

    def _predict_proba_tree(self, X: np.ndarray, feature_names: list | None = None) -> np.ndarray:
        """XGBoost/LightGBM probability estimation."""
        # Make sure it's 2D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        # Add feature names (to prevent warnings)
        # First, use the names received as parameters, otherwise use the ones saved from training.
        names = feature_names or getattr(self, '_feature_names', None)
        if names and len(names) == X.shape[1]:
            X = pd.DataFrame(X, columns=names)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Positive class probability

    def _predict_proba_lstm(self, X: np.ndarray) -> np.ndarray:
        """LSTM probability estimation."""
        import torch

        # Make sure it's 3D
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            proba = self.model(X_tensor).cpu().numpy().flatten()

        return proba

    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series
    ) -> dict[str, float]:
        """
        Evaluates the model's performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        return metrics

    def save(self, path: str | Path) -> None:
        """
        Model'i kaydeder.

        Args:
            path: Record path
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == "lstm":
            import torch
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "lstm_params": self.lstm_params,
                "is_fitted": self.is_fitted
            }, path)
        else:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "config": self.config,
                    "is_fitted": self.is_fitted
                }, f)

        logger.info(f"💾 Model kaydedildi: {path}")

    def load(self, path: str | Path) -> "EntryModel":
        """
        Loads the model.

        Args:
            path: Path to the model file

        Returns:
            self: For the fluent interface.
        """
        import pickle

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        if self.model_type == "lstm":
            import torch
            checkpoint = torch.load(path, map_location=self.device)
            self.config = checkpoint["config"]
            self.lstm_params = checkpoint["lstm_params"]
            self.is_fitted = checkpoint["is_fitted"]

            # Recreate the model and load the weights
            # Input size cannot be retrieved from the checkpoint, it will cause an error in the first prediction.
            # Document this limitation
            logger.warning("⚠️ Before loading the LSTM model, the fit() method must be called.")

        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.config = data["config"]
                self.is_fitted = data["is_fitted"]

        logger.info(f"📂 Model loaded: {path}")
        return self

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Returns feature importance (only for tree models).

        Returns:
            dict: Feature name -> importance score
        """
        if self.model_type not in ["xgboost", "lightgbm"]:
            logger.warning("⚠️ Feature importance is only supported for XGBoost/LightGBM")
            return None

        if not self.is_fitted:
            logger.warning("⚠️ Model has not been trained yet")
            return None

        importances = self.model.feature_importances_

        # Feature names (if any)
        if hasattr(self, "feature_names_") and self.feature_names_ is not None:
            return dict(zip(self.feature_names_, importances))

        # Index-based
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


# =============================================================================
# LABEL GENERATOR
# =============================================================================

class LabelGenerator:
    """
    Creates a label from trade results.

    Methods:
    - trade_result: Label based on TP/SL/Timeout result (1=profitable, 0=loss)
    - price_change: Based on price change after N bars
    - bucketed: Based on profit/loss buckets (3+ classes)
    """

    def __init__(
        self,
        method: str = "trade_result",
        config: dict | None = None
    ):
        """
        Initializes the LabelGenerator.

        Args:
            method: Label creation method
            config: Method-specific configuration
        """
        self.method = method
        self.config = config or {}

        logger.info(f"🏷️ LabelGenerator started: method={method}")

    def generate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        tp_percent: float = 3.0,
        sl_percent: float = 2.0,
        timeout_bars: int = 100,
        strategy = None  # Strategy instance for exit logic
    ) -> pd.Series:
        """
        Creates labels.

        Args:
            df: OHLCV DataFrame
            signals: Trade signals (1=LONG, -1=SHORT, 0=HOLD)
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            timeout_bars: Maximum position duration (bars)
            strategy: Strategy instance (optional, for BE/PE/TS)

        Returns:
            pd.Series: Labels (1=profitable trade, 0=lossful trade)
        """
        if self.method == "trade_result":
            return self._generate_trade_result(df, signals, tp_percent, sl_percent, timeout_bars, strategy)
        elif self.method == "price_change":
            return self._generate_price_change(df, signals)
        elif self.method == "bucketed":
            return self._generate_bucketed(df, signals, tp_percent, sl_percent, timeout_bars, strategy)
        else:
            logger.warning(f"⚠️ Unknown method: {self.method}, trade_result is being used")
            return self._generate_trade_result(df, signals, tp_percent, sl_percent, timeout_bars, strategy)

    def _generate_trade_result(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        tp_percent: float,
        sl_percent: float,
        timeout_bars: int,
        strategy = None
    ) -> pd.Series:
        """
        Creates a label based on the trade result.
        
        Uses the SAME logic as the backtest engine:
        - BreakEven
        - PartialExit
        - TrailingStop
        - TP/SL
        
        For each signal:
        1. Entry price = close[signal_bar]
        2. Exit logic uygula (BE/PE/TS/TP/SL)
        3. Final PnL > 0 -> Label = 1 (profitable)
        4. Final PnL <= 0 -> Label = 0 (loss-making)

        Returns:
            pd.Series: Labels
        """
        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = np.nan

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        signal_indices = signals[signals != 0].index

        for idx in signal_indices:
            pos = df.index.get_loc(idx)
            if pos >= len(df) - 1:
                continue

            signal_type = signals.loc[idx]  # 1=LONG, -1=SHORT
            entry_price = close[pos]

            if signal_type == 1:  # LONG
                tp_price = entry_price * (1 + tp_percent / 100)
                sl_price = entry_price * (1 - sl_percent / 100)
            else:  # SHORT
                tp_price = entry_price * (1 - tp_percent / 100)
                sl_price = entry_price * (1 + sl_percent / 100)

            # Simulate the trade result (with backtest logic)
            pnl = self._simulate_trade_with_backtest_logic(
                high[pos+1:min(pos+1+timeout_bars, len(df))],
                low[pos+1:min(pos+1+timeout_bars, len(df))],
                close[pos+1:min(pos+1+timeout_bars, len(df))],
                signal_type, entry_price, tp_price, sl_price, strategy
            )
            
            # PnL > 0 -> Profitable (1), PnL <= 0 -> Loss-making (0)
            labels.iloc[pos] = 1 if pnl > 0 else 0

        return labels

    def _simulate_trade_with_backtest_logic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        signal_type: int,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        strategy = None
    ) -> float:
        """
        Performs a trading simulation (using the BacktestEngine logic).
        
        Features:
        - BreakEven: Moves the SL to the BE level at a specific profit level.
        - PartialExit: Gradual exit (e.g., 2% at 40%, 4% at 40%, 10% at 20%).
        - TrailingStop: Profit protection (trailing).
        - TP/SL: Standard take profit and stop loss.
        
        Returns:
            float: Final PnL (percentage) - positive = profit, negative = loss
        """
        if len(high) == 0:
            return -100.0  # Timeout without data
        
        # Strategy exit config (if it exists)
        if strategy is not None and hasattr(strategy, 'exit_strategy'):
            exit_cfg = strategy.exit_strategy
            be_enabled = exit_cfg.break_even_enabled
            be_trigger = exit_cfg.break_even_trigger_profit_percent
            be_offset = exit_cfg.break_even_offset
            
            pe_enabled = exit_cfg.partial_exit_enabled
            pe_levels = exit_cfg.partial_exit_levels  # [3, 4, 10]
            pe_sizes = exit_cfg.partial_exit_sizes    # [0.40, 0.40, 0.20]
            
            ts_enabled = exit_cfg.trailing_stop_enabled
            ts_activation = exit_cfg.trailing_activation_profit_percent
            ts_callback = exit_cfg.trailing_callback_percent
        else:
            # Default: Only TP/SL (no backtest logic)
            be_enabled = False
            pe_enabled = False
            ts_enabled = False
        
        # Position tracking
        position_size = 1.0  # Full size initial value
        current_sl = sl_price
        trailing_active = False
        highest_profit = 0.0  # For trailing
        
        # Partial exit tracking
        pe_executed = [False] * len(pe_levels) if pe_enabled else []
        
        # Bar-by-bar simulation
        for i in range(len(high)):
            current_high = high[i]
            current_low = low[i]
            current_close = close[i]
            
            # Current profit calculation
            if signal_type == 1:  # LONG
                profit_pct = (current_close - entry_price) / entry_price * 100
                unrealized_high_pct = (current_high - entry_price) / entry_price * 100
                unrealized_low_pct = (current_low - entry_price) / entry_price * 100
            else:  # SHORT
                profit_pct = (entry_price - current_close) / entry_price * 100
                unrealized_high_pct = (entry_price - current_low) / entry_price * 100
                unrealized_low_pct = (entry_price - current_high) / entry_price * 100
            
            # Track highest profit for trailing
            if profit_pct > highest_profit:
                highest_profit = profit_pct
            
            # ================================================================
            # 1. CHECK STOP LOSS (check stop loss first - most critical)
            # ================================================================
            if signal_type == 1:  # LONG
                if current_low <= current_sl:
                    # SL hit - close the remaining position
                    sl_pnl = (current_sl - entry_price) / entry_price * 100
                    return sl_pnl * position_size
            else:  # SHORT
                if current_high >= current_sl:
                    # SL hit
                    sl_pnl = (entry_price - current_sl) / entry_price * 100
                    return sl_pnl * position_size
            
            # ================================================================
            # 2. CHECK TAKE PROFIT
            # ================================================================
            if signal_type == 1:  # LONG
                if current_high >= tp_price:
                    # TP hit - close the remaining position at the TP.
                    tp_pnl = (tp_price - entry_price) / entry_price * 100
                    return tp_pnl * position_size
            else:  # SHORT
                if current_low <= tp_price:
                    # TP hit
                    tp_pnl = (entry_price - tp_price) / entry_price * 100
                    return tp_pnl * position_size
            
            # ================================================================
            # 3. BREAK-EVEN (Move the stop loss to the entry price)
            # ================================================================
            if be_enabled and profit_pct >= be_trigger:
                if signal_type == 1:
                    new_be_sl = entry_price * (1 + be_offset / 100)
                    if new_be_sl > current_sl:
                        current_sl = new_be_sl
                else:
                    new_be_sl = entry_price * (1 - be_offset / 100)
                    if new_be_sl < current_sl:
                        current_sl = new_be_sl
            
            # ================================================================
            # 4. PARTIAL EXIT (gradual exit)
            # ================================================================
            if pe_enabled:
                for idx, (level, size) in enumerate(zip(pe_levels, pe_sizes)):
                    if pe_executed[idx]:
                        continue
                    
                    # Did we reach the level?
                    if profit_pct >= level:
                        # Execute the partial exit for this level
                        pe_executed[idx] = True
                        position_size -= size
                        
                        # Is the position fully closed?
                        if position_size <= 0.01:
                            # All positions closed - returns the weighted average PnL.
                            return profit_pct
            
            # ================================================================
            # 5. TRAILING STOP
            # ================================================================
            if ts_enabled:
                # Trailing aktivasyonu
                if not trailing_active and profit_pct >= ts_activation:
                    trailing_active = True
                
                # Trailing logic
                if trailing_active:
                    # Has the data been retracted?
                    pullback = highest_profit - profit_pct
                    
                    if pullback >= ts_callback:
                        # Trailing stop hit - close at the current price
                        return profit_pct * position_size
        
        # ================================================================
        # 6. TIMEOUT - position is still open
        # ================================================================
        final_price = close[-1]
        if signal_type == 1:
            timeout_pnl = (final_price - entry_price) / entry_price * 100
        else:
            timeout_pnl = (entry_price - final_price) / entry_price * 100
        
        return timeout_pnl * position_size

    def _generate_price_change(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        Label based on the price change after N bars.

        Config:
            horizon: How many bars to look ahead (default: 20)
            threshold: Minimum change percentage for Label=1 (default: 0.5)
        """
        horizon = self.config.get("horizon", 20)
        threshold = self.config.get("threshold", 0.5)

        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = np.nan

        close = df["close"].values
        signal_indices = signals[signals != 0].index

        for idx in signal_indices:
            pos = df.index.get_loc(idx)
            future_pos = min(pos + horizon, len(df) - 1)

            if future_pos <= pos:
                continue

            signal_type = signals.loc[idx]
            entry_price = close[pos]
            future_price = close[future_pos]

            pct_change = (future_price - entry_price) / entry_price * 100

            if signal_type == 1:  # LONG
                labels.iloc[pos] = 1 if pct_change > threshold else 0
            else:  # SHORT
                labels.iloc[pos] = 1 if pct_change < -threshold else 0

        return labels

    def _generate_bucketed(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        tp_percent: float,
        sl_percent: float,
        timeout_bars: int,
        strategy = None
    ) -> pd.Series:
        """
        Label (multi-class) based on profit/loss buckets.

        Labels:
            0: Large loss (SL hit)
            1: Small loss (timeout, loss)
            2: Small profit (timeout, profit)
            3: Large profit (TP hit)
        """
        # For now, use a simple binary format.
        logger.warning("⚠️ Bucketed labeling is not yet implemented, using trade_result")
        return self._generate_trade_result(df, signals, tp_percent, sl_percent, timeout_bars, strategy)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EntryModel Test")
    print("=" * 60)

    # Test 1: XGBoost initialization
    print("\nTest 1: XGBoost EntryModel initialization")
    try:
        model = EntryModel()
        print(f"   ✅ Initialization successful")
        print(f"   📊 Model type: {model.model_type}")
        print(f"   📊 Mode: {model.mode}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Training with dummy data
    print("\nTest 2: XGBoost training")
    try:
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Dummy features
        X = np.random.randn(n_samples, n_features)
        # Dummy labels (basit pattern)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Train/val split
        split = int(0.8 * n_samples)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Training
        model.fit(X_train, y_train, X_val, y_val)
        print(f"   ✅ Training successful")

        # Evaluation
        metrics = model.evaluate(X_val, y_val)
        print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📊 F1 Score: {metrics['f1_score']:.4f}")
        print(f"   📊 ROC AUC: {metrics['roc_auc']:.4f}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: LabelGenerator
    print("\nTest 3: LabelGenerator")
    try:
        # Dummy OHLCV
        n = 200
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(105, 115, n),
            "low": np.random.uniform(95, 105, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 5000, n)
        })

        # Ensure high > low
        df["high"] = df[["open", "close"]].max(axis=1) + np.random.uniform(0, 5, n)
        df["low"] = df[["open", "close"]].min(axis=1) - np.random.uniform(0, 5, n)

        # Dummy signals
        signals = pd.Series(0, index=df.index)
        signals.iloc[10] = 1   # LONG
        signals.iloc[50] = -1  # SHORT
        signals.iloc[100] = 1  # LONG
        signals.iloc[150] = -1 # SHORT

        # Label generator
        label_gen = LabelGenerator(method="trade_result")
        labels = label_gen.generate(df, signals, tp_percent=3.0, sl_percent=2.0, timeout_bars=50)

        # Count labels
        signal_labels = labels.dropna()
        print(f"   ✅ Label creation successful")
        print(f"   📊 Total signals: {len(signal_labels)}")
        print(f"   📊 Profitable: {(signal_labels == 1).sum()}")
        print(f"   📊 Harmful: {(signal_labels == 0).sum()}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: LSTM (optional - requires torch)
    print("\nTest 4: LSTM EntryModel")
    try:
        lstm_config = {
            "type": "lstm",
            "mode": "filter",
            "filter": {"threshold": 0.5},
            "lstm": {
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "sequence_length": 10
            }
        }

        lstm_model = EntryModel(config=lstm_config)
        print(f"   ✅ LSTM model created")

        # 3D data (batch, seq, features)
        X_seq = np.random.randn(100, 10, 5)
        y_seq = (X_seq[:, -1, 0] > 0).astype(int)

        # Training
        lstm_model.fit(X_seq[:80], y_seq[:80], X_seq[80:], y_seq[80:], epochs=20, batch_size=16)
        print(f"   ✅ LSTM training successful")

        # Evaluation
        metrics = lstm_model.evaluate(X_seq[80:], y_seq[80:])
        print(f"   📊 LSTM Accuracy: {metrics['accuracy']:.4f}")

    except ImportError:
        print(f"   ⚠️ PyTorch is not installed, LSTM test skipped")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
