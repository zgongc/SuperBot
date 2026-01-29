#!/usr/bin/env python3
"""
modules/simple_train/training/entry_trainer.py
SuperBot - Entry Model Trainer
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Entry model training class.

Usage:
    from modules.simple_train.training import EntryTrainer

    trainer = EntryTrainer(strategy_name="simple_rsi")
    trainer.load_data()
    trainer.prepare_features()
    trainer.train()
    trainer.evaluate()
    trainer.save()

Dependencies:
    - modules.simple_train.core (DataLoader, FeatureExtractor, Normalizer)
    - modules.simple_train.models (EntryModel, LabelGenerator)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from datetime import datetime

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
    logger = get_logger("modules.simple_train.training.entry_trainer")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.training.entry_trainer")


# =============================================================================
# ENTRY TRAINER
# =============================================================================

class EntryTrainer:
    """
    Entry model training class.

    Pipeline:
    1. Data loading (ParquetsEngine)
    2. Strategy signal generation
    3. Feature extraction
    4. Label generation (trade results)
    5. Train/Val/Test split
    6. Model training
    7. Evaluation
    """

    def __init__(
        self,
        strategy_name: str = "simple_rsi",
        config_path: str | Path | None = None,
        output_dir: str | Path | None = None
    ):
        """
        Initializes the EntryTrainer.

        Args:
            strategy_name: The name of the strategy to be trained.
            config_path: training.yaml yolu (None=default)
            output_dir: Output directory for models and logs.
        """
        self.strategy_name = strategy_name

        # Load configuration
        if config_path is None:
            config_path = SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
        self.config = self._load_config(config_path)

        # Output directory (strategy based: data/ai/checkpoints/simple_train/strategy_name/)
        if output_dir is None:
            base_dir = Path(self.config.get("checkpoints", {}).get(
                "save_path", "data/ai/checkpoints/simple_train"
            ))
            output_dir = base_dir / strategy_name  # Add strategy subfolder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df: pd.DataFrame | None = None  # Base timeframe data
        self.mtf_data: dict[str, pd.DataFrame] | None = None  # MTF data
        self.features_df: pd.DataFrame | None = None
        self.signals: pd.Series | None = None
        self.labels: pd.Series | None = None

        # MTF info
        self.is_mtf: bool = False
        self.timeframes: list[str] = []
        self.base_timeframe: str = ""

        # Split data
        self.X_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        # Components (lazy load)
        self._data_loader = None
        self._feature_extractor = None
        self._normalizer = None
        self._model = None
        self._label_generator = None
        self._strategy = None

        # Training state
        self.is_trained = False
        self.metrics: dict = {}

        logger.info(f"🎓 EntryTrainer started: {strategy_name}")

    def _load_config(self, config_path: str | Path) -> dict:
        """Loads the configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Configuration not found: {config_path}, using default")
            return self._default_config()

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        """Default config."""
        return {
            "data": {
                "source": "parquet",
                "parquet_path": "data/parquets",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "symbols": ["BTCUSDT"],
                "timeframe": "5m",
                "warmup_bars": 200
            },
            "model": {
                "lookback_window": 100,
                "labeling": {"method": "trade_result"}
            },
            "entry_model": {
                "type": "xgboost",
                "mode": "filter"
            },
            "training": {
                "split": {
                    "method": "time_based",
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15
                }
            },
            "environment": {
                "commission": 0.0004,
                "slippage": 0.0001
            }
        }

    # =========================================================================
    # LAZY LOADING COMPONENTS
    # =========================================================================

    @property
    def data_loader(self):
        """DataLoader lazy load."""
        if self._data_loader is None:
            from modules.simple_train.core import DataLoader
            self._data_loader = DataLoader(
                config_path=SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
            )
        return self._data_loader

    @property
    def feature_extractor(self):
        """FeatureExtractor lazy load."""
        if self._feature_extractor is None:
            from modules.simple_train.core import FeatureExtractor
            self._feature_extractor = FeatureExtractor(
                strategy_name=self.strategy_name,
                features_config_path=SIMPLE_TRAIN_ROOT / "configs" / "features.yaml"
            )
        return self._feature_extractor

    @property
    def normalizer(self):
        """Normalizer lazy load."""
        if self._normalizer is None:
            from modules.simple_train.core import Normalizer
            self._normalizer = Normalizer(
                config_path=SIMPLE_TRAIN_ROOT / "configs" / "features.yaml"
            )
        return self._normalizer

    @property
    def model(self):
        """EntryModel lazy load."""
        if self._model is None:
            from modules.simple_train.models import EntryModel
            self._model = EntryModel(
                config_path=SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
            )
        return self._model

    @property
    def label_generator(self):
        """LabelGenerator lazy load."""
        if self._label_generator is None:
            from modules.simple_train.models.entry_model import LabelGenerator
            labeling_config = self.config.get("model", {}).get("labeling", {})
            self._label_generator = LabelGenerator(
                method=labeling_config.get("method", "trade_result"),
                config=labeling_config
            )
        return self._label_generator

    @property
    def strategy(self):
        """Strategy lazy load."""
        if self._strategy is None:
            self._strategy = self._load_strategy()
        return self._strategy

    def _load_strategy(self):
        """Loads the strategy class."""
        import importlib.util

        strategy_path = SUPERBOT_ROOT / "components" / "strategies" / "templates" / f"{self.strategy_name}.py"

        if not strategy_path.exists():
            logger.error(f"❌ Strategy not found: {strategy_path}")
            return None

        spec = importlib.util.spec_from_file_location(
            f"strategy_{self.strategy_name}",
            strategy_path
        )
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)

        if hasattr(strategy_module, "Strategy"):
            return strategy_module.Strategy()

        logger.error(f"❌ Strategy class not found")
        return None

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_data(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> bool:
        """
        Loads data. Supports MTF ("5m,15m,30m" format).

        Args:
            symbol: Trading symbol (default: from config)
            timeframe: Timeframe or MTF string (default: from config)
            start_date: Start date
            end_date: End date

        Returns:
            bool: Is it successful?
        """
        data_config = self.config.get("data", {})

        symbol = symbol or data_config.get("symbols", ["BTCUSDT"])[0]
        timeframe = timeframe or data_config.get("timeframe", "5m")
        start_date = start_date or data_config.get("start_date", "2024-01-01")
        end_date = end_date or data_config.get("end_date", "2025-01-01")
        warmup_bars = data_config.get("warmup_bars", 200)

        # MTF check
        self.is_mtf = self.data_loader.is_mtf(timeframe)
        self.timeframes = self.data_loader.parse_timeframes(timeframe)
        self.base_timeframe = self.timeframes[0]

        if self.is_mtf:
            logger.info(f"📂 Loading MTF Data: {symbol}")
            logger.info(f"   ⏱️ Timeframes: {self.timeframes}")
        else:
            logger.info(f"📂 Loading data: {symbol} {timeframe}")

        logger.info(f"   📅 {start_date} → {end_date}")
        logger.info(f"   🔥 Warmup: {warmup_bars} bar")

        try:
            if self.is_mtf:
                # Load multi-timeframe data
                self.mtf_data = self.data_loader.load_mtf(
                    symbol=symbol,
                    timeframes=self.timeframes,
                    start_date=start_date,
                    end_date=end_date,
                    warmup_bars=warmup_bars
                )

                if not self.mtf_data:
                    logger.error("❌ MTF data could not be loaded")
                    return False

                # Use the base timeframe as the main DataFrame.
                self.df = self.mtf_data.get(self.base_timeframe)
                if self.df is None:
                    logger.error(f"❌ Base timeframe {self.base_timeframe} could not be loaded")
                    return False

                logger.info(f"   ✅ MTF loaded: {list(self.mtf_data.keys())}")
            else:
                # Load a single timeframe
                self.df = self.data_loader.load(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    warmup_bars=warmup_bars
                )

                if self.df is None or len(self.df) == 0:
                    logger.error("❌ Data could not be loaded")
                    return False

                logger.info(f"   ✅ {len(self.df)} bars loaded")

            return True

        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_signals(self) -> bool:
        """
        Creates strategy signals.

        Returns:
            bool: Is it successful?
        """
        if self.df is None:
            logger.error("❌ Please load the data first: load_data()")
            return False

        logger.info("📊 Creating strategy signals...")

        try:
            # Load strategy
            if self.strategy is None:
                return False

            # Create signals
            self.signals = self._generate_strategy_signals()

            if self.signals is None:
                return False

            # Statistics
            long_signals = (self.signals == 1).sum()
            short_signals = (self.signals == -1).sum()
            total_signals = long_signals + short_signals

            logger.info(f"   ✅ {total_signals} signal was created")
            logger.info(f"      📈 LONG: {long_signals}")
            logger.info(f"      📉 SHORT: {short_signals}")

            return True

        except Exception as e:
            logger.error(f"❌ Error creating signal: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_strategy_signals(self) -> pd.Series | None:
        """
        Generates a signal using a strategy.

        For simple_rsi:
        - RSI < oversold AND close > EMA → LONG (1)
        - RSI > overbought AND close < EMA → SHORT (-1)
        - Other -> HOLD (0)
        """
        signals = pd.Series(0, index=self.df.index)

        # Calculate indicators
        indicators = self.feature_extractor.strategy_indicators

        # Get the RSI and EMA parameters
        rsi_params = indicators.get("rsi_14", {})
        ema_params = indicators.get("ema_50", {})

        rsi_period = rsi_params.get("period", 14)
        rsi_oversold = rsi_params.get("oversold", 20)
        rsi_overbought = rsi_params.get("overbought", 75)
        ema_period = ema_params.get("period", 50)

        # Calculate indicators
        close = self.df["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # EMA
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # Signal conditions
        # LONG: RSI < oversold AND close > EMA
        long_condition = (rsi < rsi_oversold) & (close > ema)
        signals[long_condition] = 1

        # SHORT: RSI > overbought AND close < EMA
        short_condition = (rsi > rsi_overbought) & (close < ema)
        signals[short_condition] = -1

        return signals

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def prepare_features(self) -> bool:
        """
        Prepares and normalizes features.

        Returns:
            bool: Is it successful?
        """
        if self.df is None:
            logger.error("❌ Please load the data first: load_data()")
            return False

        logger.info("🔧 Feature extraction is starting...")

        try:
            # Feature extraction
            self.features_df = self.feature_extractor.extract(self.df)

            if self.features_df is None or len(self.features_df) == 0:
                logger.error("❌ Feature extraction failed")
                return False

            # Feature names
            feature_names = self.feature_extractor.get_feature_names()

            logger.info(f"   📊 {len(feature_names)} features extracted")
            logger.debug(f"   Features: {feature_names}")

            # Normalization
            logger.info("📊 Normalization is starting...")
            self.normalizer.fit(self.features_df, feature_names)
            self.features_df = self.normalizer.normalize(self.features_df, feature_names)

            logger.info("   ✅ Normalization completed")

            return True

        except Exception as e:
            logger.error(f"❌ Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # LABEL GENERATION
    # =========================================================================

    def generate_labels(
        self,
        tp_percent: float | None = None,
        sl_percent: float | None = None,
        timeout_bars: int | None = None
    ) -> bool:
        """
        Creates labels.

        Args:
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            timeout_bars: Maximum position duration

        Returns:
            bool: Is it successful?
        """
        if self.df is None or self.signals is None:
            logger.error("❌ Please load data and signals first")
            return False

        # Default values (from config or hardcoded)
        exit_config = self.config.get("exit_model", {}).get("profiles", {}).get("balanced", {})
        tp_percent = tp_percent or exit_config.get("tp_percent", 3.0)
        sl_percent = sl_percent or exit_config.get("sl_percent", 2.0)
        timeout_bars = timeout_bars or self.config.get("model", {}).get("lookback_window", 100)

        logger.info(f"🏷️ Creating label...")
        logger.info(f"   📈 TP: %{tp_percent}")
        logger.info(f"   📉 SL: %{sl_percent}")
        logger.info(f"   ⏱️ Timeout: {timeout_bars} bar")

        try:
            self.labels = self.label_generator.generate(
                df=self.df,
                signals=self.signals,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                timeout_bars=timeout_bars
            )

            # Label istatistikleri
            signal_labels = self.labels.dropna()
            if len(signal_labels) > 0:
                profitable = (signal_labels == 1).sum()
                losing = (signal_labels == 0).sum()
                win_rate = profitable / len(signal_labels) * 100

                logger.info(f"   ✅ {len(signal_labels)} label created")
                logger.info(f"      📈 Profitable: {profitable} ({win_rate:.1f}%)")
                logger.info(f"      📉 Loss: {losing}")
            else:
                logger.warning("⚠️ No labels could be created")

            return True

        except Exception as e:
            logger.error(f"❌ Error creating label: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # DATA SPLITTING
    # =========================================================================

    def split_data(self) -> bool:
        """
        Performs train/validation/test split.

        Returns:
            bool: Is it successful?
        """
        if self.features_df is None or self.labels is None:
            logger.error("❌ Prepare features and labels first")
            return False

        split_config = self.config.get("training", {}).get("split", {})
        method = split_config.get("method", "time_based")
        train_ratio = split_config.get("train_ratio", 0.7)
        val_ratio = split_config.get("val_ratio", 0.15)
        test_ratio = split_config.get("test_ratio", 0.15)

        logger.info(f"✂️ Data split: {method}")
        logger.info(f"   Train: {train_ratio*100:.0f}%")
        logger.info(f"   Val: {val_ratio*100:.0f}%")
        logger.info(f"   Test: {test_ratio*100:.0f}%")

        try:
            # Get only the places that have a signal
            signal_mask = self.labels.notna()
            X = self.features_df[signal_mask].values
            y = self.labels[signal_mask].values

            # Save feature names
            self.feature_names = self.feature_extractor.get_feature_names()

            if len(X) == 0:
                logger.error("❌ No signal detected")
                return False

            n = len(X)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            if method == "time_based":
                # Time-based split (to prevent leakage)
                self.X_train = X[:train_end]
                self.y_train = y[:train_end]
                self.X_val = X[train_end:val_end]
                self.y_val = y[train_end:val_end]
                self.X_test = X[val_end:]
                self.y_test = y[val_end:]
            else:
                # Random split
                from sklearn.model_selection import train_test_split
                X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                    X, y, test_size=test_ratio, random_state=42
                )
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio/(1-test_ratio), random_state=42
                )

            logger.info(f"   ✅ Split completed")
            logger.info(f"      Train: {len(self.X_train)} samples")
            logger.info(f"      Val: {len(self.X_val)} samples")
            logger.info(f"      Test: {len(self.X_test)} samples")

            return True

        except Exception as e:
            logger.error(f"❌ Split error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, epochs: int = 100, batch_size: int = 32) -> bool:
        """
        Trains the model.

        Args:
            epochs: The number of epochs for the LSTM.
            batch_size: The batch size for the LSTM.

        Returns:
            bool: Is it successful?
        """
        if self.X_train is None or self.y_train is None:
            logger.error("❌ Call split_data() first")
            return False

        logger.info("🎓 Education is starting...")

        try:
            # Train the model
            self.model.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=epochs,
                batch_size=batch_size
            )

            self.is_trained = True
            logger.info("✅ Training completed")

            return True

        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate(self) -> dict:
        """
        Evaluates the model's performance.

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            logger.error("❌ Model has not been trained yet")
            return {}

        logger.info("📊 Evaluation is starting...")

        try:
            # Train metrics
            train_metrics = self.model.evaluate(self.X_train, self.y_train)
            logger.info(f"   📈 Train Accuracy: {train_metrics['accuracy']:.4f}")

            # Val metrics
            val_metrics = self.model.evaluate(self.X_val, self.y_val)
            logger.info(f"   📈 Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Test metrics
            test_metrics = self.model.evaluate(self.X_test, self.y_test)
            logger.info(f"   📈 Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"   📊 Test F1: {test_metrics['f1_score']:.4f}")
            logger.info(f"   📊 Test ROC AUC: {test_metrics['roc_auc']:.4f}")

            self.metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics
            }

            return self.metrics

        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    # =========================================================================
    # SAVE/LOAD
    # =========================================================================

    def save(self, name: str | None = None) -> Path | None:
        """
        Saves the model and metadata.

        Args:
            name: Model name (default: timestamp)

        Returns:
            Path: Record path
        """
        if not self.is_trained:
            logger.error("❌ Model has not been trained yet")
            return None

        if name is None:
            name = f"entry_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_dir = self.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            model_path = model_dir / "model.pkl"
            self.model.save(model_path)

            # Save metadata
            metadata = {
                "strategy_name": self.strategy_name,
                "model_type": self.model.model_type,
                "feature_names": self.feature_names if hasattr(self, 'feature_names') else [],
                "metrics": self.metrics,
                "config": {
                    "entry_model": self.config.get("entry_model", {}),
                    "training": self.config.get("training", {})
                },
                "created_at": datetime.now().isoformat(),
                "train_samples": len(self.X_train) if self.X_train is not None else 0,
                "val_samples": len(self.X_val) if self.X_val is not None else 0,
                "test_samples": len(self.X_test) if self.X_test is not None else 0
            }

            metadata_path = model_dir / "metadata.yaml"
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"💾 Model kaydedildi: {model_dir}")

            return model_dir

        except Exception as e:
            logger.error(f"❌ Error saving: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load(self, model_dir: str | Path) -> bool:
        """
        Loads the saved model.

        Args:
            model_dir: Model directory

        Returns:
            bool: Is it successful?
        """
        model_dir = Path(model_dir)

        if not model_dir.exists():
            logger.error(f"❌ Model folder not found: {model_dir}")
            return False

        try:
            # Load the model
            model_path = model_dir / "model.pkl"
            self.model.load(model_path)
            self.is_trained = True

            # Load metadata
            metadata_path = model_dir / "metadata.yaml"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = yaml.safe_load(f)
                self.metrics = metadata.get("metrics", {})
                self.feature_names = metadata.get("feature_names", [])

            logger.info(f"📂 Model loaded: {model_dir}")
            return True

        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    def run_pipeline(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        save_model: bool = True
    ) -> dict:
        """
        Runs the complete training pipeline.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            save_model: Save the model

        Returns:
            dict: Results and metrics
        """
        logger.info("=" * 60)
        logger.info("🚀 Entry Model Training Pipeline")
        logger.info("=" * 60)

        results = {"success": False}

        # 1. Load data
        if not self.load_data(symbol, timeframe, start_date, end_date):
            return results

        # 2. Strategy signals
        if not self.generate_signals():
            return results

        # 3. Feature extraction
        if not self.prepare_features():
            return results

        # 4. Label generation
        if not self.generate_labels():
            return results

        # 5. Split
        if not self.split_data():
            return results

        # 6. Training
        if not self.train():
            return results

        # 7. Evaluation
        self.metrics = self.evaluate()

        # 8. Save
        if save_model:
            model_path = self.save()
            results["model_path"] = str(model_path) if model_path else None

        results["success"] = True
        results["metrics"] = self.metrics

        logger.info("=" * 60)
        logger.info("✅ Pipeline completed!")
        logger.info("=" * 60)

        return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EntryTrainer Test")
    print("=" * 60)

    # Test 1: Initialization
    print("\nTest 1: Starting EntryTrainer")
    try:
        trainer = EntryTrainer(strategy_name="simple_rsi")
        print(f"   ✅ Initialization successful")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
