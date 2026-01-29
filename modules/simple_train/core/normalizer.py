#!/usr/bin/env python3
"""
modules/simple_train/core/normalizer.py
SuperBot - Feature Normalizer for AI Training
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Feature normalization class - Supports rolling window.

Features:
- MinMax normalization (sabit min/max)
- Z-Score normalization (global)
- Rolling Z-Score (adaptif)
- Rolling Percentile (rank-based)
- Log Z-Score (for skewed data)

Usage:
    from modules.simple_train.core import Normalizer

    normalizer = Normalizer(config_path="modules/simple_train/configs/features.yaml")
    normalized_df = normalizer.normalize(df, feature_names)

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
    - pyyaml>=6.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    logger = get_logger("modules.simple_train.core.normalizer")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.core.normalizer")


# =============================================================================
# NORMALIZER
# =============================================================================

class Normalizer:
    """
    Feature normalization class.

    Desteklenen metodlar:
    - minmax: Min-Max scaling (0-1 or custom range)
    - zscore: Global Z-Score normalization
    - rolling_zscore: Rolling window Z-Score
    - rolling_percentile: Rolling percentile rank (0-1)
    - log_zscore: Log transform + Z-Score (for skewed data)
    """

    SUPPORTED_METHODS = [
        "minmax",
        "zscore",
        "rolling_zscore",
        "rolling_percentile",
        "log_zscore"
    ]

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None
    ):
        """
        Initializes the normalizer.

        Args:
            config: Normalizer configuration dictionary (priority)
            config_path: Path to the features.yaml file
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Default config path
            default_path = SIMPLE_TRAIN_ROOT / "configs" / "features.yaml"
            self.config = self._load_config(default_path)

        # Normalizer parameters
        self.normalizers_config = self.config.get("normalizers", {})
        self.default_config = self.normalizers_config.get("default", {
            "method": "rolling_zscore",
            "window": 100
        })

        # Statistics saved during the fitting process
        self._stats: dict[str, dict] = {}

        logger.info("📊 Normalizer started")

    def _load_config(self, config_path: str | Path) -> dict:
        """Loads the configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Configuration not found: {config_path}")
            return {"normalizers": {}}

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_config(self, feature_name: str) -> dict:
        """
        Returns the normalizer config for the feature.

        Args:
            feature_name: Feature name

        Returns:
            dict: Normalizer config (method, window, min, max, etc.)
        """
        if feature_name in self.normalizers_config:
            return self.normalizers_config[feature_name]
        return self.default_config.copy()

    def fit(self, df: pd.DataFrame, feature_names: list[str] | None = None) -> "Normalizer":
        """
        Calculates statistics (for training data).

        Args:
            df: Training DataFrame
            feature_names: Features to be normalized (None=all)

        Returns:
            self: For the fluent interface.
        """
        if feature_names is None:
            feature_names = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume", "timestamp"]]

        for fname in feature_names:
            if fname not in df.columns:
                continue

            config = self.get_config(fname)
            method = config.get("method", "rolling_zscore")

            # Global statistics (for z-score)
            if method in ["zscore", "minmax"]:
                self._stats[fname] = {
                    "mean": df[fname].mean(),
                    "std": df[fname].std(),
                    "min": df[fname].min(),
                    "max": df[fname].max()
                }

            # Log transform istatistikleri
            elif method == "log_zscore":
                # Log transform for positive values
                positive_vals = df[fname].clip(lower=1e-10)
                log_vals = np.log1p(positive_vals)
                self._stats[fname] = {
                    "log_mean": log_vals.mean(),
                    "log_std": log_vals.std()
                }

        logger.info(f"📊 Normalizer fit completed: {len(self._stats)} feature")
        return self

    def normalize(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Normalizes the features.

        Args:
            df: DataFrame
            feature_names: Features to be normalized (None=all)
            inplace: If True, modifies the original DataFrame

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        if not inplace:
            df = df.copy()

        if feature_names is None:
            feature_names = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume", "timestamp"]]

        for fname in feature_names:
            if fname not in df.columns:
                continue

            try:
                df[fname] = self._normalize_feature(df, fname)
            except Exception as e:
                logger.warning(f"⚠️ {fname} normalize edilemedi: {e}")

        return df

    def _normalize_feature(self, df: pd.DataFrame, feature_name: str) -> pd.Series:
        """
        Normalizes a single feature.

        Args:
            df: DataFrame
            feature_name: Feature name

        Returns:
            pd.Series: Normalized values
        """
        config = self.get_config(feature_name)
        method = config.get("method", "rolling_zscore")
        series = df[feature_name]

        if method == "minmax":
            return self._minmax_normalize(series, config)

        elif method == "zscore":
            return self._zscore_normalize(series, feature_name)

        elif method == "rolling_zscore":
            window = config.get("window", 100)
            return self._rolling_zscore_normalize(series, window)

        elif method == "rolling_percentile":
            window = config.get("window", 200)
            return self._rolling_percentile_normalize(series, window)

        elif method == "log_zscore":
            return self._log_zscore_normalize(series, feature_name)

        else:
            logger.warning(f"⚠️ Unknown method: {method}, using rolling_zscore")
            return self._rolling_zscore_normalize(series, 100)

    def _minmax_normalize(self, series: pd.Series, config: dict) -> pd.Series:
        """
        Min-Max normalization.

        Args:
            series: Feature values
            config: config containing min, max values

        Returns:
            pd.Series: Normalized values between 0 and 1.
        """
        min_val = config.get("min", series.min())
        max_val = config.get("max", series.max())

        if max_val == min_val:
            return pd.Series(0.5, index=series.index)

        normalized = (series - min_val) / (max_val - min_val)
        return normalized.clip(0, 1)

    def _zscore_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """
        Global Z-Score normalization.

        Args:
            series: Feature values
            feature_name: Feature name to retrieve statistics for

        Returns:
            pd.Series: Z-score normalized values
        """
        if feature_name in self._stats:
            mean = self._stats[feature_name]["mean"]
            std = self._stats[feature_name]["std"]
        else:
            mean = series.mean()
            std = series.std()

        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)

        return (series - mean) / std

    def _rolling_zscore_normalize(self, series: pd.Series, window: int) -> pd.Series:
        """
        Rolling Z-Score normalization.

        Args:
            series: Feature values
            window: Rolling window size

        Returns:
            pd.Series: Rolling Z-Score normalized values
        """
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()

        # Returns 0 when Std is 0
        rolling_std = rolling_std.replace(0, np.nan)

        normalized = (series - rolling_mean) / rolling_std
        return normalized.fillna(0)

    def _rolling_percentile_normalize(self, series: pd.Series, window: int) -> pd.Series:
        """
        Rolling percentile rank normalization.

        Args:
            series: Feature values
            window: Rolling window size

        Returns:
            pd.Series: Percentile rank values between 0 and 1.
        """
        def percentile_rank(x):
            if len(x) < 2:
                return 0.5
            current = x.iloc[-1]
            return (x < current).sum() / (len(x) - 1)

        return series.rolling(window=window, min_periods=2).apply(percentile_rank, raw=False).fillna(0.5)

    def _log_zscore_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """
        Log transform + Z-Score normalization.
        Suitable for skewed data (such as volume).

        Args:
            series: Feature values
            feature_name: Feature name to retrieve statistics for

        Returns:
            pd.Series: Log + Z-Score normalized values
        """
        # Log transform (safe for negative values)
        log_series = np.log1p(series.clip(lower=0))

        if feature_name in self._stats:
            mean = self._stats[feature_name].get("log_mean", log_series.mean())
            std = self._stats[feature_name].get("log_std", log_series.std())
        else:
            mean = log_series.mean()
            std = log_series.std()

        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)

        return (log_series - mean) / std

    def inverse_normalize(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Converts normalized values back to the original scale.
        (Only works for minmax and zscore)

        Args:
            df: Normalized DataFrame
            feature_names: Features to be inverted

        Returns:
            pd.DataFrame: Values in the original scale
        """
        result = df.copy()

        if feature_names is None:
            feature_names = list(self._stats.keys())

        for fname in feature_names:
            if fname not in df.columns or fname not in self._stats:
                continue

            config = self.get_config(fname)
            method = config.get("method", "rolling_zscore")
            stats = self._stats[fname]

            if method == "minmax":
                min_val = config.get("min", stats.get("min", 0))
                max_val = config.get("max", stats.get("max", 1))
                result[fname] = df[fname] * (max_val - min_val) + min_val

            elif method == "zscore":
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)
                result[fname] = df[fname] * std + mean

        return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Normalizer Test")
    print("=" * 60)

    # Test 1: Initialization
    print("\nTest 1: Normalizer initialization")
    try:
        normalizer = Normalizer()
        print(f"   ✅ Initialization successful")
        print(f"   📊 Default config: {normalizer.default_config}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Normalization
    print("\nTest 2: Feature normalization")
    try:
        # Dummy data
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            "rsi_14": np.random.uniform(0, 100, n),
            "ema_distance": np.random.randn(n) * 0.02,
            "volume_change": np.random.exponential(1, n),  # Skewed data
            "price_momentum": np.random.randn(n) * 0.01,
            "atr_14": np.random.uniform(0.5, 2.0, n)
        })

        print(f"   Orijinal istatistikler:")
        print(f"   - rsi_14: min={df['rsi_14'].min():.2f}, max={df['rsi_14'].max():.2f}")
        print(f"   - volume_change: min={df['volume_change'].min():.2f}, max={df['volume_change'].max():.2f}")

        # Fit and normalize
        normalizer.fit(df)
        normalized_df = normalizer.normalize(df)

        print(f"\n   Normalized statistics:")
        for col in normalized_df.columns:
            print(f"   - {col}: min={normalized_df[col].min():.2f}, max={normalized_df[col].max():.2f}, mean={normalized_df[col].mean():.2f}")

        print(f"\n   ✅ Normalization successful")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Config-based normalization
    print("\nTest 3: Config-based normalization")
    try:
        # Check config for each feature
        for fname in ["rsi_14", "ema_distance", "volume_change", "unknown"]:
            config = normalizer.get_config(fname)
            print(f"   - {fname}: {config.get('method', 'N/A')}")

        print(f"   ✅ Normalization based on configuration was successful")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 4: Rolling normalization
    print("\nTest 4: Rolling normalization")
    try:
        # For rolling zscore, window=50
        rolling_config = {"normalizers": {"test_feature": {"method": "rolling_zscore", "window": 50}}}
        rolling_normalizer = Normalizer(config=rolling_config)

        test_df = pd.DataFrame({"test_feature": np.cumsum(np.random.randn(100))})
        normalized = rolling_normalizer.normalize(test_df)

        print(f"   - Input range: {test_df['test_feature'].min():.2f} to {test_df['test_feature'].max():.2f}")
        print(f"   - Output range: {normalized['test_feature'].min():.2f} to {normalized['test_feature'].max():.2f}")
        print(f"   ✅ Rolling normalization successful")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
