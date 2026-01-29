#!/usr/bin/env python3
"""
modules/simple_train/core/feature_extractor.py
SuperBot - Feature Extractor for AI Training
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

Extracts strategy indicators and extra features from the config.

Features:
- Automatically reads indicators from the strategy file.
- Config-driven ekstra feature'lar
- Current indicator_manager usage
- Derived features (momentum, change, distance)

Usage:
    from modules.simple_train.core import FeatureExtractor

    extractor = FeatureExtractor(
        strategy_name="simple_rsi_ai",
        features_config_path="modules/simple_train/configs/features.yaml"
    )
    features_df = extractor.extract(df)

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
    logger = get_logger("modules.simple_train.core.feature_extractor")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.core.feature_extractor")


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """
    A class for extracting features from strategy and config.

    Feature sources:
    1. Strategy indicators (required) - strategy.technical_parameters.indicators
    2. Config'den ekstra feature'lar - features.yaml
    3. Derived features (momentum, change, distance calculations)
    """

    def __init__(
        self,
        strategy_name: str = "simple_rsi",
        features_config_path: str | Path | None = None
    ):
        """
        Initializes the FeatureExtractor.

        Args:
            strategy_name: Strategy name (from the templates folder)
            features_config_path: Path to the features.yaml file
        """
        self.strategy_name = strategy_name
        self.features_config_path = features_config_path or (
            SIMPLE_TRAIN_ROOT / "configs" / "features.yaml"
        )

        # Load configuration
        self.config = self._load_config()

        # Get strategy indicators
        self.strategy_indicators = self._get_strategy_indicators()

        logger.info(f"📊 FeatureExtractor started: {strategy_name}")
        logger.info(f"   Strategy indicators: {list(self.strategy_indicators.keys())}")

    def _load_config(self) -> dict:
        """Loads the features.yaml configuration file."""
        config_path = Path(self.features_config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Configuration not found: {config_path}, using default")
            return self._get_default_config()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.debug(f"🔍 Configuration loaded: {config_path}")
        return config

    def _get_default_config(self) -> dict:
        """Returns the default feature configuration."""
        return {
            "strategy_indicators": "auto",
            "features": {},
            "normalizers": {
                "default": {"method": "rolling_zscore", "window": 100}
            }
        }

    def _get_strategy_indicators(self) -> dict[str, dict]:
        """
        Reads indicator information from the strategy file.

        Imports the Strategy class and reads technical_parameters.indicators.

        Returns:
            dict: Indicators in the format of {indicator_name: {params}}.
        """
        indicators = {}

        try:
            # Dynamically import the strategy module
            import importlib.util

            strategy_path = SUPERBOT_ROOT / "components" / "strategies" / "templates" / f"{self.strategy_name}.py"

            if not strategy_path.exists():
                logger.warning(f"⚠️ Strategy file not found: {strategy_path}")
                return {}

            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"strategy_{self.strategy_name}",
                strategy_path
            )
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)

            # Get the Strategy class
            if hasattr(strategy_module, "Strategy"):
                strategy_instance = strategy_module.Strategy()

                # Read technical_parameters.indicators
                if hasattr(strategy_instance, "technical_parameters"):
                    raw_indicators = strategy_instance.technical_parameters.indicators

                    # Extract type and period for each indicator
                    for ind_name, ind_params in raw_indicators.items():
                        # ema_50 -> type: ema, period: 50
                        parts = ind_name.rsplit("_", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            ind_type = parts[0]
                            period = int(parts[1])
                        else:
                            ind_type = ind_name
                            period = ind_params.get("period", 14)

                        indicators[ind_name] = {
                            "type": ind_type,
                            "period": period,
                            **ind_params  # Add the original parameters as well
                        }

                    logger.debug(f"🔍 Strategy indicators loaded: {list(indicators.keys())}")
                else:
                    logger.warning(f"⚠️ Technical parameters not found in the strategy")
            else:
                logger.warning(f"⚠️ Strategy class not found in the strategy.")

        except Exception as e:
            logger.error(f"❌ Error loading strategy: {e}")
            import traceback
            traceback.print_exc()

        return indicators



    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all features from the DataFrame.

        Args:
            df: OHLCV DataFrame (open, high, low, close, volume)

        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        if df.empty:
            logger.warning("⚠️ Empty DataFrame, feature extraction is skipped")
            return df

        result = df.copy()

        # 1. Calculate strategy indicators
        result = self._add_strategy_indicators(result)

        # 2. Calculate extra features from the config.
        result = self._add_extra_features(result)

        # 3. Derived features
        result = self._add_derived_features(result)

        logger.info(f"✅ Feature extraction completed: {len(result.columns)} columns")
        return result


    def _add_strategy_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates strategy indicators."""
        from components.indicators import get_indicator_class

        # Prepare calc data once (timestamp column needed)
        calc_df = self._prepare_data_for_indicator(df)

        for ind_name, params in self.strategy_indicators.items():
            if ind_name in df.columns:
                continue

            try:
                ind_type = params.get("type", "")
                period = params.get("period", 14)
                
                # Resolve alias (config name -> registry name)
                ind_type = self._resolve_indicator_type(ind_type)
                
                # Basic indicators (pandas/numpy optimized)
                if ind_type == "ema":
                    df[ind_name] = df["close"].ewm(span=period, adjust=False).mean()
                elif ind_type == "sma":
                    df[ind_name] = df["close"].rolling(window=period).mean()
                elif ind_type == "rsi":
                    df[ind_name] = self._calculate_rsi(df["close"], period)
                else:
                    # Get the indicator class from the registry
                    try:
                        IndicatorClass = get_indicator_class(ind_type)
                        
                        # Prepare parameters
                        ind_params = params.copy()
                        if 'type' in ind_params: del ind_params['type']
                        
                        # Parameter mapping (period -> rsi_period vs.)
                        ind_params = self._map_indicator_params(ind_type, ind_params)

                        # Create an instance
                        indicator = IndicatorClass(**ind_params)
                        
                        # Batch hesapla
                        result = indicator.calculate_batch(calc_df)
                        
                        # Add the results
                        if isinstance(result, pd.Series):
                            df[ind_name] = result
                        elif isinstance(result, pd.DataFrame):
                            # Multi-output: add additional outputs like stochrsi_14_k, stochrsi_14_d
                            for col in result.columns:
                                df[f"{ind_name}_{col}"] = result[col]
                        else:
                            logger.warning(f"⚠️ {ind_name}: Unexpected result type {type(result)}")

                    except Exception as e:
                        logger.warning(f"⚠️ {ind_name} could not be calculated: {e}")

                logger.debug(f"📊 {ind_name} calculated")

            except Exception as e:
                logger.error(f"❌ {ind_name} calculation error: {e}")

        return df

    def _add_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates extra features from the config."""
        features_config = self.config.get("features", {})

        for category, features in features_config.items():
            if not isinstance(features, dict):
                continue

            for feature_name, feature_params in features.items():
                if not isinstance(feature_params, dict):
                    continue

                if not feature_params.get("enabled", False):
                    continue

                try:
                    df = self._calculate_feature(df, feature_name, feature_params)
                except Exception as e:
                    logger.warning(f"⚠️ {feature_name} could not be calculated: {e}")

        return df

    def _calculate_feature(
        self,
        df: pd.DataFrame,
        name: str,
        params: dict
    ) -> pd.DataFrame:
        """
        Tek bir feature hesaplar.
        """
        from components.indicators import get_indicator_class

        window = params.get("window", 10)

        # Price-based features
        if name == "ema_distance":
            # Distance to EMA (normalized)
            ema_col = self._find_ema_column(df)
            if ema_col:
                df[name] = (df["close"] - df[ema_col]) / df[ema_col]

        elif name == "price_momentum":
            # Price momentum (last N bar changes)
            df[name] = (df["close"] - df["close"].shift(window)) / df["close"].shift(window)

        elif name == "price_volatility":
            # Price volatility (rolling std)
            df[name] = df["close"].pct_change().rolling(window=window).std()

        # Momentum features
        elif name == "rsi_change":
            # RSI change
            rsi_col = self._find_rsi_column(df)
            if rsi_col:
                df[name] = df[rsi_col] - df[rsi_col].shift(window)

        elif name == "rsi_slope":
            # RSI slope (trend)
            rsi_col = self._find_rsi_column(df)
            if rsi_col:
                df[name] = df[rsi_col].diff(window) / window

        # Volume features
        elif name == "volume_change":
            # Volume / SMA(volume, N)
            vol_sma = df["volume"].rolling(window=window).mean()
            df[name] = df["volume"] / vol_sma

        elif name == "volume_spike":
            # Volume spike detection
            threshold = params.get("threshold", 2.0)
            vol_sma = df["volume"].rolling(window=20).mean()
            df[name] = (df["volume"] > threshold * vol_sma).astype(int)
            
        elif name == "obv_change":
             # OBV Change
             calc_df = self._prepare_data_for_indicator(df)
             try:
                 OBV = get_indicator_class("obv")
                 obv_ind = OBV()
                 obv_val = obv_ind.calculate_batch(calc_df)
                 if isinstance(obv_val, pd.Series):
                     df[name] = obv_val.pct_change(window)
                 elif isinstance(obv_val, pd.DataFrame) and 'obv' in obv_val.columns:
                     df[name] = obv_val['obv'].pct_change(window)
                 else:
                     raise ValueError("OBV output format unknown")
             except Exception as e:
                 # Fallback
                 change = df['close'].diff()
                 direction = np.where(change > 0, 1, -1)
                 direction[change == 0] = 0
                 obv = (direction * df['volume']).cumsum()
                 df[name] = obv.pct_change(window)

        # Volatility features
        elif name in ["atr_14", "atr_20"]:
            # ATR hesapla
            period = params.get("params", {}).get("period", 14)
            df[name] = self._calculate_atr(df, period)

        elif name == "atr_change":
            # ATR change
            atr_col = self._find_atr_column(df)
            if atr_col:
                df[name] = df[atr_col].pct_change(window)

        # Indicator-based features via Registry
        elif "indicator" in params:
            ind_type = params["indicator"]
            ind_type = self._resolve_indicator_type(ind_type) # <--- ALIAS
            ind_params = params.get("params", {})
            
            try:
                IndicatorClass = get_indicator_class(ind_type)
                
                # Parameter mapping
                ind_params = self._map_indicator_params(ind_type, ind_params)
                
                # Instance
                indicator = IndicatorClass(**ind_params)
                
                # Prepare data
                calc_df = self._prepare_data_for_indicator(df)
                
                # Batch calculate
                result = indicator.calculate_batch(calc_df)
                
                if isinstance(result, pd.Series):
                    df[name] = result
                elif isinstance(result, pd.DataFrame):
                    # Multi-output
                    for col in result.columns:
                        df[f"{name}_{col}"] = result[col]
                        
            except Exception as e:
                logger.warning(f"⚠️ {name} ({ind_type}) could not be calculated: {e}")

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds automatically derived features."""
        # Close price percentage change (always add)
        if "close_pct" not in df.columns:
            df["close_pct"] = df["close"].pct_change()

        # High-Low range
        if "hl_range" not in df.columns:
            df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        return df

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesaplar."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR hesaplar."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def _find_ema_column(self, df: pd.DataFrame) -> str | None:
        """EMA kolonunu bulur."""
        for col in df.columns:
            if col.startswith("ema_"):
                return col
        return None

    def _find_rsi_column(self, df: pd.DataFrame) -> str | None:
        """RSI kolonunu bulur."""
        for col in df.columns:
            if col.startswith("rsi_"):
                return col
        return None

    def _find_atr_column(self, df: pd.DataFrame) -> str | None:
        """ATR kolonunu bulur."""
        for col in df.columns:
            if col.startswith("atr_"):
                return col
        return None

    def _prepare_data_for_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the data for indicator calculation (adds timestamp)."""
        calc_df = df.copy()
        
        # If timestamp is not available, get it from the index or create it.
        if 'timestamp' not in calc_df.columns:
            if isinstance(calc_df.index, pd.DatetimeIndex):
                # DatetimeIndex -> int64 timestamp (ms)
                calc_df['timestamp'] = calc_df.index.astype(np.int64) // 10**6
            else:
                 # Dummy timestamp
                 calc_df['timestamp'] = range(len(calc_df))
        
        return calc_df

    def _resolve_indicator_type(self, ind_type: str) -> str:
        """Resolves indicator aliases."""
        aliases = {
            "stochrsi": "stochasticrsi",
            "bb": "bollinger",
            "bollinger_bands": "bollinger",
        }
        return aliases.get(ind_type.lower(), ind_type)

    def _map_indicator_params(self, ind_type: str, params: dict) -> dict:
        """Maps indicator parameters (e.g., period -> rsi_period)."""
        new_params = params.copy()
        
        # StochRSI: period -> rsi_period, stoch_period; k_period -> k_smooth; d_period -> d_smooth
        if ind_type == "stochasticrsi":
             if "period" in new_params:
                 period = new_params.pop("period")
                 if "rsi_period" not in new_params:
                     new_params["rsi_period"] = period
                 if "stoch_period" not in new_params:
                     new_params["stoch_period"] = period
             
             if "k_period" in new_params:
                 new_params["k_smooth"] = new_params.pop("k_period")
            
             if "d_period" in new_params:
                 new_params["d_smooth"] = new_params.pop("d_period")
                     
        # If mapping is needed for other indicators, add it here.
        
        return new_params

    def get_feature_names(self) -> list[str]:
        """
        Returns all feature names.

        Returns:
            list: Feature isimleri
        """
        features = []

        # Strategy indicators
        features.extend(self.strategy_indicators.keys())

        # Ekstra feature'lar
        features_config = self.config.get("features", {})
        for category, cat_features in features_config.items():
            if isinstance(cat_features, dict):
                for fname, fparams in cat_features.items():
                    if isinstance(fparams, dict) and fparams.get("enabled", False):
                        features.append(fname)

        # Derived features
        features.extend(["close_pct", "hl_range"])

        return features

    def get_normalizer_config(self, feature_name: str) -> dict:
        """
        Returns the normalizer config for the feature.

        Args:
            feature_name: Feature name

        Returns:
            dict: Normalizer config
        """
        normalizers = self.config.get("normalizers", {})

        if feature_name in normalizers:
            return normalizers[feature_name]

        return normalizers.get("default", {
            "method": "rolling_zscore",
            "window": 100
        })


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 FeatureExtractor Test")
    print("=" * 60)

    # Test 1: Initialization
    print("\nTest 1: FeatureExtractor initialization")
    try:
        extractor = FeatureExtractor(strategy_name="simple_rsi")
        print(f"   ✅ Initialization successful")
        print(f"   📊 Strategy indicators: {list(extractor.strategy_indicators.keys())}")
        print(f"   📊 All features: {extractor.get_feature_names()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Feature extraction
    print("\nTest 2: Feature extraction")
    try:
        # Create dummy data
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "high": 0,
            "low": 0,
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": np.random.randint(1000, 10000, n)
        })
        df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.randn(n) * 0.3)
        df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.randn(n) * 0.3)

        result = extractor.extract(df)
        print(f"   ✅ Extraction successful")
        print(f"   📊 Orijinal kolonlar: {len(df.columns)}")
        print(f"   📊 Result columns: {len(result.columns)}")
        print(f"   📊 New columns: {set(result.columns) - set(df.columns)}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Normalizer config
    print("\nTest 3: Normalizer config")
    try:
        for fname in ["rsi_14", "ema_distance", "unknown_feature"]:
            config = extractor.get_normalizer_config(fname)
            print(f"   📊 {fname}: {config}")
        print(f"   ✅ Normalizer config successful")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
