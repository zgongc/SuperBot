#!/usr/bin/env python3
"""
modules/simple_train/core/data_loader.py
SuperBot - Data Loader for AI Training
Author: SuperBot Team
Date: 2026-01-14
Versiyon: 1.0.0

ParquetsEngine wrapper - Loading training data.

Features:
- Reading data from Parquet files
- Support for multiple symbols
- Multi-timeframe (MTF) support
- Time range filtering
- Train/Val/Test split
- Support for warm-up sets (by going backward from the start_date).

Usage:
    from modules.simple_train.core import DataLoader

    loader = DataLoader(config_path="modules/simple_train/configs/training.yaml")

    # Single timeframe
    df = loader.load("BTCUSDT", "5m")

    # Multi-timeframe (config'den: "5m,15m,30m")
    mtf_data = loader.load_mtf("BTCUSDT")  # Returns dict: {"5m": df, "15m": df, "30m": df}

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - pyyaml>=6.0
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

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
    logger = get_logger("modules.simple_train.core.data_loader")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("modules.simple_train.core.data_loader")


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """
    Training data loading class.

    ParquetsEngine'i kullanarak:
    - Loads data for the specified symbol and timeframe.
    - Supports multi-timeframe (MTF) ("5m,15m,30m").
    - Supports warmup bars (goes back from start_date).
    - Performs time range filtering.
    - Performs train/val/test split.
    """

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
        parquets_engine: Any | None = None
    ):
        """
        Initializes the DataLoader.

        Args:
            config: Data configuration dictionary (priority).
            config_path: Path to the training.yaml file.
            parquets_engine: Optional ParquetsEngine instance.
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Default config path
            default_path = SIMPLE_TRAIN_ROOT / "configs" / "training.yaml"
            self.config = self._load_config(default_path)

        # Data config
        self.data_config = self.config.get("data", {})
        self.split_config = self.config.get("training", {}).get("split", {})

        # ParquetsEngine (lazy load)
        self._parquets_engine = parquets_engine

        logger.info("📊 DataLoader started")

    def _load_config(self, config_path: str | Path) -> dict:
        """Loads the configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"⚠️ Configuration not found: {config_path}")
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def parquets_engine(self):
        """ParquetsEngine lazy load."""
        if self._parquets_engine is None:
            try:
                from components.managers.parquets_engine import ParquetsEngine
                self._parquets_engine = ParquetsEngine()
                logger.debug("🔍 ParquetsEngine loaded")
            except ImportError as e:
                logger.error(f"❌ ParquetsEngine import error: {e}")
                raise
        return self._parquets_engine

    def load(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> pd.DataFrame:
        """
        Loads all data for the specified symbol (sync wrapper).

        Args:
            symbol: Symbol name (default: from config)
            timeframe: Timeframe (default: config'den)
            start_date: Start date (default: from config)
            end_date: End date (default: from config)
            warmup_bars: Number of warmup bars (default: from config)

        Returns:
            pd.DataFrame: OHLCV DataFrame (warmup dahil)
        """
        # Call the async load synchronously.
        try:
            # Use asyncio.run() for Python 3.10+
            return asyncio.run(
                self.load_async(symbol, timeframe, start_date, end_date, warmup_bars)
            )
        except RuntimeError as e:
            # The event loop is already running (Jupyter, async context)
            if "running event loop" in str(e):
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        self.load_async(symbol, timeframe, start_date, end_date, warmup_bars)
                    )
                except ImportError:
                    logger.error("❌ nest_asyncio is required: pip install nest_asyncio")
                    raise
            raise

    async def load_async(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> pd.DataFrame:
        """
        Loads all data for the specified symbol (async).

        Args:
            symbol: Symbol name (default: from config)
            timeframe: Timeframe (default: config'den)
            start_date: Start date (default: from config)
            end_date: End date (default: from config)
            warmup_bars: Number of warmup bars (default: from config)

        Returns:
            pd.DataFrame: OHLCV DataFrame (warmup dahil)

        Note:
            Warmup bars are loaded before the start_date.
            Example: start_date=2024-01-01, warmup=200 -> Approximately 200 bars are pulled from the year 2023.
        """
        # From config, default values
        symbol = symbol or self.data_config.get("symbols", ["BTCUSDT"])[0]
        timeframe = timeframe or self.data_config.get("timeframe", "5m")
        start_date = start_date or self.data_config.get("start_date", "2024-01-01")
        end_date = end_date or self.data_config.get("end_date", "2025-01-01")
        warmup_bars = warmup_bars if warmup_bars is not None else self.data_config.get("warmup_bars", 200)

        logger.info(f"📂 Loading data: {symbol} {timeframe}")
        logger.info(f"   Date range: {start_date} - {end_date}")
        logger.info(f"   Warmup bars: {warmup_bars}")

        try:
            # Use ParquetsEngine.get_historical_data
            # This method goes backward from the start_date using warmup_candles.
            df = await self.parquets_engine.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                warmup_candles=warmup_bars,
                utc_offset=0  # Use UTC
            )

            if df is None or df.empty:
                logger.warning(f"⚠️ Data not found: {symbol}")
                return pd.DataFrame()

            # Prepare the DataFrame
            df = self._prepare_dataframe(df)

            logger.info(f"✅ {len(df)} bar loaded (including warmup: {warmup_bars})")
            return df

        except FileNotFoundError as e:
            logger.error(f"❌ Parquet file not found: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return pd.DataFrame()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the DataFrame for AI training.

        - Kolon isimlerini standardize eder
        - Index'i ayarlar
        - Cleans up unnecessary columns.

        Args:
            df: Raw DataFrame from ParquetsEngine

        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        # Kolon mapping (ParquetsEngine output → standard)
        column_mapping = {
            "open_time": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }

        # Rename existing columns (if any)
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Required columns
        required_cols = ["open", "high", "low", "close", "volume"]

        # Timestamp'i index yap
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif "open_time" in df.columns:
            df = df.set_index("open_time")

        # Keep only the necessary columns
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]

        # Sorting (chronological)
        df = df.sort_index()

        return df

    def load_split(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train/val/test sets.

        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            warmup_bars: Number of warmup bars

        Returns:
            tuple: (train_df, val_df, test_df)

        Note:
            Warmup bars are included only in the training set.
            When splitting, the warmup part remains in the training set.
        """
        # Load all data (including warmup)
        df = self.load(symbol, timeframe, start_date, end_date, warmup_bars)

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Split ratios
        train_ratio = self.split_config.get("train_ratio", 0.7)
        val_ratio = self.split_config.get("val_ratio", 0.15)
        # test_ratio = 1 - train_ratio - val_ratio

        # Split method
        method = self.split_config.get("method", "time_based")

        if method == "time_based":
            return self._time_based_split(df, train_ratio, val_ratio)
        elif method == "random":
            return self._random_split(df, train_ratio, val_ratio)
        elif method == "walk_forward":
            # A separate method is required for walk-forward.
            logger.warning("⚠️ Walk-forward split is not yet supported, using time_based")
            return self._time_based_split(df, train_ratio, val_ratio)
        else:
            logger.warning(f"⚠️ Unknown split method: {method}, time_based is being used")
            return self._time_based_split(df, train_ratio, val_ratio)

    def _time_based_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split (to prevent data leakage).

        Args:
            df: DataFrame
            train_ratio: Training ratio (0-1)
            val_ratio: Validation ratio (0-1)

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(f"📊 Time-based split:")
        logger.info(f"   Train: {len(train_df)} bars ({train_ratio*100:.0f}%)")
        logger.info(f"   Val: {len(val_df)} bars ({val_ratio*100:.0f}%)")
        logger.info(f"   Test: {len(test_df)} bars ({(1-train_ratio-val_ratio)*100:.0f}%)")

        return train_df, val_df, test_df

    def _random_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split (using shuffle).
        WARNING: Not recommended for time series data (risk of data leakage).

        Args:
            df: DataFrame
            train_ratio: Training ratio (0-1)
            val_ratio: Validation ratio (0-1)

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.warning("⚠️ Using random split - risk of data leakage!")

        # Shuffle
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_shuffled.iloc[:train_end].copy()
        val_df = df_shuffled.iloc[train_end:val_end].copy()
        test_df = df_shuffled.iloc[val_end:].copy()

        logger.info(f"📊 Random split:")
        logger.info(f"   Train: {len(train_df)} bars")
        logger.info(f"   Val: {len(val_df)} bars")
        logger.info(f"   Test: {len(test_df)} bars")

        return train_df, val_df, test_df

    async def load_all_symbols_async(
        self,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Loads all symbols (async).

        Args:
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            warmup_bars: Number of warmup bars

        Returns:
            dict: in the format of {symbol: DataFrame}
        """
        symbols = self.data_config.get("symbols", ["BTCUSDT"])
        result = {}

        for symbol in symbols:
            df = await self.load_async(symbol, timeframe, start_date, end_date, warmup_bars)
            if not df.empty:
                result[symbol] = df

        logger.info(f"✅ {len(result)}/{len(symbols)} symbols loaded")
        return result

    def parse_timeframes(self, timeframe: str | None = None) -> list[str]:
        """
        Timeframe string'ini parse eder.

        Args:
            timeframe: in the format "5m" or "5m,15m,30m"

        Returns:
            list: ["5m"] or ["5m", "15m", "30m"]
        """
        tf = timeframe or self.data_config.get("timeframe", "5m")
        return [t.strip() for t in tf.split(",")]

    def load_mtf(
        self,
        symbol: str | None = None,
        timeframes: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Loads data from multiple timeframes.

        Args:
            symbol: Symbol name
            timeframes: "5m,15m,30m" or ["5m", "15m", "30m"]
            start_date: Start date
            end_date: End date
            warmup_bars: Number of warmup bars

        Returns:
            dict: {"5m": df, "15m": df, "30m": df}
        """
        # Timeframes parse
        if timeframes is None:
            tf_list = self.parse_timeframes()
        elif isinstance(timeframes, str):
            tf_list = self.parse_timeframes(timeframes)
        else:
            tf_list = timeframes

        symbol = symbol or self.data_config.get("symbols", ["BTCUSDT"])[0]

        logger.info(f"📂 Loading MTF data: {symbol}")
        logger.info(f"   Timeframes: {tf_list}")

        result = {}
        for tf in tf_list:
            df = self.load(symbol, tf, start_date, end_date, warmup_bars)
            if not df.empty:
                result[tf] = df
                logger.info(f"   ✅ {tf}: {len(df)} bar")

        return result

    async def load_mtf_async(
        self,
        symbol: str | None = None,
        timeframes: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Loads data from multiple timeframes (async).

        Args:
            symbol: Symbol name
            timeframes: "5m,15m,30m" or ["5m", "15m", "30m"]
            start_date: Start date
            end_date: End date
            warmup_bars: Number of warmup bars

        Returns:
            dict: {"5m": df, "15m": df, "30m": df}
        """
        # Timeframes parse
        if timeframes is None:
            tf_list = self.parse_timeframes()
        elif isinstance(timeframes, str):
            tf_list = self.parse_timeframes(timeframes)
        else:
            tf_list = timeframes

        symbol = symbol or self.data_config.get("symbols", ["BTCUSDT"])[0]

        logger.info(f"📂 Loading MTF data: {symbol}")
        logger.info(f"   Timeframes: {tf_list}")

        result = {}
        for tf in tf_list:
            df = await self.load_async(symbol, tf, start_date, end_date, warmup_bars)
            if not df.empty:
                result[tf] = df
                logger.info(f"   ✅ {tf}: {len(df)} bar")

        return result

    def is_mtf(self, timeframe: str | None = None) -> bool:
        """
        Checks if multi-timeframe is present in the configuration.

        Returns:
            bool: True if MTF
        """
        tf_list = self.parse_timeframes(timeframe)
        return len(tf_list) > 1

    def get_base_timeframe(self, timeframe: str | None = None) -> str:
        """
        Returns the base (smallest) timeframe.

        Returns:
            str: Base timeframe (e.g., "5m")
        """
        tf_list = self.parse_timeframes(timeframe)
        return tf_list[0]  # The first timeframe is accepted as the base.

    def load_all_symbols(
        self,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        warmup_bars: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Loads all symbols (sync wrapper).

        Args:
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            warmup_bars: Number of warmup bars

        Returns:
            dict: in the format of {symbol: DataFrame}
        """
        try:
            return asyncio.run(
                self.load_all_symbols_async(timeframe, start_date, end_date, warmup_bars)
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.load_all_symbols_async(timeframe, start_date, end_date, warmup_bars)
            )

    def get_info(self) -> dict:
        """
        Returns the data configuration information.

        Returns:
            dict: Config bilgileri
        """
        return {
            "source": self.data_config.get("source", "parquet"),
            "parquet_path": self.data_config.get("parquet_path", "data/parquets"),
            "symbols": self.data_config.get("symbols", []),
            "timeframe": self.data_config.get("timeframe", "5m"),
            "start_date": self.data_config.get("start_date"),
            "end_date": self.data_config.get("end_date"),
            "warmup_bars": self.data_config.get("warmup_bars", 200),
            "split_method": self.split_config.get("method", "time_based"),
            "train_ratio": self.split_config.get("train_ratio", 0.7),
            "val_ratio": self.split_config.get("val_ratio", 0.15),
            "test_ratio": self.split_config.get("test_ratio", 0.15),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DataLoader Test")
    print("=" * 60)

    # Test 1: Initialization
    print("\nTest 1: DataLoader initialization")
    try:
        loader = DataLoader()
        print(f"   ✅ Initialization successful")
        info = loader.get_info()
        print(f"   📊 Config:")
        for key, value in info.items():
            print(f"      - {key}: {value}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Data loading (if ParquetsEngine is available)
    print("\nTest 2: Data loading (including warmup)")
    try:
        # Check if ParquetsEngine can be imported.
        from components.managers.parquets_engine import ParquetsEngine

        df = loader.load("BTCUSDT", "5m", warmup_bars=200)
        if not df.empty:
            print(f"   ✅ Data loaded: {len(df)} rows")
            print(f"   📊 Kolonlar: {list(df.columns)}")
            if hasattr(df.index, 'min'):
                print(f"   📊 Date range: {df.index.min()} - {df.index.max()}")
        else:
            print(f"   ⚠️ Data not found")

    except ImportError:
        print(f"   ⚠️ ParquetsEngine could not be imported, test skipped")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Split
    print("\nTest 3: Train/Val/Test split")
    try:
        # Test with dummy data
        import numpy as np

        dummy_df = pd.DataFrame({
            "open": np.random.randn(1000),
            "high": np.random.randn(1000),
            "low": np.random.randn(1000),
            "close": np.random.randn(1000),
            "volume": np.random.randint(100, 1000, 1000)
        })

        # Time-based split test
        train, val, test = loader._time_based_split(dummy_df, 0.7, 0.15)
        print(f"   ✅ Split successful")
        print(f"      Train: {len(train)} ({len(train)/len(dummy_df)*100:.1f}%)")
        print(f"      Val: {len(val)} ({len(val)/len(dummy_df)*100:.1f}%)")
        print(f"      Test: {len(test)} ({len(test)/len(dummy_df)*100:.1f}%)")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
