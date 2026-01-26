#!/usr/bin/env python3
"""
components/managers/parquets_engine.py
SuperBot - Parquet Historical Data Manager
Author: SuperBot Team
Date: 2025-11-11
Versiyon: 1.0.0

Management of historical data from Parquet files.

Features:
- Multi-year support (automatically merges files for 2023, 2024, 2025).
- Timezone conversion (UTC ↔ Local)
- Support for warmup period (N candles before the start)
- Smart file finding (skip missing years)
- Memory efficient (lazy loading)
- TODO: MTF resample (1m -> 5m, 15m, 1h...) Sum for volume, resample for OHLC.

Usage:
    from components.managers.parquets_engine import ParquetsEngine

    engine = ParquetsEngine(data_path='data/parquets', logger_engine=logger)

    # Historical data al
    df = await engine.get_historical_data(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date='2023-01-01T00:00',
        end_date='2025-01-03T00:00',
        warmup_candles=200,
        utc_offset=3
    )

Dependencies:
    - python>=3.10
    - pandas
    - pyarrow (for parquet reading)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class ParquetsEngine:
    """
    Management of historical data from Parquet files.

    Multi-year support, timezone conversion, warmup period
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        config_engine: Any = None,
        logger_engine: Any = None
    ):
        """
        Initialize ParquetsEngine

        Args:
            data_path: The folder containing the Parquet files (optional - read from config)
            config_engine: ConfigEngine instance (optional - reads the path from the config)
            logger_engine: LoggerEngine instance (optional)
        """
        self.config_engine = config_engine
        self.logger_engine = logger_engine
        self.logger = logger_engine.get_logger(__name__) if logger_engine else None

        # Data path - read from config or use fallback
        if data_path:
            self.data_path = Path(data_path)
        elif config_engine:
            # Config'den oku
            parquet_config = config_engine.get('parquet', {})
            path = parquet_config.get('path', 'data/parquets')
            self.data_path = Path(path)
            if self.logger:
                self.logger.info(f"📂 ParquetsEngine: Path read from config: {self.data_path}")
        else:
            # Fallback
            self.data_path = Path('data/parquets')
            if self.logger:
                self.logger.warning(f"⚠️ ParquetsEngine: Config not found, using default path: {self.data_path}")

        # For caching (to avoid reading the same file repeatedly)
        self._file_cache: Dict[str, pd.DataFrame] = {}

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        warmup_candles: int = 0,
        utc_offset: int = 0
    ) -> pd.DataFrame:
        """
        Retrieves historical data within the specified date range.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h)
            start_date: Start date (local time, ISO format)
            end_date: End date (local time, ISO format)
            warmup_candles: How many candles are needed before the start (for warmup)
            utc_offset: UTC offset (hours) - e.g., 3 = UTC+3

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume, timestamp

        Raises:
            FileNotFoundError: The required file was not found.
            RuntimeError: Insufficient data
        """
        # Validate required parameters
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for ParquetsEngine")

        # Ensure warmup_candles is integer (may come as float from YAML config)
        warmup_candles = int(warmup_candles)

        if self.logger:
            self.logger.info(f"📂 ParquetsEngine: Loading historical data")
            self.logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}")
            self.logger.info(f"   Start (local): {start_date}")
            self.logger.info(f"   End (local): {end_date}")
            self.logger.info(f"   Warmup: {warmup_candles} mum")
            self.logger.info(f"   Timezone: UTC{utc_offset:+d}")

        # Convert local time to UTC
        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)

        start_utc = (dt_start - pd.Timedelta(hours=utc_offset)).tz_localize('UTC')
        end_utc = (dt_end - pd.Timedelta(hours=utc_offset)).tz_localize('UTC')

        if self.logger:
            self.logger.info(f"   Start (UTC): {start_utc}")
            self.logger.info(f"   End (UTC): {end_utc}")

        # Which year files are needed? (from start_utc - warmup to end_utc)
        # Data is also needed before the beginning for warmup.
        years = self._get_required_years(start_utc, end_utc, warmup_candles, timeframe)

        if self.logger:
            self.logger.info(f"   Required years: {years}")

        # Read and merge multi-year files
        df_list = []
        for year in years:
            df_year = self._read_parquet_file(symbol, timeframe, year)
            if df_year is not None:
                df_list.append(df_year)

        if len(df_list) == 0:
            raise FileNotFoundError(f"No parquet file found: {symbol}_{timeframe}")

        # Merge
        df = pd.concat(df_list, ignore_index=True)

        # CRITICAL: Normalize open_time to UTC-aware for consistent comparison
        # Some files might be tz-naive (old downloads) and some tz-aware (new downloads)
        if 'open_time' in df.columns:
            # Ensure open_time is datetime type (concat might lose dtype)
            if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
                df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
            else:
                # Convert to UTC-aware if not already
                if df['open_time'].dt.tz is None:
                    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
                else:
                    df['open_time'] = df['open_time'].dt.tz_convert('UTC')

            # Add timestamp column (int64 ms)
            # Remove timezone info before converting to int64
            df['timestamp'] = df['open_time'].dt.tz_localize(None).astype('int64') // 10**6

        if self.logger:
            self.logger.info(f"   ✅ Total {len(df)} rows loaded (merged)")

        # warmup for initialization, requires warmup_candles amount of data BEFORE the start.
        if warmup_candles > 0:
            df_before_start = df[df['open_time'] < start_utc]

            if len(df_before_start) < warmup_candles:
                # WARNING: Insufficient warmup, but continue
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Insufficient warmup data!"
                        f"A total of {warmup_candles} candles are required before the start time ({start_utc}), "
                        f"but there are only {len(df_before_start)} candles."
                        f"Indicator values may be missing for the first {warmup_candles - len(df_before_start)} candles."
                    )

                # Use what exists
                if len(df_before_start) > 0:
                    warmup_start = df_before_start.iloc[0]['open_time']
                    if self.logger:
                        self.logger.info(f"   📊 Partial warmup starting: {warmup_start} ({len(df_before_start)} samples)")
                    df = df[(df['open_time'] >= warmup_start) & (df['open_time'] <= end_utc)].copy()
                else:
                    # No warmup, start from start_utc
                    if self.logger:
                        self.logger.warning(f"⚠️ No warmup data available, starting from scratch ({start_utc})")
                    df = df[(df['open_time'] >= start_utc) & (df['open_time'] <= end_utc)].copy()
            else:
                # There is enough warmup.
                warmup_start = df_before_start.iloc[-warmup_candles]['open_time']

                if self.logger:
                    self.logger.info(f"   📊 Warmup start: {warmup_start}")

                # Filter from the warmup start to end_utc
                df = df[(df['open_time'] >= warmup_start) & (df['open_time'] <= end_utc)].copy()
        else:
            # No warmup, only the start_utc - end_utc range.
            df = df[(df['open_time'] >= start_utc) & (df['open_time'] <= end_utc)].copy()

        if self.logger:
            self.logger.info(f"   ✅ After filtering: {len(df)} rows")
            self.logger.info(f"   📅 Date range: {df.iloc[0]['open_time']} - {df.iloc[-1]['open_time']}")

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def _get_required_years(
        self,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        warmup_candles: int,
        timeframe: str
    ) -> List[int]:
        """
        Determine the required year files.

        Data is also needed before the start of the warmup, so data from previous years may be necessary.
        """
        # Start and end years
        start_year = start_utc.year
        end_year = end_utc.year

        # How many days should we go back for the warmup?
        if warmup_candles > 0:
            # Convert the timeframe to minutes
            tf_minutes = self._parse_timeframe_to_minutes(timeframe)

            # Total duration required for warmup (minutes)
            warmup_minutes = warmup_candles * tf_minutes

            # Convert minutes to days
            warmup_days = warmup_minutes / (60 * 24)

            # Warmup start date
            warmup_start = start_utc - pd.Timedelta(days=warmup_days)

            # Warmup start year
            warmup_start_year = warmup_start.year

            if self.logger:
                self.logger.debug(f"   Warmup calculation: {warmup_candles} x {tf_minutes}min = {warmup_days:.1f} days")
                self.logger.debug(f"   Warmup start year: {warmup_start_year}")

            # Update start_year
            start_year = min(start_year, warmup_start_year)

        # Create a list of years
        years = list(range(start_year, end_year + 1))

        return years

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert the timeframe string to minutes (e.g., '15m' -> 15, '1h' -> 60)"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        else:
            # Default: 1m
            return 1

    def _read_parquet_file(
        self,
        symbol: str,
        timeframe: str,
        year: int
    ) -> Optional[pd.DataFrame]:
        """
        Reads a single parquet file.

        Uses a cache; returns None if the file does not exist (does not raise an error).
        """
        # Windows case-insensitive fix: 1M (month) → 1MO
        # (To avoid confusion with 'm' for minute)
        file_timeframe = "1MO" if timeframe == "1M" else timeframe

        # New format: data/parquets/{symbol}/{symbol}_{timeframe}_{year}.parquet
        filename = f"{symbol}_{file_timeframe}_{year}.parquet"
        symbol_dir = self.data_path / symbol
        filepath = symbol_dir / filename

        # Is it in the cache?
        cache_key = str(filepath)
        if cache_key in self._file_cache:
            if self.logger:
                self.logger.debug(f"   📦 Read from cache: {filename}")
            return self._file_cache[cache_key]

        # Does the file exist?
        if not filepath.exists():
            if self.logger:
                self.logger.warning(f"   ⚠️  File not found (skipping): {filename}")
            return None

        # Oku
        try:
            df = pd.read_parquet(filepath)

            if self.logger:
                self.logger.info(f"   ✅ Read: {filename} ({len(df)} rows)")

            # Add to cache
            self._file_cache[cache_key] = df

            return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"   ❌ Read error: {filename} - {e}")
            return None

    def clear_cache(self):
        """Clear the cache"""
        self._file_cache.clear()
        if self.logger:
            self.logger.info("🧹 ParquetsEngine cache cleaned")

    # ========================================================================
    # TODO: MTF RESAMPLE SUPPORT
    # ========================================================================

    async def resample_timeframe(
        self,
        df: pd.DataFrame,
        source_tf: str,
        target_tf: str
    ) -> pd.DataFrame:
        """
        TODO: Resample from one timeframe to another.

        Example: 1m -> 5m, 15m, 1h

        Rules:
        - OHLC: first, max, min, last
        - Volume: sum
        - open_time: first

        Args:
            df: Source DataFrame
            source_tf: Source timeframe (e.g., 1m)
            target_tf: Target timeframe (e.g., 5m)

        Returns:
            Resampled DataFrame
        """
        raise NotImplementedError("MTF resample is not yet implemented - TODO")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from core.logger_engine import LoggerEngine
    from core.config_engine import ConfigEngine

    async def test_parquets_engine():
        """ParquetsEngine test"""
        print("=" * 80)
        print("ParquetsEngine Test")
        print("=" * 80)

        # Config & Logger
        config_engine = ConfigEngine(config_path='config/main.yaml')
        logger_engine = LoggerEngine()
        logger = logger_engine.get_logger(__name__)

        # Engine - config'den path oku
        engine = ParquetsEngine(
            config_engine=config_engine,
            logger_engine=logger_engine
        )

        # Test: 2025-01-01 - 2025-01-03 (200 warmup)
        logger.info("\n📊 Test 1: Multi-year warmup (2024 + 2025)")

        try:
            df = await engine.get_historical_data(
                symbol='BTCUSDT',
                timeframe='15m',
                start_date='2025-01-01T00:00',
                end_date='2025-01-03T00:00',
                warmup_candles=200,
                utc_offset=3
            )

            logger.info(f"\n✅ Test 1 SUCCESSFUL!")
            logger.info(f"   Total rows: {len(df)}")
            logger.info(f"   First candle: {df.iloc[0]['open_time']}")
            logger.info(f"   Last candle: {df.iloc[-1]['open_time']}")

        except Exception as e:
            logger.error(f"❌ Test 1 FAILED: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST COMPLETED!")
        logger.info("=" * 80)

    asyncio.run(test_parquets_engine())
