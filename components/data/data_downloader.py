#!/usr/bin/env python3
"""
managers/data_downloader.py
SuperBot - Data Downloader
Author: SuperBot Team
Date: 2025-10-17
Versiyon: 1.0.0

Description:
    Binance'den historical data indirir.
    Smart incremental update (no duplicates).

Features:
    1. Binance API integration
    2. All timeframes support
    3. Smart incremental update (son timestamp'ten devam)
    4. Duplicate detection & removal
    5. Parquet save
    6. Progress tracking

Usage:
    downloader = DataDownloader()

    # Download
    await downloader.download(
        symbol='BTCUSDT',
        timeframe='1m',
        start_date='2025-01-01',
        output_dir='data/parquets'
    )

    # Update (incremental)
    await downloader.update(
        symbol='BTCUSDT',
        timeframe='1m',
        output_dir='data/parquets'
    )

Dependencies:
    - python-binance
    - pandas
    - pyarrow
"""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Timezone utilities
try:
    from core.timezone_utils import get_utc_now
except ImportError:
    # Fallback if core module not available
    def get_utc_now():
        return datetime.now(timezone.utc)


# ============================================================================
# TIMEFRAME MAP
# ============================================================================

TIMEFRAME_MAP = {
    # Minutes
    "1m": Client.KLINE_INTERVAL_1MINUTE if BINANCE_AVAILABLE else "1m",
    "3m": Client.KLINE_INTERVAL_3MINUTE if BINANCE_AVAILABLE else "3m",
    "5m": Client.KLINE_INTERVAL_5MINUTE if BINANCE_AVAILABLE else "5m",
    "15m": Client.KLINE_INTERVAL_15MINUTE if BINANCE_AVAILABLE else "15m",
    "30m": Client.KLINE_INTERVAL_30MINUTE if BINANCE_AVAILABLE else "30m",

    # Hours
    "1h": Client.KLINE_INTERVAL_1HOUR if BINANCE_AVAILABLE else "1h",
    "2h": Client.KLINE_INTERVAL_2HOUR if BINANCE_AVAILABLE else "2h",
    "4h": Client.KLINE_INTERVAL_4HOUR if BINANCE_AVAILABLE else "4h",
    "6h": Client.KLINE_INTERVAL_6HOUR if BINANCE_AVAILABLE else "6h",
    "8h": Client.KLINE_INTERVAL_8HOUR if BINANCE_AVAILABLE else "8h",
    "12h": Client.KLINE_INTERVAL_12HOUR if BINANCE_AVAILABLE else "12h",

    # Days
    "1d": Client.KLINE_INTERVAL_1DAY if BINANCE_AVAILABLE else "1d",
    "3d": Client.KLINE_INTERVAL_3DAY if BINANCE_AVAILABLE else "3d",

    # Week/Month
    "1w": Client.KLINE_INTERVAL_1WEEK if BINANCE_AVAILABLE else "1w",
    "1M": Client.KLINE_INTERVAL_1MONTH if BINANCE_AVAILABLE else "1M",
}


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    """
    Data Downloader

    Binance'den historical data indirir.
    Prevents duplicates with smart incremental updates.

    Attributes:
        client: Binance client
        batch_size: Download batch size (default: 1000)
    """

    def __init__(self, api_key: str = "", api_secret: str = "", debug: bool = False):
        """
        Initialize Data Downloader

        Args:
            api_key: Binance API key (optional, not required for public data)
            api_secret: Binance API secret (optional)
            debug: Enable debug logging (default: False)
        """
        if not BINANCE_AVAILABLE:
            raise ImportError(
                "python-binance required! "
                "Install: pip install python-binance"
            )

        self.client = Client(api_key, api_secret)
        self.batch_size = 1000  # Binance limit (per request)
        self.write_chunk_size = 5000  # Write to disk every 5000 rows (streaming)
        self.debug = debug  # Debug mode

    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Returns the time delta based on the timeframe."""
        if timeframe.endswith('m'):
            return timedelta(minutes=int(timeframe[:-1]))
        elif timeframe.endswith('h'):
            return timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith('d'):
            return timedelta(days=int(timeframe[:-1]))
        elif timeframe.endswith('w'):
            return timedelta(weeks=int(timeframe[:-1]))
        elif timeframe == '1M':
            return timedelta(days=30)
        else:
            return timedelta(hours=1)  # Default

    # ========================================================================
    # GAP DETECTION & FILL
    # ========================================================================

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        time_col: str = 'open_time',
        year_start: Optional[pd.Timestamp] = None,
        year_end: Optional[pd.Timestamp] = None
    ) -> List[Dict]:
        """
        Detect missing data gaps in DataFrame

        Args:
            df: DataFrame with time series data
            timeframe: Timeframe (1m, 5m, 15m, etc.)
            time_col: Time column name
            year_start: Optional year start boundary (UTC) - only detect gaps after this
            year_end: Optional year end boundary (UTC) - only detect gaps before this

        Returns:
            List of gaps: [{'start': datetime, 'end': datetime, 'count': int}, ...]
        """
        if len(df) == 0:
            return []

        # Sort by time
        df = df.sort_values(time_col)

        # Get expected frequency (pandas 2.x: lowercase h/d/w)
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '3d': '3D',
            '1w': '1W', '1M': '1MS'  # Month start
        }

        freq = freq_map.get(timeframe, '1h')

        # Create expected timeline within year boundaries
        data_start = df[time_col].min()
        data_end = df[time_col].max()

        # Use year boundaries to detect gaps BEFORE existing data too
        # This allows downloading historical data before current data
        start = year_start if year_start else data_start
        end = year_end if year_end else data_end

        # Extend end to include current data if year_end is beyond data
        if data_end and end < data_end:
            end = data_end

        expected_timeline = pd.date_range(start, end, freq=freq)

        # Ensure both are timezone-aware or both timezone-naive for comparison
        existing_times = set(df[time_col])

        # Normalize timezone awareness
        if len(existing_times) > 0:
            sample_time = next(iter(existing_times))
            is_tz_aware = hasattr(sample_time, 'tz') and sample_time.tz is not None

            if is_tz_aware:
                # Make expected_timeline timezone-aware
                if expected_timeline.tz is None:
                    expected_timeline = expected_timeline.tz_localize('UTC')
            else:
                # Make expected_timeline timezone-naive
                if expected_timeline.tz is not None:
                    expected_timeline = expected_timeline.tz_localize(None)

        missing_times = sorted([t for t in expected_timeline if t not in existing_times])

        if not missing_times:
            return []

        # Group consecutive missing times into gaps
        gaps = []
        gap_start = missing_times[0]
        gap_end = missing_times[0]
        gap_count = 1

        time_delta = self._get_timeframe_delta(timeframe)

        for i in range(1, len(missing_times)):
            current = missing_times[i]
            prev = missing_times[i-1]

            # Check if consecutive
            if current - prev == time_delta:
                gap_end = current
                gap_count += 1
            else:
                # Save current gap
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'count': gap_count
                })

                # Start new gap
                gap_start = current
                gap_end = current
                gap_count = 1

        # Save last gap
        gaps.append({
            'start': gap_start,
            'end': gap_end,
            'count': gap_count
        })

        return gaps

    async def _fill_gaps(
        self,
        symbol: str,
        timeframe: str,
        gaps: List[Dict],
        output_path: Path
    ) -> int:
        """
        Fill detected gaps by downloading missing data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            gaps: List of gaps from _detect_gaps()
            output_path: Output parquet file

        Returns:
            Total rows downloaded
        """
        if not gaps:
            return 0

        print(f"\n🔍 Gap Detection:")
        print(f"   Found {len(gaps)} gap(s)")

        total_rows = 0
        gaps_filled = 0
        gaps_no_data = 0

        for i, gap in enumerate(gaps, 1):
            gap_start_str = gap['start'].strftime("%Y-%m-%d %H:%M:%S")
            gap_end_str = gap['end'].strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n   Gap {i}/{len(gaps)}: {gap_start_str} to {gap_end_str} ({gap['count']} candles)")

            # Download gap data
            new_rows = await self._download_and_append(
                symbol, timeframe, gap_start_str, gap_end_str, output_path
            )

            total_rows += new_rows

            if new_rows > 0:
                print(f"   ✅ Filled: +{new_rows} rows")
                gaps_filled += 1
            else:
                print(f"   ⚠️  No data available on Binance (exchange downtime or missing data)")
                gaps_no_data += 1

        # Summary
        if gaps_no_data > 0:
            print(f"\n   📊 Gap Fill Summary:")
            print(f"      ✅ Successfully filled: {gaps_filled}/{len(gaps)} gaps")
            print(f"      ⚠️  No data on Binance: {gaps_no_data}/{len(gaps)} gaps (permanent gaps)")
            print(f"      💡 Permanent gaps are likely due to exchange downtime in early 2018")

        return total_rows

    # ========================================================================
    # DOWNLOAD (FULL)
    # ========================================================================

    async def download(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
        output_dir: str = "data/parquets"
    ) -> pd.DataFrame:
        """
        Download historical data (full) - Multi-year support

        If start_date and end_date are in different years, it creates a separate file for each year:
        - 2024-01-01 to 2025-12-31 -> BTCUSDT_1m_2024.parquet + BTCUSDT_1m_2025.parquet

        Args:
            symbol: Trading pair (BTCUSDT)
            timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
            start_date: Start date (2024-01-01)
            end_date: End date or None (today)
            output_dir: Output directory

        Returns:
            Downloaded DataFrame (all years combined)
        """
        print(f"\n{'='*70}")
        print(f"DOWNLOADING: {symbol} {timeframe}")
        print(f"From: {start_date} to {end_date or 'now'}")
        print(f"{'='*70}\n")

        # Validate timeframe
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Available: {list(TIMEFRAME_MAP.keys())}"
            )

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            # When end_date is provided as YYYY-MM-DD, interpret as END of that day
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        else:
            # When no end_date, use current moment
            end_dt = get_utc_now()

        # Get years to download
        start_year = start_dt.year
        end_year = end_dt.year

        all_dfs = []

        # Download year by year
        for year in range(start_year, end_year + 1):
            # Year boundaries
            year_start = start_dt if year == start_year else datetime(year, 1, 1)
            year_end = end_dt if year == end_year else datetime(year, 12, 31, 23, 59, 59)

            year_start_str = year_start.strftime("%Y-%m-%d")
            year_end_str = year_end.strftime("%Y-%m-%d")

            print(f"\n--- Year {year}: {year_start_str} to {year_end_str} ---")

            # Get output path for this year
            output_path = self._get_output_path(symbol, timeframe, output_dir, year)

            # SMART DOWNLOAD: Check if file exists
            if output_path.exists():
                print(f"File exists: {output_path.name}")

                # Read full file for gap detection + duplicate removal
                # The column name can be 'timestamp' or 'open_time'
                try:
                    existing_df = pd.read_parquet(output_path)
                    time_col = 'open_time' if 'open_time' in existing_df.columns else 'timestamp'
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
                    raise

                initial_rows = len(existing_df)
                print(f"   Initial rows: {initial_rows:,}")

                # Check if file is empty from the start
                if initial_rows == 0:
                    print(f"   ⚠️  File is empty, downloading from scratch...")
                    output_path.unlink(missing_ok=True)  # Delete the empty file
                    total_rows = await self._download_klines_streaming(
                        symbol, timeframe, year_start_str, year_end_str, output_path
                    )
                    print(f"Year {year} complete: {total_rows:,} rows saved to {output_path.name}")
                    if total_rows == 0:
                        print(f"⚠️  No data for {year}")
                    continue

                # 1. Remove duplicates
                existing_df = existing_df.drop_duplicates(subset=[time_col], keep='last')
                existing_df = existing_df.sort_values(time_col).reset_index(drop=True)

                after_dedup_rows = len(existing_df)
                duplicates_removed = initial_rows - after_dedup_rows

                if duplicates_removed > 0:
                    print(f"   🧹 Removed {duplicates_removed:,} duplicates")

                # Create year boundary timestamps
                year_start_ts = pd.Timestamp(year_start).tz_localize('UTC')
                if hasattr(year_end, 'tzinfo') and year_end.tzinfo is not None:
                    year_end_ts = pd.Timestamp(year_end).tz_convert('UTC')
                else:
                    year_end_ts = pd.Timestamp(year_end).tz_localize('UTC')

                # 1b. Filter to only this year's data (remove cross-year contamination)
                # Ensure time column is timezone-aware for comparison
                if pd.api.types.is_datetime64_any_dtype(existing_df[time_col]):
                    if existing_df[time_col].dt.tz is None:
                        existing_df[time_col] = pd.to_datetime(existing_df[time_col], utc=True)
                    else:
                        existing_df[time_col] = existing_df[time_col].dt.tz_convert('UTC')

                # Filter: only keep data within this CALENDAR year (not start_date)
                # IMPORTANT: Use calendar year boundaries, not --start date
                # This prevents deleting existing data before --start date
                calendar_year_start_ts = pd.Timestamp(f"{year}-01-01", tz='UTC')
                calendar_year_end_ts = pd.Timestamp(f"{year}-12-31 23:59:59", tz='UTC')

                year_filtered_df = existing_df[
                    (existing_df[time_col] >= calendar_year_start_ts) &
                    (existing_df[time_col] <= calendar_year_end_ts)
                ].copy()

                rows_outside_year = len(existing_df) - len(year_filtered_df)
                if rows_outside_year > 0:
                    print(f"   🧹 Removed {rows_outside_year:,} rows outside year {year}")
                    existing_df = year_filtered_df
                    # Save cleaned data
                    self._save_to_parquet(existing_df, output_path)
                    print(f"   ✅ File cleaned: {len(existing_df):,} rows (duplicates: {duplicates_removed:,}, wrong year: {rows_outside_year:,})")
                elif duplicates_removed > 0:
                    # Save if only duplicates were removed
                    self._save_to_parquet(existing_df, output_path)
                    print(f"   ✅ File cleaned: {len(existing_df):,} rows")

                last_timestamp = existing_df[time_col].max()
                first_timestamp = existing_df[time_col].min()

                # Check if file became empty after cleaning (all rows were outside year)
                if len(existing_df) == 0 or pd.isna(last_timestamp):
                    print(f"   ⚠️  The file was cleared and is now empty, downloading from scratch...")
                    output_path.unlink(missing_ok=True)  # Delete the empty file
                    total_rows = await self._download_klines_streaming(
                        symbol, timeframe, year_start_str, year_end_str, output_path
                    )
                    print(f"Year {year} complete: {total_rows:,} rows saved to {output_path.name}")
                    if total_rows == 0:
                        print(f"⚠️  No data for {year}")
                    continue

                # Ensure timezone-aware for comparison
                if hasattr(last_timestamp, 'tz') and last_timestamp.tz is None:
                    last_timestamp = pd.Timestamp(last_timestamp, tz='UTC')
                if hasattr(first_timestamp, 'tz') and first_timestamp.tz is None:
                    first_timestamp = pd.Timestamp(first_timestamp, tz='UTC')

                print(f"   Date range: {first_timestamp} -> {last_timestamp}")

                # 2. Check if user wants data BEFORE existing data
                # If start_date is before first_timestamp, we need to download that earlier data first
                if year_start_ts < first_timestamp:
                    print(f"\n   📥 User requested earlier data: {year_start_ts} (before existing: {first_timestamp})")
                    # Download from user's start to first existing timestamp
                    earlier_start = year_start_ts.strftime("%Y-%m-%d %H:%M:%S")
                    earlier_end = (first_timestamp - self._get_timeframe_delta(timeframe)).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"      Downloading: {earlier_start} to {earlier_end}")

                    earlier_rows = await self._download_and_append(
                        symbol, timeframe, earlier_start, earlier_end, output_path
                    )
                    print(f"   ✅ Earlier data downloaded: +{earlier_rows:,} rows")

                    # Reload file to get updated state
                    existing_df = pd.read_parquet(output_path)
                    if pd.api.types.is_datetime64_any_dtype(existing_df[time_col]):
                        if existing_df[time_col].dt.tz is None:
                            existing_df[time_col] = pd.to_datetime(existing_df[time_col], utc=True)
                        else:
                            existing_df[time_col] = existing_df[time_col].dt.tz_convert('UTC')
                    existing_df = existing_df.sort_values(time_col).reset_index(drop=True)
                    first_timestamp = existing_df[time_col].min()
                    last_timestamp = existing_df[time_col].max()
                    print(f"   📊 After earlier download - Range: {first_timestamp} -> {last_timestamp}")

                # 3. Detect gaps in existing data (within year boundaries)
                gaps = self._detect_gaps(
                    existing_df,
                    timeframe,
                    time_col,
                    year_start=year_start_ts,
                    year_end=year_end_ts
                )

                if gaps:
                    total_missing = sum(g['count'] for g in gaps)
                    print(f"\n   🔍 Gap Analysis:")
                    print(f"      Found {len(gaps)} gap(s) with {total_missing:,} missing candles")

                    # Fill gaps
                    filled_rows = await self._fill_gaps(symbol, timeframe, gaps, output_path)
                    print(f"   ✅ Gaps filled: +{filled_rows:,} rows")

                    # CRITICAL: Reload ENTIRE file after gap filling to get accurate state
                    existing_df = pd.read_parquet(output_path)

                    # Ensure timezone-aware for comparison
                    if pd.api.types.is_datetime64_any_dtype(existing_df[time_col]):
                        if existing_df[time_col].dt.tz is None:
                            existing_df[time_col] = pd.to_datetime(existing_df[time_col], utc=True)
                        else:
                            existing_df[time_col] = existing_df[time_col].dt.tz_convert('UTC')

                    existing_df = existing_df.sort_values(time_col).reset_index(drop=True)
                    last_timestamp = existing_df[time_col].max()
                    first_timestamp = existing_df[time_col].min()

                    # Ensure timezone-aware
                    if hasattr(last_timestamp, 'tz') and last_timestamp.tz is None:
                        last_timestamp = pd.Timestamp(last_timestamp, tz='UTC')
                    if hasattr(first_timestamp, 'tz') and first_timestamp.tz is None:
                        first_timestamp = pd.Timestamp(first_timestamp, tz='UTC')

                    print(f"   📊 After gap fill - Total rows: {len(existing_df):,}, Range: {first_timestamp} -> {last_timestamp}")
                else:
                    print(f"   ✅ No gaps detected")

                # 3. Update to latest (append new data after last_timestamp)
                # But only within this year's boundary!
                # Convert year_end to timezone-aware timestamp
                if hasattr(year_end, 'tzinfo') and year_end.tzinfo is not None:
                    year_end_ts = pd.Timestamp(year_end).tz_convert('UTC')
                else:
                    year_end_ts = pd.Timestamp(year_end).tz_localize('UTC')
                time_delta = self._get_timeframe_delta(timeframe)
                next_expected = last_timestamp + time_delta

                # Check if we need more data for this year
                if last_timestamp >= year_end_ts:
                    print(f"\n   ✅ Year {year} is complete (ends at {last_timestamp})")
                    total_rows = len(existing_df)
                elif next_expected > year_end_ts:
                    print(f"\n   ✅ Data is up-to-date for year {year}!")
                    print(f"      Last: {last_timestamp}, Year end: {year_end_ts}")
                    total_rows = len(existing_df)
                else:
                    # Download up to year_end, not beyond
                    update_end = year_end_ts if year < get_utc_now().year else None
                    update_end_str = year_end_ts.strftime("%Y-%m-%d %H:%M:%S") if update_end else "now"

                    print(f"\n   📥 Updating from {last_timestamp} to {update_end_str}...")
                    # Download missing data and append
                    next_start = (last_timestamp + time_delta).strftime("%Y-%m-%d %H:%M:%S")
                    new_rows = await self._download_and_append(
                        symbol, timeframe, next_start, update_end_str if update_end else None, output_path
                    )
                    total_rows = len(existing_df) + new_rows
                    print(f"   ✅ Update complete: +{new_rows:,} new rows (total: {total_rows:,})")

                del existing_df  # Free memory
            else:
                # Fresh download
                print(f"No existing file, downloading from scratch...")
                total_rows = await self._download_klines_streaming(
                    symbol, timeframe, year_start_str, year_end_str, output_path
                )
                print(f"Year {year} complete: {total_rows:,} rows saved to {output_path.name}")

            if total_rows == 0:
                print(f"⚠️  No data for {year}")
                continue

            # Read metadata only (don't load full file)
            df_year_meta = pd.read_parquet(output_path, columns=['open_time'])
            print(f"📅 Date range: {df_year_meta['open_time'].min()} -> {df_year_meta['open_time'].max()}")

            all_dfs.append(pd.DataFrame({'rows': [total_rows]}))  # Just count

        # Summary
        if all_dfs:
            total_rows_all = sum(df['rows'].iloc[0] for df in all_dfs)
            print(f"\n{'='*70}")
            print(f"✅ DOWNLOAD COMPLETE!")
            print(f"   Total: {total_rows_all:,} rows across {len(all_dfs)} year(s)")
            print(f"   Files saved to: {output_dir}")
            print(f"{'='*70}")
            # Return empty df since data is already on disk
            return pd.DataFrame({'total_rows': [total_rows_all]})
        else:
            # Informative error message
            tf_info = {
                '1w': 'Weekly (1w) candles open on Monday',
                '1M': 'Monthly (1M) candles open on the 1st of the month',
            }
            hint = tf_info.get(timeframe, '')
            print(f"\n⚠️  No data downloaded for {timeframe}!")
            if hint:
                print(f"   💡 {hint}")
                print(f"   💡 Try expanding the date range (e.g., use --start with an earlier date)")
            return pd.DataFrame()

    # ========================================================================
    # UPDATE (INCREMENTAL)
    # ========================================================================

    async def update(
        self,
        symbol: str,
        timeframe: str,
        output_dir: str = "data/parquets",
        chunk_size: int = 50000
    ) -> int:
        """
        Update data (smart incremental) - Memory efficient

        Logic:
        1. Read the existing file (metadata only - last timestamp)
        2. Download from the last timestamp to today
        3. Chunk chunk append (memory efficient)
        4. Remove duplicates (final pass)

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            output_dir: Output directory
            chunk_size: Chunk size for memory efficiency

        Returns:
            Total rows count
        """
        print(f"\n{'='*70}")
        print(f"UPDATING: {symbol} {timeframe}")
        print(f"{'='*70}\n")

        # Determine which year files exist and get the latest one
        # Yeni format: data/parquets/{symbol}/
        output_dir_path = Path(output_dir)
        symbol_dir = output_dir_path / symbol

        # Fix Windows case-insensitive issue: 1m vs 1M
        timeframe_safe = "1MO" if timeframe == "1M" else timeframe

        # Find all year files for this symbol+timeframe
        pattern = f"{symbol}_{timeframe_safe}_*.parquet"
        existing_files = list(symbol_dir.glob(pattern)) if symbol_dir.exists() else []

        if existing_files:
            # Sort by year (newest first)
            existing_files.sort(reverse=True)
            latest_file = existing_files[0]
            year_from_filename = int(latest_file.stem.split('_')[-1])

            print(f"Found {len(existing_files)} existing file(s):")
            for f in existing_files:
                print(f"  - {f.name}")

            output_path = latest_file
        else:
            # No existing files, start from 2025
            year_from_filename = 2025
            output_path = self._get_output_path(symbol, timeframe, output_dir, year_from_filename)

        # Get last timestamp (without loading full file)
        if output_path.exists():
            # Read only last few rows to get last timestamp
            existing_df = pd.read_parquet(output_path)
            last_n = existing_df.tail(100)  # Son 100 row yeterli
            last_timestamp = last_n['open_time'].max()
            existing_rows = len(existing_df)

            print(f"Existing data: {existing_rows:,} rows")
            print(f"Last timestamp: {last_timestamp}")

            next_start = last_timestamp + timedelta(minutes=1)
            start_date = next_start.strftime("%Y-%m-%d")
            del last_n  # Memory free

            # Check if we need to create a new year file
            # If last_timestamp is in 2024 but now we're in 2025, create new file
            last_year = last_timestamp.year
            current_year = get_utc_now().year

            if last_year < current_year and next_start.year > last_year:
                print(f"\n[INFO] Year changed: {last_year} -> {next_start.year}")
                print(f"[INFO] Will create new file for year {next_start.year}")

                # Close current year file (already has data up to last_timestamp)
                print(f"[INFO] {output_path.name} is complete (ends at {last_timestamp})")

                # Create new year file path
                output_path = self._get_output_path(symbol, timeframe, output_dir, next_start.year)
                year_from_filename = next_start.year

                # Don't merge with previous year data
                existing_df = None
                existing_rows = 0

        else:
            print("No existing data, downloading from 2025-01-01")
            existing_df = None
            existing_rows = 0
            start_date = "2025-01-01"

        # Download new data
        new_df = await self._download_klines(symbol, timeframe, start_date, None)

        if len(new_df) == 0:
            print("No new data to download")
            return existing_rows

        print(f"New data: {len(new_df):,} rows")

        # Check if new data spans multiple years
        if len(new_df) > 0:
            new_start_year = new_df['open_time'].min().year
            new_end_year = new_df['open_time'].max().year

            if new_start_year != new_end_year:
                print(f"\n[INFO] New data spans multiple years: {new_start_year} to {new_end_year}")
                print(f"[INFO] Splitting by year...")

                # Split by year and save separately
                for year in range(new_start_year, new_end_year + 1):
                    df_year = new_df[new_df['open_time'].dt.year == year].copy()

                    if len(df_year) == 0:
                        continue

                    print(f"\n  Year {year}: {len(df_year):,} rows")

                    # Get or create year file
                    year_path = self._get_output_path(symbol, timeframe, output_dir, year)

                    # Load existing data for this year (if any)
                    if year_path.exists() and year == year_from_filename and existing_df is not None:
                        # CRITICAL: Normalize timezone before merge
                        if pd.api.types.is_datetime64_any_dtype(existing_df['open_time']):
                            if existing_df['open_time'].dt.tz is None:
                                existing_df['open_time'] = pd.to_datetime(existing_df['open_time'], utc=True)
                            else:
                                existing_df['open_time'] = existing_df['open_time'].dt.tz_convert('UTC')

                        if pd.api.types.is_datetime64_any_dtype(df_year['open_time']):
                            if df_year['open_time'].dt.tz is None:
                                df_year['open_time'] = pd.to_datetime(df_year['open_time'], utc=True)
                            else:
                                df_year['open_time'] = df_year['open_time'].dt.tz_convert('UTC')

                        # Merge with existing
                        df_merged = pd.concat([existing_df, df_year], ignore_index=True)
                        df_merged = df_merged.drop_duplicates(subset=['open_time'], keep='last')
                        df_merged = df_merged.sort_values('open_time').reset_index(drop=True)
                    else:
                        df_merged = df_year

                    # Save
                    self._save_to_parquet(df_merged, year_path)
                    print(f"  Saved: {year_path.name} ({len(df_merged):,} rows)")

                # Return total rows across all years
                total_rows = len(new_df)
                if existing_df is not None:
                    total_rows += len(existing_df)

                del existing_df
                del new_df
                return total_rows

        # Single year update (normal case)
        # Merge strategy
        if existing_df is not None:
            # CRITICAL: Normalize timezone before concat to avoid comparison errors
            # Ensure both dataframes have same timezone (UTC-aware)
            if pd.api.types.is_datetime64_any_dtype(existing_df['open_time']):
                if existing_df['open_time'].dt.tz is None:
                    existing_df['open_time'] = pd.to_datetime(existing_df['open_time'], utc=True)
                else:
                    existing_df['open_time'] = existing_df['open_time'].dt.tz_convert('UTC')

            if pd.api.types.is_datetime64_any_dtype(new_df['open_time']):
                if new_df['open_time'].dt.tz is None:
                    new_df['open_time'] = pd.to_datetime(new_df['open_time'], utc=True)
                else:
                    new_df['open_time'] = new_df['open_time'].dt.tz_convert('UTC')

            # Concat (memory spike here, but unavoidable for dedup)
            df = pd.concat([existing_df, new_df], ignore_index=True)
            del existing_df  # Free memory
            del new_df

            # Remove duplicates
            df = df.drop_duplicates(subset=['open_time'], keep='last')
            df = df.sort_values('open_time').reset_index(drop=True)
        else:
            df = new_df

        print(f"After merge: {len(df):,} rows")

        # Save (chunked if large)
        if len(df) > chunk_size:
            print(f"Large dataset, saving in chunks...")
            # Save with compression
            df.to_parquet(output_path, index=False, compression='snappy',
                         row_group_size=chunk_size)
        else:
            self._save_to_parquet(df, output_path)

        total_rows = len(df)
        del df  # Free memory

        return total_rows

    # ========================================================================
    # DOWNLOAD AND APPEND (UPDATE MODE)
    # ========================================================================

    async def _download_and_append(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str],
        output_path: Path
    ) -> int:
        """
        Download missing data and append to existing file

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date (usually last_timestamp + 1 minute)
            end_date: End date or None
            output_path: Existing parquet file

        Returns:
            Number of new rows downloaded
        """
        interval = TIMEFRAME_MAP[timeframe]

        # Parse start date (can be datetime string with time)
        # CRITICAL: Use UTC timezone to avoid local timezone conversion
        if ' ' in start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            start_dt = start_dt.replace(tzinfo=timezone.utc)  # Treat as UTC
            start_ts = int(start_dt.timestamp() * 1000)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_dt = start_dt.replace(tzinfo=timezone.utc)  # Treat as UTC
            start_ts = int(start_dt.timestamp() * 1000)

        if end_date:
            # Parse end date (can also be datetime string with time)
            # CRITICAL: Use UTC timezone to avoid local timezone conversion
            if ' ' in end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                end_dt = end_dt.replace(tzinfo=timezone.utc)  # Treat as UTC
                end_ts = int(end_dt.timestamp() * 1000)
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_dt = end_dt.replace(tzinfo=timezone.utc)  # Treat as UTC
                end_ts = int(end_dt.timestamp() * 1000)
        else:
            end_ts = int(get_utc_now().timestamp() * 1000)

        # Download new data
        batch_num = 0
        total_new_rows = 0
        accumulated_chunks = []
        accumulated_rows = 0
        current_ts = start_ts

        # DEBUG: Log request details
        if self.debug:
            print(f"\n   [DEBUG] _download_and_append:")
            print(f"      Symbol: {symbol}, Interval: {interval}")
            print(f"      Start: {start_date} (UTC) -> {start_ts} ms epoch")
            print(f"      End: {end_date} (UTC) -> {end_ts} ms epoch")
            print(f"      Range: {datetime.fromtimestamp(start_ts/1000, tz=timezone.utc)} -> {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc)}")

        while current_ts < end_ts:
            batch_num += 1

            # Get klines
            klines = await asyncio.to_thread(
                self.client.get_historical_klines,
                symbol=symbol,
                interval=interval,
                start_str=current_ts,
                end_str=end_ts,
                limit=self.batch_size
            )

            if not klines:
                # DEBUG: Log why no data returned
                if self.debug:
                    print(f"\n   [DEBUG] Binance returned NO DATA for:")
                    print(f"      Timestamp: {current_ts} ms epoch")
                    print(f"      Date: {datetime.fromtimestamp(current_ts/1000, tz=timezone.utc)}")
                    print(f"      This could indicate exchange downtime or missing data on Binance's side")
                break

            # Convert to DataFrame
            df_chunk = self._klines_to_dataframe(klines, symbol, timeframe)
            accumulated_chunks.append(df_chunk)
            accumulated_rows += len(df_chunk)
            total_new_rows += len(df_chunk)

            # Progress
            print(f"   Batch {batch_num}: +{len(df_chunk):,} rows (total new: {total_new_rows:,})", end='\r')

            # Write to disk when buffer is full
            if accumulated_rows >= self.write_chunk_size:
                df_to_write = pd.concat(accumulated_chunks, ignore_index=True)
                self._append_to_parquet(df_to_write, output_path)
                print(f"\n   💾 Checkpoint: {accumulated_rows:,} rows appended")
                accumulated_chunks.clear()
                accumulated_rows = 0

            # Update timestamp
            current_ts = klines[-1][0] + 1

            # Rate limit
            await asyncio.sleep(0.1)

        # Write remaining data
        if accumulated_chunks:
            df_to_write = pd.concat(accumulated_chunks, ignore_index=True)
            self._append_to_parquet(df_to_write, output_path)
            print(f"\n   💾 Final append: {accumulated_rows:,} rows")

        return total_new_rows

    # ========================================================================
    # DOWNLOAD KLINES (STREAMING - MEMORY EFFICIENT)
    # ========================================================================

    async def _download_klines_streaming(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str],
        output_path: Path
    ) -> int:
        """
        Download klines with STREAMING WRITE (memory efficient)

        Writes to disk every 5000 rows instead of accumulating in memory.
        If connection drops, can resume from last written position.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date string
            end_date: End date string or None
            output_path: Output parquet file path

        Returns:
            Total rows downloaded
        """
        interval = TIMEFRAME_MAP[timeframe]

        # Convert to timestamps
        # CRITICAL: Use UTC timezone to avoid local timezone conversion
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = int(start_dt.timestamp() * 1000)

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_ts = int(end_dt.timestamp() * 1000)
        else:
            end_ts = int(get_utc_now().timestamp() * 1000)

        # Streaming state
        current_ts = start_ts
        batch_num = 0
        total_rows = 0
        accumulated_chunks = []
        accumulated_rows = 0

        # NOTE: This function is for FRESH downloads only
        # Existing file check is done in download() function
        # If this function is called, we're doing a fresh download
        if output_path.exists():
            print(f"⚠️  Warning: Overwriting existing file {output_path.name}")
            output_path.unlink()

        # Download loop with streaming write
        while current_ts < end_ts:
            batch_num += 1

            # Get klines
            klines = await asyncio.to_thread(
                self.client.get_historical_klines,
                symbol=symbol,
                interval=interval,
                start_str=current_ts,
                end_str=end_ts,
                limit=self.batch_size
            )

            if not klines:
                break

            # Convert to DataFrame
            df_chunk = self._klines_to_dataframe(klines, symbol, timeframe)
            accumulated_chunks.append(df_chunk)
            accumulated_rows += len(df_chunk)
            total_rows += len(df_chunk)

            # Progress
            print(f"Batch {batch_num}: +{len(df_chunk):,} rows (total: {total_rows:,}, buffered: {accumulated_rows:,})", end='\r')

            # Write to disk when buffer is full (5000+ rows)
            if accumulated_rows >= self.write_chunk_size:
                df_to_write = pd.concat(accumulated_chunks, ignore_index=True)

                # Append to parquet
                self._append_to_parquet(df_to_write, output_path)

                print(f"\n💾 Checkpoint: {accumulated_rows:,} rows written to disk (total: {total_rows:,})")

                # Clear buffer
                accumulated_chunks.clear()
                accumulated_rows = 0

            # Update current timestamp
            current_ts = klines[-1][0] + 1  # Next candle

            # Avoid rate limit
            await asyncio.sleep(0.1)

        # Write remaining data
        if accumulated_chunks:
            df_to_write = pd.concat(accumulated_chunks, ignore_index=True)
            self._append_to_parquet(df_to_write, output_path)
            print(f"\n💾 Final write: {accumulated_rows:,} rows written to disk")

        print(f"\n✅ Downloaded: {batch_num} batches, {total_rows:,} total rows")

        return total_rows

    # ========================================================================
    # DOWNLOAD KLINES (LEGACY - KEPT FOR UPDATE FUNCTION)
    # ========================================================================

    async def _download_klines(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Download klines from Binance

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date string
            end_date: End date string or None

        Returns:
            DataFrame with klines
        """
        interval = TIMEFRAME_MAP[timeframe]

        # Convert to timestamps
        # CRITICAL: Use UTC timezone to avoid local timezone conversion
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = int(start_dt.timestamp() * 1000)

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_ts = int(end_dt.timestamp() * 1000)
        else:
            end_ts = int(get_utc_now().timestamp() * 1000)

        # Download in batches (STREAMING - memory efficient)
        chunks = []
        current_ts = start_ts
        batch_num = 0
        total_rows = 0

        while current_ts < end_ts:
            batch_num += 1

            # Get klines
            klines = await asyncio.to_thread(
                self.client.get_historical_klines,
                symbol=symbol,
                interval=interval,
                start_str=current_ts,
                end_str=end_ts,
                limit=self.batch_size
            )

            if not klines:
                break

            # Convert batch to DataFrame immediately (process in chunks)
            df_chunk = self._klines_to_dataframe(klines, symbol, timeframe)
            chunks.append(df_chunk)
            total_rows += len(df_chunk)

            # Progress (show accumulated total)
            print(f"Batch {batch_num}: +{len(df_chunk):,} rows (total: {total_rows:,})", end='\r')

            # Update current timestamp
            current_ts = klines[-1][0] + 1  # Next candle

            # Avoid rate limit
            await asyncio.sleep(0.1)

        print(f"\nDownloaded: {batch_num} batches, {total_rows:,} rows")

        if not chunks:
            return pd.DataFrame()

        # Concat all chunks at the end
        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory

        return df

    def _klines_to_dataframe(self, klines: list, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Convert klines list to DataFrame (helper function)

        Args:
            klines: Raw klines from Binance
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            DataFrame with processed klines
        """
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_volume', 'taker_buy_quote_volume']:
            df[col] = df[col].astype(float)

        df['trades'] = df['trades'].astype(int)

        # Drop ignore column
        df = df.drop(columns=['ignore'])

        # Add symbol and timeframe columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe

        return df

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    def _get_output_path(self, symbol: str, timeframe: str, output_dir: str, year: Optional[int] = None) -> Path:
        """
        Get output file path (year-based, symbol subfolder)

        Yeni format: data/parquets/{symbol}/{symbol}_{timeframe}_{year}.parquet

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            output_dir: Base output directory (e.g., data/parquets)
            year: Year (default: current year)

        Returns:
            Path to parquet file (e.g., data/parquets/BTCUSDT/BTCUSDT_1m_2024.parquet)
        """
        base_dir = Path(output_dir)
        # New format: subfolder based on symbol
        symbol_dir = base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Default to current year
        if year is None:
            year = get_utc_now().year

        # Fix Windows case-insensitive issue: 1m (minute) vs 1M (month)
        # Convert 1M → 1MO to avoid collision (only uppercase M for months)
        timeframe_safe = "1MO" if timeframe == "1M" else timeframe

        # Filename: BTCUSDT_1m_2024.parquet or BTCUSDT_1MO_2024.parquet
        filename = f"{symbol}_{timeframe_safe}_{year}.parquet"

        return symbol_dir / filename

    def _append_to_parquet(self, df: pd.DataFrame, filepath: Path):
        """
        Append DataFrame to parquet file (streaming write) with duplicate removal

        Uses pyarrow for efficient append operations.
        Creates file if doesn't exist, appends if exists.
        Automatically removes duplicates based on 'open_time' column.

        Args:
            df: DataFrame to append
            filepath: Parquet file path
        """
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not filepath.exists():
            # Create new file (no duplicates possible)
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, filepath, compression='snappy')
        else:
            # Append to existing file with duplicate removal
            # Read existing data
            existing_df = pd.read_parquet(filepath)

            # Normalize timezone awareness for both dataframes
            # Ensure both are timezone-aware (UTC) for comparison
            if pd.api.types.is_datetime64_any_dtype(existing_df['open_time']):
                if existing_df['open_time'].dt.tz is None:
                    existing_df['open_time'] = pd.to_datetime(existing_df['open_time'], utc=True)
                else:
                    existing_df['open_time'] = existing_df['open_time'].dt.tz_convert('UTC')

            if pd.api.types.is_datetime64_any_dtype(df['open_time']):
                if df['open_time'].dt.tz is None:
                    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
                else:
                    df['open_time'] = df['open_time'].dt.tz_convert('UTC')

            # Concat new data
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates (keep last - newest data wins)
            combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='last')

            # Sort by timestamp
            combined_df = combined_df.sort_values('open_time').reset_index(drop=True)

            # Write back
            table = pa.Table.from_pandas(combined_df, preserve_index=False)
            pq.write_table(table, filepath, compression='snappy')

            # Memory cleanup
            del existing_df
            del combined_df

    def _save_to_parquet(self, df: pd.DataFrame, filepath: Path):
        """Save DataFrame to parquet with symbol and timeframe columns (legacy)"""
        # Parse symbol and timeframe from filename
        # Format: BTCUSDT_1m_2025.parquet
        filename = filepath.stem  # Remove .parquet extension
        parts = filename.split('_')

        if len(parts) >= 3:
            symbol = parts[0]  # BTCUSDT
            timeframe = parts[1]  # 1m

            # Add symbol and timeframe columns if not exists
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            if 'timeframe' not in df.columns:
                df['timeframe'] = timeframe

        df.to_parquet(filepath, index=False, compression='snappy')
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"💾 Saved: {filepath.name} ({size_mb:.2f} MB)")

    # ========================================================================
    # BATCH DOWNLOAD
    # ========================================================================

    async def download_multiple(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: str,
        output_dir: str = "data/parquets"
    ):
        """
        Download multiple symbols/timeframes

        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            start_date: Start date
            output_dir: Output directory
        """
        total = len(symbols) * len(timeframes)
        current = 0

        print(f"\n{'='*70}")
        print(f"BATCH DOWNLOAD: {len(symbols)} symbols x {len(timeframes)} timeframes = {total} tasks")
        print(f"{'='*70}")

        for symbol in symbols:
            for timeframe in timeframes:
                current += 1
                print(f"\n[{current}/{total}] {symbol} {timeframe}")

                try:
                    await self.download(symbol, timeframe, start_date, None, output_dir)
                except Exception as e:
                    print(f"ERROR: {e}")

                # Rate limit
                await asyncio.sleep(0.5)

        print(f"\n{'='*70}")
        print(f"BATCH DOWNLOAD COMPLETED: {current}/{total}")
        print(f"{'='*70}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    """
    CLI Interface for Data Downloader

    SMART DOWNLOAD:
    - If the file does not exist -> Fresh download
    - If the file exists -> Gap detection + fill missing data
    - If the last timestamp is older than today -> Update to today

    Usage:
        # Single symbol/timeframe
        python data_downloader.py BTCUSDT 1m --start 2025-01-01

        # Multiple symbols
        python data_downloader.py --symbols BTCUSDT,ETHUSDT --timeframe 1m --start 2025-01-01

        # Multiple timeframes
        python data_downloader.py BTCUSDT --timeframes 1m,5m,15m --start 2025-01-01

        # Multiple symbols AND timeframes
        python data_downloader.py --symbols BTCUSDT,ETHUSDT --timeframes 1m,5m,15m --start 2025-01-01

        # Custom end date
        python data_downloader.py BTCUSDT 1m --start 2024-01-01 --end 2024-12-31
    """
    import sys
    import argparse

    # Windows UTF-8 fix
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(
        description='SuperBot Data Downloader - Smart historical data downloader with gap detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download BTCUSDT 1m from 2025-01-01 to now
  python data_downloader.py BTCUSDT 1m --start 2025-01-01

  # Multiple symbols
  python data_downloader.py --symbols BTCUSDT,ETHUSDT --timeframe 1m --start 2025-01-01

  # Multiple timeframes
  python data_downloader.py BTCUSDT --timeframes 1m,5m,15m,30m,1h,4h --start 2025-01-01

  # Multiple symbols AND timeframes
  python data_downloader.py --symbols BTC,ETH --timeframes 1m,5m --start 2025-01-01

  # Custom date range
  python data_downloader.py BTCUSDT 1m --start 2024-01-01 --end 2024-12-31

Smart Features:
  - Auto-detects existing files
  - Fills gaps in historical data
  - Updates to latest data
  - Streaming write (5000-row chunks)
  - Resume on connection failure
        '''
    )

    # Positional arguments (optional - can use --symbols/--timeframes instead)
    parser.add_argument('symbol', type=str, nargs='?', help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('timeframe', type=str, nargs='?', help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d)')

    # Optional arguments
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols (e.g., BTCUSDT or BTCUSDT ETHUSDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', help='Timeframes (e.g., 1m or 1m 5m 15m or "1m,5m,15m")')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD) or None for today')
    parser.add_argument('--output', type=str, default='data/parquets', help='Output directory (default: data/parquets)')

    args = parser.parse_args()

    # Parse symbols and timeframes
    symbols = []
    timeframes = []

    # From positional args
    if args.symbol:
        symbols.append(args.symbol.upper())
    if args.timeframe:
        timeframes.append(args.timeframe)

    # From --symbols/--timeframes (override positional)
    if args.symbols:
        # Support both: "1m,5m" and 1m 5m
        if isinstance(args.symbols, list):
            # PowerShell array: ['1m', '5m', '15m']
            symbols_flat = []
            for item in args.symbols:
                if ',' in item:
                    # "1m,5m" format
                    symbols_flat.extend([s.strip().upper() for s in item.split(',')])
                else:
                    # "1m" format
                    symbols_flat.append(item.strip().upper())
            symbols = symbols_flat
        else:
            # Single string: "1m,5m,15m"
            symbols = [s.strip().upper() for s in args.symbols.split(',')]

    if args.timeframes:
        # Support both: "1m,5m" and 1m 5m
        if isinstance(args.timeframes, list):
            # PowerShell array: ['1m', '5m', '15m']
            timeframes_flat = []
            for item in args.timeframes:
                if ',' in item:
                    # "1m,5m" format
                    timeframes_flat.extend([t.strip() for t in item.split(',')])
                else:
                    # "1m" format
                    timeframes_flat.append(item.strip())
            timeframes = timeframes_flat
        else:
            # Single string: "1m,5m,15m"
            timeframes = [t.strip() for t in args.timeframes.split(',')]

    # Validation
    if not symbols:
        parser.error("Please provide symbol(s): BTCUSDT or --symbols BTCUSDT,ETHUSDT")
    if not timeframes:
        parser.error("Please provide timeframe(s): 1m or --timeframes 1m,5m,15m")

    # Execute smart download
    async def main():
        downloader = DataDownloader()

        total = len(symbols) * len(timeframes)
        current = 0

        print(f"\n{'='*70}")
        print(f"SMART DOWNLOAD")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Date range: {args.start} -> {args.end or 'now'}")
        print(f"Output: {args.output}")
        print(f"Total tasks: {total}")
        print(f"{'='*70}\n")

        for symbol in symbols:
            for timeframe in timeframes:
                current += 1
                print(f"\n{'-'*70}")
                print(f"[{current}/{total}] {symbol} {timeframe}")
                print(f"{'-'*70}")

                try:
                    # SMART DOWNLOAD: Auto-detects existing files and handles gaps
                    await downloader.download(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=args.start,
                        end_date=args.end,
                        output_dir=args.output
                    )
                except Exception as e:
                    print(f"\nERROR: {e}\n")
                    import traceback
                    traceback.print_exc()

                # Rate limit between tasks
                if current < total:
                    await asyncio.sleep(0.5)

        print(f"\n{'='*70}")
        print(f"ALL DONE! {current}/{total} tasks completed")
        print(f"Files saved to: {args.output}")
        print(f"{'='*70}\n")

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
