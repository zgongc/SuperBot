#!/usr/bin/env python3
"""
fix_year_split.py
SuperBot - Year-based File Split Fixer
Yazar: SuperBot Team
Tarih: 2025-01-06
Versiyon: 2.0.0

Year-based files are cleaned and year boundaries are forced.

Sorun:
The file from 2024 contains data from 2025 (UTC+3 timezone error)
The file from 2025 contains data from 2024.
Duplicate data exists
Mixed year boundaries

Solution:
Read each file.
Check year boundaries based on epoch time (time zone safe)
Only store data for the relevant year.
- Duplicate'leri temizle
align

Usage:
    python fix_year_split.py

Dependencies:
    - python>=3.10
    - pandas>=2.0
    - pyarrow>=10.0
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import sys


def fix_year_split(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1m',
    data_dir: str = 'data/parquets'
):
    """
    Fix year-based file split

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        data_dir: Data directory (base dir, will use {data_dir}/{symbol}/)
    """
    print("=" * 70)
    print(f"🔧 YEAR-SPLIT FIXER: {symbol} {timeframe}")
    print("=" * 70)

    # Yeni format: data/parquets/{symbol}/
    data_path = Path(data_dir) / symbol

    if not data_path.exists():
        print(f"❌ Symbol dictionary not found: {data_path}")
        return

    # Find all year files
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = sorted(data_path.glob(pattern))

    if not files:
        print(f"❌ No files found matching: {pattern}")
        return

    print(f"\n📂 Found {len(files)} file(s):")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)

    # Process each file
    for filepath in files:
        # Extract year from filename: BTCUSDT_1m_2024.parquet
        year_str = filepath.stem.split('_')[-1]

        try:
            year = int(year_str)
        except ValueError:
            print(f"⚠️  Skipping {filepath.name} - invalid year: {year_str}")
            continue

        print(f"\n📅 Processing: {filepath.name} (Year {year})")
        print("-" * 70)

        # 1. Load file
        df = pd.read_parquet(filepath)
        initial_rows = len(df)
        print(f"   Initial: {initial_rows:,} rows")

        # Get time column
        time_col = 'open_time' if 'open_time' in df.columns else 'timestamp'

        # Show date range
        date_min = df[time_col].min()
        date_max = df[time_col].max()
        print(f"   📅 Date range: {date_min} -> {date_max}")

        # 2. Year filter based on epoch time (timezone-safe)
        Binance epoch time is used, no timezone issue.
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

        Year boundaries based on epoch time (UTC)
        # 2024-01-01 00:00:00 UTC = 1704067200000 ms epoch
        # 2024-12-31 23:59:59 UTC = 1735689599000 ms epoch
        year_start_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        year_end_utc = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Convert to pandas Timestamp for comparison
        year_start_ts = pd.Timestamp(year_start_utc)
        year_end_ts = pd.Timestamp(year_end_utc)

        # Filter: only keep data within this year (UTC epoch time based)
        df_year = df[
            (df[time_col] >= year_start_ts) &
            (df[time_col] <= year_end_ts)
        ].copy()

        year_rows = len(df_year)

        rows_removed = initial_rows - year_rows
        if rows_removed > 0:
            print(f"   🧹 Filtered: {rows_removed:,} row removed from other years")
        else:
            print(f"   ✔️ All lines are for the year {year}")

        # 3. Duplicate'leri temizle
        df_year = df_year.drop_duplicates(subset=[time_col], keep='last')
        df_year = df_year.sort_values(time_col).reset_index(drop=True)

        after_dedup = len(df_year)
        duplicates = year_rows - after_dedup

        if duplicates > 0:
            print(f"   🧹 Cleaned: {duplicates:,} duplicate row")

        # Check expected date range (UTC epoch based)
        expected_start = year_start_ts
        expected_end = pd.Timestamp(datetime(year, 12, 31, 23, 59, 0, tzinfo=timezone.utc))

        actual_start = df_year[time_col].min()
        actual_end = df_year[time_col].max()

        print(f"\n   Beklenen (UTC): {expected_start} -> {expected_end}")
        print(f"   Real (UTC):   {actual_start} -> {actual_end}")

        # Insufficient data control (start too early or end too late)
        needs_download = False

        if actual_start > expected_start:
            missing_days = (actual_start - expected_start).days
            if missing_days > 0:
                print(f"   ⚠️  Missing: Starting with {missing_days} days missing")
                needs_download = True

        # Only for past years, control the last date (not current year)
        current_year_utc = datetime.now(timezone.utc).year
        if year < current_year_utc and actual_end < expected_end:
            missing_days = (expected_end - actual_end).days
            if missing_days > 0:
                print(f"   ⚠️  Missing: Last {missing_days} days")
                needs_download = True

        if needs_download:
            print(f"   💡 Hint: Fill gaps with data_downloader to run")

        # 5. Cleaned file to save
        print(f"\n   Cleaned file saved...")
        df_year.to_parquet(filepath, index=False, compression='snappy')

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   ✔️ Recorded: {after_dedup:,} rows ({size_mb:.2f} MB)")

        # Memory cleanup
        del df
        del df_year

    print("\n" + "=" * 70)
    YIL-SPLIT ADJUSTMENT COMPLETE! 🎉
    print("=" * 70)

    # Final summary
    "Print:\n\n📊 Last File Summary:"
    for filepath in files:
        df_check = pd.read_parquet(filepath)
        time_col = 'open_time' if 'open_time' in df_check.columns else 'timestamp'
        date_min = df_check[time_col].min()
        date_max = df_check[time_col].max()
        size_mb = filepath.stat().st_size / (1024 * 1024)

        print(f"\n   {filepath.name}:")
        print(f"      Row: {len(df_check):,}")
        print(f"    December: {date_min} -> {date_max}")
        print(f"      Boyut: {size_mb:.2f} MB")

        del df_check

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Windows UTF-8 fix (for emoji display)
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            # Fallback for older Python or IDLE
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    fix_year_split(
        symbol='BTCUSDT',
        timeframe='1m',
        data_dir='data/parquets'
    )
