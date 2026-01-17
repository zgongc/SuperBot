#!/usr/bin/env python3
"""
components/data/timeframe_resampler.py
SuperBot - Timeframe Resampler

Purpose:
    Resample lower timeframe data to higher timeframes.
    Used when requested timeframe data doesn't exist.

Features:
    - Smart source selection (closest lower timeframe)
    - OHLCV aggregation (pandas resample)
    - File naming with _re suffix (_re1h, _re15m, etc.)
    - Volume summation
    - Validation

Usage:
    from components.data.timeframe_resampler import TimeframeResampler

    resampler = TimeframeResampler(data_dir='data/parquets')

    # Resample 1h → 2h
    df_2h = resampler.resample(
        symbol='BTCUSDT',
        target_tf='2h',
        year=2025
    )
    # → Saves to: BTCUSDT_2h_2025_re1h.parquet

Dependencies:
    - pandas>=2.0.0
    - pyarrow (parquet)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from core.logger_engine import LoggerEngine

# LoggerEngine setup
logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


# ============================================================================
# TIMEFRAME MAPPING
# ============================================================================

# Binance supported timeframes
BINANCE_TIMEFRAMES = [
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
]

# Resample hierarchy: target → [sources in priority order]
# Priority: Use closest lower timeframe for better accuracy
RESAMPLE_HIERARCHY: Dict[str, List[str]] = {
    # Minutes
    '3m':  ['1m'],
    '45m': ['15m', '5m', '1m'],

    # Hours
    '2h':  ['1h', '30m', '15m'],
    '3h':  ['1h', '30m'],
    '6h':  ['2h', '1h', '30m'],
    '8h':  ['4h', '2h', '1h'],
    '12h': ['4h', '2h', '6h', '1h'],

    # Days
    '1d':  ['12h', '8h', '6h', '4h', '2h', '1h'],
    '3d':  ['1d', '12h'],
    '1w':  ['1d', '3d'],
}

# Pandas resample rules (using 'min' instead of deprecated 'T')
TIMEFRAME_TO_RULE = {
    '1m': '1min',   '3m': '3min',   '5m': '5min',   '15m': '15min',  '30m': '30min',  '45m': '45min',
    '1h': '1H',   '2h': '2H',   '3h': '3H',   '4h': '4H',    '6h': '6H',    '8h': '8H',   '12h': '12H',
    '1d': '1D',   '3d': '3D',   '1w': '1W',   '1M': '1M',
}


# ============================================================================
# TIMEFRAME RESAMPLER
# ============================================================================

class TimeframeResampler:
    """
    Timeframe Resampler

    Resample lower timeframe OHLCV data to higher timeframes.

    Attributes:
        data_dir: Directory containing parquet files
    """

    def __init__(self, data_dir: str = 'data/parquets'):
        """
        Initialize resampler

        Args:
            data_dir: Path to parquet files directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TimeframeResampler initialized: {self.data_dir}")

    def find_source_file(
        self,
        symbol: str,
        target_tf: str,
        year: int
    ) -> Optional[tuple]:
        """
        Find best source file for resampling

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            target_tf: Target timeframe (e.g., '2h')
            year: Year

        Returns:
            (source_tf, file_path) or None if not found
        """
        # Check if target exists in Binance (shouldn't resample these)
        if target_tf in BINANCE_TIMEFRAMES:
            logger.warning(f"⚠️  {target_tf} is a Binance timeframe - download instead of resample!")

        # Get possible source timeframes
        sources = RESAMPLE_HIERARCHY.get(target_tf, [])

        if not sources:
            logger.error(f"❌ No resample hierarchy defined for {target_tf}")
            return None

        # Try each source in priority order
        for source_tf in sources:
            # Check for original file
            file_path = self.data_dir / f"{symbol}_{source_tf}_{year}.parquet"
            if file_path.exists():
                logger.info(f"✅ Found source: {file_path.name}")
                return (source_tf, file_path)

            # Check for already-resampled file
            file_path_re = self.data_dir / f"{symbol}_{source_tf}_{year}_re*.parquet"
            matches = list(self.data_dir.glob(f"{symbol}_{source_tf}_{year}_re*.parquet"))
            if matches:
                logger.info(f"✅ Found resampled source: {matches[0].name}")
                return (source_tf, matches[0])

        logger.error(f"❌ No source file found for {symbol} {target_tf} {year}")
        logger.error(f"   Tried: {sources}")
        return None

    def resample_dataframe(
        self,
        df: pd.DataFrame,
        target_tf: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV DataFrame to target timeframe

        Args:
            df: Source DataFrame with OHLCV columns
            target_tf: Target timeframe (e.g., '2h')

        Returns:
            Resampled DataFrame
        """
        # Handle different timestamp column names
        time_col = None
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'open_time' in df.columns:
            time_col = 'open_time'
        else:
            raise ValueError("DataFrame must have 'timestamp' or 'open_time' column")

        # Get pandas resample rule
        rule = TIMEFRAME_TO_RULE.get(target_tf)
        if not rule:
            raise ValueError(f"Unknown timeframe: {target_tf}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], unit='ms')

        # Set timestamp as index
        df = df.set_index(time_col)

        # Resample
        df_resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Reset index
        df_resampled = df_resampled.reset_index()

        # Rename back to original column name
        df_resampled = df_resampled.rename(columns={time_col: 'open_time' if time_col == 'open_time' else 'timestamp'})

        logger.info(f"✅ Resampled: {len(df)} → {len(df_resampled)} bars ({target_tf})")

        return df_resampled

    def resample(
        self,
        symbol: str,
        target_tf: str,
        year: int,
        force: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Resample data to target timeframe

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            target_tf: Target timeframe (e.g., '2h', '45m')
            year: Year
            force: Force resample even if target file exists

        Returns:
            Resampled DataFrame or None
        """
        logger.info("=" * 70)
        logger.info(f"🔄 RESAMPLE: {symbol} {target_tf} ({year})")
        logger.info("=" * 70)

        # Check if resampled file already exists
        existing_files = list(self.data_dir.glob(f"{symbol}_{target_tf}_{year}*.parquet"))
        if existing_files and not force:
            logger.info(f"✅ Resampled file exists: {existing_files[0].name}")
            return pd.read_parquet(existing_files[0])

        # Find source file
        result = self.find_source_file(symbol, target_tf, year)
        if not result:
            return None

        source_tf, source_file = result

        # Load source data
        logger.info(f"📂 Loading: {source_file.name}")
        df_source = pd.read_parquet(source_file)
        logger.info(f"   Rows: {len(df_source)}")

        # Resample
        df_resampled = self.resample_dataframe(df_source, target_tf)

        # Save with _re suffix
        output_file = self.data_dir / f"{symbol}_{target_tf}_{year}_re{source_tf}.parquet"
        df_resampled.to_parquet(output_file, index=False)
        logger.info(f"💾 Saved: {output_file.name}")

        # Summary
        time_col = 'open_time' if 'open_time' in df_resampled.columns else 'timestamp'
        logger.info("")
        logger.info("📊 Summary:")
        logger.info(f"   Source: {source_tf} ({len(df_source)} bars)")
        logger.info(f"   Target: {target_tf} ({len(df_resampled)} bars)")
        logger.info(f"   Ratio: {len(df_source) / len(df_resampled):.2f}x")
        logger.info(f"   Date range: {df_resampled[time_col].min()} → {df_resampled[time_col].max()}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ RESAMPLE COMPLETE")
        logger.info("=" * 70)

        return df_resampled


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resample timeframe data')
    parser.add_argument('symbol', help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('target_tf', help='Target timeframe (e.g., 2h, 45m)')
    parser.add_argument('year', type=int, help='Year (e.g., 2025)')
    parser.add_argument('--data-dir', default='data/parquets', help='Data directory')
    parser.add_argument('--force', action='store_true', help='Force resample')

    args = parser.parse_args()

    resampler = TimeframeResampler(data_dir=args.data_dir)
    df = resampler.resample(
        symbol=args.symbol,
        target_tf=args.target_tf,
        year=args.year,
        force=args.force
    )

    if df is not None:
        print(f"\nSuccess: {len(df)} bars resampled")
    else:
        print("\nFailed")
