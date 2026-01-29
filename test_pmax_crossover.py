#!/usr/bin/env python3
"""
test_pmax_crossover.py
SuperBot - PMax Crossover Analysis Test
Author: SuperBot Team
Date: 2026-01-28

Test to analyze PMax crossover detection timing differences
between TradingView and our backtest.

Purpose:
- Load real BTCUSDT 15m data from January 3, 2026
- Calculate PMax indicator values
- Detect crossover signals
- Compare with TradingView chart
- Find why trade opened at 17:30 instead of 22:30
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timezone

from core.logger_engine import get_logger
from core.config_engine import get_config
from components.indicators import INDICATOR_REGISTRY, get_indicator_class
from components.managers.parquets_engine import ParquetsEngine

logger = get_logger(__name__)


def format_timestamp(ts: int) -> str:
    """Convert timestamp to readable format"""
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def detect_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossover: series1 crosses above series2

    Returns:
        pd.Series: Boolean series where True = crossover occurred
    """
    # series1 was below series2, now above
    prev_below = series1.shift(1) <= series2.shift(1)
    curr_above = series1 > series2
    return prev_below & curr_above


def detect_crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossunder: series1 crosses below series2

    Returns:
        pd.Series: Boolean series where True = crossunder occurred
    """
    # series1 was above series2, now below
    prev_above = series1.shift(1) >= series2.shift(1)
    curr_below = series1 < series2
    return prev_above & curr_below


async def main():
    logger.info("="*80)
    logger.info("🔍 PMAX CROSSOVER ANALYSIS TEST")
    logger.info("="*80)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    logger.info("\n📂 Loading BTCUSDT 15m data from January 3, 2026...")

    # Get config and create ParquetsEngine
    config = get_config()
    parquets_engine = ParquetsEngine(
        config_engine=config,
        logger_engine=None  # Will use internal logger
    )

    # Load data for Jan 2-4, 2026 (include buffer for warmup)
    start_date = "2026-01-02T00:00:00"
    end_date = "2026-01-04T23:59:59"

    data = await parquets_engine.get_historical_data(
        symbol="BTCUSDT",
        timeframe="15m",
        start_date=start_date,
        end_date=end_date,
        warmup_candles=200,
        utc_offset=3
    )

    if data is None or len(data) == 0:
        logger.error("❌ Failed to load data!")
        return

    logger.info(f"✅ Loaded {len(data)} candles")
    logger.info(f"   From: {format_timestamp(data.iloc[0]['timestamp'])}")
    logger.info(f"   To:   {format_timestamp(data.iloc[-1]['timestamp'])}")

    # ========================================================================
    # 2. CALCULATE INDICATORS
    # ========================================================================
    logger.info("\n📊 Calculating indicators...")

    # Get indicator classes from registry
    EMA = get_indicator_class('ema')
    PMax = get_indicator_class('pmax')

    # EMA 200
    ema200 = EMA(period=200)
    ema200_series = ema200.calculate_batch(data)

    # PMax (TradingView params - multiplier 3.0!)
    pmax = PMax(atr_period=10, atr_multiplier=3.0, ma_period=10)
    pmax_result = pmax.calculate_batch(data)

    # Add to dataframe
    data['ema200'] = ema200_series
    data['pmax_mavg'] = pmax_result['mavg']
    data['pmax_pmax'] = pmax_result['pmax']
    data['pmax_trend'] = pmax_result['trend_direction']
    data['pmax_long_stop'] = pmax_result['long_stop']
    data['pmax_short_stop'] = pmax_result['short_stop']

    logger.info("✅ Indicators calculated")

    # ========================================================================
    # 3. DETECT CROSSOVERS
    # ========================================================================
    logger.info("\n🔄 Detecting crossovers...")

    # Detect crossover: mavg crosses above pmax
    data['long_crossover'] = detect_crossover(data['pmax_mavg'], data['pmax_pmax'])

    # Detect crossunder: mavg crosses below pmax
    data['short_crossunder'] = detect_crossunder(data['pmax_mavg'], data['pmax_pmax'])

    # Filter only January 3, 2026
    jan3_data = data[
        (data['timestamp'] >= pd.Timestamp('2026-01-03 00:00:00').timestamp() * 1000) &
        (data['timestamp'] < pd.Timestamp('2026-01-04 00:00:00').timestamp() * 1000)
    ].copy()

    logger.info(f"✅ Filtered to January 3: {len(jan3_data)} candles")

    # ========================================================================
    # 4. FIND CROSSOVERS ON JANUARY 3
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("📈 LONG CROSSOVERS ON JANUARY 3, 2026")
    logger.info("="*80)

    long_signals = jan3_data[jan3_data['long_crossover'] == True]

    if len(long_signals) == 0:
        logger.info("❌ No LONG crossovers found!")
    else:
        for idx, row in long_signals.iterrows():
            time_str = format_timestamp(row['timestamp'])
            logger.info(f"\n🟢 LONG CROSSOVER at {time_str}")
            logger.info(f"   Close: ${row['close']:.2f}")
            logger.info(f"   EMA200: ${row['ema200']:.2f}")
            logger.info(f"   PMax MAVG: ${row['pmax_mavg']:.2f}")
            logger.info(f"   PMax Value: ${row['pmax_pmax']:.2f}")
            logger.info(f"   Trend: {int(row['pmax_trend'])}")

            # Check EMA200 condition
            if row['close'] > row['ema200']:
                logger.info(f"   ✅ Close > EMA200: LONG entry condition MET")
            else:
                logger.info(f"   ❌ Close < EMA200: LONG entry condition NOT met")

    # ========================================================================
    # 5. ANALYZE TREND CHANGES
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("🔄 PMAX TREND CHANGES ON JANUARY 3, 2026")
    logger.info("="*80)

    # Detect trend changes
    jan3_data['trend_change'] = jan3_data['pmax_trend'] != jan3_data['pmax_trend'].shift(1)
    trend_changes = jan3_data[jan3_data['trend_change'] == True]

    for idx, row in trend_changes.iterrows():
        time_str = format_timestamp(row['timestamp'])
        prev_trend = jan3_data.loc[jan3_data.index < idx, 'pmax_trend'].iloc[-1] if idx > jan3_data.index[0] else 0
        curr_trend = int(row['pmax_trend'])

        trend_str = "UP (1)" if curr_trend == 1 else "DOWN (-1)" if curr_trend == -1 else "NEUTRAL (0)"
        prev_trend_str = "UP (1)" if prev_trend == 1 else "DOWN (-1)" if prev_trend == -1 else "NEUTRAL (0)"

        logger.info(f"\n🔄 TREND CHANGE at {time_str}")
        logger.info(f"   From: {prev_trend_str} → To: {trend_str}")
        logger.info(f"   Close: ${row['close']:.2f}")
        logger.info(f"   PMax MAVG: ${row['pmax_mavg']:.2f}")
        logger.info(f"   PMax Value: ${row['pmax_pmax']:.2f}")
        logger.info(f"   Long Stop: ${row['pmax_long_stop']:.2f}")
        logger.info(f"   Short Stop: ${row['pmax_short_stop']:.2f}")

    # ========================================================================
    # 6. DETAILED ANALYSIS AROUND 17:30 AND 22:30
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("🔍 DETAILED ANALYSIS: 17:00 - 23:00")
    logger.info("="*80)

    # Filter 17:00 - 23:00
    analysis_data = jan3_data[
        (jan3_data['timestamp'] >= pd.Timestamp('2026-01-03 17:00:00').timestamp() * 1000) &
        (jan3_data['timestamp'] <= pd.Timestamp('2026-01-03 23:00:00').timestamp() * 1000)
    ].copy()

    for idx, row in analysis_data.iterrows():
        time_str = format_timestamp(row['timestamp'])

        # Previous row for comparison
        prev_idx = analysis_data.index.get_loc(idx) - 1
        if prev_idx >= 0:
            prev_row = analysis_data.iloc[prev_idx]
            prev_mavg = prev_row['pmax_mavg']
            prev_pmax = prev_row['pmax_pmax']
        else:
            prev_mavg = row['pmax_mavg']
            prev_pmax = row['pmax_pmax']

        logger.info(f"\n⏰ {time_str}")
        logger.info(f"   Close: ${row['close']:.2f} | EMA200: ${row['ema200']:.2f}")
        logger.info(f"   MAVG: ${row['pmax_mavg']:.2f} (prev: ${prev_mavg:.2f})")
        logger.info(f"   PMax: ${row['pmax_pmax']:.2f} (prev: ${prev_pmax:.2f})")
        logger.info(f"   Trend: {int(row['pmax_trend'])}")

        # Crossover detection
        if row['long_crossover']:
            logger.info(f"   🟢 LONG CROSSOVER DETECTED!")
        if row['short_crossunder']:
            logger.info(f"   🔴 SHORT CROSSUNDER DETECTED!")

        # Relationship
        if row['pmax_mavg'] > row['pmax_pmax']:
            logger.info(f"   📊 MAVG > PMax (bullish)")
        else:
            logger.info(f"   📊 MAVG < PMax (bearish)")

    # ========================================================================
    # 7. SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)

    logger.info(f"\n📈 Total LONG crossovers on Jan 3: {len(long_signals)}")
    if len(long_signals) > 0:
        logger.info(f"   Times: {[format_timestamp(t) for t in long_signals['timestamp'].tolist()]}")

    logger.info(f"\n🔄 Total trend changes on Jan 3: {len(trend_changes)}")

    # ========================================================================
    # 8. EXPORT TO CSV
    # ========================================================================
    logger.info("\n💾 Exporting data to CSV...")

    # Export January 3 data
    export_file = project_root / "test_pmax_jan3_analysis.csv"

    jan3_export = jan3_data[[
        'timestamp', 'open', 'high', 'low', 'close',
        'ema200', 'pmax_mavg', 'pmax_pmax', 'pmax_trend',
        'pmax_long_stop', 'pmax_short_stop',
        'long_crossover', 'short_crossunder'
    ]].copy()

    # Add readable time column
    jan3_export['time'] = jan3_export['timestamp'].apply(format_timestamp)

    jan3_export.to_csv(export_file, index=False)
    logger.info(f"✅ Exported to: {export_file}")

    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
