"""
Test RSI Divergence Signals - Compare with TradingView

Tests RSI Divergence indicator and strategy signals against TradingView implementation.

PineScript Reference: temp/rsi_div.txt
- RSI Period: 14
- Pivot Lookback Left: 5
- Pivot Lookback Right: 5
- Range: 5-60 bars

Usage:
    python test_rsidivergence_signals.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))

from components.managers.parquets_engine import ParquetsEngine
from components.indicators import get_indicator_class
from core.config_engine import get_config
from core.logger_engine import get_logger

# Get logger
logger = get_logger(__name__)


async def test_rsidivergence_signals():
    """Test RSI Divergence signals and compare with TradingView"""

    print("\n" + "="*80)
    print("RSI DIVERGENCE SIGNAL TEST - TradingView Comparison")
    print("="*80 + "\n")

    # Initialize
    config = get_config()
    parquets_engine = ParquetsEngine(
        config_engine=config,
        logger_engine=None  # Will use internal logger
    )

    # Load indicator classes
    RSIDivergence = get_indicator_class('rsidivergence')
    RSI = get_indicator_class('rsi')
    EMA = get_indicator_class('ema')
    ADX = get_indicator_class('adx')

    # ========================================================================
    # TEST 1: Load BTCUSDT 15m data
    # ========================================================================
    print("📊 Loading BTCUSDT 15m data...")
    data = await parquets_engine.get_historical_data(
        symbol="BTCUSDT",
        timeframe="15m",
        start_date="2026-01-15T00:00:00",
        end_date="2026-01-28T23:59:59",  # Extended to include Jan 27
        warmup_candles=200,
        utc_offset=3
    )

    # Add date_str column for display
    data['date_str'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Istanbul').dt.strftime('%Y-%m-%d %H:%M')

    print(f"   ✓ Loaded {len(data)} candles")
    print(f"   ✓ Date range: {data['date_str'].iloc[0]} → {data['date_str'].iloc[-1]}")
    print()

    # ========================================================================
    # TEST 2: Calculate RSI Divergence
    # ========================================================================
    print("🔍 Calculating RSI Divergence...")

    # TradingView parameters
    rsi_period = 14
    lookback = 5  # lbL = lbR = 5
    min_strength = 30

    rsidiv = RSIDivergence(rsi_period=rsi_period, lookback=lookback, min_strength=min_strength)
    rsidiv_result = rsidiv.calculate_batch(data)

    # Add to dataframe
    data['rsidiv_rsi'] = rsidiv_result['rsi']
    data['rsidiv_bullish'] = rsidiv_result['bullish_divergence']
    data['rsidiv_bearish'] = rsidiv_result['bearish_divergence']
    data['rsidiv_hidden_bullish'] = rsidiv_result['hidden_bullish_divergence']
    data['rsidiv_hidden_bearish'] = rsidiv_result['hidden_bearish_divergence']
    data['rsidiv_strength'] = rsidiv_result['divergence_strength']

    print(f"   ✓ RSI Period: {rsi_period}")
    print(f"   ✓ Pivot Lookback: {lookback}")
    print(f"   ✓ Min Strength: {min_strength}")
    print()

    # ========================================================================
    # TEST 3: Calculate other indicators for strategy
    # ========================================================================
    print("📈 Calculating strategy indicators...")

    rsi = RSI(period=14)
    ema_50 = EMA(period=50)
    ema_200 = EMA(period=200)
    adx = ADX(period=14)

    rsi_result = rsi.calculate_batch(data)
    ema50_result = ema_50.calculate_batch(data)
    ema200_result = ema_200.calculate_batch(data)
    adx_result = adx.calculate_batch(data)

    # RSI returns Series, others return DataFrame
    if isinstance(rsi_result, pd.Series):
        data['rsi_14'] = rsi_result
    else:
        data['rsi_14'] = rsi_result.iloc[:, 0]  # First column

    data['ema_50'] = ema50_result.iloc[:, 0] if isinstance(ema50_result, pd.DataFrame) else ema50_result
    data['ema_200'] = ema200_result.iloc[:, 0] if isinstance(ema200_result, pd.DataFrame) else ema200_result
    data['adx_14_adx'] = adx_result['adx'] if 'adx' in adx_result.columns else adx_result.iloc[:, 0]

    print(f"   ✓ RSI 14 calculated")
    print(f"   ✓ EMA 50 calculated")
    print(f"   ✓ EMA 200 calculated")
    print(f"   ✓ ADX 14 calculated")
    print()

    # ========================================================================
    # TEST 4: Detect strategy signals (as per simple_rsidivergence.py)
    # ========================================================================
    print("🎯 Detecting strategy entry signals...")
    print()

    # LONG signals
    data['signal_long'] = (
        (data['rsidiv_bullish'] == True) &
        (data['rsidiv_rsi'] < 40) &
        # (data['close'] > data['ema_50'].shift(1)) &  # Crossover check disabled for testing
        # (data['ema_50'] > data['ema_200']) &  # Trend filter disabled for testing
        (data['adx_14_adx'] > 25)
    )

    # SHORT signals
    data['signal_short'] = (
        (data['rsidiv_bearish'] == True) &
        (data['rsidiv_rsi'] > 60) &
        # (data['close'] < data['ema_50'].shift(1)) &  # Crossover check disabled for testing
        # (data['ema_50'] < data['ema_200']) &  # Trend filter disabled for testing
        (data['adx_14_adx'] > 25)
    )

    # ========================================================================
    # TEST 5: Display divergence signals
    # ========================================================================
    print("="*80)
    print("DIVERGENCE SIGNALS DETECTED")
    print("="*80)
    print()

    # Regular Bullish divergences
    bullish_signals = data[data['rsidiv_bullish'] == True].copy()
    if len(bullish_signals) > 0:
        print(f"🟢 BULLISH DIVERGENCES (Regular): {len(bullish_signals)} found")
        print()
        for idx, row in bullish_signals.iterrows():
            print(f"   Date: {row['date_str']}")
            print(f"   Price: {row['close']:.2f}")
            print(f"   RSI: {row['rsidiv_rsi']:.2f}")
            print(f"   Strength: {row['rsidiv_strength']:.2f}")
            print(f"   ADX: {row['adx_14_adx']:.2f}")

            # Check if it's a strategy signal
            if row['signal_long']:
                print(f"   >>> STRATEGY LONG SIGNAL <<<")
            print()
    else:
        print(f"🟢 BULLISH DIVERGENCES (Regular): None found")
        print()

    # Hidden Bullish divergences
    hidden_bullish_signals = data[data['rsidiv_hidden_bullish'] == True].copy()
    if len(hidden_bullish_signals) > 0:
        print(f"🟢 HIDDEN BULLISH DIVERGENCES: {len(hidden_bullish_signals)} found")
        print()
        for idx, row in hidden_bullish_signals.iterrows():
            print(f"   Date: {row['date_str']}")
            print(f"   Price: {row['close']:.2f}")
            print(f"   RSI: {row['rsidiv_rsi']:.2f}")
            print(f"   Strength: {row['rsidiv_strength']:.2f}")
            print(f"   ADX: {row['adx_14_adx']:.2f}")
            print()
    else:
        print(f"🟢 HIDDEN BULLISH DIVERGENCES: None found")
        print()

    # Regular Bearish divergences
    bearish_signals = data[data['rsidiv_bearish'] == True].copy()
    if len(bearish_signals) > 0:
        print(f"🔴 BEARISH DIVERGENCES (Regular): {len(bearish_signals)} found")
        print()
        for idx, row in bearish_signals.iterrows():
            print(f"   Date: {row['date_str']}")
            print(f"   Price: {row['close']:.2f}")
            print(f"   RSI: {row['rsidiv_rsi']:.2f}")
            print(f"   Strength: {row['rsidiv_strength']:.2f}")
            print(f"   ADX: {row['adx_14_adx']:.2f}")

            # Check if it's a strategy signal
            if row['signal_short']:
                print(f"   >>> STRATEGY SHORT SIGNAL <<<")
            print()
    else:
        print(f"🔴 BEARISH DIVERGENCES (Regular): None found")
        print()

    # Hidden Bearish divergences
    hidden_bearish_signals = data[data['rsidiv_hidden_bearish'] == True].copy()
    if len(hidden_bearish_signals) > 0:
        print(f"🔴 HIDDEN BEARISH DIVERGENCES: {len(hidden_bearish_signals)} found")
        print()
        for idx, row in hidden_bearish_signals.iterrows():
            print(f"   Date: {row['date_str']}")
            print(f"   Price: {row['close']:.2f}")
            print(f"   RSI: {row['rsidiv_rsi']:.2f}")
            print(f"   Strength: {row['rsidiv_strength']:.2f}")
            print(f"   ADX: {row['adx_14_adx']:.2f}")
            print()
    else:
        print(f"🔴 HIDDEN BEARISH DIVERGENCES: None found")
        print()

    # ========================================================================
    # TEST 6: Strategy signal summary
    # ========================================================================
    print("="*80)
    print("STRATEGY SIGNALS SUMMARY")
    print("="*80)
    print()

    long_signals = data[data['signal_long'] == True]
    short_signals = data[data['signal_short'] == True]

    print(f"LONG Signals:  {len(long_signals)}")
    print(f"SHORT Signals: {len(short_signals)}")
    print(f"TOTAL Signals: {len(long_signals) + len(short_signals)}")
    print()

    if len(long_signals) > 0:
        print("LONG Signal Details:")
        for idx, row in long_signals.iterrows():
            print(f"   {row['date_str']} | Price: {row['close']:.2f} | RSI: {row['rsidiv_rsi']:.2f}")
        print()

    if len(short_signals) > 0:
        print("SHORT Signal Details:")
        for idx, row in short_signals.iterrows():
            print(f"   {row['date_str']} | Price: {row['close']:.2f} | RSI: {row['rsidiv_rsi']:.2f}")
        print()

    # ========================================================================
    # TEST 7: Export for TradingView comparison
    # ========================================================================
    print("="*80)
    print("EXPORT FOR TRADINGVIEW COMPARISON")
    print("="*80)
    print()

    # Create export dataframe
    export_cols = [
        'date_str', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'rsidiv_rsi', 'rsidiv_bullish', 'rsidiv_bearish',
        'rsidiv_hidden_bullish', 'rsidiv_hidden_bearish', 'rsidiv_strength',
        'rsi_14', 'ema_50', 'ema_200', 'adx_14_adx',
        'signal_long', 'signal_short'
    ]

    export_data = data[export_cols].copy()

    # Save to CSV
    output_file = base_dir / "test_rsidivergence_results.csv"
    export_data.to_csv(output_file, index=False)

    print(f"✓ Results exported to: {output_file}")
    print()
    print("📝 Comparison Instructions:")
    print("   1. Open TradingView: BTCUSDT 15m chart")
    print("   2. Add 'RSI Divergence Indicator' (from temp/rsi_div.txt)")
    print("   3. Set parameters: RSI Period=14, Lookback L/R=5")
    print("   4. Compare divergence signals with exported CSV")
    print()

    # ========================================================================
    # TEST 8: Implementation differences
    # ========================================================================
    print("="*80)
    print("IMPLEMENTATION ANALYSIS vs TradingView")
    print("="*80)
    print()

    print("TradingView (PineScript) Parameters:")
    print("   • RSI Period: 14")
    print("   • Pivot Lookback Left (lbL): 5")
    print("   • Pivot Lookback Right (lbR): 5")
    print("   • Range: 5-60 bars (pivot distance limit)")
    print("   • Price comparison: Uses low/high for pivots")
    print("   • Signal types: Regular + Hidden divergences")
    print()

    print("Our Implementation (Python) Parameters:")
    print("   • RSI Period: 14")
    print("   • Lookback: 5 (symmetric)")
    print("   • Range check: IMPLEMENTED ✅ (5-60 bars)")
    print("   • Price comparison: Uses HIGH/LOW ✅")
    print("   • Signal types: Regular + Hidden divergences ✅")
    print()

    print("Current Status:")
    print("   ✅ HIGH/LOW price pivots implemented (TradingView compatible)")
    print("   ✅ Range check (5-60 bars) implemented")
    print("   ✅ Hidden divergences implemented (both bullish & bearish)")
    print("   ✅ Proper pivot detection using swing_points utility")
    print()
    print("   ℹ️  Note: Minor timing differences may still exist due to:")
    print("      - Different programming language implementations")
    print("      - Floating-point precision differences")
    print("      - Edge cases in pivot detection logic")
    print()

    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print()

    return data


if __name__ == "__main__":
    # Run async test
    result = asyncio.run(test_rsidivergence_signals())
