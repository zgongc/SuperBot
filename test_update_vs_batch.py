"""
Test update() vs calculate_batch() consistency
"""
import sys
import asyncio
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))

from components.managers.parquets_engine import ParquetsEngine
from components.indicators import get_indicator_class
from core.config_engine import get_config


async def test_update_vs_batch():
    print("\n" + "="*80)
    print("TEST: update() vs calculate_batch() Consistency")
    print("="*80 + "\n")

    # Initialize
    config = get_config()
    parquets = ParquetsEngine(config_engine=config, logger_engine=None)
    RSIDivergence = get_indicator_class('rsidivergence')

    # Load BTCUSDT 1m data
    print("Loading BTCUSDT 1m data...")
    data = await parquets.get_historical_data(
        symbol="BTCUSDT",
        timeframe="1m",
        start_date="2026-01-29T00:00:00",
        end_date="2026-01-29T23:59:59",
        warmup_candles=100,
        utc_offset=3
    )
    print(f"Loaded {len(data)} candles\n")

    # Split data: warmup + last 5 candles
    warmup_data = data.iloc[:-5]
    test_candles = data.iloc[-5:]

    print(f"Warmup: {len(warmup_data)} candles")
    print(f"Test: {len(test_candles)} candles\n")

    # Test both methods
    rsidiv = RSIDivergence(rsi_period=14, lookback=5, min_strength=30)

    # Method 1: calculate_batch on ALL data
    print("Method 1: calculate_batch() on all data")
    batch_result = rsidiv.calculate_batch(data)

    # Method 2: warmup + update for each test candle
    print("Method 2: warmup() + update() for each test candle")
    rsidiv2 = RSIDivergence(rsi_period=14, lookback=5, min_strength=30)
    rsidiv2.warmup_buffer(warmup_data)

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")

    for idx, row in test_candles.iterrows():
        # Get batch result
        batch_idx = data.index.get_loc(idx)
        batch_values = {
            'rsi': batch_result['rsi'].iloc[batch_idx],
            'bullish': batch_result['bullish_divergence'].iloc[batch_idx],
            'bearish': batch_result['bearish_divergence'].iloc[batch_idx],
            'strength': batch_result['divergence_strength'].iloc[batch_idx]
        }

        # Get update result
        update_result = rsidiv2.update(row.to_dict())
        update_values = update_result.value

        # Compare
        timestamp = pd.to_datetime(row['timestamp'], unit='ms', utc=True).tz_convert('Europe/Istanbul')
        print(f"Candle: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  RSI:      batch={batch_values['rsi']:.2f}  update={update_values['rsi']:.2f}  {'✅' if abs(batch_values['rsi'] - update_values['rsi']) < 0.1 else '❌'}")
        print(f"  Bullish:  batch={batch_values['bullish']}   update={update_values['bullish_divergence']}   {'✅' if batch_values['bullish'] == update_values['bullish_divergence'] else '❌'}")
        print(f"  Bearish:  batch={batch_values['bearish']}   update={update_values['bearish_divergence']}   {'✅' if batch_values['bearish'] == update_values['bearish_divergence'] else '❌'}")
        print(f"  Strength: batch={batch_values['strength']:.2f}  update={update_values['divergence_strength']:.2f}  {'✅' if abs(batch_values['strength'] - update_values['divergence_strength']) < 0.1 else '❌'}")
        print()

    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_update_vs_batch())
