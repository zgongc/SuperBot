"""
Test different strength thresholds for RSI Divergence strategy
"""
import asyncio
import pandas as pd
from pathlib import Path
from components.managers.parquets_engine import ParquetsEngine
from components.indicators import get_indicator_class
from core.config_engine import get_config

async def test_thresholds():
    config = get_config()
    parquets = ParquetsEngine(config_engine=config, logger_engine=None)
    
    data = await parquets.get_historical_data(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date='2026-01-01T00:00:00',
        end_date='2026-01-30T23:59:59',
        warmup_candles=200,
        utc_offset=3
    )
    
    # Calculate indicators
    RSIDivergence = get_indicator_class('rsidivergence')
    ADX = get_indicator_class('adx')
    
    rsidiv = RSIDivergence(rsi_period=14, lookback=5, min_strength=30)
    adx = ADX(period=14)
    
    rsidiv_result = rsidiv.calculate_batch(data)
    adx_result = adx.calculate_batch(data)
    
    data['rsidiv_bullish'] = rsidiv_result['bullish_divergence']
    data['rsidiv_bearish'] = rsidiv_result['bearish_divergence']
    data['rsidiv_rsi'] = rsidiv_result['rsi']
    data['rsidiv_strength'] = rsidiv_result['divergence_strength']
    data['adx'] = adx_result['adx']
    
    print("\n" + "="*80)
    print("STRENGTH THRESHOLD COMPARISON")
    print("="*80)
    print()
    
    # Test different configurations
    configs = [
        ("No filters", None, None, None),
        ("Strength ≥20", 20, None, None),
        ("Strength ≥30", 30, None, None),
        ("Strength ≥40", 40, None, None),
        ("Strength ≥50", 50, None, None),
        ("Strength ≥40 + RSI", 40, True, None),
        ("Strength ≥40 + RSI + ADX", 40, True, True),
        ("Strength ≥30 + RSI + ADX", 30, True, True),
    ]
    
    for name, strength_thresh, use_rsi, use_adx in configs:
        # Long signals
        long_cond = data['rsidiv_bullish'] == True
        if strength_thresh:
            long_cond = long_cond & (data['rsidiv_strength'] >= strength_thresh)
        if use_rsi:
            long_cond = long_cond & (data['rsidiv_rsi'] < 40)
        if use_adx:
            long_cond = long_cond & (data['adx'] > 25)
        
        # Short signals
        short_cond = data['rsidiv_bearish'] == True
        if strength_thresh:
            short_cond = short_cond & (data['rsidiv_strength'] >= strength_thresh)
        if use_rsi:
            short_cond = short_cond & (data['rsidiv_rsi'] > 60)
        if use_adx:
            short_cond = short_cond & (data['adx'] > 25)
        
        long_signals = len(data[long_cond])
        short_signals = len(data[short_cond])
        total_signals = long_signals + short_signals
        
        print(f"{name:30s}: {total_signals:3d} signals (L:{long_signals:2d} S:{short_signals:2d})")
    
    print()
    print("="*80)

asyncio.run(test_thresholds())
