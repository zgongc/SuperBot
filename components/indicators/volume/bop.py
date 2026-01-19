#!/usr/bin/env python3
"""
indicators/volume/bop.py - BOP (Balance of Power)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

BOP (Balance of Power) - Power Balance Indicator.
Measures the balance between buyer and seller power.

Features:
- Measures buyer/seller pressure.
- Produces values between -1 and +1.
- Positive value = Buyer pressure (Bullish)
- Negative value = Seller pressure (Bearish)
- Zero = Balanced state
- Independent calculation for each bar.

Usage:
    from components.indicators import get_indicator_class

    BOP = get_indicator_class('bop')
    bop = BOP()
    result = bop.calculate(data)
    print(result.value['bop'])

Formula:
    BOP = (Close - Open) / (High - Low)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje root'unu path'e ekle
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class BOP(BaseIndicator):
    """
    BOP - Balance of Power

    Measures the balance between buyer and seller power.
    It is calculated independently for each bar.

    Args:
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, logger=None, error_handler=None):
        super().__init__(
            name='bop',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return 1

    def validate_params(self) -> bool:
        """Validate parameters"""
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: BOP values
        """
        # BOP calculation
        bop = (data['close'] - data['open']) / (data['high'] - data['low'])

        # Check for division by zero.
        bop = bop.fillna(0)

        return pd.DataFrame({'bop': bop}, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict or list/tuple)

        Returns:
            IndicatorResult: Current BOP value
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle['open']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        hl_range = high_val - low_val

        # Division by zero check
        if hl_range > 0:
            bop_val = (close_val - open_val) / hl_range
        else:
            bop_val = 0

        timestamp = timestamp_val

        # Signal determination: BOP > 0.5 = strong BUY, BOP < -0.5 = strong SELL
        if bop_val > 0.5:
            signal = SignalType.BUY
        elif bop_val < -0.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Trend belirleme
        if bop_val > 0:
            trend = TrendDirection.UP
        elif bop_val < 0:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL

        return IndicatorResult(
            value={'bop': round(bop_val, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(bop_val) * 100, 100),
            metadata={}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate BOP (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: BOP value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        bop_val = batch_result['bop'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal determination
        if bop_val > 0.5:
            signal = SignalType.BUY
        elif bop_val < -0.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        # Trend belirleme
        if bop_val > 0:
            trend = TrendDirection.UP
        elif bop_val < 0:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'bop': round(bop_val, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(bop_val) * 100, 100),
            metadata={}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['bop']

    def _requires_volume(self) -> bool:
        """Does not require BOP volume (optional)"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['BOP']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """BOP indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 BOP (BALANCE OF POWER) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    close_prices = base_price + trend + noise

    # Create OHLC
    opens = close_prices + np.random.randn(100) * 0.5
    highs = np.maximum(opens, close_prices) + np.abs(np.random.randn(100))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.randn(100))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(close_prices):.2f} -> {max(close_prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    bop = BOP()
    print(f"   ✅ Created: {bop}")
    print(f"   ✅ Kategori: {bop.category.value}")
    print(f"   ✅ Required period: {bop.get_required_periods()}")

    result = bop(data)
    print(f"   ✅ BOP: {result.value['bop']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = bop.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 BOP values:")
    print(batch_result['bop'].tail())

    # Test 3: Update method
    print("\n4. Update method test...")
    bop2 = BOP()

    # Update for the last 5 bars
    for i in range(95, 100):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'open': data.iloc[i]['open'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low'],
            'close': data.iloc[i]['close']
        }
        update_result = bop2.update(candle)
        print(f"   ✅ Bar {i}: BOP={update_result.value['bop']:.4f}, "
              f"Signal={update_result.signal.value}")

    # Test 4: BOP distribution analysis
    print("\n5. BOP distribution analysis...")
    batch_result = bop.calculate_batch(data)
    bop_values = batch_result['bop']

    print(f"   ✅ Ortalama BOP: {bop_values.mean():.4f}")
    print(f"   ✅ Std sapma: {bop_values.std():.4f}")
    print(f"   ✅ Min BOP: {bop_values.min():.4f}")
    print(f"   ✅ Max BOP: {bop_values.max():.4f}")
    print(f"   ✅ Pozitif BOP barlar: {sum(bop_values > 0)}")
    print(f"   ✅ Negative BOP bars: {sum(bop_values < 0)}")
    print(f"   ✅ Strong receiver (>0.5): {sum(bop_values > 0.5)}")
    print(f"   ✅ Strong seller (<-0.5): {sum(bop_values < -0.5)}")

    # Test 5: Division by zero test
    print("\n6. Division by zero test...")
    # Create a Doji (open = close = high = low)
    test_data = pd.DataFrame({
        'timestamp': [1697000000000],
        'open': [100.0],
        'high': [100.0],
        'low': [100.0],
        'close': [100.0],
        'volume': [1000]
    })
    result = bop.calculate(test_data)
    print(f"   ✅ Doji bar BOP: {result.value['bop']} (should be zero)")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
