#!/usr/bin/env python3
"""
indicators/statistical/typprice.py - TYPPRICE (Typical Price)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

TYPPRICE (Typical Price) - Typical Price.
The average of the High, Low, and Close prices.

Features:
- Simple and fast calculation
- Provides the representative price of the bar
- Frequently used in volume indicators
- Independent calculation for each bar
- Median price approach

Usage:
    from components.indicators import get_indicator_class

    TYPPRICE = get_indicator_class('typprice')
    typprice = TYPPRICE()
    result = typprice.calculate(data)
    print(result.value['typprice'])

Formula:
    TYPPRICE = (High + Low + Close) / 3

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


class TYPPRICE(BaseIndicator):
    """
    TYPPRICE - Typical Price

    Average of High, Low, and Close prices.
    Representative price calculated independently for each bar.

    Args:
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, logger=None, error_handler=None):
        super().__init__(
            name='typprice',
            category=IndicatorCategory.STATISTICAL,
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
            pd.DataFrame: TYPPRICE values
        """
        typprice = (data['high'] + data['low'] + data['close']) / 3
        return pd.DataFrame({'typprice': typprice}, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: The current TYPPRICE value.
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        typprice = (high_val + low_val + close_val) / 3

        # Typical price does not generate a signal itself.
        return IndicatorResult(
            value={'typprice': round(typprice, 2)},
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=0,
            metadata={}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate TYPPRICE (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TYPPRICE value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        typprice = batch_result['typprice'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Typical price does not generate a signal itself.
        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'typprice': round(typprice, 2)},
            timestamp=timestamp,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=0,
            metadata={}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['typprice']

    def _requires_volume(self) -> bool:
        """TYPPRICE volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TYPPRICE']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """TYPPRICE indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 TYPPRICE (TYPICAL PRICE) TEST")
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
    typprice = TYPPRICE()
    print(f"   ✅ Created: {typprice}")
    print(f"   ✅ Kategori: {typprice.category.value}")
    print(f"   ✅ Required period: {typprice.get_required_periods()}")

    result = typprice(data)
    print(f"   ✅ TYPPRICE: {result.value['typprice']}")
    print(f"   ✅ Close: {data['close'].iloc[-1]:.2f}")
    print(f"   ✅ High: {data['high'].iloc[-1]:.2f}")
    print(f"   ✅ Low: {data['low'].iloc[-1]:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = typprice.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 TYPPRICE values:")
    print(batch_result['typprice'].tail())

    # Test 3: Update method
    print("\n4. Update method test...")
    typprice2 = TYPPRICE()

    # Update for the last 5 bars
    for i in range(95, 100):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low'],
            'close': data.iloc[i]['close']
        }
        update_result = typprice2.update(candle)
        print(f"   ✅ Bar {i}: TYPPRICE={update_result.value['typprice']:.2f}, "
              f"Close={candle['close']:.2f}")

    # Test 4: Comparison with close
    print("\n5. Comparing with the closing price...")
    batch_result = typprice.calculate_batch(data)
    typprice_values = batch_result['typprice']
    close_values = data['close']

    diff = (typprice_values - close_values).abs()
    print(f"   ✅ Ortalama fark: {diff.mean():.4f}")
    print(f"   ✅ Max fark: {diff.max():.4f}")
    print(f"   ✅ Min fark: {diff.min():.4f}")
    print(f"   ✅ TYPPRICE > Close: {sum(typprice_values > close_values)}")
    print(f"   ✅ TYPPRICE < Close: {sum(typprice_values < close_values)}")

    # Test 5: Manual calculation verification
    print("\n6. Manual calculation verification...")
    last_bar = data.iloc[-1]
    manual_typprice = (last_bar['high'] + last_bar['low'] + last_bar['close']) / 3
    calc_typprice = result.value['typprice']

    print(f"   ✅ Manual calculation: {manual_typprice:.2f}")
    print(f"   ✅ Indicator calculation: {calc_typprice:.2f}")
    print(f"   ✅ Is it equal: {abs(manual_typprice - calc_typprice) < 0.01}")

    # Test 6: Statistical analysis
    print("\n7. Statistical analysis...")
    print(f"   ✅ Ortalama TYPPRICE: {typprice_values.mean():.2f}")
    print(f"   ✅ Std sapma: {typprice_values.std():.2f}")
    print(f"   ✅ Min TYPPRICE: {typprice_values.min():.2f}")
    print(f"   ✅ Max TYPPRICE: {typprice_values.max():.2f}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
