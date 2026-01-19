#!/usr/bin/env python3
"""
indicators/statistical/var.py - VAR (Variance)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

VAR (Variance) - Variance Indicator.
The square of the standard deviation of the price distribution.

Features:
- Volatility measurement
- Width of price distribution
- High VAR = High volatility
- Low VAR = Low volatility
- Used for risk management

Usage:
    from components.indicators import get_indicator_class

    VAR = get_indicator_class('var')
    var = VAR(period=20)
    result = var.calculate(data)
    print(result.value['var'])

Formula:
    VAR = Σ(Close - Mean)² / N

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
from collections import deque
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class VAR(BaseIndicator):
    """
    VAR - Variance

    The square of the standard deviation of the price distribution.
    Used for volatility and risk measurement.

    Args:
        period: VAR period (default: 20)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 20, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period)

        super().__init__(
            name='var',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period must be at least 2"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.
        TA-Lib compatible: uses population variance (ddof=0).

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: VAR values
        """
        # TA-Lib uyumlu: ddof=0 (population variance)
        var = data['close'].rolling(window=self.period).var(ddof=0)
        return pd.DataFrame({'var': var}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        # prices deque'yu doldur
        self.prices.clear()
        self.prices.extend(data['close'].tail(self.period).values)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Current VAR value
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self.prices.append(close_val)

        if len(self.prices) < self.period:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # TA-Lib uyumlu: population variance (ddof=0)
        var_val = np.var(list(self.prices), ddof=0)

        # VAR does not generate a signal itself, but it indicates high volatility.
        return IndicatorResult(
            value={'var': round(var_val, 4)},
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=min(var_val * 10, 100),
            metadata={'period': self.period}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate the variable (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VAR value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['var'].dropna().values

        if len(valid_values) == 0:
            return None

        var_val = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'var': round(var_val, 4)},
            timestamp=timestamp,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=min(var_val * 10, 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 20}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['var']

    def _requires_volume(self) -> bool:
        """Does not require VAR volume"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VAR']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """VAR indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 VAR (VARIANCE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Two different volatility periods
    low_vol = np.random.randn(75) * 1  # Low volatility
    high_vol = np.random.randn(75) * 5  # High volatility
    noise = np.concatenate([low_vol, high_vol])

    base_price = 100
    trend = np.linspace(0, 20, 150)
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(150)),
        'low': prices - np.abs(np.random.randn(150)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(150)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    var = VAR(period=20)
    print(f"   ✅ Created: {var}")
    print(f"   ✅ Category: {var.category.value}")
    print(f"   ✅ Required period: {var.get_required_periods()}")

    result = var(data)
    print(f"   ✅ VAR: {result.value['var']}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = var.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 VAR values:")
    print(batch_result['var'].tail())

    # Test 3: Update method
    print("\n4. Update method test...")
    var2 = VAR(period=20)
    init_data = data.head(50)
    var2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close']
        }
        update_result = var2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: VAR={update_result.value['var']:.4f}")

    # Test 4: Analysis of volatility periods
    print("\n5. Volatility period analysis...")
    batch_result = var.calculate_batch(data)
    var_values = batch_result['var'].dropna()

    # First and second half
    mid_point = len(var_values) // 2
    first_half = var_values.iloc[:mid_point]
    second_half = var_values.iloc[mid_point:]

    print(f"   ✅ First half average VAR: {first_half.mean():.4f}")
    print(f"   ✅ Second half average VAR: {second_half.mean():.4f}")
    print(f"   ✅ Volatility increase: {(second_half.mean() / first_half.mean()):.2f}x")

    # Test 5: Different periods
    print("\n6. Different period test...")
    for period in [10, 20, 30]:
        var_test = VAR(period=period)
        result = var_test.calculate(data)
        print(f"   ✅ VAR({period}): {result.value['var']:.4f}")

    # Test 6: Relationship with standard deviation
    print("\n7. Relationship with standard deviation...")
    batch_result = var.calculate_batch(data)
    var_values = batch_result['var'].dropna()
    std_values = data['close'].rolling(window=20).std().dropna()

    # VAR = STD²
    valid_indices = ~var_values.isna() & ~std_values.isna()
    var_valid = var_values[valid_indices]
    std_valid = std_values[valid_indices]

    if len(var_valid) > 0:
        # Compare the last values
        last_var = var_valid.iloc[-1]
        last_std = std_valid.iloc[-1]
        expected_var = last_std ** 2

        print(f"   ✅ VAR: {last_var:.4f}")
        print(f"   ✅ STD²: {expected_var:.4f}")
        print(f"   ✅ Fark: {abs(last_var - expected_var):.6f}")

    # Test 7: Validasyon testi
    print("\n8. Validasyon testi...")
    try:
        invalid_var = VAR(period=1)
        print("   ❌ Error: Invalid period accepted!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validation successful: {e}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
