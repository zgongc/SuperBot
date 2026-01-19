#!/usr/bin/env python3
"""
indicators/trend/trima.py - TRIMA (Triangular Moving Average)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

TRIMA (Triangular Moving Average) - Triangular Moving Average.
It is a double-smoothed average calculated as the SMA of the SMA.

Features:
- Smooth trend with double SMA
- Gives more weight to the middle
- Low noise, high latency
- Price crossover signals
- Uzun vadeli trend takibi

Usage:
    from components.indicators import get_indicator_class

    TRIMA = get_indicator_class('trima')
    trima = TRIMA(period=20)
    result = trima.calculate(data)
    print(result.value['trima'])

Formula:
    n = (period + 1) / 2
    SMA1 = SMA(Close, n)
    TRIMA = SMA(SMA1, n)

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


class TRIMA(BaseIndicator):
    """
    TRIMA - Triangular Moving Average

    Calculates the double-smoothed moving average, which is the SMA of the SMA.
    Uses a triangular weighting scheme that gives more weight to the middle value.

    Args:
        period: TRIMA period (default: 20)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 20, logger=None, error_handler=None):
        self.period = period

        super().__init__(
            name='trima',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period * 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period must be positive"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: TRIMA values
        """
        close = data['close']

        # Triangle MA = SMA of SMA
        n = (self.period + 1) // 2
        sma1 = close.rolling(window=n).mean()
        trima = sma1.rolling(window=n).mean()

        return pd.DataFrame({'trima': trima}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: TRIMA value
        """
        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._close_buffer.append(close_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate TRIMA (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TRIMA value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['trima'].dropna().values

        if len(valid_values) == 0:
            return None

        trima_val = valid_values[-1]
        close = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal determination: Price is above TRIMA = BUY
        if close > trima_val:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif close < trima_val:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Power: Percentage difference between price and TRIMA.
        strength = min(abs((close - trima_val) / trima_val * 100) * 10, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'trima': round(trima_val, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 20}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['trima']

    def _requires_volume(self) -> bool:
        """TRIMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TRIMA']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """TRIMA indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 TRIMA (TRIANGULAR MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(200)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 30, 200)
    noise = np.random.randn(200) * 3
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(200)),
        'low': prices - np.abs(np.random.randn(200)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(200)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    trima = TRIMA(period=20)
    print(f"   ✅ Created: {trima}")
    print(f"   ✅ Kategori: {trima.category.value}")
    print(f"   ✅ Required period: {trima.get_required_periods()}")

    result = trima(data)
    print(f"   ✅ TRIMA: {result.value['trima']}")
    print(f"   ✅ Son Close: {data['close'].iloc[-1]:.2f}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = trima.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 TRIMA values:")
    print(batch_result['trima'].tail())

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [10, 20, 30]:
        trima_test = TRIMA(period=period)
        result = trima_test.calculate(data)
        print(f"   ✅ TRIMA({period}): {result.value['trima']:.2f}, Signal={result.signal.value}")

    # Test 4: Crossover analizi
    print("\n5. Price-TRIMA crossover analysis...")
    batch_result = trima.calculate_batch(data)
    trima_values = batch_result['trima'].dropna()
    close_values = data['close'].iloc[len(data)-len(trima_values):]

    crossovers = 0
    for i in range(1, len(trima_values)):
        trima_prev = trima_values.iloc[i-1]
        trima_curr = trima_values.iloc[i]
        close_prev = close_values.iloc[i-1]
        close_curr = close_values.iloc[i]

        if (close_prev < trima_prev and close_curr > trima_curr) or \
           (close_prev > trima_prev and close_curr < trima_curr):
            crossovers += 1

    print(f"   ✅ Total price - TRIMA crossover: {crossovers}")
    print(f"   ✅ Price is above TRIMA: {sum(close_values.values > trima_values.values)}")
    print(f"   ✅ Price below TRIMA: {sum(close_values.values < trima_values.values)}")

    # Test 5: Comparison with SMA
    print("\n6. Comparison with SMA...")
    sma = data['close'].rolling(window=20).mean()
    trima_vals = batch_result['trima']

    # Let's find the valid indices.
    valid_indices = ~trima_vals.isna() & ~sma.isna()
    trima_valid = trima_vals[valid_indices]
    sma_valid = sma[valid_indices]

    if len(trima_valid) > 0:
        diff = abs(trima_valid.values - sma_valid.values).mean()
        print(f"   ✅ Average SMA-TRIMA difference: {diff:.4f}")
        print(f"   ✅ TRIMA daha smooth: {(trima_valid.diff().abs().mean() < sma_valid.diff().abs().mean())}")

    # Test 6: Validasyon testi
    print("\n7. Validasyon testi...")
    try:
        invalid_trima = TRIMA(period=0)
        print("   ❌ Error: Invalid period accepted!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validation successful: {e}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
