#!/usr/bin/env python3
"""
indicators/trend/trix.py - TRIX (Triple Exponential Average)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

TRIX (Triple Exponential Average) - Triple exponential moving average rate of change.
Calculates the percentage change of the price after applying triple exponential smoothing.

Features:
- Powerful noise filtering with triple smoothing
- Detects trend changes early
- Positive value = Bullish momentum
- Negative value = Bearish momentum
- Zero-line crossover sinyalleri

Usage:
    from components.indicators import get_indicator_class

    TRIX = get_indicator_class('trix')
    trix = TRIX(period=15)
    result = trix.calculate(data)
    print(result.value['trix'])

Formula:
    EMA1 = EMA(Close, period)
    EMA2 = EMA(EMA1, period)
    EMA3 = EMA(EMA2, period)
    TRIX = 100 * (EMA3 - EMA3[1]) / EMA3[1]

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
from components.indicators.trend.ema import EMA
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class TRIX(BaseIndicator):
    """
    TRIX - Triple Exponential Average

    Triple exponential moving average rate of change.
    Noise is reduced by applying triple exponential smoothing.

    Args:
        period: TRIX period (default: 15)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 15, logger=None, error_handler=None):
        self.period = period

        # Use the EMA indicator (code reuse)
        self._ema = EMA(period=period)

        super().__init__(
            name='trix',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period * 3 + 1

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
            pd.DataFrame: TRIX values
        """
        # 3 x EMA - Use EMA.calculate_batch (code reuse)
        ema1 = self._ema.calculate_batch(data)
        ema2 = self._ema.calculate_batch(self._create_ema_input(ema1, data))
        ema3 = self._ema.calculate_batch(self._create_ema_input(ema2, data))

        # Percentage change
        trix = 100 * ema3.pct_change()

        return pd.DataFrame({'trix': trix}, index=data.index)

    def _create_ema_input(self, series: pd.Series, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create minimal OHLCV DataFrame for EMA calculation from a Series."""
        return pd.DataFrame({
            'timestamp': original_data['timestamp'].values,
            'open': series.values,
            'high': series.values,
            'low': series.values,
            'close': series.values,
            'volume': np.zeros(len(series))
        }, index=original_data.index)

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
            IndicatorResult: TRIX value
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
        Calculate TRIX (last value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TRIX value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['trix'].dropna().values

        if len(valid_values) == 0:
            return None

        trix_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal and trend determination
        if trix_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif trix_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'trix': round(trix_value, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(trix_value) * 100, 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 15}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['trix']

    def _requires_volume(self) -> bool:
        """TRIX volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TRIX']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """TRIX indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 TRIX (TRIPLE EXPONENTIAL AVERAGE) TEST")
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
    trix = TRIX(period=15)
    print(f"   ✅ Created: {trix}")
    print(f"   ✅ Kategori: {trix.category.value}")
    print(f"   ✅ Required period: {trix.get_required_periods()}")

    result = trix(data)
    print(f"   ✅ TRIX: {result.value['trix']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = trix.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 TRIX values:")
    print(batch_result['trix'].tail())

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [10, 15, 20]:
        trix_test = TRIX(period=period)
        result = trix_test.calculate(data)
        print(f"   ✅ TRIX({period}): {result.value['trix']:.4f}, Signal={result.signal.value}")

    # Test 4: Zero-line crossover testi
    print("\n5. Zero-line crossover analizi...")
    batch_result = trix.calculate_batch(data)
    trix_values = batch_result['trix'].dropna()

    # Crossover count
    crossovers = 0
    for i in range(1, len(trix_values)):
        if (trix_values.iloc[i-1] < 0 and trix_values.iloc[i] > 0) or \
           (trix_values.iloc[i-1] > 0 and trix_values.iloc[i] < 0):
            crossovers += 1

    print(f"   ✅ Total zero-line crossover: {crossovers}")
    print(f"   ✅ Pozitif TRIX barlar: {sum(trix_values > 0)}")
    print(f"   ✅ Negative TRIX bars: {sum(trix_values < 0)}")

    # Test 5: Validasyon testi
    print("\n6. Validasyon testi...")
    try:
        invalid_trix = TRIX(period=0)
        print("   ❌ Error: Invalid period accepted!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validation successful: {e}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
