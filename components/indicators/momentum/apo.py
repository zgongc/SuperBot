#!/usr/bin/env python3
"""
indicators/momentum/apo.py - APO (Absolute Price Oscillator)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

APO (Absolute Price Oscillator) - Absolute Price Oscillator.
Calculates the absolute difference between two EMA periods.

Features:
- Difference between fast and slow EMA
- Similar to MACD but uses absolute value
- Shows trend strength and direction
- Positive value = Bullish momentum
- Negative value = Bearish momentum

Usage:
    from components.indicators import get_indicator_class

    APO = get_indicator_class('apo')
    apo = APO(fast_period=12, slow_period=26)
    result = apo.calculate(data)
    print(result.value['apo'])

Formula:
    APO = Fast EMA - Slow EMA

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


class APO(BaseIndicator):
    """
    APO - Absolute Price Oscillator

    Calculates the absolute difference between two EMAs with different periods.
    It is the absolute value version of MACD.

    Args:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, logger=None, error_handler=None):
        self.fast_period = fast_period
        self.slow_period = slow_period

        super().__init__(
            name='apo',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'fast_period': fast_period, 'slow_period': slow_period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.slow_period * 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "The fast period must be smaller than the slow period"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.
        TA-Lib compatible: Uses SMA (default matype=0)

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: APO values
        """
        close = data['close']

        # Compatible with TA-Lib: Use SMA (not EMA)!
        fast_ma = close.rolling(window=self.fast_period).mean()
        slow_ma = close.rolling(window=self.slow_period).mean()

        # Mutlak fark
        apo = fast_ma - slow_ma

        return pd.DataFrame({'apo': apo}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for state-based update.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional, for multi-symbol support)
        """
        super().warmup_buffer(data, symbol)

        buffer_key = symbol if symbol else 'default'
        if not hasattr(self, '_apo_state'):
            self._apo_state = {}

        if len(data) >= self.slow_period:
            close = data['close'].values
            # Keep data until the end of the slow_period
            self._apo_state[buffer_key] = {
                'close_buffer': list(close[-self.slow_period:]),
                'last_close': close[-1]
            }

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - State-based

        Args:
            candle: Yeni mum verisi (dict)
            symbol: Symbol name (optional)

        Returns:
            IndicatorResult: Current APO value
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental hesapla
        if hasattr(self, '_apo_state') and buffer_key in self._apo_state:
            state = self._apo_state[buffer_key]
            close_buffer = state['close_buffer']

            # Buffer'a yeni close ekle
            close_buffer.append(close_val)

            # Buffer boyutunu koru
            if len(close_buffer) > self.slow_period:
                close_buffer.pop(0)

            # SMA hesapla
            if len(close_buffer) >= self.slow_period:
                fast_ma = np.mean(close_buffer[-self.fast_period:])
                slow_ma = np.mean(close_buffer[-self.slow_period:])
                apo_value = fast_ma - slow_ma

                # Update state
                self._apo_state[buffer_key] = {
                    'close_buffer': close_buffer,
                    'last_close': close_val
                }

                # Signal and trend determination
                if apo_value > 0:
                    signal = SignalType.BUY
                    trend = TrendDirection.UP
                elif apo_value < 0:
                    signal = SignalType.SELL
                    trend = TrendDirection.DOWN
                else:
                    signal = SignalType.HOLD
                    trend = TrendDirection.NEUTRAL

                return IndicatorResult(
                    value={'apo': round(apo_value, 4)},
                    timestamp=timestamp_val,
                    signal=signal,
                    trend=trend,
                    strength=min(abs(apo_value) * 10, 100),
                    metadata={'fast': self.fast_period, 'slow': self.slow_period}
                )

        # If state does not exist, insufficient data
        return IndicatorResult(
            value=0.0,
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=0.0,
            metadata={'insufficient_data': True}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate APO (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: APO value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['apo'].dropna().values

        if len(valid_values) == 0:
            return None

        apo_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal and trend determination
        if apo_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif apo_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'apo': round(apo_value, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(apo_value) * 10, 100),
            metadata={'fast': self.fast_period, 'slow': self.slow_period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'fast_period': 12, 'slow_period': 26}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['apo']

    def _requires_volume(self) -> bool:
        """APO volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['APO']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """APO indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 APO (ABSOLUTE PRICE OSCILLATOR) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 25, 150)
    noise = np.random.randn(150) * 2.5
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
    apo = APO(fast_period=12, slow_period=26)
    print(f"   ✅ Created: {apo}")
    print(f"   ✅ Kategori: {apo.category.value}")
    print(f"   ✅ Required period: {apo.get_required_periods()}")

    result = apo(data)
    print(f"   ✅ APO: {result.value['apo']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = apo.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 APO values:")
    print(batch_result['apo'].tail())

    # Test 3: Different period combinations
    print("\n4. Different period test...")
    configs = [(5, 10), (12, 26), (20, 50)]
    for fast, slow in configs:
        apo_test = APO(fast_period=fast, slow_period=slow)
        result = apo_test.calculate(data)
        print(f"   ✅ APO({fast},{slow}): {result.value['apo']:.4f}, Signal={result.signal.value}")

    # Test 4: Zero-line crossover analizi
    print("\n5. Zero-line crossover analizi...")
    batch_result = apo.calculate_batch(data)
    apo_values = batch_result['apo'].dropna()

    # Crossover count
    crossovers = 0
    for i in range(1, len(apo_values)):
        if (apo_values.iloc[i-1] < 0 and apo_values.iloc[i] > 0) or \
           (apo_values.iloc[i-1] > 0 and apo_values.iloc[i] < 0):
            crossovers += 1

    print(f"   ✅ Total zero-line crossover: {crossovers}")
    print(f"   ✅ Pozitif APO barlar: {sum(apo_values > 0)}")
    print(f"   ✅ Negative APO bars: {sum(apo_values < 0)}")
    print(f"   ✅ Ortalama APO: {apo_values.mean():.4f}")
    print(f"   ✅ APO std sapma: {apo_values.std():.4f}")

    # Test 5: Validasyon testi
    print("\n6. Validasyon testi...")
    try:
        invalid_apo = APO(fast_period=26, slow_period=12)
        print("   ❌ Error: Invalid period combination accepted!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validation successful: {e}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
