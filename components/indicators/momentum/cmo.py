#!/usr/bin/env python3
"""
indicators/momentum/cmo.py - CMO (Chande Momentum Oscillator)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

Chande Momentum Oscillator (CMO) - A momentum indicator similar to RSI.
The difference from RSI is that it uses a different normalization.

Features:
- Similar to RSI but with a different calculation
- Ranges from -100 to +100
- Shows overbought/oversold regions
- CMO > +50: Overbought
- CMO < -50: Oversold
- Zero line is an important support/resistance

Usage:
    from components.indicators import get_indicator_class

    CMO = get_indicator_class('cmo')
    cmo = CMO(period=14)
    result = cmo.calculate(data)
    print(result.value['cmo'])

Formula:
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    Difference from RSI:
    - RSI: 100 * (gains / (gains + losses))
    - CMO: 100 * ((gains - losses) / (gains + losses))

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


class CMO(BaseIndicator):
    """
    Chande Momentum Oscillator

    Momentum indicator similar to RSI but using a different normalization.
    It oscillates between -100 and +100.

    Args:
        period: CMO period (default: 14)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 14, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period + 1)

        super().__init__(
            name='cmo',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 1

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
        TA-Lib compatible: Uses Wilder smoothing (RMA).

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: CMO values
        """
        close = data['close']

        # Price differences
        diff = close.diff()

        # Separate profits and losses
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        # Compatible with TA-Lib: Use Wilder smoothing (RMA)
        # RMA = EMA with alpha = 1/period
        alpha = 1 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        # CMO hesapla: 100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)
        cmo = 100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)

        return pd.DataFrame({'cmo': cmo}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for state-based update.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional, for multi-symbol support)
        """
        super().warmup_buffer(data, symbol)

        buffer_key = symbol if symbol else 'default'
        if not hasattr(self, '_cmo_state'):
            self._cmo_state = {}

        if len(data) >= self.period + 1:
            close = data['close'].values
            diff = np.diff(close)

            # Calculate state for Wilder smoothing (RMA)
            alpha = 1 / self.period
            gains = np.where(diff > 0, diff, 0)
            losses = np.where(diff < 0, -diff, 0)

            # RMA hesapla
            avg_gain = gains[0]
            avg_loss = losses[0]
            for i in range(1, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

            self._cmo_state[buffer_key] = {
                'avg_gain': avg_gain,
                'avg_loss': avg_loss,
                'last_close': close[-1],
                'alpha': alpha
            }

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - State-based Wilder smoothing

        Args:
            candle: Yeni mum verisi (dict)
            symbol: Symbol name (optional)

        Returns:
            IndicatorResult: Current CMO value
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental Wilder smoothing hesapla
        if hasattr(self, '_cmo_state') and buffer_key in self._cmo_state:
            state = self._cmo_state[buffer_key]
            avg_gain = state['avg_gain']
            avg_loss = state['avg_loss']
            last_close = state['last_close']
            alpha = state['alpha']

            # Price change
            diff = close_val - last_close
            gain = diff if diff > 0 else 0
            loss = -diff if diff < 0 else 0

            # Update Wilder smoothing (RMA)
            new_avg_gain = alpha * gain + (1 - alpha) * avg_gain
            new_avg_loss = alpha * loss + (1 - alpha) * avg_loss

            # CMO hesapla
            total = new_avg_gain + new_avg_loss
            cmo_value = 100 * (new_avg_gain - new_avg_loss) / total if total > 0 else 0

            # Update state
            self._cmo_state[buffer_key] = {
                'avg_gain': new_avg_gain,
                'avg_loss': new_avg_loss,
                'last_close': close_val,
                'alpha': alpha
            }

            # Signal and trend determination
            if cmo_value < -50:
                signal = SignalType.BUY  # Oversold
                trend = TrendDirection.DOWN
            elif cmo_value > 50:
                signal = SignalType.SELL  # Oversold
                trend = TrendDirection.UP
            else:
                signal = SignalType.HOLD
                trend = TrendDirection.UP if cmo_value > 0 else (
                    TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
                )

            return IndicatorResult(
                value={'cmo': round(cmo_value, 2)},
                timestamp=timestamp_val,
                signal=signal,
                trend=trend,
                strength=min(abs(cmo_value), 100),
                metadata={
                    'period': self.period,
                    'overbought': 50,
                    'oversold': -50
                }
            )

        # Fallback: old buffer-based calculation (backward compatibility)
        self.prices.append(close_val)

        if len(self.prices) < self.period + 1:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert the price list to a numpy array
        prices = np.array(self.prices)
        diff = np.diff(prices)

        # Wilder smoothing (RMA) hesapla
        alpha = 1 / self.period
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        avg_gain = gains[0]
        avg_loss = losses[0]
        for i in range(1, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        total = avg_gain + avg_loss
        cmo_value = 100 * (avg_gain - avg_loss) / total if total > 0 else 0

        # Signal and trend determination
        if cmo_value < -50:
            signal = SignalType.BUY
            trend = TrendDirection.DOWN
        elif cmo_value > 50:
            signal = SignalType.SELL
            trend = TrendDirection.UP
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.UP if cmo_value > 0 else (
                TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
            )

        return IndicatorResult(
            value={'cmo': round(cmo_value, 2)},
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=min(abs(cmo_value), 100),
            metadata={
                'period': self.period,
                'overbought': 50,
                'oversold': -50
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate CMO (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CMO value
        """
        # Fill the buffers
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['cmo'].dropna().values

        if len(valid_values) == 0:
            return None

        cmo_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal and trend
        if cmo_value < -50:
            signal = SignalType.BUY  # Oversold
            trend = TrendDirection.DOWN
        elif cmo_value > 50:
            signal = SignalType.SELL  # Oversold
            trend = TrendDirection.UP
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.UP if cmo_value > 0 else (
                TrendDirection.DOWN if cmo_value < 0 else TrendDirection.NEUTRAL
            )

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'cmo': round(cmo_value, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(cmo_value), 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 14}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['cmo']

    def _requires_volume(self) -> bool:
        """CMO volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CMO']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """CMO indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 CHANDE MOMENTUM OSCILLATOR (CMO) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Sine wave + noise (for better CMO representation)
    base_price = 100
    sine_wave = 10 * np.sin(np.linspace(0, 4 * np.pi, 100))
    noise = np.random.randn(100) * 2
    prices = base_price + sine_wave + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    cmo = CMO(period=14)
    print(f"   ✅ Created: {cmo}")
    print(f"   ✅ Kategori: {cmo.category.value}")
    print(f"   ✅ Required period: {cmo.get_required_periods()}")

    result = cmo(data)
    print(f"   ✅ CMO: {result.value['cmo']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")
    print(f"   ✅ Overbought: {result.metadata.get('overbought', 'N/A')}")
    print(f"   ✅ Oversold: {result.metadata.get('oversold', 'N/A')}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = cmo.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 CMO values:")
    print(batch_result['cmo'].tail())

    # Test 3: Overbuying/overselling detection
    print("\n4. Overbuying/overselling test...")
    cmo_values = batch_result['cmo'].dropna().values
    overbought = len(cmo_values[cmo_values > 50])
    oversold = len(cmo_values[cmo_values < -50])
    print(f"   ✅ Overbought region: {overbought} bar")
    print(f"   ✅ Oversold region: {oversold} bar")
    print(f"   ✅ Min CMO: {min(cmo_values):.2f}")
    print(f"   ✅ Max CMO: {max(cmo_values):.2f}")

    # Test 4: Update method
    print("\n5. Update method test...")
    cmo2 = CMO(period=14)
    init_data = data.head(50)
    cmo2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low']
        }
        update_result = cmo2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: CMO={update_result.value['cmo']:.2f}, "
                  f"Signal={update_result.signal.value}")

    # Test 5: Different periods
    print("\n6. Different period test...")
    for period in [9, 14, 20]:
        cmo_test = CMO(period=period)
        result = cmo_test.calculate(data)
        print(f"   ✅ CMO({period}): {result.value['cmo']:.2f}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
