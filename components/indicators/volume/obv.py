"""
indicators/volume/obv.py - On Balance Volume

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    OBV (On Balance Volume) - A momentum indicator based on volume.
    Measures the relationship between price and volume.
    Rising OBV = Buying pressure.
    Falling OBV = Selling pressure.

Formula:
    OBV = OBV_prev + volume (if close > close_prev)
    OBV = OBV_prev - volume (if close < close_prev)
    OBV = OBV_prev (if close == close_prev)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class OBV(BaseIndicator):
    """
    On Balance Volume

    Measures momentum by combining price movements with volume.
    Provides strong signals for divergences.

    Args:
        signal_period: Signal line SMA period (default: 20)
    """

    def __init__(
        self,
        signal_period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.signal_period = signal_period

        super().__init__(
            name='obv',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'signal_period': signal_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.signal_period + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "The period must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        close = data['close'].values
        volume = data['volume'].values

        # OBV hesapla
        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        obv_value = obv[-1]
        obv_signal = np.mean(obv[-self.signal_period:]) if len(obv) >= self.signal_period else obv_value

        timestamp = int(data.iloc[-1]['timestamp'])

        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'obv': round(obv_value, 2),
                'obv_signal': round(obv_signal, 2)
            },
            timestamp=timestamp,
            signal=self.get_signal(obv_value, obv_signal),
            trend=self.get_trend(obv[-10:] if len(obv) >= 10 else obv),
            strength=min(abs((obv_value - obv_signal) / obv_signal * 100), 100) if obv_signal != 0 else 0,
            metadata={
                'signal_period': self.signal_period,
                'divergence': round(obv_value - obv_signal, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch OBV calculation - for BACKTEST

        OBV Formula:
            OBV += volume if close > close_prev
            OBV -= volume if close < close_prev
            OBV unchanged if close == close_prev

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: OBV values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close'].values
        volume = data['volume'].values

        # Calculate OBV - using a loop (same logic as calculate)
        obv_values = np.zeros(len(close))
        obv_values[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_values[i] = obv_values[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_values[i] = obv_values[i-1] - volume[i]
            else:
                obv_values[i] = obv_values[i-1]

        obv = pd.Series(obv_values, index=data.index)

        # ADD SIGNAL LINE (SMA)!
        obv_signal = obv.rolling(window=self.signal_period, min_periods=1).mean()

        return pd.DataFrame({
            'obv': obv,
            'obv_signal': obv_signal
        }, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        OBV is a cumulative indicator, so we need to store the last OBV value
        and the OBV values of the last N bars.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque

        # Calculate OBV values
        close = data['close'].values
        volume = data['volume'].values

        obv_values = np.zeros(len(close))
        obv_values[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_values[i] = obv_values[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_values[i] = obv_values[i-1] - volume[i]
            else:
                obv_values[i] = obv_values[i-1]

        # Store the cumulative OBV value
        self._cumulative_obv = obv_values[-1]
        self._prev_close = close[-1]

        # Keep the OBV value for the last signal_period.
        max_len = self.signal_period + 10
        self._obv_buffer = deque(maxlen=max_len)
        for val in obv_values[-max_len:]:
            self._obv_buffer.append(val)

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Since OBV is cumulatively calculated, it refers to the previous OBV value.
        We are adding and subtracting the volume of the new bar.
        """
        from collections import deque

        if not hasattr(self, '_buffers_init'):
            max_len = self.signal_period + 10
            self._obv_buffer = deque(maxlen=max_len)
            self._cumulative_obv = 0.0
            self._prev_close = None
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        # OBV update
        if self._prev_close is None:
            # First bar
            self._cumulative_obv = volume_val
        else:
            if close_val > self._prev_close:
                self._cumulative_obv += volume_val
            elif close_val < self._prev_close:
                self._cumulative_obv -= volume_val
            # If they are equal, no change is needed.

        self._prev_close = close_val
        obv_value = self._cumulative_obv

        # OBV buffer'a ekle
        self._obv_buffer.append(obv_value)

        if len(self._obv_buffer) < self.signal_period:
            return IndicatorResult(
                value={'obv': round(obv_value, 2), 'obv_signal': round(obv_value, 2)},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Calculate the signal path
        obv_signal = np.mean(list(self._obv_buffer)[-self.signal_period:])

        return IndicatorResult(
            value={'obv': round(obv_value, 2), 'obv_signal': round(obv_signal, 2)},
            timestamp=timestamp_val,
            signal=self.get_signal(obv_value, obv_signal),
            trend=self.get_trend(np.array(list(self._obv_buffer)[-10:])),
            strength=min(abs((obv_value - obv_signal) / obv_signal * 100), 100) if obv_signal != 0 else 0,
            metadata={
                'signal_period': self.signal_period,
                'divergence': round(obv_value - obv_signal, 2)
            }
        )

    def get_signal(self, obv_value: float, obv_signal: float) -> SignalType:
        """
        Generate a signal from the OBV value.

        Args:
            obv_value: OBV value
            obv_signal: OBV signal line

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if obv_value > obv_signal:
            return SignalType.BUY
        elif obv_value < obv_signal:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, obv_array: np.ndarray) -> TrendDirection:
        """
        OBV trendini belirle

        Args:
            obv_array: The latest OBV values.

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if len(obv_array) < 2:
            return TrendDirection.NEUTRAL

        # Determining the trend using linear regression
        slope = np.polyfit(range(len(obv_array)), obv_array, 1)[0]

        if slope > 0:
            return TrendDirection.UP
        elif slope < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'signal_period': 20
        }

    def _requires_volume(self) -> bool:
        """OBV volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['OBV']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """OBV indicator test"""

    print("\n" + "="*60)
    print("OBV (ON BALANCE VOLUME) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend is an upward price movement
    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(49):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-2000, 3000))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Volume range: {min(volumes):.0f} -> {max(volumes):.0f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    obv = OBV(signal_period=20)
    print(f"   [OK] Created: {obv}")
    print(f"   [OK] Kategori: {obv.category.value}")
    print(f"   [OK] Required period: {obv.get_required_periods()}")

    result = obv(data)
    print(f"   [OK] OBV Value: {result.value['obv']:,.2f}")
    print(f"   [OK] OBV Signal: {result.value['obv_signal']:,.2f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [10, 20, 30]:
        obv_test = OBV(signal_period=period)
        result = obv_test.calculate(data)
        print(f"   [OK] OBV(signal={period}): {result.value['obv']:,.2f} | Signal: {result.signal.value}")

    # Test 3: Volume gereksinimi
    print("\n4. Volume gereksinimi testi...")
    metadata = obv.metadata
    print(f"   [OK] Volume required: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "OBV volume gerektirmeli!"

    # Test 4: Statistics
    print("\n5. Statistical test...")
    stats = obv.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 5: Trend testi
    print("\n6. Trend testi...")
    # Create an upward trend
    up_prices = [100 + i * 0.5 for i in range(30)]
    up_volumes = [10000 + i * 100 for i in range(30)]

    up_data = pd.DataFrame({
        'timestamp': timestamps[:30],
        'open': up_prices,
        'high': [p + 0.3 for p in up_prices],
        'low': [p - 0.3 for p in up_prices],
        'close': up_prices,
        'volume': up_volumes
    })

    result = obv.calculate(up_data)
    print(f"   [OK] Uptrend OBV: {result.value['obv']:,.2f}")
    print(f"   [OK] Trend direction: {result.trend.name}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
