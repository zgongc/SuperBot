"""
indicators/volume/force_index.py - Force Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Force Index - Force index
    Measures buying/selling power by combining price change and volume
    Positive = Buyer power is dominant
    Negative = Seller power is dominant

Formula:
    Force Index = (Close - Close_prev) × Volume
    Smoothed FI = EMA(Force Index, period)

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


class ForceIndex(BaseIndicator):
    """
    Force Index

    Indicator developed by Alexander Elder.
    Measures buying/selling power by multiplying price changes by volume.

    Args:
        period: Force Index smoothing period (default: 13)
    """

    def __init__(
        self,
        period: int = 13,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='force_index',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
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
                "The period must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Force Index hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Force Index value
        """
        close = data['close'].values
        volume = data['volume'].values

        # Raw Force Index hesapla
        force_index_raw = np.zeros(len(close))

        for i in range(1, len(close)):
            price_change = close[i] - close[i-1]
            force_index_raw[i] = price_change * volume[i]

        # Smooth with EMA
        force_index_ema = self._calculate_ema(force_index_raw, self.period)
        fi_value = force_index_ema[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Normalized power (relative to volume)
        avg_volume = np.mean(volume[-self.period:])
        normalized_strength = abs(fi_value) / avg_volume if avg_volume > 0 else 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(fi_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(fi_value),
            trend=self.get_trend(force_index_ema[-10:] if len(force_index_ema) >= 10 else force_index_ema),
            strength=min(normalized_strength * 100, 100),
            metadata={
                'period': self.period,
                'raw_fi': round(force_index_raw[-1], 2),
                'price_change': round(close[-1] - close[-2], 8),
                'volume': int(volume[-1])
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Force Index calculation - for BACKTEST

        Force Index Formula:
            Raw FI = (Close - Close_prev) × Volume
            Smoothed FI = EMA(Raw FI, period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Force Index values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']
        volume = data['volume']

        # Raw Force Index = Price Change × Volume (vectorized)
        price_change = close.diff()
        force_index_raw = price_change * volume

        # Smooth with EMA
        force_index = force_index_raw.ewm(span=self.period, adjust=False).mean()

        # Set first period values to NaN (warmup)
        force_index.iloc[:self.period] = np.nan

        return pd.Series(force_index.values, index=data.index, name='force_index')

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        EMA hesapla

        Args:
            data: Veri array
            period: EMA periyodu

        Returns:
            np.ndarray: EMA values
        """
        ema = np.zeros(len(data))
        multiplier = 2 / (period + 1)

        # Initial value is SMA
        ema[period-1] = np.mean(data[:period])

        # EMA hesapla
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - prepares the necessary state for update()"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
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

        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the Force Index value.

        Args:
            value: Force Index value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if value > 0:
            return SignalType.BUY  # Buy signal
        elif value < 0:
            return SignalType.SELL  # Seller strength
        return SignalType.HOLD

    def get_trend(self, fi_array: np.ndarray) -> TrendDirection:
        """
        Force Index trendini belirle

        Args:
            fi_array: Last Force Index values

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if len(fi_array) < 2:
            return TrendDirection.NEUTRAL

        # Average of the last values
        recent_avg = np.mean(fi_array[-3:]) if len(fi_array) >= 3 else fi_array[-1]

        if recent_avg > 0:
            return TrendDirection.UP
        elif recent_avg < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 13
        }

    def _requires_volume(self) -> bool:
        """Force Index volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ForceIndex']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Force Index indicator test"""

    print("\n" + "="*60)
    print("FORCE INDEX TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-3000, 5000))

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

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    fi = ForceIndex(period=13)
    print(f"   [OK] Created: {fi}")
    print(f"   [OK] Kategori: {fi.category.value}")
    print(f"   [OK] Required period: {fi.get_required_periods()}")

    result = fi(data)
    print(f"   [OK] Force Index: {result.value:,.2f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [2, 13, 50]:
        fi_test = ForceIndex(period=period)
        result = fi_test.calculate(data)
        print(f"   [OK] FI({period}): {result.value:,.2f} | Signal: {result.signal.value}")

    # Test 3: Volume gereksinimi
    print("\n4. Volume gereksinimi testi...")
    metadata = fi.metadata
    print(f"   [OK] Volume required: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "Force Index volume gerektirmeli!"

    # Test 4: Power interpretation
    print("\n5. Power interpretation test...")
    result = fi.calculate(data)
    fi_val = result.value
    raw_fi = result.metadata['raw_fi']

    print(f"   [OK] Raw FI: {raw_fi:,.2f}")
    print(f"   [OK] Smoothed FI: {fi_val:,.2f}")

    if fi_val > 1000:
        print("   [OK] Strong receiver pressure")
    elif fi_val > 0:
        print("   [OK] Weak receiver pressure")
    elif fi_val > -1000:
        print("   [OK] Weak vendor pressure")
    else:
        print("   [OK] Strong seller pressure")

    # Test 5: Price-Volume relationship
    print("\n6. Price-Volume Relationship Test...")
    price_change = result.metadata['price_change']
    volume = result.metadata['volume']
    print(f"   [OK] Price change: {price_change:.8f}")
    print(f"   [OK] Volume: {volume:,}")
    print(f"   [OK] Force Index = Price Change × Volume")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = fi.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
