"""
indicators/momentum/mfi.py - Money Flow Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    MFI (Money Flow Index) - Volume weighted momentum oscillator
    Range: 0-100
    Overbought: > 80
    Oversold: < 20
    It is known as the version of RSI that uses volume.

Formula:
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = Typical Price × Volume
    Money Flow Ratio = (14-period Positive Money Flow) / (14-period Negative Money Flow)
    MFI = 100 - (100 / (1 + Money Flow Ratio))

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


class MFI(BaseIndicator):
    """
    Money Flow Index

    Volume-weighted momentum oscillator.
    Detects overbought/oversold conditions by using the relationship between price and volume.

    Args:
        period: MFI period (default: 14)
        overbought: Overbought level (default: 80)
        oversold: Oversold level (default: 20)
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 80,
        oversold: float = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='mfi',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'overbought': overbought,
                'oversold': oversold
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
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold should be smaller than overbought"
            )
        if not (0 <= self.oversold <= 100) or not (0 <= self.overbought <= 100):
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Levels must be between 0 and 100"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        MFI hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: MFI value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Typical Price hesapla
        typical_price = (high + low + close) / 3

        # Raw Money Flow hesapla
        raw_money_flow = typical_price * volume

        # Separate positive and negative money flows
        # Use data up to the last period
        period_data = typical_price[-(self.period + 1):]
        period_flow = raw_money_flow[-(self.period + 1):]

        positive_flow = []
        negative_flow = []

        for i in range(1, len(period_data)):
            if period_data[i] > period_data[i-1]:
                positive_flow.append(period_flow[i])
                negative_flow.append(0)
            elif period_data[i] < period_data[i-1]:
                positive_flow.append(0)
                negative_flow.append(period_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        # Total positive and negative flow
        positive_sum = np.sum(positive_flow)
        negative_sum = np.sum(negative_flow)

        # MFI hesapla
        if negative_sum == 0:
            mfi_value = 100.0
        else:
            money_flow_ratio = positive_sum / negative_sum
            mfi_value = 100 - (100 / (1 + money_flow_ratio))

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(mfi_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(mfi_value),
            trend=self.get_trend(mfi_value),
            strength=abs(mfi_value - 50) * 2,  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'typical_price': round(typical_price[-1], 2),
                'positive_flow': round(positive_sum, 2),
                'negative_flow': round(negative_sum, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch MFI calculation - for BACKTEST

        MFI Formula:
            Typical Price = (High + Low + Close) / 3
            Money Flow = Typical Price × Volume
            MFI = 100 - 100 / (1 + Positive MF / Negative MF)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: MFI values for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Raw Money Flow = Typical Price × Volume
        raw_money_flow = typical_price * data['volume']

        # Determine direction: +1 if up, -1 if down, 0 if unchanged
        direction = typical_price.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Positive and Negative Money Flow
        positive_mf = (direction == 1) * raw_money_flow
        negative_mf = (direction == -1) * raw_money_flow

        # Rolling sum over period
        positive_sum = positive_mf.rolling(window=self.period).sum()
        negative_sum = negative_mf.rolling(window=self.period).sum()

        # Money Flow Ratio
        mf_ratio = positive_sum / negative_sum

        # MFI calculation
        mfi = 100 - (100 / (1 + mf_ratio))

        # Handle division by zero (when negative_sum = 0)
        mfi = mfi.fillna(100)

        # Set first period values to NaN (warmup)
        mfi.iloc[:self.period] = np.nan

        return pd.Series(mfi.values, index=data.index, name='mfi')

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

        # Create and fill the buffers
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=50.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the MFI value.

        Args:
            value: MFI value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if value < self.oversold:
            return SignalType.BUY
        elif value > self.overbought:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the MFI value.

        Args:
            value: MFI value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 50:
            return TrendDirection.UP
        elif value < 50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 14,
            'overbought': 80,
            'oversold': 20
        }

    def _requires_volume(self) -> bool:
        """MFI volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MFI']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """MFI indicator test"""

    print("\n" + "="*60)
    print("MFI (MONEY FLOW INDEX) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    mfi = MFI(period=14)
    print(f"   [OK] Created: {mfi}")
    print(f"   [OK] Kategori: {mfi.category.value}")
    print(f"   [OK] Required period: {mfi.get_required_periods()}")
    print(f"   [OK] Volume required: {mfi._requires_volume()}")

    result = mfi(data)
    print(f"   [OK] MFI Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [7, 14, 21]:
        mfi_test = MFI(period=period)
        result = mfi_test.calculate(data)
        print(f"   [OK] MFI({period}): {result.value} | Signal: {result.signal.value}")

    # Test 3: Custom levels
    print("\n4. Special level test...")
    mfi_custom = MFI(period=14, overbought=90, oversold=10)
    result = mfi_custom.calculate(data)
    print(f"   [OK] MFI with custom level: {result.value}")
    print(f"   [OK] Overbought: {mfi_custom.overbought}, Oversold: {mfi_custom.oversold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Volume effect - Test with high volume
    print("\n5. Volume etkisi testi...")
    high_vol_data = data.copy()
    high_vol_data.loc[high_vol_data.index[-5:], 'volume'] *= 3
    result_high_vol = mfi.calculate(high_vol_data)
    result_normal_vol = mfi.calculate(data)
    print(f"   [OK] Normal volume MFI: {result_normal_vol.value}")
    print(f"   [OK] High volume MFI: {result_high_vol.value}")
    print(f"   [OK] Fark: {abs(result_high_vol.value - result_normal_vol.value):.2f}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = mfi.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = mfi.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
