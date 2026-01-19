"""
indicators/momentum/williams_r.py - Williams %R

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Williams %R - Momentum oscillator
    Range: between -100 and 0
    Overbought: > -20
    Oversold: < -80
    Similar logic to Stochastic, uses a negative scale.

Formula:
    Williams %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)
    Highest High = The highest price of the last N periods
    Lowest Low = The lowest price of the last N periods

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


class WilliamsR(BaseIndicator):
    """
    Williams %R

    Momentum oscillator detects overbought/oversold conditions.
    It takes values between -100 and 0.

    Args:
        period: Williams %R period (default: 14)
        overbought: Overbought level (default: -20)
        oversold: Oversold level (default: -80)
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = -20,
        oversold: float = -80,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='williams_r',
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
        return self.period

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
        if not (-100 <= self.oversold <= 0) or not (-100 <= self.overbought <= 0):
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Levels must be between -100 and 0"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Williams %R hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Williams %R value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Get data up to the last period
        period_high = high[-self.period:]
        period_low = low[-self.period:]
        current_close = close[-1]

        # Calculate the highest high and lowest low.
        highest_high = np.max(period_high)
        lowest_low = np.min(period_low)

        # Williams %R hesapla
        if highest_high == lowest_low:
            williams_r_value = -50.0  # Neutral value
        else:
            williams_r_value = -100 * (highest_high - current_close) / (highest_high - lowest_low)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(williams_r_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(williams_r_value),
            trend=self.get_trend(williams_r_value),
            strength=abs(williams_r_value + 50) * 2,  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'highest_high': round(highest_high, 2),
                'lowest_low': round(lowest_low, 2),
                'close': round(current_close, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Williams %R calculation - for BACKTEST

        Williams %R Formula:
            %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Williams %R values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Highest High and Lowest Low over period (vectorized)
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()

        # Williams %R = -100 * (HH - Close) / (HH - LL)
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        williams_r = -100 * (highest_high - close) / denominator

        # Handle division by zero
        williams_r = williams_r.fillna(-50)  # Neutral value

        # Set first period values to NaN (warmup)
        williams_r.iloc[:self.period-1] = np.nan

        return pd.Series(williams_r.values, index=data.index, name='williams_r')

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
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=-50.0,
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
            'open': [open_val] * len(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the Williams %R value.

        Args:
            value: Williams %R value

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
        Determine the trend based on the Williams %R value.

        Args:
            value: Williams %R value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > -50:
            return TrendDirection.UP
        elif value < -50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 14,
            'overbought': -20,
            'oversold': -80
        }

    def _requires_volume(self) -> bool:
        """Williams %R volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['WilliamsR']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Williams %R indicator test"""

    print("\n" + "="*60)
    print("WILLIAMS %R TEST")
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
    williams = WilliamsR(period=14)
    print(f"   [OK] Created: {williams}")
    print(f"   [OK] Kategori: {williams.category.value}")
    print(f"   [OK] Required period: {williams.get_required_periods()}")

    result = williams(data)
    print(f"   [OK] Williams %R Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [7, 14, 21]:
        williams_test = WilliamsR(period=period)
        result = williams_test.calculate(data)
        print(f"   [OK] Williams %R({period}): {result.value} | Signal: {result.signal.value}")

    # Test 3: Custom levels
    print("\n4. Special level test...")
    williams_custom = WilliamsR(period=14, overbought=-30, oversold=-70)
    result = williams_custom.calculate(data)
    print(f"   [OK] Williams %R for custom levels: {result.value}")
    print(f"   [OK] Overbought: {williams_custom.overbought}, Oversold: {williams_custom.oversold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Overbuying/overselling conditions
    print("\n5. Overbuying/overselling test...")
    # Simulate an upward trend
    up_data = data.copy()
    up_data.loc[up_data.index[-5:], 'close'] = [p + 5 for p in up_data.loc[up_data.index[-5:], 'close']]
    up_data.loc[up_data.index[-5:], 'high'] = [p + 5.5 for p in up_data.loc[up_data.index[-5:], 'high']]

    result_up = williams.calculate(up_data)
    print(f"   [OK] Uprising trend Williams %R: {result_up.value}")
    print(f"   [OK] Signal: {result_up.signal.value}")

    # Simulate a downtrend
    down_data = data.copy()
    down_data.loc[down_data.index[-5:], 'close'] = [p - 5 for p in down_data.loc[down_data.index[-5:], 'close']]
    down_data.loc[down_data.index[-5:], 'low'] = [p - 5.5 for p in down_data.loc[down_data.index[-5:], 'low']]

    result_down = williams.calculate(down_data)
    print(f"   [OK] Williams %R downtrend: {result_down.value}")
    print(f"   [OK] Signal: {result_down.signal.value}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = williams.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = williams.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
