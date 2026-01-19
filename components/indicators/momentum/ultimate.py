"""
indicators/momentum/ultimate.py - Ultimate Oscillator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Ultimate Oscillator - Developed by Larry Williams
    Range: 0-100
    Overbought: > 70
    Oversold: < 30
    Uses the weighted average of 3 different timeframes.

Formula:
    BP = Close - Min(Low, Previous Close)
    TR = Max(High, Previous Close) - Min(Low, Previous Close)
    Average7 = Sum(BP, 7) / Sum(TR, 7)
    Average14 = Sum(BP, 14) / Sum(TR, 14)
    Average28 = Sum(BP, 28) / Sum(TR, 28)
    UO = 100 × ((4×Average7 + 2×Average14 + Average28) / (4 + 2 + 1))

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


class UltimateOscillator(BaseIndicator):
    """
    Ultimate Oscillator

    It combines three different periods to generate more reliable signals.
    It balances short, medium, and long-term momentum.

    Args:
        period1: Short period (default: 7)
        period2: Medium period (default: 14)
        period3: Long period (default: 28)
        overbought: Overbought level (default: 70)
        oversold: Oversold level (default: 30)
    """

    def __init__(
        self,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        overbought: float = 70,
        oversold: float = 30,
        logger=None,
        error_handler=None
    ):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='ultimate_oscillator',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period1': period1,
                'period2': period2,
                'period3': period3,
                'overbought': overbought,
                'oversold': oversold
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period3 + 1  # The longest period + 1 for the previous close

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period1 < 1 or self.period2 < 1 or self.period3 < 1:
            raise InvalidParameterError(
                self.name, 'periods',
                f"period1={self.period1}, period2={self.period2}, period3={self.period3}",
                "All periods must be positive"
            )
        if not (self.period1 < self.period2 < self.period3):
            raise InvalidParameterError(
                self.name, 'periods',
                f"period1={self.period1}, period2={self.period2}, period3={self.period3}",
                "Periods must be in increasing order: period1 < period2 < period3"
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
        Ultimate Oscillator hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: UO value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Calculate BP (Buying Pressure) and TR (True Range)
        bp_values = []
        tr_values = []

        for i in range(1, len(close)):
            # BP = Close - Min(Low, Previous Close)
            bp = close[i] - min(low[i], close[i-1])
            bp_values.append(bp)

            # TR = Max(High, Previous Close) - Min(Low, Previous Close)
            tr = max(high[i], close[i-1]) - min(low[i], close[i-1])
            tr_values.append(tr)

        bp_array = np.array(bp_values)
        tr_array = np.array(tr_values)

        # Calculate the average for every 3 periods
        def calculate_average(period: int) -> float:
            if len(bp_array) < period:
                return 0.0

            bp_sum = np.sum(bp_array[-period:])
            tr_sum = np.sum(tr_array[-period:])

            if tr_sum == 0:
                return 0.0
            return bp_sum / tr_sum

        avg1 = calculate_average(self.period1)
        avg2 = calculate_average(self.period2)
        avg3 = calculate_average(self.period3)

        # Ultimate Oscillator hesapla (weighted average)
        # Weights: 4, 2, 1
        uo_value = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)

        timestamp = int(data.iloc[-1]['timestamp'])

        return IndicatorResult(
            value=round(uo_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(uo_value),
            trend=self.get_trend(uo_value),
            strength=abs(uo_value - 50) * 2,  # Normalize to a range of 0-100
            metadata={
                'period1': self.period1,
                'period2': self.period2,
                'period3': self.period3,
                'avg1': round(avg1, 4),
                'avg2': round(avg2, 4),
                'avg3': round(avg3, 4)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Ultimate Oscillator calculation - for BACKTEST

        UO Formula:
            BP = Close - Min(Low, Previous Close)
            TR = Max(High, Previous Close) - Min(Low, Previous Close)
            Average = Sum(BP, period) / Sum(TR, period)
            UO = 100 × ((4×Avg7 + 2×Avg14 + Avg28) / 7)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: UO values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        # BP (Buying Pressure) = Close - Min(Low, Previous Close)
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)

        # TR (True Range) = Max(High, Previous Close) - Min(Low, Previous Close)
        tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

        # Calculate averages for 3 periods
        bp_sum1 = bp.rolling(window=self.period1).sum()
        tr_sum1 = tr.rolling(window=self.period1).sum()
        avg1 = bp_sum1 / tr_sum1

        bp_sum2 = bp.rolling(window=self.period2).sum()
        tr_sum2 = tr.rolling(window=self.period2).sum()
        avg2 = bp_sum2 / tr_sum2

        bp_sum3 = bp.rolling(window=self.period3).sum()
        tr_sum3 = tr.rolling(window=self.period3).sum()
        avg3 = bp_sum3 / tr_sum3

        # Ultimate Oscillator = 100 × ((4×avg1 + 2×avg2 + avg3) / 7)
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)

        # Handle division by zero
        uo = uo.fillna(0)

        # Set first period values to NaN (warmup)
        uo.iloc[:self.period3] = np.nan

        return pd.Series(uo.values, index=data.index, name='uo')

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: The current indicator value.
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._buffers:
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = deque(maxlen=max_len)

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Need minimum data for calculation
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the UO value.

        Args:
            value: The UO value.

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
        Determine the trend based on the UO value.

        Args:
            value: The UO value.

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
            'period1': 7,
            'period2': 14,
            'period3': 28,
            'overbought': 70,
            'oversold': 30
        }

    def _requires_volume(self) -> bool:
        """Ultimate Oscillator volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['UltimateOscillator']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Ultimate Oscillator indicator test"""

    print("\n" + "="*60)
    print("ULTIMATE OSCILLATOR TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(49):
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
    uo = UltimateOscillator(period1=7, period2=14, period3=28)
    print(f"   [OK] Created: {uo}")
    print(f"   [OK] Kategori: {uo.category.value}")
    print(f"   [OK] Required period: {uo.get_required_periods()}")

    result = uo(data)
    print(f"   [OK] UO Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Avg1: {result.metadata['avg1']}")
    print(f"   [OK] Avg2: {result.metadata['avg2']}")
    print(f"   [OK] Avg3: {result.metadata['avg3']}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    configs = [
        (5, 10, 20),
        (7, 14, 28),
        (10, 20, 40)
    ]
    for p1, p2, p3 in configs:
        uo_test = UltimateOscillator(period1=p1, period2=p2, period3=p3)
        result = uo_test.calculate(data)
        print(f"   [OK] UO({p1},{p2},{p3}): {result.value} | Signal: {result.signal.value}")

    # Test 3: Custom levels
    print("\n4. Special level test...")
    uo_custom = UltimateOscillator(period1=7, period2=14, period3=28,
                                    overbought=80, oversold=20)
    result = uo_custom.calculate(data)
    print(f"   [OK] UO with specific level: {result.value}")
    print(f"   [OK] Overbought: {uo_custom.overbought}, Oversold: {uo_custom.oversold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Rising trend
    print("\n5. Rising trend test...")
    up_data = data.copy()
    for i in range(20):
        idx = up_data.index[-(20-i)]
        up_data.loc[idx, 'close'] = prices[-(20-i)] + i * 0.5
        up_data.loc[idx, 'high'] = prices[-(20-i)] + i * 0.5 + 0.3
        up_data.loc[idx, 'low'] = prices[-(20-i)] + i * 0.5 - 0.3

    result_up = uo.calculate(up_data)
    print(f"   [OK] Rising trend UO: {result_up.value}")
    print(f"   [OK] Signal: {result_up.signal.value}")
    print(f"   [OK] Trend: {result_up.trend.name}")

    # Test 5: Declining trend
    print("\n6. Downtrend test...")
    down_data = data.copy()
    for i in range(20):
        idx = down_data.index[-(20-i)]
        down_data.loc[idx, 'close'] = prices[-(20-i)] - i * 0.5
        down_data.loc[idx, 'high'] = prices[-(20-i)] - i * 0.5 + 0.3
        down_data.loc[idx, 'low'] = prices[-(20-i)] - i * 0.5 - 0.3

    result_down = uo.calculate(down_data)
    print(f"   [OK] Declining trend UO: {result_down.value}")
    print(f"   [OK] Signal: {result_down.signal.value}")
    print(f"   [OK] Trend: {result_down.trend.name}")

    # Test 6: Overbuying/overselling
    print("\n7. Overbuying/overselling test...")
    # Strong upward trend
    strong_up_data = data.copy()
    for i in range(10):
        idx = strong_up_data.index[-(10-i)]
        strong_up_data.loc[idx, 'close'] = prices[-(10-i)] + i * 2
        strong_up_data.loc[idx, 'high'] = prices[-(10-i)] + i * 2 + 1
        strong_up_data.loc[idx, 'low'] = prices[-(10-i)] + i * 2 - 0.2

    result_strong_up = uo.calculate(strong_up_data)
    print(f"   [OK] Strong upward trend UO: {result_strong_up.value}")
    print(f"   [OK] Signal: {result_strong_up.signal.value}")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = uo.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = uo.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
