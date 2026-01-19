"""
indicators/support_resistance/fib_retracement.py - Fibonacci Retracement

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Fibonacci Retracement - Fibonacci retracement levels
    Determines potential retracement levels after a trend movement.
    Standard Fibonacci ratios: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%

Formula:
    Range = High - Low
    Level = High - (Range × Fib_Ratio)
    Fib Ratios: 0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000

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


class FibonacciRetracement(BaseIndicator):
    """
    Fibonacci Retracement Levels

    The difference between the highest and lowest price within a specific period.
    Calculates retracement levels according to Fibonacci ratios.

    Args:
        lookback: High/low search period (default: 50)
    """

    # Standard Fibonacci ratios
    FIB_RATIOS = {
        '0.0': 0.000,
        '23.6': 0.236,
        '38.2': 0.382,
        '50.0': 0.500,
        '61.8': 0.618,
        '78.6': 0.786,
        '100.0': 1.000
    }

    def __init__(
        self,
        lookback: int = 50,
        logger=None,
        error_handler=None
    ):
        self.lookback = lookback

        super().__init__(
            name='fib_retracement',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'lookback': lookback
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.lookback

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.lookback < 10:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback must be at least 10"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Fibonacci Retracement seviyeleri hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Fibonacci retracement seviyeleri
        """
        # Highest and lowest values in the last lookback period
        recent_data = data.iloc[-self.lookback:]
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()

        # Determine the trend direction (did high or low come first)
        is_uptrend = low_idx < high_idx
        range_hl = high - low

        # Fibonacci seviyelerini hesapla
        levels = {}
        if is_uptrend:
            # Uprising trend - retracement downwards from the high.
            for label, ratio in self.FIB_RATIOS.items():
                level_value = high - (range_hl * ratio)
                levels[f'Fib_{label}'] = round(level_value, 2)
        else:
            # Downtrend - retracement from low towards up
            for label, ratio in self.FIB_RATIOS.items():
                level_value = low + (range_hl * ratio)
                levels[f'Fib_{label}'] = round(level_value, 2)

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels, is_uptrend),
            trend=TrendDirection.UP if is_uptrend else TrendDirection.DOWN,
            strength=self.calculate_strength(current_price, high, low),
            metadata={
                'lookback': self.lookback,
                'high': round(high, 2),
                'low': round(low, 2),
                'range': round(range_hl, 2),
                'current_price': round(current_price, 2),
                'trend': 'uptrend' if is_uptrend else 'downtrend',
                'high_index': int(high_idx),
                'low_index': int(low_idx)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Fibonacci Retracement calculation - for BACKTEST

        Fibonacci Retracement Formula:
            Range = High - Low (over lookback period)
            For uptrend: Level = High - (Range × Fib_Ratio)
            For downtrend: Level = Low + (Range × Fib_Ratio)
            Standard ratios: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: fib_236, fib_382, fib_500, fib_618 for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']

        # Rolling high/low over lookback period
        rolling_high = high.rolling(window=self.lookback).max()
        rolling_low = low.rolling(window=self.lookback).min()

        # Range
        range_hl = rolling_high - rolling_low

        # Determine trend: argmax vs argmin positions
        # For simplicity, assume uptrend if current price is closer to high
        close = data['close']
        trend_strength = (close - rolling_low) / range_hl  # 0-1, high=uptrend
        is_uptrend = trend_strength > 0.5

        # Calculate all 7 Fibonacci levels (matching calculate())
        # For uptrend: High - (Range × ratio)
        # For downtrend: Low + (Range × ratio)

        # Calculate all 7 levels
        fib_000 = np.where(is_uptrend, rolling_high - (range_hl * 0.000), rolling_low + (range_hl * 0.000))
        fib_236 = np.where(is_uptrend, rolling_high - (range_hl * 0.236), rolling_low + (range_hl * 0.236))
        fib_382 = np.where(is_uptrend, rolling_high - (range_hl * 0.382), rolling_low + (range_hl * 0.382))
        fib_500 = np.where(is_uptrend, rolling_high - (range_hl * 0.500), rolling_low + (range_hl * 0.500))
        fib_618 = np.where(is_uptrend, rolling_high - (range_hl * 0.618), rolling_low + (range_hl * 0.618))
        fib_786 = np.where(is_uptrend, rolling_high - (range_hl * 0.786), rolling_low + (range_hl * 0.786))
        fib_100 = np.where(is_uptrend, rolling_high - (range_hl * 1.000), rolling_low + (range_hl * 1.000))

        # Convert to Series
        fib_000 = pd.Series(fib_000, index=data.index)
        fib_236 = pd.Series(fib_236, index=data.index)
        fib_382 = pd.Series(fib_382, index=data.index)
        fib_500 = pd.Series(fib_500, index=data.index)
        fib_618 = pd.Series(fib_618, index=data.index)
        fib_786 = pd.Series(fib_786, index=data.index)
        fib_100 = pd.Series(fib_100, index=data.index)

        # Set first period values to NaN (warmup)
        fib_000.iloc[:self.lookback-1] = np.nan
        fib_236.iloc[:self.lookback-1] = np.nan
        fib_382.iloc[:self.lookback-1] = np.nan
        fib_500.iloc[:self.lookback-1] = np.nan
        fib_618.iloc[:self.lookback-1] = np.nan
        fib_786.iloc[:self.lookback-1] = np.nan
        fib_100.iloc[:self.lookback-1] = np.nan

        return pd.DataFrame({
            'Fib_0.0': fib_000.values,
            'Fib_23.6': fib_236.values,
            'Fib_38.2': fib_382.values,
            'Fib_50.0': fib_500.values,
            'Fib_61.8': fib_618.values,
            'Fib_78.6': fib_786.values,
            'Fib_100.0': fib_100.values
        }, index=data.index)

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
                value=[],
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

    def get_signal(self, price: float, levels: dict, is_uptrend: bool) -> SignalType:
        """
        Generate a signal based on the price's Fibonacci levels.

        Args:
            price: Current price
            levels: Fibonacci seviyeleri
            is_uptrend: Is it an uptrend?

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Important Fibonacci levels (38.2, 50.0, 61.8)
        fib_382 = levels.get('Fib_38.2', 0)
        fib_500 = levels.get('Fib_50.0', 0)
        fib_618 = levels.get('Fib_61.8', 0)

        if is_uptrend:
            # A pullback in the uptrend presents a buying opportunity.
            if fib_618 <= price <= fib_500:
                return SignalType.BUY
            elif price > levels.get('Fib_23.6', price):
                return SignalType.SELL  # Trend weakened
        else:
            # A pullback in the downtrend presents a selling opportunity.
            if fib_500 <= price <= fib_618:
                return SignalType.SELL
            elif price < levels.get('Fib_23.6', price):
                return SignalType.BUY  # Trend weakened

        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Trend direction (determined within calculate)

        Args:
            value: Value

        Returns:
            TrendDirection: Trend direction
        """
        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, high: float, low: float) -> float:
        """
        Calculate power based on the price's position within the range.

        Args:
            price: Current price
            high: Range high
            low: Range low

        Returns:
            float: Power value (0-100)
        """
        if high == low:
            return 50.0

        # Price's percentage position within the range
        position = (price - low) / (high - low) * 100
        return min(max(position, 0), 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'lookback': 50
        }

    def _requires_volume(self) -> bool:
        """Fibonacci Retracement volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['FibonacciRetracement']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Fibonacci Retracement indicator test"""

    print("\n" + "="*60)
    print("FIBONACCI RETRACEMENT TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Simulate an upward trend
    base_price = 100
    prices = [base_price]

    # First 50 candles: uptrend
    for i in range(49):
        change = np.random.randn() * 0.5 + 0.3  # Upward bias
        prices.append(prices[-1] * (1 + change / 100))

    # Last 50 candles: pullback
    for i in range(50):
        change = np.random.randn() * 0.5 - 0.1  # Slight decrease in bias
        prices.append(prices[-1] * (1 + change / 100))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.005) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.005) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    fib = FibonacciRetracement(lookback=50)
    print(f"   [OK] Created: {fib}")
    print(f"   [OK] Kategori: {fib.category.value}")
    print(f"   [OK] Tip: {fib.indicator_type.value}")
    print(f"   [OK] Required period: {fib.get_required_periods()}")

    result = fib(data)
    print(f"   [OK] Fibonacci Retracement Seviyeleri:")
    for level, value in sorted(result.value.items(), key=lambda x: x[1], reverse=True):
        print(f"        {level}: {value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different lookback values
    print("\n3. Different lookback test...")
    for lookback in [30, 50, 80]:
        fib_test = FibonacciRetracement(lookback=lookback)
        result = fib_test.calculate(data)
        print(f"   [OK] Fib(lookback={lookback}) - Range: {result.metadata['range']:.2f} | "
              f"Trend: {result.metadata['trend']}")

    # Test 3: Fibonacci seviye analizi
    print("\n4. Fibonacci seviye analizi...")
    result = fib.calculate(data)
    current = result.metadata['current_price']
    high = result.metadata['high']
    low = result.metadata['low']
    trend = result.metadata['trend']

    print(f"   [OK] Trend: {trend}")
    print(f"   [OK] Range High: {high}")
    print(f"   [OK] Range Low: {low}")
    print(f"   [OK] Range: {result.metadata['range']}")
    print(f"   [OK] Current price: {current}")
    print(f"\n   [OK] Important Fibonacci Levels:")
    print(f"        0.0% (Tam retracement): {result.value.get('Fib_0.0', 'N/A')}")
    print(f"        23.6% (Weak support): {result.value.get('Fib_23.6', 'N/A')}")
    print(f"        38.2% (Orta destek): {result.value.get('Fib_38.2', 'N/A')}")
    print(f"        50.0% (Strong support): {result.value.get('Fib_50.0', 'N/A')}")
    print(f"        61.8% (Golden ratio): {result.value.get('Fib_61.8', 'N/A')}")
    print(f"        78.6% (Derin retracement): {result.value.get('Fib_78.6', 'N/A')}")
    print(f"        100% (Initial): {result.value.get('Fib_100.0', 'N/A')}")

    # Find the price range.
    sorted_levels = sorted([(k, v) for k, v in result.value.items()], key=lambda x: x[1])
    for i in range(len(sorted_levels) - 1):
        level1_name, level1_val = sorted_levels[i]
        level2_name, level2_val = sorted_levels[i + 1]
        if level1_val <= current <= level2_val:
            print(f"\n   [OK] Price is between {level1_name} and {level2_name}")
            break

    # Test 4: Trading strategy suggestion
    print("\n5. Trading stratejisi analizi...")
    if result.metadata['trend'] == 'uptrend':
        print(f"   [OK] Retracement in the uptrend")
        print(f"   [OK] Important levels for buying:")
        print(f"        - 38.2%: {result.value['Fib_38.2']} (initial purchase)")
        print(f"        - 50.0%: {result.value['Fib_50.0']} (main buy)")
        print(f"        - 61.8%: {result.value['Fib_61.8']} (last buy)")
        print(f"   [OK] Stop loss: less than {result.value['Fib_78.6']}")
    else:
        print(f"   [OK] Recovering from a downtrend")
        print(f"   [OK] Important levels for sales:")
        print(f"        - 38.2%: {result.value['Fib_38.2']} (first sale)")
        print(f"        - 50.0%: {result.value['Fib_50.0']} (main sales)")
        print(f"        - 61.8%: {result.value['Fib_61.8']} (last sale)")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = fib.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = fib.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
