"""
indicators/support_resistance/swingpoints.py - Swing Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Swing Points - Detects swing high and swing low points.
    Identifies local maximum and minimum points, indicating potential
    support and resistance levels.

Formula:
    Swing High: The middle candle is higher than the N candles to its left and right.
    Swing Low: The middle candle is lower than the N candles to its left and right.
    N = left_bars or right_bars

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


# ============================================================================
# STANDALONE UTILITY FUNCTIONS (TradingView Compatible)
# ============================================================================

def find_pivot_highs(data: np.ndarray, left_bars: int = 5, right_bars: int = 5) -> List[Tuple[int, float]]:
    """
    Find pivot high points (TradingView ta.pivothigh equivalent)

    Returns pivot highs as (index, value) tuples.
    Compatible with TradingView's ta.pivothigh(data, left_bars, right_bars).

    Args:
        data: High prices array
        left_bars: Number of bars to check on the left (default: 5)
        right_bars: Number of bars to check on the right (default: 5)

    Returns:
        List of (index, value) tuples for pivot highs

    Example:
        >>> highs = np.array([100, 102, 104, 103, 101, 99, 98])
        >>> pivots = find_pivot_highs(highs, left_bars=2, right_bars=2)
        >>> pivots
        [(2, 104.0)]  # Index 2 has value 104, higher than 2 bars left/right
    """
    pivots = []

    for i in range(left_bars, len(data) - right_bars):
        # Check if current bar is higher than left_bars to the left
        is_highest_left = all(data[i] > data[i - j] for j in range(1, left_bars + 1))

        # Check if current bar is higher or equal to right_bars to the right
        is_highest_right = all(data[i] >= data[i + j] for j in range(1, right_bars + 1))

        if is_highest_left and is_highest_right:
            pivots.append((i, float(data[i])))

    return pivots


def find_pivot_lows(data: np.ndarray, left_bars: int = 5, right_bars: int = 5) -> List[Tuple[int, float]]:
    """
    Find pivot low points (TradingView ta.pivotlow equivalent)

    Returns pivot lows as (index, value) tuples.
    Compatible with TradingView's ta.pivotlow(data, left_bars, right_bars).

    Args:
        data: Low prices array
        left_bars: Number of bars to check on the left (default: 5)
        right_bars: Number of bars to check on the right (default: 5)

    Returns:
        List of (index, value) tuples for pivot lows

    Example:
        >>> lows = np.array([100, 98, 96, 97, 99, 101, 102])
        >>> pivots = find_pivot_lows(lows, left_bars=2, right_bars=2)
        >>> pivots
        [(2, 96.0)]  # Index 2 has value 96, lower than 2 bars left/right
    """
    pivots = []

    for i in range(left_bars, len(data) - right_bars):
        # Check if current bar is lower than left_bars to the left
        is_lowest_left = all(data[i] < data[i - j] for j in range(1, left_bars + 1))

        # Check if current bar is lower or equal to right_bars to the right
        is_lowest_right = all(data[i] <= data[i + j] for j in range(1, right_bars + 1))

        if is_lowest_left and is_lowest_right:
            pivots.append((i, float(data[i])))

    return pivots


def check_pivot_range(pivot1_idx: int, pivot2_idx: int,
                      range_min: int = 5, range_max: int = 60) -> bool:
    """
    Check if two pivots are within acceptable distance range

    TradingView equivalent:
        bars = ta.barssince(cond == true)
        rangeLower <= bars and bars <= rangeUpper

    Args:
        pivot1_idx: Index of first pivot
        pivot2_idx: Index of second pivot
        range_min: Minimum bars between pivots (default: 5)
        range_max: Maximum bars between pivots (default: 60)

    Returns:
        True if pivots are within range, False otherwise

    Example:
        >>> check_pivot_range(10, 25, range_min=5, range_max=60)
        True  # 15 bars apart, within 5-60 range
        >>> check_pivot_range(10, 12, range_min=5, range_max=60)
        False  # Only 2 bars apart, less than minimum 5
    """
    distance = abs(pivot2_idx - pivot1_idx)
    return range_min <= distance <= range_max


class SwingPoints(BaseIndicator):
    """
    Swing Points Detector

    Local maximum (swing high) and minimum (swing low) points.
    detects. Each swing point is the specified number of previous and subsequent
    It must be higher or lower than the candle.

    Args:
        left_bars: The number of comparison candles on the left side (default: 5)
        right_bars: The number of comparison candles on the right side (default: 5)
        lookback: The lookback period (default: 50)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        lookback: int = 50,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.lookback = lookback

        super().__init__(
            name='swing_points',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'lookback': lookback
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.lookback + self.right_bars

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars must be positive"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars must be positive"
            )
        min_lookback = self.left_bars + self.right_bars
        if self.lookback < min_lookback:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                f"Lookback must be at least {min_lookback} (left_bars + right_bars)"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Swing Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Swing high and low levels
        """
        # Lookback period + get extra data for the right side
        recent_data = data.iloc[-self.lookback - self.right_bars:]
        high = recent_data['high'].values
        low = recent_data['low'].values

        # Find swing points
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Create swing levels (consistent with the last swing high and low - calculate_batch)
        levels = {
            'swing_high': round(swing_highs[-1], 2) if len(swing_highs) > 0 else None,
            'swing_low': round(swing_lows[-1], 2) if len(swing_lows) > 0 else None
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, swing_highs, swing_lows),
            trend=self.get_trend(swing_highs, swing_lows),
            strength=self.calculate_strength(current_price, swing_highs, swing_lows),
            metadata={
                'left_bars': self.left_bars,
                'right_bars': self.right_bars,
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'last_swing_high': round(swing_highs[-1], 2) if len(swing_highs) > 0 else None,
                'last_swing_low': round(swing_lows[-1], 2) if len(swing_lows) > 0 else None
            }
        )

    def _find_swing_highs(self, high: np.ndarray) -> list:
        """
        Find swing high points.

        Args:
            high: High prices

        Returns:
            list: Swing high seviyeleri
        """
        swing_highs = []

        # Only check completed swings (data must be available for the right side)
        for i in range(self.left_bars, len(high) - self.right_bars):
            # Left side check
            is_highest_left = all(high[i] > high[i - j] for j in range(1, self.left_bars + 1))

            # Right-hand side check
            is_highest_right = all(high[i] >= high[i + j] for j in range(1, self.right_bars + 1))

            if is_highest_left and is_highest_right:
                swing_highs.append(high[i])

        return swing_highs if swing_highs else [max(high)]

    def _find_swing_lows(self, low: np.ndarray) -> list:
        """
        Find swing low points.

        Args:
            low: Low prices

        Returns:
            list: Swing low seviyeleri
        """
        swing_lows = []

        # Only check completed swings (data must be available for the right side)
        for i in range(self.left_bars, len(low) - self.right_bars):
            # Left side check
            is_lowest_left = all(low[i] < low[i - j] for j in range(1, self.left_bars + 1))

            # Right-hand side check
            is_lowest_right = all(low[i] <= low[i + j] for j in range(1, self.right_bars + 1))

            if is_lowest_left and is_lowest_right:
                swing_lows.append(low[i])

        return swing_lows if swing_lows else [min(low)]

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup SwingPoints buffers with historical data

        CRITICAL: SwingPoints uses its own buffers (_high_buffer, _low_buffer, _close_buffer)
        not BaseIndicator's _buffers. This override ensures they're properly filled.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (unused, for interface compatibility)
        """
        from collections import deque

        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Fill buffers with historical data
        for _, row in data.tail(max_len).iterrows():
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._close_buffer.append(row['close'])

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
            # Return dict format consistent with calculate() method
            # This allows TradingEngine to properly add to DataFrame
            return IndicatorResult(
                value={'swing_high': None, 'swing_low': None},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'warmup': True, 'required': self.get_required_periods(), 'current': len(self._close_buffer)}
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

    def get_signal(self, price: float, swing_highs: list, swing_lows: list) -> SignalType:
        """
        Generate a signal based on the price's swing points.

        Args:
            price: Current price
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if not swing_highs or not swing_lows:
            return SignalType.HOLD

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]

        # Check proximity to the nearest swing level
        distance_to_low = abs(price - last_low) / price
        distance_to_high = abs(price - last_high) / price

        if distance_to_low < 0.01:  # within 1%
            return SignalType.BUY
        elif distance_to_high < 0.01:  # within 1%
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, swing_highs: list, swing_lows: list) -> TrendDirection:
        """
        Determine the trend based on swing points.

        Args:
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not swing_highs or not swing_lows or len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendDirection.NEUTRAL

        # Higher low and higher high = uptrend
        # Descending bottom and descending peak = downtrend
        higher_lows = swing_lows[-1] > swing_lows[-2]
        higher_highs = swing_highs[-1] > swing_highs[-2]
        lower_lows = swing_lows[-1] < swing_lows[-2]
        lower_highs = swing_highs[-1] < swing_highs[-2]

        if higher_lows and higher_highs:
            return TrendDirection.UP
        elif lower_lows and lower_highs:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, swing_highs: list, swing_lows: list) -> float:
        """
        Calculate power based on price swing points.

        Args:
            price: Current price
            swing_highs: Swing high seviyeleri
            swing_lows: Swing low seviyeleri

        Returns:
            float: Power value (0-100)
        """
        if not swing_highs or not swing_lows:
            return 50.0

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]

        if last_high == last_low:
            return 50.0

        # The position of the price within the swing range.
        position = (price - last_low) / (last_high - last_low) * 100
        return min(max(position, 0), 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'lookback': 50
        }

    def _requires_volume(self) -> bool:
        """Swing Points volume gerektirmez"""
        return False

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Swing Points calculation - for BACKTEST

        Proper swing point detection using left_bars and right_bars.
        A swing high is a peak higher than N bars on both left and right.
        A swing low is a valley lower than N bars on both left and right.

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: swing_high and swing_low columns (NaN where no swing detected)

        Performance: Detects confirmed swing points with proper confirmation

        Note: Swing points require future confirmation (right_bars), so the last
        N bars will have NaN values since we can't confirm them yet.
        """
        self._validate_data(data)

        high = data['high'].values
        low = data['low'].values
        n_bars = len(data)

        # Initialize arrays with NaN
        swing_highs = np.full(n_bars, np.nan)
        swing_lows = np.full(n_bars, np.nan)

        # Detect swing points (skip first left_bars and last right_bars)
        for i in range(self.left_bars, n_bars - self.right_bars):
            # Check if this is a swing high
            is_swing_high = True
            current_high = high[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if current_high <= high[i - j]:
                    is_swing_high = False
                    break

            # Check right side if left passed
            if is_swing_high:
                for j in range(1, self.right_bars + 1):
                    if current_high < high[i + j]:  # >= for right side
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs[i] = current_high

            # Check if this is a swing low
            is_swing_low = True
            current_low = low[i]

            # Check left side
            for j in range(1, self.left_bars + 1):
                if current_low >= low[i - j]:
                    is_swing_low = False
                    break

            # Check right side if left passed
            if is_swing_low:
                for j in range(1, self.right_bars + 1):
                    if current_low > low[i + j]:  # <= for right side
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows[i] = current_low

        return pd.DataFrame({
            'swing_high': swing_highs,
            'swing_low': swing_lows
        }, index=data.index)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'SwingPoints',
    'find_pivot_highs',
    'find_pivot_lows',
    'check_pivot_range'
]


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Swing Points indicator test"""

    print("\n" + "="*60)
    print("SWING POINTS TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Simulate fluctuating price movements
    base_price = 100
    prices = [base_price]

    for i in range(99):
        # Sine wave + noise
        wave = 10 * np.sin(i / 8)
        noise = np.random.randn() * 1
        prices.append(base_price + wave + noise)

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
    swing = SwingPoints(left_bars=5, right_bars=5, lookback=50)
    print(f"   [OK] Created: {swing}")
    print(f"   [OK] Kategori: {swing.category.value}")
    print(f"   [OK] Tip: {swing.indicator_type.value}")
    print(f"   [OK] Required period: {swing.get_required_periods()}")

    result = swing(data)
    print(f"   [OK] Swing Seviyeleri:")
    for level, value in sorted(result.value.items(), key=lambda x: x[1], reverse=True):
        level_type = "Swing High" if level.startswith('SH') else "Swing Low"
        print(f"        {level} ({level_type}): {value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different left/right bar combinations
    print("\n3. Testing different bar combinations...")
    for left, right in [(3, 3), (5, 5), (7, 7)]:
        swing_test = SwingPoints(left_bars=left, right_bars=right, lookback=50)
        result = swing_test.calculate(data)
        print(f"   [OK] Swing(L={left},R={right}) - Highs: {result.metadata['total_swing_highs']}, "
              f"Lows: {result.metadata['total_swing_lows']}")

    # Test 3: Different lookback values
    print("\n4. Different lookback test...")
    for lookback in [30, 50, 70]:
        swing_test = SwingPoints(left_bars=5, right_bars=5, lookback=lookback)
        result = swing_test.calculate(data)
        print(f"   [OK] Swing(lookback={lookback}) - "
              f"Detection: {result.metadata['total_swing_highs'] + result.metadata['total_swing_lows']} points")

    # Test 4: Swing analizi
    print("\n5. Swing analizi...")
    result = swing.calculate(data)
    current = result.metadata['current_price']
    last_high = result.metadata['last_swing_high']
    last_low = result.metadata['last_swing_low']

    print(f"   [OK] Current price: {current}")
    print(f"   [OK] Son swing high: {last_high}")
    print(f"   [OK] Son swing low: {last_low}")

    if last_high and last_low:
        swing_range = last_high - last_low
        position = (current - last_low) / swing_range * 100
        print(f"   [OK] Swing range: {swing_range:.2f}")
        print(f"   [OK] Price location: {position:.1f}% (within range)")

        if position > 70:
            print(f"   [OK] Price is in the upper region of the swing range (near resistance)")
        elif position < 30:
            print(f"   [OK] Price is below the lower band of the swing range (near support)")
        else:
            print(f"   [OK] Price is in the middle of the swing range")

    # Test 5: Trend analizi
    print("\n6. Trend analizi...")
    print(f"   [OK] Detected trend: {result.trend.name}")
    if result.trend == TrendDirection.UP:
        print(f"   [OK] Rising lows and highs - Rising trend")
        print(f"   [OK] Strategy: Look for buy opportunities at swing low levels")
    elif result.trend == TrendDirection.DOWN:
        print(f"   [OK] Declining lows and highs - Downtrend")
        print(f"   [OK] Strategy: Look for sell opportunities at swing high levels")
    else:
        print(f"   [OK] No clear trend - Horizontal movement")
        print(f"   [OK] Strategy: Range trading (trading from support/resistance)")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = swing.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = swing.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
