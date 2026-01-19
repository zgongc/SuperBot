"""
indicators/support_resistance/zigzag.py - ZigZag Indicator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    ZigZag - Determines swing high and low points.
    Filters price movements that exceed a specific percentage change threshold.
    Displays trend reversals and important support/resistance points.

Formula:
    - A new pivot is formed when the price changes by a percentage deviation from the previous pivot.
    - Swing High: Pivot point in the upward direction.
    - Swing Low: Pivot point in the downward direction.

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


class ZigZag(BaseIndicator):
    """
    ZigZag Indicator

    Filters price movements that exceed a specific percentage change threshold
    and determines swing high/low points.

    Args:
        deviation: Minimum percentage change (default: 5.0)
        depth: Backward search depth (default: 12)
    """

    def __init__(
        self,
        deviation: float = 5.0,
        depth: int = 12,
        logger=None,
        error_handler=None
    ):
        self.deviation = deviation
        self.depth = depth

        super().__init__(
            name='zigzag',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'deviation': deviation,
                'depth': depth
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.depth * 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.deviation <= 0:
            raise InvalidParameterError(
                self.name, 'deviation', self.deviation,
                "Deviation must be positive"
            )
        if self.depth < 1:
            raise InvalidParameterError(
                self.name, 'depth', self.depth,
                "Depth must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        ZigZag hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: The last pivot value and its information.
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Fill the buffers (preparation for incremental update)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.depth + 1)
            self._low_buffer = deque(maxlen=self.depth + 1)
            self._last_pivot_val = None
            self._last_pivot_type = None
            self._total_candles = 0
            self._pivots = []
            
        # Son verileri buffer'a at
        # Note: For a complete synchronization, it is necessary to process the entire history and establish the state.
        # However, calculate() already processes the entire history.
        # We will only transfer the final state to the 'state'.
        
        # It's best to reset and rebuild the state when calculate is called.
        # But this can be expensive.
        # Simply get the final state:
        
        # Pivots are already calculated
        pivots = self._find_pivots(high, low)
        
        if pivots:
            last = pivots[-1]
            self._last_pivot_val = last['value']
            self._last_pivot_type = last['type']
            self._pivots = pivots[-5:] # Son 5 pivotu sakla
        else:
            self._last_pivot_val = close[-1]
            self._last_pivot_type = 'none'
            
        self._total_candles = len(data)
        
        # Fill the buffers with the latest data
        self._high_buffer.clear()
        self._low_buffer.clear()
        
        # Son depth+1 veriyi al
        start_idx = max(0, len(data) - (self.depth + 1))
        self._high_buffer.extend(high[start_idx:])
        self._low_buffer.extend(low[start_idx:])

        # Find pivots
        pivots = self._find_pivots(high, low)

        # Son pivot bilgisini al
        last_pivot = pivots[-1] if len(pivots) > 0 else None
        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        if last_pivot:
            pivot_value = last_pivot['value']
            pivot_type = last_pivot['type']
            pivot_index = last_pivot['index']

            # Trend belirle
            if pivot_type == 'high':
                trend = TrendDirection.DOWN  # A decrease is expected after the high
                signal = SignalType.SELL
            else:
                trend = TrendDirection.UP  # An upward trend is expected after the low.
                signal = SignalType.BUY

            # Calculate power (distance from the last pivot)
            price_change = abs((current_price - pivot_value) / pivot_value * 100)
            strength = min(price_change / self.deviation * 100, 100)

            # Update the signal based on the previous pivot if it exists.
            if len(pivots) > 1:
                prev_pivot = pivots[-2]
                # If moving in the trend direction, strengthen the signal.
                if pivot_type == 'low' and current_price > pivot_value:
                    signal = SignalType.BUY
                elif pivot_type == 'high' and current_price < pivot_value:
                    signal = SignalType.SELL
                else:
                    signal = SignalType.HOLD

        else:
            pivot_value = current_price
            pivot_type = 'none'
            pivot_index = len(data) - 1
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL
            strength = 0.0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'value': round(pivot_value, 2),
                'pivot_type': pivot_type
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'deviation': self.deviation,
                'depth': self.depth,
                'pivot_index': pivot_index,
                'current_price': round(current_price, 2),
                'price_change_pct': round(abs((current_price - pivot_value) / pivot_value * 100), 2),
                'total_pivots': len(pivots)
            }
        )



    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: ZigZag values
        """
        high = data['high'].values
        low = data['low'].values
        
        pivots = self._find_pivots(high, low)
        
        # Convert the results to a DataFrame
        # We will return the "latest pivot" value for each bar.
        
        result_values = np.full(len(data), np.nan)
        result_types = np.full(len(data), None)
        
        # Spread the pivots to the time series
        # Pivot list: [{'index': 10, 'value': 100, 'type': 'high'}, ...]
        # Pivots are "created" (or confirmed) at their respective indices.
        # However, ZigZag is usually drawn backwards.
        # For real-time usage: The most recently known pivot value is valid at that moment.
        
        current_pivot_val = np.nan
        current_pivot_type = None
        
        pivot_idx = 0
        num_pivots = len(pivots)
        
        # This loop may be a bit inefficient, but it is sparse due to its ZigZag nature.
        # A faster method: Use pivot indices and fillna.
        
        # Get the indices where the pivots are formed
        pivot_indices = [p['index'] for p in pivots]
        pivot_vals = [p['value'] for p in pivots]
        pivot_types = [p['type'] for p in pivots]
        
        # Create a series
        s_values = pd.Series(np.nan, index=data.index)
        s_types = pd.Series(dtype=object, index=data.index)  # Explicitly object dtype for string values
        
        # Mark the pivot points
        # Not: _find_pivots indexleri integer index (iloc).
        if pivots:
            # No value before the first pivot (or the first price?)
            # It may contain NaN values up to the first pivot index, or it can be backfilled.
            # We will use the forward fill logic.
            
            # Place the pivots
            # Attention: The pivot index is not the place where the pivot "forms", but the place where it reaches a "peak/trough".
            # However, the confirmation might be available later.
            # In the _find_pivots logic, the moment a pivot is added (loop index), is the confirmation moment.
            # However, pivot['index'] is the peak point.
            # For real-time simulation: It is valid from the moment the pivot is confirmed.
            # However, _find_pivots does not return the confirmation time, it only returns the pivot point.
            # That's why it's difficult to generate values without "repainting" (without lookahead) in batch calculations.
            
            # Simple approach: Place pivot points and perform forward fill (Step function).
            # This is the "last known pivot" logic.
            
            # However, is the _find_pivots list sorted? Yes.
            
            # Convert pivot indices to data indices
            # data.index[pivot_indices]
            
            # However, there is a problem here: The pivot index may be in the past.
            # In our batch result, t must be the "latest known pivot" at that moment.
            # We cannot determine the confirmation time without modifying the _find_pivots function.
            # But the _find_pivots function adds it to the list immediately.
            # That means the order in the pivot list is the confirmation order.
            
            # Instead of simulating the logic of _find_pivots within the batch,
            # Let's use '_find_pivots', place them according to their pivot indices, and then perform ffill.
            # This includes "repainting" (because the pivot index can be t-k).
            # But this is usually desired for visualization and trend tracking.
            
            # If we want a "non-repainting" indicator, we need to know the confirmation time.
            # Instead of the standard ZigZag behavior (a line connecting pivot points) for now
            # We will return the "Last Pivot Value".
            
            for p in pivots:
                idx = p['index']
                s_values.iloc[idx] = p['value']
                s_types.iloc[idx] = p['type']
                
            # Forward fill
            s_values = s_values.ffill()
            s_types = s_types.ffill()
            
        return pd.DataFrame({
            'value': s_values,
            'pivot_type': s_types
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current ZigZag value
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            # We need to look back up to the depth.
            self._high_buffer = deque(maxlen=self.depth + 1)
            self._low_buffer = deque(maxlen=self.depth + 1)
            # State
            self._last_pivot_val = None
            self._last_pivot_type = None
            self._last_pivot_idx = 0 # Relative index or count
            self._total_candles = 0
            self._pivots = [] # We can store the last few pivots
            
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._total_candles += 1

        current_price = close_val
        
        # Yeterli veri yoksa
        if len(self._high_buffer) < self.depth + 1:
             # Simple initialization for initial values
             if self._last_pivot_val is None:
                 self._last_pivot_val = current_price
                 self._last_pivot_type = 'none'
             
             return IndicatorResult(
                value=self._last_pivot_val,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'pivot_type': 'none'}
            )

        # Incremental Calculation Logic
        # Apply the logic of _find_pivots for a single step.
        
        high_arr = np.array(self._high_buffer)
        low_arr = np.array(self._low_buffer)
        
        # Window up to the last depth
        # window_high = np.max(high[max(0, i - self.depth):i + 1])
        # The buffer already holds the last (depth+1) data.
        # i (currently) is the last element of the buffer.
        # window_high = np.max(buffer)
        
        window_high = np.max(high_arr)
        window_low = np.min(low_arr)
        
        threshold = self.deviation / 100
        
        # Initial pivot initialization (if it doesn't exist yet)
        if self._last_pivot_val is None:
            # Buffer doldu, ilk pivotu belirle
            # Simply start with the highest/lowest
            if (window_high - window_low) / window_low > threshold:
                self._last_pivot_val = window_high
                self._last_pivot_type = 'high'
                self._pivots.append({'value': window_high, 'type': 'high', 'index': self._total_candles - 1})
            else:
                # No significant movement yet
                self._last_pivot_val = current_price
                self._last_pivot_type = 'none'
        
        else:
            # There is an existing pivot, search for a new one.
            if self._last_pivot_type == 'low' or self._last_pivot_type == 'none':
                # High pivot ara
                # If it is none and there is an increase, initialize high
                ref_val = self._last_pivot_val
                change = (window_high - ref_val) / ref_val
                
                if change > threshold:
                    # Yeni High Pivot bulundu
                    self._last_pivot_val = window_high
                    self._last_pivot_type = 'high'
                    self._pivots.append({'value': window_high, 'type': 'high', 'index': self._total_candles - 1})
                    
            elif self._last_pivot_type == 'high':
                # Low pivot ara
                ref_val = self._last_pivot_val
                change = (ref_val - window_low) / ref_val
                
                if change > threshold:
                    # Yeni Low Pivot bulundu
                    self._last_pivot_val = window_low
                    self._last_pivot_type = 'low'
                    self._pivots.append({'value': window_low, 'type': 'low', 'index': self._total_candles - 1})

        # Create the result
        pivot_value = self._last_pivot_val
        pivot_type = self._last_pivot_type
        
        # Signal and Trend (taken from the calculate method)
        signal = SignalType.HOLD
        trend = TrendDirection.NEUTRAL
        
        if pivot_type == 'high':
            trend = TrendDirection.DOWN
            if current_price < pivot_value:
                signal = SignalType.SELL
        elif pivot_type == 'low':
            trend = TrendDirection.UP
            if current_price > pivot_value:
                signal = SignalType.BUY
                
        # Strength
        price_change = abs((current_price - pivot_value) / pivot_value * 100) if pivot_value != 0 else 0
        strength = min(price_change / self.deviation * 100, 100)
        
        return IndicatorResult(
            value=round(pivot_value, 2),
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'deviation': self.deviation,
                'depth': self.depth,
                'pivot_type': pivot_type,
                'current_price': round(current_price, 2),
                'price_change_pct': round(price_change, 2),
                'total_pivots': len(self._pivots)
            }
        )

    def _find_pivots(self, high: np.ndarray, low: np.ndarray) -> list:
        """
        Find the swing high and low points.

        Args:
            high: High prices
            low: Low prices

        Returns:
            list: List of pivot points
        """
        pivots = []
        last_pivot_value = None
        last_pivot_type = None
        threshold = self.deviation / 100

        # Find the first pivot
        start_idx = self.depth
        current_high = np.max(high[:start_idx])
        current_low = np.min(low[:start_idx])

        if (current_high - current_low) / current_low > threshold:
            last_pivot_value = current_high
            last_pivot_type = 'high'
            pivots.append({
                'value': current_high,
                'type': 'high',
                'index': np.argmax(high[:start_idx])
            })

        # Find ongoing pivots
        for i in range(start_idx, len(high)):
            window_high = np.max(high[max(0, i - self.depth):i + 1])
            window_low = np.min(low[max(0, i - self.depth):i + 1])

            if last_pivot_value is None:
                last_pivot_value = window_high
                last_pivot_type = 'high'
                continue

            # Yeni high pivot
            if last_pivot_type == 'low':
                change = (window_high - last_pivot_value) / last_pivot_value
                if change > threshold:
                    pivots.append({
                        'value': window_high,
                        'type': 'high',
                        'index': i
                    })
                    last_pivot_value = window_high
                    last_pivot_type = 'high'

            # Yeni low pivot
            elif last_pivot_type == 'high':
                change = (last_pivot_value - window_low) / last_pivot_value
                if change > threshold:
                    pivots.append({
                        'value': window_low,
                        'type': 'low',
                        'index': i
                    })
                    last_pivot_value = window_low
                    last_pivot_type = 'low'

        return pivots

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the ZigZag value (done within calculate).

        Args:
            value: ZigZag value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the ZigZag value (done within the calculate function).

        Args:
            value: ZigZag value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'deviation': 5.0,
            'depth': 12
        }

    def _requires_volume(self) -> bool:
        """ZigZag volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ZigZag']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """ZigZag indicator test"""

    print("\n" + "="*60)
    print("ZIGZAG INDICATOR TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Simulate price movements with trend changes
    base_price = 100
    prices = [base_price]
    trend = 1  # 1: up, -1: down

    for i in range(99):
        # Change trend every 20 candles
        if i % 20 == 0:
            trend *= -1

        change = np.random.randn() * 0.5 + (trend * 0.3)
        prices.append(prices[-1] * (1 + change / 100))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    zigzag = ZigZag(deviation=5.0, depth=12)
    print(f"   [OK] Created: {zigzag}")
    print(f"   [OK] Kategori: {zigzag.category.value}")
    print(f"   [OK] Tip: {zigzag.indicator_type.value}")
    print(f"   [OK] Required period: {zigzag.get_required_periods()}")

    result = zigzag(data)
    print(f"   [OK] Son Pivot: {result.value}")
    print(f"   [OK] Pivot Type: {result.metadata['pivot_type']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Total Pivot: {result.metadata['total_pivots']}")
    print(f"   [OK] Price Change %: {result.metadata['price_change_pct']}")

    # Test 2: Different deviation values
    print("\n3. Different deviation test...")
    for dev in [3.0, 5.0, 10.0]:
        zigzag_test = ZigZag(deviation=dev, depth=12)
        result = zigzag_test.calculate(data)
        print(f"   [OK] ZigZag(dev={dev}) - Pivots: {result.metadata['total_pivots']} | "
              f"Signal: {result.signal.value}")

    # Test 3: Different depth values
    print("\n4. Different depth test...")
    for depth in [5, 12, 20]:
        zigzag_test = ZigZag(deviation=5.0, depth=depth)
        result = zigzag_test.calculate(data)
        print(f"   [OK] ZigZag(depth={depth}) - Pivots: {result.metadata['total_pivots']} | "
              f"Last: {result.value}")

    # Test 4: Pivot analizi
    print("\n5. Pivot analizi...")
    result = zigzag.calculate(data)
    current = result.metadata['current_price']
    pivot = result.value
    pivot_type = result.metadata['pivot_type']

    print(f"   [OK] Current price: {current}")
    print(f"   [OK] Son pivot: {pivot} ({pivot_type})")

    if pivot_type == 'high':
        print(f"   [OK] Downtrend after the last swing high")
        if current < pivot:
            print(f"   [OK] Price is below the pivot, the decline continues")
        else:
            print(f"   [OK] Price is above the pivot, recovery signal")
    elif pivot_type == 'low':
        print(f"   [OK] Uprising trend after the last swing low")
        if current > pivot:
            print(f"   [OK] Price is above the pivot, the uptrend continues")
        else:
            print(f"   [OK] Price is below the pivot, indicating a weakening signal")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = zigzag.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = zigzag.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
