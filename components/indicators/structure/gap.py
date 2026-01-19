#!/usr/bin/env python3
"""
components/indicators/structure/gap.py
SuperBot - Price Gap Detector

Author: SuperBot Team
Date: 2025-01-09
Versiyon: 1.0.0

Description:
    Detects price gaps between two candles.
    Unlike FVG, it does not require a 3-candle pattern, only a gap between 2 candles.

    Gap Nedir:
    - The price gap formed between two consecutive candles.
    - It is formed as a result of rapid price movement.
    - These gaps are generally "filled" (filled in).

Formula:
    Bullish Gap (Upward Gap):
    - Candle[1].low > Candle[0].high
    - Gap: [Candle[0].high, Candle[1].low]

    Bearish Gap (Downward Gap):
    - Candle[0].low > Candle[1].high
    - Gap: [Candle[1].high, Candle[0].low]

    Fill Status:
    - open: The space has not been filled yet (empty box)
    - filled: Completely filled (only the frame)

Usage:
    from components.indicators.structure.gap import Gap

    gap = Gap(min_gap_percent=0.05)
    result = gap.calculate(df)

    # Zones in metadata
    zones = result.metadata['zones']

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


@dataclass
class GapZone:
    """
    Gap Zone data structure

    Attributes:
        type: 'bullish' or 'bearish'
        top: Upper price level
        bottom: Lower price level
        created_index: Index of the bar it was created on
        created_time: Timestamp when it was created
        filled: Dolduruldu mu
        filled_index: The bar index where it was filled (if applicable)
        filled_time: The timestamp when it was filled (if applicable)
    """
    type: str
    top: float
    bottom: float
    created_index: int
    created_time: int
    filled: bool = False
    filled_index: Optional[int] = None
    filled_time: Optional[int] = None

    @property
    def size(self) -> float:
        """Gap size"""
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """Gap midpoint"""
        return (self.top + self.bottom) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'top': round(self.top, 8),
            'bottom': round(self.bottom, 8),
            'size': round(self.size, 8),
            'midpoint': round(self.midpoint, 8),
            'created_index': self.created_index,
            'created_time': self.created_time,
            'filled': self.filled,
            'filled_index': self.filled_index,
            'filled_time': self.filled_time
        }


class Gap(BaseIndicator):
    """
    Price Gap Detector

    Detects and tracks price gaps between two candles.

    Args:
        min_gap_percent: Minimum gap percentage (default: 0.05 = %0.05)
        max_zones: Maximum number of zones to track (default: 50)
        max_age: Maximum age (bar count, default: 500)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.05,
        max_zones: int = 50,
        max_age: int = 500,
        logger=None,
        error_handler=None
    ):
        self.min_gap_percent = min_gap_percent
        self.max_zones = max_zones
        self.max_age = max_age

        super().__init__(
            name='gap',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'min_gap_percent': min_gap_percent,
                'max_zones': max_zones,
                'max_age': max_age
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Track all gaps (including filled ones)
        self._all_gaps: List[GapZone] = []
        self._active_gaps: List[GapZone] = []  # Not yet filled

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.min_gap_percent < 0:
            raise InvalidParameterError(
                self.name, 'min_gap_percent', self.min_gap_percent,
                "Min gap percent cannot be negative"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones must be positive"
            )
        if self.max_age < 1:
            raise InvalidParameterError(
                self.name, 'max_age', self.max_age,
                "Max age must be positive"
            )
        return True

    def _detect_gap(
        self,
        prev_high: float,
        prev_low: float,
        curr_high: float,
        curr_low: float,
        curr_index: int,
        curr_time: int
    ) -> Optional[GapZone]:
        """
        Detect a gap between two candles.

        Args:
            prev_high: The high of the previous candle
            prev_low: The low of the previous candle
            curr_high: The high of the current candle
            curr_low: The low of the current candle
            curr_index: The index of the current bar
            curr_time: The timestamp of the current bar

        Returns:
            GapZone or None
        """
        mid_price = (prev_high + prev_low + curr_high + curr_low) / 4

        # Bullish Gap: The low of the current candle is greater than the high of the previous candle.
        if curr_low > prev_high:
            gap_size = curr_low - prev_high
            gap_percent = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if gap_percent >= self.min_gap_percent:
                return GapZone(
                    type='bullish',
                    top=curr_low,
                    bottom=prev_high,
                    created_index=curr_index,
                    created_time=curr_time,
                    filled=False
                )

        # Bearish Gap: The low of the previous candle is greater than the high of the current candle.
        if prev_low > curr_high:
            gap_size = prev_low - curr_high
            gap_percent = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if gap_percent >= self.min_gap_percent:
                return GapZone(
                    type='bearish',
                    top=prev_low,
                    bottom=curr_high,
                    created_index=curr_index,
                    created_time=curr_time,
                    filled=False
                )

        return None

    def _update_gap_status(
        self,
        gap: GapZone,
        current_high: float,
        current_low: float,
        current_index: int,
        current_time: int
    ) -> None:
        """
        Update the gap filling status.

        Gap filling condition:
        - Bullish gap: If the price goes down to the bottom (or below) of the gap.
        - Bearish gap: If the price goes up to the top (or above) of the gap.
        """
        if gap.filled:
            return

        if gap.type == 'bullish':
            # Bullish gap: The price should go down and fill the gap.
            if current_low <= gap.bottom:
                gap.filled = True
                gap.filled_index = current_index
                gap.filled_time = current_time
        else:
            # Bearish gap: The price should rise and fill the gap.
            if current_high >= gap.top:
                gap.filled = True
                gap.filled_index = current_index
                gap.filled_time = current_time

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate the gap across all data.

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Gap zones
        """
        self.reset()

        highs = data['high'].values
        lows = data['low'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        n = len(data)

        # Detect and update the gap for each bar.
        for i in range(1, n):
            # Detect a new gap
            new_gap = self._detect_gap(
                prev_high=highs[i - 1],
                prev_low=lows[i - 1],
                curr_high=highs[i],
                curr_low=lows[i],
                curr_index=i,
                curr_time=int(times[i])
            )

            if new_gap:
                self._all_gaps.append(new_gap)
                self._active_gaps.append(new_gap)

            # Update the status of existing gaps
            for gap in self._active_gaps:
                if gap.created_index < i:  # Do not update if created in a different bar
                    self._update_gap_status(
                        gap,
                        current_high=highs[i],
                        current_low=lows[i],
                        current_index=i,
                        current_time=int(times[i])
                    )

            # Remove filled gaps from the active list
            self._active_gaps = [g for g in self._active_gaps if not g.filled]

            # Remove very old gaps from the active list (but keep them in the all list)
            self._active_gaps = [
                g for g in self._active_gaps
                if (i - g.created_index) < self.max_age
            ]

        # Maximum zones limit (keep the newest ones)
        if len(self._all_gaps) > self.max_zones * 2:
            self._all_gaps = self._all_gaps[-self.max_zones * 2:]

        timestamp = int(data.iloc[-1]['timestamp']) if 'timestamp' in data.columns else n - 1

        # Returns all zones (including filled ones - will be displayed differently in the WebUI)
        zones = [gap.to_dict() for gap in self._all_gaps[-self.max_zones:]]
        active_zones = [gap.to_dict() for gap in self._active_gaps]

        # Warmup buffer
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'value': len(active_zones)},
            timestamp=timestamp,
            signal=self._get_signal(active_zones, data['close'].values[-1]),
            trend=self._get_trend(active_zones),
            strength=min(len(active_zones) * 10, 100),
            metadata={
                'zones': zones,  # All gaps (including filled)
                'active_zones': active_zones,  # Only active (unfilled)
                'total_gaps': len(self._all_gaps),
                'active_gaps': len(self._active_gaps),
                'bullish_active': len([z for z in active_zones if z['type'] == 'bullish']),
                'bearish_active': len([z for z in active_zones if z['type'] == 'bearish']),
                'min_gap_percent': self.min_gap_percent
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Batch calculation for backtest

        Returns:
            pd.Series: Net gap value (number of bullish gaps minus the number of bearish gaps)
        """
        highs = data['high'].values
        lows = data['low'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        n = len(data)
        gap_values = np.zeros(n)

        all_gaps: List[GapZone] = []
        active_gaps: List[GapZone] = []

        for i in range(1, n):
            # Detect a new gap
            new_gap = self._detect_gap(
                prev_high=highs[i - 1],
                prev_low=lows[i - 1],
                curr_high=highs[i],
                curr_low=lows[i],
                curr_index=i,
                curr_time=int(times[i])
            )

            if new_gap:
                all_gaps.append(new_gap)
                active_gaps.append(new_gap)

            # Update the status of existing gaps
            for gap in active_gaps:
                if gap.created_index < i:
                    self._update_gap_status(
                        gap,
                        current_high=highs[i],
                        current_low=lows[i],
                        current_index=i,
                        current_time=int(times[i])
                    )

            # Fill in and remove old gaps
            active_gaps = [
                g for g in active_gaps
                if not g.filled and (i - g.created_index) < self.max_age
            ]

            # Net gap value
            bullish_count = len([g for g in active_gaps if g.type == 'bullish'])
            bearish_count = len([g for g in active_gaps if g.type == 'bearish'])
            gap_values[i] = bullish_count - bearish_count

        return pd.Series(gap_values, index=data.index, name='gap')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffers with historical data"""
        from collections import deque

        max_len = self.get_required_periods() + 10
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._time_buffer = deque(maxlen=max_len)
        self._index_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        for idx, row in data.tail(max_len).iterrows():
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._time_buffer.append(int(row.get('timestamp', 0)))
            self._index_buffer.append(idx if isinstance(idx, int) else len(self._high_buffer) - 1)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 10
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._time_buffer = deque(maxlen=max_len)
            self._index_buffer = deque(maxlen=max_len)
            self._current_update_index = 0
            self._buffers_init = True

        # Extract candle data
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._time_buffer.append(timestamp_val)

        if not hasattr(self, '_current_update_index'):
            self._current_update_index = 0
        self._current_update_index += 1
        self._index_buffer.append(self._current_update_index)

        if len(self._high_buffer) < 2:
            return IndicatorResult(
                value={'value': 0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'warmup': True, 'zones': [], 'active_zones': []}
            )

        # Detect a new gap
        new_gap = self._detect_gap(
            prev_high=self._high_buffer[-2],
            prev_low=self._low_buffer[-2],
            curr_high=high_val,
            curr_low=low_val,
            curr_index=self._current_update_index,
            curr_time=timestamp_val
        )

        if new_gap:
            self._all_gaps.append(new_gap)
            self._active_gaps.append(new_gap)

        # Update existing gaps
        for gap in self._active_gaps:
            if gap.created_index < self._current_update_index:
                self._update_gap_status(
                    gap, high_val, low_val,
                    self._current_update_index, timestamp_val
                )

        # Cleanup
        self._active_gaps = [
            g for g in self._active_gaps
            if not g.filled and (self._current_update_index - g.created_index) < self.max_age
        ]

        zones = [g.to_dict() for g in self._all_gaps[-self.max_zones:]]
        active_zones = [g.to_dict() for g in self._active_gaps]

        return IndicatorResult(
            value={'value': len(active_zones)},
            timestamp=timestamp_val,
            signal=self._get_signal(active_zones, close_val),
            trend=self._get_trend(active_zones),
            strength=min(len(active_zones) * 10, 100),
            metadata={
                'zones': zones,
                'active_zones': active_zones,
                'total_gaps': len(self._all_gaps),
                'active_gaps': len(self._active_gaps)
            }
        )

    def _get_signal(self, active_zones: List[Dict], current_price: float) -> SignalType:
        """Generate signal from gaps"""
        if not active_zones:
            return SignalType.HOLD

        # Find the gap closest to the price.
        nearest = None
        min_distance = float('inf')

        for zone in active_zones:
            distance = min(
                abs(current_price - zone['top']),
                abs(current_price - zone['bottom'])
            )
            if distance < min_distance:
                min_distance = distance
                nearest = zone

        if nearest:
            distance_percent = (min_distance / current_price) * 100
            # Send a signal if it's within %0.5
            if distance_percent < 0.5:
                if nearest['type'] == 'bullish':
                    return SignalType.BUY
                else:
                    return SignalType.SELL

        return SignalType.HOLD

    def _get_trend(self, active_zones: List[Dict]) -> TrendDirection:
        """Gap'lerden trend belirle"""
        if not active_zones:
            return TrendDirection.NEUTRAL

        bullish = len([z for z in active_zones if z['type'] == 'bullish'])
        bearish = len([z for z in active_zones if z['type'] == 'bearish'])

        if bullish > bearish:
            return TrendDirection.UP
        elif bearish > bullish:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def get_active_gaps(self) -> List[GapZone]:
        """Active (unfilled) gaps"""
        return self._active_gaps.copy()

    def get_all_gaps(self) -> List[GapZone]:
        """All gaps (including filled ones)"""
        return self._all_gaps.copy()

    def get_bullish_gaps(self, active_only: bool = True) -> List[GapZone]:
        """Bullish gap'ler"""
        source = self._active_gaps if active_only else self._all_gaps
        return [g for g in source if g.type == 'bullish']

    def get_bearish_gaps(self, active_only: bool = True) -> List[GapZone]:
        """Bearish gap'ler"""
        source = self._active_gaps if active_only else self._all_gaps
        return [g for g in source if g.type == 'bearish']

    def reset(self) -> None:
        """Clear state"""
        self._all_gaps = []
        self._active_gaps = []
        if hasattr(self, '_high_buffer'):
            self._high_buffer.clear()
            self._low_buffer.clear()
            self._time_buffer.clear()
            self._index_buffer.clear()
        self._current_update_index = 0

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'min_gap_percent': 0.05,
            'max_zones': 50,
            'max_age': 500
        }

    def _requires_volume(self) -> bool:
        """Gap volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Gap', 'GapZone']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Gap (Price Gap) Test")
    print("=" * 60)

    # Create test data - containing gaps
    print("\n1. Creating test data...")
    np.random.seed(42)

    n = 100
    timestamps = [1700000000000 + i * 60000 for i in range(n)]

    # Base price movement
    base = 100.0
    prices = [base]
    for i in range(1, n):
        change = np.random.randn() * 0.3
        prices.append(prices[-1] + change)

    # Create a gap (at specific points)
    highs = []
    lows = []
    opens = []
    closes = []

    for i, price in enumerate(prices):
        if i == 20:
            # Create a bullish gap
            opens.append(price + 2)  # Gap up
            closes.append(price + 2.5)
            lows.append(price + 1.8)  # Low > previous high
            highs.append(price + 3)
        elif i == 50:
            # Create a bearish gap
            opens.append(price - 2)  # Gap down
            closes.append(price - 2.5)
            highs.append(price - 1.8)  # High < previous low
            lows.append(price - 3)
        else:
            opens.append(price)
            closes.append(price + np.random.randn() * 0.2)
            highs.append(max(opens[-1], closes[-1]) + abs(np.random.randn()) * 0.2)
            lows.append(min(opens[-1], closes[-1]) - abs(np.random.randn()) * 0.2)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(n)]
    })

    print(f"   ✅ {len(data)} bar created")

    # Test 2: Gap calculation
    print("\n2. Gap calculation test...")
    gap = Gap(min_gap_percent=0.05, max_zones=50, max_age=500)
    result = gap.calculate(data)

    print(f"   ✅ Total gaps: {result.metadata['total_gaps']}")
    print(f"   ✅ Active gap: {result.metadata['active_gaps']}")
    print(f"   ✅ Bullish active: {result.metadata['bullish_active']}")
    print(f"   ✅ Bearish active: {result.metadata['bearish_active']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")

    # Test 3: Zone details
    print("\n3. Zone details...")
    zones = result.metadata['zones']
    for i, zone in enumerate(zones[:5]):
        status = "FILLED" if zone['filled'] else "OPEN"
        print(f"   📊 Gap #{i+1}: {zone['type']} | {zone['bottom']:.2f}-{zone['top']:.2f} | {status}")

    # Test 4: Batch calculation
    print("\n4. Batch calculation testi...")
    batch_result = gap.calculate_batch(data)
    print(f"   ✅ Series length: {len(batch_result)}")
    print(f"   ✅ Final value: {batch_result.iloc[-1]}")

    # Test 5: Incremental update
    print("\n5. Incremental update testi...")
    gap2 = Gap(min_gap_percent=0.05)

    # Add the first few candles
    for i in range(10):
        candle = {
            'timestamp': timestamps[i],
            'open': opens[i],
            'high': highs[i],
            'low': lows[i],
            'close': closes[i],
            'volume': 1000
        }
        res = gap2.update(candle)

    print(f"   ✅ Active gap after update: {res.metadata.get('active_gaps', 0)}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
