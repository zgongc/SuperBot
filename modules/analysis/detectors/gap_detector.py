"""
modules/analysis/detectors/gap_detector.py

Price Gap detection (space between 2 candles)

Unlike FVG, it does not require 3 candle patterns.
It only detects the price gap between two consecutive candles.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from .base_detector import BaseDetector
from ..models.formations import GapFormation


class GapDetector(BaseDetector):
    """
    Price Gap detector

    Detects the price gaps between two consecutive candles.

    Bullish Gap (Uptrend Gap):
    - Mum2.low > Mum1.high
    - Gap: [Mum1.high, Mum2.low]

    Bearish Gap (Downward Gap):
    - Mum1.low > Mum2.high
    - Gap: [Mum2.high, Mum1.low]

    Gaps are usually "filled" by the price.
    Filled gaps are displayed only as a frame.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.min_size_pct = self.config.get('min_size_pct', 0.05)  # Min %0.05 gap
        self.max_age = self.config.get('max_age', 500)  # Max 500 bar active for.
        self.max_zones = self.config.get('max_zones', 100)  # Max 100 gap sakla

        # Streaming state
        self._high_buffer: List[float] = []
        self._low_buffer: List[float] = []
        self._time_buffer: List[int] = []
        self._current_index = 0

    def detect(self, data: pd.DataFrame) -> List[GapFormation]:
        """
        Batch detection for True GAP (wick gap - space between high/low).

        TradingView ICT True GAP definition:
        - Bullish GAP: high[1] < low (previous candle's HIGH < current candle's LOW)
        - Bearish GAP: low[1] > high (previous candle's LOW > current candle's HIGH)

        Args:
            data: OHLCV DataFrame

        Returns:
            List of GapFormation
        """
        self.reset()

        highs = data['high'].values
        lows = data['low'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        n = len(data)
        formations = []

        for i in range(1, n):
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_time = int(times[i])

            mid_price = (prev_high + prev_low + curr_high + curr_low) / 4

            # Bullish True GAP: Previous candle's HIGH < Current candle's LOW
            # That means there is no contact between the two candles, there is space above.
            if prev_high < curr_low:
                gap_size = curr_low - prev_high
                gap_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

                if gap_pct >= self.min_size_pct:
                    gap = GapFormation(
                        type='bullish',
                        top=float(curr_low),
                        bottom=float(prev_high),
                        created_time=curr_time,
                        created_index=i,
                        filled=False
                    )
                    formations.append(gap)
                    self._add_formation(gap)

            # Bearish True GAP: The low of the previous candle is greater than the high of the current candle.
            # That means there is no touch between two candles, there is a space below.
            elif prev_low > curr_high:
                gap_size = prev_low - curr_high
                gap_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

                if gap_pct >= self.min_size_pct:
                    gap = GapFormation(
                        type='bearish',
                        top=float(prev_low),
                        bottom=float(curr_high),
                        created_time=curr_time,
                        created_index=i,
                        filled=False
                    )
                    formations.append(gap)
                    self._add_formation(gap)

            # Update the status of existing gaps (with wick)
            self._update_active_gaps(curr_high, curr_low, i, curr_time)

        return formations

    def update(self, candle: dict, current_index: int) -> Optional[GapFormation]:
        """
        Incremental update for True GAP (wick gap)

        TradingView ICT True GAP definition:
        - Bullish GAP: high[1] < low (previous candle's HIGH < current candle's LOW)
        - Bearish GAP: low[1] > high (previous candle's LOW > current candle's HIGH)

        Args:
            candle: New candle data
            current_index: Current bar index

        Returns:
            New Gap if detected, else None
        """
        curr_high = candle.get('high', candle.get('h', 0))
        curr_low = candle.get('low', candle.get('l', 0))
        time = candle.get('timestamp', candle.get('t', current_index))

        self._high_buffer.append(curr_high)
        self._low_buffer.append(curr_low)
        self._time_buffer.append(time)
        self._current_index = current_index

        # Buffer limit
        max_buffer = 10
        if len(self._high_buffer) > max_buffer:
            self._high_buffer = self._high_buffer[-max_buffer:]
            self._low_buffer = self._low_buffer[-max_buffer:]
            self._time_buffer = self._time_buffer[-max_buffer:]

        # At least 2 candles are required
        if len(self._high_buffer) < 2:
            return None

        # Check the last 2 candles (wick - high/low)
        prev_high = self._high_buffer[-2]
        prev_low = self._low_buffer[-2]

        mid_price = (prev_high + prev_low + curr_high + curr_low) / 4
        new_gap = None

        # Bullish True GAP: Previous candle's HIGH < Current candle's LOW
        if prev_high < curr_low:
            gap_size = curr_low - prev_high
            gap_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if gap_pct >= self.min_size_pct:
                new_gap = GapFormation(
                    type='bullish',
                    top=float(curr_low),
                    bottom=float(prev_high),
                    created_time=int(time),
                    created_index=current_index,
                    filled=False
                )
                self._add_formation(new_gap)

        # Bearish True GAP: The low of the previous candle is greater than the high of the current candle.
        elif prev_low > curr_high:
            gap_size = prev_low - curr_high
            gap_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if gap_pct >= self.min_size_pct:
                new_gap = GapFormation(
                    type='bearish',
                    top=float(prev_low),
                    bottom=float(curr_high),
                    created_time=int(time),
                    created_index=current_index,
                    filled=False
                )
                self._add_formation(new_gap)

        # Update existing gaps (with wick)
        self._update_active_gaps(curr_high, curr_low, current_index, int(time))

        return new_gap

    def _update_active_gaps(
        self,
        current_high: float,
        current_low: float,
        current_index: int,
        current_time: int
    ) -> None:
        """
        Update the fill status of active gaps.

        Gap filling condition:
        - Bullish gap: If the price goes down to the bottom (or below) of the gap.
        - Bearish gap: If the price goes up to the top (or above) of the gap.
        """
        still_active = []

        for gap in self._active_formations:
            # Do not update in the bar where it was created.
            if gap.created_index >= current_index:
                still_active.append(gap)
                continue

            # Fill check
            if not gap.filled:
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

            # Keep in the active list: If it's not filled and not too old.
            age = current_index - gap.created_index
            if not gap.filled and age < self.max_age:
                still_active.append(gap)

        self._active_formations = still_active

        # Maximum zones limit (for history)
        if len(self._history) > self.max_zones:
            self._history = self._history[-self.max_zones:]

    def get_unfilled_gaps(self) -> List[GapFormation]:
        """Gaps that have not yet been filled"""
        return [g for g in self._active_formations if not g.filled]

    def get_filled_gaps(self) -> List[GapFormation]:
        """Filled gaps (from history)"""
        return [g for g in self._history if g.filled]

    def get_bullish_gaps(self, active_only: bool = True) -> List[GapFormation]:
        """Bullish gap'ler"""
        source = self._active_formations if active_only else self._history
        return [g for g in source if g.type == 'bullish']

    def get_bearish_gaps(self, active_only: bool = True) -> List[GapFormation]:
        """Bearish gap'ler"""
        source = self._active_formations if active_only else self._history
        return [g for g in source if g.type == 'bearish']

    def get_nearest_gap(self, current_price: float) -> Optional[GapFormation]:
        """
        The active gap closest to the price.

        Args:
            current_price: Current price

        Returns:
            Nearest Gap or None
        """
        if not self._active_formations:
            return None

        nearest = None
        min_distance = float('inf')

        for gap in self._active_formations:
            if gap.filled:
                continue
            distance = abs(current_price - gap.midpoint)
            if distance < min_distance:
                min_distance = distance
                nearest = gap

        return nearest

    def reset(self) -> None:
        """Clear state"""
        super().reset()
        self._high_buffer = []
        self._low_buffer = []
        self._time_buffer = []
        self._current_index = 0
