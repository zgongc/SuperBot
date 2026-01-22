"""
modules/analysis/detectors/fvg_detector.py

FVG (Fair Value Gap) / Imbalance detection
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from .base_detector import BaseDetector
from ..models.formations import FVGFormation


class FVGDetector(BaseDetector):
    """
    Fair Value Gap detector (SMC Style)

    FVG (Imbalance): 3 candle pattern where candle 2 creates an imbalance

    - Bullish FVG: Candle 2 moves up, with a gap between Candle 1's HIGH and Candle 3's LOW
      -> Gap = Candle3.low - Candle1.high (if positive)

    - Bearish FVG: Candle 2 moves down, there is a gap between Candle 1's LOW and Candle 3's HIGH
      -> Gap = Candle1.low - Candle3.high (if positive)

    FVGs are usually "filled" by the price.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.min_size_pct = self.config.get('min_size_pct', 0.1)  # Min %0.1 gap
        self.max_age = self.config.get('max_age', 50)  # Max 50 bars active for a limited time

        # Streaming state
        self._high_buffer: List[float] = []
        self._low_buffer: List[float] = []
        self._close_buffer: List[float] = []
        self._time_buffer: List[int] = []
        self._current_index = 0

    def detect(self, data: pd.DataFrame) -> List[FVGFormation]:
        """
        Batch detection for FVG

        Args:
            data: OHLCV DataFrame

        Returns:
            List of FVGFormation
        """
        self.reset()

        highs = data['high'].values
        lows = data['low'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        n = len(data)
        formations = []

        # First pass: Detect all FVGs
        for i in range(2, n):
            # Candle indices
            candle1_high = highs[i - 2]
            candle1_low = lows[i - 2]
            candle3_high = highs[i]
            candle3_low = lows[i]

            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle3_low > candle1_high:
                gap_size = candle3_low - candle1_high
                gap_pct = (gap_size / candle1_high) * 100 if candle1_high > 0 else 0

                if gap_pct >= self.min_size_pct:
                    fvg = FVGFormation(
                        type='bullish',
                        top=float(candle3_low),
                        bottom=float(candle1_high),
                        created_time=int(times[i]),
                        created_index=i,
                        filled=False,
                        filled_percent=0.0,
                        age=0
                    )
                    formations.append(fvg)
                    self._add_formation(fvg)

            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif candle1_low > candle3_high:
                gap_size = candle1_low - candle3_high
                gap_pct = (gap_size / candle3_high) * 100 if candle3_high > 0 else 0

                if gap_pct >= self.min_size_pct:
                    fvg = FVGFormation(
                        type='bearish',
                        top=float(candle1_low),
                        bottom=float(candle3_high),
                        created_time=int(times[i]),
                        created_index=i,
                        filled=False,
                        filled_percent=0.0,
                        age=0
                    )
                    formations.append(fvg)
                    self._add_formation(fvg)

        # Second pass: Check fill status for ALL formations against ALL subsequent candles
        # This ensures even old FVGs get properly filled
        for fvg in formations:
            if fvg.filled:
                continue

            # Check all candles after FVG creation
            for j in range(fvg.created_index + 1, n):
                if fvg.filled:
                    break

                current_high = highs[j]
                current_low = lows[j]
                current_time = int(times[j])

                # Update age
                fvg.age = j - fvg.created_index

                # Fill check (wick-based) with history tracking
                if fvg.type == 'bullish':
                    # Bullish FVG: Price wick enters FVG from above
                    if current_low <= fvg.top:
                        fill_level = max(current_low, fvg.bottom)
                        fvg.fill_history.append((current_time, fill_level))

                        fill_pct = ((fvg.top - current_low) / fvg.size) * 100
                        fvg.filled_percent = min(fill_pct, 100.0)

                        if current_low <= fvg.bottom or fvg.filled_percent >= 95.0:
                            fvg.filled = True
                            fvg.filled_percent = 100.0
                            fvg.filled_time = current_time
                            fvg.filled_index = j
                else:
                    # Bearish FVG: Price wick enters FVG from below
                    if current_high >= fvg.bottom:
                        fill_level = min(current_high, fvg.top)
                        fvg.fill_history.append((current_time, fill_level))

                        fill_pct = ((current_high - fvg.bottom) / fvg.size) * 100
                        fvg.filled_percent = min(fill_pct, 100.0)

                        if current_high >= fvg.top or fvg.filled_percent >= 95.0:
                            fvg.filled = True
                            fvg.filled_percent = 100.0
                            fvg.filled_time = current_time
                            fvg.filled_index = j

        # Update active formations list (unfilled and not too old relative to last bar)
        last_index = n - 1
        self._active_formations = [
            f for f in formations
            if not f.filled and (last_index - f.created_index) < self.max_age
        ]

        return formations

    def update(self, candle: dict, current_index: int) -> Optional[FVGFormation]:
        """
        Incremental update

        Args:
            candle: New candle
            current_index: Current bar index

        Returns:
            New FVG if detected
        """
        high = candle.get('high', candle.get('h', 0))
        low = candle.get('low', candle.get('l', 0))
        close = candle.get('close', candle.get('c', 0))
        time = candle.get('timestamp', candle.get('t', current_index))

        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(close)
        self._time_buffer.append(time)
        self._current_index = current_index

        # Buffer limit
        max_buffer = 10
        if len(self._high_buffer) > max_buffer:
            self._high_buffer = self._high_buffer[-max_buffer:]
            self._low_buffer = self._low_buffer[-max_buffer:]
            self._close_buffer = self._close_buffer[-max_buffer:]
            self._time_buffer = self._time_buffer[-max_buffer:]

        # Need at least 3 candles
        if len(self._high_buffer) < 3:
            return None

        # Check for new FVG (using last 3 candles)
        candle1_high = self._high_buffer[-3]
        candle1_low = self._low_buffer[-3]
        candle3_high = self._high_buffer[-1]
        candle3_low = self._low_buffer[-1]

        new_fvg = None

        # Bullish FVG
        if candle3_low > candle1_high:
            gap_size = candle3_low - candle1_high
            gap_pct = (gap_size / candle1_high) * 100 if candle1_high > 0 else 0

            if gap_pct >= self.min_size_pct:
                new_fvg = FVGFormation(
                    type='bullish',
                    top=float(candle3_low),
                    bottom=float(candle1_high),
                    created_time=int(time),
                    created_index=current_index,
                    filled=False,
                    filled_percent=0.0,
                    age=0
                )
                self._add_formation(new_fvg)

        # Bearish FVG
        elif candle1_low > candle3_high:
            gap_size = candle1_low - candle3_high
            gap_pct = (gap_size / candle3_high) * 100 if candle3_high > 0 else 0

            if gap_pct >= self.min_size_pct:
                new_fvg = FVGFormation(
                    type='bearish',
                    top=float(candle1_low),
                    bottom=float(candle3_high),
                    created_time=int(time),
                    created_index=current_index,
                    filled=False,
                    filled_percent=0.0,
                    age=0
                )
                self._add_formation(new_fvg)

        # Update existing FVGs - wick-based
        self._update_active_fvgs(high, low, current_index, int(time))

        return new_fvg

    def _update_active_fvgs(
        self,
        current_high: float,
        current_low: float,
        current_index: int,
        current_time: int = None
    ) -> None:
        """
        Update active FVG values (fill check, age)

        Fill detection based on wick (high/low):
        - Bullish FVG: low <= bottom ise doldu
        - Bearish FVG: high >= top ise doldu

        Args:
            current_high: Current candle high
            current_low: Current candle low
            current_index: Current bar index
            current_time: Current timestamp (ms)
        """
        still_active = []

        for fvg in self._active_formations:
            # Skip if already filled
            if fvg.filled:
                continue

            # Age update
            fvg.age = current_index - fvg.created_index

            # Fill check (wick-based) with history tracking
            if fvg.type == 'bullish':
                # Bullish FVG: Did the price wick enter the FVG?
                if current_low <= fvg.top:
                    # Price entered the price FVG, fill level = the level at which the price fell.
                    fill_level = max(current_low, fvg.bottom)  # It cannot go below the bottom.

                    # Add to history (only if it's inside FVG)
                    fvg.fill_history.append((current_time, fill_level))

                    # Calculate partial filling
                    fill_pct = ((fvg.top - current_low) / fvg.size) * 100
                    fvg.filled_percent = min(fill_pct, 100.0)

                    # To be considered completely full: reached the bottom OR is 95%+ full
                    if current_low <= fvg.bottom or fvg.filled_percent >= 95.0:
                        fvg.filled = True
                        fvg.filled_percent = 100.0
                        fvg.filled_time = current_time
                        fvg.filled_index = current_index
            else:
                # Bearish FVG: Did the price wick enter the FVG?
                if current_high >= fvg.bottom:
                    # Price entered the price FVG, fill level = the level from which the price emerged.
                    fill_level = min(current_high, fvg.top)  # cannot go above the top

                    # Add to history (only if it's within FVG)
                    fvg.fill_history.append((current_time, fill_level))

                    # Calculate partial filling
                    fill_pct = ((current_high - fvg.bottom) / fvg.size) * 100
                    fvg.filled_percent = min(fill_pct, 100.0)

                    # To be considered completely full: the ball reached the top OR it is 95%+ full.
                    if current_high >= fvg.top or fvg.filled_percent >= 95.0:
                        fvg.filled = True
                        fvg.filled_percent = 100.0
                        fvg.filled_time = current_time
                        fvg.filled_index = current_index

            # Keep if not filled and not too old
            if not fvg.filled and fvg.age < self.max_age:
                still_active.append(fvg)

        self._active_formations = still_active

    def get_unfilled_fvgs(self) -> List[FVGFormation]:
        """Unfilled FVG values"""
        return [f for f in self._active_formations if not f.filled]

    def get_bullish_fvgs(self) -> List[FVGFormation]:
        """Bullish FVG (active)"""
        return [f for f in self._active_formations if f.type == 'bullish']

    def get_bearish_fvgs(self) -> List[FVGFormation]:
        """Bearish FVG (active)"""
        return [f for f in self._active_formations if f.type == 'bearish']

    def get_nearest_fvg(self, current_price: float) -> Optional[FVGFormation]:
        """
        The most recent FVG closest to the price.

        Args:
            current_price: Current price

        Returns:
            Nearest FVG or None
        """
        if not self._active_formations:
            return None

        # Find the nearest FVG.
        nearest = None
        min_distance = float('inf')

        for fvg in self._active_formations:
            # Distance to the midpoint of FVG
            distance = abs(current_price - fvg.midpoint)
            if distance < min_distance:
                min_distance = distance
                nearest = fvg

        return nearest

    def reset(self) -> None:
        """Clear state"""
        super().reset()
        self._high_buffer = []
        self._low_buffer = []
        self._close_buffer = []
        self._time_buffer = []
        self._current_index = 0
