"""
modules/analysis/detectors/structure_detector.py

BOS (Break of Structure) and CHoCH (Change of Character) detection.

WRAPPER: This file only uses bos.py and choch.py.
Single source: components/indicators/structure/bos.py and choch.py
"""

from typing import List, Optional, Dict, Any, Set
import pandas as pd
import numpy as np
import sys
import os
from .base_detector import BaseDetector
from ..models.formations import BOSFormation, CHoCHFormation, SwingPoint

# Set the components path.
_components_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _components_path not in sys.path:
    sys.path.insert(0, _components_path)

from components.indicators.structure.bos import BOS
from components.indicators.structure.choch import CHoCH


class StructureDetector(BaseDetector):
    """
    Market Structure detector - Detects BOS and CHoCH.

    WRAPPER: Uses the calculate_batch() methods in bos.py and choch.py.

    BOS (Break of Structure):
        - Bullish BOS: Close > previous swing high (uptrend/ranging)
        - Bearish BOS: Close < previous swing low (downtrend/ranging)

    CHoCH (Change of Character):
        - Bullish CHoCH: Breakout of the swing high in a downtrend (trend reversal)
        - Bearish CHoCH: Breakout of the swing low in an uptrend (trend reversal)

    CHoCH vs BOS difference:
        - BOS: Structure break in the same direction as the trend (continuation)
        - CHoCH: Structure break in the opposite direction of the trend (reversal)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.left_bars = self.config.get('left_bars', 5)
        self.right_bars = self.config.get('right_bars', 5)
        self.max_levels = self.config.get('max_levels', 5)
        self.trend_strength = self.config.get('trend_strength', 2)

        # BOS and CHoCH indicator instances (single source)
        self._bos_indicator = BOS(
            left_bars=self.left_bars,
            right_bars=self.right_bars,
            max_levels=self.max_levels
        )
        self._choch_indicator = CHoCH(
            left_bars=self.left_bars,
            right_bars=self.right_bars,
            max_levels=self.max_levels,
            trend_strength=self.trend_strength
        )

        # State for tracking (for update() method and get_current_trend())
        self._current_trend: str = 'ranging'
        self._recent_swing_highs: List[SwingPoint] = []
        self._recent_swing_lows: List[SwingPoint] = []

    def detect(self, data: pd.DataFrame) -> List[Any]:
        """
        Batch detection for BOS and CHoCH

        Uses bos.py and choch.py calculate_batch() methods.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of BOSFormation and CHoCHFormation
        """
        self.reset()

        # Calculate BOS and CHoCH using indicators (tek kaynak)
        bos_series = self._bos_indicator.calculate_batch(data)
        choch_series = self._choch_indicator.calculate_batch(data)

        # Get swing points for formation metadata
        swing_highs, swing_lows = self._bos_indicator._find_swings(data)
        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        closes = data['close'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        formations = []

        # Convert BOS signals to BOSFormation objects
        for i in range(len(data)):
            bos_val = bos_series.iloc[i]
            choch_val = choch_series.iloc[i]

            if bos_val == 1:  # Bullish BOS
                # Find the broken swing high
                broken_level = self._find_broken_level(i, swing_highs, closes, 'high')
                formation = BOSFormation(
                    type='bullish',
                    broken_level=broken_level,
                    break_price=float(closes[i]),
                    break_time=int(times[i]),
                    swing_index=self._find_swing_index(i, swing_highs, closes, 'high'),
                    break_index=i,
                    strength=self._calculate_strength(broken_level, closes[i])
                )
                formations.append(formation)
                self._add_formation(formation)

            elif bos_val == -1:  # Bearish BOS
                broken_level = self._find_broken_level(i, swing_lows, closes, 'low')
                formation = BOSFormation(
                    type='bearish',
                    broken_level=broken_level,
                    break_price=float(closes[i]),
                    break_time=int(times[i]),
                    swing_index=self._find_swing_index(i, swing_lows, closes, 'low'),
                    break_index=i,
                    strength=self._calculate_strength(broken_level, closes[i])
                )
                formations.append(formation)
                self._add_formation(formation)

            if choch_val == 1:  # Bullish CHoCH
                broken_level = self._find_broken_level(i, swing_highs, closes, 'high')
                formation = CHoCHFormation(
                    type='bullish',
                    previous_trend='downtrend',
                    broken_level=broken_level,
                    break_price=float(closes[i]),
                    break_time=int(times[i]),
                    break_index=i,
                    significance=self._calculate_significance(broken_level, closes[i])
                )
                formations.append(formation)
                self._add_formation(formation)

            elif choch_val == -1:  # Bearish CHoCH
                broken_level = self._find_broken_level(i, swing_lows, closes, 'low')
                formation = CHoCHFormation(
                    type='bearish',
                    previous_trend='uptrend',
                    broken_level=broken_level,
                    break_price=float(closes[i]),
                    break_time=int(times[i]),
                    break_index=i,
                    significance=self._calculate_significance(broken_level, closes[i])
                )
                formations.append(formation)
                self._add_formation(formation)

        # Update current trend based on last swings
        self._update_trend_state(swing_highs, swing_lows)

        return formations

    def _find_broken_level(self, break_index: int, swings: List[Dict], closes: np.ndarray, swing_type: str) -> float:
        """Find the swing level that was broken at break_index"""
        prev_close = closes[break_index - 1] if break_index > 0 else closes[break_index]
        current_close = closes[break_index]

        for swing in reversed(swings):
            if swing['index'] >= break_index:
                continue
            swing_val = swing['value']

            if swing_type == 'high':
                # Bullish break: current > swing, prev <= swing
                if current_close > swing_val and prev_close <= swing_val:
                    return swing_val
            else:
                # Bearish break: current < swing, prev >= swing
                if current_close < swing_val and prev_close >= swing_val:
                    return swing_val

        # Fallback: return closest swing
        for swing in reversed(swings):
            if swing['index'] < break_index:
                return swing['value']

        return closes[break_index]

    def _find_swing_index(self, break_index: int, swings: List[Dict], closes: np.ndarray, swing_type: str) -> int:
        """Find the swing index that was broken"""
        prev_close = closes[break_index - 1] if break_index > 0 else closes[break_index]
        current_close = closes[break_index]

        for swing in reversed(swings):
            if swing['index'] >= break_index:
                continue
            swing_val = swing['value']

            if swing_type == 'high':
                if current_close > swing_val and prev_close <= swing_val:
                    return swing['index']
            else:
                if current_close < swing_val and prev_close >= swing_val:
                    return swing['index']

        # Fallback
        for swing in reversed(swings):
            if swing['index'] < break_index:
                return swing['index']

        return 0

    def _update_trend_state(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> None:
        """Update current trend based on swing structure"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return

        last_highs = [h['value'] for h in swing_highs[-3:]]
        last_lows = [l['value'] for l in swing_lows[-3:]]

        if len(last_highs) < 2 or len(last_lows) < 2:
            return

        last_high_direction = 'higher' if last_highs[-1] > last_highs[-2] else 'lower'
        last_low_direction = 'higher' if last_lows[-1] > last_lows[-2] else 'lower'

        if last_high_direction == 'higher' and last_low_direction == 'higher':
            self._current_trend = 'uptrend'
        elif last_high_direction == 'lower' and last_low_direction == 'lower':
            self._current_trend = 'downtrend'
        elif last_high_direction == 'higher':
            self._current_trend = 'uptrend'
        elif last_low_direction == 'lower':
            self._current_trend = 'downtrend'

        # Store recent swings for get methods
        self._recent_swing_highs = [
            SwingPoint(type='high', price=s['value'], time=0, index=s['index'], broken=False)
            for s in swing_highs[-self.max_levels:]
        ]
        self._recent_swing_lows = [
            SwingPoint(type='low', price=s['value'], time=0, index=s['index'], broken=False)
            for s in swing_lows[-self.max_levels:]
        ]

    def update(self, candle: dict, current_index: int) -> Optional[Any]:
        """
        Incremental update - delegates to indicator update methods

        Note: For real-time, consider using batch re-detection periodically
        as SwingPoints requires right_bars confirmation.
        """
        # For now, return None - real-time updates need batch re-detection
        # because swing points can only be confirmed after right_bars
        return None

    def _calculate_strength(self, broken_level: float, break_price: float) -> float:
        """BOS strength calculation (0-100)"""
        if broken_level <= 0:
            return 0.0
        break_size = abs(break_price - broken_level)
        relative_break = (break_size / broken_level) * 100
        strength = min(relative_break * 100, 100)
        return round(strength, 2)

    def _calculate_significance(self, broken_level: float, break_price: float) -> float:
        """CHoCH significance calculation (0-100)"""
        base_strength = self._calculate_strength(broken_level, break_price)
        return min(base_strength * 1.2, 100)

    def get_bos_formations(self) -> List[BOSFormation]:
        """Only BOS formations"""
        return [f for f in self._history if isinstance(f, BOSFormation)]

    def get_choch_formations(self) -> List[CHoCHFormation]:
        """Only CHoCH formations"""
        return [f for f in self._history if isinstance(f, CHoCHFormation)]

    def get_current_trend(self) -> str:
        """Current trend"""
        return self._current_trend

    def reset(self) -> None:
        """Clear state"""
        super().reset()
        self._current_trend = 'ranging'
        self._recent_swing_highs = []
        self._recent_swing_lows = []
