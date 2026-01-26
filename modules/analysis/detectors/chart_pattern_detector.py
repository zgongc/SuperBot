"""
modules/analysis/detectors/chart_pattern_detector.py
SuperBot - Trendline Breakout Detector
Author: SuperBot Team
Date: 2025-01-23
Version: 5.0.0

TSR (Trend Lines, Supports and Resistances) Style:
- Uptrend: Rising LOW pivots (each low higher than previous)
- Downtrend: Falling HIGH pivots (each high lower than previous)
- Violation check instead of R² fitting
- Extension to current bar

Algorithm:
1. Find pivot highs and lows
2. For uptrend: find consecutive lows where p1 < p2 (rising)
3. For downtrend: find consecutive highs where p1 > p2 (falling)
4. Check violations (bars crossing the line)
5. Extend valid trendlines to current bar
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base_detector import BaseDetector
from .swing_detector import SwingDetector
from ..models.formations import SwingPoint, ChartPattern


class ChartPatternDetector(BaseDetector):
    """
    Trendline Breakout Detector (v5.0 - TSR Style)

    Detects:
    - Uptrend lines (rising lows) → Support
    - Downtrend lines (falling highs) → Resistance
    - Breakouts when price crosses these lines

    Config:
        - pivot_length: Pivot detection period (default: 20)
        - points_to_check: How many recent pivots to check (default: 3)
        - max_violation: Max allowed violations, 0 = strict (default: 0)
        - except_bars: Ignore last N bars for violation (default: 3)
        - extend_lines: Extend lines to current bar (default: true)
    """

    PATTERN_INFO = {
        'resistance_breakout': {'name': 'Resistance Breakout', 'type': 'bullish'},
        'support_breakout': {'name': 'Support Breakout', 'type': 'bearish'},
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Pivot detection (like TradingView pivothigh/pivotlow)
        self.pivot_length = self.config.get('pivot_length', 20)

        # TSR parameters
        self.points_to_check = self.config.get('points_to_check', 3)
        self.max_violation = self.config.get('max_violation', 0)
        self.except_bars = self.config.get('except_bars', 3)
        self.extend_lines = self.config.get('extend_lines', True)

        # Swing detector
        self.swing_detector = SwingDetector({
            'left_bars': self.pivot_length,
            'right_bars': self.pivot_length
        })

        self._patterns: List[ChartPattern] = []
        self._swings: List[SwingPoint] = []
        self._trendlines: List[Dict] = []

    def detect(self, data: pd.DataFrame) -> List[ChartPattern]:
        """Detect trendlines and breakouts using TSR method."""
        self.reset()

        if len(data) < self.pivot_length * 2 + 10:
            return []

        # Get pivot points
        swings = self.swing_detector.detect(data)
        self._swings = swings

        # Separate and sort by time (most recent first)
        high_pivots = sorted(
            [s for s in swings if s.type == 'high'],
            key=lambda x: x.index,
            reverse=True
        )
        low_pivots = sorted(
            [s for s in swings if s.type == 'low'],
            key=lambda x: x.index,
            reverse=True
        )

        patterns = []
        all_trendlines = []

        # Find UPTREND lines (rising lows) - Support
        uptrends = self._find_uptrends(low_pivots, data)
        all_trendlines.extend(uptrends)

        # Find DOWNTREND lines (falling highs) - Resistance
        downtrends = self._find_downtrends(high_pivots, data)
        all_trendlines.extend(downtrends)

        # Check for breakouts
        for tl in all_trendlines:
            if not tl.get('is_violated', False):
                breakout = self._check_breakout(tl, data)
                if breakout:
                    patterns.append(breakout)

        self._trendlines = all_trendlines
        self._patterns = patterns

        for p in patterns:
            self._add_formation(p)

        return patterns

    def _find_uptrends(self, low_pivots: List[SwingPoint], data: pd.DataFrame) -> List[Dict]:
        """
        Find uptrend lines from rising low pivots.

        Uptrend: Connect two LOW pivots where first is LOWER than second (rising).
        This forms a support trendline.
        """
        trendlines = []

        if len(low_pivots) < 2:
            return trendlines

        # Get recent pivots to check (reversed for chronological order)
        recent_pivots = low_pivots[:self.points_to_check][::-1]

        # Try all pairs where p1.price < p2.price (rising)
        for i in range(len(recent_pivots)):
            for j in range(i + 1, len(recent_pivots)):
                p1 = recent_pivots[i]  # Earlier (older)
                p2 = recent_pivots[j]  # Later (newer)

                # Uptrend: first point must be LOWER than second (rising lows)
                if p1.price >= p2.price:
                    continue

                # Calculate line: y = mx + b
                m = (p2.price - p1.price) / (p2.index - p1.index)
                b = p1.price - m * p1.index

                # Check violations
                violations = self._count_violations_below(
                    m, b, p1.index, data, 'low'
                )

                is_violated = violations > self.max_violation

                # Calculate current line value
                last_index = len(data) - 1
                current_value = m * last_index + b

                # End point for drawing
                if self.extend_lines and not is_violated:
                    end_index = last_index
                    end_value = current_value
                else:
                    end_index = p2.index
                    end_value = p2.price

                trendlines.append({
                    'type': 'uptrend',  # Rising lows = support
                    'direction': 'up',
                    'slope': m,
                    'intercept': b,
                    'start_index': p1.index,
                    'end_index': end_index,
                    'start_value': p1.price,
                    'end_value': end_value,
                    'current_value': current_value,
                    'start_time': p1.time,
                    'end_time': int(data['timestamp'].iloc[end_index]) if 'timestamp' in data.columns else 0,
                    'touches': 2,
                    'touch_points': [p1, p2],
                    'violations': violations,
                    'is_violated': is_violated,
                    'r_squared': 1.0,  # Perfect fit for 2 points
                    'score': 100 if not is_violated else 0,
                })

        return trendlines

    def _find_downtrends(self, high_pivots: List[SwingPoint], data: pd.DataFrame) -> List[Dict]:
        """
        Find downtrend lines from falling high pivots.

        Downtrend: Connect two HIGH pivots where first is HIGHER than second (falling).
        This forms a resistance trendline.
        """
        trendlines = []

        if len(high_pivots) < 2:
            return trendlines

        # Get recent pivots to check (reversed for chronological order)
        recent_pivots = high_pivots[:self.points_to_check][::-1]

        # Try all pairs where p1.price > p2.price (falling)
        for i in range(len(recent_pivots)):
            for j in range(i + 1, len(recent_pivots)):
                p1 = recent_pivots[i]  # Earlier (older)
                p2 = recent_pivots[j]  # Later (newer)

                # Downtrend: first point must be HIGHER than second (falling highs)
                if p1.price <= p2.price:
                    continue

                # Calculate line: y = mx + b
                m = (p2.price - p1.price) / (p2.index - p1.index)
                b = p1.price - m * p1.index

                # Check violations
                violations = self._count_violations_above(
                    m, b, p1.index, data, 'high'
                )

                is_violated = violations > self.max_violation

                # Calculate current line value
                last_index = len(data) - 1
                current_value = m * last_index + b

                # End point for drawing
                if self.extend_lines and not is_violated:
                    end_index = last_index
                    end_value = current_value
                else:
                    end_index = p2.index
                    end_value = p2.price

                trendlines.append({
                    'type': 'downtrend',  # Falling highs = resistance
                    'direction': 'down',
                    'slope': m,
                    'intercept': b,
                    'start_index': p1.index,
                    'end_index': end_index,
                    'start_value': p1.price,
                    'end_value': end_value,
                    'current_value': current_value,
                    'start_time': p1.time,
                    'end_time': int(data['timestamp'].iloc[end_index]) if 'timestamp' in data.columns else 0,
                    'touches': 2,
                    'touch_points': [p1, p2],
                    'violations': violations,
                    'is_violated': is_violated,
                    'r_squared': 1.0,  # Perfect fit for 2 points
                    'score': 100 if not is_violated else 0,
                })

        return trendlines

    def _count_violations_below(
        self,
        m: float,
        b: float,
        start_index: int,
        data: pd.DataFrame,
        price_col: str
    ) -> int:
        """
        Count how many bars have lows BELOW the trendline.
        Used for uptrend (support) validation.
        """
        last_index = len(data) - 1
        end_check = last_index - self.except_bars

        if end_check <= start_index:
            return 0

        violations = 0
        for i in range(start_index, end_check + 1):
            line_price = m * i + b
            actual_price = float(data[price_col].iloc[i])
            if actual_price < line_price:
                violations += 1

        return violations

    def _count_violations_above(
        self,
        m: float,
        b: float,
        start_index: int,
        data: pd.DataFrame,
        price_col: str
    ) -> int:
        """
        Count how many bars have highs ABOVE the trendline.
        Used for downtrend (resistance) validation.
        """
        last_index = len(data) - 1
        end_check = last_index - self.except_bars

        if end_check <= start_index:
            return 0

        violations = 0
        for i in range(start_index, end_check + 1):
            line_price = m * i + b
            actual_price = float(data[price_col].iloc[i])
            if actual_price > line_price:
                violations += 1

        return violations

    def _check_breakout(self, trendline: Dict, data: pd.DataFrame) -> Optional[ChartPattern]:
        """
        Check if current price breaks the trendline.

        - Uptrend (support) breakout: low goes BELOW the line → Bearish
        - Downtrend (resistance) breakout: high goes ABOVE the line → Bullish
        """
        last_idx = len(data) - 1
        last_low = float(data['low'].iloc[-1])
        last_high = float(data['high'].iloc[-1])
        last_close = float(data['close'].iloc[-1])

        current_value = trendline['current_value']
        direction = trendline['direction']

        breakout_detected = False
        breakout_price = None
        pattern_type = None
        display_name = None

        if direction == 'up':
            # Uptrend (support) breakout: price goes BELOW
            if last_low < current_value:
                breakout_detected = True
                breakout_price = last_low
                pattern_type = 'bearish'
                display_name = 'Support Breakout'
        else:
            # Downtrend (resistance) breakout: price goes ABOVE
            if last_high > current_value:
                breakout_detected = True
                breakout_price = last_high
                pattern_type = 'bullish'
                display_name = 'Resistance Breakout'

        if not breakout_detected:
            return None

        # Calculate target (simple projection)
        touches = trendline.get('touch_points', [])
        target = None

        if len(touches) >= 2:
            p1, p2 = touches[0], touches[1]
            height = abs(p2.price - p1.price)

            if pattern_type == 'bullish':
                target = current_value + height
            else:
                target = current_value - height

        # Confidence based on no violations
        confidence = 70.0 if trendline['violations'] == 0 else 50.0

        timestamp = int(data['timestamp'].iloc[-1]) if 'timestamp' in data.columns else 0
        pattern_name = 'resistance_breakout' if pattern_type == 'bullish' else 'support_breakout'

        return ChartPattern(
            name=pattern_name,
            display_name=display_name,
            type=pattern_type,
            status='confirmed',
            swings=touches,
            start_time=trendline['start_time'],
            end_time=timestamp,
            start_index=trendline['start_index'],
            end_index=last_idx,
            neckline=current_value,
            target=target,
            confidence=confidence,
            breakout_price=breakout_price,
            breakout_confirmed=True
        )

    def get_patterns(self) -> List[ChartPattern]:
        return self._patterns.copy()

    def get_swings(self) -> List[SwingPoint]:
        return self._swings.copy()

    def get_trendlines(self) -> List[Dict]:
        """Get detected trendlines (for chart drawing)."""
        return self._trendlines.copy()

    def reset(self) -> None:
        super().reset()
        self._patterns = []
        self._swings = []
        self._trendlines = []

    def update(self, candle: dict, current_index: int) -> Optional[ChartPattern]:
        """Incremental update - not implemented."""
        return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trendline Detector v5.0 (TSR Style) Test")
    print("=" * 60)

    np.random.seed(42)
    n = 200

    # Generate trending price data
    base = 100
    prices = [base]
    for i in range(n - 1):
        trend = 0.001 if i < 100 else -0.0005
        cycle = np.sin(i / 20) * 0.003
        noise = np.random.randn() * 0.003
        prices.append(prices[-1] * (1 + trend + cycle + noise))

    timestamps = [1700000000000 + i * 900000 for i in range(n)]  # 15m candles

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.002) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.002) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"\n1. Sample data: {len(df)} candles")
    print(f"   Start: {df['close'].iloc[0]:.2f}")
    print(f"   End: {df['close'].iloc[-1]:.2f}")

    detector = ChartPatternDetector({
        'pivot_length': 10,
        'points_to_check': 3,
        'max_violation': 0,
        'except_bars': 3,
        'extend_lines': True,
    })

    print(f"\n2. Config: pivot={detector.pivot_length}, check={detector.points_to_check}")

    patterns = detector.detect(df)
    trendlines = detector.get_trendlines()
    swings = detector.get_swings()

    print(f"\n3. Results:")
    print(f"   Pivots: {len(swings)}")
    print(f"   Trendlines: {len(trendlines)}")

    for tl in trendlines:
        status = "VIOLATED" if tl['is_violated'] else "VALID"
        print(f"     - {tl['type']}: {tl['start_value']:.2f} → {tl['end_value']:.2f} [{status}]")

    print(f"\n   Breakouts: {len(patterns)}")
    for p in patterns:
        print(f"     - {p.display_name} ({p.type})")

    print("\n" + "=" * 60)
