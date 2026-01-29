"""
modules/analysis/detectors/swing_detector.py

Swing High/Low detection - TradingView ZigZag Compatible

Tek kaynak: SwingPoints (components/indicators/support_resistance/swingpoints.py)
TradingView uyumlu pivot algoritmasi kullanir.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
import os
from .base_detector import BaseDetector
from ..models.formations import SwingPoint

# Set the SwingPoints path (single source)
_components_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _components_path not in sys.path:
    sys.path.insert(0, os.path.join(_components_path, 'components'))


class SwingDetector(BaseDetector):
    """
    Detects Swing High/Low points - Compatible with TradingView ZigZag.

    Tek kaynak: SwingPoints kullanir
    - TradingView ta.pivothigh/ta.pivotlow uyumlu
    - HH, HL, LH, LL etiketleri hesaplanir
    - Unbroken swing levels are tracked.

    Args:
        config:
            - left_bars: Sol taraftaki bar sayisi (default: 3)
            - right_bars: Sag taraftaki bar sayisi (default: 3)
            - max_levels: The maximum number of levels to follow (default: 10)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.left_bars = self.config.get('left_bars', 3)
        self.right_bars = self.config.get('right_bars', 3)
        self.max_levels = self.config.get('max_levels', 10)

        # SwingPoints - tek kaynak (lazy load)
        self._swing_points = None

        # Current swing levels
        self._current_swing_high: Optional[SwingPoint] = None
        self._current_swing_low: Optional[SwingPoint] = None

        # All unbroken swing levels (for horizontal lines)
        self._unbroken_highs: List[SwingPoint] = []
        self._unbroken_lows: List[SwingPoint] = []

    def _get_swing_points(self):
        """SwingPoints instance'i lazy load et"""
        if self._swing_points is None:
            from pathlib import Path
            import sys
            # SuperBot base directory'yi path'e ekle
            base_dir = Path(__file__).parent.parent.parent.parent
            if str(base_dir) not in sys.path:
                sys.path.insert(0, str(base_dir))
            # Add the components folder as well (for indicators.__init__.py)
            components_path = base_dir / "components"
            if str(components_path) not in sys.path:
                sys.path.insert(0, str(components_path))

            from components.indicators.support_resistance.swingpoints import SwingPoints
            self._swing_points = SwingPoints(
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                lookback=50
            )
        return self._swing_points

    def detect(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        SwingPoints kullanarak batch detection.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of SwingPoint
        """
        self.reset()

        swing_points = self._get_swing_points()
        swing_df = swing_points.calculate_batch(data)
        times = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        # First, collect all swings (index, type, price)
        raw_swings = []
        for i in range(len(swing_df)):
            if not np.isnan(swing_df['swing_high'].iloc[i]):
                raw_swings.append((i, 'high', swing_df['swing_high'].iloc[i]))
            if not np.isnan(swing_df['swing_low'].iloc[i]):
                raw_swings.append((i, 'low', swing_df['swing_low'].iloc[i]))

        # Sort by index (chronological)
        raw_swings.sort(key=lambda x: x[0])

        # Alternating swing'leri filtrele (High-Low-High-Low pattern)
        # Keep only the most extreme value in consecutive swings of the same type.
        filtered_swings = []
        for idx, swing_type, price in raw_swings:
            if not filtered_swings:
                filtered_swings.append((idx, swing_type, price))
            elif filtered_swings[-1][1] != swing_type:
                # Different type - add
                filtered_swings.append((idx, swing_type, price))
            else:
                # Same type - keep the more extreme one
                last_idx, last_type, last_price = filtered_swings[-1]
                if swing_type == 'high' and price > last_price:
                    filtered_swings[-1] = (idx, swing_type, price)
                elif swing_type == 'low' and price < last_price:
                    filtered_swings[-1] = (idx, swing_type, price)

        # Create a SwingPoint list from the filtered swings
        swings = []
        prev_high_price = None
        prev_low_price = None

        # OHLC data - for broken check
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        for idx, swing_type, price in filtered_swings:
            if swing_type == 'high':
                # HH/LH label hesapla
                if prev_high_price is None:
                    label = 'HH'  # First high
                elif price > prev_high_price:
                    label = 'HH'
                else:
                    label = 'LH'
                prev_high_price = price

                swing = SwingPoint(
                    type='high',
                    price=price,
                    time=int(times[idx]),
                    index=idx,
                    broken=False,
                    label=label
                )
                swings.append(swing)
                self._add_formation(swing)
                self._current_swing_high = swing
            else:
                # HL/LL label hesapla
                if prev_low_price is None:
                    label = 'HL'  # First low
                elif price > prev_low_price:
                    label = 'HL'
                else:
                    label = 'LL'
                prev_low_price = price

                swing = SwingPoint(
                    type='low',
                    price=price,
                    time=int(times[idx]),
                    index=idx,
                    broken=False,
                    label=label
                )
                swings.append(swing)
                self._add_formation(swing)
                self._current_swing_low = swing

        # Broken check: Check the close price from the next bars for each swing.
        # Swing High: close > swing.price ise broken
        # Swing Low: close < swing.price ise broken
        n = len(data)
        for swing in swings:
            for i in range(swing.index + 1, n):
                if swing.type == 'high' and closes[i] > swing.price:
                    swing.broken = True
                    swing.broken_index = i
                    swing.broken_time = int(times[i])
                    break
                elif swing.type == 'low' and closes[i] < swing.price:
                    swing.broken = True
                    swing.broken_index = i
                    swing.broken_time = int(times[i])
                    break

        # Sum up the unbroken levels (up to the last max_levels)
        self._unbroken_highs = [s for s in swings if s.type == 'high' and not s.broken][-self.max_levels:]
        self._unbroken_lows = [s for s in swings if s.type == 'low' and not s.broken][-self.max_levels:]

        return swings

    def update(self, candle: dict, current_index: int) -> Optional[SwingPoint]:
        """
        Incremental update (streaming mode).

        Note: Swing detection requires right_bars,
        so a new swing can only be confirmed after right_bars.
        """
        # TODO: Implement streaming mode
        return None

    def get_current_swing_high(self) -> Optional[float]:
        """The latest swing high value"""
        return self._current_swing_high.price if self._current_swing_high else None

    def get_current_swing_low(self) -> Optional[float]:
        """The latest swing low value"""
        return self._current_swing_low.price if self._current_swing_low else None

    def get_recent_swings(self, count: int = 5) -> List[SwingPoint]:
        """Last N swing point"""
        return self._history[-count:] if self._history else []

    def get_unbroken_highs(self) -> List[SwingPoint]:
        """Return unbroken swing highs (for the horizontal line)"""
        return self._unbroken_highs

    def get_unbroken_lows(self) -> List[SwingPoint]:
        """Flip unbroken swing lows (for horizontal line)"""
        return self._unbroken_lows

    def get_all_unbroken(self) -> List[SwingPoint]:
        """Tum kirilmamis swing seviyelerini dondur"""
        return self._unbroken_highs + self._unbroken_lows

    def reset(self) -> None:
        """Clear state"""
        super().reset()
        self._swing_points = None  # Lazy load reset
        self._current_swing_high = None
        self._current_swing_low = None
        self._unbroken_highs = []
        self._unbroken_lows = []
