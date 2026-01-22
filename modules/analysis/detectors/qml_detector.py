"""
modules/analysis/detectors/qml_detector.py

QML (Quasimodo) Detector - TradingView EmreKb Style

TradingView "Quasimodo Pattern" indicator logic:
https://www.tradingview.com/script/xxx/

Bullish QML: trend == -1 and h2 > h1 and l1 > l0 and h0 > h1 and close > l1
Bearish QML: trend == 1 and l2 < l1 and h1 < h0 and l0 < l1 and close < h1

5 point zigzag pattern uses (h0, h1, h2 or l0, l1, l2 + opposite side).
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .base_detector import BaseDetector


@dataclass
class QMLFormation:
    """
    Quasimodo pattern formation - TradingView Style

    Bullish QML (5 nokta):
        h2 -> l1 -> h1 -> l0 -> h0
        Conditions: h2 > h1, l1 > l0, h0 > h1, close > l1

    Bearish QML (5 nokta):
        l2 -> h1 -> l1 -> h0 -> l0
        Conditions: l2 < l1, h1 < h0, l0 < l1, close < h1
    """
    type: str  # 'bullish' or 'bearish'
    index: int  # MSB (confirm) bar index
    timestamp: Optional[int] = None

    # 5 swing points (TradingView style)
    # Bullish: h2, l1, h1, l0, h0
    # Bearish: l2, h1, l1, h0, l0
    h2: Optional[float] = None
    h2_idx: Optional[int] = None
    h2_time: Optional[int] = None

    l1: Optional[float] = None
    l1_idx: Optional[int] = None
    l1_time: Optional[int] = None

    h1: Optional[float] = None
    h1_idx: Optional[int] = None
    h1_time: Optional[int] = None

    l0: Optional[float] = None
    l0_idx: Optional[int] = None
    l0_time: Optional[int] = None

    h0: Optional[float] = None
    h0_idx: Optional[int] = None
    h0_time: Optional[int] = None

    # Additional points for bearish scenarios
    l2: Optional[float] = None
    l2_idx: Optional[int] = None
    l2_time: Optional[int] = None

    # QML Zone (entry zone)
    zone_top: Optional[float] = None
    zone_bottom: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSON serializable dictionary"""
        def safe_int(val):
            return int(val) if val is not None else None

        def safe_float(val):
            return float(val) if val is not None else None

        return {
            'type': self.type,
            'index': safe_int(self.index),
            'timestamp': safe_int(self.timestamp),
            # Swing points
            'h2': safe_float(self.h2), 'h2_idx': safe_int(self.h2_idx), 'h2_time': safe_int(self.h2_time),
            'l1': safe_float(self.l1), 'l1_idx': safe_int(self.l1_idx), 'l1_time': safe_int(self.l1_time),
            'h1': safe_float(self.h1), 'h1_idx': safe_int(self.h1_idx), 'h1_time': safe_int(self.h1_time),
            'l0': safe_float(self.l0), 'l0_idx': safe_int(self.l0_idx), 'l0_time': safe_int(self.l0_time),
            'h0': safe_float(self.h0), 'h0_idx': safe_int(self.h0_idx), 'h0_time': safe_int(self.h0_time),
            'l2': safe_float(self.l2), 'l2_idx': safe_int(self.l2_idx), 'l2_time': safe_int(self.l2_time),
            # Zone
            'zone_top': safe_float(self.zone_top),
            'zone_bottom': safe_float(self.zone_bottom),
        }

    def to_chart_annotation(self) -> Dict[str, Any]:
        """Annotation format for the chart"""
        return {
            'type': 'marker',
            'subtype': 'qml',
            'direction': self.type,
            'index': self.index,
            'price': self.l1 if self.type == 'bullish' else self.h1,
            'color': '#22c55e' if self.type == 'bullish' else '#ef4444',
            'shape': 'arrowUp' if self.type == 'bullish' else 'arrowDown',
            'text': 'QM!',
        }


class QMLDetector(BaseDetector):
    """
    QML (Quasimodo) Detector - TradingView EmreKb Style

    It uses the logic of the TradingView "Quasimodo Pattern" indicator.

    Bullish QML (downtrend'de):
        h2 > h1 (LH formed)
        l1 > l0 (LL formed - Lower Low)
        h0 > h1 (the last high exceeded h1)
        close > l1 (MSB - l1 level broken)

    Bearish QML (uptrend'de):
        l2 < l1 (HL formed)
        h1 < h0 (HH formed - Higher High)
        l0 < l1 (the last low exceeded l1)
        close < h1 (MSB - the h1 level was broken)

    Args:
        config: Configuration
            - left_bars: ZigZag pivot left bars (default: 13)
            - right_bars: ZigZag pivot right bars (default: 13)
            - lookback_bars: The number of bars to look back for pattern search (default: 200)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.config = config or {}

        self.left_bars = self.config.get('left_bars', 13)
        self.right_bars = self.config.get('right_bars', 13)
        self.lookback_bars = self.config.get('lookback_bars', 200)

        self._formations: List[QMLFormation] = []
        # Swing arrays (TradingView style)
        self._high_points: List[float] = []
        self._high_indices: List[int] = []
        self._low_points: List[float] = []
        self._low_indices: List[int] = []
        self._trend: int = 1  # 1 = uptrend, -1 = downtrend

    def detect(self, data: pd.DataFrame) -> List[QMLFormation]:
        """
        TradingView EmreKb Style QML Detection

        It collects swing points during trend changes using the ZigZag logic,
        then checks the QML pattern conditions.

        Args:
            data: OHLCV DataFrame

        Returns:
            List[QMLFormation]: Detected QML patterns.
        """
        self.reset()

        timestamps = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        n = len(data)

        # TradingView style: calculate to_up and to_down
        # to_up = high >= ta.highest(zigzag_len)
        # to_down = low <= ta.lowest(zigzag_len)
        zigzag_len = self.left_bars

        # Rolling highest/lowest
        highest = pd.Series(highs).rolling(window=zigzag_len, min_periods=1).max().values
        lowest = pd.Series(lows).rolling(window=zigzag_len, min_periods=1).min().values

        to_up = highs >= highest
        to_down = lows <= lowest

        # Trend tracking
        trend = np.ones(n, dtype=np.int32)
        for i in range(1, n):
            if trend[i-1] == 1 and to_down[i]:
                trend[i] = -1
            elif trend[i-1] == -1 and to_up[i]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]

        # Collect swing points (when the trend changes)
        high_points = []
        high_indices = []
        low_points = []
        low_indices = []

        for i in range(1, n):
            trend_changed = trend[i] != trend[i-1]

            if trend_changed:
                if trend[i] == 1:
                    # Trend reversed upwards - save the previous low
                    # Find the lowest low since the last to_down
                    lookback = 1
                    for j in range(i-1, max(0, i - zigzag_len * 2) - 1, -1):
                        if to_down[j]:
                            lookback = i - j
                            break
                    low_val = np.min(lows[max(0, i - lookback):i+1])
                    low_idx = max(0, i - lookback) + np.argmin(lows[max(0, i - lookback):i+1])
                    low_points.append(low_val)
                    low_indices.append(low_idx)

                elif trend[i] == -1:
                    # Trend reversed downwards - save the previous high
                    lookback = 1
                    for j in range(i-1, max(0, i - zigzag_len * 2) - 1, -1):
                        if to_up[j]:
                            lookback = i - j
                            break
                    high_val = np.max(highs[max(0, i - lookback):i+1])
                    high_idx = max(0, i - lookback) + np.argmax(highs[max(0, i - lookback):i+1])
                    high_points.append(high_val)
                    high_indices.append(high_idx)

        # QML pattern tespiti
        # Check the last 3 highs and 3 lows for each bar
        for i in range(zigzag_len, n):
            current_close = closes[i]

            # Get the last swing points (from the bar before this one)
            valid_highs = [(idx, val) for idx, val in zip(high_indices, high_points) if idx < i]
            valid_lows = [(idx, val) for idx, val in zip(low_indices, low_points) if idx < i]

            if len(valid_highs) < 3 or len(valid_lows) < 3:
                continue

            # Last 3 high and low
            h0_idx, h0 = valid_highs[-1]
            h1_idx, h1 = valid_highs[-2]
            h2_idx, h2 = valid_highs[-3]

            l0_idx, l0 = valid_lows[-1]
            l1_idx, l1 = valid_lows[-2]
            l2_idx, l2 = valid_lows[-3]

            # Bullish QML: trend == -1 and h2 > h1 and l1 > l0 and h0 > h1 and close > l1
            # Downtrend: h2 > h1 (LH), l1 > l0 (LL), h0 > h1 (exceeds h1), close > l1 (MSB)
            bu_cond = (
                trend[i] == -1 and
                h2 > h1 and      # LH occurred (h2 is higher)
                l1 > l0 and      # LL occurred (l0 is lower)
                h0 > h1 and      # The last high h1 has been exceeded
                current_close > l1  # MSB - l1 level is broken
            )

            # Check if this condition existed in the previous bar (not bu_cond[1])
            if bu_cond:
                prev_close = closes[i-1] if i > 0 else 0
                bu_cond_prev = (
                    trend[i-1] == -1 and
                    h2 > h1 and l1 > l0 and h0 > h1 and
                    prev_close > l1
                )
                if not bu_cond_prev:
                    # Yeni bullish QML!
                    formation = QMLFormation(
                        type='bullish',
                        index=i,
                        timestamp=int(timestamps[i]),
                        h2=h2, h2_idx=h2_idx, h2_time=int(timestamps[h2_idx]),
                        l1=l1, l1_idx=l1_idx, l1_time=int(timestamps[l1_idx]),
                        h1=h1, h1_idx=h1_idx, h1_time=int(timestamps[h1_idx]),
                        l0=l0, l0_idx=l0_idx, l0_time=int(timestamps[l0_idx]),
                        h0=h0, h0_idx=h0_idx, h0_time=int(timestamps[h0_idx]),
                        zone_top=l1,
                        zone_bottom=l0,
                    )
                    self._formations.append(formation)

            # Bearish QML: trend == 1 and l2 < l1 and h1 < h0 and l0 < l1 and close < h1
            # In an uptrend: l2 < l1 (HL), h1 < h0 (HH), l0 < l1 (exceeds l1), close < h1 (MSB)
            be_cond = (
                trend[i] == 1 and
                l2 < l1 and      # HL occurred (l2 is lower)
                h1 < h0 and      # HH occurred (h0 is higher)
                l0 < l1 and      # The last low has been exceeded by l1
                current_close < h1  # MSB - h1 level is broken
            )

            if be_cond:
                prev_close = closes[i-1] if i > 0 else float('inf')
                be_cond_prev = (
                    trend[i-1] == 1 and
                    l2 < l1 and h1 < h0 and l0 < l1 and
                    prev_close < h1
                )
                if not be_cond_prev:
                    # Yeni bearish QML!
                    formation = QMLFormation(
                        type='bearish',
                        index=i,
                        timestamp=int(timestamps[i]),
                        l2=l2, l2_idx=l2_idx, l2_time=int(timestamps[l2_idx]),
                        h1=h1, h1_idx=h1_idx, h1_time=int(timestamps[h1_idx]),
                        l1=l1, l1_idx=l1_idx, l1_time=int(timestamps[l1_idx]),
                        h0=h0, h0_idx=h0_idx, h0_time=int(timestamps[h0_idx]),
                        l0=l0, l0_idx=l0_idx, l0_time=int(timestamps[l0_idx]),
                        zone_top=h0,
                        zone_bottom=h1,
                    )
                    self._formations.append(formation)

        return self._formations

    def update(self, candle: dict, index: int) -> Optional[QMLFormation]:
        """
        Incremental update - requires warmup with historical data first

        Args:
            candle: Yeni mum verisi
            index: Bar index

        Returns:
            Returns the new QML pattern if it exists.
        """
        # For realtime, need to maintain swing arrays
        # This is a simplified version - full implementation would track swings incrementally
        return None

    def get_history(self) -> List[QMLFormation]:
        """Returns all QML patterns"""
        return self._formations

    def get_bullish(self) -> List[QMLFormation]:
        """Returns bullish QMLs"""
        return [f for f in self._formations if f.type == 'bullish']

    def get_bearish(self) -> List[QMLFormation]:
        """Returns bearish QMLs"""
        return [f for f in self._formations if f.type == 'bearish']

    def reset(self) -> None:
        """Reset the state"""
        self._formations = []
        self._high_points = []
        self._high_indices = []
        self._low_points = []
        self._low_indices = []
        self._trend = 1


__all__ = ['QMLDetector', 'QMLFormation']
