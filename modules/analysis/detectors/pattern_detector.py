"""
modules/analysis/detectors/pattern_detector.py

Candlestick pattern detection wrapper
It uses the existing CandlestickPatterns and TALibPatterns indicators.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import pandas as pd
import uuid


@dataclass
class CandlePattern:
    """Detected candlestick pattern"""
    name: str
    type: str  # 'bullish', 'bearish', 'neutral'
    time: int
    index: int
    strength: int = 100  # 100 for bullish, -100 for bearish, 0 for neutral
    source: str = 'custom'  # 'custom' or 'talib'
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'time': self.time,
            'index': self.index,
            'strength': self.strength,
            'source': self.source
        }

    def to_chart_annotation(self) -> dict:
        """LightweightCharts marker format"""
        colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#9e9e9e'
        }
        return {
            'time': self.time // 1000,
            'position': 'belowBar' if self.type == 'bullish' else 'aboveBar',
            'color': colors.get(self.type, '#9e9e9e'),
            'shape': 'arrowUp' if self.type == 'bullish' else 'arrowDown',
            'text': self.name,
            'size': 1
        }


class PatternDetector:
    """
    Candlestick pattern detector

    Wraps existing indicators:
    - CandlestickPatterns (custom, 21 pattern)
    - TALibPatterns (TA-Lib, 61 pattern)
    """

    # Bullish patterns (as a value of True or 100)
    BULLISH_PATTERNS = {
        # Custom
        'hammer', 'inverted_hammer', 'marubozu_bullish', 'engulfing_bullish',
        'harami_bullish', 'piercing_line', 'morning_star', 'three_white_soldiers',
        'dragonfly_doji',
        # TALib
        'three_inside', 'three_outside', 'three_stars_in_south', 'three_white_soldiers',
        'abandoned_baby', 'belt_hold', 'breakaway', 'homing_pigeon', 'ladder_bottom',
        'matching_low', 'morning_star', 'morning_doji_star', 'piercing', 'takuri',
        'unique_three_river',
    }

    # Bearish patterns
    BEARISH_PATTERNS = {
        # Custom
        'hanging_man', 'shooting_star', 'marubozu_bearish', 'engulfing_bearish',
        'harami_bearish', 'dark_cloud_cover', 'evening_star', 'three_black_crows',
        'gravestone_doji',
        # TALib
        'two_crows', 'three_black_crows', 'advance_block', 'dark_cloud_cover',
        'evening_star', 'evening_doji_star', 'identical_three_crows', 'stalled_pattern',
        'upside_gap_two_crows',
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.use_talib = self.config.get('use_talib', False)
        self._history: List[CandlePattern] = []

        # Lazy load indicators
        self._custom_detector = None
        self._talib_detector = None

    def _get_custom_detector(self):
        """Lazy load custom pattern detector"""
        if self._custom_detector is None:
            try:
                from components.indicators.patterns.candlestick_patterns import CandlestickPatterns
                self._custom_detector = CandlestickPatterns(
                    doji_threshold=self.config.get('doji_threshold', 0.1),
                    shadow_ratio=self.config.get('shadow_ratio', 2.0),
                    min_body_size=self.config.get('min_body_size', 0.0001)
                )
            except ImportError:
                pass
        return self._custom_detector

    def _get_talib_detector(self):
        """Lazy load TALib pattern detector"""
        if self._talib_detector is None:
            try:
                from components.indicators.patterns.talib_patterns import TALibPatterns
                self._talib_detector = TALibPatterns()
            except ImportError:
                pass
        return self._talib_detector

    def detect(self, data: pd.DataFrame) -> List[CandlePattern]:
        """
        Batch detection for candlestick patterns

        Args:
            data: OHLCV DataFrame

        Returns:
            List of CandlePattern
        """
        self._history = []
        patterns = []

        times = data['timestamp'].values if 'timestamp' in data.columns else list(range(len(data)))

        # Custom patterns
        custom = self._get_custom_detector()
        if custom:
            try:
                patterns_df = custom.calculate_batch(data)

                for i in range(len(patterns_df)):
                    row = patterns_df.iloc[i]
                    for col in patterns_df.columns:
                        if row[col] == True:
                            pattern_type = self._get_pattern_type(col)
                            pattern = CandlePattern(
                                name=col,
                                type=pattern_type,
                                time=int(times[i]),
                                index=i,
                                strength=100 if pattern_type == 'bullish' else (-100 if pattern_type == 'bearish' else 0),
                                source='custom'
                            )
                            patterns.append(pattern)
                            self._history.append(pattern)
            except Exception as e:
                pass

        # TALib patterns
        if self.use_talib:
            talib = self._get_talib_detector()
            if talib:
                try:
                    patterns_df = talib.calculate_batch(data)

                    for i in range(len(patterns_df)):
                        row = patterns_df.iloc[i]
                        for col in patterns_df.columns:
                            val = row[col]
                            if val != 0:
                                pattern_type = 'bullish' if val > 0 else 'bearish'
                                pattern = CandlePattern(
                                    name=col,
                                    type=pattern_type,
                                    time=int(times[i]),
                                    index=i,
                                    strength=int(val),
                                    source='talib'
                                )
                                patterns.append(pattern)
                                self._history.append(pattern)
                except Exception as e:
                    pass

        return patterns

    def update(self, candle: dict, current_index: int) -> List[CandlePattern]:
        """
        Incremental update

        Args:
            candle: New candle
            current_index: Current bar index

        Returns:
            List of new patterns (if any)
        """
        # Pattern detection usually looks at the last N bars.
        # Use the update() method of the custom detector for streaming.
        new_patterns = []

        time = candle.get('timestamp', candle.get('t', current_index))

        custom = self._get_custom_detector()
        if custom:
            try:
                result = custom.update(candle)
                if result and result.value.get('patterns'):
                    for pattern_name in result.value['patterns']:
                        pattern_type = self._get_pattern_type(pattern_name)
                        pattern = CandlePattern(
                            name=pattern_name,
                            type=pattern_type,
                            time=int(time),
                            index=current_index,
                            strength=100 if pattern_type == 'bullish' else (-100 if pattern_type == 'bearish' else 0),
                            source='custom'
                        )
                        new_patterns.append(pattern)
                        self._history.append(pattern)
            except Exception:
                pass

        return new_patterns

    def _get_pattern_type(self, pattern_name: str) -> str:
        """
        Pattern tipini belirle

        Args:
            pattern_name: Pattern name

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        name_lower = pattern_name.lower()

        if name_lower in self.BULLISH_PATTERNS:
            return 'bullish'
        elif name_lower in self.BEARISH_PATTERNS:
            return 'bearish'
        elif 'bullish' in name_lower:
            return 'bullish'
        elif 'bearish' in name_lower:
            return 'bearish'
        else:
            return 'neutral'

    def get_bullish_patterns(self) -> List[CandlePattern]:
        """Bullish patterns"""
        return [p for p in self._history if p.type == 'bullish']

    def get_bearish_patterns(self) -> List[CandlePattern]:
        """Bearish patterns"""
        return [p for p in self._history if p.type == 'bearish']

    def get_patterns_at(self, index: int) -> List[CandlePattern]:
        """Belirli bir bar'daki patterns"""
        return [p for p in self._history if p.index == index]

    def get_recent_patterns(self, count: int = 10) -> List[CandlePattern]:
        """Last N pattern"""
        return self._history[-count:] if self._history else []

    def reset(self) -> None:
        """Clear state"""
        self._history = []
