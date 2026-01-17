#!/usr/bin/env python3
"""
components/indicators/patterns/talib_patterns.py
SuperBot - TA-Lib Candlestick Pattern Detector (All 61 Patterns)
Yazar: SuperBot Team
Tarih: 2025-10-26
Versiyon: 2.0.0

🎯 Görev:
    TA-Lib'in tüm 61 candlestick pattern'ini tespit eder

📊 Pattern'ler (61 adet):
    - CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE
    - CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS
    - CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD
    - CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL
    - CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI
    - CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING
    - CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE
    - CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN
    - CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE
    - CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON
    - CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER
    - CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM
    - CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU
    - CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR
    - CDLMORNINGSTAR, CDLONNECK, CDLPIERCING
    - CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES
    - CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP
    - CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI
    - CDLTASUKIGAP, CDLTHRUSTING, CDLTRISTAR
    - CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS

🔧 Kullanım:
    # Strategy template'de
    technical_parameters:
      indicators:
        talib_patterns:
          enabled: true

    # Entry condition - Pattern adı lowercase, CDL prefix yok
    entry_conditions:
      buy:
        - ["hammer", "==", 100, "1m"]          # Bullish pattern
        - ["engulfing", "==", 100, "1m"]       # Bullish engulfing
      sell:
        - ["shootingstar", "==" -100, "1m"]    # Bearish pattern
        - ["eveningstar", "==", -100, "1m"]    # Bearish evening star

Pattern Return Values:
    - 100  = Bullish pattern
    - -100 = Bearish pattern
    - 0    = No pattern

Bağımlılıklar:
    - TA-Lib>=0.4.0 (pip install TA-Lib)
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import pandas as pd
import numpy as np
from typing import Optional
from collections import deque
import talib

from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import SignalType, TrendDirection
from components.indicators.indicator_types import (
    IndicatorResult,
    IndicatorCategory,
    IndicatorType
)


# ============================================================================
# TA-LIB PATTERN WRAPPER
# ============================================================================

class TALibPatterns(BaseIndicator):
    """
    TA-Lib Candlestick Pattern Detector (All 61 Patterns)

    Wraps TA-Lib's candlestick pattern recognition functions.

    Output (DataFrame with 61 columns):
        Each column represents a pattern, values are:
        - 100: Bullish pattern detected
        - -100: Bearish pattern detected
        - 0: No pattern

    Pattern Names (lowercase, no CDL prefix):
        - two_crows, three_black_crows, three_inside, three_line_strike
        - three_outside, three_stars_in_south, three_white_soldiers
        - abandoned_baby, advance_block, belt_hold
        - breakaway, closing_marubozu, conceal_baby_swallow
        - counterattack, dark_cloud_cover, doji
        - doji_star, dragonfly_doji, engulfing
        - evening_doji_star, evening_star, gap_side_side_white
        - gravestone_doji, hammer, hanging_man
        - harami, harami_cross, high_wave
        - hikkake, hikkake_mod, homing_pigeon
        - identical_three_crows, in_neck, inverted_hammer
        - kicking, kicking_by_length, ladder_bottom
        - long_legged_doji, long_line, marubozu
        - matching_low, mat_hold, morning_doji_star
        - morning_star, on_neck, piercing
        - rickshaw_man, rise_fall_three_methods, separating_lines
        - shooting_star, short_line, spinning_top
        - stalled_pattern, stick_sandwich, takuri
        - tasuki_gap, thrusting, tristar
        - unique_three_river, upside_gap_two_crows, xside_gap_three_methods
    """

    # Map TA-Lib function names to friendly column names
    PATTERN_MAP = {
        'CDL2CROWS': 'two_crows',
        'CDL3BLACKCROWS': 'three_black_crows',
        'CDL3INSIDE': 'three_inside',
        'CDL3LINESTRIKE': 'three_line_strike',
        'CDL3OUTSIDE': 'three_outside',
        'CDL3STARSINSOUTH': 'three_stars_in_south',
        'CDL3WHITESOLDIERS': 'three_white_soldiers',
        'CDLABANDONEDBABY': 'abandoned_baby',
        'CDLADVANCEBLOCK': 'advance_block',
        'CDLBELTHOLD': 'belt_hold',
        'CDLBREAKAWAY': 'breakaway',
        'CDLCLOSINGMARUBOZU': 'closing_marubozu',
        'CDLCONCEALBABYSWALL': 'conceal_baby_swallow',
        'CDLCOUNTERATTACK': 'counterattack',
        'CDLDARKCLOUDCOVER': 'dark_cloud_cover',
        'CDLDOJI': 'doji',
        'CDLDOJISTAR': 'doji_star',
        'CDLDRAGONFLYDOJI': 'dragonfly_doji',
        'CDLENGULFING': 'engulfing',
        'CDLEVENINGDOJISTAR': 'evening_doji_star',
        'CDLEVENINGSTAR': 'evening_star',
        'CDLGAPSIDESIDEWHITE': 'gap_side_side_white',
        'CDLGRAVESTONEDOJI': 'gravestone_doji',
        'CDLHAMMER': 'hammer',
        'CDLHANGINGMAN': 'hanging_man',
        'CDLHARAMI': 'harami',
        'CDLHARAMICROSS': 'harami_cross',
        'CDLHIGHWAVE': 'high_wave',
        'CDLHIKKAKE': 'hikkake',
        'CDLHIKKAKEMOD': 'hikkake_mod',
        'CDLHOMINGPIGEON': 'homing_pigeon',
        'CDLIDENTICAL3CROWS': 'identical_three_crows',
        'CDLINNECK': 'in_neck',
        'CDLINVERTEDHAMMER': 'inverted_hammer',
        'CDLKICKING': 'kicking',
        'CDLKICKINGBYLENGTH': 'kicking_by_length',
        'CDLLADDERBOTTOM': 'ladder_bottom',
        'CDLLONGLEGGEDDOJI': 'long_legged_doji',
        'CDLLONGLINE': 'long_line',
        'CDLMARUBOZU': 'marubozu',
        'CDLMATCHINGLOW': 'matching_low',
        'CDLMATHOLD': 'mat_hold',
        'CDLMORNINGDOJISTAR': 'morning_doji_star',
        'CDLMORNINGSTAR': 'morning_star',
        'CDLONNECK': 'on_neck',
        'CDLPIERCING': 'piercing',
        'CDLRICKSHAWMAN': 'rickshaw_man',
        'CDLRISEFALL3METHODS': 'rise_fall_three_methods',
        'CDLSEPARATINGLINES': 'separating_lines',
        'CDLSHOOTINGSTAR': 'shooting_star',
        'CDLSHORTLINE': 'short_line',
        'CDLSPINNINGTOP': 'spinning_top',
        'CDLSTALLEDPATTERN': 'stalled_pattern',
        'CDLSTICKSANDWICH': 'stick_sandwich',
        'CDLTAKURI': 'takuri',
        'CDLTASUKIGAP': 'tasuki_gap',
        'CDLTHRUSTING': 'thrusting',
        'CDLTRISTAR': 'tristar',
        'CDLUNIQUE3RIVER': 'unique_three_river',
        'CDLUPSIDEGAP2CROWS': 'upside_gap_two_crows',
        'CDLXSIDEGAP3METHODS': 'xside_gap_three_methods',
    }

    def __init__(
        self,
        logger=None,
        error_handler=None
    ):
        """
        Initialize TA-Lib Pattern Detector

        Args:
            logger: Logger instance
            error_handler: Error handler instance
        """
        super().__init__(
            name="TALibPatterns",
            category=IndicatorCategory.PATTERNS,
            logger=logger,
            error_handler=error_handler
        )

    def _requires_volume(self) -> bool:
        """Volume gerekli mi?"""
        return False

    def get_required_periods(self) -> int:
        """
        Minimum kaç bar gerekli?

        Multi-candle pattern'ler için 5 bar güvenli

        Returns:
            5
        """
        return 5

    def calculate(self, data: pd.DataFrame) -> Optional[IndicatorResult]:
        """
        Son bar için pattern tespit et (single bar mode)

        NOTE: Pattern detection için calculate_batch() kullanılmalı (daha hızlı)
        Bu metod sadece geriye dönük uyumluluk için mevcut.

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult with pattern dict
        """
        if len(data) < 5:
            return None

        # Son 5 bar'a bak (multi-candle pattern'ler için)
        patterns_df = self.calculate_batch(data.tail(5))

        if patterns_df is None or len(patterns_df) == 0:
            return None

        # Son bar'ın pattern'lerini al
        last_patterns = patterns_df.iloc[-1].to_dict()

        # Detected pattern'leri bul (100 veya -100)
        detected = {k: v for k, v in last_patterns.items() if v != 0}

        return IndicatorResult(
            value={
                'patterns': list(detected.keys()),
                'count': len(detected),
                'details': last_patterns
            },
            timestamp=int(data.iloc[-1]['timestamp'])
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Buffer'a yeni mum ekler ve yeterli veri varsa pattern hesaplar.

        Args:
            candle: New candle data (dict with open, high, low, close, timestamp)
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Detected patterns
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_candle_buffers'):
            self._candle_buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed (10 bars for safety)
        if buffer_key not in self._candle_buffers:
            self._candle_buffers[buffer_key] = deque(maxlen=10)

        # Add new candle to buffer
        self._candle_buffers[buffer_key].append(candle)

        # Check minimum data (5 bars required for multi-candle patterns)
        if len(self._candle_buffers[buffer_key]) < self.get_required_periods():
            return IndicatorResult(
                value={'patterns': [], 'count': 0, 'details': {}},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame(list(self._candle_buffers[buffer_key]))

        # Calculate patterns
        return self.calculate(buffer_data)

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm data için pattern tespit et (vectorized via TA-Lib)

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with 61 pattern columns (values: 100, -100, or 0)
        """
        if len(data) < 5:
            self._log('warning', "En az 5 bar gerekli (multi-candle pattern'ler için)")
            return pd.DataFrame(index=data.index)

        result = pd.DataFrame(index=data.index)

        # OHLC arrays
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        # ====================================================================
        # CALCULATE ALL 61 TA-LIB PATTERNS
        # ====================================================================

        for talib_name, friendly_name in self.PATTERN_MAP.items():
            try:
                # Get TA-Lib function dynamically
                talib_func = getattr(talib, talib_name)

                # Calculate pattern
                pattern_result = talib_func(open_prices, high_prices, low_prices, close_prices)

                # Store result
                result[friendly_name] = pattern_result

            except Exception as e:
                self._log('warning', f"Pattern {friendly_name} hesaplanamadı: {e}")
                result[friendly_name] = 0

        return result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pattern_list():
    """
    Tüm 61 pattern ismini döndür (lowercase)

    Returns:
        List of 61 pattern names
    """
    return list(TALibPatterns.PATTERN_MAP.values())


def get_bullish_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece bullish pattern'leri filtrele (value == 100)

    Args:
        data: Pattern DataFrame from calculate_batch()

    Returns:
        Filtered DataFrame with only bullish patterns
    """
    return data[data == 100].fillna(0)


def get_bearish_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece bearish pattern'leri filtrele (value == -100)

    Args:
        data: Pattern DataFrame from calculate_batch()

    Returns:
        Filtered DataFrame with only bearish patterns
    """
    return data[data == -100].fillna(0)


def count_patterns_per_bar(data: pd.DataFrame) -> pd.Series:
    """
    Her bar'da kaç pattern tespit edildiğini say

    Args:
        data: Pattern DataFrame from calculate_batch()

    Returns:
        Series with pattern counts per bar
    """
    return (data != 0).sum(axis=1)
