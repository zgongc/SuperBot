#!/usr/bin/env python3
"""
components/indicators/patterns/candlestick_patterns.py
SuperBot - Candlestick Pattern Detector
Yazar: SuperBot Team
Tarih: 2025-10-26
Versiyon: 1.0.0

🎯 Görev:
    Popüler candlestick pattern'lerini tespit eder (vectorized)

📊 Pattern'ler (15 adet):
    Single Candle:
    - Doji (ve varyantları: Dragonfly, Gravestone, Long-legged)
    - Hammer / Inverted Hammer
    - Hanging Man / Shooting Star
    - Marubozu (Bullish/Bearish)
    - Spinning Top

    Multi Candle:
    - Engulfing (Bullish/Bearish)
    - Harami (Bullish/Bearish)
    - Morning Star / Evening Star
    - Piercing Line / Dark Cloud Cover
    - Three White Soldiers / Three Black Crows

🔧 Kullanım:
    # Strategy template'de
    technical_parameters:
      indicators:
        candlestick_patterns:
          enabled: true
          doji_threshold: 0.1           # Body/Range ratio
          shadow_ratio: 2.0             # Shadow/Body ratio
          min_body_size: 0.0001         # Minimum body size (anti-noise)

    # Entry condition
    entry_conditions:
      buy:
        - ["hammer", "equals", true, "1m"]
        - ["rsi_14", "below", 30, "1m"]

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import pandas as pd
import numpy as np
from typing import Optional
from collections import deque

from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import SignalType, TrendDirection
from components.indicators.indicator_types import (
    IndicatorResult,
    IndicatorCategory,
    IndicatorType
)


# ============================================================================
# CANDLESTICK PATTERNS INDICATOR
# ============================================================================

class CandlestickPatterns(BaseIndicator):
    """
    Candlestick Pattern Detector

    15 popüler candlestick pattern'ini vectorized şekilde tespit eder.

    Attributes:
        doji_threshold: Doji için body/range oranı (default: 0.1)
        shadow_ratio: Hammer/Shooting Star için shadow/body oranı (default: 2.0)
        min_body_size: Minimum body size (noise filtreleme) (default: 0.0001)

    Output (DataFrame):
        - doji: Doji pattern
        - dragonfly_doji: Dragonfly Doji
        - gravestone_doji: Gravestone Doji
        - longlegged_doji: Long-legged Doji
        - hammer: Hammer (bullish)
        - inverted_hammer: Inverted Hammer (bullish)
        - hanging_man: Hanging Man (bearish)
        - shooting_star: Shooting Star (bearish)
        - marubozu_bullish: Bullish Marubozu
        - marubozu_bearish: Bearish Marubozu
        - spinning_top: Spinning Top
        - engulfing_bullish: Bullish Engulfing
        - engulfing_bearish: Bearish Engulfing
        - harami_bullish: Bullish Harami
        - harami_bearish: Bearish Harami
        - morning_star: Morning Star (3-bar bullish)
        - evening_star: Evening Star (3-bar bearish)
        - piercing_line: Piercing Line (bullish)
        - dark_cloud_cover: Dark Cloud Cover (bearish)
        - three_white_soldiers: Three White Soldiers (bullish)
        - three_black_crows: Three Black Crows (bearish)
    """

    def __init__(
        self,
        doji_threshold: float = 0.1,
        shadow_ratio: float = 2.0,
        min_body_size: float = 0.0001,
        logger=None,
        error_handler=None
    ):
        """
        Initialize Candlestick Patterns Indicator

        Args:
            doji_threshold: Doji için body/range oranı (0.1 = %10)
            shadow_ratio: Hammer/Star için shadow/body oranı (2.0 = 2x)
            min_body_size: Minimum body size (noise filter)
            logger: Logger instance
            error_handler: Error handler instance
        """
        super().__init__(
            name="CandlestickPatterns",
            category=IndicatorCategory.PATTERNS,
            logger=logger,
            error_handler=error_handler
        )

        self.doji_threshold = doji_threshold
        self.shadow_ratio = shadow_ratio
        self.min_body_size = min_body_size

        self._validate_parameters()

    def _validate_parameters(self):
        """Parametre validasyonu"""
        if self.doji_threshold <= 0 or self.doji_threshold >= 1:
            raise ValueError(f"doji_threshold 0 ile 1 arasında olmalı, aldı: {self.doji_threshold}")

        if self.shadow_ratio < 1:
            raise ValueError(f"shadow_ratio >= 1 olmalı, aldı: {self.shadow_ratio}")

        if self.min_body_size < 0:
            raise ValueError(f"min_body_size >= 0 olmalı, aldı: {self.min_body_size}")

    def _requires_volume(self) -> bool:
        """Volume gerekli mi?"""
        return False

    def get_required_periods(self) -> int:
        """
        Minimum kaç bar gerekli?

        Multi-candle pattern'ler için 3 bar gerekli (Morning Star, Evening Star, etc.)

        Returns:
            3
        """
        return 3

    def calculate(self, data: pd.DataFrame) -> Optional[IndicatorResult]:
        """
        Son bar için pattern tespit et (single bar mode)

        NOT: Pattern detection için calculate_batch() kullanılmalı (daha hızlı)
        Bu metod sadece geriye dönük uyumluluk için mevcut.

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult with pattern dict
        """
        if len(data) < 3:
            return None

        # Son 3 bar'a bak (multi-candle pattern'ler için)
        patterns_df = self.calculate_batch(data.tail(3))

        if patterns_df is None or len(patterns_df) == 0:
            return None

        # Son bar'ın pattern'lerini al
        last_patterns = patterns_df.iloc[-1].to_dict()

        # True olan pattern'leri bul
        active_patterns = [k for k, v in last_patterns.items() if v == True]

        return IndicatorResult(
            value={
                'patterns': active_patterns,
                'count': len(active_patterns),
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

        # Initialize buffer for this symbol if needed (5 bars for safety)
        if buffer_key not in self._candle_buffers:
            self._candle_buffers[buffer_key] = deque(maxlen=5)

        # Add new candle to buffer
        self._candle_buffers[buffer_key].append(candle)

        # Check minimum data (3 bars required for multi-candle patterns)
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
        Tüm data için pattern tespit et (vectorized)

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with pattern columns (bool)
        """
        if len(data) < 3:
            self._log('warning', "En az 3 bar gerekli (multi-candle pattern'ler için)")
            return pd.DataFrame(index=data.index)

        result = pd.DataFrame(index=data.index)

        # Temel hesaplamalar
        body = abs(data['close'] - data['open'])
        total_range = data['high'] - data['low']
        upper_shadow = data['high'] - data[['close', 'open']].max(axis=1)
        lower_shadow = data[['close', 'open']].min(axis=1) - data['low']

        # Body ratio (body size / total range)
        body_ratio = body / total_range.replace(0, np.nan)

        # Bullish/Bearish
        is_bullish = data['close'] > data['open']
        is_bearish = data['close'] < data['open']

        # ====================================================================
        # SINGLE CANDLE PATTERNS
        # ====================================================================

        # 1. Doji (body çok küçük)
        result['doji'] = (body_ratio < self.doji_threshold) & (body > self.min_body_size)

        # 2. Dragonfly Doji (body küçük, üst shadow yok, alt shadow uzun)
        result['dragonfly_doji'] = (
            result['doji'] &
            (upper_shadow < body) &
            (lower_shadow > self.shadow_ratio * body)
        )

        # 3. Gravestone Doji (body küçük, alt shadow yok, üst shadow uzun)
        result['gravestone_doji'] = (
            result['doji'] &
            (lower_shadow < body) &
            (upper_shadow > self.shadow_ratio * body)
        )

        # 4. Long-legged Doji (body küçük, her iki shadow da uzun)
        result['longlegged_doji'] = (
            result['doji'] &
            (upper_shadow > self.shadow_ratio * body) &
            (lower_shadow > self.shadow_ratio * body)
        )

        # 5. Hammer (bullish, alt shadow uzun, üst shadow küçük)
        result['hammer'] = (
            is_bullish &
            (lower_shadow > self.shadow_ratio * body) &
            (upper_shadow < body) &
            (body > self.min_body_size)
        )

        # 6. Inverted Hammer (bullish, üst shadow uzun, alt shadow küçük)
        result['inverted_hammer'] = (
            is_bullish &
            (upper_shadow > self.shadow_ratio * body) &
            (lower_shadow < body) &
            (body > self.min_body_size)
        )

        # 7. Hanging Man (bearish, alt shadow uzun, üst shadow küçük)
        result['hanging_man'] = (
            is_bearish &
            (lower_shadow > self.shadow_ratio * body) &
            (upper_shadow < body) &
            (body > self.min_body_size)
        )

        # 8. Shooting Star (bearish, üst shadow uzun, alt shadow küçük)
        result['shooting_star'] = (
            is_bearish &
            (upper_shadow > self.shadow_ratio * body) &
            (lower_shadow < body) &
            (body > self.min_body_size)
        )

        # 9. Bullish Marubozu (shadow'lar çok küçük, body büyük, bullish)
        result['marubozu_bullish'] = (
            is_bullish &
            (body_ratio > 0.9) &
            (upper_shadow < 0.05 * body) &
            (lower_shadow < 0.05 * body)
        )

        # 10. Bearish Marubozu (shadow'lar çok küçük, body büyük, bearish)
        result['marubozu_bearish'] = (
            is_bearish &
            (body_ratio > 0.9) &
            (upper_shadow < 0.05 * body) &
            (lower_shadow < 0.05 * body)
        )

        # 11. Spinning Top (body küçük, shadow'lar orta boy)
        result['spinning_top'] = (
            (body_ratio > self.doji_threshold) &
            (body_ratio < 0.3) &
            (upper_shadow > body * 0.5) &
            (lower_shadow > body * 0.5)
        )

        # ====================================================================
        # MULTI CANDLE PATTERNS (2-bar)
        # ====================================================================

        # Shift values (previous bar)
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        prev_high = data['high'].shift(1)
        prev_low = data['low'].shift(1)
        prev_body = abs(prev_close - prev_open)

        # 12. Bullish Engulfing (bullish bar engulfs previous bearish bar)
        result['engulfing_bullish'] = (
            is_bullish &
            (prev_close < prev_open) &  # Previous bearish
            (data['open'] <= prev_close) &  # Opens at/below prev close
            (data['close'] >= prev_open)  # Closes at/above prev open
        )

        # 13. Bearish Engulfing (bearish bar engulfs previous bullish bar)
        result['engulfing_bearish'] = (
            is_bearish &
            (prev_close > prev_open) &  # Previous bullish
            (data['open'] >= prev_close) &  # Opens at/above prev close
            (data['close'] <= prev_open)  # Closes at/below prev open
        )

        # 14. Bullish Harami (small bullish inside previous large bearish)
        result['harami_bullish'] = (
            is_bullish &
            (prev_close < prev_open) &  # Previous bearish
            (data['open'] >= prev_close) &  # Opens above prev close
            (data['close'] <= prev_open) &  # Closes below prev open
            (body < prev_body * 0.5)  # Current body < 50% of prev body
        )

        # 15. Bearish Harami (small bearish inside previous large bullish)
        result['harami_bearish'] = (
            is_bearish &
            (prev_close > prev_open) &  # Previous bullish
            (data['open'] <= prev_close) &  # Opens below prev close
            (data['close'] >= prev_open) &  # Closes above prev open
            (body < prev_body * 0.5)  # Current body < 50% of prev body
        )

        # 16. Piercing Line (bullish reversal, 2-bar)
        result['piercing_line'] = (
            is_bullish &
            (prev_close < prev_open) &  # Previous bearish
            (data['open'] < prev_low) &  # Gap down
            (data['close'] > (prev_open + prev_close) / 2) &  # Close above 50% of prev body
            (data['close'] < prev_open)  # But below prev open
        )

        # 17. Dark Cloud Cover (bearish reversal, 2-bar)
        result['dark_cloud_cover'] = (
            is_bearish &
            (prev_close > prev_open) &  # Previous bullish
            (data['open'] > prev_high) &  # Gap up
            (data['close'] < (prev_open + prev_close) / 2) &  # Close below 50% of prev body
            (data['close'] > prev_open)  # But above prev open
        )

        # ====================================================================
        # MULTI CANDLE PATTERNS (3-bar)
        # ====================================================================

        # Shift values (2 bars ago)
        prev2_open = data['open'].shift(2)
        prev2_close = data['close'].shift(2)
        prev2_high = data['high'].shift(2)
        prev2_low = data['low'].shift(2)

        # 18. Morning Star (bullish reversal, 3-bar)
        result['morning_star'] = (
            # First bar: Large bearish
            (prev2_close < prev2_open) &
            (abs(prev2_close - prev2_open) > total_range.shift(2) * 0.5) &

            # Second bar: Small body (star)
            (prev_body < body.shift(2) * 0.3) &
            (prev_close < prev2_close) &  # Gap down

            # Third bar: Large bullish
            is_bullish &
            (body > total_range * 0.5) &
            (data['close'] > (prev2_open + prev2_close) / 2)  # Close above 50% of first bar
        )

        # 19. Evening Star (bearish reversal, 3-bar)
        result['evening_star'] = (
            # First bar: Large bullish
            (prev2_close > prev2_open) &
            (abs(prev2_close - prev2_open) > total_range.shift(2) * 0.5) &

            # Second bar: Small body (star)
            (prev_body < body.shift(2) * 0.3) &
            (prev_close > prev2_close) &  # Gap up

            # Third bar: Large bearish
            is_bearish &
            (body > total_range * 0.5) &
            (data['close'] < (prev2_open + prev2_close) / 2)  # Close below 50% of first bar
        )

        # 20. Three White Soldiers (strong bullish, 3-bar)
        result['three_white_soldiers'] = (
            # All three bars bullish
            is_bullish &
            (prev_close > prev_open) &
            (prev2_close > prev2_open) &

            # Each bar closes higher
            (data['close'] > prev_close) &
            (prev_close > prev2_close) &

            # Each bar opens within previous body
            (data['open'] > prev_open) &
            (data['open'] < prev_close) &
            (prev_open > prev2_open) &
            (prev_open < prev2_close) &

            # Strong bodies
            (body_ratio > 0.6) &
            (body_ratio.shift(1) > 0.6) &
            (body_ratio.shift(2) > 0.6)
        )

        # 21. Three Black Crows (strong bearish, 3-bar)
        result['three_black_crows'] = (
            # All three bars bearish
            is_bearish &
            (prev_close < prev_open) &
            (prev2_close < prev2_open) &

            # Each bar closes lower
            (data['close'] < prev_close) &
            (prev_close < prev2_close) &

            # Each bar opens within previous body
            (data['open'] < prev_open) &
            (data['open'] > prev_close) &
            (prev_open < prev2_open) &
            (prev_open > prev2_close) &

            # Strong bodies
            (body_ratio > 0.6) &
            (body_ratio.shift(1) > 0.6) &
            (body_ratio.shift(2) > 0.6)
        )

        # Fill NaN with False (ilk barlar için)
        result = result.fillna(False)

        self._log('debug', f"Pattern detection tamamlandı: {len(result)} bar, {len(result.columns)} pattern")

        return result

    def get_pattern_summary(self, data: pd.DataFrame) -> dict:
        """
        Pattern özeti (kaç adet pattern tespit edildi)

        Args:
            data: OHLCV DataFrame

        Returns:
            Dict with pattern counts
        """
        patterns_df = self.calculate_batch(data)

        if patterns_df is None or len(patterns_df) == 0:
            return {}

        # Her pattern için toplam sayı
        summary = {}
        for col in patterns_df.columns:
            count = patterns_df[col].sum()
            if count > 0:
                summary[col] = int(count)

        return summary


# ============================================================================
# TEST SECTION (if __name__ == "__main__")
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    print("="*80)
    print("CANDLESTICK PATTERNS TEST")
    print("="*80)

    # Sample data (10 bars)
    sample_data = pd.DataFrame({
        'timestamp': [1704067200000 + i*60000 for i in range(10)],
        'open': [100, 102, 101, 103, 102, 105, 104, 107, 106, 108],
        'high': [103, 104, 103, 105, 104, 107, 106, 109, 108, 110],
        'low': [99, 101, 100, 102, 101, 104, 103, 106, 105, 107],
        'close': [102, 101, 103, 102, 105, 104, 107, 106, 108, 109],
        'volume': [1000]*10
    })

    # Initialize
    detector = CandlestickPatterns(
        doji_threshold=0.1,
        shadow_ratio=2.0,
        min_body_size=0.01
    )

    # Test calculate_batch
    print("\n" + "-"*80)
    print("TEST 1: calculate_batch()")
    print("-"*80)

    patterns = detector.calculate_batch(sample_data)
    print(f"\nPattern DataFrame shape: {patterns.shape}")
    print(f"Pattern columns: {list(patterns.columns)}")

    # Count patterns
    print("\nPattern Summary:")
    for col in patterns.columns:
        count = patterns[col].sum()
        if count > 0:
            print(f"  {col:25s}: {count} bar")

    # Test calculate (single bar)
    print("\n" + "-"*80)
    print("TEST 2: calculate() - Son bar")
    print("-"*80)

    result = detector.calculate(sample_data)
    if result:
        print(f"\nActive patterns: {result.value['patterns']}")
        print(f"Pattern count: {result.value['count']}")
    else:
        print("No patterns detected")

    # Test get_pattern_summary
    print("\n" + "-"*80)
    print("TEST 3: get_pattern_summary()")
    print("-"*80)

    summary = detector.get_pattern_summary(sample_data)
    print(f"\nPattern Summary: {summary}")

    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)
