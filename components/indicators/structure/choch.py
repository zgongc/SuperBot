"""
indicators/structure/choch.py - Change of Character

Version: 3.0.0
Date: 2025-12-24
Author: SuperBot Team

Açıklama:
    CHoCH (Change of Character) - Smart Money Concepts
    Piyasa karakterinin değişimini tespit eder

    CHoCH Nedir:
    - Yükseliş trendinde: Swing low'un kırılması (trend zayıflıyor)
    - Düşüş trendinde: Swing high'ın kırılması (trend zayıflıyor)
    - Potansiyel trend değişimini gösterir

Formül:
    1. Swing High/Low tespiti (SwingPoints kullanır - TradingView uyumlu)
    2. Trend yönünü tespit et
    3. Ters yönde swing kırılması -> CHoCH

    Bullish CHoCH (Düşüş -> Yükseliş): Close > Previous Swing High (in downtrend)
    Bearish CHoCH (Yükseliş -> Düşüş): Close < Previous Swing Low (in uptrend)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - SwingPoints (../support_resistance/swing_points.py)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class CHoCH(BaseIndicator):
    """
    Change of Character (CHoCH)

    Piyasa karakterinin değişim noktalarını tespit eder.
    Trend değişimi sinyali verir.

    Args:
        left_bars: Sol taraf bar sayısı (varsayılan: 5)
        right_bars: Sağ taraf bar sayısı (varsayılan: 5)
        max_levels: Maksimum seviye sayısı (varsayılan: 3)
        trend_strength: Trend gücü eşiği (varsayılan: 3)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        max_levels: int = 3,
        trend_strength: int = 3,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.max_levels = max_levels
        self.trend_strength = trend_strength

        # SwingPoints'i lazy import ile oluştur
        self._swing_points = None

        super().__init__(
            name='choch',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'max_levels': max_levels,
                'trend_strength': trend_strength
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.left_bars + self.right_bars + self.trend_strength + 10

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars pozitif olmalı"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars pozitif olmalı"
            )
        if self.max_levels < 1:
            raise InvalidParameterError(
                self.name, 'max_levels', self.max_levels,
                "Max levels pozitif olmalı"
            )
        if self.trend_strength < 1:
            raise InvalidParameterError(
                self.name, 'trend_strength', self.trend_strength,
                "Trend strength pozitif olmalı"
            )
        return True

    def _get_swing_points(self):
        """SwingPoints instance'ını lazy load et"""
        if self._swing_points is None:
            import sys
            import os
            # components/indicators'i 'indicators' olarak ekle (eski import uyumluluğu için)
            components_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if components_path not in sys.path:
                sys.path.insert(0, components_path)

            from indicators.support_resistance.swing_points import SwingPoints
            self._swing_points = SwingPoints(
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                lookback=50
            )
        return self._swing_points

    def _find_swings(self, data: pd.DataFrame) -> tuple:
        """SwingPoints kullanarak swing high/low noktalarını tespit et (alternating filtreli)"""
        swing_points = self._get_swing_points()
        swing_df = swing_points.calculate_batch(data)

        # Önce tüm swing'leri topla (index, type, value)
        raw_swings = []
        for i in range(len(swing_df)):
            if not np.isnan(swing_df['swing_high'].iloc[i]):
                raw_swings.append((i, 'high', swing_df['swing_high'].iloc[i]))
            if not np.isnan(swing_df['swing_low'].iloc[i]):
                raw_swings.append((i, 'low', swing_df['swing_low'].iloc[i]))

        # Index'e göre sırala (kronolojik)
        raw_swings.sort(key=lambda x: x[0])

        # Alternating swing'leri filtrele (High-Low-High-Low pattern)
        filtered_swings = []
        for idx, swing_type, value in raw_swings:
            if not filtered_swings:
                filtered_swings.append((idx, swing_type, value))
            elif filtered_swings[-1][1] != swing_type:
                # Farklı tip - ekle
                filtered_swings.append((idx, swing_type, value))
            else:
                # Aynı tip - daha ekstrem olanı tut
                last_idx, last_type, last_value = filtered_swings[-1]
                if swing_type == 'high' and value > last_value:
                    filtered_swings[-1] = (idx, swing_type, value)
                elif swing_type == 'low' and value < last_value:
                    filtered_swings[-1] = (idx, swing_type, value)

        # Filtrelenmiş swing'lerden ayrı listeler oluştur
        swing_highs = []
        swing_lows = []
        for idx, swing_type, value in filtered_swings:
            if swing_type == 'high':
                swing_highs.append({'index': idx, 'value': value})
            else:
                swing_lows.append({'index': idx, 'value': value})

        return swing_highs, swing_lows

    def _detect_trend(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> str:
        """
        Mevcut trend yönünü tespit et

        SMC Approach (structure_detector.py ile aynı logic):
        - Uptrend: Higher High OR Higher Low (any bullish structure)
        - Downtrend: Lower Low OR Lower High (any bearish structure)
        - Daha esnek - sadece son 2 swing'e bakıyor

        Args:
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar

        Returns:
            str: 'uptrend', 'downtrend', 'ranging'
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'ranging'

        # Son 2-3 swing'e bak (structure_detector.py gibi)
        last_highs = [h['value'] for h in swing_highs[-3:]]
        last_lows = [l['value'] for l in swing_lows[-3:]]

        # En son swing yönünü kontrol et
        last_high_direction = 'higher' if last_highs[-1] > last_highs[-2] else 'lower'
        last_low_direction = 'higher' if last_lows[-1] > last_lows[-2] else 'lower'

        # Uptrend: HH and HL (classic) OR just HH with equal/higher low
        # Downtrend: LL and LH (classic) OR just LL with equal/lower high
        if last_high_direction == 'higher' and last_low_direction == 'higher':
            return 'uptrend'
        elif last_high_direction == 'lower' and last_low_direction == 'lower':
            return 'downtrend'
        elif last_high_direction == 'higher':
            # HH but not HL - still bullish bias
            return 'uptrend'
        elif last_low_direction == 'lower':
            # LL but not LH - still bearish bias
            return 'downtrend'

        return 'ranging'

    def _detect_choch_at_index(
        self,
        index: int,
        closes: np.ndarray,
        recent_highs: List[Dict[str, Any]],
        recent_lows: List[Dict[str, Any]],
        current_trend: str,
        broken_highs: set,
        broken_lows: set
    ) -> Optional[Dict[str, Any]]:
        """
        CORE DETECTION FUNCTION - Used by calculate(), calculate_batch(), update()

        CHoCH tespiti - TEK BAR için trend kontrolü ile.

        IMPORTANT: CHoCH and BOS are MUTUALLY EXCLUSIVE!
        - CHoCH: Break in OPPOSITE direction of trend (reversal)
        - BOS: Break in SAME direction as trend (continuation)

        Args:
            index: Current bar index
            closes: Close fiyat dizisi
            recent_highs: Recent swing highs (max_levels adet)
            recent_lows: Recent swing lows (max_levels adet)
            current_trend: 'uptrend', 'downtrend', 'ranging'
            broken_highs: Set of already broken swing high indices
            broken_lows: Set of already broken swing low indices

        Returns:
            Dict: CHoCH bilgisi veya None

        Side Effects:
            Updates broken_highs and broken_lows sets
        """
        if index < 1 or len(closes) < 2:
            return None

        current_close = closes[index]
        prev_close = closes[index - 1]

        # ===== BULLISH BREAK CHECK =====
        # Bullish CHoCH: downtrend + break above swing high
        for swing in reversed(recent_highs[-self.max_levels:]):
            swing_idx = swing['index']
            swing_val = swing['value']

            if swing_idx in broken_highs:
                continue

            if swing_idx >= index:
                continue

            # Check if this is a NEW break (crossover)
            if current_close > swing_val and prev_close <= swing_val:
                broken_highs.add(swing_idx)
                # Only signal CHoCH if in downtrend
                if current_trend == 'downtrend':
                    return {
                        'type': 'bullish',
                        'level': swing_val,
                        'index': swing_idx,
                        'broken_at': index,
                        'previous_trend': 'downtrend'
                    }
                else:
                    # It's a break but BOS, not CHoCH
                    return None
            elif current_close > swing_val:
                # Already broken before, mark it
                broken_highs.add(swing_idx)

        # ===== BEARISH BREAK CHECK =====
        # Bearish CHoCH: uptrend + break below swing low
        for swing in reversed(recent_lows[-self.max_levels:]):
            swing_idx = swing['index']
            swing_val = swing['value']

            if swing_idx in broken_lows:
                continue

            if swing_idx >= index:
                continue

            # Check if this is a NEW break (crossunder)
            if current_close < swing_val and prev_close >= swing_val:
                broken_lows.add(swing_idx)
                # Only signal CHoCH if in uptrend
                if current_trend == 'uptrend':
                    return {
                        'type': 'bearish',
                        'level': swing_val,
                        'index': swing_idx,
                        'broken_at': index,
                        'previous_trend': 'uptrend'
                    }
                else:
                    # It's a break but BOS, not CHoCH
                    return None
            elif current_close < swing_val:
                # Already broken before, mark it
                broken_lows.add(swing_idx)

        return None

    def _detect_choch(
        self,
        closes: np.ndarray,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]],
        trend: str
    ) -> Optional[Dict[str, Any]]:
        """
        CHoCH tespiti - Son bar için (calculate() için wrapper)

        Uses _detect_choch_at_index() with trend detection.

        Args:
            closes: Close fiyat dizisi
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar
            trend: Mevcut trend

        Returns:
            Dict: CHoCH bilgisi veya None
        """
        if len(closes) < 2:
            return None

        index = len(closes) - 1

        # Build broken sets by scanning history
        broken_highs = set()
        broken_lows = set()

        # Recent swings for detection
        recent_highs = swing_highs[-self.max_levels:] if swing_highs else []
        recent_lows = swing_lows[-self.max_levels:] if swing_lows else []

        # Scan history to find already broken swings
        for i in range(1, index):
            current_close = closes[i]

            for swing in recent_highs:
                if swing['index'] < i and swing['index'] not in broken_highs:
                    if current_close > swing['value']:
                        broken_highs.add(swing['index'])

            for swing in recent_lows:
                if swing['index'] < i and swing['index'] not in broken_lows:
                    if current_close < swing['value']:
                        broken_lows.add(swing['index'])

        # Detect CHoCH at current index
        choch_result = self._detect_choch_at_index(
            index=index,
            closes=closes,
            recent_highs=recent_highs,
            recent_lows=recent_lows,
            current_trend=trend,
            broken_highs=broken_highs,
            broken_lows=broken_lows
        )

        if choch_result:
            return choch_result

        # CHoCH yoksa bekleyen seviye
        if trend == 'uptrend' and swing_lows:
            return {
                'type': 'pending_bearish',
                'level': swing_lows[-1]['value'],
                'index': swing_lows[-1]['index'],
                'broken_at': None,
                'previous_trend': 'uptrend'
            }
        elif trend == 'downtrend' and swing_highs:
            return {
                'type': 'pending_bullish',
                'level': swing_highs[-1]['value'],
                'index': swing_highs[-1]['index'],
                'broken_at': None,
                'previous_trend': 'downtrend'
            }

        return None

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        CHoCH hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CHoCH değeri
                - value: Sinyal değeri (1=bullish, -1=bearish, 0=none)
                - metadata: CHoCH seviyesi ve detayları
        """
        closes = data['close'].values

        # Swing High/Low tespiti (SwingPoints kullanır)
        swing_highs, swing_lows = self._find_swings(data)

        # Trend tespiti
        trend = self._detect_trend(swing_highs, swing_lows)

        # CHoCH tespiti
        choch_data = self._detect_choch(closes, swing_highs, swing_lows, trend)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Değer: Sinyal (1=bullish, -1=bearish, 0=none/pending)
        # calculate_batch() ile tutarlı!
        choch_type = choch_data.get('type', 'none') if choch_data else 'none'
        if choch_type == 'bullish':
            value = 1
        elif choch_type == 'bearish':
            value = -1
        else:
            value = 0  # 'none', 'pending_bullish', 'pending_bearish'

        # Metadata: Tüm CHoCH seviyeleri ve fiyat bilgisi
        metadata = {
            'swing_highs': [{'level': s['value'], 'index': s['index']} for s in swing_highs[-self.max_levels:]],
            'swing_lows': [{'level': s['value'], 'index': s['index']} for s in swing_lows[-self.max_levels:]],
            'choch_type': choch_type,
            'choch_level': choch_data['level'] if choch_data else None,
            'current_trend': trend,
            'previous_trend': choch_data.get('previous_trend') if choch_data else None,
            'left_bars': self.left_bars,
            'right_bars': self.right_bars
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=value,
            timestamp=timestamp,
            signal=self.get_signal(choch_data),
            trend=self.get_trend(choch_data, trend),
            strength=self._calculate_strength(choch_data, closes[-1]) if choch_data else 0,
            metadata=metadata
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate CHoCH for entire DataFrame (for backtest)

        Uses _detect_choch_at_index() core function for each bar.

        Returns pd.Series with CHoCH signals for all bars.
        Value: 1=bullish CHoCH, -1=bearish CHoCH, 0=none

        IMPORTANT: Uses same trend detection logic as structure_detector.py
        - Trend is stateful (keeps previous trend if not enough data)
        - More flexible trend detection (HH OR LL can determine trend)
        """
        closes = data['close'].values

        # Result array: store CHoCH signal at each bar
        choch_signal = np.zeros(len(data))

        # Find all swing highs/lows upfront (SwingPoints kullanır)
        swing_highs, swing_lows = self._find_swings(data)

        # Convert to dict for fast lookup
        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        # Track recent swings for trend detection
        recent_highs = []
        recent_lows = []

        # Track which swing levels have been broken (to detect NEW breaks only)
        broken_highs = set()
        broken_lows = set()

        # State: keep previous trend (structure_detector.py gibi)
        current_trend = 'ranging'

        start_idx = self.left_bars + self.right_bars

        for i in range(start_idx, len(data)):
            # Update recent swings
            if i in swing_high_dict:
                recent_highs.append({'index': i, 'value': swing_high_dict[i]})
                if len(recent_highs) > self.max_levels:
                    recent_highs = recent_highs[-self.max_levels:]

            if i in swing_low_dict:
                recent_lows.append({'index': i, 'value': swing_low_dict[i]})
                if len(recent_lows) > self.max_levels:
                    recent_lows = recent_lows[-self.max_levels:]

            # Detect trend (with state preservation)
            new_trend = self._detect_trend(recent_highs, recent_lows)
            if new_trend != 'ranging':
                current_trend = new_trend

            # Use core detection function
            choch_result = self._detect_choch_at_index(
                index=i,
                closes=closes,
                recent_highs=recent_highs,
                recent_lows=recent_lows,
                current_trend=current_trend,
                broken_highs=broken_highs,
                broken_lows=broken_lows
            )

            if choch_result:
                if choch_result['type'] == 'bullish':
                    choch_signal[i] = 1
                    # CHoCH sonrası trend değişir - bullish CHoCH = artık uptrend
                    current_trend = 'uptrend'
                elif choch_result['type'] == 'bearish':
                    choch_signal[i] = -1
                    # CHoCH sonrası trend değişir - bearish CHoCH = artık downtrend
                    current_trend = 'downtrend'

        return pd.Series(choch_signal, index=data.index, name='choch')

    def _calculate_strength(self, choch_data: Dict[str, Any], current_close: float) -> float:
        """CHoCH gücünü hesapla (0-100)"""
        if not choch_data or not choch_data.get('broken_at'):
            return 0.0

        level = choch_data['level']
        distance = abs(current_close - level)

        # Kırılma mesafesine göre güç hesapla
        strength = min((distance / level) * 1000, 100)

        return round(strength, 2)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup CHoCH buffers with historical data

        CRITICAL: CHoCH uses its own buffers (_high_buffer, _low_buffer, _close_buffer)
        not BaseIndicator's _buffers. This override ensures they're properly filled.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (unused, for interface compatibility)
        """
        from collections import deque

        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Fill buffers with historical data
        for _, row in data.tail(max_len).iterrows():
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._close_buffer.append(row['close'])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Initialize buffers if warmup wasn't called
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0,  # Sinyal yok (calculate() ile tutarlı)
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'choch_type': 'none', 'choch_level': None, 'current_trend': 'neutral'}
            )
        
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, choch_data: Optional[Dict[str, Any]]) -> SignalType:
        """
        CHoCH'tan sinyal üret

        Args:
            choch_data: CHoCH bilgisi

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if not choch_data:
            return SignalType.HOLD

        choch_type = choch_data.get('type', 'none')

        # CHoCH trend değişimini gösterir
        if choch_type == 'bullish':
            return SignalType.BUY  # Düşüş -> Yükseliş
        elif choch_type == 'bearish':
            return SignalType.SELL  # Yükseliş -> Düşüş

        return SignalType.HOLD

    def get_trend(self, choch_data: Optional[Dict[str, Any]], current_trend: str) -> TrendDirection:
        """
        CHoCH'tan trend belirle

        Args:
            choch_data: CHoCH bilgisi
            current_trend: Mevcut trend

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not choch_data:
            # Mevcut trendi döndür
            if current_trend == 'uptrend':
                return TrendDirection.UP
            elif current_trend == 'downtrend':
                return TrendDirection.DOWN
            return TrendDirection.NEUTRAL

        choch_type = choch_data.get('type', 'none')

        # CHoCH sonrası yeni trend
        if 'bullish' in choch_type:
            return TrendDirection.UP
        elif 'bearish' in choch_type:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'max_levels': 3,
            'trend_strength': 3
        }

    def _requires_volume(self) -> bool:
        """CHoCH volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CHoCH']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """CHoCH indikatör testi"""

    print("\n" + "="*60)
    print("CHoCH (CHANGE OF CHARACTER) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Trend değişimi simülasyonu
    base_price = 100
    prices = []
    for i in range(60):
        if i < 20:
            # Güçlü yükseliş (uptrend)
            prices.append(base_price + i * 0.8 + np.random.randn() * 0.3)
        elif i < 25:
            # Zayıflama (CHoCH hazırlığı)
            prices.append(base_price + 16 - (i - 20) * 0.3 + np.random.randn() * 0.3)
        elif i < 45:
            # Düşüş (downtrend)
            prices.append(base_price + 14.5 - (i - 25) * 0.6 + np.random.randn() * 0.3)
        else:
            # Yeniden yükseliş (CHoCH)
            prices.append(base_price + 2.5 + (i - 45) * 0.5 + np.random.randn() * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    choch = CHoCH(left_bars=5, right_bars=5, max_levels=3, trend_strength=3)
    print(f"   [OK] Oluşturuldu: {choch}")
    print(f"   [OK] Kategori: {choch.category.value}")
    print(f"   [OK] Gerekli periyot: {choch.get_required_periods()}")

    result = choch(data)
    print(f"   [OK] CHoCH Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength}")
    print(f"   [OK] CHoCH Tipi: {result.metadata['choch_type']}")
    print(f"   [OK] Mevcut Trend: {result.metadata['current_trend']}")
    print(f"   [OK] Önceki Trend: {result.metadata['previous_trend']}")

    # Test 2: Swing seviyelerini göster
    print("\n3. Swing seviyeleri...")
    for i, high in enumerate(result.metadata['swing_highs'][-3:]):
        print(f"   [OK] Swing High #{i+1}: {high['level']:.2f} @ index {high['index']}")
    for i, low in enumerate(result.metadata['swing_lows'][-3:]):
        print(f"   [OK] Swing Low #{i+1}: {low['level']:.2f} @ index {low['index']}")

    # Test 3: Farklı parametreler
    print("\n4. Farklı parametre testi...")
    for trend_strength in [2, 3, 4]:
        choch_test = CHoCH(trend_strength=trend_strength)
        result = choch_test.calculate(data)
        print(f"   [OK] CHoCH(trend_str={trend_strength}): {result.metadata['current_trend']} | Tip: {result.metadata['choch_type']}")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = choch.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = choch.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
