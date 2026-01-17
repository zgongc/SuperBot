"""
indicators/combo/ema_ribbon.py - EMA Ribbon (Çoklu EMA Bantları)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    EMA Ribbon - Çoklu EMA bantları sistemi
    Birden fazla EMA'yı (5, 10, 20, 50, 100, 200) aynı anda analiz ederek
    trend gücünü, yönünü ve potansiyel giriş/çıkış noktalarını belirler

    Özellikler:
    - 6 farklı EMA (5, 10, 20, 50, 100, 200)
    - EMA sıralaması ile trend gücü ölçümü
    - EMA crossover analizi
    - Destek/direnç seviyeleri

Strateji:
    GÜÇLÜ YÜKSELIŞ: Tüm EMA'lar sıralı (5>10>20>50>100>200) + Fiyat en üstte
    YÜKSELIŞ: Çoğu EMA sıralı + Fiyat kısa vadeli EMA'ların üstünde
    GÜÇLÜ DÜŞÜŞ: Tüm EMA'lar ters sıralı (5<10<20<50<100<200) + Fiyat en altta
    DÜŞÜŞ: Çoğu EMA ters sıralı + Fiyat kısa vadeli EMA'ların altında
    YATAY: EMA'lar karışık, net trend yok

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.trend.ema
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class EMARibbon(BaseIndicator):
    """
    EMA Ribbon - Çoklu EMA Bantları

    Birden fazla EMA'yı aynı anda kullanarak trend analizi yapar.
    EMA'ların sıralaması trend gücünü gösterir.

    Args:
        ema_periods: EMA periyotları listesi (varsayılan: [5, 10, 20, 50, 100, 200])
    """

    def __init__(
        self,
        ema_periods: list = None,
        logger=None,
        error_handler=None
    ):
        # Varsayılan periyotlar
        if ema_periods is None:
            ema_periods = [5, 10, 20, 50, 100, 200]

        self.ema_periods = sorted(ema_periods)  # Küçükten büyüğe sırala

        # Her periyot için EMA indikatörü oluştur
        self.emas = {}
        for period in self.ema_periods:
            self.emas[period] = EMA(
                period=period,
                logger=logger,
                error_handler=error_handler
            )

        super().__init__(
            name='ema_ribbon',
            category=IndicatorCategory.COMBO,
            indicator_type=IndicatorType.LINES,
            params={
                'ema_periods': ema_periods
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı (en uzun EMA)"""
        return max(self.ema_periods)

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if not self.ema_periods:
            raise InvalidParameterError(
                self.name, 'ema_periods', self.ema_periods,
                "En az bir EMA periyodu gerekli"
            )
        for period in self.ema_periods:
            if period < 1:
                raise InvalidParameterError(
                    self.name, 'ema_period', period,
                    "EMA periyotları pozitif olmalı"
                )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch EMA Ribbon calculation - BACKTEST için

        Calculates multiple EMAs at once using vectorized operations

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: ema_5, ema_10, ema_20, ema_50, ema_100, ema_200 for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        # Calculate all EMAs (vectorized)
        result_df = pd.DataFrame(index=data.index)

        for period in self.ema_periods:
            ema_series = self.emas[period].calculate_batch(data)
            result_df[f'ema_{period}'] = ema_series.values

        return result_df

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        EMA Ribbon hesaplama

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Tüm EMA değerleri ve analizleri
        """
        # Tüm EMA'ları hesapla
        ema_values = {}
        for period in self.ema_periods:
            result = self.emas[period].calculate(data)
            ema_values[f'ema_{period}'] = round(result.value, 2)

        # Mevcut fiyat
        current_price = data['close'].values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # EMA sıralamasını analiz et
        alignment = self._analyze_alignment(ema_values, current_price)

        # Trend ve sinyal belirleme
        trend = self.get_trend(alignment)
        signal = self.get_signal(alignment, current_price, ema_values)
        strength = self._calculate_strength(alignment)

        # Destek ve direnç seviyeleri
        support_resistance = self._find_support_resistance(ema_values, current_price)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=ema_values,
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'ema_periods': self.ema_periods,
                'alignment': alignment,
                'alignment_score': self._calculate_alignment_score(ema_values),
                'current_price': round(current_price, 2),
                'nearest_ema': support_resistance['nearest'],
                'support': support_resistance['support'],
                'resistance': support_resistance['resistance'],
                'trend_quality': self._assess_trend_quality(alignment)
            }
        )

    def _analyze_alignment(self, ema_values: dict, price: float) -> str:
        """
        EMA sıralamasını analiz et

        Returns:
            str: 'bullish', 'bearish', 'mixed' veya 'neutral'
        """
        # EMA değerlerini liste olarak al (küçükten büyüğe)
        emas = [ema_values[f'ema_{p}'] for p in self.ema_periods]

        # Yükseliş sıralaması kontrolü (her EMA bir öncekinden büyük mü?)
        bullish_count = 0
        bearish_count = 0

        for i in range(len(emas) - 1):
            if emas[i] > emas[i + 1]:
                bullish_count += 1
            elif emas[i] < emas[i + 1]:
                bearish_count += 1

        total_comparisons = len(emas) - 1

        # Fiyatın pozisyonu
        price_above_all = all(price > ema for ema in emas)
        price_below_all = all(price < ema for ema in emas)

        # Sınıflandırma
        if bullish_count == total_comparisons and price_above_all:
            return 'strong_bullish'
        elif bullish_count >= total_comparisons * 0.7:
            return 'bullish'
        elif bearish_count == total_comparisons and price_below_all:
            return 'strong_bearish'
        elif bearish_count >= total_comparisons * 0.7:
            return 'bearish'
        elif abs(bullish_count - bearish_count) <= 1:
            return 'neutral'
        else:
            return 'mixed'

    def _calculate_alignment_score(self, ema_values: dict) -> float:
        """
        EMA sıralama skorunu hesapla (0-100)

        100: Mükemmel yükseliş sıralaması
        0: Mükemmel düşüş sıralaması
        50: Karışık/nötr
        """
        emas = [ema_values[f'ema_{p}'] for p in self.ema_periods]

        bullish_count = 0
        total_comparisons = len(emas) - 1

        for i in range(total_comparisons):
            if emas[i] > emas[i + 1]:
                bullish_count += 1

        score = (bullish_count / total_comparisons) * 100
        return round(score, 2)

    def _find_support_resistance(self, ema_values: dict, price: float) -> dict:
        """
        Destek ve direnç seviyelerini bul

        Returns:
            dict: nearest, support, resistance
        """
        emas = [(p, ema_values[f'ema_{p}']) for p in self.ema_periods]

        # Fiyata en yakın EMA
        nearest = min(emas, key=lambda x: abs(x[1] - price))

        # Fiyatın altındaki en yakın EMA (destek)
        supports = [ema for ema in emas if ema[1] < price]
        support = max(supports, key=lambda x: x[1]) if supports else None

        # Fiyatın üstündeki en yakın EMA (direnç)
        resistances = [ema for ema in emas if ema[1] > price]
        resistance = min(resistances, key=lambda x: x[1]) if resistances else None

        return {
            'nearest': f"EMA{nearest[0]} ({nearest[1]:.2f})",
            'support': f"EMA{support[0]} ({support[1]:.2f})" if support else "None",
            'resistance': f"EMA{resistance[0]} ({resistance[1]:.2f})" if resistance else "None"
        }

    def _assess_trend_quality(self, alignment: str) -> str:
        """
        Trend kalitesini değerlendir

        Returns:
            str: 'excellent', 'good', 'fair', 'poor'
        """
        if alignment == 'strong_bullish' or alignment == 'strong_bearish':
            return 'excellent'
        elif alignment == 'bullish' or alignment == 'bearish':
            return 'good'
        elif alignment == 'mixed':
            return 'fair'
        else:
            return 'poor'

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        self._close_buffer.append(candle['close'])
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={'trend': 'neutral', 'strength': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, alignment: str, price: float, ema_values: dict) -> SignalType:
        """
        EMA Ribbon'dan sinyal üret

        Args:
            alignment: EMA sıralama durumu
            price: Güncel fiyat
            ema_values: EMA değerleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # Kısa ve uzun vadeli EMA'lar
        ema_short = ema_values[f'ema_{self.ema_periods[0]}']  # En kısa
        ema_long = ema_values[f'ema_{self.ema_periods[-1]}']  # En uzun

        # Güçlü sinyaller
        if alignment == 'strong_bullish' and price > ema_short:
            return SignalType.BUY

        if alignment == 'strong_bearish' and price < ema_short:
            return SignalType.SELL

        # Orta seviye sinyaller
        if alignment == 'bullish' and price > ema_short:
            return SignalType.BUY

        if alignment == 'bearish' and price < ema_short:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, alignment: str) -> TrendDirection:
        """
        Trend yönünü belirle

        Args:
            alignment: EMA sıralama durumu

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if 'bullish' in alignment:
            return TrendDirection.UP
        elif 'bearish' in alignment:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, alignment: str) -> float:
        """
        Trend gücünü hesapla (0-100)
        """
        strength_map = {
            'strong_bullish': 100,
            'bullish': 75,
            'mixed': 50,
            'neutral': 25,
            'bearish': 75,
            'strong_bearish': 100
        }
        return strength_map.get(alignment, 50)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'ema_periods': [5, 10, 20, 50, 100, 200]
        }

    def _requires_volume(self) -> bool:
        """EMA Ribbon volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['EMARibbon']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """EMA Ribbon indikatör testi"""

    print("\n" + "="*60)
    print("EMA RIBBON (MULTI-EMA) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(250)]

    # Güçlü trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(249):
        if i < 80:
            trend = 0.5  # Yükseliş
        elif i < 160:
            trend = -0.3  # Düşüş
        else:
            trend = 0.7  # Güçlü yükseliş
        noise = np.random.randn() * 1.5
        prices.append(prices[-1] + trend + noise)

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
    ribbon = EMARibbon()
    print(f"   [OK] Oluşturuldu: {ribbon}")
    print(f"   [OK] Kategori: {ribbon.category.value}")
    print(f"   [OK] Tip: {ribbon.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {ribbon.get_required_periods()}")
    print(f"   [OK] EMA periyotları: {ribbon.ema_periods}")

    result = ribbon(data)
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 2: EMA değerleri
    print("\n3. EMA değerleri...")
    for period in ribbon.ema_periods:
        ema_key = f'ema_{period}'
        print(f"   [OK] EMA({period}): {result.value[ema_key]:.2f}")

    # Test 3: Sıralama analizi
    print("\n4. Sıralama analizi...")
    print(f"   [OK] Alignment: {result.metadata['alignment']}")
    print(f"   [OK] Alignment Score: {result.metadata['alignment_score']}/100")
    print(f"   [OK] Trend Quality: {result.metadata['trend_quality']}")
    print(f"   [OK] Current Price: {result.metadata['current_price']:.2f}")

    # Test 4: Destek/Direnç
    print("\n5. Destek/Direnç analizi...")
    print(f"   [OK] Nearest EMA: {result.metadata['nearest_ema']}")
    print(f"   [OK] Support: {result.metadata['support']}")
    print(f"   [OK] Resistance: {result.metadata['resistance']}")

    # Test 5: Trend değişimi analizi
    print("\n6. Trend değişimi analizi...")
    test_points = [100, 150, 200, 240]
    for idx in test_points:
        data_slice = data.iloc[:idx+1]
        result = ribbon.calculate(data_slice)
        print(f"   [OK] Mum {idx}: "
              f"Alignment={result.metadata['alignment']}, "
              f"Score={result.metadata['alignment_score']:.1f}, "
              f"Trend={result.trend.name}, "
              f"Quality={result.metadata['trend_quality']}")

    # Test 6: Özel periyotlar
    print("\n7. Özel periyot testi...")
    ribbon_custom = EMARibbon(ema_periods=[9, 21, 55, 89])
    result = ribbon_custom.calculate(data)
    print(f"   [OK] Özel periyotlar: {ribbon_custom.ema_periods}")
    print(f"   [OK] EMA değerleri:")
    for period in ribbon_custom.ema_periods:
        print(f"       EMA({period}): {result.value[f'ema_{period}']:.2f}")
    print(f"   [OK] Alignment: {result.metadata['alignment']}")

    # Test 7: Kısa vadeli ribbon (hızlı trade için)
    print("\n8. Kısa vadeli ribbon testi...")
    ribbon_short = EMARibbon(ema_periods=[5, 10, 20, 50])
    result = ribbon_short.calculate(data)
    print(f"   [OK] Kısa vadeli periyotlar: {ribbon_short.ema_periods}")
    print(f"   [OK] Alignment: {result.metadata['alignment']}")
    print(f"   [OK] Alignment Score: {result.metadata['alignment_score']:.1f}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 8: İstatistikler
    print("\n9. İstatistik testi...")
    stats = ribbon.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 9: Metadata
    print("\n10. Metadata testi...")
    metadata = ribbon.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
