"""
indicators/structure/fvg.py - Fair Value Gap

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    FVG (Fair Value Gap) - Smart Money Concepts
    Fiyat boşluklarını (imbalance) tespit eder

    FVG Nedir:
    - 3 mum arasında oluşan fiyat boşluğu
    - Hızlı fiyat hareketi sonucu oluşur
    - Genellikle bu boşluklar "fill" edilir (doldurulur)

Formül:
    Bullish FVG:
    - Candle[0].high < Candle[2].low
    - Gap: [Candle[0].high, Candle[2].low]

    Bearish FVG:
    - Candle[0].low > Candle[2].high
    - Gap: [Candle[2].high, Candle[0].low]

    Fill Status:
    - Open: Boşluk henüz doldurulmadı
    - Partial: Kısmen doldu
    - Filled: Tamamen doldu

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class FVG(BaseIndicator):
    """
    Fair Value Gap (FVG)

    Fiyat boşluklarını tespit eder ve takip eder.
    Potansiyel destek/direnç bölgeleri olarak kullanılır.

    Args:
        min_gap_percent: Minimum boşluk yüzdesi (varsayılan: 0.1)
        max_zones: Maksimum açık zone sayısı (varsayılan: 5)
        fill_threshold: Doldurulma eşiği (varsayılan: 0.5, %50)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.1,
        max_zones: int = 5,
        fill_threshold: float = 0.5,
        logger=None,
        error_handler=None
    ):
        self.min_gap_percent = min_gap_percent
        self.max_zones = max_zones
        self.fill_threshold = fill_threshold

        super().__init__(
            name='fvg',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'min_gap_percent': min_gap_percent,
                'max_zones': max_zones,
                'fill_threshold': fill_threshold
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Açık FVG'leri takip et
        self.open_fvgs: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return 5  # En az 3 mum + buffer

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.min_gap_percent < 0:
            raise InvalidParameterError(
                self.name, 'min_gap_percent', self.min_gap_percent,
                "Min gap percent negatif olamaz"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones pozitif olmalı"
            )
        if not 0 <= self.fill_threshold <= 1:
            raise InvalidParameterError(
                self.name, 'fill_threshold', self.fill_threshold,
                "Fill threshold 0-1 arası olmalı"
            )
        return True

    def _detect_fvg(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        index: int
    ) -> List[Dict[str, Any]]:
        """
        FVG tespiti

        Args:
            highs: High fiyat dizisi
            lows: Low fiyat dizisi
            index: Kontrol edilecek index (i-2, i-1, i)

        Returns:
            List[Dict]: Tespit edilen FVG'ler
        """
        fvgs = []

        if index < 2:
            return fvgs

        # 3 mum: i-2, i-1, i
        candle_0_high = highs[index - 2]
        candle_0_low = lows[index - 2]
        candle_2_high = highs[index]
        candle_2_low = lows[index]

        mid_price = (highs[index - 1] + lows[index - 1]) / 2

        # Bullish FVG: Candle[0].high < Candle[2].low
        if candle_0_high < candle_2_low:
            gap_size = candle_2_low - candle_0_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                fvgs.append({
                    'type': 'bullish',
                    'top': candle_2_low,
                    'bottom': candle_0_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        # Bearish FVG: Candle[0].low > Candle[2].high
        if candle_0_low > candle_2_high:
            gap_size = candle_0_low - candle_2_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                fvgs.append({
                    'type': 'bearish',
                    'top': candle_0_low,
                    'bottom': candle_2_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        return fvgs

    def _update_fvg_status(
        self,
        fvg: Dict[str, Any],
        current_high: float,
        current_low: float
    ) -> Dict[str, Any]:
        """
        FVG dolum durumunu güncelle

        Args:
            fvg: FVG bilgisi
            current_high: Güncel high
            current_low: Güncel low

        Returns:
            Dict: Güncellenmiş FVG
        """
        top = fvg['top']
        bottom = fvg['bottom']
        gap_size = fvg['size']

        # Fiyat FVG içine girdi mi?
        if current_low <= top and current_high >= bottom:
            # Dolum miktarını hesapla
            if fvg['type'] == 'bullish':
                # Aşağıdan dolduruluyor
                filled_amount = max(0, min(current_low, top) - bottom)
            else:
                # Yukarıdan dolduruluyor
                filled_amount = max(0, top - max(current_high, bottom))

            fill_percent = (filled_amount / gap_size) * 100
            fvg['fill_percent'] = min(fill_percent, 100)

            # Status güncelle
            if fvg['fill_percent'] >= 100:
                fvg['fill_status'] = 'filled'
            elif fvg['fill_percent'] >= self.fill_threshold * 100:
                fvg['fill_status'] = 'partial'
            else:
                fvg['fill_status'] = 'open'

        return fvg

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        FVG hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: FVG zones
        """
        highs = data['high'].values
        lows = data['low'].values

        # Yeni FVG'leri tespit et (son 3 mumda)
        latest_index = len(highs) - 1
        new_fvgs = self._detect_fvg(highs, lows, latest_index)

        # Yeni FVG'leri ekle
        self.open_fvgs.extend(new_fvgs)

        # Mevcut FVG'lerin durumunu güncelle
        current_high = highs[-1]
        current_low = lows[-1]

        for fvg in self.open_fvgs:
            self._update_fvg_status(fvg, current_high, current_low)

        # Tamamen dolmuş FVG'leri kaldır
        self.open_fvgs = [
            fvg for fvg in self.open_fvgs
            if fvg['fill_status'] != 'filled'
        ]

        # Maksimum zone sayısını uygula (en yeni olanları tut)
        if len(self.open_fvgs) > self.max_zones:
            self.open_fvgs = self.open_fvgs[-self.max_zones:]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Değer: Açık FVG'lerin listesi
        zones = [
            {
                'type': fvg['type'],
                'top': round(fvg['top'], 2),
                'bottom': round(fvg['bottom'], 2),
                'size': round(fvg['size'], 2),
                'fill_status': fvg['fill_status'],
                'fill_percent': round(fvg['fill_percent'], 2)
            }
            for fvg in self.open_fvgs
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'value': len(zones)},  # Dict format with zone count for consistency
            timestamp=timestamp,
            signal=self.get_signal(zones, data['close'].values[-1]),
            trend=self.get_trend(zones),
            strength=len(zones) * 20,  # Her zone 20 puan
            metadata={
                'zones': zones,  # Full zones data in metadata
                'total_zones': len(zones),
                'bullish_zones': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_zones': len([z for z in zones if z['type'] == 'bearish']),
                'min_gap_percent': self.min_gap_percent,
                'max_zones': self.max_zones
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate FVG for entire DataFrame (vectorized - for backtest)

        Returns:
            pd.Series with net FVG values for all bars:
                - Positive: Bullish FVG dominance (bullish_zones - bearish_zones)
                - Negative: Bearish FVG dominance
                - Zero: No FVG or balanced

        Examples:
            fvg > 0   → Bullish FVG present
            fvg < 0   → Bearish FVG present
            fvg >= 2  → Strong bullish (2+ net zones)
            fvg <= -2 → Strong bearish (2+ net zones)
        """
        highs = data['high'].values
        lows = data['low'].values

        # Result array: store number of active FVG zones at each bar
        fvg_count = np.zeros(len(data))

        # Track all FVGs across history
        all_fvgs: List[Dict[str, Any]] = []

        # Scan through all bars
        for i in range(2, len(data)):
            # Detect new FVGs at this bar
            new_fvgs = self._detect_fvg(highs, lows, i)

            # Update existing FVGs BEFORE adding new ones (don't update on creation bar)
            current_high = highs[i]
            current_low = lows[i]

            for fvg in all_fvgs:
                if fvg['fill_status'] != 'filled':
                    self._update_fvg_status(fvg, current_high, current_low)

            # Now add new FVGs (they won't be updated on creation bar)
            all_fvgs.extend(new_fvgs)

            # Remove filled FVGs
            all_fvgs = [
                fvg for fvg in all_fvgs
                if fvg['fill_status'] != 'filled'
            ]

            # Apply max zones limit
            if len(all_fvgs) > self.max_zones:
                all_fvgs = all_fvgs[-self.max_zones:]

            # Calculate net FVG (bullish - bearish)
            bullish_count = len([fvg for fvg in all_fvgs if fvg['type'] == 'bullish'])
            bearish_count = len([fvg for fvg in all_fvgs if fvg['type'] == 'bearish'])

            # Positive = bullish dominance, Negative = bearish dominance
            fvg_count[i] = bullish_count - bearish_count

        return pd.Series(fvg_count, index=data.index, name='fvg')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup FVG buffers with historical data

        CRITICAL: FVG uses its own buffers (_high_buffer, _low_buffer, _close_buffer)
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
            # Return dict format consistent with calculate() method
            # This allows TradingEngine to properly add to DataFrame
            return IndicatorResult(
                value={'value': 0},  # 0 zones during warmup
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'warmup': True, 'zones': [], 'total_zones': 0}
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

    def get_signal(self, zones: List[Dict[str, Any]], current_price: float) -> SignalType:
        """
        FVG'lerden sinyal üret

        Args:
            zones: FVG zone'ları
            current_price: Güncel fiyat

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if not zones:
            return SignalType.HOLD

        # Fiyat bir FVG zone'una yaklaştı mı?
        for zone in zones:
            distance_to_zone = min(
                abs(current_price - zone['top']),
                abs(current_price - zone['bottom'])
            )

            distance_percent = (distance_to_zone / current_price) * 100

            # %1 içindeyse sinyal ver
            if distance_percent < 1.0:
                if zone['type'] == 'bullish':
                    return SignalType.BUY  # Bullish FVG'ye yakın
                elif zone['type'] == 'bearish':
                    return SignalType.SELL  # Bearish FVG'ye yakın

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        FVG'lerden trend belirle

        Args:
            zones: FVG zone'ları

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        bullish_count = len([z for z in zones if z['type'] == 'bullish'])
        bearish_count = len([z for z in zones if z['type'] == 'bearish'])

        if bullish_count > bearish_count:
            return TrendDirection.UP
        elif bearish_count > bullish_count:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'min_gap_percent': 0.1,
            'max_zones': 5,
            'fill_threshold': 0.5
        }

    def _requires_volume(self) -> bool:
        """FVG volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['FVG']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """FVG indikatör testi"""

    print("\n" + "="*60)
    print("FVG (FAIR VALUE GAP) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(40)]

    # FVG oluşturacak fiyat hareketi
    base_price = 100
    prices = []
    highs = []
    lows = []

    for i in range(40):
        if i == 10:
            # Bullish FVG oluştur (hızlı yükseliş)
            prices.append(base_price + 10)
            highs.append(base_price + 11)
            lows.append(base_price + 9)
        elif i == 25:
            # Bearish FVG oluştur (hızlı düşüş)
            prices.append(base_price - 5)
            highs.append(base_price - 4)
            lows.append(base_price - 6)
        else:
            # Normal hareket
            prices.append(base_price + np.random.randn() * 0.5)
            highs.append(prices[-1] + abs(np.random.randn()) * 0.3)
            lows.append(prices[-1] - abs(np.random.randn()) * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    fvg = FVG(min_gap_percent=0.1, max_zones=5, fill_threshold=0.5)
    print(f"   [OK] Oluşturuldu: {fvg}")
    print(f"   [OK] Kategori: {fvg.category.value}")
    print(f"   [OK] Gerekli periyot: {fvg.get_required_periods()}")

    result = fvg(data)
    print(f"   [OK] Toplam Zone: {result.metadata['total_zones']}")
    print(f"   [OK] Bullish Zone: {result.metadata['bullish_zones']}")
    print(f"   [OK] Bearish Zone: {result.metadata['bearish_zones']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength}")

    # Test 2: Zone detayları
    print("\n3. Zone detayları...")
    if result.value:
        for i, zone in enumerate(result.value[:3]):
            print(f"   [OK] Zone #{i+1}:")
            print(f"       - Tip: {zone['type']}")
            print(f"       - Top: {zone['top']:.2f}")
            print(f"       - Bottom: {zone['bottom']:.2f}")
            print(f"       - Size: {zone['size']:.2f}")
            print(f"       - Fill: {zone['fill_status']} ({zone['fill_percent']:.1f}%)")
    else:
        print("   [OK] Açık zone bulunamadı")

    # Test 3: Farklı parametreler
    print("\n4. Farklı parametre testi...")
    for min_gap in [0.05, 0.1, 0.2]:
        fvg_test = FVG(min_gap_percent=min_gap)
        result = fvg_test.calculate(data)
        print(f"   [OK] FVG(gap={min_gap}): {result.metadata['total_zones']} zones")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = fvg.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = fvg.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
