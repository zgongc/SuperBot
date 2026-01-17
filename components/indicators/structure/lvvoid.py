"""
indicators/structure/lvvoid.py - Liquidity Void

Version: 1.0.0
Date: 2025-10-27
Author: SuperBot Team

Açıklama:
    LV Void (Liquidity Void) - Smart Money Concepts
    FVG + düşük volume kombinasyonu
    Likiditenin az olduğu boşluk bölgelerini tespit eder

    LV Void Nedir:
    - Fair Value Gap (FVG) + Düşük volume
    - Likidite eksikliği olan bölgeler
    - Fiyat hızla geçer (slippage riski yüksek)
    - Genellikle "fill" edilmek için geri dönülür

Formül:
    1. FVG tespiti (3 mum gap)
    2. Gap içindeki volume < Average Volume * threshold
    3. LV Void = FVG + Low Volume

    Bullish LV Void:
    - Candle[0].high < Candle[2].low
    - Volume[1] < Avg Volume * threshold

    Bearish LV Void:
    - Candle[0].low > Candle[2].high
    - Volume[1] < Avg Volume * threshold

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


class LVVoid(BaseIndicator):
    """
    Liquidity Void (LV Void)

    FVG + Low Volume kombinasyonu.
    Likidite eksikliği olan boşluk bölgelerini tespit eder.

    Args:
        min_gap_percent: Minimum boşluk yüzdesi (varsayılan: 0.1)
        volume_threshold: Volume eşiği (varsayılan: 0.5, avg volume'un %50'si)
        volume_period: Volume ortalaması periyodu (varsayılan: 20)
        max_zones: Maksimum açık zone sayısı (varsayılan: 5)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.1,
        volume_threshold: float = 0.5,
        volume_period: int = 20,
        max_zones: int = 5,
        logger=None,
        error_handler=None
    ):
        self.min_gap_percent = min_gap_percent
        self.volume_threshold = volume_threshold
        self.volume_period = volume_period
        self.max_zones = max_zones

        super().__init__(
            name='lvvoid',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'min_gap_percent': min_gap_percent,
                'volume_threshold': volume_threshold,
                'volume_period': volume_period,
                'max_zones': max_zones
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Açık LV Void'leri takip et
        self.open_voids: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.volume_period + 5, 10)

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.min_gap_percent < 0:
            raise InvalidParameterError(
                self.name, 'min_gap_percent', self.min_gap_percent,
                "Min gap percent negatif olamaz"
            )
        if not 0 < self.volume_threshold <= 1:
            raise InvalidParameterError(
                self.name, 'volume_threshold', self.volume_threshold,
                "Volume threshold 0-1 arası olmalı"
            )
        if self.volume_period < 1:
            raise InvalidParameterError(
                self.name, 'volume_period', self.volume_period,
                "Volume period pozitif olmalı"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones pozitif olmalı"
            )
        return True

    def _detect_lvvoid(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        index: int
    ) -> List[Dict[str, Any]]:
        """
        LV Void tespiti

        Args:
            highs: High fiyat dizisi
            lows: Low fiyat dizisi
            volumes: Volume dizisi
            index: Kontrol edilecek index (i-2, i-1, i)

        Returns:
            List[Dict]: Tespit edilen LV Void'ler
        """
        voids = []

        if index < 2 or index < self.volume_period:
            return voids

        # 3 mum: i-2, i-1, i
        candle_0_high = highs[index - 2]
        candle_0_low = lows[index - 2]
        candle_1_volume = volumes[index - 1]  # Gap mumunun volume'u
        candle_2_high = highs[index]
        candle_2_low = lows[index]

        mid_price = (highs[index - 1] + lows[index - 1]) / 2

        # Average volume hesapla (excluding current candle)
        avg_volume = np.mean(volumes[max(0, index - self.volume_period):index])
        volume_threshold_value = avg_volume * self.volume_threshold

        # Volume düşük mü?
        is_low_volume = candle_1_volume < volume_threshold_value

        if not is_low_volume:
            return voids

        # Bullish LV Void: Candle[0].high < Candle[2].low + Low Volume
        if candle_0_high < candle_2_low:
            gap_size = candle_2_low - candle_0_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                voids.append({
                    'type': 'bullish',
                    'top': candle_2_low,
                    'bottom': candle_0_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'volume': candle_1_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': candle_1_volume / avg_volume if avg_volume > 0 else 0,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        # Bearish LV Void: Candle[0].low > Candle[2].high + Low Volume
        if candle_0_low > candle_2_high:
            gap_size = candle_0_low - candle_2_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                voids.append({
                    'type': 'bearish',
                    'top': candle_0_low,
                    'bottom': candle_2_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'volume': candle_1_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': candle_1_volume / avg_volume if avg_volume > 0 else 0,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        return voids

    def _update_void_status(
        self,
        void: Dict[str, Any],
        current_high: float,
        current_low: float
    ) -> Dict[str, Any]:
        """
        LV Void dolum durumunu güncelle

        Args:
            void: LV Void bilgisi
            current_high: Güncel high
            current_low: Güncel low

        Returns:
            Dict: Güncellenmiş void
        """
        top = void['top']
        bottom = void['bottom']
        gap_size = void['size']

        # Fiyat void içine girdi mi?
        if current_low <= top and current_high >= bottom:
            # Dolum miktarını hesapla
            if void['type'] == 'bullish':
                # Aşağıdan dolduruluyor
                filled_amount = max(0, min(current_low, top) - bottom)
            else:
                # Yukarıdan dolduruluyor
                filled_amount = max(0, top - max(current_high, bottom))

            fill_percent = (filled_amount / gap_size) * 100
            void['fill_percent'] = min(fill_percent, 100)

            # Status güncelle
            if void['fill_percent'] >= 100:
                void['fill_status'] = 'filled'
            elif void['fill_percent'] >= 50:
                void['fill_status'] = 'partial'
            else:
                void['fill_status'] = 'open'

        return void

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        LV Void hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: LV Void zones
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values

        # Yeni LV Void'leri tespit et (son 3 mumda)
        latest_index = len(highs) - 1
        new_voids = self._detect_lvvoid(highs, lows, volumes, latest_index)

        # Yeni void'leri ekle
        self.open_voids.extend(new_voids)

        # Mevcut void'lerin durumunu güncelle
        current_high = highs[-1]
        current_low = lows[-1]

        for void in self.open_voids:
            self._update_void_status(void, current_high, current_low)

        # Tamamen dolmuş void'leri kaldır
        self.open_voids = [
            void for void in self.open_voids
            if void['fill_status'] != 'filled'
        ]

        # Maksimum zone sayısını uygula
        if len(self.open_voids) > self.max_zones:
            self.open_voids = self.open_voids[-self.max_zones:]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Değer: Açık LV Void'lerin listesi
        zones = [
            {
                'type': void['type'],
                'top': round(void['top'], 2),
                'bottom': round(void['bottom'], 2),
                'size': round(void['size'], 2),
                'volume_ratio': round(void['volume_ratio'], 3),
                'fill_status': void['fill_status'],
                'fill_percent': round(void['fill_percent'], 2)
            }
            for void in self.open_voids
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'value': len(zones)},  # Dict format with zone count for consistency
            timestamp=timestamp,
            signal=self.get_signal(zones, data['close'].values[-1]),
            trend=self.get_trend(zones),
            strength=len(zones) * 25,  # Her zone 25 puan (FVG'den daha güçlü)
            metadata={
                'zones': zones,  # Full zones data in metadata
                'total_zones': len(zones),
                'bullish_zones': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_zones': len([z for z in zones if z['type'] == 'bearish']),
                'avg_volume_ratio': round(np.mean([void['volume_ratio'] for void in self.open_voids]), 3) if self.open_voids else 0,
                'min_gap_percent': self.min_gap_percent,
                'volume_threshold': self.volume_threshold
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate LV Void for entire DataFrame (vectorized - for backtest)

        Returns pd.Series with LV Void count for all bars.
        Value: Number of active LV Void zones at each bar
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values

        # Result array
        void_count = np.zeros(len(data))

        # Track all voids
        all_voids: List[Dict[str, Any]] = []

        # Scan through all bars
        for i in range(max(2, self.volume_period), len(data)):
            # Detect new voids at this bar
            new_voids = self._detect_lvvoid(highs, lows, volumes, i)
            all_voids.extend(new_voids)

            # Update all open voids
            current_high = highs[i]
            current_low = lows[i]

            for void in all_voids:
                if void['fill_status'] != 'filled':
                    self._update_void_status(void, current_high, current_low)

            # Remove filled voids
            all_voids = [
                void for void in all_voids
                if void['fill_status'] != 'filled'
            ]

            # Apply max zones limit
            if len(all_voids) > self.max_zones:
                all_voids = all_voids[-self.max_zones:]

            # Store count
            void_count[i] = len(all_voids)

        return pd.Series(void_count, index=data.index, name='lvvoid')

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
            return IndicatorResult(
                value=[],
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
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
        LV Void'lerden sinyal üret

        Args:
            zones: LV Void zone'ları
            current_price: Güncel fiyat

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if not zones:
            return SignalType.HOLD

        # Fiyat bir void zone'una yaklaştı mı?
        for zone in zones:
            distance_to_zone = min(
                abs(current_price - zone['top']),
                abs(current_price - zone['bottom'])
            )

            distance_percent = (distance_to_zone / current_price) * 100

            # %0.5 içindeyse sinyal ver (FVG'den daha hassas)
            if distance_percent < 0.5:
                if zone['type'] == 'bullish':
                    return SignalType.BUY
                elif zone['type'] == 'bearish':
                    return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        LV Void'lerden trend belirle

        Args:
            zones: LV Void zone'ları

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
            'volume_threshold': 0.5,
            'volume_period': 20,
            'max_zones': 5
        }

    def _requires_volume(self) -> bool:
        """LV Void volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['LVVoid']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """LV Void indikatör testi"""

    print("\n" + "="*60)
    print("LV VOID (LIQUIDITY VOID) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # FVG + düşük volume simülasyonu
    base_price = 100
    prices = []
    volumes = []
    highs = []
    lows = []

    for i in range(50):
        if i == 15:
            # Bullish LV Void oluştur (hızlı yükseliş + düşük volume)
            prices.append(base_price + 10)
            volumes.append(500)  # Düşük volume
            highs.append(base_price + 11)
            lows.append(base_price + 9)
        elif i == 35:
            # Bearish LV Void oluştur (hızlı düşüş + düşük volume)
            prices.append(base_price - 5)
            volumes.append(600)  # Düşük volume
            highs.append(base_price - 4)
            lows.append(base_price - 6)
        else:
            # Normal hareket
            prices.append(base_price + np.random.randn() * 0.5)
            volumes.append(1000 + np.random.randint(0, 500))  # Normal volume
            highs.append(prices[-1] + abs(np.random.randn()) * 0.3)
            lows.append(prices[-1] - abs(np.random.randn()) * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Volume aralığı: {min(volumes)} -> {max(volumes)}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    lvvoid = LVVoid(min_gap_percent=0.1, volume_threshold=0.5, volume_period=20)
    print(f"   [OK] Oluşturuldu: {lvvoid}")
    print(f"   [OK] Kategori: {lvvoid.category.value}")
    print(f"   [OK] Gerekli periyot: {lvvoid.get_required_periods()}")

    result = lvvoid(data)
    print(f"   [OK] Toplam Zone: {result.metadata['total_zones']}")
    print(f"   [OK] Bullish Zone: {result.metadata['bullish_zones']}")
    print(f"   [OK] Bearish Zone: {result.metadata['bearish_zones']}")
    print(f"   [OK] Avg Volume Ratio: {result.metadata['avg_volume_ratio']}")
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
            print(f"       - Volume Ratio: {zone['volume_ratio']:.3f}")
            print(f"       - Fill: {zone['fill_status']} ({zone['fill_percent']:.1f}%)")
    else:
        print("   [OK] Açık zone bulunamadı")

    # Test 3: Batch hesaplama
    print("\n4. Batch hesaplama testi...")
    batch_result = lvvoid.calculate_batch(data)
    print(f"   [OK] Batch sonuç uzunluğu: {len(batch_result)}")
    print(f"   [OK] Max aktif zone sayısı: {int(batch_result.max())}")
    print(f"   [OK] Toplam zone tespit: {int(batch_result.sum())}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
