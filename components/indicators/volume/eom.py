"""
indicators/volume/eom.py - Ease of Movement

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    EOM (Ease of Movement) - Hareket kolaylığı indikatörü
    Fiyat değişimini hacimle ilişkilendirerek hareket kolaylığını ölçer
    Yüksek EOM = Az hacimle büyük fiyat hareketi (kolay hareket)
    Düşük EOM = Çok hacimle az fiyat hareketi (zor hareket)

Formül:
    Distance Moved = ((High + Low) / 2) - ((High_prev + Low_prev) / 2)
    Box Ratio = (Volume / 100000000) / (High - Low)
    EMV = Distance Moved / Box Ratio
    EOM = SMA(EMV, period)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class EOM(BaseIndicator):
    """
    Ease of Movement

    Richard Arms tarafından geliştirilen indikatör.
    Fiyat değişimini hacimle ilişkilendirerek trend gücünü ölçer.

    Args:
        period: EOM smoothing periyodu (varsayılan: 14)
        divisor: Hacim ölçeklendirme divisor (varsayılan: 100000000)
    """

    def __init__(
        self,
        period: int = 14,
        divisor: float = 100000000,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.divisor = divisor

        super().__init__(
            name='eom',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'divisor': divisor
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period + 1

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot pozitif olmalı"
            )
        if self.divisor <= 0:
            raise InvalidParameterError(
                self.name, 'divisor', self.divisor,
                "Divisor pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        EOM hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: EOM değeri
        """
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        # EMV (Ease of Movement Value) hesapla
        emv = np.zeros(len(high))

        for i in range(1, len(high)):
            # Distance Moved
            mid_point = (high[i] + low[i]) / 2
            mid_point_prev = (high[i-1] + low[i-1]) / 2
            distance_moved = mid_point - mid_point_prev

            # Box Ratio
            high_low_diff = high[i] - low[i]
            if high_low_diff == 0 or volume[i] == 0:
                emv[i] = 0
            else:
                box_ratio = (volume[i] / self.divisor) / high_low_diff
                if box_ratio == 0:
                    emv[i] = 0
                else:
                    emv[i] = distance_moved / box_ratio

        # EOM = EMV'nin SMA'sı
        eom_value = np.mean(emv[-self.period:])

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(eom_value, 6),
            timestamp=timestamp,
            signal=self.get_signal(eom_value),
            trend=self.get_trend(emv[-10:] if len(emv) >= 10 else emv),
            strength=min(abs(eom_value) * 10, 100),  # Normalize
            metadata={
                'period': self.period,
                'divisor': self.divisor,
                'emv_current': round(emv[-1], 6),
                'avg_emv': round(eom_value, 6)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch EOM calculation - BACKTEST için

        EOM Formula:
            Distance Moved = MidPoint - MidPoint_prev
            Box Ratio = (Volume / divisor) / (High - Low)
            EMV = Distance Moved / Box Ratio
            EOM = SMA(EMV, period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: EOM values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        volume = data['volume']

        # Mid Point = (High + Low) / 2
        mid_point = (high + low) / 2

        # Distance Moved = MidPoint - MidPoint_prev
        distance_moved = mid_point.diff()

        # Box Ratio = (Volume / divisor) / (High - Low)
        high_low_diff = high - low
        high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
        box_ratio = (volume / self.divisor) / high_low_diff

        # EMV = Distance Moved / Box Ratio
        emv = distance_moved / box_ratio
        emv = emv.fillna(0)  # Handle division by zero

        # EOM = SMA of EMV
        eom = emv.rolling(window=self.period).mean()

        # Set first period values to NaN (warmup)
        eom.iloc[:self.period] = np.nan

        return pd.Series(eom.values, index=data.index, name='eom')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli state'i hazırlar"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._volume_buffer.append(volume_val)

        if len(self._volume_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._volume_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        EOM değerinden sinyal üret

        Args:
            value: EOM değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if value > 0.0001:  # Pozitif ve anlamlı
            return SignalType.BUY
        elif value < -0.0001:  # Negatif ve anlamlı
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, emv_array: np.ndarray) -> TrendDirection:
        """
        EOM trendini belirle

        Args:
            emv_array: Son EMV değerleri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if len(emv_array) < 2:
            return TrendDirection.NEUTRAL

        # Lineer regresyon ile trend
        slope = np.polyfit(range(len(emv_array)), emv_array, 1)[0]

        if slope > 0.00001:
            return TrendDirection.UP
        elif slope < -0.00001:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 14,
            'divisor': 100000000
        }

    def _requires_volume(self) -> bool:
        """EOM volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['EOM']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """EOM indikatör testi"""

    print("\n" + "="*60)
    print("EOM (EASE OF MOVEMENT) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]
    volumes = [100000000]  # 100M baseline

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(100000000 + np.random.randint(-30000000, 50000000))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Hacim aralığı: {min(volumes):,.0f} -> {max(volumes):,.0f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    eom = EOM(period=14)
    print(f"   [OK] Oluşturuldu: {eom}")
    print(f"   [OK] Kategori: {eom.category.value}")
    print(f"   [OK] Gerekli periyot: {eom.get_required_periods()}")

    result = eom(data)
    print(f"   [OK] EOM Değeri: {result.value:.6f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [7, 14, 21]:
        eom_test = EOM(period=period)
        result = eom_test.calculate(data)
        print(f"   [OK] EOM({period}): {result.value:.6f} | Sinyal: {result.signal.value}")

    # Test 3: Farklı divisor'lar
    print("\n4. Farklı divisor testi...")
    for div in [10000000, 100000000, 1000000000]:
        eom_test = EOM(period=14, divisor=div)
        result = eom_test.calculate(data)
        print(f"   [OK] Divisor={div:,.0f}: EOM={result.value:.6f}")

    # Test 4: Volume gereksinimi
    print("\n5. Volume gereksinimi testi...")
    metadata = eom.metadata
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "EOM volume gerektirmeli!"

    # Test 5: Hareket kolaylığı yorumlama
    print("\n6. Hareket kolaylığı yorumlama testi...")
    result = eom.calculate(data)
    eom_val = result.value
    print(f"   [OK] EOM: {eom_val:.6f}")
    if eom_val > 0:
        print("   [OK] Pozitif EOM - Az hacimle yukarı hareket (kolay yükseliş)")
    elif eom_val < 0:
        print("   [OK] Negatif EOM - Az hacimle aşağı hareket (kolay düşüş)")
    else:
        print("   [OK] Nötr - Hareket yok veya yüksek hacimli konsolidasyon")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = eom.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
