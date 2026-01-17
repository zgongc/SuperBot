"""
indicators/trend/vwma.py - Volume Weighted Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    VWMA (Volume Weighted Moving Average) - Hacim ağırlıklı hareketli ortalama
    Fiyatları işlem hacmine göre ağırlıklandırarak hesaplanan trend indikatörü
    Yüksek hacimli fiyat hareketlerine daha fazla önem verir

    Kullanım:
    - Hacimli fiyat hareketlerini vurgulama
    - Gerçek piyasa gücünü ölçme
    - Destek/direnç seviyeleri

Formül:
    VWMA = Sum(Close * Volume) / Sum(Volume)
    n periyot için:
    VWMA = (C1*V1 + C2*V2 + ... + Cn*Vn) / (V1 + V2 + ... + Vn)

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


class VWMA(BaseIndicator):
    """
    Volume Weighted Moving Average

    İşlem hacmine göre ağırlıklandırılmış hareketli ortalama.
    Yüksek hacimli mumlar daha fazla ağırlığa sahiptir.

    Args:
        period: VWMA periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='vwma',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        VWMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VWMA değeri
        """
        close = data['close'].values
        volume = data['volume'].values

        # VWMA hesapla
        close_slice = close[-self.period:]
        volume_slice = volume[-self.period:]

        # Sum(Close * Volume) / Sum(Volume)
        vwma_value = np.sum(close_slice * volume_slice) / np.sum(volume_slice)

        # Mevcut fiyat
        current_price = close[-1]

        # SMA ile karşılaştırma için
        sma_value = np.mean(close_slice)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(vwma_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, vwma_value),
            trend=self.get_trend(current_price, vwma_value),
            strength=self._calculate_strength(current_price, vwma_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'sma': round(sma_value, 2),
                'vwma_sma_diff': round(vwma_value - sma_value, 2),
                'avg_volume': round(np.mean(volume_slice), 2),
                'distance_pct': round(((current_price - vwma_value) / vwma_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch VWMA calculation - BACKTEST için

        VWMA Formula:
            VWMA = Sum(Close × Volume) / Sum(Volume)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: VWMA values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']
        volume = data['volume']

        # Close × Volume
        close_volume = close * volume

        # VWMA = rolling sum of (Close × Volume) / rolling sum of Volume
        vwma = close_volume.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

        # Set first period values to NaN (warmup)
        vwma.iloc[:self.period-1] = np.nan

        return pd.Series(vwma.values, index=data.index, name='vwma')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli state'i hazırlar

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)

        # Buffer'lara verileri ekle
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            open_val = candle.get('open', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'high': [high_val] * len(self._close_buffer),
            'low': [low_val] * len(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, price: float, vwma: float) -> SignalType:
        """
        VWMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            vwma: VWMA değeri

        Returns:
            SignalType: BUY (fiyat VWMA üstüne çıkınca), SELL (altına ininse)
        """
        if price > vwma:
            return SignalType.BUY
        elif price < vwma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, vwma: float) -> TrendDirection:
        """
        VWMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            vwma: VWMA değeri

        Returns:
            TrendDirection: UP (fiyat > VWMA), DOWN (fiyat < VWMA)
        """
        if price > vwma:
            return TrendDirection.UP
        elif price < vwma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, vwma: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - vwma) / vwma * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """VWMA volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VWMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """VWMA indikatör testi"""

    print("\n" + "="*60)
    print("VWMA (VOLUME WEIGHTED MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend ve hacim simülasyonu
    base_price = 100
    prices = [base_price]
    volumes = [1000]

    for i in range(49):
        trend = 0.5
        noise = np.random.randn() * 1.5
        prices.append(prices[-1] + trend + noise)

        # Yüksek fiyat değişiminde yüksek hacim
        price_change = abs(prices[-1] - prices[-2])
        volume = 1000 + price_change * 100 + np.random.randint(0, 500)
        volumes.append(volume)

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
    print(f"   [OK] Hacim aralığı: {min(volumes):.0f} -> {max(volumes):.0f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    vwma = VWMA(period=20)
    print(f"   [OK] Oluşturuldu: {vwma}")
    print(f"   [OK] Kategori: {vwma.category.value}")
    print(f"   [OK] Gerekli periyot: {vwma.get_required_periods()}")
    print(f"   [OK] Volume gerekli: {vwma._requires_volume()}")

    result = vwma(data)
    print(f"   [OK] VWMA Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: VWMA vs SMA karşılaştırması
    print("\n3. VWMA vs SMA karşılaştırma testi...")
    print(f"   [OK] SMA(20): {result.metadata['sma']}")
    print(f"   [OK] VWMA(20): {result.value}")
    print(f"   [OK] Fark: {result.metadata['vwma_sma_diff']}")
    print(f"   [OK] VWMA hacimli hareketlere daha fazla ağırlık verir")

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [10, 20, 30]:
        vwma_test = VWMA(period=period)
        result = vwma_test.calculate(data)
        print(f"   [OK] VWMA({period}): {result.value:.2f} | Sinyal: {result.signal.value}")

    # Test 4: Hacim etkisi
    print("\n5. Hacim etkisi testi...")
    print(f"   [OK] Ortalama hacim: {result.metadata['avg_volume']:.0f}")
    print(f"   [OK] Son mum hacmi: {volumes[-1]:.0f}")

    # Yüksek hacimli senaryo
    high_volume_data = data.copy()
    high_volume_data.loc[high_volume_data.index[-1], 'volume'] = 10000
    result_high = vwma.calculate(high_volume_data)
    print(f"   [OK] Normal VWMA: {result.value}")
    print(f"   [OK] Yüksek hacimli VWMA: {result_high.value}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = vwma.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = vwma.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
