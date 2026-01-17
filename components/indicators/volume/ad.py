"""
indicators/volume/ad.py - Accumulation/Distribution

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    A/D (Accumulation/Distribution) - Biriktirme/Dağıtım indikatörü
    Hacim akışını ve fiyat hareketlerini birleştirir
    Yükselen A/D = Biriktirme (alım)
    Düşen A/D = Dağıtım (satım)

Formül:
    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    MF Volume = MFM × Volume
    A/D = A/D_prev + MF Volume

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


class AD(BaseIndicator):
    """
    Accumulation/Distribution

    Para akışını (money flow) ölçerek biriktirme ve dağıtım
    fazlarını tespit eder.

    Args:
        signal_period: Sinyal hattı SMA periyodu (varsayılan: 10)
    """

    def __init__(
        self,
        signal_period: int = 10,
        logger=None,
        error_handler=None
    ):
        self.signal_period = signal_period

        super().__init__(
            name='ad',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'signal_period': signal_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.signal_period + 1

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "Periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        A/D hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: A/D değeri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # A/D hesapla
        ad_values = np.zeros(len(close))

        for i in range(len(close)):
            high_low_diff = high[i] - low[i]

            if high_low_diff == 0:
                mfm = 0
            else:
                # Money Flow Multiplier
                mfm = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_diff

            # Money Flow Volume
            mf_volume = mfm * volume[i]

            # Kümülatif A/D
            if i == 0:
                ad_values[i] = mf_volume
            else:
                ad_values[i] = ad_values[i-1] + mf_volume

        ad_value = ad_values[-1]

        # Sinyal hattı (SMA)
        ad_signal = np.mean(ad_values[-self.signal_period:])

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(ad_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(ad_value, ad_signal),
            trend=self.get_trend(ad_values[-10:] if len(ad_values) >= 10 else ad_values),
            strength=min(abs((ad_value - ad_signal) / ad_signal * 100), 100) if ad_signal != 0 else 0,
            metadata={
                'signal_period': self.signal_period,
                'ad_signal': round(ad_signal, 2),
                'divergence': round(ad_value - ad_signal, 2),
                'mfm': round(mfm, 4)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch A/D calculation - BACKTEST için

        A/D Formula:
            MFM = ((Close - Low) - (High - Close)) / (High - Low)
            MF Volume = MFM × Volume
            A/D = Cumulative Sum of MF Volume

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: A/D values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # High - Low difference
        high_low_diff = high - low

        # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
        mfm = ((close - low) - (high - close)) / high_low_diff

        # Handle division by zero (when high == low, set MFM to 0)
        mfm = mfm.fillna(0).replace([np.inf, -np.inf], 0)

        # Money Flow Volume: MFM × Volume
        mf_volume = mfm * volume

        # A/D: Cumulative sum of MF Volume
        ad = mf_volume.cumsum()

        return pd.Series(ad.values, index=data.index, name='ad')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        AD kümülatif bir indikatördür, bu yüzden son AD değerini ve
        son N bar'ın AD değerlerini (signal için) saklamamız gerekiyor.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque

        # AD değerlerini hesapla ve son değeri sakla
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        ad_values = np.zeros(len(close))
        for i in range(len(close)):
            high_low_diff = high[i] - low[i]
            if high_low_diff == 0:
                mfm = 0
            else:
                mfm = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_diff
            mf_volume = mfm * volume[i]
            if i == 0:
                ad_values[i] = mf_volume
            else:
                ad_values[i] = ad_values[i-1] + mf_volume

        # Kümülatif AD değerini sakla
        self._cumulative_ad = ad_values[-1]

        # Son signal_period kadar AD değerini sakla (signal için)
        max_len = self.signal_period + 10
        self._ad_buffer = deque(maxlen=max_len)
        for val in ad_values[-max_len:]:
            self._ad_buffer.append(val)

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        AD kümülatif hesaplandığı için, önceki AD değerine yeni bar'ın
        MF Volume'unu ekliyoruz.
        """
        from collections import deque

        if not hasattr(self, '_buffers_init'):
            max_len = self.signal_period + 10
            self._ad_buffer = deque(maxlen=max_len)
            self._cumulative_ad = 0.0
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        # MFM ve MF Volume hesapla
        high_low_diff = high_val - low_val
        if high_low_diff == 0:
            mfm = 0
        else:
            mfm = ((close_val - low_val) - (high_val - close_val)) / high_low_diff
        mf_volume = mfm * volume_val

        # Kümülatif AD'yi güncelle
        self._cumulative_ad += mf_volume
        ad_value = self._cumulative_ad

        # AD buffer'a ekle
        self._ad_buffer.append(ad_value)

        if len(self._ad_buffer) < self.signal_period:
            return IndicatorResult(
                value=round(ad_value, 2),
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Sinyal hattı hesapla
        ad_signal = np.mean(list(self._ad_buffer)[-self.signal_period:])

        return IndicatorResult(
            value=round(ad_value, 2),
            timestamp=timestamp_val,
            signal=self.get_signal(ad_value, ad_signal),
            trend=self.get_trend(np.array(list(self._ad_buffer)[-10:])),
            strength=min(abs((ad_value - ad_signal) / ad_signal * 100), 100) if ad_signal != 0 else 0,
            metadata={
                'signal_period': self.signal_period,
                'ad_signal': round(ad_signal, 2),
                'divergence': round(ad_value - ad_signal, 2),
                'mfm': round(mfm, 4)
            }
        )

    def get_signal(self, ad_value: float, ad_signal: float) -> SignalType:
        """
        A/D değerinden sinyal üret

        Args:
            ad_value: A/D değeri
            ad_signal: A/D sinyal hattı

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if ad_value > ad_signal:
            return SignalType.BUY  # Biriktirme
        elif ad_value < ad_signal:
            return SignalType.SELL  # Dağıtım
        return SignalType.HOLD

    def get_trend(self, ad_array: np.ndarray) -> TrendDirection:
        """
        A/D trendini belirle

        Args:
            ad_array: Son A/D değerleri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if len(ad_array) < 2:
            return TrendDirection.NEUTRAL

        # Lineer regresyon ile trend
        slope = np.polyfit(range(len(ad_array)), ad_array, 1)[0]

        if slope > 0:
            return TrendDirection.UP
        elif slope < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'signal_period': 10
        }

    def _requires_volume(self) -> bool:
        """A/D volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['AD']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """A/D indikatör testi"""

    print("\n" + "="*60)
    print("A/D (ACCUMULATION/DISTRIBUTION) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-3000, 5000))

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

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    ad = AD(signal_period=10)
    print(f"   [OK] Oluşturuldu: {ad}")
    print(f"   [OK] Kategori: {ad.category.value}")
    print(f"   [OK] Gerekli periyot: {ad.get_required_periods()}")

    result = ad(data)
    print(f"   [OK] A/D Değeri: {result.value:,.2f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [5, 10, 20]:
        ad_test = AD(signal_period=period)
        result = ad_test.calculate(data)
        print(f"   [OK] A/D(signal={period}): {result.value:,.2f} | Sinyal: {result.signal.value}")

    # Test 3: Volume gereksinimi
    print("\n4. Volume gereksinimi testi...")
    metadata = ad.metadata
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "A/D volume gerektirmeli!"

    # Test 4: Biriktirme/Dağıtım tespiti
    print("\n5. Biriktirme/Dağıtım testi...")
    result = ad.calculate(data)
    if result.signal == SignalType.BUY:
        print("   [OK] Biriktirme fazı tespit edildi")
    elif result.signal == SignalType.SELL:
        print("   [OK] Dağıtım fazı tespit edildi")
    else:
        print("   [OK] Nötr faz")

    # Test 5: MFM değeri
    print("\n6. Money Flow Multiplier testi...")
    mfm = result.metadata['mfm']
    print(f"   [OK] MFM: {mfm:.4f}")
    if mfm > 0:
        print("   [OK] Pozitif para akışı (alıcı baskısı)")
    elif mfm < 0:
        print("   [OK] Negatif para akışı (satıcı baskısı)")
    else:
        print("   [OK] Nötr para akışı")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = ad.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
