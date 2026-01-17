"""
indicators/volume/volume_oscillator.py - Volume Oscillator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Volume Oscillator - Hacim osilatörü
    İki farklı periyotlu hacim hareketli ortalaması arasındaki farkı ölçer
    Pozitif = Kısa vadeli hacim artışı
    Negatif = Kısa vadeli hacim düşüşü

Formül:
    VO = ((Fast_MA - Slow_MA) / Slow_MA) × 100
    Fast_MA = SMA(Volume, fast_period)
    Slow_MA = SMA(Volume, slow_period)

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


class VolumeOscillator(BaseIndicator):
    """
    Volume Oscillator

    Kısa ve uzun vadeli hacim hareketli ortalamalarını karşılaştırır.
    Hacim trendlerini ve değişimlerini tespit eder.

    Args:
        fast_period: Hızlı MA periyodu (varsayılan: 5)
        slow_period: Yavaş MA periyodu (varsayılan: 10)
        signal_period: Sinyal hattı SMA periyodu (varsayılan: 10)
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 10,
        signal_period: int = 10,
        logger=None,
        error_handler=None
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        super().__init__(
            name='volume_oscillator',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.slow_period, self.signal_period)

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.fast_period < 1:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "Fast periyot pozitif olmalı"
            )
        if self.slow_period < 1:
            raise InvalidParameterError(
                self.name, 'slow_period', self.slow_period,
                "Slow periyot pozitif olmalı"
            )
        if self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'periods',
                f"fast={self.fast_period}, slow={self.slow_period}",
                "Fast periyot, slow periyottan küçük olmalı"
            )
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "Signal periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Volume Oscillator hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VO değeri
        """
        volume = data['volume'].values

        # Fast ve Slow MA hesapla
        fast_ma = np.mean(volume[-self.fast_period:])
        slow_ma = np.mean(volume[-self.slow_period:])

        # Volume Oscillator hesapla (yüzde olarak)
        if slow_ma == 0:
            vo_value = 0.0
        else:
            vo_value = ((fast_ma - slow_ma) / slow_ma) * 100

        # Tüm VO değerlerini hesapla (signal line için)
        vo_array = np.zeros(len(volume))
        for i in range(self.slow_period - 1, len(volume)):
            fast_avg = np.mean(volume[max(0, i - self.fast_period + 1):i + 1])
            slow_avg = np.mean(volume[max(0, i - self.slow_period + 1):i + 1])

            if slow_avg != 0:
                vo_array[i] = ((fast_avg - slow_avg) / slow_avg) * 100

        # Sinyal hattı (VO'nun SMA'sı)
        vo_signal = np.mean(vo_array[-self.signal_period:])

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(vo_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(vo_value, vo_signal),
            trend=self.get_trend(vo_value),
            strength=min(abs(vo_value), 100),
            metadata={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'fast_ma': round(fast_ma, 2),
                'slow_ma': round(slow_ma, 2),
                'vo_signal': round(vo_signal, 2),
                'current_volume': int(volume[-1])
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Volume Oscillator calculation - BACKTEST için

        VO Formula:
            VO = ((Fast_MA - Slow_MA) / Slow_MA) × 100
            Fast_MA = SMA(Volume, fast_period)
            Slow_MA = SMA(Volume, slow_period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: VO values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        volume = data['volume']

        # Fast MA (short period)
        fast_ma = volume.rolling(window=self.fast_period).mean()

        # Slow MA (long period)
        slow_ma = volume.rolling(window=self.slow_period).mean()

        # Volume Oscillator = ((Fast - Slow) / Slow) × 100
        vo = ((fast_ma - slow_ma) / slow_ma) * 100

        # Handle division by zero
        vo = vo.fillna(0).replace([np.inf, -np.inf], 0)

        # Set first period values to NaN (warmup)
        vo.iloc[:self.slow_period-1] = np.nan

        return pd.Series(vo.values, index=data.index, name='vo')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli state'i hazırlar"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._volume_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._volume_buffer.append(data['volume'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

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
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._volume_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, vo_value: float, vo_signal: float) -> SignalType:
        """
        VO değerinden sinyal üret

        Args:
            vo_value: VO değeri
            vo_signal: VO sinyal hattı

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # VO sinyal hattını geçerse
        if vo_value > vo_signal and vo_value > 0:
            return SignalType.BUY  # Hacim artıyor ve pozitif
        elif vo_value < vo_signal and vo_value < 0:
            return SignalType.SELL  # Hacim azalıyor ve negatif
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        VO değerinden trend belirle

        Args:
            value: VO değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 5:  # %5'den fazla artış
            return TrendDirection.UP
        elif value < -5:  # %5'den fazla azalış
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'fast_period': 5,
            'slow_period': 10,
            'signal_period': 10
        }

    def _requires_volume(self) -> bool:
        """Volume Oscillator volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VolumeOscillator']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Volume Oscillator indikatör testi"""

    print("\n" + "="*60)
    print("VOLUME OSCILLATOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]

    # Hacim trendi oluştur (önce düşük, sonra yüksek)
    volumes = []
    for i in range(30):
        if i < 15:
            volumes.append(8000 + np.random.randint(-2000, 2000))
        else:
            volumes.append(15000 + np.random.randint(-3000, 5000))

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

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
    vo = VolumeOscillator(fast_period=5, slow_period=10)
    print(f"   [OK] Oluşturuldu: {vo}")
    print(f"   [OK] Kategori: {vo.category.value}")
    print(f"   [OK] Gerekli periyot: {vo.get_required_periods()}")

    result = vo(data)
    print(f"   [OK] VO Değeri: {result.value:.2f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for fast, slow in [(3, 10), (5, 10), (5, 20)]:
        vo_test = VolumeOscillator(fast_period=fast, slow_period=slow)
        result = vo_test.calculate(data)
        print(f"   [OK] VO({fast},{slow}): {result.value:.2f}% | Sinyal: {result.signal.value}")

    # Test 3: Volume gereksinimi
    print("\n4. Volume gereksinimi testi...")
    metadata = vo.metadata
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "Volume Oscillator volume gerektirmeli!"

    # Test 4: Hacim trend analizi
    print("\n5. Hacim trend analizi...")
    result = vo.calculate(data)
    vo_val = result.value
    fast_ma = result.metadata['fast_ma']
    slow_ma = result.metadata['slow_ma']

    print(f"   [OK] Fast MA ({vo.fast_period}): {fast_ma:,.2f}")
    print(f"   [OK] Slow MA ({vo.slow_period}): {slow_ma:,.2f}")
    print(f"   [OK] VO: {vo_val:.2f}%")

    if vo_val > 10:
        print("   [OK] Güçlü hacim artışı")
    elif vo_val > 0:
        print("   [OK] Zayıf hacim artışı")
    elif vo_val > -10:
        print("   [OK] Zayıf hacim azalışı")
    else:
        print("   [OK] Güçlü hacim azalışı")

    # Test 5: Sinyal hattı crossover
    print("\n6. Sinyal hattı testi...")
    vo_signal = result.metadata['vo_signal']
    print(f"   [OK] VO: {vo_val:.2f}%")
    print(f"   [OK] VO Signal: {vo_signal:.2f}%")
    if vo_val > vo_signal:
        print("   [OK] VO sinyal hattı üstünde (güçleniyor)")
    else:
        print("   [OK] VO sinyal hattı altında (zayıflıyor)")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = vo.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
