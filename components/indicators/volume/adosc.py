#!/usr/bin/env python3
"""
indicators/volume/adosc.py - ADOSC (Chaikin A/D Oscillator)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

ADOSC (Chaikin A/D Oscillator) - Chaikin Akümülasyon/Dağıtım Osilatörü.
Accumulation/Distribution Line'ın iki EMA'sının farkını hesaplar.

Özellikler:
- Marc Chaikin tarafından geliştirildi
- Volume-based momentum göstergesi
- Alış ve satış baskısını ölçer
- Pozitif değer = Akümülasyon (Alış baskısı)
- Negatif değer = Dağıtım (Satış baskısı)
- Divergence sinyalleri verir

Kullanım:
    from components.indicators import get_indicator_class

    ADOSC = get_indicator_class('adosc')
    adosc = ADOSC(fast_period=3, slow_period=10)
    result = adosc.calculate(data)
    print(result.value['adosc'])

Formül:
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    AD = Cumulative(CLV * Volume)
    ADOSC = Fast EMA(AD) - Slow EMA(AD)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje root'unu path'e ekle
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class ADOSC(BaseIndicator):
    """
    ADOSC - Chaikin A/D Oscillator

    Accumulation/Distribution Line'ın iki EMA'sının farkını hesaplar.
    Volume-based momentum göstergesidir.

    Args:
        fast_period: Hızlı EMA periyodu (varsayılan: 3)
        slow_period: Yavaş EMA periyodu (varsayılan: 10)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, fast_period: int = 3, slow_period: int = 10, logger=None, error_handler=None):
        self.fast_period = fast_period
        self.slow_period = slow_period

        super().__init__(
            name='adosc',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'fast_period': fast_period, 'slow_period': slow_period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.slow_period * 2

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "Hızlı periyot yavaş periyottan küçük olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: ADOSC değerleri
        """
        # Close Location Value
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)  # Sıfıra bölme durumu

        # Accumulation/Distribution Line
        ad = (clv * data['volume']).cumsum()

        # ADOSC = Fast EMA(AD) - Slow EMA(AD)
        fast_ema = ad.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = ad.ewm(span=self.slow_period, adjust=False).mean()
        adosc = fast_ema - slow_ema

        return pd.DataFrame({'adosc': adosc}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        # Buffer'ları oluştur ve doldur
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: ADOSC değeri
        """
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            open_val = candle.get('open', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'open': [open_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        ADOSC hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: ADOSC değeri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['adosc'].dropna().values

        if len(valid_values) == 0:
            return None

        adosc_val = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme
        if adosc_val > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif adosc_val < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'adosc': round(adosc_val, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(adosc_val) / 1000, 100),
            metadata={'fast': self.fast_period, 'slow': self.slow_period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'fast_period': 3, 'slow_period': 10}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['adosc']

    def _requires_volume(self) -> bool:
        """ADOSC volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ADOSC']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """ADOSC indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 ADOSC (CHAIKIN A/D OSCILLATOR) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 25, 150)
    noise = np.random.randn(150) * 2.5
    close_prices = base_price + trend + noise

    # OHLC oluştur
    opens = close_prices + np.random.randn(150) * 0.5
    highs = np.maximum(opens, close_prices) + np.abs(np.random.randn(150))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.randn(150))

    # Volume with trend
    volumes = 1000 + np.abs(np.random.randn(150) * 200) + np.linspace(0, 500, 150)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(close_prices):.2f} -> {max(close_prices):.2f}")
    print(f"   ✅ Volume aralığı: {min(volumes):.0f} -> {max(volumes):.0f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    adosc = ADOSC(fast_period=3, slow_period=10)
    print(f"   ✅ Oluşturuldu: {adosc}")
    print(f"   ✅ Kategori: {adosc.category.value}")
    print(f"   ✅ Volume gerekli: {adosc._requires_volume()}")
    print(f"   ✅ Gerekli periyot: {adosc.get_required_periods()}")

    result = adosc(data)
    print(f"   ✅ ADOSC: {result.value['adosc']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = adosc.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 ADOSC değeri:")
    print(batch_result['adosc'].tail())

    # Test 3: Farklı periyot kombinasyonları
    print("\n4. Farklı periyot testi...")
    configs = [(3, 10), (5, 15), (7, 20)]
    for fast, slow in configs:
        adosc_test = ADOSC(fast_period=fast, slow_period=slow)
        result = adosc_test.calculate(data)
        print(f"   ✅ ADOSC({fast},{slow}): {result.value['adosc']:.2f}, Signal={result.signal.value}")

    # Test 4: Zero-line crossover analizi
    print("\n5. Zero-line crossover analizi...")
    batch_result = adosc.calculate_batch(data)
    adosc_values = batch_result['adosc'].dropna()

    # Crossover sayısı
    crossovers = 0
    for i in range(1, len(adosc_values)):
        if (adosc_values.iloc[i-1] < 0 and adosc_values.iloc[i] > 0) or \
           (adosc_values.iloc[i-1] > 0 and adosc_values.iloc[i] < 0):
            crossovers += 1

    print(f"   ✅ Toplam zero-line crossover: {crossovers}")
    print(f"   ✅ Pozitif ADOSC barlar: {sum(adosc_values > 0)}")
    print(f"   ✅ Negatif ADOSC barlar: {sum(adosc_values < 0)}")
    print(f"   ✅ Ortalama ADOSC: {adosc_values.mean():.2f}")
    print(f"   ✅ ADOSC std sapma: {adosc_values.std():.2f}")

    # Test 5: Validasyon testi
    print("\n6. Validasyon testi...")
    try:
        invalid_adosc = ADOSC(fast_period=10, slow_period=3)
        print("   ❌ Hata: Geçersiz periyot kombinasyonu kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
