#!/usr/bin/env python3
"""
indicators/statistical/tsf.py - TSF (Time Series Forecast)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

TSF (Time Series Forecast) - Zaman Serisi Tahmini.
Lineer regresyon kullanarak gelecek fiyat tahmini.

Özellikler:
- Lineer regresyon ile trend tahmini
- Bir sonraki bar için fiyat tahmini
- Trend yönü ve gücünü gösterir
- Fiyat-TSF farkı sinyal üretir
- Destek/direnç seviyesi olarak kullanılabilir

Kullanım:
    from components.indicators import get_indicator_class

    TSF = get_indicator_class('tsf')
    tsf = TSF(period=14)
    result = tsf.calculate(data)
    print(result.value['tsf'])

Formül:
    Linear Regression: y = mx + b
    TSF = m * (period) + b
    (Bir sonraki değer tahmini)

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


class TSF(BaseIndicator):
    """
    TSF - Time Series Forecast

    Lineer regresyon kullanarak gelecek fiyat tahmini yapar.
    Bir sonraki bar için fiyat projeksiyonu.

    Args:
        period: TSF periyodu (varsayılan: 14)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 14, logger=None, error_handler=None):
        self.period = period

        super().__init__(
            name='tsf',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period en az 2 olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: TSF değerleri
        """
        close = data['close']
        tsf_values = []

        for i in range(len(close)):
            if i < self.period - 1:
                tsf_values.append(np.nan)
            else:
                # Son period kadar veriyi al
                y = close.iloc[i - self.period + 1:i + 1].values
                x = np.arange(self.period)

                # Lineer regresyon
                slope, intercept = np.polyfit(x, y, 1)

                # Bir sonraki değer tahmini
                forecast = slope * self.period + intercept
                tsf_values.append(forecast)

        return pd.DataFrame({'tsf': tsf_values}, index=data.index)

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

        self._close_buffer = deque(maxlen=max_len)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: TSF değeri
        """
        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._close_buffer.append(close_val)

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
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        TSF hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TSF değeri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['tsf'].dropna().values

        if len(valid_values) == 0:
            return None

        tsf_val = valid_values[-1]
        close = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme: TSF > Close = forecast yükselecek (BUY)
        if tsf_val > close:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif tsf_val < close:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Güç: Tahmin ile gerçek arasındaki yüzde fark
        strength = min(abs((tsf_val - close) / close * 100) * 10, 100)

        # Forecast farkını metadata'ya ekle
        forecast_diff = tsf_val - close

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'tsf': round(tsf_val, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'period': self.period,
                'forecast_diff': round(forecast_diff, 2)
            }
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 14}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['tsf']

    def _requires_volume(self) -> bool:
        """TSF volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TSF']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """TSF indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 TSF (TIME SERIES FORECAST) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Güçlü trend + noise
    base_price = 100
    trend = np.linspace(0, 30, 150)
    noise = np.random.randn(150) * 2
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(150)),
        'low': prices - np.abs(np.random.randn(150)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(150)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    tsf = TSF(period=14)
    print(f"   ✅ Oluşturuldu: {tsf}")
    print(f"   ✅ Kategori: {tsf.category.value}")
    print(f"   ✅ Gerekli periyot: {tsf.get_required_periods()}")

    result = tsf(data)
    print(f"   ✅ TSF: {result.value['tsf']}")
    print(f"   ✅ Close: {data['close'].iloc[-1]:.2f}")
    print(f"   ✅ Forecast farkı: {result.metadata['forecast_diff']:.2f}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = tsf.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 TSF değeri:")
    print(batch_result['tsf'].tail())

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [7, 14, 21]:
        tsf_test = TSF(period=period)
        result = tsf_test.calculate(data)
        print(f"   ✅ TSF({period}): {result.value['tsf']:.2f}, "
              f"Diff={result.metadata['forecast_diff']:.2f}")

    # Test 4: Tahmin doğruluğu analizi
    print("\n5. Tahmin doğruluğu analizi...")
    batch_result = tsf.calculate_batch(data)
    tsf_values = batch_result['tsf'].dropna()

    # TSF'in bir sonraki bar'ı ne kadar doğru tahmin ettiğini kontrol et
    errors = []
    for i in range(len(tsf_values) - 1):
        forecast = tsf_values.iloc[i]
        actual = data['close'].iloc[i + 1]
        error = abs(forecast - actual)
        errors.append(error)

    if len(errors) > 0:
        print(f"   ✅ Ortalama tahmin hatası: {np.mean(errors):.4f}")
        print(f"   ✅ Min hata: {min(errors):.4f}")
        print(f"   ✅ Max hata: {max(errors):.4f}")
        print(f"   ✅ Hata std sapması: {np.std(errors):.4f}")

    # Test 5: Trend çizgisi ile karşılaştırma
    print("\n6. Fiyat ile TSF karşılaştırması...")
    batch_result = tsf.calculate_batch(data)
    tsf_values = batch_result['tsf'].dropna()
    close_values = data['close'].iloc[len(data)-len(tsf_values):]

    # TSF ne kadar fiyatın üstünde/altında
    above_count = sum(tsf_values.values > close_values.values)
    below_count = sum(tsf_values.values < close_values.values)

    print(f"   ✅ TSF > Close: {above_count}")
    print(f"   ✅ TSF < Close: {below_count}")
    print(f"   ✅ Ortalama fark: {(tsf_values.values - close_values.values).mean():.4f}")

    # Test 6: Trend gücü analizi
    print("\n7. Trend gücü analizi...")
    # Son N bar için trend eğimini hesapla
    last_n = 30
    recent_closes = data['close'].tail(last_n).values
    x = np.arange(last_n)
    slope, intercept = np.polyfit(x, recent_closes, 1)

    print(f"   ✅ Son {last_n} bar trend eğimi: {slope:.4f}")
    print(f"   ✅ Trend yönü: {'Yükseliş' if slope > 0 else 'Düşüş'}")

    # Test 7: Validasyon testi
    print("\n8. Validasyon testi...")
    try:
        invalid_tsf = TSF(period=1)
        print("   ❌ Hata: Geçersiz period kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
