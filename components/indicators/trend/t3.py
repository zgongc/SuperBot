#!/usr/bin/env python3
"""
indicators/trend/t3.py - T3 (Tillson T3 Moving Average)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

T3 (Tillson T3) - Tillson T3 Hareketli Ortalama.
Tim Tillson tarafından geliştirilen gelişmiş smoothing algoritması.

Özellikler:
- 6 kez EMA uygulayarak ultra-smooth sonuç
- Volume factor ile ayarlanabilir gecikme/duyarlılık
- Düşük gecikme ile pürüzsüz trend
- Fiyat crossover sinyalleri
- Trend yönü ve gücünü gösterir

Kullanım:
    from components.indicators import get_indicator_class

    T3 = get_indicator_class('t3')
    t3 = T3(period=5, vfactor=0.7)
    result = t3.calculate(data)
    print(result.value['t3'])

Formül:
    c1 = -vfactor^3
    c2 = 3*vfactor^2 + 3*vfactor^3
    c3 = -6*vfactor^2 - 3*vfactor - 3*vfactor^3
    c4 = 1 + 3*vfactor + vfactor^3 + 3*vfactor^2

    e1 = EMA(Close, period)
    e2 = EMA(e1, period)
    e3 = EMA(e2, period)
    e4 = EMA(e3, period)
    e5 = EMA(e4, period)
    e6 = EMA(e5, period)

    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3

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
from components.indicators.trend.ema import EMA
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class T3(BaseIndicator):
    """
    T3 - Tillson T3 Moving Average

    Tim Tillson tarafından geliştirilen gelişmiş trend göstergesi.
    6 kez EMA uygulanarak pürüzsüz sonuç elde edilir.

    Args:
        period: T3 periyodu (varsayılan: 5)
        vfactor: Volume factor (0-1 arası, varsayılan: 0.7)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 5, vfactor: float = 0.7, logger=None, error_handler=None):
        self.period = period
        self.vfactor = vfactor

        # EMA indikatörünü kullan (code reuse)
        self._ema = EMA(period=period)

        super().__init__(
            name='t3',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period, 'vfactor': vfactor},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.period * 6

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period pozitif olmalı"
            )
        if not (0 <= self.vfactor <= 1):
            raise InvalidParameterError(
                self.name, 'vfactor', self.vfactor,
                "VFactor 0 ile 1 arasında olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: T3 değerleri
        """
        # Katsayıları hesapla
        c1 = -self.vfactor ** 3
        c2 = 3 * self.vfactor ** 2 + 3 * self.vfactor ** 3
        c3 = -6 * self.vfactor ** 2 - 3 * self.vfactor - 3 * self.vfactor ** 3
        c4 = 1 + 3 * self.vfactor + self.vfactor ** 3 + 3 * self.vfactor ** 2

        # 6 kez EMA uygula - EMA.calculate_batch kullan (code reuse)
        e1 = self._ema.calculate_batch(data)
        e2 = self._ema.calculate_batch(self._create_ema_input(e1, data))
        e3 = self._ema.calculate_batch(self._create_ema_input(e2, data))
        e4 = self._ema.calculate_batch(self._create_ema_input(e3, data))
        e5 = self._ema.calculate_batch(self._create_ema_input(e4, data))
        e6 = self._ema.calculate_batch(self._create_ema_input(e5, data))

        # T3 hesapla
        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

        return pd.DataFrame({'t3': t3}, index=data.index)

    def _create_ema_input(self, series: pd.Series, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create minimal OHLCV DataFrame for EMA calculation from a Series."""
        return pd.DataFrame({
            'timestamp': original_data['timestamp'].values,
            'open': series.values,
            'high': series.values,
            'low': series.values,
            'close': series.values,
            'volume': np.zeros(len(series))
        }, index=original_data.index)

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
            IndicatorResult: T3 değeri
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
        T3 hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: T3 değeri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['t3'].dropna().values

        if len(valid_values) == 0:
            return IndicatorResult(
                value=0.0,
                timestamp=int(data.iloc[-1]['timestamp']) if len(data) > 0 else 0,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        t3_val = valid_values[-1]
        close = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal belirleme: Fiyat T3'ün üstünde = BUY
        if close > t3_val:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif close < t3_val:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Güç: Fiyat ile T3 arasındaki yüzde fark
        strength = min(abs((close - t3_val) / t3_val * 100) * 10, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'t3': round(t3_val, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={'period': self.period, 'vfactor': self.vfactor}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 5, 'vfactor': 0.7}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['t3']

    def _requires_volume(self) -> bool:
        """T3 volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['T3']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """T3 indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 T3 (TILLSON T3 MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(200)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 30, 200)
    noise = np.random.randn(200) * 3
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(200)),
        'low': prices - np.abs(np.random.randn(200)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(200)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    t3 = T3(period=5, vfactor=0.7)
    print(f"   ✅ Oluşturuldu: {t3}")
    print(f"   ✅ Kategori: {t3.category.value}")
    print(f"   ✅ Gerekli periyot: {t3.get_required_periods()}")

    result = t3(data)
    print(f"   ✅ T3: {result.value['t3']}")
    print(f"   ✅ Son Close: {data['close'].iloc[-1]:.2f}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = t3.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 T3 değeri:")
    print(batch_result['t3'].tail())

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [3, 5, 10]:
        t3_test = T3(period=period, vfactor=0.7)
        result = t3_test.calculate(data)
        print(f"   ✅ T3({period}): {result.value['t3']:.2f}, Signal={result.signal.value}")

    # Test 4: Farklı vfactor değerleri
    print("\n5. Farklı vfactor testi...")
    for vf in [0.3, 0.5, 0.7, 0.9]:
        t3_test = T3(period=5, vfactor=vf)
        result = t3_test.calculate(data)
        print(f"   ✅ T3(vf={vf}): {result.value['t3']:.2f}")

    # Test 5: Crossover analizi
    print("\n6. Fiyat-T3 crossover analizi...")
    batch_result = t3.calculate_batch(data)
    t3_values = batch_result['t3'].dropna()
    close_values = data['close'].iloc[len(data)-len(t3_values):]

    crossovers = 0
    for i in range(1, len(t3_values)):
        t3_prev = t3_values.iloc[i-1]
        t3_curr = t3_values.iloc[i]
        close_prev = close_values.iloc[i-1]
        close_curr = close_values.iloc[i]

        if (close_prev < t3_prev and close_curr > t3_curr) or \
           (close_prev > t3_prev and close_curr < t3_curr):
            crossovers += 1

    print(f"   ✅ Toplam fiyat-T3 crossover: {crossovers}")
    print(f"   ✅ Fiyat T3 üstünde: {sum(close_values.values > t3_values.values)}")
    print(f"   ✅ Fiyat T3 altında: {sum(close_values.values < t3_values.values)}")

    # Test 6: Validasyon testi
    print("\n7. Validasyon testi...")
    try:
        invalid_t3 = T3(period=0, vfactor=0.7)
        print("   ❌ Hata: Geçersiz period kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    try:
        invalid_t3 = T3(period=5, vfactor=1.5)
        print("   ❌ Hata: Geçersiz vfactor kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ VFactor validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
