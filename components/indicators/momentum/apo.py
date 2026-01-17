#!/usr/bin/env python3
"""
indicators/momentum/apo.py - APO (Absolute Price Oscillator)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

APO (Absolute Price Oscillator) - Mutlak Fiyat Osilatörü.
İki farklı periyotlu EMA arasındaki mutlak farkı hesaplar.

Özellikler:
- Hızlı ve yavaş EMA farkı
- MACD'ye benzer ancak mutlak değer kullanır
- Trend gücünü ve yönünü gösterir
- Pozitif değer = Bullish momentum
- Negatif değer = Bearish momentum

Kullanım:
    from components.indicators import get_indicator_class

    APO = get_indicator_class('apo')
    apo = APO(fast_period=12, slow_period=26)
    result = apo.calculate(data)
    print(result.value['apo'])

Formül:
    APO = Fast EMA - Slow EMA

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


class APO(BaseIndicator):
    """
    APO - Absolute Price Oscillator

    İki farklı periyotlu EMA arasındaki mutlak farkı hesaplar.
    MACD'nin mutlak değer versiyonudur.

    Args:
        fast_period: Hızlı EMA periyodu (varsayılan: 12)
        slow_period: Yavaş EMA periyodu (varsayılan: 26)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, logger=None, error_handler=None):
        self.fast_period = fast_period
        self.slow_period = slow_period

        super().__init__(
            name='apo',
            category=IndicatorCategory.MOMENTUM,
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
        TA-Lib uyumlu: SMA kullanır (varsayılan matype=0)

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: APO değerleri
        """
        close = data['close']

        # TA-Lib uyumlu: SMA kullan (EMA değil!)
        fast_ma = close.rolling(window=self.fast_period).mean()
        slow_ma = close.rolling(window=self.slow_period).mean()

        # Mutlak fark
        apo = fast_ma - slow_ma

        return pd.DataFrame({'apo': apo}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - state-based update için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel, multi-symbol desteği için)
        """
        super().warmup_buffer(data, symbol)

        buffer_key = symbol if symbol else 'default'
        if not hasattr(self, '_apo_state'):
            self._apo_state = {}

        if len(data) >= self.slow_period:
            close = data['close'].values
            # Son slow_period kadar veriyi tut
            self._apo_state[buffer_key] = {
                'close_buffer': list(close[-self.slow_period:]),
                'last_close': close[-1]
            }

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - State-based

        Args:
            candle: Yeni mum verisi (dict)
            symbol: Sembol adı (opsiyonel)

        Returns:
            IndicatorResult: Güncel APO değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # State varsa incremental hesapla
        if hasattr(self, '_apo_state') and buffer_key in self._apo_state:
            state = self._apo_state[buffer_key]
            close_buffer = state['close_buffer']

            # Buffer'a yeni close ekle
            close_buffer.append(close_val)

            # Buffer boyutunu koru
            if len(close_buffer) > self.slow_period:
                close_buffer.pop(0)

            # SMA hesapla
            if len(close_buffer) >= self.slow_period:
                fast_ma = np.mean(close_buffer[-self.fast_period:])
                slow_ma = np.mean(close_buffer[-self.slow_period:])
                apo_value = fast_ma - slow_ma

                # State güncelle
                self._apo_state[buffer_key] = {
                    'close_buffer': close_buffer,
                    'last_close': close_val
                }

                # Sinyal ve trend belirleme
                if apo_value > 0:
                    signal = SignalType.BUY
                    trend = TrendDirection.UP
                elif apo_value < 0:
                    signal = SignalType.SELL
                    trend = TrendDirection.DOWN
                else:
                    signal = SignalType.HOLD
                    trend = TrendDirection.NEUTRAL

                return IndicatorResult(
                    value={'apo': round(apo_value, 4)},
                    timestamp=timestamp_val,
                    signal=signal,
                    trend=trend,
                    strength=min(abs(apo_value) * 10, 100),
                    metadata={'fast': self.fast_period, 'slow': self.slow_period}
                )

        # State yoksa yetersiz veri
        return IndicatorResult(
            value=0.0,
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=0.0,
            metadata={'insufficient_data': True}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        APO hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: APO değeri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['apo'].dropna().values

        if len(valid_values) == 0:
            return None

        apo_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal ve trend belirleme
        if apo_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif apo_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'apo': round(apo_value, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(apo_value) * 10, 100),
            metadata={'fast': self.fast_period, 'slow': self.slow_period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'fast_period': 12, 'slow_period': 26}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['apo']

    def _requires_volume(self) -> bool:
        """APO volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['APO']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """APO indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 APO (ABSOLUTE PRICE OSCILLATOR) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 25, 150)
    noise = np.random.randn(150) * 2.5
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
    apo = APO(fast_period=12, slow_period=26)
    print(f"   ✅ Oluşturuldu: {apo}")
    print(f"   ✅ Kategori: {apo.category.value}")
    print(f"   ✅ Gerekli periyot: {apo.get_required_periods()}")

    result = apo(data)
    print(f"   ✅ APO: {result.value['apo']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = apo.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 APO değeri:")
    print(batch_result['apo'].tail())

    # Test 3: Farklı periyot kombinasyonları
    print("\n4. Farklı periyot testi...")
    configs = [(5, 10), (12, 26), (20, 50)]
    for fast, slow in configs:
        apo_test = APO(fast_period=fast, slow_period=slow)
        result = apo_test.calculate(data)
        print(f"   ✅ APO({fast},{slow}): {result.value['apo']:.4f}, Signal={result.signal.value}")

    # Test 4: Zero-line crossover analizi
    print("\n5. Zero-line crossover analizi...")
    batch_result = apo.calculate_batch(data)
    apo_values = batch_result['apo'].dropna()

    # Crossover sayısı
    crossovers = 0
    for i in range(1, len(apo_values)):
        if (apo_values.iloc[i-1] < 0 and apo_values.iloc[i] > 0) or \
           (apo_values.iloc[i-1] > 0 and apo_values.iloc[i] < 0):
            crossovers += 1

    print(f"   ✅ Toplam zero-line crossover: {crossovers}")
    print(f"   ✅ Pozitif APO barlar: {sum(apo_values > 0)}")
    print(f"   ✅ Negatif APO barlar: {sum(apo_values < 0)}")
    print(f"   ✅ Ortalama APO: {apo_values.mean():.4f}")
    print(f"   ✅ APO std sapma: {apo_values.std():.4f}")

    # Test 5: Validasyon testi
    print("\n6. Validasyon testi...")
    try:
        invalid_apo = APO(fast_period=26, slow_period=12)
        print("   ❌ Hata: Geçersiz periyot kombinasyonu kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
