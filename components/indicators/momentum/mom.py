#!/usr/bin/env python3
"""
indicators/momentum/mom.py - MOM (Momentum Indicator)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

Momentum (MOM) - En basit momentum indikatörü.
Mevcut fiyat ile N periyot önceki fiyat arasındaki farkı hesaplar.

Özellikler:
- Basit ve hızlı hesaplama
- Trend yönü ve gücünü ölçer
- Pozitif değer = Bullish momentum
- Negatif değer = Bearish momentum

Kullanım:
    from components.indicators import get_indicator_class

    MOM = get_indicator_class('mom')
    mom = MOM(period=10)
    result = mom.calculate(data)
    print(result.value['mom'])

Formül:
    MOM = Close - Close[n periods ago]

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
from collections import deque
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class MOM(BaseIndicator):
    """
    Momentum Indicator

    En basit momentum göstergesi. Mevcut fiyat ile N periyot önceki
    fiyat arasındaki farkı hesaplar.

    Args:
        period: Momentum periyodu (varsayılan: 10)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 10, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period + 1)

        super().__init__(
            name='mom',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
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
                "Period pozitif olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: MOM değerleri
        """
        close = data['close']
        mom = close - close.shift(self.period)
        return pd.DataFrame({'mom': mom}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        # prices deque'yu doldur
        self.prices.clear()
        # Son period+1 veri gerekli
        tail_data = data['close'].tail(self.period + 1).values
        for val in tail_data:
            self.prices.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Güncel MOM değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self.prices.append(close_val)

        if len(self.prices) < self.period + 1:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        mom_value = self.prices[-1] - self.prices[0]

        # Sinyal ve trend belirleme
        if mom_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif mom_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        return IndicatorResult(
            value={'mom': round(mom_value, 4)},
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=min(abs(mom_value) * 10, 100),
            metadata={'period': self.period}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        MOM hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: MOM değeri
        """
        # Buffer'ları doldur
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['mom'].dropna().values

        if len(valid_values) == 0:
            return None

        mom_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Sinyal ve trend
        if mom_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif mom_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'mom': round(mom_value, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(mom_value) * 10, 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 10}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['mom']

    def _requires_volume(self) -> bool:
        """MOM volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MOM']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """MOM indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 MOMENTUM (MOM) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   ✅ {len(data)} mum oluşturuldu")
    print(f"   ✅ Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    mom = MOM(period=10)
    print(f"   ✅ Oluşturuldu: {mom}")
    print(f"   ✅ Kategori: {mom.category.value}")
    print(f"   ✅ Gerekli periyot: {mom.get_required_periods()}")

    result = mom(data)
    print(f"   ✅ MOM: {result.value['mom']}")
    print(f"   ✅ Sinyal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = mom.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 MOM değeri:")
    print(batch_result['mom'].tail())

    # Test 3: Update metodu
    print("\n4. Update metodu testi...")
    mom2 = MOM(period=10)
    init_data = data.head(50)
    mom2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low']
        }
        update_result = mom2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: MOM={update_result.value['mom']:.4f}, "
                  f"Signal={update_result.signal.value}")

    # Test 4: Farklı periyotlar
    print("\n5. Farklı periyot testi...")
    for period in [5, 10, 20]:
        mom_test = MOM(period=period)
        result = mom_test.calculate(data)
        print(f"   ✅ MOM({period}): {result.value['mom']:.4f}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
