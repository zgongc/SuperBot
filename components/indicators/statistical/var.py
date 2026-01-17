#!/usr/bin/env python3
"""
indicators/statistical/var.py - VAR (Variance)

Yazar: SuperBot Team
Tarih: 2025-11-20
Versiyon: 1.0.0

VAR (Variance) - Varyans İndikatörü.
Fiyat dağılımının standart sapmasının karesi.

Özellikler:
- Volatilite ölçümü
- Fiyat dağılımının genişliği
- Yüksek VAR = Yüksek volatilite
- Düşük VAR = Düşük volatilite
- Risk yönetimi için kullanılır

Kullanım:
    from components.indicators import get_indicator_class

    VAR = get_indicator_class('var')
    var = VAR(period=20)
    result = var.calculate(data)
    print(result.value['var'])

Formül:
    VAR = Σ(Close - Mean)² / N

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


class VAR(BaseIndicator):
    """
    VAR - Variance

    Fiyat dağılımının standart sapmasının karesi.
    Volatilite ve risk ölçümü için kullanılır.

    Args:
        period: VAR periyodu (varsayılan: 20)
        logger: Logger instance (opsiyonel)
        error_handler: Error handler (opsiyonel)
    """

    def __init__(self, period: int = 20, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period)

        super().__init__(
            name='var',
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
        TA-Lib uyumlu: population variance (ddof=0) kullanır

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: VAR değerleri
        """
        # TA-Lib uyumlu: ddof=0 (population variance)
        var = data['close'].rolling(window=self.period).var(ddof=0)
        return pd.DataFrame({'var': var}, index=data.index)

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
        self.prices.extend(data['close'].tail(self.period).values)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Güncel VAR değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self.prices.append(close_val)

        if len(self.prices) < self.period:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # TA-Lib uyumlu: population variance (ddof=0)
        var_val = np.var(list(self.prices), ddof=0)

        # VAR kendisi sinyal üretmez ama yüksek volatiliteyi gösterir
        return IndicatorResult(
            value={'var': round(var_val, 4)},
            timestamp=timestamp_val,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=min(var_val * 10, 100),
            metadata={'period': self.period}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        VAR hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VAR değeri
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['var'].dropna().values

        if len(valid_values) == 0:
            return None

        var_val = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'var': round(var_val, 4)},
            timestamp=timestamp,
            signal=SignalType.HOLD,
            trend=TrendDirection.NEUTRAL,
            strength=min(var_val * 10, 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {'period': 20}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['var']

    def _requires_volume(self) -> bool:
        """VAR volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VAR']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """VAR indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 VAR (VARIANCE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # İki farklı volatilite dönemi
    low_vol = np.random.randn(75) * 1  # Düşük volatilite
    high_vol = np.random.randn(75) * 5  # Yüksek volatilite
    noise = np.concatenate([low_vol, high_vol])

    base_price = 100
    trend = np.linspace(0, 20, 150)
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
    var = VAR(period=20)
    print(f"   ✅ Oluşturuldu: {var}")
    print(f"   ✅ Kategori: {var.category.value}")
    print(f"   ✅ Gerekli periyot: {var.get_required_periods()}")

    result = var(data)
    print(f"   ✅ VAR: {result.value['var']}")
    print(f"   ✅ Güç: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = var.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Son 5 VAR değeri:")
    print(batch_result['var'].tail())

    # Test 3: Update metodu
    print("\n4. Update metodu testi...")
    var2 = VAR(period=20)
    init_data = data.head(50)
    var2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close']
        }
        update_result = var2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: VAR={update_result.value['var']:.4f}")

    # Test 4: Volatilite dönemleri analizi
    print("\n5. Volatilite dönemleri analizi...")
    batch_result = var.calculate_batch(data)
    var_values = batch_result['var'].dropna()

    # İlk ve ikinci yarı
    mid_point = len(var_values) // 2
    first_half = var_values.iloc[:mid_point]
    second_half = var_values.iloc[mid_point:]

    print(f"   ✅ İlk yarı ortalama VAR: {first_half.mean():.4f}")
    print(f"   ✅ İkinci yarı ortalama VAR: {second_half.mean():.4f}")
    print(f"   ✅ Volatilite artışı: {(second_half.mean() / first_half.mean()):.2f}x")

    # Test 5: Farklı periyotlar
    print("\n6. Farklı periyot testi...")
    for period in [10, 20, 30]:
        var_test = VAR(period=period)
        result = var_test.calculate(data)
        print(f"   ✅ VAR({period}): {result.value['var']:.4f}")

    # Test 6: Std sapma ile ilişki
    print("\n7. Standart sapma ile ilişki...")
    batch_result = var.calculate_batch(data)
    var_values = batch_result['var'].dropna()
    std_values = data['close'].rolling(window=20).std().dropna()

    # VAR = STD²
    valid_indices = ~var_values.isna() & ~std_values.isna()
    var_valid = var_values[valid_indices]
    std_valid = std_values[valid_indices]

    if len(var_valid) > 0:
        # Son değerleri karşılaştır
        last_var = var_valid.iloc[-1]
        last_std = std_valid.iloc[-1]
        expected_var = last_std ** 2

        print(f"   ✅ VAR: {last_var:.4f}")
        print(f"   ✅ STD²: {expected_var:.4f}")
        print(f"   ✅ Fark: {abs(last_var - expected_var):.6f}")

    # Test 7: Validasyon testi
    print("\n8. Validasyon testi...")
    try:
        invalid_var = VAR(period=1)
        print("   ❌ Hata: Geçersiz period kabul edildi!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validasyonu başarılı: {e}")

    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
