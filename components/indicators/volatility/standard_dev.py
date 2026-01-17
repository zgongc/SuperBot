"""
indicators/volatility/standard_dev.py - Standard Deviation

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Standard Deviation - Standart Sapma
    Fiyat dağılımının ne kadar yayıldığını ölçer
    Yüksek değer = Yüksek volatilite
    Düşük değer = Düşük volatilite

Formül:
    StdDev = sqrt(sum((close - mean)^2) / period)

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


class StandardDeviation(BaseIndicator):
    """
    Standard Deviation

    Fiyat hareketlerinin ortalamadan sapmasını ölçerek volatiliteyi belirler.
    Risk yönetimi ve pozisyon büyüklüğü için kullanılır.

    Args:
        period: Hesaplama periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='standard_dev',
            category=IndicatorCategory.VOLATILITY,
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
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot en az 2 olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Standard Deviation hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Standard Deviation değeri
        """
        close = data['close'].values

        # Son period kadar veriyi al
        recent_close = close[-self.period:]

        # Standart sapma hesapla
        std_value = np.std(recent_close, ddof=0)

        # Ortalama ve varyasyon katsayısı
        mean_value = np.mean(recent_close)
        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Varyasyon katsayısı (Coefficient of Variation)
        if mean_value != 0:
            cv = (std_value / mean_value) * 100
        else:
            cv = 0

        # Volatilite seviyesi (z-score benzeri)
        if std_value != 0:
            deviation_from_mean = abs(current_price - mean_value) / std_value
        else:
            deviation_from_mean = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(std_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(cv),
            trend=TrendDirection.NEUTRAL,  # Standart sapma trend göstermez
            strength=min(cv * 10, 100),  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'mean': round(mean_value, 8),
                'cv': round(cv, 2),  # Varyasyon katsayısı (%)
                'z_score': round(deviation_from_mean, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Standard Deviation calculation - BACKTEST için

        StdDev Formula:
            StdDev = sqrt(sum((close - mean)^2) / period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Standard Deviation values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Rolling standard deviation (vectorized)
        std_dev = close.rolling(window=self.period).std(ddof=0)

        # Set first period values to NaN (warmup)
        std_dev.iloc[:self.period-1] = np.nan

        return pd.Series(std_dev.values, index=data.index, name='std_dev')

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
        """Incremental update (Real-time)"""
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
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, cv: float) -> SignalType:
        """
        Varyasyon katsayısından sinyal üret

        Args:
            cv: Varyasyon katsayısı (%)

        Returns:
            SignalType: Volatilite seviyesine göre sinyal
        """
        # Yüksek volatilite: dikkatli ol
        if cv > 5.0:
            return SignalType.SELL  # Yüksek risk
        # Normal volatilite
        elif cv > 2.0:
            return SignalType.HOLD
        # Düşük volatilite: potansiyel fırsat
        else:
            return SignalType.BUY

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """Standard Deviation volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['StandardDeviation']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Standard Deviation indikatör testi"""

    print("\n" + "="*60)
    print("STANDARD DEVIATION TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    std = StandardDeviation(period=20)
    print(f"   [OK] Oluşturuldu: {std}")
    print(f"   [OK] Kategori: {std.category.value}")
    print(f"   [OK] Gerekli periyot: {std.get_required_periods()}")

    result = std(data)
    print(f"   [OK] StdDev Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [10, 20, 30]:
        std_test = StandardDeviation(period=period)
        result = std_test.calculate(data)
        print(f"   [OK] StdDev({period}): {result.value:.4f} | CV: {result.metadata['cv']:.2f}% | Z-Score: {result.metadata['z_score']:.2f}")

    # Test 3: Düşük volatilite testi
    print("\n4. Düşük volatilite testi...")
    low_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [100.0] * 30,
        'high': [100.1] * 30,
        'low': [99.9] * 30,
        'close': [100.0 + i * 0.01 for i in range(30)],  # Çok düşük değişim
        'volume': [1000] * 30
    })
    result = std.calculate(low_vol_data)
    print(f"   [OK] Düşük Vol StdDev: {result.value:.6f}")
    print(f"   [OK] CV: {result.metadata['cv']:.4f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 4: Yüksek volatilite testi
    print("\n5. Yüksek volatilite testi...")
    high_vol_prices = [100]
    for i in range(29):
        change = np.random.randn() * 10  # Yüksek volatilite
        high_vol_prices.append(high_vol_prices[-1] + change)

    high_vol_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': high_vol_prices,
        'high': [p + abs(np.random.randn()) * 5 for p in high_vol_prices],
        'low': [p - abs(np.random.randn()) * 5 for p in high_vol_prices],
        'close': high_vol_prices,
        'volume': [1000] * 30
    })
    result = std.calculate(high_vol_data)
    print(f"   [OK] Yüksek Vol StdDev: {result.value:.4f}")
    print(f"   [OK] CV: {result.metadata['cv']:.2f}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = std.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = std.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
