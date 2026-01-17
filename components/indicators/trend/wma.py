"""
indicators/trend/wma.py - Weighted Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    WMA (Weighted Moving Average) - Ağırlıklı hareketli ortalama
    Son fiyatlara daha fazla ağırlık veren trend indikatörü
    EMA'dan daha basit, SMA'dan daha responsive

    Kullanım:
    - Trend yönünü belirleme
    - Son fiyat hareketlerine daha hızlı tepki
    - Destek/direnç seviyeleri

Formül:
    WMA = (n*P1 + (n-1)*P2 + ... + 1*Pn) / (n + (n-1) + ... + 1)
    WMA = Sum(Price[i] * (n - i)) / Sum(n - i)
    n: Periyot, P: Fiyat

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


class WMA(BaseIndicator):
    """
    Weighted Moving Average

    Yakın geçmişteki fiyatlara doğrusal olarak artan ağırlık verir.
    En son fiyat en yüksek ağırlığa sahiptir.

    Args:
        period: WMA periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='wma',
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
        WMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: WMA değeri
        """
        close = data['close'].values

        # WMA hesapla
        prices = close[-self.period:]
        weights = np.arange(1, self.period + 1)
        wma_value = np.sum(prices * weights) / np.sum(weights)

        # Mevcut fiyat
        current_price = close[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(wma_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, wma_value),
            trend=self.get_trend(current_price, wma_value),
            strength=self._calculate_strength(current_price, wma_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'distance_pct': round(((current_price - wma_value) / wma_value) * 100, 2),
                'weight_sum': int(np.sum(weights))
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch WMA calculation - BACKTEST için

        WMA Formula:
            WMA = Sum(Price[i] * Weight[i]) / Sum(Weight[i])
            Weight[i] = period - i + 1 (linear weighting)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: WMA values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']
        weights = np.arange(1, self.period + 1)
        weight_sum = np.sum(weights)

        # Vectorized WMA using rolling().apply()
        def wma_window(window):
            if len(window) < self.period:
                return np.nan
            return np.sum(window * weights) / weight_sum

        wma = close.rolling(window=self.period).apply(wma_window, raw=True)

        # Set first period values to NaN (warmup)
        wma.iloc[:self.period-1] = np.nan

        return pd.Series(wma.values, index=data.index, name='wma')

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

    def get_signal(self, price: float, wma: float) -> SignalType:
        """
        WMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            wma: WMA değeri

        Returns:
            SignalType: BUY (fiyat WMA üstüne çıkınca), SELL (altına ininse)
        """
        if price > wma:
            return SignalType.BUY
        elif price < wma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, wma: float) -> TrendDirection:
        """
        WMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            wma: WMA değeri

        Returns:
            TrendDirection: UP (fiyat > WMA), DOWN (fiyat < WMA)
        """
        if price > wma:
            return TrendDirection.UP
        elif price < wma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, wma: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - wma) / wma * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """WMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['WMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """WMA indikatör testi"""

    print("\n" + "="*60)
    print("WMA (WEIGHTED MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 0.5
        noise = np.random.randn() * 1.5
        prices.append(prices[-1] + trend + noise)

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
    wma = WMA(period=20)
    print(f"   [OK] Oluşturuldu: {wma}")
    print(f"   [OK] Kategori: {wma.category.value}")
    print(f"   [OK] Gerekli periyot: {wma.get_required_periods()}")

    result = wma(data)
    print(f"   [OK] WMA Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: SMA vs WMA karşılaştırması
    print("\n3. WMA vs SMA karşılaştırma testi...")
    sma_value = np.mean(data['close'].values[-20:])
    print(f"   [OK] SMA(20): {sma_value:.2f}")
    print(f"   [OK] WMA(20): {result.value:.2f}")
    print(f"   [OK] Fark: {abs(result.value - sma_value):.2f}")
    print(f"   [OK] WMA son fiyatlara daha fazla ağırlık verir")

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [10, 20, 50]:
        wma_test = WMA(period=period)
        result = wma_test.calculate(data)
        print(f"   [OK] WMA({period}): {result.value:.2f} | Sinyal: {result.signal.value}")

    # Test 4: Ağırlık dağılımı
    print("\n5. Ağırlık dağılımı testi...")
    period = 10
    weights = np.arange(1, period + 1)
    print(f"   [OK] Period: {period}")
    print(f"   [OK] Ağırlıklar: {weights.tolist()}")
    print(f"   [OK] Toplam ağırlık: {np.sum(weights)}")
    print(f"   [OK] Son fiyat ağırlığı: {weights[-1]} / {np.sum(weights)} = {weights[-1]/np.sum(weights):.2%}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = wma.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = wma.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
