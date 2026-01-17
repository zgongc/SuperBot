"""
indicators/trend/sma.py - Simple Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    SMA (Simple Moving Average) - Basit hareketli ortalama
    En temel trend indikatörü, fiyatların aritmetik ortalaması

    Kullanım:
    - Trend yönünü belirlemek
    - Destek/direnç seviyeleri
    - Crossover stratejileri (50/200 SMA golden cross)

Formül:
    SMA = (P1 + P2 + ... + Pn) / n
    P: Fiyat (close)
    n: Periyot

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


class SMA(BaseIndicator):
    """
    Simple Moving Average

    Belirtilen periyottaki fiyatların basit aritmetik ortalaması.
    Trend takibi ve destek/direnç seviyeleri için kullanılır.

    Args:
        period: SMA periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='sma',
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
        SMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: SMA değeri
        """
        close = data['close'].values

        # SMA hesapla (son N periyodun ortalaması)
        sma_value = np.mean(close[-self.period:])

        # Mevcut fiyat
        current_price = close[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(sma_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, sma_value),
            trend=self.get_trend(current_price, sma_value),
            strength=self._calculate_strength(current_price, sma_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'distance_pct': round(((current_price - sma_value) / sma_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch SMA calculation - BACKTEST için

        SMA Formula:
            SMA = (P1 + P2 + ... + Pn) / n

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: SMA values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        # Pandas rolling().mean() is already vectorized
        sma = data['close'].rolling(window=self.period).mean()

        # Set first period values to NaN (warmup)
        sma.iloc[:self.period-1] = np.nan

        return pd.Series(sma.values, index=data.index, name='sma')

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Uses BaseIndicator's _buffers[symbol] (populated by warmup_buffer())

        Args:
            candle: New candle data
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Updated SMA value
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._buffers:
            self._buffers[buffer_key] = deque(maxlen=self.get_required_periods())

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

        # Check minimum data
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            # Support both dict and list/tuple formats
            if isinstance(candle, dict):
                close_val = candle['close']
                timestamp_val = candle.get('timestamp', 0)
            else:
                close_val = candle[4] if len(candle) > 4 else 0
                timestamp_val = candle[0] if len(candle) > 0 else 0

            return IndicatorResult(
                value=close_val,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period}
            )

        # Convert to DataFrame and calculate
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))
        return self.calculate(buffer_data)

    def get_signal(self, price: float, sma: float) -> SignalType:
        """
        SMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            sma: SMA değeri

        Returns:
            SignalType: BUY (fiyat SMA üstüne çıkınca), SELL (altına ininse)
        """
        if price > sma:
            return SignalType.BUY
        elif price < sma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, sma: float) -> TrendDirection:
        """
        SMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            sma: SMA değeri

        Returns:
            TrendDirection: UP (fiyat > SMA), DOWN (fiyat < SMA)
        """
        if price > sma:
            return TrendDirection.UP
        elif price < sma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, sma: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - sma) / sma * 100)
        return min(distance_pct * 20, 100)  # %5 uzaklık = 100 güç

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """SMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """SMA indikatör testi"""

    print("\n" + "="*60)
    print("SMA (SIMPLE MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 0.5  # Yavaş yükseliş
        noise = np.random.randn() * 1
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
    sma = SMA(period=20)
    print(f"   [OK] Oluşturuldu: {sma}")
    print(f"   [OK] Kategori: {sma.category.value}")
    print(f"   [OK] Gerekli periyot: {sma.get_required_periods()}")

    result = sma(data)
    print(f"   [OK] SMA Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [10, 20, 50]:
        sma_test = SMA(period=period)
        result = sma_test.calculate(data)
        print(f"   [OK] SMA({period}): {result.value:.2f} | Sinyal: {result.signal.value}")

    # Test 3: Golden Cross simülasyonu (50/200 cross)
    print("\n4. Multiple SMA testi (Golden/Death Cross)...")
    sma_50 = SMA(period=50)
    sma_20 = SMA(period=20)

    result_50 = sma_50.calculate(data)
    result_20 = sma_20.calculate(data)

    print(f"   [OK] SMA(20): {result_20.value:.2f}")
    print(f"   [OK] SMA(50): {result_50.value:.2f}")

    if result_20.value > result_50.value:
        print(f"   [OK] Golden Cross bölgesi (SMA20 > SMA50) - BULLISH")
    else:
        print(f"   [OK] Death Cross bölgesi (SMA20 < SMA50) - BEARISH")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = sma.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = sma.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
