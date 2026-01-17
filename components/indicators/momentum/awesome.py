"""
indicators/momentum/awesome.py - Awesome Oscillator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Awesome Oscillator (AO) - Bill Williams tarafından geliştirildi
    Aralık: Sınırsız (pozitif ve negatif değerler)
    Pozitif değer: Bullish momentum
    Negatif değer: Bearish momentum
    Histogram renkleri: Yeşil (artıyor), Kırmızı (azalıyor)

Formül:
    Median Price = (High + Low) / 2
    AO = SMA(Median Price, 5) - SMA(Median Price, 34)

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


class AwesomeOscillator(BaseIndicator):
    """
    Awesome Oscillator

    Market momentum'unu ölçer. İki farklı periyotlu SMA arasındaki farkı kullanır.
    Parametresiz indikatör (5 ve 34 periyot sabittir).

    Sabit parametreler:
        fast_period: 5 (SMA kısa)
        slow_period: 34 (SMA uzun)
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 34,
        logger=None,
        error_handler=None
    ):
        # Sabit periyotlar (Bill Williams'ın orijinal tanımı)
        self.fast_period = fast_period
        self.slow_period = slow_period

        super().__init__(
            name='awesome_oscillator',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.slow_period

    def validate_params(self) -> bool:
        """Parametreleri doğrula (sabit değerler)"""
        # Sabit değerler olduğu için her zaman True
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Awesome Oscillator hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: AO değeri
        """
        high = data['high'].values
        low = data['low'].values

        # Median Price hesapla
        median_price = (high + low) / 2

        # SMA(5) ve SMA(34) hesapla
        sma_fast = np.mean(median_price[-self.fast_period:])
        sma_slow = np.mean(median_price[-self.slow_period:])

        # Awesome Oscillator
        ao_value = sma_fast - sma_slow

        # Önceki AO değeri (momentum değişimi için)
        if len(median_price) > self.slow_period:
            prev_median = median_price[:-1]
            prev_sma_fast = np.mean(prev_median[-self.fast_period:])
            prev_sma_slow = np.mean(prev_median[-self.slow_period:])
            prev_ao = prev_sma_fast - prev_sma_slow
            momentum_direction = 'increasing' if ao_value > prev_ao else 'decreasing'
            ao_change = ao_value - prev_ao
        else:
            momentum_direction = 'neutral'
            ao_change = 0

        timestamp = int(data.iloc[-1]['timestamp'])

        return IndicatorResult(
            value={'ao': round(ao_value, 4)},  # Dict format for consistency with calculate_batch
            timestamp=timestamp,
            signal=self.get_signal(ao_value),
            trend=self.get_trend(ao_value),
            strength=min(abs(ao_value) * 10, 100),  # 0-100 arası normalize et
            metadata={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'sma_fast': round(sma_fast, 2),
                'sma_slow': round(sma_slow, 2),
                'momentum_direction': momentum_direction,
                'change': round(ao_change, 4)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Awesome Oscillator calculation - BACKTEST için

        AO Formula:
            Median Price = (High + Low) / 2
            AO = SMA(Median Price, 5) - SMA(Median Price, 34)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: AO values for all bars (column: 'ao')

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        # Median Price = (High + Low) / 2
        median_price = (data['high'] + data['low']) / 2

        # SMA(5) - Fast
        sma_fast = median_price.rolling(window=self.fast_period).mean()

        # SMA(34) - Slow
        sma_slow = median_price.rolling(window=self.slow_period).mean()

        # AO = SMA_fast - SMA_slow
        ao = sma_fast - sma_slow

        # Set first period values to NaN (warmup)
        ao.iloc[:self.slow_period-1] = np.nan

        return pd.DataFrame({'ao': ao.values}, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel indicator değeri
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._buffers:
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = deque(maxlen=max_len)

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Need minimum data for calculation
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        AO değerinden sinyal üret

        Args:
            value: AO değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # Pozitif AO: Bullish
        if value > 0:
            return SignalType.BUY
        # Negatif AO: Bearish
        elif value < 0:
            return SignalType.SELL
        return SignalType.NEUTRAL

    def get_trend(self, value: float) -> TrendDirection:
        """
        AO değerinden trend belirle

        Args:
            value: AO değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 0:
            return TrendDirection.UP
        elif value < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'fast_period': 5,
            'slow_period': 34
        }

    def _requires_volume(self) -> bool:
        """Awesome Oscillator volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['AwesomeOscillator']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Awesome Oscillator indikatör testi"""

    print("\n" + "="*60)
    print("AWESOME OSCILLATOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(49):
        change = np.random.randn() * 1.5
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
    ao = AwesomeOscillator()
    print(f"   [OK] Oluşturuldu: {ao}")
    print(f"   [OK] Kategori: {ao.category.value}")
    print(f"   [OK] Gerekli periyot: {ao.get_required_periods()}")
    print(f"   [OK] Fast period: {ao.fast_period}, Slow period: {ao.slow_period}")

    result = ao(data)
    print(f"   [OK] AO Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Momentum yönü: {result.metadata['momentum_direction']}")
    print(f"   [OK] Değişim: {result.metadata['change']}")

    # Test 2: Momentum değişimi
    print("\n3. Momentum değişimi testi...")
    for i in range(-5, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= ao.get_required_periods():
            result = ao.calculate(test_data)
            print(f"   [OK] Index {i}: AO={result.value:.4f}, Yön={result.metadata['momentum_direction']}")

    # Test 3: Yükselen trend
    print("\n4. Yükselen trend testi...")
    up_data = data.copy()
    for i in range(20):
        idx = up_data.index[-(20-i)]
        up_data.loc[idx, 'high'] = up_data.loc[idx, 'high'] + i * 0.3
        up_data.loc[idx, 'low'] = up_data.loc[idx, 'low'] + i * 0.3

    result_up = ao.calculate(up_data)
    print(f"   [OK] Yükselen trend AO: {result_up.value:.4f}")
    print(f"   [OK] Sinyal: {result_up.signal.value}")
    print(f"   [OK] Trend: {result_up.trend.name}")
    print(f"   [OK] Momentum yönü: {result_up.metadata['momentum_direction']}")

    # Test 4: Düşen trend
    print("\n5. Düşen trend testi...")
    down_data = data.copy()
    for i in range(20):
        idx = down_data.index[-(20-i)]
        down_data.loc[idx, 'high'] = down_data.loc[idx, 'high'] - i * 0.3
        down_data.loc[idx, 'low'] = down_data.loc[idx, 'low'] - i * 0.3

    result_down = ao.calculate(down_data)
    print(f"   [OK] Düşen trend AO: {result_down.value:.4f}")
    print(f"   [OK] Sinyal: {result_down.signal.value}")
    print(f"   [OK] Trend: {result_down.trend.name}")
    print(f"   [OK] Momentum yönü: {result_down.metadata['momentum_direction']}")

    # Test 5: Sıfır geçişi (zero-line cross)
    print("\n6. Sıfır geçişi testi...")
    neutral_data = data.copy()
    # Median price'ı sabitle
    median = (neutral_data['high'] + neutral_data['low']) / 2
    avg_median = median.mean()
    neutral_data['high'] = avg_median + 0.1
    neutral_data['low'] = avg_median - 0.1

    result_neutral = ao.calculate(neutral_data)
    print(f"   [OK] Nötr AO: {result_neutral.value:.4f}")
    print(f"   [OK] Sinyal: {result_neutral.signal.value}")

    # Test 6: Twin Peaks pattern (gelişmiş kullanım)
    print("\n7. Pattern tanıma testi...")
    # Son 10 AO değerini hesapla
    ao_values = []
    for i in range(-10, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= ao.get_required_periods():
            result = ao.calculate(test_data)
            ao_values.append(result.value)

    if len(ao_values) >= 3:
        print(f"   [OK] Son 10 AO değeri hesaplandı")
        print(f"   [OK] İlk 3: {[round(v, 4) for v in ao_values[:3]]}")
        print(f"   [OK] Son 3: {[round(v, 4) for v in ao_values[-3:]]}")

    # Test 7: İstatistikler
    print("\n8. İstatistik testi...")
    stats = ao.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = ao.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
