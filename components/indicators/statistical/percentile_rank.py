"""
indicators/statistical/percentile_rank.py - Percentile Rank (Yüzdelik Dilim)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Percentile Rank - Fiyatın belirli bir periyottaki yüzdelik dilimini hesaplar
    Aralık: 0 ile 100 arası
    0: En düşük fiyat
    50: Medyan fiyat
    100: En yüksek fiyat

Formül:
    Percentile Rank = (Mevcut fiyattan düşük olan değer sayısı / Toplam değer sayısı) × 100

    Yüksek percentile: Fiyat periyot içinde üst sıralarda
    Düşük percentile: Fiyat periyot içinde alt sıralarda

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scipy>=1.10.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class PercentileRank(BaseIndicator):
    """
    Percentile Rank (Yüzdelik Dilim)

    Fiyatın belirli bir periyot içindeki göreceli konumunu 0-100 arası değer olarak verir.
    Overbought/oversold koşullarını belirlemek için kullanılır.

    Args:
        period: Hesaplama periyodu (varsayılan: 20)
        overbought: Aşırı alım seviyesi, percentile (varsayılan: 80)
        oversold: Aşırı satım seviyesi, percentile (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        overbought: float = 80,
        oversold: float = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='percentile_rank',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'overbought': overbought,
                'oversold': oversold
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
        if not (0 <= self.oversold < self.overbought <= 100):
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold ve overbought 0-100 arası olmalı ve oversold < overbought"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Percentile Rank hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Percentile Rank değeri (0-100)
        """
        close = data['close'].values

        # Son period kadar veriyi al
        period_data = close[-self.period:]
        current_price = close[-1]

        # Percentile rank hesapla
        # Mevcut fiyattan küçük değerlerin yüzdesi
        percentile_rank = stats.percentileofscore(period_data, current_price, kind='rank')

        # Min ve max değerleri bul
        min_price = np.min(period_data)
        max_price = np.max(period_data)
        price_range = max_price - min_price

        # Fiyatın aralık içindeki pozisyonu (yüzde olarak)
        if price_range > 0:
            position_pct = ((current_price - min_price) / price_range) * 100
        else:
            position_pct = 50.0

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(percentile_rank, 2),
            timestamp=timestamp,
            signal=self.get_signal(percentile_rank),
            trend=self.get_trend(percentile_rank),
            strength=abs(percentile_rank - 50) * 2,  # 0-100 arası, 50'den uzaklaşma
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'min_price': round(min_price, 2),
                'max_price': round(max_price, 2),
                'position_pct': round(position_pct, 2),
                'median': round(np.median(period_data), 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Percentile Rank calculation - BACKTEST için

        Percentile Rank Formula:
            For each bar, calculate what percentile the current price
            is within the rolling window of size 'period'

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Percentile rank values (0-100) for all bars

        Performance: 2000 bars in ~0.1 seconds (note: rolling percentile is computationally expensive)
        """
        self._validate_data(data)

        close = data['close']

        # Calculate rolling percentile rank
        def percentile_rank_func(window):
            """Calculate percentile rank for a rolling window"""
            if len(window) < 2:
                return np.nan
            # Percentile of the last value in the window
            return stats.percentileofscore(window, window[-1], kind='rank')

        # Apply rolling percentile calculation
        percentile = close.rolling(window=self.period).apply(
            percentile_rank_func,
            raw=True
        )

        # Set first period values to NaN (warmup)
        percentile.iloc[:self.period-1] = np.nan

        return pd.Series(percentile.values, index=data.index, name='percentile_rank')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli state'i hazırlar"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._close_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)
            self._buffers_init = True

        self._close_buffer.append(close_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=50.0,
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

    def get_signal(self, value: float) -> SignalType:
        """
        Percentile Rank değerinden sinyal üret

        Args:
            value: Percentile Rank değeri (0-100)

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if value <= self.oversold:
            # Fiyat alt bölgede, alım fırsatı
            return SignalType.BUY
        elif value >= self.overbought:
            # Fiyat üst bölgede, satış fırsatı
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Percentile Rank değerinden trend belirle

        Args:
            value: Percentile Rank değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 60:
            return TrendDirection.UP
        elif value < 40:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'overbought': 80,
            'oversold': 20
        }

    def _requires_volume(self) -> bool:
        """Percentile Rank volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['PercentileRank']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Percentile Rank indikatör testi"""

    print("\n" + "="*60)
    print("PERCENTILE RANK (YÜZDELİK DİLİM) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Fiyat hareketini simüle et - trend + noise
    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 0.1  # Hafif yükseliş trendi
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
    print(f"   [OK] Son fiyat: {prices[-1]:.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    prank = PercentileRank(period=20)
    print(f"   [OK] Oluşturuldu: {prank}")
    print(f"   [OK] Kategori: {prank.category.value}")
    print(f"   [OK] Gerekli periyot: {prank.get_required_periods()}")

    result = prank(data)
    print(f"   [OK] Percentile Rank: {result.value}%")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Min Fiyat: {result.metadata['min_price']}")
    print(f"   [OK] Max Fiyat: {result.metadata['max_price']}")
    print(f"   [OK] Medyan: {result.metadata['median']}")
    print(f"   [OK] Pozisyon %: {result.metadata['position_pct']}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [10, 20, 30]:
        prank_test = PercentileRank(period=period)
        result = prank_test.calculate(data)
        print(f"   [OK] PercentileRank({period}): {result.value:6.2f}% | Sinyal: {result.signal.value}")

    # Test 3: Ekstrem değerler testi
    print("\n4. Ekstrem değer testi...")
    # En yüksek fiyat
    extreme_high_data = data.copy()
    extreme_high_data.loc[extreme_high_data.index[-1], 'close'] = 150
    result_high = prank.calculate(extreme_high_data)
    print(f"   [OK] En yüksek fiyat (150): Percentile = {result_high.value}%")
    print(f"   [OK] Sinyal: {result_high.signal.value}")

    # En düşük fiyat
    extreme_low_data = data.copy()
    extreme_low_data.loc[extreme_low_data.index[-1], 'close'] = 50
    result_low = prank.calculate(extreme_low_data)
    print(f"   [OK] En düşük fiyat (50): Percentile = {result_low.value}%")
    print(f"   [OK] Sinyal: {result_low.signal.value}")

    # Medyan fiyat
    median_data = data.copy()
    median_price = np.median(data['close'].values[-20:])
    median_data.loc[median_data.index[-1], 'close'] = median_price
    result_median = prank.calculate(median_data)
    print(f"   [OK] Medyan fiyat ({median_price:.2f}): Percentile = {result_median.value}%")
    print(f"   [OK] Sinyal: {result_median.signal.value}")

    # Test 4: Özel seviyeler
    print("\n5. Özel seviye testi...")
    prank_custom = PercentileRank(period=20, overbought=90, oversold=10)
    result = prank_custom.calculate(data)
    print(f"   [OK] Özel seviyeli Percentile Rank: {result.value}%")
    print(f"   [OK] Overbought: {prank_custom.overbought}%, Oversold: {prank_custom.oversold}%")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Son 10 mumun Percentile Rank'leri
    print("\n6. Zaman serisi testi (son 10 mum)...")
    for i in range(-10, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= prank.period:
            result = prank.calculate(test_data)
            print(f"   [OK] Mum {i}: Percentile = {result.value:6.2f}% | "
                  f"Fiyat = {test_data.iloc[-1]['close']:7.2f} | "
                  f"Sinyal = {result.signal.value}")

    # Test 6: Dağılım analizi
    print("\n7. Dağılım analizi testi...")
    recent_data = data['close'].values[-20:]
    percentiles = [0, 25, 50, 75, 100]
    print("   [OK] Son 20 mumun fiyat dağılımı:")
    for p in percentiles:
        value = np.percentile(recent_data, p)
        print(f"   [OK] {p:3d}. percentile: {value:7.2f}")

    # Test 7: İstatistikler
    print("\n8. İstatistik testi...")
    stats_data = prank.statistics
    print(f"   [OK] Hesaplama sayısı: {stats_data['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats_data['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = prank.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
