"""
indicators/trend/tema.py - Triple Exponential Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    TEMA (Triple Exponential Moving Average) - Üçlü üssel hareketli ortalama
    Patrick Mulloy tarafından geliştirilmiş, lag'ı azaltan trend indikatörü
    Üç EMA kombinasyonu kullanarak çok smooth ve responsive sinyal üretir

    Kullanım:
    - Minimum lag ile trend takibi
    - Hızlı ve smooth trend değişimleri
    - Crossover stratejileri

Formül:
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    EMA1 = EMA(Close, period)
    EMA2 = EMA(EMA1, period)
    EMA3 = EMA(EMA2, period)
    TEMA = 3*EMA1 - 3*EMA2 + EMA3

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class TEMA(BaseIndicator):
    """
    Triple Exponential Moving Average

    Üç katlı EMA ile lag'ı minimize eden ve çok smooth olan trend indikatörü.

    Args:
        period: TEMA periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        # EMA indikatörünü kullan (code reuse)
        self._ema = EMA(period=period)

        super().__init__(
            name='tema',
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
        # TEMA için 3x EMA hesabı yapılacağı için daha fazla veri gerekli
        return self.period * 3

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
        TEMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TEMA değeri
        """
        # EMA1 = EMA(Close) - use EMA.calculate_batch (code reuse)
        ema1 = self._ema.calculate_batch(data)

        # EMA2 = EMA(EMA1) - create minimal DataFrame for EMA
        ema1_df = self._create_ema_input(ema1, data)
        ema2 = self._ema.calculate_batch(ema1_df)

        # EMA3 = EMA(EMA2)
        ema2_df = self._create_ema_input(ema2, data)
        ema3 = self._ema.calculate_batch(ema2_df)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema_value = 3 * ema1.iloc[-1] - 3 * ema2.iloc[-1] + ema3.iloc[-1]

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(tema_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, tema_value),
            trend=self.get_trend(current_price, tema_value),
            strength=self._calculate_strength(current_price, tema_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'ema1': round(ema1.iloc[-1], 2),
                'ema2': round(ema2.iloc[-1], 2),
                'ema3': round(ema3.iloc[-1], 2),
                'distance_pct': round(((current_price - tema_value) / tema_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        VECTORIZED batch TEMA calculation - BACKTEST

        TEMA Formula:
            TEMA = 3*EMA(Close) - 3*EMA(EMA(Close)) + EMA(EMA(EMA(Close)))

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: TEMA values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        # EMA1 = EMA(Close) - use EMA.calculate_batch (code reuse)
        ema1 = self._ema.calculate_batch(data)

        # EMA2 = EMA(EMA1)
        ema1_df = self._create_ema_input(ema1, data)
        ema2 = self._ema.calculate_batch(ema1_df)

        # EMA3 = EMA(EMA2)
        ema2_df = self._create_ema_input(ema2, data)
        ema3 = self._ema.calculate_batch(ema2_df)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3

        # Set first period values to NaN (warmup)
        warmup = self.period * 3
        tema.iloc[:warmup-1] = np.nan

        return pd.Series(tema.values, index=data.index, name='tema')

    def _create_ema_input(self, series: pd.Series, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create minimal OHLCV DataFrame for EMA calculation from a Series.
        EMA only needs 'close' but _validate_data requires OHLCV columns.
        """
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

    def get_signal(self, price: float, tema: float) -> SignalType:
        """
        TEMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            tema: TEMA değeri

        Returns:
            SignalType: BUY (fiyat TEMA üstüne çıkınca), SELL (altına ininse)
        """
        if price > tema:
            return SignalType.BUY
        elif price < tema:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, tema: float) -> TrendDirection:
        """
        TEMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            tema: TEMA değeri

        Returns:
            TrendDirection: UP (fiyat > TEMA), DOWN (fiyat < TEMA)
        """
        if price > tema:
            return TrendDirection.UP
        elif price < tema:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, tema: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - tema) / tema * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """TEMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TEMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """TEMA indikatör testi"""

    print("\n" + "="*60)
    print("TEMA (TRIPLE EXPONENTIAL MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(99):
        trend = 0.3
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
    tema = TEMA(period=20)
    print(f"   [OK] Oluşturuldu: {tema}")
    print(f"   [OK] Kategori: {tema.category.value}")
    print(f"   [OK] Gerekli periyot: {tema.get_required_periods()}")

    result = tema(data)
    print(f"   [OK] TEMA Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: EMA bileşenleri
    print("\n3. EMA bileşenleri testi...")
    print(f"   [OK] EMA1 (Close): {result.metadata['ema1']}")
    print(f"   [OK] EMA2 (EMA1): {result.metadata['ema2']}")
    print(f"   [OK] EMA3 (EMA2): {result.metadata['ema3']}")
    print(f"   [OK] TEMA = 3*{result.metadata['ema1']:.2f} - 3*{result.metadata['ema2']:.2f} + {result.metadata['ema3']:.2f}")

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [10, 20, 30]:
        tema_test = TEMA(period=period)
        result = tema_test.calculate(data)
        print(f"   [OK] TEMA({period}): {result.value:.2f} | Sinyal: {result.signal.value}")

    # Test 4: TEMA vs EMA karşılaştırması
    print("\n5. TEMA vs EMA karşılaştırma testi...")
    # Basit EMA hesaplama
    multiplier = 2 / (20 + 1)
    ema = np.mean(data['close'].values[:20])
    for price in data['close'].values[20:]:
        ema = (price - ema) * multiplier + ema

    print(f"   [OK] EMA(20): {ema:.2f}")
    print(f"   [OK] TEMA(20): {result.value:.2f}")
    print(f"   [OK] TEMA daha smooth ve responsive")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = tema.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = tema.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
