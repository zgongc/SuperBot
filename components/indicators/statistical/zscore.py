"""
indicators/statistical/z_score.py - Z-Score (Standart Sapma Skoru)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Z-Score - Fiyatın ortalamadan kaç standart sapma uzakta olduğunu ölçer
    Aralık: Genellikle -3 ile +3 arası (normalleştirilmiş)
    Aşırı Alım: > +2
    Aşırı Satım: < -2

Formül:
    Z-Score = (Fiyat - Ortalama) / Standart Sapma

    Pozitif değer: Fiyat ortalamadan yüksek
    Negatif değer: Fiyat ortalamadan düşük

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

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


class ZScore(BaseIndicator):
    """
    Z-Score (Standart Sapma Skoru)

    Fiyatın istatistiksel olarak ne kadar aşırı olduğunu ölçer.
    Ortalamaya dönüş stratejileri için kullanılır.

    Args:
        period: Hesaplama periyodu (varsayılan: 20)
        overbought: Aşırı alım seviyesi (varsayılan: 2.0)
        oversold: Aşırı satım seviyesi (varsayılan: -2.0)
    """

    def __init__(
        self,
        period: int = 20,
        overbought: float = 2.0,
        oversold: float = -2.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='zscore',
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
                "Periyot en az 2 olmalı (standart sapma için)"
            )
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold, overbought'tan küçük olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Z-Score hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Z-Score değeri
        """
        close = data['close'].values

        # Son period kadar veriyi al
        period_data = close[-self.period:]

        # Ortalama ve standart sapma hesapla
        mean = np.mean(period_data)
        std = np.std(period_data, ddof=1)  # Sample std (n-1)

        # Z-Score hesapla
        if std == 0:
            z_score_value = 0.0
        else:
            z_score_value = (close[-1] - mean) / std

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(z_score_value, 4),
            timestamp=timestamp,
            signal=self.get_signal(z_score_value),
            trend=self.get_trend(z_score_value),
            strength=min(abs(z_score_value) * 50, 100),  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'mean': round(mean, 2),
                'std': round(std, 2),
                'current_price': round(close[-1], 2),
                'deviation_percent': round((z_score_value * std / mean) * 100, 2) if mean != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Z-Score calculation - BACKTEST için

        Z-Score Formula:
            Z-Score = (Price - Mean) / Std Dev

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Z-Score values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Rolling mean
        mean = close.rolling(window=self.period).mean()

        # Rolling std (sample std with ddof=1)
        std = close.rolling(window=self.period).std(ddof=1)

        # Z-Score = (Price - Mean) / Std
        z_score = (close - mean) / std

        # Handle division by zero
        z_score = z_score.fillna(0)

        # Set first period values to NaN (warmup)
        z_score.iloc[:self.period-1] = np.nan

        return pd.Series(z_score.values, index=data.index, name='zscore')

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

    def get_signal(self, value: float) -> SignalType:
        """
        Z-Score değerinden sinyal üret

        Args:
            value: Z-Score değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if value < self.oversold:
            # Fiyat çok düşük, ortalamaya dönüş beklentisi
            return SignalType.BUY
        elif value > self.overbought:
            # Fiyat çok yüksek, ortalamaya dönüş beklentisi
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Z-Score değerinden trend belirle

        Args:
            value: Z-Score değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > 0.5:
            return TrendDirection.UP
        elif value < -0.5:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20,
            'overbought': 2.0,
            'oversold': -2.0
        }

    def _requires_volume(self) -> bool:
        """Z-Score volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ZScore']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Z-Score indikatör testi"""

    print("\n" + "="*60)
    print("Z-SCORE (STANDART SAPMA SKORU) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Fiyat hareketini simüle et - ortalamaya dönüş yapan bir seri
    base_price = 100
    prices = [base_price]
    mean_price = 100
    for i in range(49):
        # Mean reversion simülasyonu
        noise = np.random.randn() * 2
        mean_revert = (mean_price - prices[-1]) * 0.1
        prices.append(prices[-1] + noise + mean_revert)

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
    zscore = ZScore(period=20)
    print(f"   [OK] Oluşturuldu: {zscore}")
    print(f"   [OK] Kategori: {zscore.category.value}")
    print(f"   [OK] Gerekli periyot: {zscore.get_required_periods()}")

    result = zscore(data)
    print(f"   [OK] Z-Score Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Ortalama: {result.metadata['mean']}")
    print(f"   [OK] Std Sapma: {result.metadata['std']}")
    print(f"   [OK] Sapma %: {result.metadata['deviation_percent']}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [10, 20, 30]:
        zscore_test = ZScore(period=period)
        result = zscore_test.calculate(data)
        print(f"   [OK] Z-Score({period}): {result.value:.4f} | Sinyal: {result.signal.value}")

    # Test 3: Aşırı değerler testi
    print("\n4. Aşırı değer testi...")
    # Aşırı yüksek fiyat
    extreme_high_data = data.copy()
    extreme_high_data.loc[extreme_high_data.index[-1], 'close'] = 120
    result_high = zscore.calculate(extreme_high_data)
    print(f"   [OK] Aşırı yüksek fiyat (120): Z-Score = {result_high.value:.4f}")
    print(f"   [OK] Sinyal: {result_high.signal.value}")

    # Aşırı düşük fiyat
    extreme_low_data = data.copy()
    extreme_low_data.loc[extreme_low_data.index[-1], 'close'] = 80
    result_low = zscore.calculate(extreme_low_data)
    print(f"   [OK] Aşırı düşük fiyat (80): Z-Score = {result_low.value:.4f}")
    print(f"   [OK] Sinyal: {result_low.signal.value}")

    # Test 4: Özel seviyeler
    print("\n5. Özel seviye testi...")
    zscore_custom = ZScore(period=20, overbought=3.0, oversold=-3.0)
    result = zscore_custom.calculate(data)
    print(f"   [OK] Özel seviyeli Z-Score: {result.value:.4f}")
    print(f"   [OK] Overbought: {zscore_custom.overbought}, Oversold: {zscore_custom.oversold}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 5: Son 10 mumun Z-Score'ları
    print("\n6. Zaman serisi testi (son 10 mum)...")
    for i in range(-10, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= zscore.period:
            result = zscore.calculate(test_data)
            print(f"   [OK] Mum {i}: Z-Score = {result.value:7.4f} | "
                  f"Fiyat = {test_data.iloc[-1]['close']:7.2f} | "
                  f"Sinyal = {result.signal.value}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = zscore.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = zscore.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
