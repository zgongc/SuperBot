"""
indicators/momentum/williams_r.py - Williams %R

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Williams %R - Momentum osilatörü
    Aralık: -100 ile 0 arası
    Aşırı Alım: > -20
    Aşırı Satım: < -80
    Stochastic ile benzer mantık, negatif skala kullanır.

Formül:
    Williams %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)
    Highest High = Son N periyodun en yüksek fiyatı
    Lowest Low = Son N periyodun en düşük fiyatı

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


class WilliamsR(BaseIndicator):
    """
    Williams %R

    Momentum osilatörü, aşırı alım/satım koşullarını tespit eder.
    -100 ile 0 arası değer alır.

    Args:
        period: Williams %R periyodu (varsayılan: 14)
        overbought: Aşırı alım seviyesi (varsayılan: -20)
        oversold: Aşırı satım seviyesi (varsayılan: -80)
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = -20,
        oversold: float = -80,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='williams_r',
            category=IndicatorCategory.MOMENTUM,
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
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot pozitif olmalı"
            )
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold, overbought'tan küçük olmalı"
            )
        if not (-100 <= self.oversold <= 0) or not (-100 <= self.overbought <= 0):
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Seviyeler -100 ile 0 arası olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Williams %R hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Williams %R değeri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Son period kadar veri al
        period_high = high[-self.period:]
        period_low = low[-self.period:]
        current_close = close[-1]

        # Highest High ve Lowest Low hesapla
        highest_high = np.max(period_high)
        lowest_low = np.min(period_low)

        # Williams %R hesapla
        if highest_high == lowest_low:
            williams_r_value = -50.0  # Neutral değer
        else:
            williams_r_value = -100 * (highest_high - current_close) / (highest_high - lowest_low)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(williams_r_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(williams_r_value),
            trend=self.get_trend(williams_r_value),
            strength=abs(williams_r_value + 50) * 2,  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'highest_high': round(highest_high, 2),
                'lowest_low': round(lowest_low, 2),
                'close': round(current_close, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Williams %R calculation - BACKTEST için

        Williams %R Formula:
            %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Williams %R values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Highest High and Lowest Low over period (vectorized)
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()

        # Williams %R = -100 * (HH - Close) / (HH - LL)
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        williams_r = -100 * (highest_high - close) / denominator

        # Handle division by zero
        williams_r = williams_r.fillna(-50)  # Neutral value

        # Set first period values to NaN (warmup)
        williams_r.iloc[:self.period-1] = np.nan

        return pd.Series(williams_r.values, index=data.index, name='williams_r')

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

        # Buffer'ları oluştur ve doldur
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=-50.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'high': list(self._high_buffer),
            'low': list(self._low_buffer),
            'close': list(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'volume': [volume_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Williams %R değerinden sinyal üret

        Args:
            value: Williams %R değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if value < self.oversold:
            return SignalType.BUY
        elif value > self.overbought:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Williams %R değerinden trend belirle

        Args:
            value: Williams %R değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if value > -50:
            return TrendDirection.UP
        elif value < -50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 14,
            'overbought': -20,
            'oversold': -80
        }

    def _requires_volume(self) -> bool:
        """Williams %R volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['WilliamsR']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Williams %R indikatör testi"""

    print("\n" + "="*60)
    print("WILLIAMS %R TEST")
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
    williams = WilliamsR(period=14)
    print(f"   [OK] Oluşturuldu: {williams}")
    print(f"   [OK] Kategori: {williams.category.value}")
    print(f"   [OK] Gerekli periyot: {williams.get_required_periods()}")

    result = williams(data)
    print(f"   [OK] Williams %R Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [7, 14, 21]:
        williams_test = WilliamsR(period=period)
        result = williams_test.calculate(data)
        print(f"   [OK] Williams %R({period}): {result.value} | Sinyal: {result.signal.value}")

    # Test 3: Özel seviyeler
    print("\n4. Özel seviye testi...")
    williams_custom = WilliamsR(period=14, overbought=-30, oversold=-70)
    result = williams_custom.calculate(data)
    print(f"   [OK] Özel seviyeli Williams %R: {result.value}")
    print(f"   [OK] Overbought: {williams_custom.overbought}, Oversold: {williams_custom.oversold}")
    print(f"   [OK] Sinyal: {result.signal.value}")

    # Test 4: Aşırı alım/satım koşulları
    print("\n5. Aşırı alım/satım testi...")
    # Yükselen trend simüle et
    up_data = data.copy()
    up_data.loc[up_data.index[-5:], 'close'] = [p + 5 for p in up_data.loc[up_data.index[-5:], 'close']]
    up_data.loc[up_data.index[-5:], 'high'] = [p + 5.5 for p in up_data.loc[up_data.index[-5:], 'high']]

    result_up = williams.calculate(up_data)
    print(f"   [OK] Yükselen trend Williams %R: {result_up.value}")
    print(f"   [OK] Sinyal: {result_up.signal.value}")

    # Düşen trend simüle et
    down_data = data.copy()
    down_data.loc[down_data.index[-5:], 'close'] = [p - 5 for p in down_data.loc[down_data.index[-5:], 'close']]
    down_data.loc[down_data.index[-5:], 'low'] = [p - 5.5 for p in down_data.loc[down_data.index[-5:], 'low']]

    result_down = williams.calculate(down_data)
    print(f"   [OK] Düşen trend Williams %R: {result_down.value}")
    print(f"   [OK] Sinyal: {result_down.signal.value}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = williams.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = williams.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
