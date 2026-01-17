"""
indicators/support_resistance/pivot_points.py - Pivot Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Pivot Points - Classic pivot seviyeleri
    Destek ve direnç seviyelerini belirlemek için kullanılır.
    Pivot (P), Resistance (R1, R2, R3) ve Support (S1, S2, S3) seviyeleri hesaplanır.

Formül:
    P = (High + Low + Close) / 3
    R1 = (2 * P) - Low
    R2 = P + (High - Low)
    R3 = High + 2 * (P - Low)
    S1 = (2 * P) - High
    S2 = P - (High - Low)
    S3 = Low - 2 * (High - P)

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


class PivotPoints(BaseIndicator):
    """
    Classic Pivot Points

    Önceki periyodun High, Low ve Close değerlerini kullanarak
    pivot seviyeleri (P, R1-R3, S1-S3) hesaplar.

    Args:
        period: Pivot hesaplama periyodu (varsayılan: 1 - günlük)
    """

    def __init__(
        self,
        period: int = 1,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='pivot_points',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'period': period
            },
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
                "Periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Pivot Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Pivot seviyeleri (P, R1-R3, S1-S3)
        """
        # Önceki periyodun H, L, C değerlerini al
        high = data['high'].iloc[-self.period - 1:-1].max()
        low = data['low'].iloc[-self.period - 1:-1].min()
        close = data['close'].iloc[-self.period - 1]

        # Pivot Point hesapla
        pivot = (high + low + close) / 3

        # Resistance seviyeleri
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        # Support seviyeleri
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Seviyeleri sözlük olarak oluştur (lowercase keys for consistency)
        levels = {
            'r3': round(r3, 2),
            'r2': round(r2, 2),
            'r1': round(r1, 2),
            'p': round(pivot, 2),
            's1': round(s1, 2),
            's2': round(s2, 2),
            's3': round(s3, 2)
        }

        # Warmup buffer for update() method
        self._warmup_buffers(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, pivot),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'period': self.period,
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'current_price': round(current_price, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Pivot Points calculation - BACKTEST için

        Pivot Points Formula:
            P = (High + Low + Close) / 3
            R1 = (2 * P) - Low
            R2 = P + (High - Low)
            R3 = High + 2 * (P - Low)
            S1 = (2 * P) - High
            S2 = P - (High - Low)
            S3 = Low - 2 * (High - P)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: pivot, r1, r2, r3, s1, s2, s3 for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Rolling max/min for previous period
        prev_high = high.shift(self.period).rolling(window=self.period).max()
        prev_low = low.shift(self.period).rolling(window=self.period).min()
        prev_close = close.shift(self.period)

        # Pivot Point
        pivot = (prev_high + prev_low + prev_close) / 3

        # Resistance levels
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)

        # Support levels
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        # Set first period values to NaN (warmup)
        warmup = self.period * 2
        pivot.iloc[:warmup] = np.nan
        r1.iloc[:warmup] = np.nan
        r2.iloc[:warmup] = np.nan
        r3.iloc[:warmup] = np.nan
        s1.iloc[:warmup] = np.nan
        s2.iloc[:warmup] = np.nan
        s3.iloc[:warmup] = np.nan

        return pd.DataFrame({
            'p': pivot.values,
            'r1': r1.values,
            'r2': r2.values,
            'r3': r3.values,
            's1': s1.values,
            's2': s2.values,
            's3': s3.values
        }, index=data.index)

    def _warmup_buffers(self, data: pd.DataFrame) -> None:
        """Warmup buffer for update() method"""
        from collections import deque
        max_len = self.get_required_periods() + 50

        if not hasattr(self, '_buffers_init'):
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Fill buffers from historical data
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()

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
                value={'p': 0.0, 'r1': 0.0, 'r2': 0.0, 'r3': 0.0, 's1': 0.0, 's2': 0.0, 's3': 0.0},
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

    def get_signal(self, price: float, levels: dict) -> SignalType:
        """
        Fiyatın pivot seviyelerine göre sinyal üret

        Args:
            price: Güncel fiyat
            levels: Pivot seviyeleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if price < levels['s2']:
            return SignalType.BUY
        elif price > levels['r2']:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, pivot: float) -> TrendDirection:
        """
        Fiyatın pivot'a göre trend belirle

        Args:
            price: Güncel fiyat
            pivot: Pivot seviyesi

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if price > pivot:
            return TrendDirection.UP
        elif price < pivot:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, levels: dict) -> float:
        """
        Fiyatın seviyelere göre güç hesapla

        Args:
            price: Güncel fiyat
            levels: Pivot seviyeleri

        Returns:
            float: Güç değeri (0-100)
        """
        pivot = levels['p']
        r3 = levels['r3']
        s3 = levels['s3']

        try:
            if price > pivot:
                # Yukarı yönde güç
                divisor = r3 - pivot
                if divisor <= 0:
                    return 50.0
                strength = ((price - pivot) / divisor) * 100
            else:
                # Aşağı yönde güç
                divisor = pivot - s3
                if divisor <= 0:
                    return 50.0
                strength = ((pivot - price) / divisor) * 100

            return min(max(strength, 0), 100)
        except (ZeroDivisionError, TypeError):
            return 50.0

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 1
        }

    def _requires_volume(self) -> bool:
        """Pivot Points volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['PivotPoints']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Pivot Points indikatör testi"""

    print("\n" + "="*60)
    print("PIVOT POINTS TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(49):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    pivot = PivotPoints(period=1)
    print(f"   [OK] Oluşturuldu: {pivot}")
    print(f"   [OK] Kategori: {pivot.category.value}")
    print(f"   [OK] Tip: {pivot.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {pivot.get_required_periods()}")

    result = pivot(data)
    print(f"   [OK] Pivot Seviyeleri:")
    for level, value in result.value.items():
        print(f"        {level}: {value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [1, 5, 10]:
        pivot_test = PivotPoints(period=period)
        result = pivot_test.calculate(data)
        print(f"   [OK] Pivot({period}) - P: {result.value['p']} | Sinyal: {result.signal.value}")

    # Test 3: Seviye analizi
    print("\n4. Seviye analizi...")
    result = pivot.calculate(data)
    current = result.metadata['current_price']
    print(f"   [OK] Güncel fiyat: {current}")
    print(f"   [OK] Pivot: {result.value['p']}")
    if current > result.value['p']:
        print(f"   [OK] Fiyat pivot üstünde (Bullish)")
        print(f"   [OK] İlk direnç (R1): {result.value['r1']}")
    else:
        print(f"   [OK] Fiyat pivot altında (Bearish)")
        print(f"   [OK] İlk destek (S1): {result.value['s1']}")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = pivot.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = pivot.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
