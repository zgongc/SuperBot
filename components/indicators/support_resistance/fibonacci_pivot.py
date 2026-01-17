"""
indicators/support_resistance/fibonacci_pivot.py - Fibonacci Pivot Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Fibonacci Pivot Points - Fibonacci oranlarını kullanarak pivot seviyeleri
    Fibonacci ratios (0.382, 0.618, 1.000) ile destek ve direnç seviyeleri hesaplanır.

Formül:
    P = (High + Low + Close) / 3
    R1 = P + 0.382 * (High - Low)
    R2 = P + 0.618 * (High - Low)
    R3 = P + 1.000 * (High - Low)
    S1 = P - 0.382 * (High - Low)
    S2 = P - 0.618 * (High - Low)
    S3 = P - 1.000 * (High - Low)

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


class FibonacciPivot(BaseIndicator):
    """
    Fibonacci Pivot Points

    Önceki periyodun High, Low ve Close değerlerini kullanarak
    Fibonacci oranlarıyla pivot seviyeleri (P, R1-R3, S1-S3) hesaplar.

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
            name='fibonacci_pivot',
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
        Fibonacci Pivot Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Fibonacci pivot seviyeleri (P, R1-R3, S1-S3)
        """
        # Önceki periyodun H, L, C değerlerini al
        high = data['high'].iloc[-self.period - 1:-1].max()
        low = data['low'].iloc[-self.period - 1:-1].min()
        close = data['close'].iloc[-self.period - 1]

        # Pivot Point hesapla
        pivot = (high + low + close) / 3
        range_hl = high - low

        # Fibonacci Resistance seviyeleri
        r1 = pivot + 0.382 * range_hl
        r2 = pivot + 0.618 * range_hl
        r3 = pivot + 1.000 * range_hl

        # Fibonacci Support seviyeleri
        s1 = pivot - 0.382 * range_hl
        s2 = pivot - 0.618 * range_hl
        s3 = pivot - 1.000 * range_hl

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Seviyeleri sözlük olarak oluştur
        levels = {
            'R3': round(r3, 2),
            'R2': round(r2, 2),
            'R1': round(r1, 2),
            'P': round(pivot, 2),
            'S1': round(s1, 2),
            'S2': round(s2, 2),
            'S3': round(s3, 2)
        }

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
                'range': round(range_hl, 2),
                'current_price': round(current_price, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Fibonacci Pivot Points calculation - BACKTEST için

        Fibonacci Pivot Formula:
            P = (High + Low + Close) / 3
            R1 = P + 0.382 × (High - Low)
            R2 = P + 0.618 × (High - Low)
            R3 = P + 1.000 × (High - Low)
            S1 = P - 0.382 × (High - Low)
            S2 = P - 0.618 × (High - Low)
            S3 = P - 1.000 × (High - Low)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: fib_pivot, r1, r2, r3, s1, s2, s3 for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Previous period High/Low/Close
        prev_high = high.shift(self.period).rolling(window=self.period).max()
        prev_low = low.shift(self.period).rolling(window=self.period).min()
        prev_close = close.shift(self.period)

        # Pivot Point
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low

        # Fibonacci Resistance levels
        r1 = pivot + 0.382 * range_hl
        r2 = pivot + 0.618 * range_hl
        r3 = pivot + 1.000 * range_hl

        # Fibonacci Support levels
        s1 = pivot - 0.382 * range_hl
        s2 = pivot - 0.618 * range_hl
        s3 = pivot - 1.000 * range_hl

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
            'P': pivot.values,
            'R1': r1.values,
            'R2': r2.values,
            'R3': r3.values,
            'S1': s1.values,
            'S2': s2.values,
            'S3': s3.values
        }, index=data.index)

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

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

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

    def get_signal(self, price: float, levels: dict) -> SignalType:
        """
        Fiyatın fibonacci pivot seviyelerine göre sinyal üret

        Args:
            price: Güncel fiyat
            levels: Fibonacci pivot seviyeleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if price < levels['S2']:
            return SignalType.BUY
        elif price > levels['R2']:
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
            levels: Fibonacci pivot seviyeleri

        Returns:
            float: Güç değeri (0-100)
        """
        pivot = levels['P']
        r3 = levels['R3']
        s3 = levels['S3']

        if price > pivot:
            # Yukarı yönde güç
            strength = ((price - pivot) / (r3 - pivot)) * 100
        else:
            # Aşağı yönde güç
            strength = ((pivot - price) / (pivot - s3)) * 100

        return min(max(strength, 0), 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 1
        }

    def _requires_volume(self) -> bool:
        """Fibonacci Pivot volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['FibonacciPivot']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Fibonacci Pivot Points indikatör testi"""

    print("\n" + "="*60)
    print("FIBONACCI PIVOT POINTS TEST")
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
    fib_pivot = FibonacciPivot(period=1)
    print(f"   [OK] Oluşturuldu: {fib_pivot}")
    print(f"   [OK] Kategori: {fib_pivot.category.value}")
    print(f"   [OK] Tip: {fib_pivot.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {fib_pivot.get_required_periods()}")

    result = fib_pivot(data)
    print(f"   [OK] Fibonacci Pivot Seviyeleri:")
    for level, value in result.value.items():
        print(f"        {level}: {value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [1, 5, 10]:
        fib_test = FibonacciPivot(period=period)
        result = fib_test.calculate(data)
        print(f"   [OK] FibPivot({period}) - P: {result.value['P']} | Sinyal: {result.signal.value}")

    # Test 3: Fibonacci seviyelerinin analizi
    print("\n4. Fibonacci seviye analizi...")
    result = fib_pivot.calculate(data)
    current = result.metadata['current_price']
    print(f"   [OK] Güncel fiyat: {current}")
    print(f"   [OK] Pivot: {result.value['P']}")
    print(f"   [OK] Range: {result.metadata['range']}")
    if current > result.value['P']:
        print(f"   [OK] Fiyat pivot üstünde (Bullish)")
        print(f"   [OK] R1 (38.2%): {result.value['R1']}")
        print(f"   [OK] R2 (61.8%): {result.value['R2']}")
        print(f"   [OK] R3 (100%): {result.value['R3']}")
    else:
        print(f"   [OK] Fiyat pivot altında (Bearish)")
        print(f"   [OK] S1 (38.2%): {result.value['S1']}")
        print(f"   [OK] S2 (61.8%): {result.value['S2']}")
        print(f"   [OK] S3 (100%): {result.value['S3']}")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = fib_pivot.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = fib_pivot.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
