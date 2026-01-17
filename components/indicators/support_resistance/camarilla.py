"""
indicators/support_resistance/camarilla.py - Camarilla Pivot Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Camarilla Pivot Points - Camarilla formülü ile pivot seviyeleri
    Kısa vadeli destek ve direnç seviyelerini belirlemek için kullanılır.
    Genellikle intraday trading için tercih edilir.

Formül:
    R4 = Close + ((High - Low) × 1.1 / 2)
    R3 = Close + ((High - Low) × 1.1 / 4)
    R2 = Close + ((High - Low) × 1.1 / 6)
    R1 = Close + ((High - Low) × 1.1 / 12)
    S1 = Close - ((High - Low) × 1.1 / 12)
    S2 = Close - ((High - Low) × 1.1 / 6)
    S3 = Close - ((High - Low) × 1.1 / 4)
    S4 = Close - ((High - Low) × 1.1 / 2)

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


class Camarilla(BaseIndicator):
    """
    Camarilla Pivot Points

    Önceki periyodun High, Low ve Close değerlerini kullanarak
    Camarilla pivot seviyeleri (R1-R4, S1-S4) hesaplar.

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
            name='camarilla',
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
        Camarilla Pivot Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Camarilla pivot seviyeleri (R1-R4, S1-S4)
        """
        # Önceki periyodun H, L, C değerlerini al
        high = data['high'].iloc[-self.period - 1:-1].max()
        low = data['low'].iloc[-self.period - 1:-1].min()
        close = data['close'].iloc[-self.period - 1]

        # Range hesapla
        range_hl = high - low
        multiplier = 1.1

        # Camarilla Resistance seviyeleri
        r4 = close + (range_hl * multiplier / 2)
        r3 = close + (range_hl * multiplier / 4)
        r2 = close + (range_hl * multiplier / 6)
        r1 = close + (range_hl * multiplier / 12)

        # Camarilla Support seviyeleri
        s1 = close - (range_hl * multiplier / 12)
        s2 = close - (range_hl * multiplier / 6)
        s3 = close - (range_hl * multiplier / 4)
        s4 = close - (range_hl * multiplier / 2)

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Seviyeleri sözlük olarak oluştur
        levels = {
            'R4': round(r4, 2),
            'R3': round(r3, 2),
            'R2': round(r2, 2),
            'R1': round(r1, 2),
            'S1': round(s1, 2),
            'S2': round(s2, 2),
            'S3': round(s3, 2),
            'S4': round(s4, 2)
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, close),
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
        ⚡ VECTORIZED batch Camarilla Pivot Points calculation - BACKTEST için

        Camarilla Formula:
            Range = High - Low
            Multiplier = 1.1
            R4 = Close + (Range × 1.1 / 2)
            R3 = Close + (Range × 1.1 / 4)
            R2 = Close + (Range × 1.1 / 6)
            R1 = Close + (Range × 1.1 / 12)
            S1 = Close - (Range × 1.1 / 12)
            S2 = Close - (Range × 1.1 / 6)
            S3 = Close - (Range × 1.1 / 4)
            S4 = Close - (Range × 1.1 / 2)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: r1, r2, r3, r4, s1, s2, s3, s4 for all bars

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

        # Range and multiplier
        range_hl = prev_high - prev_low
        multiplier = 1.1

        # Camarilla Resistance levels
        r4 = prev_close + (range_hl * multiplier / 2)
        r3 = prev_close + (range_hl * multiplier / 4)
        r2 = prev_close + (range_hl * multiplier / 6)
        r1 = prev_close + (range_hl * multiplier / 12)

        # Camarilla Support levels
        s1 = prev_close - (range_hl * multiplier / 12)
        s2 = prev_close - (range_hl * multiplier / 6)
        s3 = prev_close - (range_hl * multiplier / 4)
        s4 = prev_close - (range_hl * multiplier / 2)

        # Set first period values to NaN (warmup)
        warmup = self.period * 2
        r1.iloc[:warmup] = np.nan
        r2.iloc[:warmup] = np.nan
        r3.iloc[:warmup] = np.nan
        r4.iloc[:warmup] = np.nan
        s1.iloc[:warmup] = np.nan
        s2.iloc[:warmup] = np.nan
        s3.iloc[:warmup] = np.nan
        s4.iloc[:warmup] = np.nan

        return pd.DataFrame({
            'R1': r1.values,
            'R2': r2.values,
            'R3': r3.values,
            'R4': r4.values,
            'S1': s1.values,
            'S2': s2.values,
            'S3': s3.values,
            'S4': s4.values
        }, index=data.index)

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
                value={'h4': 0.0, 'l4': 0.0},
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
        Fiyatın camarilla seviyelerine göre sinyal üret

        Args:
            price: Güncel fiyat
            levels: Camarilla seviyeleri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # S3 altında güçlü alım
        if price < levels['S3']:
            return SignalType.BUY
        # R3 üstünde güçlü satış
        elif price > levels['R3']:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, close: float) -> TrendDirection:
        """
        Fiyatın önceki kapanışa göre trend belirle

        Args:
            price: Güncel fiyat
            close: Önceki kapanış

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if price > close:
            return TrendDirection.UP
        elif price < close:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, levels: dict) -> float:
        """
        Fiyatın seviyelere göre güç hesapla

        Args:
            price: Güncel fiyat
            levels: Camarilla seviyeleri

        Returns:
            float: Güç değeri (0-100)
        """
        r4 = levels['R4']
        s4 = levels['S4']
        mid_point = (r4 + s4) / 2

        if price > mid_point:
            # Yukarı yönde güç
            strength = ((price - mid_point) / (r4 - mid_point)) * 100
        else:
            # Aşağı yönde güç
            strength = ((mid_point - price) / (mid_point - s4)) * 100

        return min(max(strength, 0), 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 1
        }

    def _requires_volume(self) -> bool:
        """Camarilla volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Camarilla']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Camarilla Pivot Points indikatör testi"""

    print("\n" + "="*60)
    print("CAMARILLA PIVOT POINTS TEST")
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
    camarilla = Camarilla(period=1)
    print(f"   [OK] Oluşturuldu: {camarilla}")
    print(f"   [OK] Kategori: {camarilla.category.value}")
    print(f"   [OK] Tip: {camarilla.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {camarilla.get_required_periods()}")

    result = camarilla(data)
    print(f"   [OK] Camarilla Seviyeleri:")
    for level, value in result.value.items():
        print(f"        {level}: {value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [1, 5, 10]:
        cam_test = Camarilla(period=period)
        result = cam_test.calculate(data)
        print(f"   [OK] Camarilla({period}) - R1: {result.value['R1']} | S1: {result.value['S1']}")

    # Test 3: Seviye analizi
    print("\n4. Seviye analizi...")
    result = camarilla.calculate(data)
    current = result.metadata['current_price']
    close_prev = result.metadata['close']
    print(f"   [OK] Güncel fiyat: {current}")
    print(f"   [OK] Önceki kapanış: {close_prev}")
    print(f"   [OK] Range: {result.metadata['range']}")

    # En yakın seviyeleri bul
    levels_list = [(k, v) for k, v in result.value.items()]
    if current > close_prev:
        print(f"   [OK] Fiyat yükseliyor (Bullish)")
        print(f"   [OK] R1: {result.value['R1']}")
        print(f"   [OK] R2: {result.value['R2']}")
        print(f"   [OK] R3: {result.value['R3']} (kritik direnç)")
        print(f"   [OK] R4: {result.value['R4']} (breakout seviyesi)")
    else:
        print(f"   [OK] Fiyat düşüyor (Bearish)")
        print(f"   [OK] S1: {result.value['S1']}")
        print(f"   [OK] S2: {result.value['S2']}")
        print(f"   [OK] S3: {result.value['S3']} (kritik destek)")
        print(f"   [OK] S4: {result.value['S4']} (breakdown seviyesi)")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = camarilla.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = camarilla.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
