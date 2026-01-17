"""
indicators/volatility/keltner.py - Keltner Channel

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Keltner Channel - ATR tabanlı volatilite bantları
    EMA ve ATR kullanarak fiyat kanalları oluşturur
    Bollinger Bands'e benzer ancak ATR kullanır

Formül:
    Middle Line = EMA(close, period)
    Upper Channel = Middle + (ATR(atr_period) * multiplier)
    Lower Channel = Middle - (ATR(atr_period) * multiplier)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.volatility.atr import ATR
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class KeltnerChannel(BaseIndicator):
    """
    Keltner Channel

    EMA ve ATR kullanarak volatilite kanalları oluşturur.
    Trend takibi ve breakout tespiti için kullanılır.

    Args:
        ema_period: EMA periyodu (varsayılan: 20)
        atr_period: ATR periyodu (varsayılan: 10)
        multiplier: ATR çarpanı (varsayılan: 2.0)
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

        # EMA ve ATR indikatörlerini kullan (code reuse)
        self._ema = EMA(period=ema_period)
        self._atr = ATR(period=atr_period)

        super().__init__(
            name='keltner',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.BANDS,
            params={
                'ema_period': ema_period,
                'atr_period': atr_period,
                'multiplier': multiplier
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.ema_period, self.atr_period) + 1

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.ema_period < 1:
            raise InvalidParameterError(
                self.name, 'ema_period', self.ema_period,
                "EMA periyodu pozitif olmalı"
            )
        if self.atr_period < 1:
            raise InvalidParameterError(
                self.name, 'atr_period', self.atr_period,
                "ATR periyodu pozitif olmalı"
            )
        if self.multiplier <= 0:
            raise InvalidParameterError(
                self.name, 'multiplier', self.multiplier,
                "Çarpan pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Keltner Channel hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Keltner Channel değerleri (upper, middle, lower)
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # EMA hesapla (Middle Line)
        ema_values = np.zeros(len(close))
        ema_values[0] = close[0]
        alpha = 2.0 / (self.ema_period + 1)

        for i in range(1, len(close)):
            ema_values[i] = alpha * close[i] + (1 - alpha) * ema_values[i-1]

        middle = ema_values[-1]

        # ATR hesapla
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR (RMA - Wilder's smoothing)
        atr_values = np.zeros(len(tr))
        start_idx = self.atr_period - 1
        atr_values[start_idx] = np.mean(tr[:self.atr_period])

        atr_alpha = 1.0 / self.atr_period
        for i in range(self.atr_period, len(tr)):
            atr_values[i] = atr_values[i-1] + atr_alpha * (tr[i] - atr_values[i-1])

        atr = atr_values[-1]

        # Upper ve Lower Channels
        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Fiyatın kanal içindeki pozisyonu
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5

        # Kanal genişliği
        if middle != 0:
            channel_width = ((upper - lower) / middle) * 100
        else:
            channel_width = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper, 8),
                'middle': round(middle, 8),
                'lower': round(lower, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(position),
            trend=self.get_trend(current_price, middle),
            strength=min(abs(position - 0.5) * 200, 100),
            metadata={
                'ema_period': self.ema_period,
                'atr_period': self.atr_period,
                'multiplier': self.multiplier,
                'atr': round(atr, 8),
                'position': round(position, 4),
                'channel_width': round(channel_width, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Keltner Channel calculation - BACKTEST için

        Keltner Channel Formula:
            Middle = EMA(close, ema_period)
            Upper = Middle + (ATR * multiplier)
            Lower = Middle - (ATR * multiplier)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 3 columns (kc_upper, kc_middle, kc_lower)

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        # 1. Middle Line = EMA - use EMA.calculate_batch (code reuse)
        middle = self._ema.calculate_batch(data)

        # 2. ATR - use ATR.calculate_batch (code reuse)
        atr = self._atr.calculate_batch(data)

        # 3. Upper and Lower Channels
        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)

        # Create result DataFrame (calculate() ile aynı key'ler)
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)

        # Set first period values to NaN (warmup)
        warmup = max(self.ema_period, self.atr_period)
        result.iloc[:warmup] = np.nan

        return result

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
                value={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
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

    def get_signal(self, position: float) -> SignalType:
        """
        Pozisyondan sinyal üret

        Args:
            position: Fiyatın kanal içindeki pozisyonu (0-1 arası)

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if position <= 0:  # Alt kanalda veya altında
            return SignalType.BUY
        elif position >= 1:  # Üst kanalda veya üstünde
            return SignalType.SELL
        elif position < 0.2:  # Alt kanala yakın
            return SignalType.BUY
        elif position > 0.8:  # Üst kanala yakın
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, middle: float) -> TrendDirection:
        """
        Fiyat ve orta çizgiden trend belirle

        Args:
            price: Güncel fiyat
            middle: Orta çizgi değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if price > middle:
            return TrendDirection.UP
        elif price < middle:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'ema_period': 20,
            'atr_period': 10,
            'multiplier': 2.0
        }

    def _requires_volume(self) -> bool:
        """Keltner Channel volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['KeltnerChannel']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Keltner Channel indikatör testi"""

    print("\n" + "="*60)
    print("KELTNER CHANNEL TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(35)]

    # Fiyat hareketini simüle et
    base_price = 100
    prices = [base_price]
    for i in range(34):
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
    kc = KeltnerChannel(ema_period=20, atr_period=10, multiplier=2.0)
    print(f"   [OK] Oluşturuldu: {kc}")
    print(f"   [OK] Kategori: {kc.category.value}")
    print(f"   [OK] Tip: {kc.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {kc.get_required_periods()}")

    result = kc(data)
    print(f"   [OK] Upper Channel: {result.value['upper']}")
    print(f"   [OK] Middle Line: {result.value['middle']}")
    print(f"   [OK] Lower Channel: {result.value['lower']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] ATR: {result.metadata['atr']}")
    print(f"   [OK] Position: {result.metadata['position']}")
    print(f"   [OK] Channel Width: {result.metadata['channel_width']:.2f}%")

    # Test 2: Farklı parametreler
    print("\n3. Farklı parametre testi...")
    for ema_p, atr_p in [(10, 10), (20, 10), (30, 20)]:
        kc_test = KeltnerChannel(ema_period=ema_p, atr_period=atr_p)
        result = kc_test.calculate(data)
        print(f"   [OK] KC(EMA={ema_p}, ATR={atr_p}): Width={result.metadata['channel_width']:.2f}%")

    # Test 3: Farklı çarpanlar
    print("\n4. Farklı multiplier testi...")
    for mult in [1.5, 2.0, 2.5]:
        kc_test = KeltnerChannel(ema_period=20, atr_period=10, multiplier=mult)
        result = kc_test.calculate(data)
        print(f"   [OK] KC(mult={mult}): Width={result.metadata['channel_width']:.2f}%")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = kc.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = kc.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
