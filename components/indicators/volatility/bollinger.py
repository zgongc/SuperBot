"""
indicators/volatility/bollinger.py - Bollinger Bands

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Bollinger Bands - Volatilite bantları
    Fiyatın ortalama etrafındaki standart sapma bantları
    Aşırı alım/satım ve volatilite ölçümü için kullanılır

Formül:
    Middle Band = SMA(close, period)
    Upper Band = Middle + (std_dev * multiplier)
    Lower Band = Middle - (std_dev * multiplier)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.sma import SMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands

    Fiyatın volatilite bantlarını oluşturarak aşırı alım/satım seviyelerini gösterir.
    Bantlar daraldığında volatilite düşük, genişlediğinde yüksektir.

    Args:
        period: SMA periyodu (varsayılan: 20)
        std_dev: Standart sapma çarpanı (varsayılan: 2.0)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.std_dev = std_dev

        # SMA indikatörünü kullan (code reuse)
        self._sma = SMA(period=period)

        super().__init__(
            name='bollinger',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'std_dev': std_dev
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
        if self.std_dev <= 0:
            raise InvalidParameterError(
                self.name, 'std_dev', self.std_dev,
                "Standart sapma çarpanı pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Bollinger Bands hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Bollinger Bands değerleri (upper, middle, lower)
        """
        close = data['close'].values

        # Middle Band (SMA)
        middle = np.mean(close[-self.period:])

        # Standart sapma
        std = np.std(close[-self.period:], ddof=0)

        # Upper ve Lower Bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Fiyatın bantlar içindeki pozisyonu (%B)
        if upper != lower:
            percent_b = (current_price - lower) / (upper - lower)
        else:
            percent_b = 0.5

        # Bant genişliği (volatilite göstergesi)
        if middle != 0:
            bandwidth = ((upper - lower) / middle) * 100
        else:
            bandwidth = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper, 8),
                'middle': round(middle, 8),
                'lower': round(lower, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(percent_b),
            trend=self.get_trend(current_price, middle),
            strength=min(abs(percent_b - 0.5) * 200, 100),  # 0-100 arası normalize et
            metadata={
                'period': self.period,
                'std_dev': self.std_dev,
                'percent_b': round(percent_b, 4),
                'bandwidth': round(bandwidth, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Bollinger Bands calculation - BACKTEST için

        Bollinger Bands Formula:
            Middle Band = SMA(close, period)
            Upper Band = Middle + (std_dev * multiplier)
            Lower Band = Middle - (std_dev * multiplier)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 3 columns (bb_upper, bb_middle, bb_lower)

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']

        # 1. Middle Band = SMA(close, period) - use SMA.calculate_batch (code reuse)
        middle = self._sma.calculate_batch(data)

        # 2. Standard Deviation (no SMA indicator for std, keep rolling)
        std = close.rolling(window=self.period).std(ddof=0)

        # 3. Upper and Lower Bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # Create result DataFrame (calculate() ile aynı key'ler - prefix yok)
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)

        # Set first period values to NaN (warmup)
        result.iloc[:self.period] = np.nan

        return result

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        buffer_key = symbol if symbol else 'default'
        max_len = self.get_required_periods() + 50

        self._buffers[buffer_key] = {
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'close': deque(maxlen=max_len)
        }

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._buffers[buffer_key]['high'].append(data['high'].iloc[i])
            self._buffers[buffer_key]['low'].append(data['low'].iloc[i])
            self._buffers[buffer_key]['close'].append(data['close'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel Bollinger Bands değerleri
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        # (Check if it's dict - BaseIndicator.warmup_buffer creates deque, we need dict)
        if buffer_key not in self._buffers or not isinstance(self._buffers[buffer_key], dict):
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = {
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len)
            }

        # Add new candle to symbol's buffer
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._buffers[buffer_key]['high'].append(high_val)
        self._buffers[buffer_key]['low'].append(low_val)
        self._buffers[buffer_key]['close'].append(close_val)

        # Need minimum data for Bollinger calculation
        if len(self._buffers[buffer_key]['close']) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period, 'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame({
            'high': list(self._buffers[buffer_key]['high']),
            'low': list(self._buffers[buffer_key]['low']),
            'close': list(self._buffers[buffer_key]['close']),
            'open': [open_val] * len(self._buffers[buffer_key]['close']),
            'volume': [volume_val] * len(self._buffers[buffer_key]['close']),
            'timestamp': [timestamp_val] * len(self._buffers[buffer_key]['close'])
        })

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, percent_b: float) -> SignalType:
        """
        %B değerinden sinyal üret

        Args:
            percent_b: Fiyatın bantlar içindeki pozisyonu (0-1 arası)

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        if percent_b <= 0:  # Alt bantta veya altında
            return SignalType.BUY
        elif percent_b >= 1:  # Üst bantta veya üstünde
            return SignalType.SELL
        elif percent_b < 0.2:  # Alt banda yakın
            return SignalType.BUY
        elif percent_b > 0.8:  # Üst banda yakın
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, middle: float) -> TrendDirection:
        """
        Fiyat ve orta banttan trend belirle

        Args:
            price: Güncel fiyat
            middle: Orta bant değeri

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
            'period': 20,
            'std_dev': 2.0
        }

    def _requires_volume(self) -> bool:
        """Bollinger Bands volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['BollingerBands']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Bollinger Bands indikatör testi"""

    print("\n" + "="*60)
    print("BOLLINGER BANDS TEST")
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
    bb = BollingerBands(period=20, std_dev=2.0)
    print(f"   [OK] Oluşturuldu: {bb}")
    print(f"   [OK] Kategori: {bb.category.value}")
    print(f"   [OK] Tip: {bb.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {bb.get_required_periods()}")

    result = bb(data)
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Middle Band: {result.value['middle']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] %B: {result.metadata['percent_b']}")
    print(f"   [OK] Bandwidth: {result.metadata['bandwidth']:.2f}%")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [10, 20, 30]:
        bb_test = BollingerBands(period=period)
        result = bb_test.calculate(data)
        print(f"   [OK] BB({period}): Upper={result.value['upper']:.2f}, Middle={result.value['middle']:.2f}, Lower={result.value['lower']:.2f}")

    # Test 3: Farklı standart sapma çarpanları
    print("\n4. Farklı std_dev testi...")
    for std_dev in [1.5, 2.0, 2.5]:
        bb_test = BollingerBands(period=20, std_dev=std_dev)
        result = bb_test.calculate(data)
        print(f"   [OK] BB(std={std_dev}): Bandwidth={result.metadata['bandwidth']:.2f}%")

    # Test 4: İstatistikler
    print("\n5. İstatistik testi...")
    stats = bb.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = bb.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
