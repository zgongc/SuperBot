"""
indicators/volume/vwap.py - Volume Weighted Average Price

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    VWAP (Volume Weighted Average Price) - Hacim ağırlıklı ortalama fiyat
    Günlük işlemlerin ortalama fiyatını hacim ile ağırlıklandırır
    Fiyat > VWAP = Güçlü (alıcı baskısı)
    Fiyat < VWAP = Zayıf (satıcı baskısı)

Formül:
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3

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


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price

    Hacim ağırlıklı ortalama fiyat hesaplar.
    Kurumsal yatırımcıların ortalama giriş fiyatını gösterir.

    Args:
        period: VWAP hesaplama periyodu (varsayılan: 20, 0=tüm data)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='vwap',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return max(self.period, 1) if self.period > 0 else 1

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.period < 0:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Periyot negatif olamaz (0=tüm data)"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        VWAP hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VWAP değeri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Typical Price hesapla
        typical_price = (high + low + close) / 3

        # Period seçimi (0 ise tüm data)
        if self.period == 0 or len(data) <= self.period:
            tp_vol = typical_price * volume
            vwap_value = np.sum(tp_vol) / np.sum(volume)
        else:
            tp_vol = typical_price[-self.period:] * volume[-self.period:]
            vwap_value = np.sum(tp_vol) / np.sum(volume[-self.period:])

        current_price = close[-1]
        distance = ((current_price - vwap_value) / vwap_value) * 100

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(vwap_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(current_price, vwap_value),
            trend=self.get_trend(current_price, vwap_value),
            strength=min(abs(distance), 100),
            metadata={
                'period': self.period if self.period > 0 else len(data),
                'current_price': round(current_price, 8),
                'distance_pct': round(distance, 2),
                'typical_price': round(typical_price[-1], 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch VWAP calculation - BACKTEST için

        VWAP Formula:
            VWAP = Σ(Typical Price × Volume) / Σ(Volume)
            Typical Price = (High + Low + Close) / 3

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: VWAP values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume = data['volume']

        # TP × Volume
        tp_vol = typical_price * volume

        if self.period == 0:
            # Cumulative VWAP (from start)
            vwap = tp_vol.cumsum() / volume.cumsum()
        else:
            # Rolling VWAP
            tp_vol_sum = tp_vol.rolling(window=self.period).sum()
            volume_sum = volume.rolling(window=self.period).sum()
            vwap = tp_vol_sum / volume_sum

            # Set first period values to NaN (warmup)
            vwap.iloc[:self.period-1] = np.nan

        return pd.Series(vwap.values, index=data.index, name='vwap')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli state'i hazırlar"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50

        # Initialize symbol-aware buffers
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        buffer_key = symbol if symbol else 'default'
        self._buffers[buffer_key] = {
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'close': deque(maxlen=max_len),
            'volume': deque(maxlen=max_len)
        }

        # Fill buffers with data
        for i in range(len(data)):
            self._buffers[buffer_key]['high'].append(data['high'].iloc[i])
            self._buffers[buffer_key]['low'].append(data['low'].iloc[i])
            self._buffers[buffer_key]['close'].append(data['close'].iloc[i])
            self._buffers[buffer_key]['volume'].append(data['volume'].iloc[i])

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel VWAP değeri
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
            self._buffers[buffer_key] = {
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len),
                'volume': deque(maxlen=max_len)
            }

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            high_val = candle['high']
            low_val = candle['low']
            volume_val = candle.get('volume', 1000)
            open_val = candle.get('open', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        # Add new candle to symbol's buffer
        self._buffers[buffer_key]['high'].append(high_val)
        self._buffers[buffer_key]['low'].append(low_val)
        self._buffers[buffer_key]['close'].append(close_val)
        self._buffers[buffer_key]['volume'].append(volume_val)

        # Need minimum data for VWAP calculation
        if len(self._buffers[buffer_key]['close']) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value=0.0,
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
            'volume': list(self._buffers[buffer_key]['volume']),
            'open': [open_val] * len(self._buffers[buffer_key]['close']),
            'timestamp': [timestamp_val] * len(self._buffers[buffer_key]['close'])
        })

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, price: float, vwap: float) -> SignalType:
        """
        VWAP'a göre sinyal üret

        Args:
            price: Güncel fiyat
            vwap: VWAP değeri

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        distance_pct = ((price - vwap) / vwap) * 100

        if distance_pct < -1.0:  # %1'den fazla VWAP altında
            return SignalType.BUY
        elif distance_pct > 1.0:  # %1'den fazla VWAP üstünde
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, vwap: float) -> TrendDirection:
        """
        VWAP'a göre trend belirle

        Args:
            price: Güncel fiyat
            vwap: VWAP değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        if price > vwap:
            return TrendDirection.UP
        elif price < vwap:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """VWAP volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VWAP']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """VWAP indikatör testi"""

    print("\n" + "="*60)
    print("VWAP (VOLUME WEIGHTED AVERAGE PRICE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    # Fiyat hareketi
    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-3000, 5000))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Hacim aralığı: {min(volumes):.0f} -> {max(volumes):.0f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    vwap = VWAP(period=20)
    print(f"   [OK] Oluşturuldu: {vwap}")
    print(f"   [OK] Kategori: {vwap.category.value}")
    print(f"   [OK] Gerekli periyot: {vwap.get_required_periods()}")

    result = vwap(data)
    print(f"   [OK] VWAP Değeri: {result.value:.8f}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    for period in [0, 10, 20, 30]:
        vwap_test = VWAP(period=period)
        result = vwap_test.calculate(data)
        period_str = "Tüm" if period == 0 else str(period)
        print(f"   [OK] VWAP({period_str}): {result.value:.8f} | Sinyal: {result.signal.value}")

    # Test 3: Volume gereksinimi
    print("\n4. Volume gereksinimi testi...")
    metadata = vwap.metadata
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "VWAP volume gerektirmeli!"

    # Test 4: Fiyat-VWAP ilişkisi
    print("\n5. Fiyat-VWAP ilişkisi testi...")
    result = vwap.calculate(data)
    current_price = data['close'].iloc[-1]
    print(f"   [OK] Güncel fiyat: {current_price:.8f}")
    print(f"   [OK] VWAP: {result.value:.8f}")
    print(f"   [OK] Fark: {result.metadata['distance_pct']:.2f}%")
    print(f"   [OK] Durum: {'Fiyat VWAP üstünde' if current_price > result.value else 'Fiyat VWAP altında'}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = vwap.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
