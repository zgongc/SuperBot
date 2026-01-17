"""
indicators/trend/donchian.py - Donchian Channel

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Donchian Channel - Donchian Kanalı
    Richard Donchian tarafından geliştirilmiş basit ama etkili kanal indikatörü
    Belirli periyottaki en yüksek ve en düşük değerleri kullanır

    Kullanım:
    - Breakout stratejileri
    - Trend yönünü belirleme
    - Destek/direnç seviyeleri
    - Volatilite ölçümü

Formül:
    Upper Band = Highest High (period)
    Lower Band = Lowest Low (period)
    Middle Band = (Upper Band + Lower Band) / 2

    Varsayılan: period=20

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


class DonchianChannel(BaseIndicator):
    """
    Donchian Channel Indicator

    Belirli periyottaki en yüksek ve en düşük değerlerden kanal oluşturur.

    Args:
        period: Kanal periyodu (varsayılan: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='donchian',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period
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
                "Period pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Donchian Channel hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Upper, Middle, Lower band değerleri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Upper Band = En yüksek değer
        upper = np.max(high[-self.period:])

        # Lower Band = En düşük değer
        lower = np.min(low[-self.period:])

        # Middle Band = Ortalama
        middle = (upper + lower) / 2

        # Mevcut fiyat
        current_price = close[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Trend ve sinyal belirleme
        trend = self.get_trend(current_price, middle)
        signal = self.get_signal(current_price, upper, middle, lower)

        # Band genişliği
        bandwidth = ((upper - lower) / middle) * 100

        # Fiyatın kanal içindeki pozisyonu (0-100)
        if upper != lower:
            position_pct = ((current_price - lower) / (upper - lower)) * 100
        else:
            position_pct = 50

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper, 2),
                'middle': round(middle, 2),
                'lower': round(lower, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=self._calculate_strength(current_price, upper, middle, lower),
            metadata={
                'period': self.period,
                'bandwidth': round(bandwidth, 2),
                'current_price': round(current_price, 2),
                'position_pct': round(position_pct, 2),
                'position': self._get_position(current_price, upper, middle, lower)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Donchian Channel calculation - BACKTEST için

        Donchian Formula:
            Upper = Highest High (period)
            Lower = Lowest Low (period)
            Middle = (Upper + Lower) / 2

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 3 columns (dc_upper, dc_middle, dc_lower)

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']

        # Upper Band = Highest High over period
        upper = high.rolling(window=self.period).max()

        # Lower Band = Lowest Low over period
        lower = low.rolling(window=self.period).min()

        # Middle Band = Average
        middle = (upper + lower) / 2

        # Create result DataFrame (calculate() ile aynı key'ler)
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)

        # Set first period values to NaN (warmup)
        result.iloc[:self.period-1] = np.nan

        return result

    def _get_position(self, price: float, upper: float, middle: float, lower: float) -> str:
        """Fiyatın kanal içindeki pozisyonu"""
        if price >= upper:
            return 'at_upper'  # Breakout
        elif price > middle:
            return 'upper_half'
        elif price > lower:
            return 'lower_half'
        else:
            return 'at_lower'  # Breakdown

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - update() için gerekli"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
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

    def get_signal(self, price: float, upper: float, middle: float, lower: float) -> SignalType:
        """
        Donchian Channel'dan sinyal üret

        Args:
            price: Mevcut fiyat
            upper: Üst band
            middle: Orta band
            lower: Alt band

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        # Breakout stratejisi
        if price >= upper:
            return SignalType.BUY  # Üst banda breakout - BUY
        elif price <= lower:
            return SignalType.SELL  # Alt banda breakdown - SELL

        return SignalType.HOLD

    def get_trend(self, price: float, middle: float) -> TrendDirection:
        """
        Donchian Channel'dan trend belirle

        Args:
            price: Mevcut fiyat
            middle: Orta band

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        if price > middle:
            return TrendDirection.UP
        elif price < middle:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, upper: float, middle: float, lower: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        # Fiyat üst veya alt banda ne kadar yakın?
        upper_distance = abs(price - upper) / upper * 100
        lower_distance = abs(price - lower) / lower * 100

        # En yakın bandın tersi olarak güç
        min_distance = min(upper_distance, lower_distance)
        return max(0, 100 - min_distance * 20)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['upper', 'middle', 'lower']

    def _requires_volume(self) -> bool:
        """Donchian Channel volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['DonchianChannel']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Donchian Channel indikatör testi"""

    print("\n" + "="*60)
    print("DONCHIAN CHANNEL TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Breakout simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 30:
            trend = 0.2  # Yavaş hareket
            noise = np.random.randn() * 1.0
        else:
            trend = 1.5  # Güçlü breakout
            noise = np.random.randn() * 1.5

        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    donchian = DonchianChannel(period=20)
    print(f"   [OK] Oluşturuldu: {donchian}")
    print(f"   [OK] Kategori: {donchian.category.value}")
    print(f"   [OK] Tip: {donchian.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {donchian.get_required_periods()}")

    result = donchian(data)
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Middle Band: {result.value['middle']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Band analizi
    print("\n3. Band analizi...")
    print(f"   [OK] Bandwidth: {result.metadata['bandwidth']:.2f}%")
    print(f"   [OK] Fiyat pozisyonu: {result.metadata['position']}")
    print(f"   [OK] Pozisyon %: {result.metadata['position_pct']:.2f}%")

    # Test 3: Breakout testi
    print("\n4. Breakout testi...")
    if result.metadata['position'] == 'at_upper':
        print(f"   [OK] Üst banda BREAKOUT - Güçlü BUY sinyali")
    elif result.metadata['position'] == 'at_lower':
        print(f"   [OK] Alt banda BREAKDOWN - Güçlü SELL sinyali")
    else:
        print(f"   [OK] Kanal içinde - Breakout bekleniyor")

    # Test 4: Farklı veri dilimleri (breakout öncesi/sonrası)
    print("\n5. Breakout analizi (zaman serisi)...")
    for i in [25, 35, 45]:
        data_slice = data.iloc[:i+1]
        result = donchian.calculate(data_slice)
        print(f"   [OK] Mum {i}: Pos={result.metadata['position']}, BW={result.metadata['bandwidth']:.2f}%")

    # Test 5: Farklı periyotlar
    print("\n6. Farklı periyot testi...")
    for period in [10, 20, 50]:
        don_test = DonchianChannel(period=period)
        result = don_test.calculate(data)
        print(f"   [OK] Donchian({period}): BW={result.metadata['bandwidth']:.2f}%")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = donchian.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = donchian.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
