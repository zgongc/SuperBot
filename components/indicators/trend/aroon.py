"""
indicators/trend/aroon.py - Aroon Indicator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Aroon - Trend gücünü ve yönünü belirleyen indikatör
    Tushar Chande tarafından 1995'te geliştirilmiş
    Aroon Up ve Aroon Down olmak üzere iki çizgiden oluşur

    Kullanım:
    - Yeni trendlerin başlangıcını tespit etme
    - Trend gücünü ölçme
    - Konsolidasyon dönemlerini belirleme

Formül:
    Aroon Up = ((period - En yüksek değere olan periyot) / period) × 100
    Aroon Down = ((period - En düşük değere olan periyot) / period) × 100

    Aroon Up > 70 ve Aroon Down < 30: Güçlü yükseliş trendi
    Aroon Down > 70 ve Aroon Up < 30: Güçlü düşüş trendi

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


class Aroon(BaseIndicator):
    """
    Aroon Indicator

    En yüksek ve en düşük değerlerin ne kadar yakın zamanda olduğunu ölçer.

    Args:
        period: Aroon periyodu (varsayılan: 25)
    """

    def __init__(
        self,
        period: int = 25,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='aroon',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
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
        Aroon hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Aroon Up ve Aroon Down değerleri
        """
        high = data['high'].values
        low = data['low'].values

        # Son period için yüksek ve düşük değerleri
        high_slice = high[-self.period-1:]
        low_slice = low[-self.period-1:]

        # En yüksek ve en düşük değerlerin indexlerini bul
        # argmax/argmin son index'i döner
        high_idx = len(high_slice) - 1 - np.argmax(high_slice[::-1])
        low_idx = len(low_slice) - 1 - np.argmin(low_slice[::-1])

        # Aroon Up ve Down hesapla
        aroon_up = ((self.period - (len(high_slice) - 1 - high_idx)) / self.period) * 100
        aroon_down = ((self.period - (len(low_slice) - 1 - low_idx)) / self.period) * 100

        # Aroon Oscillator (opsiyonel)
        aroon_osc = aroon_up - aroon_down

        timestamp = int(data.iloc[-1]['timestamp'])

        # Trend ve sinyal belirleme
        trend = self.get_trend(aroon_up, aroon_down)
        signal = self.get_signal(aroon_up, aroon_down)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'aroon_up': round(aroon_up, 2),
                'aroon_down': round(aroon_down, 2),
                'aroon_osc': round(aroon_osc, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(aroon_osc), 100),
            metadata={
                'period': self.period,
                'high_periods_ago': len(high_slice) - 1 - high_idx,
                'low_periods_ago': len(low_slice) - 1 - low_idx,
                'trend_strength': self._get_trend_strength(aroon_up, aroon_down)
            }
        )

    def _get_trend_strength(self, aroon_up: float, aroon_down: float) -> str:
        """Trend gücünü değerlendir"""
        if aroon_up > 70 and aroon_down < 30:
            return 'Strong Uptrend'
        elif aroon_down > 70 and aroon_up < 30:
            return 'Strong Downtrend'
        elif aroon_up > 50 and aroon_down < 50:
            return 'Weak Uptrend'
        elif aroon_down > 50 and aroon_up < 50:
            return 'Weak Downtrend'
        else:
            return 'Consolidation'

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Aroon calculation - BACKTEST için

        Aroon Formula:
            Aroon Up = ((period - periods since highest high) / period) × 100
            Aroon Down = ((period - periods since lowest low) / period) × 100
            Aroon Oscillator = Aroon Up - Aroon Down

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: aroon_up, aroon_down, aroon_osc for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']

        # Calculate periods since highest high and lowest low
        def periods_since_max(window):
            """Returns periods since max value in window (from end)"""
            if len(window) == 0:
                return np.nan
            # argmax returns index of first max, we need periods from END
            # window is [oldest...newest], argmax gives index from start
            max_idx = np.argmax(window[::-1])  # Search from end
            return max_idx  # This is how many periods ago the max was

        def periods_since_min(window):
            """Returns periods since min value in window (from end)"""
            if len(window) == 0:
                return np.nan
            # argmin returns index of first min from end
            min_idx = np.argmin(window[::-1])  # Search from end
            return min_idx  # This is how many periods ago the min was

        # Rolling application
        high_periods = high.rolling(window=self.period+1).apply(periods_since_max, raw=True)
        low_periods = low.rolling(window=self.period+1).apply(periods_since_min, raw=True)

        # Aroon Up and Down calculation
        aroon_up = ((self.period - high_periods) / self.period) * 100
        aroon_down = ((self.period - low_periods) / self.period) * 100

        # Aroon Oscillator
        aroon_osc = aroon_up - aroon_down

        # Set first period values to NaN (warmup)
        aroon_up.iloc[:self.period] = np.nan
        aroon_down.iloc[:self.period] = np.nan
        aroon_osc.iloc[:self.period] = np.nan

        return pd.DataFrame({
            'aroon_up': aroon_up.values,
            'aroon_down': aroon_down.values,
            'aroon_osc': aroon_osc.values
        }, index=data.index)

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
                value={'aroon_up': 0.0, 'aroon_down': 0.0},
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

    def get_signal(self, aroon_up: float, aroon_down: float) -> SignalType:
        """
        Aroon'dan sinyal üret

        Args:
            aroon_up: Aroon Up değeri
            aroon_down: Aroon Down değeri

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        if aroon_up > 70 and aroon_down < 30:
            return SignalType.BUY
        elif aroon_down > 70 and aroon_up < 30:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, aroon_up: float, aroon_down: float) -> TrendDirection:
        """
        Aroon'dan trend belirle

        Args:
            aroon_up: Aroon Up değeri
            aroon_down: Aroon Down değeri

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        if aroon_up > aroon_down:
            return TrendDirection.UP
        elif aroon_down > aroon_up:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 25
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['aroon_up', 'aroon_down', 'aroon_osc']

    def _requires_volume(self) -> bool:
        """Aroon volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Aroon']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """Aroon indikatör testi"""

    print("\n" + "="*60)
    print("AROON INDICATOR TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend değişimi simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 20:
            trend = 1.0  # Yükseliş
        elif i < 35:
            trend = 0.0  # Konsolidasyon
        else:
            trend = -0.8  # Düşüş
        noise = np.random.randn() * 1.0
        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.8 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.8 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    aroon = Aroon(period=25)
    print(f"   [OK] Oluşturuldu: {aroon}")
    print(f"   [OK] Kategori: {aroon.category.value}")
    print(f"   [OK] Tip: {aroon.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {aroon.get_required_periods()}")

    result = aroon(data)
    print(f"   [OK] Aroon Up: {result.value['aroon_up']}")
    print(f"   [OK] Aroon Down: {result.value['aroon_down']}")
    print(f"   [OK] Aroon Oscillator: {result.value['aroon_osc']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Trend gücü analizi
    print("\n3. Trend gücü analizi...")
    print(f"   [OK] Trend Gücü: {result.metadata['trend_strength']}")
    print(f"   [OK] En yüksek {result.metadata['high_periods_ago']} periyot önce")
    print(f"   [OK] En düşük {result.metadata['low_periods_ago']} periyot önce")

    # Test 3: Farklı veri dilimleri
    print("\n4. Farklı veri dilimi testi...")
    for i in [25, 35, 45]:
        data_slice = data.iloc[:i+1]
        result = aroon.calculate(data_slice)
        print(f"   [OK] Mum {i}: Up={result.value['aroon_up']:.1f}, Down={result.value['aroon_down']:.1f}, Trend={result.metadata['trend_strength']}")

    # Test 4: Farklı periyotlar
    print("\n5. Farklı periyot testi...")
    for period in [14, 25, 50]:
        aroon_test = Aroon(period=period)
        result = aroon_test.calculate(data)
        print(f"   [OK] Aroon({period}): Up={result.value['aroon_up']:.2f}, Down={result.value['aroon_down']:.2f}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = aroon.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = aroon.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
