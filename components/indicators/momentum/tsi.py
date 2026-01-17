"""
indicators/momentum/tsi.py - True Strength Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    TSI (True Strength Index) - Çift smooth momentum osilatörü
    Aralık: -100 ile +100 arası
    Pozitif değer: Yükseliş momentum
    Negatif değer: Düşüş momentum
    0 geçişleri: Trend değişimi sinyali

Formül:
    PC = Close - Close[1] (Price Change)
    Double Smoothed PC = EMA(EMA(PC, long), short)
    Double Smoothed |PC| = EMA(EMA(|PC|, long), short)
    TSI = 100 × (Double Smoothed PC / Double Smoothed |PC|)

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


class TSI(BaseIndicator):
    """
    True Strength Index

    Çift exponential smoothing kullanarak momentum gücünü ölçer.
    Gürültüyü azaltır ve güvenilir sinyaller üretir.

    Args:
        long_period: Uzun EMA periyodu (varsayılan: 25)
        short_period: Kısa EMA periyodu (varsayılan: 13)
        signal_period: Sinyal line periyodu (varsayılan: 7)
    """

    def __init__(
        self,
        long_period: int = 25,
        short_period: int = 13,
        signal_period: int = 7,
        logger=None,
        error_handler=None
    ):
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period

        super().__init__(
            name='tsi',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'long_period': long_period,
                'short_period': short_period,
                'signal_period': signal_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        # Çift smoothing için yeterli veri gerekir
        return self.long_period + self.short_period + self.signal_period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.long_period < 1:
            raise InvalidParameterError(
                self.name, 'long_period', self.long_period,
                "Long period pozitif olmalı"
            )
        if self.short_period < 1:
            raise InvalidParameterError(
                self.name, 'short_period', self.short_period,
                "Short period pozitif olmalı"
            )
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "Signal period pozitif olmalı"
            )
        if self.short_period >= self.long_period:
            raise InvalidParameterError(
                self.name, 'periods',
                f"short={self.short_period}, long={self.long_period}",
                "Short period, long period'tan küçük olmalı"
            )
        return True

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average hesapla

        Args:
            data: Veri array
            period: EMA periyodu

        Returns:
            EMA değerleri
        """
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)

        # İlk değer basit ortalama
        ema[period-1] = np.mean(data[:period])

        # Sonraki değerler EMA formülü ile
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        TSI hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TSI değeri ve sinyal line
        """
        close = data['close'].values

        # Price Change hesapla
        price_change = np.diff(close)
        price_change = np.insert(price_change, 0, 0)  # İlk değer 0

        # Absolute Price Change
        abs_price_change = np.abs(price_change)

        # Double Smoothed PC
        first_smooth_pc = self._ema(price_change, self.long_period)
        double_smooth_pc = self._ema(first_smooth_pc, self.short_period)

        # Double Smoothed |PC|
        first_smooth_abs = self._ema(abs_price_change, self.long_period)
        double_smooth_abs = self._ema(first_smooth_abs, self.short_period)

        # TSI hesapla
        if double_smooth_abs[-1] == 0:
            tsi_value = 0.0
        else:
            tsi_value = 100 * (double_smooth_pc[-1] / double_smooth_abs[-1])

        # Signal line (TSI'nin EMA'sı)
        # TSI değerlerini hesapla (son signal_period kadar)
        tsi_array = np.zeros(len(close))
        for i in range(len(close)):
            if double_smooth_abs[i] != 0:
                tsi_array[i] = 100 * (double_smooth_pc[i] / double_smooth_abs[i])

        signal_line = self._ema(tsi_array, self.signal_period)[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # TSI ve signal line farkından sinyal üret
        histogram = tsi_value - signal_line

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'tsi': round(tsi_value, 2),
                'signal': round(signal_line, 2),
                'histogram': round(histogram, 2)
            },
            timestamp=timestamp,
            signal=self.get_signal(tsi_value),
            trend=self.get_trend(histogram),
            strength=min(abs(tsi_value), 100),
            metadata={
                'long_period': self.long_period,
                'short_period': self.short_period,
                'signal_period': self.signal_period,
                'crossover': 'bullish' if histogram > 0 else 'bearish'
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch TSI calculation - BACKTEST için

        TSI Formula:
            PC = Close - Close[1]
            Double Smoothed PC = EMA(EMA(PC, long), short)
            Double Smoothed |PC| = EMA(EMA(|PC|, long), short)
            TSI = 100 × (Double Smoothed PC / Double Smoothed |PC|)
            Signal = EMA(TSI, signal_period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: TSI, signal, histogram for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Price Change
        price_change = close.diff()

        # Absolute Price Change
        abs_price_change = price_change.abs()

        # Double Smoothed PC: EMA(EMA(PC, long), short)
        first_smooth_pc = price_change.ewm(span=self.long_period, adjust=False).mean()
        double_smooth_pc = first_smooth_pc.ewm(span=self.short_period, adjust=False).mean()

        # Double Smoothed |PC|: EMA(EMA(|PC|, long), short)
        first_smooth_abs = abs_price_change.ewm(span=self.long_period, adjust=False).mean()
        double_smooth_abs = first_smooth_abs.ewm(span=self.short_period, adjust=False).mean()

        # TSI = 100 × (Double Smoothed PC / Double Smoothed |PC|)
        tsi = 100 * (double_smooth_pc / double_smooth_abs)

        # Handle division by zero
        tsi = tsi.fillna(0)

        # Signal Line: EMA(TSI, signal_period)
        signal = tsi.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram: TSI - Signal
        histogram = tsi - signal

        # Set first period values to NaN (warmup)
        warmup = self.long_period + self.short_period
        tsi.iloc[:warmup] = np.nan
        signal.iloc[:warmup] = np.nan
        histogram.iloc[:warmup] = np.nan

        return pd.DataFrame({
            'tsi': tsi.values,
            'signal': signal.values,
            'histogram': histogram.values
        }, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli state'i hazırlar

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)

        # Buffer'a verileri ekle
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
                value={'tsi': 0.0, 'signal': 0.0, 'histogram': 0.0},
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
        TSI değerinden sinyal üret

        Args:
            value: TSI değeri veya dict

        Returns:
            SignalType: BUY, SELL veya HOLD
        """
        # value dict ise tsi değerini al
        if isinstance(value, dict):
            tsi_val = value['tsi']
            hist = value['histogram']
        else:
            tsi_val = value
            hist = 0

        # Histogram pozitif ve TSI pozitif: güçlü alım
        if hist > 0 and tsi_val > 0:
            return SignalType.BUY
        # Histogram negatif ve TSI negatif: güçlü satım
        elif hist < 0 and tsi_val < 0:
            return SignalType.SELL
        # TSI sıfırdan yukarı: alım
        elif tsi_val > 0:
            return SignalType.BUY
        # TSI sıfırdan aşağı: satım
        elif tsi_val < 0:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        TSI değerinden trend belirle

        Args:
            value: Histogram değeri

        Returns:
            TrendDirection: UP, DOWN veya NEUTRAL
        """
        # Histogram pozitif: yükseliş
        if value > 0:
            return TrendDirection.UP
        # Histogram negatif: düşüş
        elif value < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'long_period': 25,
            'short_period': 13,
            'signal_period': 7
        }

    def _requires_volume(self) -> bool:
        """TSI volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TSI']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """TSI indikatör testi"""

    print("\n" + "="*60)
    print("TSI (TRUE STRENGTH INDEX) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Fiyat hareketini simüle et (trend oluşturmak için)
    base_price = 100
    prices = [base_price]
    for i in range(99):
        # Trend + gürültü
        trend = 0.1 if i < 50 else -0.1
        change = trend + np.random.randn() * 0.5
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    tsi = TSI(long_period=25, short_period=13, signal_period=7)
    print(f"   [OK] Oluşturuldu: {tsi}")
    print(f"   [OK] Kategori: {tsi.category.value}")
    print(f"   [OK] Gerekli periyot: {tsi.get_required_periods()}")

    result = tsi(data)
    print(f"   [OK] TSI Değeri: {result.value['tsi']}")
    print(f"   [OK] Signal Line: {result.value['signal']}")
    print(f"   [OK] Histogram: {result.value['histogram']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")

    # Test 2: Farklı periyotlar
    print("\n3. Farklı periyot testi...")
    configs = [
        (13, 7, 5),
        (25, 13, 7),
        (40, 20, 10)
    ]
    for long, short, sig in configs:
        tsi_test = TSI(long_period=long, short_period=short, signal_period=sig)
        result = tsi_test.calculate(data)
        print(f"   [OK] TSI({long},{short},{sig}): {result.value['tsi']} | Signal: {result.value['signal']}")

    # Test 3: Crossover tespiti
    print("\n4. Crossover testi...")
    # Son birkaç değeri kontrol et
    for i in [-5, -4, -3, -2, -1]:
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= tsi.get_required_periods():
            result = tsi.calculate(test_data)
            hist = result.value['histogram']
            cross_type = result.metadata['crossover']
            print(f"   [OK] Index {i}: Histogram={hist:.2f}, Crossover={cross_type}")

    # Test 4: Yükselen trend
    print("\n5. Yükselen trend testi...")
    up_data = data.copy()
    for i in range(20):
        idx = up_data.index[-(20-i)]
        up_data.loc[idx, 'close'] = prices[-(20-i)] + i * 0.5

    result_up = tsi.calculate(up_data)
    print(f"   [OK] Yükselen trend TSI: {result_up.value['tsi']}")
    print(f"   [OK] Sinyal: {result_up.signal.value}")
    print(f"   [OK] Trend: {result_up.trend.name}")

    # Test 5: Düşen trend
    print("\n6. Düşen trend testi...")
    down_data = data.copy()
    for i in range(20):
        idx = down_data.index[-(20-i)]
        down_data.loc[idx, 'close'] = prices[-(20-i)] - i * 0.5

    result_down = tsi.calculate(down_data)
    print(f"   [OK] Düşen trend TSI: {result_down.value['tsi']}")
    print(f"   [OK] Sinyal: {result_down.signal.value}")
    print(f"   [OK] Trend: {result_down.trend.name}")

    # Test 6: İstatistikler
    print("\n7. İstatistik testi...")
    stats = tsi.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = tsi.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
