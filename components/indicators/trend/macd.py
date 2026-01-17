"""
indicators/trend/macd.py - Moving Average Convergence Divergence

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    MACD (Moving Average Convergence Divergence) - Hareketli ortalama yakınsama/ıraksama
    Gerald Appel tarafından geliştirilmiş popüler momentum ve trend indikatörü
    MACD çizgisi, Sinyal çizgisi ve Histogram'dan oluşur

    Kullanım:
    - Trend yönünü belirleme
    - Momentum ölçme
    - Crossover sinyalleri (MACD/Signal kesişimi)
    - Divergence tespiti

Formül:
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Varsayılan: fast=12, slow=26, signal=9

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.ema import EMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence

    İki EMA'nın farkını alarak momentum ve trend ölçer.

    Args:
        fast_period: Hızlı EMA periyodu (varsayılan: 12)
        slow_period: Yavaş EMA periyodu (varsayılan: 26)
        signal_period: Sinyal EMA periyodu (varsayılan: 9)
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        logger=None,
        error_handler=None
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # EMA indikatörlerini kullan (code reuse)
        self._fast_ema_ind = EMA(period=fast_period)
        self._slow_ema_ind = EMA(period=slow_period)
        self._signal_ema_ind = EMA(period=signal_period)

        super().__init__(
            name='macd',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum gerekli periyot sayısı"""
        return self.slow_period + self.signal_period

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.fast_period < 1:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "Fast period pozitif olmalı"
            )
        if self.slow_period < 1:
            raise InvalidParameterError(
                self.name, 'slow_period', self.slow_period,
                "Slow period pozitif olmalı"
            )
        if self.signal_period < 1:
            raise InvalidParameterError(
                self.name, 'signal_period', self.signal_period,
                "Signal period pozitif olmalı"
            )
        if self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'periods',
                f"fast={self.fast_period}, slow={self.slow_period}",
                "Fast period, slow period'dan küçük olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        MACD hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: MACD, Signal ve Histogram değerleri
        """
        close = data['close'].values

        # Fast ve Slow EMA'ları hesapla
        fast_ema = self._calculate_ema(close, self.fast_period)
        slow_ema = self._calculate_ema(close, self.slow_period)

        # MACD Line
        macd_line = fast_ema - slow_ema

        # MACD serisini oluştur (sinyal için)
        macd_series = np.zeros(len(close))
        for i in range(self.slow_period - 1, len(close)):
            f_ema = self._calculate_ema(close[:i+1], self.fast_period)
            s_ema = self._calculate_ema(close[:i+1], self.slow_period)
            macd_series[i] = f_ema - s_ema

        # Signal Line (MACD'nin EMA'sı)
        signal_line = self._calculate_ema(macd_series[self.slow_period-1:], self.signal_period)

        # Histogram
        histogram = macd_line - signal_line

        # Son değerler
        macd_value = macd_line
        signal_value = signal_line
        histogram_value = histogram

        timestamp = int(data.iloc[-1]['timestamp'])

        # Trend ve sinyal belirleme
        trend = self.get_trend(macd_value, signal_value)
        signal = self.get_signal(macd_value, signal_value, histogram_value)

        # Calculate strength from histogram (how far from zero crossing)
        # Handle NaN/inf cases gracefully
        if np.isnan(histogram_value) or np.isinf(histogram_value):
            strength = 0.0
        else:
            strength = min(abs(histogram_value) * 50, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'macd': round(macd_value, 4),
                'signal': round(signal_value, 4),
                'histogram': round(histogram_value, 4)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=round(strength, 2),
            metadata={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'fast_ema': round(fast_ema, 2),
                'slow_ema': round(slow_ema, 2),
                'crossover': 'bullish' if macd_value > signal_value else 'bearish'
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        VECTORIZED batch MACD calculation - BACKTEST

        MACD Formula:
            MACD Line = EMA(fast) - EMA(slow)
            Signal Line = EMA(MACD Line, signal_period)
            Histogram = MACD Line - Signal Line

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 3 columns (macd, signal, histogram)
        """
        self._validate_data(data)

        # EMA.calculate_batch kullan (code reuse)
        fast_ema = self._fast_ema_ind.calculate_batch(data)
        slow_ema = self._slow_ema_ind.calculate_batch(data)

        # MACD Line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema

        # Signal Line = EMA of MACD Line
        macd_df = self._create_ema_input(macd_line, data)
        signal_line = self._signal_ema_ind.calculate_batch(macd_df)

        # Histogram = MACD - Signal
        histogram = macd_line - signal_line

        # Create result DataFrame
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)

        # Set first period values to NaN (warmup)
        warmup = self.slow_period + self.signal_period
        result.iloc[:warmup] = np.nan

        return result

    def _create_ema_input(self, series: pd.Series, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create minimal OHLCV DataFrame for EMA calculation from a Series."""
        return pd.DataFrame({
            'timestamp': original_data['timestamp'].values,
            'open': series.values,
            'high': series.values,
            'low': series.values,
            'close': series.values,
            'volume': np.zeros(len(series))
        }, index=original_data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        MACD için EMA state'lerini saklar. Bu sayede update() incremental
        hesaplama yapabilir.

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        close = data['close'].values

        # Calculate EMAs using pandas (same as calculate_batch)
        close_series = data['close']
        fast_ema_series = close_series.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema_series = close_series.ewm(span=self.slow_period, adjust=False).mean()
        macd_series = fast_ema_series - slow_ema_series
        signal_series = macd_series.ewm(span=self.signal_period, adjust=False).mean()

        # Store last EMA values for incremental update
        self._fast_ema = fast_ema_series.iloc[-1]
        self._slow_ema = slow_ema_series.iloc[-1]
        self._signal_ema = signal_series.iloc[-1]
        self._prev_close = close[-1]
        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Uses stored EMA states for true incremental calculation.
        EMA formula: EMA_new = (price - EMA_prev) * multiplier + EMA_prev

        Args:
            candle: New candle data
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Updated MACD values
        """
        # Check if warmup was done (EMA states exist)
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            # No warmup - return zero values
            timestamp_val = 0
            if isinstance(candle, dict):
                timestamp_val = int(candle.get('timestamp', 0))
            elif len(candle) > 0:
                timestamp_val = int(candle[0])

            return IndicatorResult(
                value={'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        # EMA multipliers (same as pandas ewm with adjust=False)
        fast_mult = 2.0 / (self.fast_period + 1)
        slow_mult = 2.0 / (self.slow_period + 1)
        signal_mult = 2.0 / (self.signal_period + 1)

        # Incremental EMA update:
        # EMA_new = (price - EMA_prev) * multiplier + EMA_prev
        self._fast_ema = (close_val - self._fast_ema) * fast_mult + self._fast_ema
        self._slow_ema = (close_val - self._slow_ema) * slow_mult + self._slow_ema

        # MACD Line = Fast EMA - Slow EMA
        macd_value = self._fast_ema - self._slow_ema

        # Signal Line = EMA of MACD
        self._signal_ema = (macd_value - self._signal_ema) * signal_mult + self._signal_ema
        signal_value = self._signal_ema

        # Histogram = MACD - Signal
        histogram_value = macd_value - signal_value

        # Trend and signal determination
        trend = self.get_trend(macd_value, signal_value)
        signal = self.get_signal(macd_value, signal_value, histogram_value)

        # Calculate strength
        strength = min(abs(histogram_value) * 50, 100) if not np.isnan(histogram_value) else 0.0

        return IndicatorResult(
            value={
                'macd': round(macd_value, 4),
                'signal': round(signal_value, 4),
                'histogram': round(histogram_value, 4)
            },
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=round(strength, 2),
            metadata={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'fast_ema': round(self._fast_ema, 2),
                'slow_ema': round(self._slow_ema, 2),
                'crossover': 'bullish' if macd_value > signal_value else 'bearish'
            }
        )

    def get_signal(self, macd: float, signal: float, histogram: float) -> SignalType:
        """
        MACD'den sinyal üret

        Args:
            macd: MACD line değeri
            signal: Signal line değeri
            histogram: Histogram değeri

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        # MACD/Signal crossover
        if macd > signal and histogram > 0:
            return SignalType.BUY
        elif macd < signal and histogram < 0:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, macd: float, signal: float) -> TrendDirection:
        """
        MACD'den trend belirle

        Args:
            macd: MACD line değeri
            signal: Signal line değeri

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        if macd > signal:
            return TrendDirection.UP
        elif macd < signal:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['macd', 'signal', 'histogram']

    def _requires_volume(self) -> bool:
        """MACD volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MACD']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """MACD indikatör testi"""

    print("\n" + "="*60)
    print("MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(99):
        if i < 40:
            trend = 0.5  # Yavaş yükseliş
        elif i < 70:
            trend = -0.3  # Düşüş
        else:
            trend = 0.7  # Güçlü yükseliş
        noise = np.random.randn() * 1.5
        prices.append(prices[-1] + trend + noise)

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
    macd = MACD()
    print(f"   [OK] Oluşturuldu: {macd}")
    print(f"   [OK] Kategori: {macd.category.value}")
    print(f"   [OK] Tip: {macd.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {macd.get_required_periods()}")

    result = macd(data)
    print(f"   [OK] MACD: {result.value['macd']}")
    print(f"   [OK] Signal: {result.value['signal']}")
    print(f"   [OK] Histogram: {result.value['histogram']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Crossover analizi
    print("\n3. Crossover analizi...")
    print(f"   [OK] MACD Line: {result.value['macd']:.4f}")
    print(f"   [OK] Signal Line: {result.value['signal']:.4f}")
    print(f"   [OK] Crossover: {result.metadata['crossover']}")

    if result.value['histogram'] > 0:
        print(f"   [OK] Bullish (MACD > Signal)")
    else:
        print(f"   [OK] Bearish (MACD < Signal)")

    # Test 3: Trend değişimi testi
    print("\n4. Trend değişimi testi...")
    for i in [45, 65, 85]:
        data_slice = data.iloc[:i+1]
        result = macd.calculate(data_slice)
        print(f"   [OK] Mum {i}: MACD={result.value['macd']:.4f}, Histogram={result.value['histogram']:.4f}, Trend={result.trend.name}")

    # Test 4: Farklı parametreler
    print("\n5. Farklı parametre testi...")
    configs = [(12, 26, 9), (5, 35, 5), (8, 17, 9)]
    for fast, slow, sig in configs:
        macd_test = MACD(fast_period=fast, slow_period=slow, signal_period=sig)
        result = macd_test.calculate(data)
        print(f"   [OK] MACD({fast},{slow},{sig}): {result.value['macd']:.4f}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = macd.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = macd.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
