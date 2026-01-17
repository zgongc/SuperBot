"""
indicators/trend/kama.py - Kaufman Adaptive Moving Average

Version: 1.0.0
Date: 2025-11-20
Author: SuperBot Team

Açıklama:
    KAMA (Kaufman Adaptive Moving Average)
    Perry Kaufman tarafından geliştirilmiş adaptive moving average
    Volatilite ve trend gücüne göre dinamik smoothing

    Kullanım:
    - Trend following
    - Dynamic support/resistance
    - Entry/Exit signals
    - Adaptive filtering

Formül:
    ER = abs(change) / sum(abs(price changes))  # Efficiency Ratio
    SC = [ER * (fastest - slowest) + slowest]^2  # Smoothing Constant
    KAMA = KAMA_prev + SC * (price - KAMA_prev)

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import sys
from pathlib import Path

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from collections import deque
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class KAMA(BaseIndicator):
    """
    Kaufman Adaptive Moving Average

    Trend gücüne göre kendini ayarlayan moving average.
    Güçlü trendde hızlı, ranging marketlerde yavaş hareket eder.

    Args:
        period: ER hesaplama periyodu (varsayılan: 10)
        fast_period: Hızlı EMA periyodu (varsayılan: 2)
        slow_period: Yavaş EMA periyodu (varsayılan: 30)
    """

    def __init__(
        self,
        period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Buffers for incremental calculation
        self.prices = deque(maxlen=period + 1)
        self.kama_value = None

        # Smoothing constants
        self.fast_sc = 2.0 / (fast_period + 1)
        self.slow_sc = 2.0 / (slow_period + 1)

        super().__init__(
            name='kama',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'fast_period': fast_period,
                'slow_period': slow_period
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
                "Period pozitif olmalı"
            )
        if self.fast_period < 1 or self.fast_period >= self.slow_period:
            raise InvalidParameterError(
                self.name, 'fast_period', self.fast_period,
                "Fast period 1 ile slow period arası olmalı"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch hesaplama (Backtest için)

        Tüm veriyi vektörel olarak hesaplar.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: KAMA değerleri
        """
        close = data['close'].values
        n = len(close)
        kama = np.full(n, np.nan)

        if n < self.period + 1:
            return pd.DataFrame({'kama': kama}, index=data.index)

        # İlk KAMA değeri = ilk close
        kama[self.period] = close[self.period]

        for i in range(self.period + 1, n):
            # Efficiency Ratio
            change = abs(close[i] - close[i - self.period])
            volatility = np.sum(np.abs(np.diff(close[i - self.period:i + 1])))

            if volatility == 0:
                er = 0
            else:
                er = change / volatility

            # Smoothing Constant
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

            # KAMA
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

        return pd.DataFrame({'kama': kama}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için gerekli

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Sembol adı (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        # Batch calculate to get KAMA values
        batch_result = self.calculate_batch(data)
        kama_values = batch_result['kama'].values
        valid_kama = kama_values[~np.isnan(kama_values)]

        if len(valid_kama) > 0:
            self.kama_value = valid_kama[-1]
        else:
            self.kama_value = data['close'].iloc[-1]  # Fallback

        # Populate prices buffer
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Güncel KAMA değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close = candle['close']
            timestamp_val = int(candle['timestamp']) if 'timestamp' in candle else 0
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close = candle[4] if len(candle) > 4 else 0

        self.prices.append(close)

        # İlk değer
        if self.kama_value is None:
            if len(self.prices) >= self.period + 1:
                self.kama_value = self.prices[self.period]
            else:
                return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Yeterli veri yok
        if len(self.prices) < self.period + 1:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Efficiency Ratio
        prices_array = np.array(self.prices)
        change = abs(prices_array[-1] - prices_array[0])
        volatility = np.sum(np.abs(np.diff(prices_array)))

        if volatility == 0:
            er = 0
        else:
            er = change / volatility

        # Smoothing Constant
        sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2

        # KAMA güncelle
        self.kama_value = self.kama_value + sc * (close - self.kama_value)

        timestamp = timestamp_val

        # Trend belirleme
        trend = self.get_trend(close, self.kama_value)
        signal = self.get_signal(close, self.kama_value)

        return IndicatorResult(
            value={'kama': round(self.kama_value, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=self._calculate_strength(close, self.kama_value, er),
            metadata={
                'period': self.period,
                'efficiency_ratio': round(er, 4),
                'smoothing_constant': round(sc, 6),
                'distance_pct': round((close - self.kama_value) / self.kama_value * 100, 2)
            }
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        KAMA hesapla (son değer)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: KAMA değeri
        """
        # Populate buffers
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

        # Batch calculate
        batch_result = self.calculate_batch(data)
        kama_series = batch_result['kama'].values

        # Son geçerli değeri bul
        valid_kama = kama_series[~np.isnan(kama_series)]
        if len(valid_kama) == 0:
            return None

        kama_value = valid_kama[-1]
        self.kama_value = kama_value

        close = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # ER hesapla
        prices_array = np.array(self.prices)
        change = abs(prices_array[-1] - prices_array[0])
        volatility = np.sum(np.abs(np.diff(prices_array)))
        er = change / volatility if volatility > 0 else 0

        # Trend ve sinyal
        trend = self.get_trend(close, kama_value)
        signal = self.get_signal(close, kama_value)

        return IndicatorResult(
            value={'kama': round(kama_value, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=self._calculate_strength(close, kama_value, er),
            metadata={
                'period': self.period,
                'efficiency_ratio': round(er, 4),
                'distance_pct': round((close - kama_value) / kama_value * 100, 2)
            }
        )

    def get_signal(self, price: float, kama: float) -> SignalType:
        """
        KAMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            kama: KAMA değeri

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        if price > kama:
            return SignalType.BUY
        elif price < kama:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, kama: float) -> TrendDirection:
        """
        KAMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            kama: KAMA değeri

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        diff_pct = (price - kama) / kama * 100

        if diff_pct > 0.5:
            return TrendDirection.UP
        elif diff_pct < -0.5:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, kama: float, er: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        # ER ve price distance kombinasyonu
        distance_pct = abs((price - kama) / kama * 100)
        strength = (er * 50) + (min(distance_pct, 5) * 10)
        return min(strength, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 10,
            'fast_period': 2,
            'slow_period': 30
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['kama']

    def _requires_volume(self) -> bool:
        """KAMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['KAMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """KAMA indikatör testi"""

    # Windows console UTF-8 desteği
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("KAMA (KAUFMAN ADAPTIVE MA) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   [OK] {len(data)} mum oluşturuldu")
    print(f"   [OK] Fiyat aralığı: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Temel hesaplama
    print("\n2. Temel hesaplama testi...")
    kama = KAMA(period=10, fast_period=2, slow_period=30)
    print(f"   [OK] Oluşturuldu: {kama}")
    print(f"   [OK] Kategori: {kama.category.value}")
    print(f"   [OK] Gerekli periyot: {kama.get_required_periods()}")

    result = kama(data)
    print(f"   [OK] KAMA: {result.value['kama']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] ER: {result.metadata['efficiency_ratio']}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = kama.calculate_batch(data)
    print(f"   [OK] Batch result shape: {batch_result.shape}")
    print(f"   [OK] Son 5 KAMA değeri:")
    print(batch_result['kama'].tail())

    # Test 3: Update metodu
    print("\n4. Update metodu testi...")
    kama2 = KAMA(period=10)
    init_data = data.head(50)
    kama2.calculate(init_data)

    # Yeni 10 mum ekle
    for i in range(50, 60):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low']
        }
        update_result = kama2.update(candle)
        if update_result:
            print(f"   [OK] Bar {i}: KAMA={update_result.value['kama']:.2f}, "
                  f"ER={update_result.metadata['efficiency_ratio']:.4f}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
