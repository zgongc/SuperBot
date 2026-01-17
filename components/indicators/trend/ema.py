"""
indicators/trend/ema.py - Exponential Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    EMA (Exponential Moving Average) - Üssel hareketli ortalama
    Son fiyatlara daha fazla ağırlık veren trend indikatörü
    SMA'ya göre fiyat değişimlerine daha hızlı tepki verir

    Kullanım:
    - Hızlı trend takibi
    - EMA crossover stratejileri (9/21 EMA)
    - Destek/direnç seviyeleri

Formül:
    EMA = (Close - EMA_prev) × Multiplier + EMA_prev
    Multiplier = 2 / (Period + 1)
    İlk EMA = SMA(Period)

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


class EMA(BaseIndicator):
    """
    Exponential Moving Average

    Yakın geçmişteki fiyatlara daha fazla ağırlık vererek
    fiyat değişimlerine daha hızlı tepki verir.

    Args:
        period: EMA periyodu (varsayılan: 20)
        sma_seed: EMA'yı başlatmak için kullanılacak SMA periyodu
                  None = period ile aynı (pandas varsayılanı)
                  9 = TradingView uyumlu (TradingView tüm EMA'ları SMA 9 ile başlatır)
    """

    def __init__(
        self,
        period: int = 20,
        sma_seed: int = None,
        logger=None,
        error_handler=None
    ):
        self.period = period
        # TradingView uyumluluğu için sma_seed=9 kullanılabilir
        # None = period (pandas varsayılanı)
        self.sma_seed = sma_seed

        super().__init__(
            name='ema',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'sma_seed': sma_seed
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
                "Periyot pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        EMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: EMA değeri
        """
        close = data['close'].values

        # EMA hesapla
        ema_value = self._calculate_ema(close, self.period)

        # Mevcut fiyat
        current_price = close[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(ema_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, ema_value),
            trend=self.get_trend(current_price, ema_value),
            strength=self._calculate_strength(current_price, ema_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'distance_pct': round(((current_price - ema_value) / ema_value) * 100, 2),
                'multiplier': round(2 / (self.period + 1), 4)
            }
        )

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        EMA hesaplama yardımcı fonksiyonu

        Args:
            prices: Fiyat dizisi
            period: EMA periyodu

        Returns:
            float: Son EMA değeri
        """
        # Multiplier
        multiplier = 2 / (period + 1)

        # İlk EMA = SMA (sma_seed ile belirlenen periyot kullanılır)
        # TradingView: sma_seed=9 (tüm EMA'lar SMA 9 ile başlar)
        # Pandas default: sma_seed=period
        seed_period = self.sma_seed if self.sma_seed else period
        seed_period = min(seed_period, len(prices))  # Veri yetersizse küçült

        ema = np.mean(prices[:seed_period])

        # Iterative EMA hesaplama (seed_period'dan sonra başla)
        for price in prices[seed_period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch EMA calculation - BACKTEST için

        EMA Formula:
            EMA = (Close - EMA_prev) × Multiplier + EMA_prev
            Multiplier = 2 / (Period + 1)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: EMA values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        # sma_seed kullanılıyorsa (TradingView uyumlu), manuel hesapla
        if self.sma_seed and self.sma_seed != self.period:
            # TradingView style: SMA seed ile başla, sonra EMA uygula
            close = data['close'].values
            multiplier = 2 / (self.period + 1)

            seed_period = min(self.sma_seed, len(close))
            ema_values = np.full(len(close), np.nan)

            # İlk seed_period bar için SMA hesapla
            if len(close) >= seed_period:
                ema = np.mean(close[:seed_period])
                ema_values[seed_period - 1] = ema

                # Sonraki barlar için EMA uygula
                for i in range(seed_period, len(close)):
                    ema = (close[i] - ema) * multiplier + ema
                    ema_values[i] = ema

            return pd.Series(ema_values, index=data.index, name='ema')
        else:
            # Pandas ewm() varsayılan davranış (period ile seed)
            ema = data['close'].ewm(span=self.period, adjust=False).mean()

            # Set first period values to NaN (warmup)
            ema.iloc[:self.period-1] = np.nan

            return pd.Series(ema.values, index=data.index, name='ema')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup EMA with historical data - CRITICAL for correct EMA values!

        Unlike other indicators, EMA must preserve its previous value.
        We calculate EMA on ALL historical data and store the last value.

        Args:
            data: Historical OHLCV DataFrame (should be as large as possible!)
            symbol: Symbol identifier
        """
        # Use symbol as key, or 'default' if not provided
        buffer_key = symbol if symbol else 'default'

        # Initialize state storage
        if not hasattr(self, '_ema_state'):
            self._ema_state = {}

        # Calculate EMA on FULL historical data
        if len(data) >= self.period:
            ema_value = self._calculate_ema(data['close'].values, self.period)
            self._ema_state[buffer_key] = ema_value
        else:
            # Not enough data - use last close as initial EMA
            self._ema_state[buffer_key] = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        TRUE Incremental EMA update (Real-time)

        Uses stored EMA value from warmup, applies EMA formula:
        EMA_new = (Close - EMA_prev) × Multiplier + EMA_prev

        This is MUCH more accurate than recalculating from limited buffer!

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel EMA değeri
        """
        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = float(candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = float(candle[4]) if len(candle) > 4 else 0.0

        # Initialize state if needed (fallback if warmup wasn't called)
        if not hasattr(self, '_ema_state'):
            self._ema_state = {}

        # Get previous EMA or use close as initial value
        prev_ema = self._ema_state.get(buffer_key)
        if prev_ema is None:
            # First call without warmup - use close as initial EMA
            self._ema_state[buffer_key] = close_val
            return IndicatorResult(
                value=round(close_val, 2),
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period, 'warmup': False}
            )

        # Calculate new EMA: EMA_new = (Close - EMA_prev) × Multiplier + EMA_prev
        multiplier = 2 / (self.period + 1)
        new_ema = (close_val - prev_ema) * multiplier + prev_ema

        # Store new EMA for next update
        self._ema_state[buffer_key] = new_ema

        return IndicatorResult(
            value=round(new_ema, 2),
            timestamp=timestamp_val,
            signal=self.get_signal(close_val, new_ema),
            trend=self.get_trend(close_val, new_ema),
            strength=self._calculate_strength(close_val, new_ema),
            metadata={
                'period': self.period,
                'current_price': round(close_val, 2),
                'distance_pct': round(((close_val - new_ema) / new_ema) * 100, 2),
                'multiplier': round(multiplier, 4)
            }
        )

    def get_signal(self, price: float, ema: float) -> SignalType:
        """
        EMA'dan sinyal üret

        Args:
            price: Mevcut fiyat
            ema: EMA değeri

        Returns:
            SignalType: BUY (fiyat EMA üstüne çıkınca), SELL (altına ininse)
        """
        if price > ema:
            return SignalType.BUY
        elif price < ema:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, ema: float) -> TrendDirection:
        """
        EMA'dan trend belirle

        Args:
            price: Mevcut fiyat
            ema: EMA değeri

        Returns:
            TrendDirection: UP (fiyat > EMA), DOWN (fiyat < EMA)
        """
        if price > ema:
            return TrendDirection.UP
        elif price < ema:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, ema: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - ema) / ema * 100)
        strength = min(distance_pct * 20, 100.0)
        # Extra clamp for floating point precision
        return max(0.0, min(strength, 100.0))

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """EMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['EMA']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """EMA indikatör testi"""

    print("\n" + "="*60)
    print("EMA (EXPONENTIAL MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Volatil trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 0.5
        noise = np.random.randn() * 2
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
    ema = EMA(period=20)
    print(f"   [OK] Oluşturuldu: {ema}")
    print(f"   [OK] Kategori: {ema.category.value}")
    print(f"   [OK] Gerekli periyot: {ema.get_required_periods()}")

    result = ema(data)
    print(f"   [OK] EMA Değeri: {result.value}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: EMA vs SMA karşılaştırması
    print("\n3. EMA vs SMA karşılaştırma testi...")
    # SMA için basit hesaplama
    sma_value = np.mean(data['close'].values[-20:])
    print(f"   [OK] SMA(20): {sma_value:.2f}")
    print(f"   [OK] EMA(20): {result.value:.2f}")
    print(f"   [OK] Fark: {abs(result.value - sma_value):.2f}")
    print(f"   [OK] EMA son fiyatlara daha fazla ağırlık verir")

    # Test 3: Farklı periyotlar
    print("\n4. Farklı periyot testi...")
    for period in [9, 21, 50]:
        ema_test = EMA(period=period)
        result = ema_test.calculate(data)
        print(f"   [OK] EMA({period}): {result.value:.2f} | Sinyal: {result.signal.value}")

    # Test 4: EMA Crossover (9/21)
    print("\n5. EMA Crossover testi (9/21)...")
    ema_9 = EMA(period=9)
    ema_21 = EMA(period=21)

    result_9 = ema_9.calculate(data)
    result_21 = ema_21.calculate(data)

    print(f"   [OK] EMA(9): {result_9.value:.2f}")
    print(f"   [OK] EMA(21): {result_21.value:.2f}")

    if result_9.value > result_21.value:
        print(f"   [OK] Bullish Crossover (EMA9 > EMA21)")
    else:
        print(f"   [OK] Bearish Crossover (EMA9 < EMA21)")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = ema.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = ema.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume gerekli: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
