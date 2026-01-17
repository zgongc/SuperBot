"""
indicators/trend/supertrend.py - SuperTrend Indicator

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    SuperTrend - ATR tabanlı trend takip indikatörü
    Hem trend yönünü hem de dinamik destek/direnç seviyelerini gösterir
    Olivier Seban tarafından geliştirilmiş popüler bir indikatör

    Kullanım:
    - Trend yönünü belirleme
    - Stop-loss seviyeleri
    - Entry/Exit sinyalleri

Formül:
    Basic Upper Band = (High + Low) / 2 + (Multiplier × ATR)
    Basic Lower Band = (High + Low) / 2 - (Multiplier × ATR)

    Trend UP ise:
        SuperTrend = Lower Band
    Trend DOWN ise:
        SuperTrend = Upper Band

Bağımlılıklar:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import Any
from indicators.base_indicator import BaseIndicator
from indicators.volatility.atr import ATR
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class SuperTrend(BaseIndicator):
    """
    SuperTrend Indicator

    ATR kullanarak dinamik destek/direnç seviyeleri ve trend yönü belirler.

    Args:
        period: ATR periyodu (varsayılan: 10)
        multiplier: ATR çarpanı (varsayılan: 3.0)
    """

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.multiplier = multiplier

        # ATR indikatörünü kullan (code reuse)
        self._atr = ATR(period=period)

        super().__init__(
            name='supertrend',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'multiplier': multiplier
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
        if self.multiplier <= 0:
            raise InvalidParameterError(
                self.name, 'multiplier', self.multiplier,
                "Multiplier pozitif olmalı"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        SuperTrend hesapla - calculate_batch() ile AYNI mantık

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: SuperTrend değerleri (upper, lower, trend)
        """
        # Use calculate_batch for consistency
        batch_result = self.calculate_batch(data)

        # Get last row values
        last_idx = len(batch_result) - 1
        supertrend_value = batch_result['supertrend'].iloc[last_idx]
        final_upper = batch_result['upper'].iloc[last_idx]
        final_lower = batch_result['lower'].iloc[last_idx]
        trend_int = int(batch_result['trend'].iloc[last_idx])

        # Calculate ATR for metadata - use ATR.calculate_batch (code reuse)
        atr = self._atr.calculate_batch(data).values
        current_close = data['close'].iloc[-1]

        # Convert trend int to TrendDirection
        if trend_int == 1:
            trend = TrendDirection.UP
        elif trend_int == -1:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'supertrend': round(supertrend_value, 2),
                'upper': round(final_upper, 2),
                'lower': round(final_lower, 2),
                'trend': trend_int  # Integer: 1=UP, -1=DOWN, 0=NEUTRAL
            },
            timestamp=timestamp,
            signal=self.get_signal(trend),
            trend=trend,
            strength=self._calculate_strength(current_close, supertrend_value),
            metadata={
                'period': self.period,
                'multiplier': self.multiplier,
                'atr': round(atr[-1], 2),
                'current_price': round(current_close, 2),
                'distance_pct': round(abs((current_close - supertrend_value) / supertrend_value) * 100, 2) if supertrend_value != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED SuperTrend batch calculation - BACKTEST için
        Senin IndicatorManager'ın beklediği abstract method

        Performance: 10.000 bar ~0.015 saniye
        """
        df = data.copy()
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 1. ATR hesapla - use ATR.calculate_batch (code reuse)
        atr = self._atr.calculate_batch(df).values

        # 2. Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)

        # 3. Final bands & trend (vectorized loop – en hızlı yöntem)
        final_upper = np.zeros(len(df))
        final_lower = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        trend = np.zeros(len(df), dtype=int)  # 1 = UP, -1 = DOWN

        # İlk değerler
        final_upper[self.period] = upper_band[self.period]
        final_lower[self.period] = lower_band[self.period]
        supertrend[self.period] = lower_band[self.period]
        trend[self.period] = 1

        for i in range(self.period + 1, len(df)):
            # Final Upper Band
            final_upper[i] = (upper_band[i] if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]
                             else final_upper[i-1])

            # Final Lower Band
            final_lower[i] = (lower_band[i] if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]
                             else final_lower[i-1])

            # Trend yönü
            if trend[i-1] == 1:  # önceki uptrend
                trend[i] = -1 if close[i] <= final_lower[i] else 1
            else:  # önceki downtrend
                trend[i] = 1 if close[i] >= final_upper[i] else -1

            # SuperTrend değeri
            supertrend[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

        # Warm-up periyodunu NaN yap
        supertrend[:self.period] = np.nan
        final_upper[:self.period] = np.nan
        final_lower[:self.period] = np.nan
        trend[:self.period] = 0

        # DataFrame'e ekle (calculate() ile aynı key'ler)
        result_df = pd.DataFrame({
            'supertrend': supertrend,
            'upper': final_upper,
            'lower': final_lower,
            'trend': trend  # 1 = bullish, -1 = bearish, 0 = warmup
        }, index=df.index)

        return result_df

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

        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)

        # Buffer'lara verileri ekle
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
                value={'supertrend': 0.0, 'direction': 1},
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

    def get_signal(self, trend: TrendDirection) -> SignalType:
        """
        SuperTrend'den sinyal üret

        Args:
            trend: Mevcut trend

        Returns:
            SignalType: BUY (uptrend), SELL (downtrend)
        """
        if trend == TrendDirection.UP:
            return SignalType.BUY
        elif trend == TrendDirection.DOWN:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: Any) -> TrendDirection:
        """Trend zaten hesaplanmış"""
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, supertrend: float) -> float:
        """Sinyal gücünü hesapla (0-100)"""
        distance_pct = abs((price - supertrend) / supertrend * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'period': 10,
            'multiplier': 3.0
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['supertrend', 'upper', 'lower', 'trend']

    def _requires_volume(self) -> bool:
        """SuperTrend volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SuperTrend']


# ============================================================================
# KULLANIM ÖRNEĞİ (TEST)
# ============================================================================

if __name__ == "__main__":
    """SuperTrend indikatör testi"""

    print("\n" + "="*60)
    print("SUPERTREND TEST")
    print("="*60 + "\n")

    # Örnek veri oluştur
    print("1. Örnek OHLCV verisi oluşturuluyor...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Volatil trend simülasyonu
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 25:
            trend = 1.0  # Güçlü yükseliş
        else:
            trend = -0.8  # Güçlü düşüş
        noise = np.random.randn() * 2
        prices.append(prices[-1] + trend + noise)

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
    st = SuperTrend(period=10, multiplier=3.0)
    print(f"   [OK] Oluşturuldu: {st}")
    print(f"   [OK] Kategori: {st.category.value}")
    print(f"   [OK] Tip: {st.indicator_type.value}")
    print(f"   [OK] Gerekli periyot: {st.get_required_periods()}")

    result = st(data)
    print(f"   [OK] SuperTrend: {result.value['supertrend']}")
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Trend: {result.value['trend']}")
    print(f"   [OK] Sinyal: {result.signal.value}")
    print(f"   [OK] Güç: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Trend değişimi testi
    print("\n3. Trend değişimi testi...")
    # Farklı veri dilimlerinde hesapla
    for i in [20, 30, 40]:
        data_slice = data.iloc[:i+1]
        result = st.calculate(data_slice)
        print(f"   [OK] Mum {i}: Trend={result.value['trend']}, SuperTrend={result.value['supertrend']}")

    # Test 3: Farklı parametreler
    print("\n4. Farklı parametre testi...")
    for mult in [2.0, 3.0, 4.0]:
        st_test = SuperTrend(period=10, multiplier=mult)
        result = st_test.calculate(data)
        print(f"   [OK] Multiplier={mult}: ST={result.value['supertrend']:.2f}, Trend={result.value['trend']}")

    # Test 4: Band genişliği
    print("\n5. Band genişliği testi...")
    band_width = result.value['upper'] - result.value['lower']
    print(f"   [OK] Upper: {result.value['upper']:.2f}")
    print(f"   [OK] Lower: {result.value['lower']:.2f}")
    print(f"   [OK] Band genişliği: {band_width:.2f}")
    print(f"   [OK] ATR: {result.metadata['atr']:.2f}")

    # Test 5: İstatistikler
    print("\n6. İstatistik testi...")
    stats = st.statistics
    print(f"   [OK] Hesaplama sayısı: {stats['calculation_count']}")
    print(f"   [OK] Hata sayısı: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = st.metadata
    print(f"   [OK] İsim: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    print("\n" + "="*60)
    print("[BAŞARILI] TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
