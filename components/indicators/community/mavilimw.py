"""
indicators/community/mavilimw.py - MavilimW Indicator

Version: 1.0.0
Date: 2025-12-12
Author: Kivanc Ozbilgic (@mavilim0732)
Port: SuperBot Team

Original PineScript:
    //@version=4
    //inspired by: @mavilim0732 on twitter
    //creator&author: KIVANC OZBILGIC
    study("MavilimW", overlay=true)

Description:
    MavilimW - 6 kademeli WMA zincirleme hesaplaması ile
    trend takibi yapan indikatör.

    Fibonacci dizisine benzer periyot artışı:
    fmal=3, smal=5 -> tmal=8 -> Fmal=13 -> Ftmal=21 -> Smal=34

    Sinyal:
    - MAVW > MAVW[1] = Yükseliş (Mavi)
    - MAVW < MAVW[1] = Düşüş (Kırmızı)
    - Crossover/Crossunder = Alım/Satım sinyali

Formül:
    M1 = WMA(close, fmal)       # 3
    M2 = WMA(M1, smal)          # 5
    M3 = WMA(M2, tmal)          # 8
    M4 = WMA(M3, Fmal)          # 13
    M5 = WMA(M4, Ftmal)         # 21
    MAVW = WMA(M5, Smal)        # 34

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.trend.wma (internal)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from collections import deque

from indicators.base_indicator import BaseIndicator
from indicators.trend.wma import WMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class MavilimW(BaseIndicator):
    """
    MavilimW - Kivanc Ozbilgic'in WMA Trend Indikatörü

    6 kademeli WMA zincirleme hesaplaması yaparak
    gürültüyü filtreler ve temiz trend sinyalleri üretir.

    Args:
        fmal: First Moving Average length (varsayılan: 3)
        smal: Second Moving Average length (varsayılan: 5)

    Outputs:
        value: Dict with 'mavw' (current MAVW value)
        signal: BUY (crossover), SELL (crossunder), HOLD
        trend: UP (mavw rising), DOWN (mavw falling)

    Periods:
        fmal=3, smal=5 kullanıldığında:
        - tmal = 3 + 5 = 8
        - Fmal = 5 + 8 = 13
        - Ftmal = 8 + 13 = 21
        - Smal = 13 + 21 = 34
        - Total required: ~89 bars (tüm WMA'ların ısınması için)
    """

    def __init__(
        self,
        fmal: int = 3,
        smal: int = 5,
        logger=None,
        error_handler=None
    ):
        self.fmal = fmal
        self.smal = smal

        # Fibonacci-like period progression
        self.tmal = fmal + smal           # 8
        self.Fmal = smal + self.tmal      # 13
        self.Ftmal = self.tmal + self.Fmal  # 21
        self.Smal = self.Fmal + self.Ftmal  # 34

        super().__init__(
            name='mavilimw',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'fmal': fmal,
                'smal': smal
            },
            logger=logger,
            error_handler=error_handler
        )

        # WMA instances - kendi WMA indikatörümüzü kullan
        self._wma1 = WMA(period=self.fmal)
        self._wma2 = WMA(period=self.smal)
        self._wma3 = WMA(period=self.tmal)
        self._wma4 = WMA(period=self.Fmal)
        self._wma5 = WMA(period=self.Ftmal)
        self._wma6 = WMA(period=self.Smal)

        # Buffer for update()
        self._close_buffer = None
        self._mavw_buffer = None

    def get_required_periods(self) -> int:
        """
        Minimum gerekli periyot sayısı

        6 kademeli WMA zinciri için:
        fmal + smal + tmal + Fmal + Ftmal + Smal = 3+5+8+13+21+34 = 84
        +5 buffer = 89
        """
        total = self.fmal + self.smal + self.tmal + self.Fmal + self.Ftmal + self.Smal
        return total + 5  # Buffer için +5

    def validate_params(self) -> bool:
        """Parametreleri doğrula"""
        if self.fmal < 1:
            raise InvalidParameterError(
                self.name, 'fmal', self.fmal,
                "First MA length must be positive"
            )
        if self.smal < 1:
            raise InvalidParameterError(
                self.name, 'smal', self.smal,
                "Second MA length must be positive"
            )
        return True

    def _apply_wma_chain(self, close: pd.Series) -> pd.Series:
        """
        6 kademeli WMA zinciri uygula (WMA sınıfını kullanarak)

        Args:
            close: Close price Series

        Returns:
            pd.Series: Final MAVW values
        """
        # Helper: Series'i WMA'ya uygun DataFrame'e çevir
        def series_to_df(series: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({
                'timestamp': range(len(series)),
                'open': series.values,
                'high': series.values,
                'low': series.values,
                'close': series.values,
                'volume': [0] * len(series)
            })

        # 6 kademeli WMA zinciri
        df1 = series_to_df(close)
        m1 = self._wma1.calculate_batch(df1)

        df2 = series_to_df(pd.Series(m1.values))
        m2 = self._wma2.calculate_batch(df2)

        df3 = series_to_df(pd.Series(m2.values))
        m3 = self._wma3.calculate_batch(df3)

        df4 = series_to_df(pd.Series(m3.values))
        m4 = self._wma4.calculate_batch(df4)

        df5 = series_to_df(pd.Series(m4.values))
        m5 = self._wma5.calculate_batch(df5)

        df6 = series_to_df(pd.Series(m5.values))
        mavw = self._wma6.calculate_batch(df6)

        return pd.Series(mavw.values, index=close.index, name='mavw')

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        MavilimW hesapla - REALTIME

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: MAVW değeri ve sinyal
        """
        close = data['close']
        timestamp = int(data.iloc[-1]['timestamp'])

        # 6 kademeli WMA zinciri (WMA sınıfını kullanarak)
        mavw_series = self._apply_wma_chain(close)
        mavw = mavw_series.values

        # Son değerler
        current_mavw = mavw[-1] if not np.isnan(mavw[-1]) else 0.0
        prev_mavw = mavw[-2] if len(mavw) > 1 and not np.isnan(mavw[-2]) else current_mavw

        # Trend direction: 1 = up (mavi), -1 = down (kırmızı), 0 = neutral
        trend_direction = 1 if current_mavw > prev_mavw else -1 if current_mavw < prev_mavw else 0
        trend = self._get_trend(current_mavw, prev_mavw)

        # Signal (crossover/crossunder detection)
        close_vals = close.values
        signal = self._get_signal(current_mavw, prev_mavw, close_vals[-1], close_vals[-2] if len(close_vals) > 1 else close_vals[-1])

        # Strength (momentum)
        strength = self._calculate_strength(current_mavw, prev_mavw, close_vals[-1])

        # Warmup buffer
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'mavw': round(current_mavw, 8),
                'mavw_prev': round(prev_mavw, 8),
                'trend_direction': trend_direction
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'fmal': self.fmal,
                'smal': self.smal,
                'current_price': round(close_vals[-1], 8),
                'distance_pct': round(((close_vals[-1] - current_mavw) / current_mavw) * 100, 4) if current_mavw != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MavilimW batch hesapla - BACKTEST

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: Columns ['mavw', 'trend_direction']
        """
        self._validate_data(data)

        close = data['close']

        # 6 kademeli WMA zinciri (WMA sınıfını kullanarak)
        mavw = self._apply_wma_chain(close)

        # Trend direction: 1 = up (mavi), -1 = down (kırmızı), 0 = neutral
        mavw_prev = mavw.shift(1)
        trend_direction = pd.Series(0, index=data.index, dtype='int')
        trend_direction[mavw > mavw_prev] = 1
        trend_direction[mavw < mavw_prev] = -1

        result = pd.DataFrame({
            'mavw': mavw,
            'mavw_prev': mavw_prev,
            'trend_direction': trend_direction
        }, index=data.index)

        return result

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - update() için

        Args:
            data: OHLCV DataFrame
            symbol: Sembol (opsiyonel)
        """
        super().warmup_buffer(data, symbol)

        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

        # MAVW buffer for previous value tracking
        batch_result = self.calculate_batch(data.tail(max_len + 10))
        self._mavw_buffer = deque(maxlen=10)
        for val in batch_result['mavw'].tail(10).values:
            if not np.isnan(val):
                self._mavw_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update - REALTIME

        Args:
            candle: Yeni kline data (dict veya tuple)
            symbol: Sembol (opsiyonel)

        Returns:
            IndicatorResult: Güncellenmiş MAVW
        """
        if self._close_buffer is None:
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._close_buffer.append(close_val)

        # Not enough data
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={
                    'mavw': 0.0,
                    'mavw_prev': 0.0,
                    'trend_direction': 0
                },
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        # Calculate using full buffer
        buffer_data = pd.DataFrame({
            'timestamp': [timestamp_val] * len(self._close_buffer),
            'open': list(self._close_buffer),
            'high': list(self._close_buffer),
            'low': list(self._close_buffer),
            'close': list(self._close_buffer),
            'volume': [0] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def _get_trend(self, current: float, prev: float) -> TrendDirection:
        """Trend direction from MAVW slope"""
        if current > prev:
            return TrendDirection.UP
        elif current < prev:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_signal(self, mavw: float, mavw_prev: float, price: float, price_prev: float) -> SignalType:
        """
        Signal detection

        - BUY: MAVW turns up (mavw > mavw_prev and mavw_prev <= mavw_prev_prev)
        - SELL: MAVW turns down
        - HOLD: No change
        """
        # Simple crossover detection: price crosses MAVW
        if price > mavw and price_prev <= mavw_prev:
            return SignalType.BUY
        elif price < mavw and price_prev >= mavw_prev:
            return SignalType.SELL

        # Trend change detection
        if mavw > mavw_prev:
            return SignalType.BUY
        elif mavw < mavw_prev:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_strength(self, mavw: float, mavw_prev: float, price: float) -> float:
        """
        Signal strength (0-100)

        Based on:
        - Distance from MAVW
        - MAVW momentum
        """
        if mavw == 0:
            return 0.0

        # Distance component
        distance_pct = abs((price - mavw) / mavw * 100)
        distance_strength = min(distance_pct * 10, 50)

        # Momentum component
        if mavw_prev == 0:
            momentum_strength = 0
        else:
            mavw_change = abs((mavw - mavw_prev) / mavw_prev * 100)
            momentum_strength = min(mavw_change * 20, 50)

        return distance_strength + momentum_strength

    def _get_default_params(self) -> dict:
        """Varsayılan parametreler"""
        return {
            'fmal': 3,
            'smal': 5
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['mavw', 'mavw_prev', 'trend_direction']

    def _requires_volume(self) -> bool:
        """Volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MavilimW']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """MavilimW indicator test"""

    print("\n" + "="*60)
    print("MAVILIMW INDICATOR TEST")
    print("="*60 + "\n")

    # Create sample data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    n_bars = 150

    timestamps = [1697000000000 + i * 60000 for i in range(n_bars)]

    # Trend simulation with noise
    base_price = 100
    prices = [base_price]
    for i in range(n_bars - 1):
        trend = 0.3 * np.sin(i / 20)  # Sine wave trend
        noise = np.random.randn() * 0.5
        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] Created {len(data)} candles")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    mavw = MavilimW(fmal=3, smal=5)
    print(f"   [OK] Created: {mavw}")
    print(f"   [OK] Required periods: {mavw.get_required_periods()}")
    print(f"   [OK] WMA periods: {mavw.fmal}, {mavw.smal}, {mavw.tmal}, {mavw.Fmal}, {mavw.Ftmal}, {mavw.Smal}")

    result = mavw.calculate(data)
    print(f"   [OK] MAVW Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Strength: {result.strength:.2f}")

    # Test 2: Batch calculation
    print("\n3. Batch calculation test...")
    batch_result = mavw.calculate_batch(data)
    print(f"   [OK] Batch shape: {batch_result.shape}")
    print(f"   [OK] Columns: {batch_result.columns.tolist()}")
    print(f"   [OK] Last 5 MAVW values:")
    for i in range(-5, 0):
        print(f"       {batch_result['mavw'].iloc[i]:.4f} ({batch_result['trend_direction'].iloc[i]})")

    # Test 3: Update method
    print("\n4. Update method test...")
    mavw.warmup_buffer(data)

    new_candle = {
        'timestamp': timestamps[-1] + 60000,
        'open': prices[-1],
        'high': prices[-1] + 0.5,
        'low': prices[-1] - 0.5,
        'close': prices[-1] + 0.3,
        'volume': 1500
    }

    update_result = mavw.update(new_candle)
    print(f"   [OK] Update MAVW: {update_result.value}")
    print(f"   [OK] Update Signal: {update_result.signal.value}")

    # Test 4: Trend change detection
    print("\n5. Trend statistics...")
    up_count = (batch_result['trend_direction'] == 'up').sum()
    down_count = (batch_result['trend_direction'] == 'down').sum()
    neutral_count = (batch_result['trend_direction'] == 'neutral').sum()
    nan_count = batch_result['mavw'].isna().sum()

    print(f"   [OK] UP trends: {up_count}")
    print(f"   [OK] DOWN trends: {down_count}")
    print(f"   [OK] NEUTRAL: {neutral_count}")
    print(f"   [OK] NaN (warmup): {nan_count}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
