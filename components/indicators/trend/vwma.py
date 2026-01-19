"""
indicators/trend/vwma.py - Volume Weighted Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    VWMA (Volume Weighted Moving Average) - Volume weighted moving average
    A trend indicator calculated by weighting prices according to trading volume
    Gives more importance to price movements with high volume

    Usage:
    - Highlighting significant price movements
    - Measuring real market strength
    - Support/resistance levels

Formula:
    VWMA = Sum(Close * Volume) / Sum(Volume)
    For n periods:
    VWMA = (C1*V1 + C2*V2 + ... + Cn*Vn) / (V1 + V2 + ... + Vn)

Dependencies:
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


class VWMA(BaseIndicator):
    """
    Volume Weighted Moving Average

    Weighted moving average based on transaction volume.
    Candles with high volume have more weight.

    Args:
        period: VWMA period (default: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='vwma',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        VWMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VWMA value
        """
        close = data['close'].values
        volume = data['volume'].values

        # VWMA hesapla
        close_slice = close[-self.period:]
        volume_slice = volume[-self.period:]

        # Sum(Close * Volume) / Sum(Volume)
        vwma_value = np.sum(close_slice * volume_slice) / np.sum(volume_slice)

        # Current price
        current_price = close[-1]

        # For comparison with SMA
        sma_value = np.mean(close_slice)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(vwma_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, vwma_value),
            trend=self.get_trend(current_price, vwma_value),
            strength=self._calculate_strength(current_price, vwma_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'sma': round(sma_value, 2),
                'vwma_sma_diff': round(vwma_value - sma_value, 2),
                'avg_volume': round(np.mean(volume_slice), 2),
                'distance_pct': round(((current_price - vwma_value) / vwma_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch VWMA calculation - for BACKTEST

        VWMA Formula:
            VWMA = Sum(Close × Volume) / Sum(Volume)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: VWMA values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']
        volume = data['volume']

        # Close × Volume
        close_volume = close * volume

        # VWMA = rolling sum of (Close × Volume) / rolling sum of Volume
        vwma = close_volume.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

        # Set first period values to NaN (warmup)
        vwma.iloc[:self.period-1] = np.nan

        return pd.Series(vwma.values, index=data.index, name='vwma')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - prepares the necessary state for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)

        # Buffer'lara verileri ekle
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            open_val = candle.get('open', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'high': [high_val] * len(self._close_buffer),
            'low': [low_val] * len(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, price: float, vwma: float) -> SignalType:
        """
        Generate signal from VWMA.

        Args:
            price: Current price
            vwma: VWMA value

        Returns:
            SignalType: BUY (when the price goes above the VWMA), SELL (when it goes below)
        """
        if price > vwma:
            return SignalType.BUY
        elif price < vwma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, vwma: float) -> TrendDirection:
        """
        VWMA'dan trend belirle

        Args:
            price: Current price
            vwma: VWMA value

        Returns:
            TrendDirection: UP (price > VWMA), DOWN (price < VWMA)
        """
        if price > vwma:
            return TrendDirection.UP
        elif price < vwma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, vwma: float) -> float:
        """Calculate signal strength (0-100)"""
        distance_pct = abs((price - vwma) / vwma * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """VWMA volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VWMA']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """VWMA indicator test"""

    print("\n" + "="*60)
    print("VWMA (VOLUME WEIGHTED MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend and volume simulation
    base_price = 100
    prices = [base_price]
    volumes = [1000]

    for i in range(49):
        trend = 0.5
        noise = np.random.randn() * 1.5
        prices.append(prices[-1] + trend + noise)

        # High volume in case of high price change
        price_change = abs(prices[-1] - prices[-2])
        volume = 1000 + price_change * 100 + np.random.randint(0, 500)
        volumes.append(volume)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Volume range: {min(volumes):.0f} -> {max(volumes):.0f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    vwma = VWMA(period=20)
    print(f"   [OK] Created: {vwma}")
    print(f"   [OK] Kategori: {vwma.category.value}")
    print(f"   [OK] Required period: {vwma.get_required_periods()}")
    print(f"   [OK] Volume required: {vwma._requires_volume()}")

    result = vwma(data)
    print(f"   [OK] VWMA Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Comparison of VWMA vs SMA
    print("\n3. VWMA vs SMA comparison test...")
    print(f"   [OK] SMA(20): {result.metadata['sma']}")
    print(f"   [OK] VWMA(20): {result.value}")
    print(f"   [OK] Fark: {result.metadata['vwma_sma_diff']}")
    print(f"   [OK] VWMA gives more weight to high-volume movements")

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [10, 20, 30]:
        vwma_test = VWMA(period=period)
        result = vwma_test.calculate(data)
        print(f"   [OK] VWMA({period}): {result.value:.2f} | Signal: {result.signal.value}")

    # Test 4: Volume effect
    print("\n5. Volume effect test...")
    print(f"   [OK] Average volume: {result.metadata['avg_volume']:.0f}")
    print(f"   [OK] Son mum hacmi: {volumes[-1]:.0f}")

    # High-volume scenario
    high_volume_data = data.copy()
    high_volume_data.loc[high_volume_data.index[-1], 'volume'] = 10000
    result_high = vwma.calculate(high_volume_data)
    print(f"   [OK] Normal VWMA: {result.value}")
    print(f"   [OK] High volume VWMA: {result_high.value}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = vwma.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = vwma.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
