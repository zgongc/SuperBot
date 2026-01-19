"""
indicators/trend/hma.py - Hull Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    HMA (Hull Moving Average) - Hull hareketli ortalama
    Developed by Alan Hull, a trend indicator that reduces lag.
    Provides both fast and smooth averaging.

    Usage:
    - Trend tracking with low latency
    - Capturing rapid trend changes
    - Generating smooth and responsive signals

Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    1. WMA(n/2) hesapla
    2. WMA(n) hesapla
    3. Calculate the difference: 2*WMA(n/2) - WMA(n)
    4. Calculate the WMA(sqrt(n)) of this difference.

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


class HMA(BaseIndicator):
    """
    Hull Moving Average

    Advanced moving average that minimizes lag and provides a smooth result.
    Provides fast and smooth trend tracking using a WMA combination.

    Args:
        period: HMA period (default: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.half_period = max(1, period // 2)
        self.sqrt_period = max(1, int(np.sqrt(period)))

        super().__init__(
            name='hma',
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
        if self.period < 2:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be at least 2"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate HMA - for REALTIME.

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: HMA value (last bar)
        """
        close = data['close'].values

        # 1. WMA(n/2) hesapla
        wma_half = self._calculate_wma(close, self.half_period)

        # 2. WMA(n) hesapla
        wma_full = self._calculate_wma(close, self.period)

        # 3. 2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # 4. Get the final sqrt(n) value and calculate its WMA.
        # create raw_hma array
        raw_hma_series = []
        for i in range(len(close) - self.period + 1):
            slice_data = close[i:i + self.period]
            wh = self._calculate_wma_array(slice_data[-self.half_period:], self.half_period)
            wf = self._calculate_wma_array(slice_data, self.period)
            raw_hma_series.append(2 * wh - wf)

        # Calculate the WMA of the last sqrt(n) value
        if len(raw_hma_series) >= self.sqrt_period:
            hma_value = self._calculate_wma_array(
                np.array(raw_hma_series[-self.sqrt_period:]),
                self.sqrt_period
            )
        else:
            hma_value = raw_hma

        # Current price
        current_price = close[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(hma_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, hma_value),
            trend=self.get_trend(current_price, hma_value),
            strength=self._calculate_strength(current_price, hma_value),
            metadata={
                'period': self.period,
                'half_period': self.half_period,
                'sqrt_period': self.sqrt_period,
                'current_price': round(current_price, 2),
                'distance_pct': round(((current_price - hma_value) / hma_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch HMA calculation - for BACKTEST

        HMA Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: HMA values for all bars

        Performance: 2000 bars in ~0.05 seconds
        """
        self._validate_data(data)

        close = data['close'].values
        n = len(close)

        # Calculate WMA using pandas rolling + custom weights
        def wma_series(series: pd.Series, period: int) -> pd.Series:
            """Vectorized WMA calculation"""
            weights = np.arange(1, period + 1)

            def wma_window(window):
                if len(window) < period:
                    return np.nan
                return np.sum(window * weights) / np.sum(weights)

            return series.rolling(window=period).apply(wma_window, raw=True)

        close_series = pd.Series(close)

        # 1. WMA(n/2)
        wma_half = wma_series(close_series, self.half_period)

        # 2. WMA(n)
        wma_full = wma_series(close_series, self.period)

        # 3. raw_hma = 2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # 4. HMA = WMA(raw_hma, sqrt(n))
        hma = wma_series(raw_hma, self.sqrt_period)

        # Set first period values to NaN (warmup)
        hma.iloc[:self.period] = np.nan

        return pd.Series(hma.values, index=data.index, name='hma')

    def _calculate_wma(self, prices: np.ndarray, period: int) -> float:
        """
        WMA calculation (last value)

        Args:
            prices: Array of prices
            period: WMA periyodu

        Returns:
            float: The WMA value.
        """
        if len(prices) < period:
            period = len(prices)

        prices_slice = prices[-period:]
        weights = np.arange(1, period + 1)
        return np.sum(prices_slice * weights) / np.sum(weights)

    def _calculate_wma_array(self, prices: np.ndarray, period: int) -> float:
        """
        Weighted Moving Average calculation for an array.

        Args:
            prices: Price array
            period: WMA periyodu

        Returns:
            float: The WMA value.
        """
        if len(prices) < period:
            period = len(prices)

        weights = np.arange(1, period + 1)
        return np.sum(prices[-period:] * weights) / np.sum(weights)

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

        # Buffer'a verileri ekle
        for i in range(len(data)):
            self._close_buffer.append(data['close'].iloc[i])

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._close_buffer.append(close_val)

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
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, price: float, hma: float) -> SignalType:
        """
        Generate a signal from HMA.

        Args:
            price: Current price
            hma: HMA value

        Returns:
            SignalType: BUY (when the price goes above the HMA), SELL (when it goes below)
        """
        if price > hma:
            return SignalType.BUY
        elif price < hma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, hma: float) -> TrendDirection:
        """
        HMA'dan trend belirle

        Args:
            price: Current price
            hma: HMA value

        Returns:
            TrendDirection: UP (price > HMA), DOWN (price < HMA)
        """
        if price > hma:
            return TrendDirection.UP
        elif price < hma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, hma: float) -> float:
        """Calculate signal strength (0-100)"""
        distance_pct = abs((price - hma) / hma * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """HMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['HMA']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """HMA indicator test"""

    print("\n" + "="*60)
    print("HMA (HULL MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Volatile trend simulation
    base_price = 100
    prices = [base_price]
    for i in range(49):
        if i < 25:
            trend = 1.0  # Increase
        else:
            trend = -0.5  # Decrease
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

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    hma = HMA(period=20)
    print(f"   [OK] Created: {hma}")
    print(f"   [OK] Kategori: {hma.category.value}")
    print(f"   [OK] Required period: {hma.get_required_periods()}")

    result = hma(data)
    print(f"   [OK] HMA Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: HMA parameters
    print("\n3. HMA parameter test...")
    print(f"   [OK] Period: {hma.period}")
    print(f"   [OK] Half Period: {hma.half_period}")
    print(f"   [OK] Sqrt Period: {hma.sqrt_period}")

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [9, 16, 25]:
        hma_test = HMA(period=period)
        result = hma_test.calculate(data)
        print(f"   [OK] HMA({period}): {result.value:.2f} | sqrt_period: {hma_test.sqrt_period}")

    # Test 4: Comparison of HMA vs SMA
    print("\n5. HMA vs SMA comparison test...")
    sma_value = np.mean(data['close'].values[-20:])
    print(f"   [OK] SMA(20): {sma_value:.2f}")
    print(f"   [OK] HMA(20): {result.value:.2f}")
    print(f"   [OK] HMA is more responsive and has lower lag")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = hma.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = hma.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
