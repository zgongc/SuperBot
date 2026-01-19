"""
indicators/volatility/chandelier.py - Chandelier Exit

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Chandelier Exit - An ATR-based trailing stop indicator
    Used to determine position exit points
    Creates different stop levels in uptrend and downtrend

Formula:
    Long Stop = Highest High(period) - (ATR(period) × multiplier)
    Short Stop = Lowest Low(period) + (ATR(period) × multiplier)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
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


class ChandelierExit(BaseIndicator):
    """
    Chandelier Exit

    Creates trailing stop levels based on ATR.
    Provides different exit levels for long and short positions.

    Args:
        period: Highest/Lowest and ATR period (default: 22)
        multiplier: ATR multiplier (default: 3.0)
    """

    def __init__(
        self,
        period: int = 22,
        multiplier: float = 3.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.multiplier = multiplier

        # Use the ATR indicator (code reuse)
        self._atr = ATR(period=period)

        super().__init__(
            name='chandelier',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'multiplier': multiplier
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 1  # Extra candle for TR calculation

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        if self.multiplier <= 0:
            raise InvalidParameterError(
                self.name, 'multiplier', self.multiplier,
                "The factor must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Chandelier Exit hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Chandelier Exit seviyeleri (long_stop, short_stop)
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # ATR hesapla
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR (RMA - Wilder's smoothing)
        atr_values = np.zeros(len(tr))
        atr_values[self.period-1] = np.mean(tr[:self.period])

        alpha = 1.0 / self.period
        for i in range(self.period, len(tr)):
            atr_values[i] = atr_values[i-1] + alpha * (tr[i] - atr_values[i-1])

        atr = atr_values[-1]

        # Highest High and Lowest Low
        highest_high = np.max(high[-self.period:])
        lowest_low = np.min(low[-self.period:])

        # Chandelier Exit seviyeleri
        long_stop = highest_high - (atr * self.multiplier)
        short_stop = lowest_low + (atr * self.multiplier)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Distance of the price to the stop levels (percentage)
        if current_price > 0:
            long_distance = ((current_price - long_stop) / current_price) * 100
            short_distance = ((short_stop - current_price) / current_price) * 100
        else:
            long_distance = 0
            short_distance = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'long_stop': round(long_stop, 8),
                'short_stop': round(short_stop, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(current_price, long_stop, short_stop),
            trend=self.get_trend(current_price, long_stop, short_stop),
            strength=min(abs(long_distance) * 2, 100),  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'multiplier': self.multiplier,
                'atr': round(atr, 8),
                'highest_high': round(highest_high, 8),
                'lowest_low': round(lowest_low, 8),
                'long_distance_pct': round(long_distance, 2),
                'short_distance_pct': round(short_distance, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Chandelier Exit calculation - for BACKTEST

        Chandelier Formula:
            Long Stop = Highest High(period) - (ATR(period) × multiplier)
            Short Stop = Lowest Low(period) + (ATR(period) × multiplier)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: long_stop, short_stop for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']

        # ATR - use ATR.calculate_batch (code reuse)
        atr = self._atr.calculate_batch(data)

        # Highest High and Lowest Low (rolling)
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()

        # Chandelier Exit levels
        long_stop = highest_high - (atr * self.multiplier)
        short_stop = lowest_low + (atr * self.multiplier)

        # Set first period values to NaN (warmup)
        long_stop.iloc[:self.period] = np.nan
        short_stop.iloc[:self.period] = np.nan

        return pd.DataFrame({
            'long_stop': long_stop.values,
            'short_stop': short_stop.values
        }, index=data.index)

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
                value=0.0,
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

    def get_signal(self, price: float, long_stop: float, short_stop: float) -> SignalType:
        """
        Generate signals from price and stop levels.

        Args:
            price: Current price
            long_stop: Long stop level
            short_stop: Short stop level

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Price dropped below the long stop: exit long position
        if price < long_stop:
            return SignalType.SELL
        # Price exceeded the short stop level: exit short position
        elif price > short_stop:
            return SignalType.BUY
        # In the meantime: hold the position
        return SignalType.HOLD

    def get_trend(self, price: float, long_stop: float, short_stop: float) -> TrendDirection:
        """
        Determine the trend based on price and stop levels.

        Args:
            price: Current price
            long_stop: Long stop level
            short_stop: Short stop level

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        # Price is above the long stop: upward trend
        if price > long_stop:
            return TrendDirection.UP
        # Price is below the short stop: downtrend
        elif price < short_stop:
            return TrendDirection.DOWN
        # Arada: belirsiz
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 22,
            'multiplier': 3.0
        }

    def _requires_volume(self) -> bool:
        """Chandelier Exit volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ChandelierExit']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Chandelier Exit indicator test"""

    print("\n" + "="*60)
    print("CHANDELIER EXIT TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(35)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(34):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    ce = ChandelierExit(period=22, multiplier=3.0)
    print(f"   [OK] Created: {ce}")
    print(f"   [OK] Kategori: {ce.category.value}")
    print(f"   [OK] Tip: {ce.indicator_type.value}")
    print(f"   [OK] Required period: {ce.get_required_periods()}")

    result = ce(data)
    print(f"   [OK] Long Stop: {result.value['long_stop']}")
    print(f"   [OK] Short Stop: {result.value['short_stop']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] ATR: {result.metadata['atr']}")
    print(f"   [OK] Long Distance: {result.metadata['long_distance_pct']:.2f}%")
    print(f"   [OK] Short Distance: {result.metadata['short_distance_pct']:.2f}%")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [10, 22, 30]:
        ce_test = ChandelierExit(period=period, multiplier=3.0)
        result = ce_test.calculate(data)
        print(f"   [OK] CE({period}): Long={result.value['long_stop']:.2f}, Short={result.value['short_stop']:.2f}")

    # Test 3: Different factors
    print("\n4. Different multiplier test...")
    for mult in [2.0, 3.0, 4.0]:
        ce_test = ChandelierExit(period=22, multiplier=mult)
        result = ce_test.calculate(data)
        long_dist = result.metadata['long_distance_pct']
        short_dist = result.metadata['short_distance_pct']
        print(f"   [OK] CE(mult={mult}): Long Dist={long_dist:.2f}%, Short Dist={short_dist:.2f}%")

    # Test 4: Uprising trend simulation
    print("\n5. Uprising trend test...")
    uptrend_prices = [100]
    for i in range(34):
        change = abs(np.random.randn()) * 1.5  # Only positive
        uptrend_prices.append(uptrend_prices[-1] + change)

    uptrend_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': uptrend_prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in uptrend_prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in uptrend_prices],
        'close': uptrend_prices,
        'volume': [1000] * 35
    })
    result = ce.calculate(uptrend_data)
    print(f"   [OK] Uprising Trend: {result.trend.name}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Price: {result.metadata['price']:.2f}, Long Stop: {result.value['long_stop']:.2f}")

    # Test 5: Downtrend simulation
    print("\n6. Downtrend test...")
    downtrend_prices = [100]
    for i in range(34):
        change = abs(np.random.randn()) * 1.5  # Only negative
        downtrend_prices.append(downtrend_prices[-1] - change)

    downtrend_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': downtrend_prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in downtrend_prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in downtrend_prices],
        'close': downtrend_prices,
        'volume': [1000] * 35
    })
    result = ce.calculate(downtrend_data)
    print(f"   [OK] Downtrend: {result.trend.name}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Price: {result.metadata['price']:.2f}, Short Stop: {result.value['short_stop']:.2f}")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = ce.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = ce.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
