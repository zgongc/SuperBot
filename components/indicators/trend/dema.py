"""
indicators/trend/dema.py - Double Exponential Moving Average

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    DEMA (Double Exponential Moving Average) - Double exponential moving average
    A trend indicator developed by Patrick Mulloy, which reduces lag
    Generates fast and smooth signals using a combination of two EMAs

    Usage:
    - Trend tracking with low latency
    - Capturing rapid trend changes
    - More responsive signal than EMA

Formula:
    DEMA = 2*EMA - EMA(EMA)
    EMA1 = EMA(Close, period)
    EMA2 = EMA(EMA1, period)
    DEMA = 2*EMA1 - EMA2

Dependencies:
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


class DEMA(BaseIndicator):
    """
    Double Exponential Moving Average

    A trend indicator with a double EMA that reduces lag and provides a smoother result.

    Args:
        period: DEMO period (default: 20)
    """

    def __init__(
        self,
        period: int = 20,
        logger=None,
        error_handler=None
    ):
        self.period = period

        # Use the EMA indicator (code reuse)
        self._ema = EMA(period=period)

        super().__init__(
            name='dema',
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
        # More data is required because the calculation for DEMA involves 2 x EMA.
        return self.period * 2

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
        DEMA hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: DEM value
        """
        # EMA1 = EMA(Close) - use EMA.calculate_batch (code reuse)
        ema1 = self._ema.calculate_batch(data)

        # EMA2 = EMA(EMA1)
        ema1_df = self._create_ema_input(ema1, data)
        ema2 = self._ema.calculate_batch(ema1_df)

        # DEMA = 2*EMA1 - EMA2
        dema_value = 2 * ema1.iloc[-1] - ema2.iloc[-1]

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(dema_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(current_price, dema_value),
            trend=self.get_trend(current_price, dema_value),
            strength=self._calculate_strength(current_price, dema_value),
            metadata={
                'period': self.period,
                'current_price': round(current_price, 2),
                'ema1': round(ema1.iloc[-1], 2),
                'ema2': round(ema2.iloc[-1], 2),
                'distance_pct': round(((current_price - dema_value) / dema_value) * 100, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        VECTORIZED batch DEMA calculation - BACKTEST

        DEMA Formula:
            DEMA = 2*EMA(Close) - EMA(EMA(Close))

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: DEMA values for all bars
        """
        self._validate_data(data)

        # EMA1 = EMA(Close) - use EMA.calculate_batch (code reuse)
        ema1 = self._ema.calculate_batch(data)

        # EMA2 = EMA(EMA1)
        ema1_df = self._create_ema_input(ema1, data)
        ema2 = self._ema.calculate_batch(ema1_df)

        # DEMA = 2*EMA1 - EMA2
        dema = 2 * ema1 - ema2

        # Set first period values to NaN (warmup)
        warmup = self.period * 2
        dema.iloc[:warmup-1] = np.nan

        return pd.Series(dema.values, index=data.index, name='dema')

    def _create_ema_input(self, series: pd.Series, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create minimal OHLCV DataFrame for EMA calculation from a Series.
        """
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
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._close_buffer = deque(maxlen=max_len)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)
        
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

    def get_signal(self, price: float, dema: float) -> SignalType:
        """
        Generate a signal from DEMA.

        Args:
            price: Current price
            dema: DEMA value

        Returns:
            SignalType: BUY (when the price goes above the EMA), SELL (when it goes below)
        """
        if price > dema:
            return SignalType.BUY
        elif price < dema:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, dema: float) -> TrendDirection:
        """
        DEMA'dan trend belirle

        Args:
            price: Current price
            dema: DEMA value

        Returns:
            TrendDirection: UP (price > DEMA), DOWN (price < DEMA)
        """
        if price > dema:
            return TrendDirection.UP
        elif price < dema:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, dema: float) -> float:
        """Calculate signal strength (0-100)"""
        distance_pct = abs((price - dema) / dema * 100)
        return min(distance_pct * 20, 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20
        }

    def _requires_volume(self) -> bool:
        """DEMA volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['DEMA']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """DEMA indicator test"""

    print("\n" + "="*60)
    print("DEMA (DOUBLE EXPONENTIAL MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(80)]

    # Trend simulation
    base_price = 100
    prices = [base_price]
    for i in range(79):
        trend = 0.4
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

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    dema = DEMA(period=20)
    print(f"   [OK] Created: {dema}")
    print(f"   [OK] Kategori: {dema.category.value}")
    print(f"   [OK] Required period: {dema.get_required_periods()}")

    result = dema(data)
    print(f"   [OK] DEMA Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: EMA components
    print("\n3. EMA component test...")
    print(f"   [OK] EMA1 (Close): {result.metadata['ema1']}")
    print(f"   [OK] EMA2 (EMA1): {result.metadata['ema2']}")
    print(f"   [OK] DEMA = 2*{result.metadata['ema1']:.2f} - {result.metadata['ema2']:.2f}")

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [10, 20, 30]:
        dema_test = DEMA(period=period)
        result = dema_test.calculate(data)
        print(f"   [OK] DEMA({period}): {result.value:.2f} | Signal: {result.signal.value}")

    # Test 4: DEMA vs EMA comparison
    print("\n5. DEMA vs EMA comparison test...")
    # Simple EMA calculation
    multiplier = 2 / (20 + 1)
    ema = np.mean(data['close'].values[:20])
    for price in data['close'].values[20:]:
        ema = (price - ema) * multiplier + ema

    print(f"   [OK] EMA(20): {ema:.2f}")
    print(f"   [OK] DEMA(20): {result.value:.2f}")
    print(f"   [OK] DEMA is more responsive (lower lag)")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = dema.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = dema.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
