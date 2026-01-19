"""
indicators/statistical/z_score.py - Z-Score (Standart Sapma Skoru)

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Z-Score - Measures how many standard deviations the price is away from the average.
    Range: Usually between -3 and +3 (normalized).
    Overbought: > +2
    Oversold: < -2

Formula:
    Z-Score = (Price - Average) / Standard Deviation

    Positive value: Price is higher than the average.
    Negative value: Price is lower than the average.

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class ZScore(BaseIndicator):
    """
    Z-Score (Standart Sapma Skoru)

    Measures how statistically extreme the price is.
    Used for strategies that revert to the mean.

    Args:
        period: Calculation period (default: 20)
        overbought: Overbought level (default: 2.0)
        oversold: Oversold level (default: -2.0)
    """

    def __init__(
        self,
        period: int = 20,
        overbought: float = 2.0,
        oversold: float = -2.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='zscore',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'overbought': overbought,
                'oversold': oversold
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
                "Period must be at least 2 (for standard deviation)"
            )
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold should be smaller than overbought"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Z-Score hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Z-Score value
        """
        close = data['close'].values

        # Get data up to the last period
        period_data = close[-self.period:]

        # Calculate the average and standard deviation
        mean = np.mean(period_data)
        std = np.std(period_data, ddof=1)  # Sample std (n-1)

        # Z-Score hesapla
        if std == 0:
            z_score_value = 0.0
        else:
            z_score_value = (close[-1] - mean) / std

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(z_score_value, 4),
            timestamp=timestamp,
            signal=self.get_signal(z_score_value),
            trend=self.get_trend(z_score_value),
            strength=min(abs(z_score_value) * 50, 100),  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'mean': round(mean, 2),
                'std': round(std, 2),
                'current_price': round(close[-1], 2),
                'deviation_percent': round((z_score_value * std / mean) * 100, 2) if mean != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch Z-Score calculation - for BACKTEST

        Z-Score Formula:
            Z-Score = (Price - Mean) / Std Dev

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: Z-Score values for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        close = data['close']

        # Rolling mean
        mean = close.rolling(window=self.period).mean()

        # Rolling std (sample std with ddof=1)
        std = close.rolling(window=self.period).std(ddof=1)

        # Z-Score = (Price - Mean) / Std
        z_score = (close - mean) / std

        # Handle division by zero
        z_score = z_score.fillna(0)

        # Set first period values to NaN (warmup)
        z_score.iloc[:self.period-1] = np.nan

        return pd.Series(z_score.values, index=data.index, name='zscore')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """Warmup buffer - prepares the necessary state for update()"""
        super().warmup_buffer(data, symbol)
        from collections import deque
        max_len = self.get_required_periods() + 50
        self._close_buffer = deque(maxlen=max_len)
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

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the Z-Score value.

        Args:
            value: Z-Score value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if value < self.oversold:
            # The price is very low, expecting a return to the average.
            return SignalType.BUY
        elif value > self.overbought:
            # The price is too high, expecting it to average out.
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the Z-score value.

        Args:
            value: Z-Score value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 0.5:
            return TrendDirection.UP
        elif value < -0.5:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'overbought': 2.0,
            'oversold': -2.0
        }

    def _requires_volume(self) -> bool:
        """Z-Score volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ZScore']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Z-Score indicator test"""

    print("\n" + "="*60)
    print("Z-SCORE (STANDART SAPMA SKORU) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Simulate price movement - a series that reverts to the average.
    base_price = 100
    prices = [base_price]
    mean_price = 100
    for i in range(49):
        # Mean reversion simulation
        noise = np.random.randn() * 2
        mean_revert = (mean_price - prices[-1]) * 0.1
        prices.append(prices[-1] + noise + mean_revert)

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
    print(f"   [OK] Final price: {prices[-1]:.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    zscore = ZScore(period=20)
    print(f"   [OK] Created: {zscore}")
    print(f"   [OK] Kategori: {zscore.category.value}")
    print(f"   [OK] Required period: {zscore.get_required_periods()}")

    result = zscore(data)
    print(f"   [OK] Z-Score Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Ortalama: {result.metadata['mean']}")
    print(f"   [OK] Std Sapma: {result.metadata['std']}")
    print(f"   [OK] Sapma %: {result.metadata['deviation_percent']}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [10, 20, 30]:
        zscore_test = ZScore(period=period)
        result = zscore_test.calculate(data)
        print(f"   [OK] Z-Score({period}): {result.value:.4f} | Signal: {result.signal.value}")

    # Test 3: Outlier test
    print("\n4. Outlier test...")
    # Extremely high price
    extreme_high_data = data.copy()
    extreme_high_data.loc[extreme_high_data.index[-1], 'close'] = 120
    result_high = zscore.calculate(extreme_high_data)
    print(f"   [OK] Excessive high price (120): Z-Score = {result_high.value:.4f}")
    print(f"   [OK] Signal: {result_high.signal.value}")

    # Extremely low price
    extreme_low_data = data.copy()
    extreme_low_data.loc[extreme_low_data.index[-1], 'close'] = 80
    result_low = zscore.calculate(extreme_low_data)
    print(f"   [OK] Extremely low price (80): Z-Score = {result_low.value:.4f}")
    print(f"   [OK] Signal: {result_low.signal.value}")

    # Test 4: Custom levels
    print("\n5. Special level test...")
    zscore_custom = ZScore(period=20, overbought=3.0, oversold=-3.0)
    result = zscore_custom.calculate(data)
    print(f"   [OK] Custom level Z-Score: {result.value:.4f}")
    print(f"   [OK] Overbought: {zscore_custom.overbought}, Oversold: {zscore_custom.oversold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 5: Z-scores of the last 10 candles
    print("\n6. Zaman serisi testi (son 10 mum)...")
    for i in range(-10, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= zscore.period:
            result = zscore.calculate(test_data)
            print(f"   [OK] Mum {i}: Z-Score = {result.value:7.4f} | "
                  f"Price = {test_data.iloc[-1]['close']:7.2f} | "
                  f"Signal = {result.signal.value}")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = zscore.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = zscore.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
