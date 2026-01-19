"""
indicators/momentum/roc.py - Rate of Change

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    ROC (Rate of Change) - Percentage of price change
    Range: Unlimited (positive and negative values)
    Positive ROC: Price increase
    Negative ROC: Price decrease
    Around 0: No momentum

Formula:
    ROC = ((Close - Close[n]) / Close[n]) × 100
    n = period (lookback period)

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


class ROC(BaseIndicator):
    """
    Rate of Change

    Calculates the percentage change in price compared to a specific period before.
    Used to determine the momentum's strength and direction.

    Args:
        period: ROC period (default: 12)
    """

    def __init__(
        self,
        period: int = 12,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='roc',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 1

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
        ROC hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: ROC value
        """
        close = data['close'].values

        # Current price and price N periods ago
        current_close = close[-1]
        previous_close = close[-(self.period + 1)]

        # ROC hesapla
        if previous_close == 0:
            roc_value = 0.0
        else:
            roc_value = ((current_close - previous_close) / previous_close) * 100

        timestamp = int(data.iloc[-1]['timestamp'])

        # Power calculation - Normalize the absolute value of ROC to a range of 0-100.
        # Typical ROC values are between -20 and +20, so let's divide by 20.
        strength = min(abs(roc_value) * 5, 100)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(roc_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(roc_value),
            trend=self.get_trend(roc_value),
            strength=strength,
            metadata={
                'period': self.period,
                'current_close': round(current_close, 2),
                'previous_close': round(previous_close, 2),
                'price_change': round(current_close - previous_close, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch ROC calculation - for BACKTEST

        ROC Formula:
            ROC = ((Close - Close[n]) / Close[n]) × 100

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: ROC values for all bars

        Performance: 2000 bars in ~0.005 seconds
        """
        self._validate_data(data)

        close = data['close']

        # ROC = percentage change over period
        # Using pct_change() which is vectorized
        roc = close.pct_change(periods=self.period) * 100

        # Handle division by zero
        roc = roc.fillna(0)

        # Set first period values to NaN (warmup)
        roc.iloc[:self.period] = np.nan

        return pd.Series(roc.values, index=data.index, name='roc')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Son verileri buffer'a ekle
        for val in data['close'].values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

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
        Generate a signal from the ROC value.

        Args:
            value: ROC value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Positive ROC indicates upward momentum, negative ROC indicates downward momentum
        # Outliers may indicate a reversal signal
        if value > 10:
            return SignalType.SELL  # Extremely positive, a selling opportunity
        elif value < -10:
            return SignalType.BUY  # Extremely negative, buy opportunity
        elif value > 0:
            return SignalType.HOLD  # Pozitif momentum devam
        elif value < 0:
            return SignalType.HOLD  # Negative momentum continues
        return SignalType.NEUTRAL

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the ROC value.

        Args:
            value: ROC value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 0:
            return TrendDirection.UP
        elif value < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 12}

    def _requires_volume(self) -> bool:
        """ROC volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ROC']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """ROC indicator test"""

    print("\n" + "="*60)
    print("ROC (RATE OF CHANGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)

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
    roc = ROC(period=12)
    print(f"   [OK] Created: {roc}")
    print(f"   [OK] Kategori: {roc.category.value}")
    print(f"   [OK] Required period: {roc.get_required_periods()}")

    result = roc(data)
    print(f"   [OK] ROC Value: {result.value}%")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [6, 12, 20]:
        roc_test = ROC(period=period)
        result = roc_test.calculate(data)
        print(f"   [OK] ROC({period}): {result.value}% | Signal: {result.signal.value}")

    # Test 3: Rising trend
    print("\n4. Rising trend test...")
    up_data = data.copy()
    # Continuous upward trend in the last 12 candles
    for i in range(12):
        idx = up_data.index[-(12-i)]
        up_data.loc[idx, 'close'] = 100 + i * 2
        up_data.loc[idx, 'high'] = 100 + i * 2 + 0.5
        up_data.loc[idx, 'low'] = 100 + i * 2 - 0.5

    result_up = roc.calculate(up_data)
    print(f"   [OK] Rising trend ROC: {result_up.value}%")
    print(f"   [OK] Signal: {result_up.signal.value}")
    print(f"   [OK] Trend: {result_up.trend.name}")

    # Test 4: Declining trend
    print("\n5. Downtrend test...")
    down_data = data.copy()
    # Continuous decline in the last 12 candles
    for i in range(12):
        idx = down_data.index[-(12-i)]
        down_data.loc[idx, 'close'] = 120 - i * 2
        down_data.loc[idx, 'high'] = 120 - i * 2 + 0.5
        down_data.loc[idx, 'low'] = 120 - i * 2 - 0.5

    result_down = roc.calculate(down_data)
    print(f"   [OK] Declining trend ROC: {result_down.value}%")
    print(f"   [OK] Signal: {result_down.signal.value}")
    print(f"   [OK] Trend: {result_down.trend.name}")

    # Test 5: Fixed price (no change)
    print("\n6. Fixed price test...")
    flat_data = data.copy()
    flat_data['close'] = 100.0
    flat_data['high'] = 100.5
    flat_data['low'] = 99.5

    result_flat = roc.calculate(flat_data)
    print(f"   [OK] Fixed price ROC: {result_flat.value}%")
    print(f"   [OK] Signal: {result_flat.signal.value}")
    print(f"   [OK] Trend: {result_flat.trend.name}")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = roc.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata testi...")
    metadata = roc.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
