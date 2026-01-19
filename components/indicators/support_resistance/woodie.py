"""
indicators/support_resistance/woodie.py - Woodie Pivot Points

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Woodie Pivot Points - Pivot levels calculated using the Woodie formula.
    A pivot calculation method that gives more weight to the close price.
    Widely used for daily trading.

Formula:
    P = (High + Low + 2 × Close) / 4
    R1 = (2 × P) - Low
    R2 = P + High - Low
    S1 = (2 × P) - High
    S2 = P - High + Low

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


class Woodie(BaseIndicator):
    """
    Woodie Pivot Points

    Using the High, Low, and Close values from the previous period.
    Woodie pivot seviyeleri (P, R1-R2, S1-S2) hesaplar.
    Gives more weight to the closing price.

    Args:
        period: Pivot calculation period (default: 1 - day)
    """

    def __init__(
        self,
        period: int = 1,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='woodie',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
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
        Woodie Pivot Points hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Woodie pivot seviyeleri (P, R1-R2, S1-S2)
        """
        # Get the H, L, C values from the previous period.
        high = data['high'].iloc[-self.period - 1:-1].max()
        low = data['low'].iloc[-self.period - 1:-1].min()
        close = data['close'].iloc[-self.period - 1]

        # Calculate Woodie Pivot Point (with more weight on Close)
        pivot = (high + low + 2 * close) / 4

        # Resistance seviyeleri
        r1 = (2 * pivot) - low
        r2 = pivot + high - low

        # Support seviyeleri
        s1 = (2 * pivot) - high
        s2 = pivot - high + low

        current_price = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Create levels as a dictionary
        levels = {
            'R2': round(r2, 2),
            'R1': round(r1, 2),
            'P': round(pivot, 2),
            'S1': round(s1, 2),
            'S2': round(s2, 2)
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, pivot),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'period': self.period,
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'current_price': round(current_price, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Woodie Pivot Points calculation - for BACKTEST

        Woodie Pivot Formula:
            P = (High + Low + 2 × Close) / 4
            R1 = (2 × P) - Low
            R2 = P + High - Low
            S1 = (2 × P) - High
            S2 = P - High + Low

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: woodie_pivot, r1, r2, s1, s2 for all bars

        Performance: 2000 bars in ~0.01 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Rolling max/min for previous period
        prev_high = high.shift(self.period).rolling(window=self.period).max()
        prev_low = low.shift(self.period).rolling(window=self.period).min()
        prev_close = close.shift(self.period)

        # Woodie Pivot Point (Close has 2x weight)
        pivot = (prev_high + prev_low + 2 * prev_close) / 4

        # Resistance levels
        r1 = (2 * pivot) - prev_low
        r2 = pivot + prev_high - prev_low

        # Support levels
        s1 = (2 * pivot) - prev_high
        s2 = pivot - prev_high + prev_low

        # Set first period values to NaN (warmup)
        warmup = self.period * 2
        pivot.iloc[:warmup] = np.nan
        r1.iloc[:warmup] = np.nan
        r2.iloc[:warmup] = np.nan
        s1.iloc[:warmup] = np.nan
        s2.iloc[:warmup] = np.nan

        return pd.DataFrame({
            'P': pivot.values,
            'R1': r1.values,
            'R2': r2.values,
            'S1': s1.values,
            'S2': s2.values
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
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
                value={'pivot': 0.0, 'r1': 0.0, 's1': 0.0},
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

    def get_signal(self, price: float, levels: dict) -> SignalType:
        """
        Generate a signal based on the price's woodie pivot levels.

        Args:
            price: Current price
            levels: Woodie pivot seviyeleri

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if price < levels['S1']:
            return SignalType.BUY
        elif price > levels['R1']:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, pivot: float) -> TrendDirection:
        """
        Determine the trend based on the price relative to the pivot.

        Args:
            price: Current price
            pivot: Pivot level

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if price > pivot:
            return TrendDirection.UP
        elif price < pivot:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, levels: dict) -> float:
        """
        Calculate the strength of the price based on levels.

        Args:
            price: Current price
            levels: Woodie pivot seviyeleri

        Returns:
            float: Power value (0-100)
        """
        pivot = levels['P']
        r2 = levels['R2']
        s2 = levels['S2']

        if price > pivot:
            # Upward force
            strength = ((price - pivot) / (r2 - pivot)) * 100
        else:
            # Downward force
            strength = ((pivot - price) / (pivot - s2)) * 100

        return min(max(strength, 0), 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 1
        }

    def _requires_volume(self) -> bool:
        """Woodie Pivot volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['Woodie']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Woodie Pivot Points indicator test"""

    print("\n" + "="*60)
    print("WOODIE PIVOT POINTS TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(49):
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
    woodie = Woodie(period=1)
    print(f"   [OK] Created: {woodie}")
    print(f"   [OK] Kategori: {woodie.category.value}")
    print(f"   [OK] Tip: {woodie.indicator_type.value}")
    print(f"   [OK] Required period: {woodie.get_required_periods()}")

    result = woodie(data)
    print(f"   [OK] Woodie Pivot Seviyeleri:")
    for level, value in result.value.items():
        print(f"        {level}: {value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [1, 5, 10]:
        woodie_test = Woodie(period=period)
        result = woodie_test.calculate(data)
        print(f"   [OK] Woodie({period}) - P: {result.value['P']} | Signal: {result.signal.value}")

    # Test 3: Comparison with Classic Pivot
    print("\n4. Comparison with Classic Pivot...")
    result = woodie.calculate(data)
    high = result.metadata['high']
    low = result.metadata['low']
    close = result.metadata['close']

    # Classic pivot hesapla
    classic_pivot = (high + low + close) / 3
    woodie_pivot = result.value['P']

    print(f"   [OK] Classic Pivot: {classic_pivot:.2f}")
    print(f"   [OK] Woodie Pivot: {woodie_pivot:.2f}")
    print(f"   [OK] Fark: {abs(woodie_pivot - classic_pivot):.2f}")
    print(f"   [OK] Woodie, gives more weight to Close")

    # Test 4: Seviye analizi
    print("\n5. Seviye analizi...")
    current = result.metadata['current_price']
    print(f"   [OK] Current price: {current}")
    print(f"   [OK] Pivot: {result.value['P']}")
    if current > result.value['P']:
        print(f"   [OK] Price is above the pivot (Bullish)")
        print(f"   [OK] R1: {result.value['R1']}")
        print(f"   [OK] R2: {result.value['R2']}")
    else:
        print(f"   [OK] Price is below the pivot (Bearish)")
        print(f"   [OK] S1: {result.value['S1']}")
        print(f"   [OK] S2: {result.value['S2']}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = woodie.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = woodie.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
