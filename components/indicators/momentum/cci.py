"""
indicators/momentum/cci.py - Commodity Channel Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    CCI (Commodity Channel Index) - Momentum oscillator
    Range: Generally between -100 and +100 (can go outside this range)
    Overbought: > +100
    Oversold: < -100

Formula:
    CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
    Typical Price = (High + Low + Close) / 3

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


class CCI(BaseIndicator):
    """
    Commodity Channel Index

    Measures the deviation from the statistical average of the price.
    It is used to determine overbought/oversold conditions and trend strength.

    Args:
        period: CCI period (default: 20)
        overbought: Overbought level (default: 100)
        oversold: Oversold level (default: -100)
    """

    def __init__(
        self,
        period: int = 20,
        overbought: float = 100,
        oversold: float = -100,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='cci',
            category=IndicatorCategory.MOMENTUM,
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
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
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
        CCI hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CCI value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Typical Price hesapla
        typical_price = (high + low + close) / 3

        # Calculate the SMA of Typical Price
        sma_tp = np.mean(typical_price[-self.period:])

        # Mean Deviation hesapla
        mean_deviation = np.mean(np.abs(typical_price[-self.period:] - sma_tp))

        # CCI hesapla
        if mean_deviation == 0:
            cci_value = 0.0
        else:
            cci_value = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(cci_value, 2),
            timestamp=timestamp,
            signal=self.get_signal(cci_value),
            trend=self.get_trend(cci_value),
            strength=min(abs(cci_value), 100),  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'typical_price': round(typical_price[-1], 2),
                'sma': round(sma_tp, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch CCI calculation - for BACKTEST

        CCI Formula:
            CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
            Typical Price = (High + Low + Close) / 3

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: CCI values for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # SMA of Typical Price
        sma_tp = typical_price.rolling(window=self.period).mean()

        # Mean Deviation (vectorized)
        mad = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        # CCI = (TP - SMA) / (0.015 * MAD)
        cci = (typical_price - sma_tp) / (0.015 * mad)

        # Handle division by zero
        cci = cci.replace([np.inf, -np.inf], 0)
        cci = cci.fillna(0)

        # Set first period values to NaN (warmup)
        cci.iloc[:self.period-1] = np.nan

        return pd.Series(cci.values, index=data.index, name='cci')

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

        # Create and fill the buffers
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])

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

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the CCI value.

        Args:
            value: CCI value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if value < self.oversold:
            return SignalType.BUY
        elif value > self.overbought:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the CCI value.

        Args:
            value: CCI value

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
        return {
            'period': 20,
            'overbought': 100,
            'oversold': -100
        }

    def _requires_volume(self) -> bool:
        """CCI volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CCI']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """CCI indicator test - similar to cache_manager.py"""

    print("=" * 70)
    print("📊 CCI (Commodity Channel Index) Test")
    print("=" * 70)

    # Create test data
    print("\n1️⃣ Create Test Data")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

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

    print(f"  ✅ {len(data)} candles created")
    print(f"  ✅ Price: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"  ✅ Change: {((max(prices) - min(prices)) / min(prices) * 100):.2f}%")

    # Test 1: Basic Calculation
    print("\n2️⃣ Basic Calculations")
    cci = CCI(period=20)
    print(f"  ✅ Indicator: {cci}")
    print(f"  ✅ Kategori: {cci.category.value}")
    print(f"  ✅ Tip: {cci.indicator_type.value}")
    print(f"  ✅ Min periyot: {cci.get_required_periods()}")

    result = cci(data)

    # Signal emoji
    signal_emoji = {
        'buy': '🟢',
        'sell': '🔴',
        'hold': '🟡'
    }.get(result.signal.value, '⚪')

    # Trend emoji
    trend_emoji = {
        'UP': '⬆️',
        'DOWN': '⬇️',
        'NEUTRAL': '➡️'
    }.get(result.trend.name, '❓')

    print(f"\n  📈 CCI Results:")
    print(f"  ✅ Value: {result.value}")
    print(f"  ✅ Signal: {signal_emoji} {result.signal.value.upper()}")
    print(f"  ✅ Trend: {trend_emoji} {result.trend.name}")
    print(f"  ✅ Power: {result.strength:.2f}/100")
    print(f"  ✅ Metadata: period={result.metadata['period']}, tp={result.metadata['typical_price']}")

    # Test 2: Different Periods
    print("\n3️⃣ Different Period Comparison")
    for period in [10, 20, 30]:
        cci_test = CCI(period=period)
        res = cci_test.calculate(data)
        sig_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}.get(res.signal.value, '⚪')
        print(f"  {sig_emoji} CCI({period:2d}): {res.value:7.2f} | {res.signal.value:4s}")

    # Test 3: Custom Levels
    print("\n4️⃣  Special Levels")
    cci_custom = CCI(period=20, overbought=150, oversold=-150)
    result = cci_custom.calculate(data)
    print(f"  ✅ OB/OS: ±{cci_custom.overbought}")
    print(f"  ✅ CCI: {result.value}")
    print(f"  ✅ Signal: {result.signal.value}")

    # Test 4: Statistics
    print("\n5️⃣  Statistics")
    stats = cci.statistics
    print(f"  📊 Calculation: {stats['calculation_count']}")
    print(f"  ❌ Error: {stats['error_count']}")
    print(f"  🕐 Son: {stats['last_calculation']}")

    # Test 5: Metadata
    print("\n6️⃣  Metadata")
    metadata = cci.metadata
    print(f"  ✅ Name: {metadata.name}")
    print(f"  ✅ Kategori: {metadata.category.value}")
    print(f"  ✅ Description: {metadata.description[:50]}...")
    print(f"  ✅ Volume required: {metadata.requires_volume}")
    print(f"  ✅ Default params: {metadata.default_params}")

    # Test 6: Signal Analysis
    print("\n7️⃣ Signal Analysis")
    if result.signal == SignalType.BUY:
        print("  🟢 BUY SIGNAL - CCI is in the oversold region")
    elif result.signal == SignalType.SELL:
        print("  🔴 SALES SIGNAL - CCI is in the overbought zone")
    else:
        print("  🟡 WAIT - CCI is within the normal range")

    print(f"\n  📋 Detay:")
    print(f"  ✅ Available: {result.value}")
    print(f"  ✅ Overbought: {cci.overbought}")
    print(f"  ✅ Oversold: {cci.oversold}")
    print(f"  ✅ Tavsiye: {'AL' if result.value < cci.oversold else 'SAT' if result.value > cci.overbought else 'BEKLE'}")

    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70 + "\n")
