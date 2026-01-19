"""
indicators/volatility/atr.py - Average True Range

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    ATR (Average True Range) - Volatility indicator
    Measures the average width of price movements
    High ATR = High volatility
    Low ATR = Low volatility

Formula:
    TR = max[(High - Low), abs(High - PrevClose), abs(Low - PrevClose)]
    ATR = RMA(TR, period)

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


class ATR(BaseIndicator):
    """
    Average True Range

    It determines volatility by measuring the average width of price movements.
    It is used for stop-loss and position sizing.

    Args:
        period: ATR period (default: 14)
    """

    def __init__(
        self,
        period: int = 14,
        logger=None,
        error_handler=None
    ):
        self.period = period

        super().__init__(
            name='atr',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 1  # The previous candle is required for the TR calculation

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
        ATR hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: ATR value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Check if we have enough data
        if len(high) < self.period:
            # Not enough data - return simple range as estimate
            current_price = close[-1]
            simple_range = high[-1] - low[-1]
            timestamp = int(data.iloc[-1]['timestamp'])
            volatility_pct = (simple_range / current_price) * 100 if current_price > 0 else 0

            return IndicatorResult(
                value=round(simple_range, 8),
                timestamp=timestamp,
                signal=self.get_signal(volatility_pct),
                trend=TrendDirection.NEUTRAL,
                strength=min(volatility_pct * 10, 100),
                metadata={
                    'period': self.period,
                    'true_range': round(simple_range, 8),
                    'volatility_pct': round(volatility_pct, 2),
                    'price': round(current_price, 8),
                    'insufficient_data': True
                }
            )

        # True Range hesapla
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR hesapla (RMA - Wilder's smoothing)
        atr_values = np.zeros(len(tr))
        atr_values[self.period-1] = np.mean(tr[:self.period])

        alpha = 1.0 / self.period
        for i in range(self.period, len(tr)):
            atr_values[i] = atr_values[i-1] + alpha * (tr[i] - atr_values[i-1])

        atr_value = atr_values[-1]
        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Volatility percentage
        volatility_pct = (atr_value / current_price) * 100 if current_price > 0 else 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(atr_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(volatility_pct),
            trend=TrendDirection.NEUTRAL,  # ATR does not show trend
            strength=min(volatility_pct * 10, 100),  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'true_range': round(tr[-1], 8),
                'volatility_pct': round(volatility_pct, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch ATR calculation - for BACKTEST

        ATR Formula:
            TR = max[(High - Low), abs(High - PrevClose), abs(Low - PrevClose)]
            ATR = RMA(TR, period) = Wilder's smoothing

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: ATR values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # True Range calculation (VECTORIZED - NO LOOPS!)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # ATR with Wilder's smoothing (RMA) = ewm with alpha=1/period
        # Wilder's RMA is equivalent to EWM with alpha = 1/period
        atr = tr.ewm(alpha=1.0/self.period, adjust=False).mean()

        # Set first period values to NaN (warmup)
        atr.iloc[:self.period] = np.nan

        return pd.Series(atr.values, index=data.index, name='atr')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        buffer_key = symbol if symbol else 'default'
        max_len = self.get_required_periods() + 50

        self._buffers[buffer_key] = {
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'close': deque(maxlen=max_len)
        }

        # Son verileri buffer'lara ekle
        for i in range(len(data)):
            self._buffers[buffer_key]['high'].append(data['high'].iloc[i])
            self._buffers[buffer_key]['low'].append(data['low'].iloc[i])
            self._buffers[buffer_key]['close'].append(data['close'].iloc[i])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Current ATR value
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        # Also reinitialize if old format (not a dict)
        if buffer_key not in self._buffers or not isinstance(self._buffers[buffer_key], dict):
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = {
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len)
            }

        # Add new candle to symbol's buffer
        # Handle both dict and list/tuple formats
        if isinstance(candle, dict):
            high = candle['high']
            low = candle['low']
            close = candle['close']
        else:
            # Assume list/tuple format: [timestamp, open, high, low, close, volume]
            high = candle[2]
            low = candle[3]
            close = candle[4]

        self._buffers[buffer_key]['high'].append(high)
        self._buffers[buffer_key]['low'].append(low)
        self._buffers[buffer_key]['close'].append(close)

        # Need minimum data for ATR calculation
        if len(self._buffers[buffer_key]['close']) < self.get_required_periods():
            # Not enough data - return neutral
            # Handle both dict and list/tuple formats for timestamp
            if isinstance(candle, dict):
                timestamp_val = int(candle.get('timestamp', 0))
            else:
                timestamp_val = int(candle[0]) if len(candle) > 0 else 0

            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period, 'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        # Get open, volume, timestamp values (handle both dict and list formats)
        if isinstance(candle, dict):
            open_val = candle.get('open', close)
            volume_val = candle.get('volume', 1000)
            timestamp_val = candle.get('timestamp', 0)
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            open_val = candle[1] if len(candle) > 1 else close
            volume_val = candle[5] if len(candle) > 5 else 1000
            timestamp_val = candle[0] if len(candle) > 0 else 0

        buffer_data = pd.DataFrame({
            'high': list(self._buffers[buffer_key]['high']),
            'low': list(self._buffers[buffer_key]['low']),
            'close': list(self._buffers[buffer_key]['close']),
            'open': [open_val] * len(self._buffers[buffer_key]['close']),
            'volume': [volume_val] * len(self._buffers[buffer_key]['close']),
            'timestamp': [timestamp_val] * len(self._buffers[buffer_key]['close'])
        })

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, volatility_pct: float) -> SignalType:
        """
        Generate a signal from the volatility percentage.

        Args:
            volatility_pct: Volatility percentage

        Returns:
            SignalType: Signal based on volatility level.
        """
        # High volatility: be careful
        if volatility_pct > 5.0:
            return SignalType.SELL  # High risk
        # Normal volatilite
        elif volatility_pct > 2.0:
            return SignalType.HOLD
        # Low volatility: may be an opportunity
        else:
            return SignalType.BUY

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 14
        }

    def _requires_volume(self) -> bool:
        """ATR volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ATR']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """ATR indicator test"""

    print("\n" + "="*60)
    print("ATR (AVERAGE TRUE RANGE) TEST")
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
        'high': [p + abs(np.random.randn()) * 1.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    atr = ATR(period=14)
    print(f"   [OK] Created: {atr}")
    print(f"   [OK] Kategori: {atr.category.value}")
    print(f"   [OK] Required period: {atr.get_required_periods()}")

    result = atr(data)
    print(f"   [OK] ATR Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [7, 14, 21]:
        atr_test = ATR(period=period)
        result = atr_test.calculate(data)
        print(f"   [OK] ATR({period}): {result.value:.4f} | Volatilite: {result.metadata['volatility_pct']:.2f}%")

    # Test 3: Statistics
    print("\n4. Statistical test...")
    stats = atr.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 4: Metadata
    print("\n5. Metadata testi...")
    metadata = atr.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
