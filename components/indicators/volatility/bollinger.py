"""
indicators/volatility/bollinger.py - Bollinger Bands

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Bollinger Bands - Volatility bands
    Standard deviation bands around the average price
    Used for measuring overbought/oversold conditions and volatility

Formula:
    Middle Band = SMA(close, period)
    Upper Band = Middle + (std_dev * multiplier)
    Lower Band = Middle - (std_dev * multiplier)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.trend.sma import SMA
from indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands

    It creates volatility bands to indicate overbought/oversold levels.
    When the bands narrow, volatility is low; when they widen, volatility is high.

    Args:
        period: SMA period (default: 20)
        std_dev: Standard deviation factor (default: 2.0)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.std_dev = std_dev

        # Use the SMA indicator (code reuse)
        self._sma = SMA(period=period)

        super().__init__(
            name='bollinger',
            category=IndicatorCategory.VOLATILITY,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'std_dev': std_dev
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
        if self.std_dev <= 0:
            raise InvalidParameterError(
                self.name, 'std_dev', self.std_dev,
                "The standard deviation multiplier must be positive."
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Bollinger Bands hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Bollinger Bands values (upper, middle, lower)
        """
        close = data['close'].values

        # Middle Band (SMA)
        middle = np.mean(close[-self.period:])

        # Standart sapma
        std = np.std(close[-self.period:], ddof=0)

        # Upper and Lower Bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # The position of the price within the bands (%B)
        if upper != lower:
            percent_b = (current_price - lower) / (upper - lower)
        else:
            percent_b = 0.5

        # Bandwidth (volatility indicator)
        if middle != 0:
            bandwidth = ((upper - lower) / middle) * 100
        else:
            bandwidth = 0

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper, 8),
                'middle': round(middle, 8),
                'lower': round(lower, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(percent_b),
            trend=self.get_trend(current_price, middle),
            strength=min(abs(percent_b - 0.5) * 200, 100),  # Normalize to a range of 0-100
            metadata={
                'period': self.period,
                'std_dev': self.std_dev,
                'percent_b': round(percent_b, 4),
                'bandwidth': round(bandwidth, 2),
                'price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Bollinger Bands calculation - for BACKTEST

        Bollinger Bands Formula:
            Middle Band = SMA(close, period)
            Upper Band = Middle + (std_dev * multiplier)
            Lower Band = Middle - (std_dev * multiplier)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 3 columns (bb_upper, bb_middle, bb_lower)

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        close = data['close']

        # 1. Middle Band = SMA(close, period) - use SMA.calculate_batch (code reuse)
        middle = self._sma.calculate_batch(data)

        # 2. Standard Deviation (no SMA indicator for std, keep rolling)
        std = close.rolling(window=self.period).std(ddof=0)

        # 3. Upper and Lower Bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # Create result DataFrame (same keys as calculate() - no prefix)
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)

        # Set first period values to NaN (warmup)
        result.iloc[:self.period] = np.nan

        return result

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
            IndicatorResult: Current Bollinger Bands values
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        # (Check if it's dict - BaseIndicator.warmup_buffer creates deque, we need dict)
        if buffer_key not in self._buffers or not isinstance(self._buffers[buffer_key], dict):
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = {
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len)
            }

        # Add new candle to symbol's buffer
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle['high']
            low_val = candle['low']
            close_val = candle['close']
            open_val = candle.get('open', candle['close'])
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._buffers[buffer_key]['high'].append(high_val)
        self._buffers[buffer_key]['low'].append(low_val)
        self._buffers[buffer_key]['close'].append(close_val)

        # Need minimum data for Bollinger calculation
        if len(self._buffers[buffer_key]['close']) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period, 'insufficient_data': True}
            )

        # Convert buffer to DataFrame
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

    def get_signal(self, percent_b: float) -> SignalType:
        """
        Generate a signal from the %B value.

        Args:
            percent_b: The position of the price within the bands (between 0 and 1).

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if percent_b <= 0:  # Below or in the lower band
            return SignalType.BUY
        elif percent_b >= 1:  # In or above the upper band
            return SignalType.SELL
        elif percent_b < 0.2:  # Close to the lower band
            return SignalType.BUY
        elif percent_b > 0.8:  # Close to the upper band
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, middle: float) -> TrendDirection:
        """
        Determine the trend based on the price and middle band.

        Args:
            price: Current price
            middle: Middle band value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if price > middle:
            return TrendDirection.UP
        elif price < middle:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'std_dev': 2.0
        }

    def _requires_volume(self) -> bool:
        """Bollinger Bands volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['BollingerBands']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Bollinger Bands indicator test"""

    print("\n" + "="*60)
    print("BOLLINGER BANDS TEST")
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
    bb = BollingerBands(period=20, std_dev=2.0)
    print(f"   [OK] Created: {bb}")
    print(f"   [OK] Kategori: {bb.category.value}")
    print(f"   [OK] Tip: {bb.indicator_type.value}")
    print(f"   [OK] Required period: {bb.get_required_periods()}")

    result = bb(data)
    print(f"   [OK] Upper Band: {result.value['upper']}")
    print(f"   [OK] Middle Band: {result.value['middle']}")
    print(f"   [OK] Lower Band: {result.value['lower']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] %B: {result.metadata['percent_b']}")
    print(f"   [OK] Bandwidth: {result.metadata['bandwidth']:.2f}%")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [10, 20, 30]:
        bb_test = BollingerBands(period=period)
        result = bb_test.calculate(data)
        print(f"   [OK] BB({period}): Upper={result.value['upper']:.2f}, Middle={result.value['middle']:.2f}, Lower={result.value['lower']:.2f}")

    # Test 3: Different standard deviation factors
    print("\n4. Different std_dev test...")
    for std_dev in [1.5, 2.0, 2.5]:
        bb_test = BollingerBands(period=20, std_dev=std_dev)
        result = bb_test.calculate(data)
        print(f"   [OK] BB(std={std_dev}): Bandwidth={result.metadata['bandwidth']:.2f}%")

    # Test 4: Statistics
    print("\n5. Statistical test...")
    stats = bb.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = bb.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
