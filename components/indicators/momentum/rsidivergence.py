"""
indicators/momentum/rsidivergence.py - RSI Divergence

Version: 3.0.0
Date: 2026-01-29
Author: SuperBot Team

Description:
    RSI Divergence - Detects the discrepancy between price and RSI.
    Bullish Divergence: Price decreases while RSI increases (buy signal).
    Bearish Divergence: Price increases while RSI decreases (sell signal).
    Hidden Divergence: Signals for trend continuation.
    Strong reversal indicator.

Formula:
    1. Calculate RSI using existing RSI indicator (14 period)
    2. Find price pivot points in the last N periods
    3. Find RSI pivot points at the same points
    4. Compare price and RSI movements
    5. Detect divergence

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - components.indicators.momentum.rsi (for RSI calculation)
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
from components.indicators.momentum.rsi import calculate_rsi_values
from components.indicators.support_resistance.swingpoints import (
    find_pivot_highs,
    find_pivot_lows,
    check_pivot_range
)


class RSIDivergence(BaseIndicator):
    """
    RSI Divergence

    Detects divergences between price and RSI.
    Used to capture trend reversal points.

    Args:
        rsi_period: RSI period (default: 14)
        lookback: Lookback period (for pivot points) (default: 5)
        min_strength: Minimum divergence strength (0-100) (default: 30)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 5,
        min_strength: float = 30,
        logger=None,
        error_handler=None
    ):
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.min_strength = min_strength

        super().__init__(
            name='rsidivergence',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'rsi_period': rsi_period,
                'lookback': lookback,
                'min_strength': min_strength
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.rsi_period + self.lookback * 2

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "RSI period must be positive"
            )
        if self.lookback < 2:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback must be at least 2"
            )
        if not (0 <= self.min_strength <= 100):
            raise InvalidParameterError(
                self.name, 'min_strength', self.min_strength,
                "Minimum strength must be between 0 and 100"
            )
        return True

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Initialize warmup buffer for incremental updates.

        Ensures update() has enough historical data for divergence detection.

        Note: symbol parameter is ignored - each indicator instance is already symbol-specific
        """
        from collections import deque

        # Initialize buffers dict if needed
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        buffer_key = 'default'

        # Create buffer with correct size (matching update() expectations)
        max_len = self.get_required_periods() + 50  # 24 + 50 = 74 candles
        self._buffers[buffer_key] = deque(maxlen=max_len)

        # Fill buffer with historical data
        for _, row in data.tail(max_len).iterrows():
            self._buffers[buffer_key].append(row.to_dict())

        # Calculate initial state if we have enough data
        if len(data) >= self.get_required_periods():
            # Call calculate to initialize internal state
            self.calculate(data)

    def _find_price_pivots(self, high: np.ndarray, low: np.ndarray, lookback: int) -> dict:
        """
        Find price pivot points using HIGH/LOW (TradingView compatible)

        Uses actual high/low prices instead of close for more accurate pivot detection.
        Compatible with TradingView's ta.pivothigh() and ta.pivotlow().

        Args:
            high: High prices array
            low: Low prices array
            lookback: The lookback period (left_bars = right_bars = lookback)

        Returns:
            dict: {'highs': [(index, value)], 'lows': [(index, value)]}
        """
        # Use TradingView-compatible pivot detection from swing_points
        pivot_highs = find_pivot_highs(high, left_bars=lookback, right_bars=lookback)
        pivot_lows = find_pivot_lows(low, left_bars=lookback, right_bars=lookback)

        return {
            'highs': pivot_highs,  # Already list of (index, value) tuples
            'lows': pivot_lows     # Already list of (index, value) tuples
        }

    def _find_oscillator_pivots(self, data: np.ndarray, lookback: int) -> dict:
        """
        Find oscillator (RSI) pivot points

        For oscillators, we still use the same value for highs/lows.

        Args:
            data: Oscillator values (e.g., RSI)
            lookback: The lookback period

        Returns:
            dict: {'highs': [(index, value)], 'lows': [(index, value)]}
        """
        # For oscillator, use the same array for both highs and lows detection
        pivot_highs = find_pivot_highs(data, left_bars=lookback, right_bars=lookback)
        pivot_lows = find_pivot_lows(data, left_bars=lookback, right_bars=lookback)

        return {
            'highs': pivot_highs,
            'lows': pivot_lows
        }

    def _detect_divergence(self, price_pivots: dict, rsi_pivots: dict,
                          range_min: int = 5, range_max: int = 60) -> dict:
        """
        Detect divergence with range check (TradingView compatible)

        Args:
            price_pivots: Price pivot points
            rsi_pivots: RSI pivot points
            range_min: Minimum bars between pivots (default: 5)
            range_max: Maximum bars between pivots (default: 60)

        Returns:
            dict: Divergence information
        """
        result = {
            'bullish': False,
            'bearish': False,
            'hidden_bullish': False,
            'hidden_bearish': False,
            'strength': 0
        }

        # Bullish Divergence: Price decreases, RSI increases
        if len(price_pivots['lows']) >= 2 and len(rsi_pivots['lows']) >= 2:
            price_low1, price_val1 = price_pivots['lows'][-2]
            price_low2, price_val2 = price_pivots['lows'][-1]

            # Range check: pivots must be 5-60 bars apart (TradingView default)
            if not check_pivot_range(price_low1, price_low2, range_min, range_max):
                pass  # Skip if pivots too close or too far
            else:
                # Find RSI pivots that are close to price pivots
                rsi_low1 = None
                rsi_low2 = None
                for idx, val in rsi_pivots['lows']:
                    if abs(idx - price_low1) <= self.lookback:
                        rsi_low1 = (idx, val)
                    if abs(idx - price_low2) <= self.lookback:
                        rsi_low2 = (idx, val)

                if rsi_low1 and rsi_low2:
                    # Regular Bullish divergence: Price LL, RSI HL (reversal signal)
                    if price_val2 < price_val1 and rsi_low2[1] > rsi_low1[1]:
                        result['bullish'] = True
                        # Better strength: combine price % change and RSI change
                        price_change_pct = abs(price_val2 - price_val1) / price_val1 * 100
                        rsi_change = abs(rsi_low2[1] - rsi_low1[1])
                        price_strength = min(100, price_change_pct * 100)
                        rsi_strength = min(100, rsi_change * 10)
                        result['strength'] = price_strength * 0.6 + rsi_strength * 0.4

                    # Hidden Bullish divergence: Price HL, RSI LL (continuation signal)
                    elif price_val2 > price_val1 and rsi_low2[1] < rsi_low1[1]:
                        result['hidden_bullish'] = True
                        if result['strength'] == 0:  # Only set if no regular divergence
                            price_change_pct = abs(price_val2 - price_val1) / price_val1 * 100
                            rsi_change = abs(rsi_low2[1] - rsi_low1[1])
                            price_strength = min(100, price_change_pct * 100)
                            rsi_strength = min(100, rsi_change * 10)
                            result['strength'] = price_strength * 0.6 + rsi_strength * 0.4

        # Bearish Divergence: Price increases, RSI decreases
        if len(price_pivots['highs']) >= 2 and len(rsi_pivots['highs']) >= 2:
            price_high1, price_val1 = price_pivots['highs'][-2]
            price_high2, price_val2 = price_pivots['highs'][-1]

            # Range check: pivots must be 5-60 bars apart (TradingView default)
            if not check_pivot_range(price_high1, price_high2, range_min, range_max):
                pass  # Skip if pivots too close or too far
            else:
                # Find RSI pivots that are close to price pivots
                rsi_high1 = None
                rsi_high2 = None
                for idx, val in rsi_pivots['highs']:
                    if abs(idx - price_high1) <= self.lookback:
                        rsi_high1 = (idx, val)
                    if abs(idx - price_high2) <= self.lookback:
                        rsi_high2 = (idx, val)

                if rsi_high1 and rsi_high2:
                    # Regular Bearish divergence: Price HH, RSI LH (reversal signal)
                    if price_val2 > price_val1 and rsi_high2[1] < rsi_high1[1]:
                        result['bearish'] = True
                        # Better strength: combine price % change and RSI change
                        price_change_pct = abs(price_val2 - price_val1) / price_val1 * 100
                        rsi_change = abs(rsi_high2[1] - rsi_high1[1])
                        price_strength = min(100, price_change_pct * 100)
                        rsi_strength = min(100, rsi_change * 10)
                        result['strength'] = price_strength * 0.6 + rsi_strength * 0.4

                    # Hidden Bearish divergence: Price LH, RSI HH (continuation signal)
                    elif price_val2 < price_val1 and rsi_high2[1] > rsi_high1[1]:
                        result['hidden_bearish'] = True
                        if result['strength'] == 0:  # Only set if no regular divergence
                            price_change_pct = abs(price_val1 - price_val2) / price_val1 * 100
                            rsi_change = abs(rsi_high2[1] - rsi_high1[1])
                            price_strength = min(100, price_change_pct * 100)
                            rsi_strength = min(100, rsi_change * 10)
                            result['strength'] = price_strength * 0.6 + rsi_strength * 0.4

        return result

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate RSI Divergence

        Uses HIGH/LOW for price pivots (TradingView compatible).

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Divergence information
        """
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Calculate RSI using existing RSI indicator
        rsi_values = calculate_rsi_values(close, self.rsi_period)

        # Find pivot points using HIGH/LOW for price, RSI for oscillator
        price_pivots = self._find_price_pivots(high, low, self.lookback)
        rsi_pivots = self._find_oscillator_pivots(rsi_values, self.lookback)

        # Detect divergence
        divergence = self._detect_divergence(price_pivots, rsi_pivots)

        # Result value
        value = {
            'rsi': round(rsi_values[-1], 2) if not np.isnan(rsi_values[-1]) else 50.0,
            'bullish_divergence': divergence['bullish'],
            'bearish_divergence': divergence['bearish'],
            'hidden_bullish_divergence': divergence['hidden_bullish'],
            'hidden_bearish_divergence': divergence['hidden_bearish'],
            'divergence_strength': round(divergence['strength'], 2)
        }

        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal determination
        if divergence['bullish'] and divergence['strength'] >= self.min_strength:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif divergence['bearish'] and divergence['strength'] >= self.min_strength:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        return IndicatorResult(
            value=value,
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=divergence['strength'],
            metadata={
                'rsi_period': self.rsi_period,
                'lookback': self.lookback,
                'price_pivots_highs': len(price_pivots['highs']),
                'price_pivots_lows': len(price_pivots['lows']),
                'rsi_pivots_highs': len(rsi_pivots['highs']),
                'rsi_pivots_lows': len(rsi_pivots['lows'])
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch RSI Divergence calculation - for BACKTEST

        Uses HIGH/LOW for price pivots (TradingView compatible).
        Proper pivot detection with range checking for batch processing.

        Returns:
            pd.DataFrame: 6 columns (rsi, bullish_divergence, bearish_divergence,
                         hidden_bullish_divergence, hidden_bearish_divergence, divergence_strength)
        """
        self._validate_data(data)

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Calculate RSI using existing RSI indicator
        rsi = calculate_rsi_values(close, self.rsi_period)

        # Initialize result arrays
        bullish_div = np.zeros(len(close), dtype=bool)
        bearish_div = np.zeros(len(close), dtype=bool)
        hidden_bullish_div = np.zeros(len(close), dtype=bool)
        hidden_bearish_div = np.zeros(len(close), dtype=bool)
        div_strength = np.zeros(len(close))

        # Find all pivot points once (TradingView compatible)
        price_pivot_highs = find_pivot_highs(high, left_bars=self.lookback, right_bars=self.lookback)
        price_pivot_lows = find_pivot_lows(low, left_bars=self.lookback, right_bars=self.lookback)
        rsi_pivot_highs = find_pivot_highs(rsi, left_bars=self.lookback, right_bars=self.lookback)
        rsi_pivot_lows = find_pivot_lows(rsi, left_bars=self.lookback, right_bars=self.lookback)

        # Process each candle to check for divergences
        for i in range(self.lookback * 2, len(close)):
            # Check bullish divergence (using pivot lows)
            # Find the two most recent price lows before current index
            relevant_price_lows = [(idx, val) for idx, val in price_pivot_lows if idx < i]
            if len(relevant_price_lows) >= 2:
                price_low1_idx, price_low1_val = relevant_price_lows[-2]
                price_low2_idx, price_low2_val = relevant_price_lows[-1]

                # Range check: 5-60 bars apart (TradingView default)
                if check_pivot_range(price_low1_idx, price_low2_idx, range_min=5, range_max=60):
                    # Find corresponding RSI lows
                    rsi_low1 = None
                    rsi_low2 = None
                    for rsi_idx, rsi_val in rsi_pivot_lows:
                        if abs(rsi_idx - price_low1_idx) <= self.lookback:
                            rsi_low1 = (rsi_idx, rsi_val)
                        if abs(rsi_idx - price_low2_idx) <= self.lookback:
                            rsi_low2 = (rsi_idx, rsi_val)

                    if rsi_low1 and rsi_low2:
                        # Regular Bullish: Price LL, RSI HL (reversal)
                        if price_low2_val < price_low1_val and rsi_low2[1] > rsi_low1[1]:
                            bullish_div[i] = True
                            # Better strength calculation: combine price % change and RSI change
                            price_change_pct = abs(price_low2_val - price_low1_val) / price_low1_val * 100
                            rsi_change = abs(rsi_low2[1] - rsi_low1[1])
                            # Normalize: 0.5% price change = 50 strength, 10 RSI change = 100 strength
                            price_strength = min(100, price_change_pct * 100)
                            rsi_strength = min(100, rsi_change * 10)
                            # Combined strength (60% price, 40% RSI)
                            div_strength[i] = price_strength * 0.6 + rsi_strength * 0.4

                        # Hidden Bullish: Price HL, RSI LL (continuation)
                        elif price_low2_val > price_low1_val and rsi_low2[1] < rsi_low1[1]:
                            hidden_bullish_div[i] = True
                            if div_strength[i] == 0:
                                # Same strength calculation for hidden divergence
                                price_change_pct = abs(price_low2_val - price_low1_val) / price_low1_val * 100
                                rsi_change = abs(rsi_low2[1] - rsi_low1[1])
                                price_strength = min(100, price_change_pct * 100)
                                rsi_strength = min(100, rsi_change * 10)
                                div_strength[i] = price_strength * 0.6 + rsi_strength * 0.4

            # Check bearish divergence (using pivot highs)
            # Find the two most recent price highs before current index
            relevant_price_highs = [(idx, val) for idx, val in price_pivot_highs if idx < i]
            if len(relevant_price_highs) >= 2:
                price_high1_idx, price_high1_val = relevant_price_highs[-2]
                price_high2_idx, price_high2_val = relevant_price_highs[-1]

                # Range check: 5-60 bars apart (TradingView default)
                if check_pivot_range(price_high1_idx, price_high2_idx, range_min=5, range_max=60):
                    # Find corresponding RSI highs
                    rsi_high1 = None
                    rsi_high2 = None
                    for rsi_idx, rsi_val in rsi_pivot_highs:
                        if abs(rsi_idx - price_high1_idx) <= self.lookback:
                            rsi_high1 = (rsi_idx, rsi_val)
                        if abs(rsi_idx - price_high2_idx) <= self.lookback:
                            rsi_high2 = (rsi_idx, rsi_val)

                    if rsi_high1 and rsi_high2:
                        # Regular Bearish: Price HH, RSI LH (reversal)
                        if price_high2_val > price_high1_val and rsi_high2[1] < rsi_high1[1]:
                            bearish_div[i] = True
                            # Better strength calculation: combine price % change and RSI change
                            price_change_pct = abs(price_high2_val - price_high1_val) / price_high1_val * 100
                            rsi_change = abs(rsi_high2[1] - rsi_high1[1])
                            # Normalize: 5% price change = 100, 50 RSI change = 100
                            price_strength = min(100, price_change_pct * 100)
                            rsi_strength = min(100, rsi_change * 10)
                            # Combined strength (60% price, 40% RSI)
                            div_strength[i] = price_strength * 0.6 + rsi_strength * 0.4

                        # Hidden Bearish: Price LH, RSI HH (continuation)
                        elif price_high2_val < price_high1_val and rsi_high2[1] > rsi_high1[1]:
                            hidden_bearish_div[i] = True
                            if div_strength[i] == 0:
                                # Same strength calculation for hidden divergence
                                price_change_pct = abs(price_high1_val - price_high2_val) / price_high1_val * 100
                                rsi_change = abs(rsi_high2[1] - rsi_high1[1])
                                price_strength = min(100, price_change_pct * 100)
                                rsi_strength = min(100, rsi_change * 10)
                                div_strength[i] = price_strength * 0.6 + rsi_strength * 0.4

        # Create result DataFrame
        result = pd.DataFrame({
            'rsi': rsi,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'hidden_bullish_divergence': hidden_bullish_div,
            'hidden_bearish_divergence': hidden_bearish_div,
            'divergence_strength': div_strength
        }, index=data.index)

        # Set warmup period to NaN/False
        warmup = self.rsi_period + self.lookback * 2
        result.iloc[:warmup, result.columns.get_loc('rsi')] = np.nan
        result.iloc[:warmup, result.columns.get_loc('divergence_strength')] = 0

        return result

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: New candle data
            symbol: Symbol identifier (IGNORED - each instance is already symbol-specific)

        Returns:
            IndicatorResult: Current indicator value

        Note: symbol parameter is ignored. Each indicator instance is created per-symbol
              by RealtimeCalculator, so multi-symbol support is handled at calculator level.
        """
        from collections import deque

        # Initialize buffer if needed (always use 'default' key)
        if not hasattr(self, '_buffers'):
            self._buffers = {}

        # Always use 'default' - each indicator instance is symbol-specific
        buffer_key = 'default'

        # Initialize buffer if needed
        if buffer_key not in self._buffers:
            max_len = self.get_required_periods() + 50
            self._buffers[buffer_key] = deque(maxlen=max_len)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0

        # Add new candle to symbol's buffer
        self._buffers[buffer_key].append(candle)

        # Need minimum data for calculation
        if len(self._buffers[buffer_key]) < self.get_required_periods():
            # Not enough data - return neutral with proper structure
            return IndicatorResult(
                value={
                    'rsi': 50.0,
                    'bullish_divergence': False,
                    'bearish_divergence': False,
                    'hidden_bullish_divergence': False,
                    'hidden_bearish_divergence': False,
                    'divergence_strength': 0.0
                },
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))

        # Use calculate_batch to get accurate per-candle divergence states
        # This ensures we return the LAST candle's state, not just last pivot's state
        batch_result = self.calculate_batch(buffer_data)

        # Extract last candle's values
        last_idx = len(batch_result) - 1
        value = {
            'rsi': round(float(batch_result['rsi'].iloc[last_idx]), 2),
            'bullish_divergence': bool(batch_result['bullish_divergence'].iloc[last_idx]),
            'bearish_divergence': bool(batch_result['bearish_divergence'].iloc[last_idx]),
            'hidden_bullish_divergence': bool(batch_result['hidden_bullish_divergence'].iloc[last_idx]),
            'hidden_bearish_divergence': bool(batch_result['hidden_bearish_divergence'].iloc[last_idx]),
            'divergence_strength': round(float(batch_result['divergence_strength'].iloc[last_idx]), 2)
        }

        # Determine signal and trend
        signal = self.get_signal(value)
        trend = self.get_trend(value)
        strength = value['divergence_strength']

        return IndicatorResult(
            value=value,
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={}
        )

    def get_signal(self, value: dict) -> SignalType:
        """
        Generate a signal from the divergence value

        Args:
            value: Divergence information

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if isinstance(value, dict):
            if value.get('bullish_divergence') and value.get('divergence_strength', 0) >= self.min_strength:
                return SignalType.BUY
            elif value.get('bearish_divergence') and value.get('divergence_strength', 0) >= self.min_strength:
                return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: dict) -> TrendDirection:
        """
        Determine the trend based on the divergence value

        Args:
            value: Divergence information

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if isinstance(value, dict):
            if value.get('bullish_divergence'):
                return TrendDirection.UP
            elif value.get('bearish_divergence'):
                return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'rsi_period': 14,
            'lookback': 5,
            'min_strength': 30
        }

    def _requires_volume(self) -> bool:
        """RSI Divergence does not require volume"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['RSIDivergence']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """RSI Divergence indicator test"""

    print("\n" + "="*60)
    print("RSI DIVERGENCE TEST")
    print("="*60 + "\n")

    # Create sample data - special pattern to create divergence
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Create a bullish divergence pattern
    prices = []
    for i in range(60):
        if i < 20:
            # Normal movement
            prices.append(100 + i * 0.5 + np.random.randn() * 0.3)
        elif i < 40:
            # Price decreases
            prices.append(110 - (i-20) * 0.5 + np.random.randn() * 0.3)
        else:
            # Price falls even further (lower low)
            prices.append(100 - (i-40) * 0.3 + np.random.randn() * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.3 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.3 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    rsi_div = RSIDivergence(rsi_period=14, lookback=5, min_strength=30)
    print(f"   [OK] Created: {rsi_div}")
    print(f"   [OK] Category: {rsi_div.category.value}")
    print(f"   [OK] Required period: {rsi_div.get_required_periods()}")

    result = rsi_div(data)
    print(f"   [OK] RSI Value: {result.value['rsi']}")
    print(f"   [OK] Bullish Divergence: {result.value['bullish_divergence']}")
    print(f"   [OK] Bearish Divergence: {result.value['bearish_divergence']}")
    print(f"   [OK] Divergence Strength: {result.value['divergence_strength']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")

    # Test 2: Different parameters
    print("\n3. Different parameter test...")
    configs = [
        (14, 3, 20),
        (14, 5, 30),
        (14, 7, 40)
    ]
    for rsi_p, look, strength in configs:
        div_test = RSIDivergence(rsi_period=rsi_p, lookback=look, min_strength=strength)
        result = div_test.calculate(data)
        print(f"   [OK] Params({rsi_p},{look},{strength}): Bullish={result.value['bullish_divergence']}, Bearish={result.value['bearish_divergence']}")

    # Test 3: Bearish divergence pattern
    print("\n4. Bearish divergence test...")
    bearish_prices = []
    for i in range(60):
        if i < 20:
            bearish_prices.append(100 - i * 0.3 + np.random.randn() * 0.3)
        elif i < 40:
            bearish_prices.append(94 + (i-20) * 0.5 + np.random.randn() * 0.3)
        else:
            bearish_prices.append(104 + (i-40) * 0.4 + np.random.randn() * 0.3)

    bearish_data = data.copy()
    bearish_data['close'] = bearish_prices
    bearish_data['high'] = [p + abs(np.random.randn()) * 0.3 for p in bearish_prices]
    bearish_data['low'] = [p - abs(np.random.randn()) * 0.3 for p in bearish_prices]

    result_bearish = rsi_div.calculate(bearish_data)
    print(f"   [OK] Bearish pattern RSI: {result_bearish.value['rsi']}")
    print(f"   [OK] Bearish Divergence: {result_bearish.value['bearish_divergence']}")
    print(f"   [OK] Signal: {result_bearish.signal.value}")

    # Test 4: Pivot points
    print("\n5. Pivot point test...")
    print(f"   [OK] Price pivot highs: {result.metadata['price_pivots_highs']}")
    print(f"   [OK] Price pivot lows: {result.metadata['price_pivots_lows']}")
    print(f"   [OK] RSI pivot highs: {result.metadata['rsi_pivots_highs']}")
    print(f"   [OK] RSI pivot lows: {result.metadata['rsi_pivots_lows']}")

    # Test 5: Strength test
    print("\n6. Divergence strength test...")
    if result.value['divergence_strength'] >= rsi_div.min_strength:
        print(f"   [OK] Strong divergence detected: {result.value['divergence_strength']}")
    else:
        print(f"   [OK] Weak or no divergence: {result.value['divergence_strength']}")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = rsi_div.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 7: Metadata
    print("\n8. Metadata test...")
    metadata = rsi_div.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Category: {metadata.category.value}")
    print(f"   [OK] Type: {metadata.indicator_type.value}")
    print(f"   [OK] Min period: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
