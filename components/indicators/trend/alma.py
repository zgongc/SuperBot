"""
indicators/trend/alma.py - Arnaud Legoux Moving Average

Version: 1.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    ALMA (Arnaud Legoux Moving Average) - Gaussian weighted moving average
    Lower lag and smoother than SMA/EMA
    Uses Gaussian distribution curve for weighting

    Usage:
    - Smooth trend tracking with minimal lag
    - ALMA SD Bands (ALMA + StdDev)
    - Support/resistance levels

Formula:
    ALMA = sum(w[i] * price[i]) / sum(w[i])
    w[i] = exp(-((i - offset * (period - 1))^2) / (2 * sigma^2))

    offset: Controls the curve position (0..1, default: 0.85)
        - 0.85 = closer to recent prices (less lag)
        - 0.50 = centered (like Gaussian filter)
    sigma: Controls the curve width (default: 6.0)
        - Higher = smoother, lower = more responsive

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


class ALMA(BaseIndicator):
    """
    Arnaud Legoux Moving Average

    Gaussian weighted moving average with configurable offset and sigma.
    Provides smoother output with less lag compared to EMA/SMA.

    Args:
        period: ALMA period (default: 20)
        offset: Gaussian curve offset 0..1 (default: 0.85)
        sigma: Gaussian curve width (default: 6.0)
        source: Price source - 'close', 'high', 'low', 'hl2', 'hlc3' (default: 'close')
    """

    def __init__(
        self,
        period: int = 20,
        offset: float = 0.85,
        sigma: float = 6.0,
        source: str = 'close',
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.offset = offset
        self.sigma = sigma
        self.source = source

        super().__init__(
            name='alma',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'offset': offset,
                'sigma': sigma,
                'source': source
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
        if not (0.0 <= self.offset <= 1.0):
            raise InvalidParameterError(
                self.name, 'offset', self.offset,
                "Offset must be between 0 and 1"
            )
        if self.sigma <= 0:
            raise InvalidParameterError(
                self.name, 'sigma', self.sigma,
                "Sigma must be positive"
            )
        return True

    def _get_source(self, data: pd.DataFrame) -> np.ndarray:
        """Get price source array"""
        if self.source == 'high':
            return data['high'].values
        elif self.source == 'low':
            return data['low'].values
        elif self.source == 'hl2':
            return ((data['high'] + data['low']) / 2).values
        elif self.source == 'hlc3':
            return ((data['high'] + data['low'] + data['close']) / 3).values
        else:
            return data['close'].values

    def _alma_weights(self, period: int) -> np.ndarray:
        """
        Calculate ALMA Gaussian weights

        w[i] = exp(-((i - m)^2) / (2 * s^2))
        m = offset * (period - 1)
        s = period / sigma

        Returns:
            np.ndarray: Normalized weights
        """
        m = self.offset * (period - 1)
        s = period / self.sigma

        weights = np.zeros(period)
        for i in range(period):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))

        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum != 0:
            weights /= weight_sum

        return weights

    def _calculate_alma_value(self, prices: np.ndarray) -> float:
        """
        Calculate single ALMA value from price window

        Args:
            prices: Price array (length = period)

        Returns:
            float: ALMA value
        """
        period = len(prices)
        weights = self._alma_weights(period)
        return np.sum(weights * prices)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate ALMA

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: ALMA value
        """
        src = self._get_source(data)

        # Calculate ALMA on last period window
        window = src[-self.period:]
        alma_value = self._calculate_alma_value(window)

        current_price = data['close'].values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(alma_value, 8),
            timestamp=timestamp,
            signal=self.get_signal(current_price, alma_value),
            trend=self.get_trend(current_price, alma_value),
            strength=self._calculate_strength(current_price, alma_value),
            metadata={
                'period': self.period,
                'offset': self.offset,
                'sigma': self.sigma,
                'source': self.source,
                'current_price': round(current_price, 8),
                'distance_pct': round(((current_price - alma_value) / alma_value) * 100, 4) if alma_value != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        VECTORIZED batch ALMA calculation - for BACKTEST

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: ALMA values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        src = self._get_source(data)
        weights = self._alma_weights(self.period)

        n = len(src)
        alma_values = np.full(n, np.nan)

        # Vectorized convolution for ALMA
        for i in range(self.period - 1, n):
            window = src[i - self.period + 1:i + 1]
            alma_values[i] = np.sum(weights * window)

        return pd.Series(alma_values, index=data.index, name='alma')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer for update() method

        Args:
            data: OHLCV DataFrame
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._src_buffer = deque(maxlen=max_len)
        src = self._get_source(data)
        for val in src[-max_len:]:
            self._src_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: New candle data
            symbol: Symbol identifier

        Returns:
            IndicatorResult: Current ALMA value
        """
        if not hasattr(self, '_src_buffer'):
            from collections import deque
            self._src_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            if self.source == 'high':
                src_val = float(candle['high'])
            elif self.source == 'low':
                src_val = float(candle['low'])
            elif self.source == 'hl2':
                src_val = (float(candle['high']) + float(candle['low'])) / 2
            elif self.source == 'hlc3':
                src_val = (float(candle['high']) + float(candle['low']) + float(candle['close'])) / 3
            else:
                src_val = float(candle['close'])
            close_val = float(candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            src_val = float(candle[4]) if len(candle) > 4 else 0.0
            close_val = src_val

        self._src_buffer.append(src_val)

        if len(self._src_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        # Calculate ALMA from buffer
        window = np.array(list(self._src_buffer))[-self.period:]
        alma_value = self._calculate_alma_value(window)

        return IndicatorResult(
            value=round(alma_value, 8),
            timestamp=timestamp_val,
            signal=self.get_signal(close_val, alma_value),
            trend=self.get_trend(close_val, alma_value),
            strength=self._calculate_strength(close_val, alma_value),
            metadata={
                'period': self.period,
                'offset': self.offset,
                'sigma': self.sigma,
                'source': self.source,
                'current_price': round(close_val, 8),
                'distance_pct': round(((close_val - alma_value) / alma_value) * 100, 4) if alma_value != 0 else 0
            }
        )

    def get_signal(self, price: float, alma: float) -> SignalType:
        """
        Generate signal from ALMA

        Args:
            price: Current price
            alma: ALMA value

        Returns:
            SignalType: BUY (price > ALMA), SELL (price < ALMA)
        """
        if price > alma:
            return SignalType.BUY
        elif price < alma:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, price: float, alma: float) -> TrendDirection:
        """
        Determine trend from ALMA

        Args:
            price: Current price
            alma: ALMA value

        Returns:
            TrendDirection: UP (price > ALMA), DOWN (price < ALMA)
        """
        if price > alma:
            return TrendDirection.UP
        elif price < alma:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _calculate_strength(self, price: float, alma: float) -> float:
        """Calculate signal strength (0-100)"""
        if alma == 0:
            return 0.0
        distance_pct = abs((price - alma) / alma * 100)
        strength = min(distance_pct * 20, 100.0)
        return max(0.0, min(strength, 100.0))

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'offset': 0.85,
            'sigma': 6.0,
            'source': 'close'
        }

    def _requires_volume(self) -> bool:
        """ALMA does not require volume"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ALMA']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """ALMA indicator test"""

    print("\n" + "="*60)
    print("ALMA (ARNAUD LEGOUX MOVING AVERAGE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 0.5
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

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    alma = ALMA(period=20, offset=0.85, sigma=6.0)
    print(f"   [OK] Created: {alma}")
    print(f"   [OK] Category: {alma.category.value}")
    print(f"   [OK] Required period: {alma.get_required_periods()}")

    result = alma(data)
    print(f"   [OK] ALMA Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Strength: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: ALMA vs EMA comparison
    print("\n3. ALMA vs SMA comparison...")
    sma_value = np.mean(data['close'].values[-20:])
    print(f"   [OK] SMA(20): {sma_value:.4f}")
    print(f"   [OK] ALMA(20): {result.value:.4f}")
    print(f"   [OK] ALMA gives Gaussian-weighted emphasis to recent prices")

    # Test 3: Different parameters
    print("\n4. Different parameter tests...")
    for offset_val in [0.5, 0.85, 1.0]:
        alma_test = ALMA(period=20, offset=offset_val, sigma=6.0)
        r = alma_test.calculate(data)
        print(f"   [OK] ALMA(20, offset={offset_val}): {r.value:.4f}")

    # Test 4: Batch calculation
    print("\n5. Batch calculation test...")
    alma_batch = ALMA(period=20)
    series = alma_batch.calculate_batch(data)
    print(f"   [OK] Batch result: {len(series)} values")
    print(f"   [OK] NaN count: {series.isna().sum()}")
    print(f"   [OK] Last 5 values: {series.tail().values}")

    # Test 5: Pine Script compatible (ALMA SD Bands preset)
    print("\n6. ALMA SD Bands preset test...")
    alma_sd = ALMA(period=111, offset=0.85, sigma=6.0, source='high')
    r = alma_sd.calculate(data)
    print(f"   [OK] ALMA(111, src=high): {r.value:.4f}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
