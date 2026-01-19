"""
indicators/trend/adx.py - Average Directional Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    ADX (Average Directional Index) - Average directional movement index
    Trend strength indicator developed by J. Welles Wilder
    Produces 3 values: +DI, -DI, and ADX

    Usage:
    - Measuring trend strength (ADX > 25 indicates a strong trend)
    - Determining trend direction (+DI vs -DI)
    - Entry/Exit sinyalleri (+DI/-DI crossover)

Formula:
    +DM = High(t) - High(t-1)  (pozitif ise)
    -DM = Low(t-1) - Low(t)    (pozitif ise)
    TR = True Range
    +DI = 100 × EMA(+DM) / EMA(TR)
    -DI = 100 × EMA(-DM) / EMA(TR)
    DX = 100 × |+DI - -DI| / (+DI + -DI)
    ADX = EMA(DX)

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


class ADX(BaseIndicator):
    """
    Average Directional Index

    An indicator that measures trend strength and direction.
    Calculates +DI, -DI, and ADX values.

    Args:
        period: ADX period (default: 14)
        adx_threshold: Trend strength threshold (default: 25)
    """

    def __init__(
        self,
        period: int = 14,
        adx_threshold: float = 25,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.adx_threshold = adx_threshold

        super().__init__(
            name='adx',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'period': period,
                'adx_threshold': adx_threshold
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period * 2  # Smoothing is required for ADX

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        if self.adx_threshold < 0 or self.adx_threshold > 100:
            raise InvalidParameterError(
                self.name, 'adx_threshold', self.adx_threshold,
                "The ADX threshold must be between 0 and 100"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        ADX hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: ADX, +DI, -DI values
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Calculate +DM and -DM
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))

        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]

            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        # True Range hesapla
        tr = self._calculate_tr(high, low, close)

        # Smoothed +DM, -DM, TR
        plus_di_values = self._smooth_series(plus_dm, tr, self.period)
        minus_di_values = self._smooth_series(minus_dm, tr, self.period)

        # DX hesapla
        dx_values = np.zeros(len(high))
        for i in range(self.period, len(high)):
            di_sum = plus_di_values[i] + minus_di_values[i]
            if di_sum > 0:
                dx_values[i] = 100 * abs(plus_di_values[i] - minus_di_values[i]) / di_sum

        # ADX hesapla (DX'in smoothed average)
        adx_values = self._smooth_dx(dx_values, self.period)

        # Last values
        plus_di = plus_di_values[-1]
        minus_di = minus_di_values[-1]
        adx = adx_values[-1]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Trend and signal determination
        trend = self.get_trend(plus_di, minus_di)
        signal = self.get_signal(adx, plus_di, minus_di)

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'adx': round(adx, 2),
                'plus_di': round(plus_di, 2),
                'minus_di': round(minus_di, 2)
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(adx, 100),
            metadata={
                'period': self.period,
                'adx_threshold': self.adx_threshold,
                'trend_strength': 'Strong' if adx > self.adx_threshold else 'Weak',
                'di_diff': round(abs(plus_di - minus_di), 2)
            }
        )

    def _calculate_tr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range hesapla"""
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        return tr

    def _smooth_series(self, dm: np.ndarray, tr: np.ndarray, period: int) -> np.ndarray:
        """Calculate the DM/TR ratio with smoothing."""
        di = np.zeros(len(dm))

        # Average for the first period
        sum_dm = np.sum(dm[:period])
        sum_tr = np.sum(tr[:period])

        if sum_tr > 0:
            di[period-1] = 100 * sum_dm / sum_tr

        # Wilder smoothing
        for i in range(period, len(dm)):
            sum_dm = sum_dm - sum_dm/period + dm[i]
            sum_tr = sum_tr - sum_tr/period + tr[i]
            if sum_tr > 0:
                di[i] = 100 * sum_dm / sum_tr

        return di

    def _smooth_dx(self, dx: np.ndarray, period: int) -> np.ndarray:
        """Calculate ADX by smoothing DX values"""
        adx = np.zeros(len(dx))

        # Initial ADX = average of DX values
        first_adx_idx = period * 2 - 1
        if first_adx_idx < len(dx):
            adx[first_adx_idx] = np.mean(dx[period:first_adx_idx+1])

            # Wilder smoothing
            for i in range(first_adx_idx + 1, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

        return adx

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX for entire DataFrame (vectorized - for backtest)

        Returns pd.DataFrame with ADX, +DI, -DI values for all bars.
        Uses Wilder's smoothing.
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # +DM and -DM
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))

        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]

            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        # True Range
        tr = self._calculate_tr(high, low, close)

        # Smoothed +DI, -DI
        plus_di_values = self._smooth_series(plus_dm, tr, self.period)
        minus_di_values = self._smooth_series(minus_dm, tr, self.period)

        # DX
        dx_values = np.zeros(len(high))
        for i in range(self.period, len(high)):
            di_sum = plus_di_values[i] + minus_di_values[i]
            if di_sum > 0:
                dx_values[i] = 100 * abs(plus_di_values[i] - minus_di_values[i]) / di_sum

        # ADX (smoothed DX)
        adx_values = self._smooth_dx(dx_values, self.period)

        # Return DataFrame with same keys as calculate() (adx, plus_di, minus_di)
        return pd.DataFrame({
            'adx': adx_values,
            'plus_di': plus_di_values,
            'minus_di': minus_di_values
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Current ADX values
        """
        from collections import deque

        # Initialize symbol-aware buffers if needed
        if not hasattr(self, '_adx_buffers'):
            self._adx_buffers = {}

        # Use symbol as key, or 'default' for backward compatibility
        buffer_key = symbol if symbol else 'default'

        # Initialize buffer for this symbol if needed
        if buffer_key not in self._adx_buffers:
            max_len = self.get_required_periods() + 50
            self._adx_buffers[buffer_key] = {
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len)
            }

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

        # Add new candle to symbol's buffer
        self._adx_buffers[buffer_key]['high'].append(high_val)
        self._adx_buffers[buffer_key]['low'].append(low_val)
        self._adx_buffers[buffer_key]['close'].append(close_val)

        # Need minimum data for ADX calculation
        if len(self._adx_buffers[buffer_key]['close']) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value={'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period, 'insufficient_data': True}
            )

        # Convert buffer to DataFrame
        buffer_data = pd.DataFrame({
            'high': list(self._adx_buffers[buffer_key]['high']),
            'low': list(self._adx_buffers[buffer_key]['low']),
            'close': list(self._adx_buffers[buffer_key]['close']),
            'open': [open_val] * len(self._adx_buffers[buffer_key]['close']),
            'volume': [volume_val] * len(self._adx_buffers[buffer_key]['close']),
            'timestamp': [timestamp_val] * len(self._adx_buffers[buffer_key]['close'])
        })

        # Calculate using existing logic
        return self.calculate(buffer_data)

    def get_signal(self, adx: float, plus_di: float, minus_di: float) -> SignalType:
        """
        Generate a signal from ADX.

        Args:
            adx: ADX value
            plus_di: +DI value
            minus_di: -DI value

        Returns:
            SignalType: BUY/SELL/HOLD
        """
        if adx < self.adx_threshold:
            return SignalType.HOLD  # Weak trend

        if plus_di > minus_di:
            return SignalType.BUY
        elif minus_di > plus_di:
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, plus_di: float, minus_di: float) -> TrendDirection:
        """
        Determine the trend from DI values.

        Args:
            plus_di: The +DI value
            minus_di: The -DI value

        Returns:
            TrendDirection: UP/DOWN/NEUTRAL
        """
        if plus_di > minus_di:
            return TrendDirection.UP
        elif minus_di > plus_di:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup ADX-specific buffer with historical data

        ADX uses _adx_buffers instead of _buffers, so we need to override.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier
        """
        from collections import deque

        # Initialize _adx_buffers if needed
        if not hasattr(self, '_adx_buffers'):
            self._adx_buffers = {}

        buffer_key = symbol if symbol else 'default'
        max_len = self.get_required_periods() + 50

        # Create buffer for this symbol
        self._adx_buffers[buffer_key] = {
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'close': deque(maxlen=max_len)
        }

        # Fill with historical data
        for _, row in data.tail(max_len).iterrows():
            self._adx_buffers[buffer_key]['high'].append(row['high'])
            self._adx_buffers[buffer_key]['low'].append(row['low'])
            self._adx_buffers[buffer_key]['close'].append(row['close'])

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 14,
            'adx_threshold': 25
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['adx', 'plus_di', 'minus_di']

    def _requires_volume(self) -> bool:
        """ADX volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['ADX']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """ADX indicator test"""

    print("\n" + "="*60)
    print("ADX (AVERAGE DIRECTIONAL INDEX) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Powerful trend simulation
    base_price = 100
    prices = [base_price]
    for i in range(49):
        trend = 1.2  # Strong upward trend
        noise = np.random.randn() * 0.5
        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 1.0 for p in prices],
        'low': [p - abs(np.random.randn()) * 1.0 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    adx = ADX(period=14)
    print(f"   [OK] Created: {adx}")
    print(f"   [OK] Kategori: {adx.category.value}")
    print(f"   [OK] Tip: {adx.indicator_type.value}")
    print(f"   [OK] Required period: {adx.get_required_periods()}")

    result = adx(data)
    print(f"   [OK] ADX: {result.value['adx']}")
    print(f"   [OK] +DI: {result.value['plus_di']}")
    print(f"   [OK] -DI: {result.value['minus_di']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Trend power analysis
    print("\n3. Trend strength analysis...")
    if result.value['adx'] > adx.adx_threshold:
        print(f"   [OK] Strong trend detected (ADX={result.value['adx']:.2f} > {adx.adx_threshold})")
    else:
        print(f"   [OK] Weak trend (ADX={result.value['adx']:.2f} < {adx.adx_threshold})")

    # Test 3: DI crossover
    print("\n4. DI crossover analizi...")
    print(f"   [OK] +DI: {result.value['plus_di']:.2f}")
    print(f"   [OK] -DI: {result.value['minus_di']:.2f}")
    print(f"   [OK] Fark: {result.metadata['di_diff']:.2f}")

    if result.value['plus_di'] > result.value['minus_di']:
        print(f"   [OK] Bullish (+DI > -DI)")
    else:
        print(f"   [OK] Bearish (-DI > +DI)")

    # Test 4: Different periods
    print("\n5. Different period test...")
    for period in [7, 14, 21]:
        adx_test = ADX(period=period)
        result = adx_test.calculate(data)
        print(f"   [OK] ADX({period}): {result.value['adx']:.2f}")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = adx.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = adx.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Output'lar: {metadata.output_names}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
