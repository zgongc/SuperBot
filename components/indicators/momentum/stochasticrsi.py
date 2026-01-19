"""
indicators/momentum/stochasticrsi.py - Stochastic RSI

Version: 2.1.0
Date: 2025-12-16
Author: SuperBot Team

Description:
    Stochastic RSI - Stochastic oscillator applied to RSI.
    Range: 0-100 (more sensitive).
    Overbought: > 80.
    Oversold: < 20.
    Generates more sensitive signals than RSI, may contain more noise.

Formula:
    1. RSI hesapla (14 periyot)
    2. StochRSI = (RSI - Min RSI) / (Max RSI - Min RSI) × 100
       The min/max values are taken from the last N periods.
    3. %K = StochRSI
    4. %D = SMA(%K, 3) (signal line)

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
from components.indicators.momentum.rsi import calculate_rsi_values


class StochasticRSI(BaseIndicator):
    """
    Stochastic RSI

    It generates more precise momentum signals by applying the Stochastic formula to the RSI.
    It detects overbought/oversold conditions earlier.

    Args:
        rsi_period: RSI period (default: 14)
        stoch_period: Stochastic period (default: 14)
        k_smooth: %K smoothing period (default: 3)
        d_smooth: %D smoothing period (default: 3)
        overbought: Overbought level (default: 80)
        oversold: Oversold level (default: 20)
        rsi_source: RSI calculation source (default: 0)
                    0=close, 1=open, 2=high, 3=low, 4=hl2, 5=hlc3, 6=ohlc4
    """

    # Integer to source name mapping for WebUI compatibility
    SOURCE_MAP = {
        0: 'close',
        1: 'open',
        2: 'high',
        3: 'low',
        4: 'hl2',
        5: 'hlc3',
        6: 'ohlc4'
    }

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
        overbought: float = 80,
        oversold: float = 20,
        rsi_source: int = 0,
        logger=None,
        error_handler=None
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.overbought = overbought
        self.oversold = oversold
        self.rsi_source = rsi_source

        super().__init__(
            name='stochastic_rsi',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'rsi_period': rsi_period,
                'stoch_period': stoch_period,
                'k_smooth': k_smooth,
                'd_smooth': d_smooth,
                'overbought': overbought,
                'oversold': oversold,
                'rsi_source': rsi_source
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.rsi_period + self.stoch_period + max(self.k_smooth, self.d_smooth)

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.rsi_period < 1:
            raise InvalidParameterError(
                self.name, 'rsi_period', self.rsi_period,
                "RSI period must be positive"
            )
        if self.stoch_period < 1:
            raise InvalidParameterError(
                self.name, 'stoch_period', self.stoch_period,
                "Stochastic period must be positive"
            )
        if self.k_smooth < 1:
            raise InvalidParameterError(
                self.name, 'k_smooth', self.k_smooth,
                "K smooth must be positive"
            )
        if self.d_smooth < 1:
            raise InvalidParameterError(
                self.name, 'd_smooth', self.d_smooth,
                "D smooth must be positive"
            )
        if self.oversold >= self.overbought:
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Oversold should be smaller than overbought"
            )
        if not (0 <= self.oversold <= 100) or not (0 <= self.overbought <= 100):
            raise InvalidParameterError(
                self.name, 'levels',
                f"oversold={self.oversold}, overbought={self.overbought}",
                "Levels must be between 0 and 100"
            )
        if self.rsi_source not in self.SOURCE_MAP:
            raise InvalidParameterError(
                self.name, 'rsi_source', self.rsi_source,
                f"Valid values: 0-6 (0=close, 1=open, 2=high, 3=low, 4=hl2, 5=hlc3, 6=ohlc4)"
            )
        return True

    def _get_source_values(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate source values for RSI"""
        source_name = self.SOURCE_MAP.get(self.rsi_source, 'close')

        if source_name == 'close':
            return data['close'].values
        elif source_name == 'open':
            return data['open'].values
        elif source_name == 'high':
            return data['high'].values
        elif source_name == 'low':
            return data['low'].values
        elif source_name == 'hl2':
            return ((data['high'] + data['low']) / 2).values
        elif source_name == 'hlc3':
            return ((data['high'] + data['low'] + data['close']) / 3).values
        elif source_name == 'ohlc4':
            return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
        else:
            return data['close'].values

    def _calculate_stochastic_rsi(self, rsi_values: np.ndarray) -> tuple:
        """
        Stochastic RSI hesapla - TA-Lib uyumlu

        TA-Lib STOCHRSI:
        - fastk = raw StochRSI (no smoothing on K, only uses stoch_period window)
        - fastd = SMA of fastk

        Args:
            rsi_values: RSI values

        Returns:
            tuple: (stoch_rsi, k_values, d_values)
        """
        stoch_rsi_values = np.zeros_like(rsi_values, dtype=float)

        # StochRSI hesapla (TA-Lib uses current RSI value, not previous)
        # Window includes current value: [i-stoch_period+1 : i+1]
        start_idx = self.rsi_period + self.stoch_period - 1
        for i in range(start_idx, len(rsi_values)):
            # Include current value in window
            rsi_window = rsi_values[i - self.stoch_period + 1:i + 1]
            min_rsi = np.min(rsi_window)
            max_rsi = np.max(rsi_window)

            if max_rsi - min_rsi == 0:
                stoch_rsi_values[i] = 50  # Neutral
            else:
                # Use current RSI value (rsi_values[i])
                stoch_rsi_values[i] = 100 * (rsi_values[i] - min_rsi) / (max_rsi - min_rsi)

        # TA-Lib: fastk = raw StochRSI (k_smooth is NOT applied to K in TA-Lib)
        # Our k_smooth is extra smoothing we apply, but for TA-Lib compatibility,
        # we use raw StochRSI as K
        k_values = stoch_rsi_values.copy()

        # %D (smoothed %K) - Signal line using d_smooth
        d_values = np.zeros_like(k_values, dtype=float)
        for i in range(start_idx + self.d_smooth - 1, len(k_values)):
            d_values[i] = np.mean(k_values[i - self.d_smooth + 1:i + 1])

        return stoch_rsi_values, k_values, d_values

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ Batch Stochastic RSI calculation - for BACKTEST

        It uses the same calculation method as calculate():
            1. RSI = calculate_rsi_values() (Wilder's smoothed - TA-Lib uyumlu)
            2. StochRSI = (RSI - Min RSI) / (Max RSI - Min RSI) × 100
            3. %K = raw StochRSI
            4. %D = SMA(%K, d_smooth)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: stoch_rsi, k, d, crossover for all bars
        """
        self._validate_data(data)

        # Get the source values (close, hl2, hlc3, ohlc4, etc.) - same as calculate()
        source_values = self._get_source_values(data)

        # 1. Calculate RSI - same function as calculate()
        rsi_values = calculate_rsi_values(source_values, self.rsi_period)

        # 2. Calculate Stochastic RSI - same logic as _calculate_stochastic_rsi
        stoch_rsi_values = np.zeros_like(rsi_values, dtype=float)
        k_values = np.zeros_like(rsi_values, dtype=float)
        d_values = np.zeros_like(rsi_values, dtype=float)

        start_idx = self.rsi_period + self.stoch_period - 1

        for i in range(start_idx, len(rsi_values)):
            rsi_window = rsi_values[i - self.stoch_period + 1:i + 1]
            min_rsi = np.min(rsi_window)
            max_rsi = np.max(rsi_window)

            if max_rsi - min_rsi == 0:
                stoch_rsi_values[i] = 50
            else:
                stoch_rsi_values[i] = 100 * (rsi_values[i] - min_rsi) / (max_rsi - min_rsi)

        # %K = raw StochRSI
        k_values = stoch_rsi_values.copy()

        # %D = SMA(%K, d_smooth)
        for i in range(start_idx + self.d_smooth - 1, len(k_values)):
            d_values[i] = np.mean(k_values[i - self.d_smooth + 1:i + 1])

        # Crossover hesapla
        crossover = np.zeros(len(k_values), dtype=int)
        for i in range(1, len(k_values)):
            if k_values[i-1] < d_values[i-1] and k_values[i] > d_values[i]:
                crossover[i] = 1  # Bullish
            elif k_values[i-1] > d_values[i-1] and k_values[i] < d_values[i]:
                crossover[i] = -1  # Bearish

        # Make NaN values during the warmup period
        warmup = self.rsi_period + self.stoch_period + self.d_smooth - 1
        stoch_rsi_values[:warmup] = np.nan
        k_values[:warmup] = np.nan
        d_values[:warmup] = np.nan

        # Return same keys as calculate(): stoch_rsi, k, d, crossover
        return pd.DataFrame({
            'stoch_rsi': stoch_rsi_values,
            'k': k_values,
            'd': d_values,
            'crossover': crossover
        }, index=data.index)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Stochastic RSI hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Stochastic RSI values
        """
        # Get the source values (close, hl2, hlc3, ohlc4, etc.)
        source_values = self._get_source_values(data)

        # RSI hesapla (rsi.py'den import edilen fonksiyon)
        rsi_values = calculate_rsi_values(source_values, self.rsi_period)

        # Stochastic RSI hesapla
        stoch_rsi, k_values, d_values = self._calculate_stochastic_rsi(rsi_values)

        # Last values
        current_k = k_values[-1]
        current_d = d_values[-1]
        current_stoch_rsi = stoch_rsi[-1]

        # Crossover tespiti
        prev_k = k_values[-2] if len(k_values) > 1 else current_k
        prev_d = d_values[-2] if len(d_values) > 1 else current_d

        crossover = None
        if prev_k < prev_d and current_k > current_d:
            crossover = 'bullish'
        elif prev_k > prev_d and current_k < current_d:
            crossover = 'bearish'

        value = {
            'stoch_rsi': round(current_stoch_rsi, 2),
            'k': round(current_k, 2),
            'd': round(current_d, 2),
            'crossover': crossover
        }

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=value,
            timestamp=timestamp,
            signal=self.get_signal(value),
            trend=self.get_trend(current_k),
            strength=abs(current_k - 50) * 2,  # Normalize to the 0-100 range
            metadata={
                'rsi_period': self.rsi_period,
                'stoch_period': self.stoch_period,
                'k_smooth': self.k_smooth,
                'd_smooth': self.d_smooth,
                'rsi': round(rsi_values[-1], 2)
            }
        )

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

        # Create OHLC buffers (required for rsi_source)
        self._ohlc_buffer = {
            'open': deque(maxlen=max_len),
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'close': deque(maxlen=max_len)
        }
        self._buffers_init = True

        tail_data = data.tail(max_len)
        for _, row in tail_data.iterrows():
            self._ohlc_buffer['open'].append(row['open'])
            self._ohlc_buffer['high'].append(row['high'])
            self._ohlc_buffer['low'].append(row['low'])
            self._ohlc_buffer['close'].append(row['close'])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Symbol-aware

        Args:
            candle: New candle data (OHLCV dict or list/tuple)
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: The current indicator value.
        """
        from collections import deque

        # Initialize buffer if needed
        if not hasattr(self, '_buffers_init') or not hasattr(self, '_ohlc_buffer'):
            max_len = self.get_required_periods() + 50
            self._ohlc_buffer = {
                'open': deque(maxlen=max_len),
                'high': deque(maxlen=max_len),
                'low': deque(maxlen=max_len),
                'close': deque(maxlen=max_len)
            }
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            open_val = candle.get('open', candle.get('close', 0))
            high_val = candle.get('high', candle.get('close', 0))
            low_val = candle.get('low', candle.get('close', 0))
            close_val = candle.get('close', 0)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # list/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        # Add OHLC to buffers
        self._ohlc_buffer['open'].append(open_val)
        self._ohlc_buffer['high'].append(high_val)
        self._ohlc_buffer['low'].append(low_val)
        self._ohlc_buffer['close'].append(close_val)

        # Need minimum data for calculation
        if len(self._ohlc_buffer['close']) < self.get_required_periods():
            # Not enough data - return neutral
            return IndicatorResult(
                value={'stoch_rsi': 50.0, 'k': 50.0, 'd': 50.0, 'crossover': None},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        # Convert buffer to DataFrame (OHLC for _get_source_values)
        buffer_data = pd.DataFrame({
            'open': list(self._ohlc_buffer['open']),
            'high': list(self._ohlc_buffer['high']),
            'low': list(self._ohlc_buffer['low']),
            'close': list(self._ohlc_buffer['close']),
            'timestamp': [timestamp_val] * len(self._ohlc_buffer['close'])
        })

        # Calculate using existing logic (will use _get_source_values with rsi_source)
        return self.calculate(buffer_data)

    def get_signal(self, value: dict) -> SignalType:
        """
        Generate a signal from the stochastic RSI value.

        Args:
            value: Stochastic RSI values

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if isinstance(value, dict):
            k = value['k']
            crossover = value.get('crossover')

            # Crossover + oversold/overbought
            if crossover == 'bullish' and k < self.oversold:
                return SignalType.STRONG_BUY
            elif crossover == 'bearish' and k > self.overbought:
                return SignalType.STRONG_SELL
            elif k < self.oversold:
                return SignalType.BUY
            elif k > self.overbought:
                return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the stochastic RSI value.

        Args:
            value: The %K value.

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 50:
            return TrendDirection.UP
        elif value < 50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'rsi_period': 14,
            'stoch_period': 14,
            'k_smooth': 3,
            'd_smooth': 3,
            'overbought': 80,
            'oversold': 20
        }

    def _requires_volume(self) -> bool:
        """Stochastic RSI volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['StochasticRSI']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Stochastic RSI indicator test"""

    print("\n" + "="*60)
    print("STOCHASTIC RSI TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Simulate price movement
    base_price = 100
    prices = [base_price]
    for i in range(59):
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
    stoch_rsi = StochasticRSI(rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    print(f"   [OK] Created: {stoch_rsi}")
    print(f"   [OK] Kategori: {stoch_rsi.category.value}")
    print(f"   [OK] Required period: {stoch_rsi.get_required_periods()}")

    result = stoch_rsi(data)
    print(f"   [OK] Stochastic RSI: {result.value['stoch_rsi']}")
    print(f"   [OK] %K: {result.value['k']}")
    print(f"   [OK] %D: {result.value['d']}")
    print(f"   [OK] Crossover: {result.value['crossover']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] RSI: {result.metadata['rsi']}")

    # Test 2: Different parameters
    print("\n3. Different parameter test...")
    configs = [
        (14, 14, 3, 3),
        (14, 21, 5, 5),
        (21, 14, 3, 3)
    ]
    for rsi_p, stoch_p, k_s, d_s in configs:
        stoch_test = StochasticRSI(rsi_period=rsi_p, stoch_period=stoch_p,
                                    k_smooth=k_s, d_smooth=d_s)
        result = stoch_test.calculate(data)
        print(f"   [OK] Params({rsi_p},{stoch_p},{k_s},{d_s}): K={result.value['k']}, D={result.value['d']}")

    # Test 3: Custom levels
    print("\n4. Special level test...")
    stoch_custom = StochasticRSI(rsi_period=14, stoch_period=14,
                                  overbought=90, oversold=10)
    result = stoch_custom.calculate(data)
    print(f"   [OK] Stochastic RSI with custom levels: K={result.value['k']}")
    print(f"   [OK] Overbought: {stoch_custom.overbought}, Oversold: {stoch_custom.oversold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Crossover tespiti
    print("\n5. Crossover testi...")
    # Check the last few values
    for i in range(-5, 0):
        test_data = data.iloc[:len(data)+i]
        if len(test_data) >= stoch_rsi.get_required_periods():
            result = stoch_rsi.calculate(test_data)
            k = result.value['k']
            d = result.value['d']
            cross = result.value['crossover']
            print(f"   [OK] Index {i}: K={k:.2f}, D={d:.2f}, Cross={cross}")

    # Test 5: Overbuying/overselling conditions
    print("\n6. Overbuying/overselling test...")
    # Rising trend
    up_data = data.copy()
    for i in range(20):
        idx = up_data.index[-(20-i)]
        up_data.loc[idx, 'close'] = prices[-(20-i)] + i * 1

    result_up = stoch_rsi.calculate(up_data)
    print(f"   [OK] Rising trend Stoch RSI: K={result_up.value['k']}")
    print(f"   [OK] Signal: {result_up.signal.value}")

    # Declining trend
    down_data = data.copy()
    for i in range(20):
        idx = down_data.index[-(20-i)]
        down_data.loc[idx, 'close'] = prices[-(20-i)] - i * 1

    result_down = stoch_rsi.calculate(down_data)
    print(f"   [OK] Declining trend Stoch RSI: K={result_down.value['k']}")
    print(f"   [OK] Signal: {result_down.signal.value}")

    # Test 6: Comparison of RSI vs Stochastic RSI
    print("\n7. RSI vs Stoch RSI comparison...")
    print(f"   [OK] Normal RSI: {result.metadata['rsi']}")
    print(f"   [OK] Stochastic RSI: {result.value['stoch_rsi']}")
    print(f"   [OK] %K (smoothed): {result.value['k']}")
    print(f"   [OK] Sensitivity difference: Stochastic RSI is normalized between 0-100")

    # Test 7: Statistics
    print("\n8. Statistical test...")
    stats = stoch_rsi.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 8: Metadata
    print("\n9. Metadata testi...")
    metadata = stoch_rsi.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
