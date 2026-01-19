"""
indicators/momentum/stochastic.py - Stochastic Oscillator

Version: 2.0.0
Date: 2025-10-14

Description:
    Stochastic Oscillator (%K and %D lines)
    Range: 0-100
    Overbought: > 80
    Oversold: < 20
    
Formula:
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
"""

import numpy as np
import pandas as pd
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory, IndicatorType, IndicatorResult, SignalType, TrendDirection
)


class Stochastic(BaseIndicator):
    """Stochastic Oscillator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, logger=None, error_handler=None):
        # IMPORTANT: Set instance variables BEFORE calling super().__init__
        # BaseIndicator.__init__ calls _build_metadata() which calls get_required_periods()
        # which needs self.k_period and self.d_period to be already set
        self.k_period = k_period
        self.d_period = d_period

        super().__init__(
            name='stochastic',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'k_period': k_period, 
                'd_period': d_period
            },
            logger=logger, error_handler=error_handler
        )
    
    def get_required_periods(self) -> int:
        return self.k_period + self.d_period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Stochastic Oscillator - matches TA-Lib STOCH

        TA-Lib STOCH uses:
        - fastk_period = k_period (default 14)
        - slowk_period = d_period (default 3) - SMA smoothing of raw %K
        - slowd_period = d_period (default 3) - SMA smoothing of slow %K

        Our output: k = slowK, d = slowD (same as TA-Lib default)
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        n = len(close)
        required = self.k_period + self.d_period * 2 - 1
        if n < required:
            return IndicatorResult(
                value={'k': 50.0, 'd': 50.0},
                timestamp=int(data.iloc[-1]['timestamp']) if n > 0 else 0,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'k_period': self.k_period, 'd_period': self.d_period}
            )

        # Calculate FastK values for the last (d_period * 2 - 1) bars
        # We need d_period FastK values for current SlowK
        # And d_period SlowK values for SlowD
        # So we need (d_period + d_period - 1) = (d_period * 2 - 1) FastK values
        num_fastk = self.d_period * 2 - 1
        fastk_values = []

        for i in range(num_fastk):
            # end_idx is the last bar index (inclusive) for this FastK calculation
            end_idx = n - (num_fastk - 1 - i)
            start_idx = end_idx - self.k_period

            lowest_low = np.min(low[start_idx:end_idx])
            highest_high = np.max(high[start_idx:end_idx])

            if highest_high == lowest_low:
                fastk_values.append(50.0)
            else:
                fastk_values.append(((close[end_idx - 1] - lowest_low) / (highest_high - lowest_low)) * 100)

        fastk_values = np.array(fastk_values)

        # SlowK = SMA of FastK over d_period
        # Calculate d_period SlowK values for SlowD
        slow_k_values = []
        for i in range(self.d_period):
            start = i
            end = i + self.d_period
            slow_k_values.append(np.mean(fastk_values[start:end]))

        slow_k_values = np.array(slow_k_values)
        k_value = slow_k_values[-1]  # Current SlowK

        # SlowD = SMA of SlowK over d_period
        d_value = np.mean(slow_k_values)
        
        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'k': round(k_value, 2), 'd': round(d_value, 2)},
            timestamp=int(data.iloc[-1]['timestamp']),
            signal=SignalType.BUY if k_value < 20 else SignalType.SELL if k_value > 80 else SignalType.HOLD,
            trend=TrendDirection.UP if k_value > d_value else TrendDirection.DOWN if k_value < d_value else TrendDirection.NEUTRAL,
            strength=abs(k_value - 50) * 2,  # 0-100 range
            metadata={
                'k_period': self.k_period,
                'd_period': self.d_period,
                'k_value': round(k_value, 2),
                'd_value': round(d_value, 2)
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
                value={'k': 50.0, 'd': 50.0},
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

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch Stochastic calculation - for BACKTEST

        TA-Lib STOCH compatible: Calculates SlowK and SlowD.

        TA-Lib Formula:
            FastK = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
            SlowK = SMA(FastK, d_period)  # smoothed FastK
            SlowD = SMA(SlowK, d_period)  # smoothed SlowK

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: 2 columns (k=SlowK, d=SlowD)

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # 1. Calculate FastK (VECTORIZED - NO LOOPS!)
        # Lowest low and highest high over k_period
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()

        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        # FastK = (close - lowest_low) / (highest_high - lowest_low) * 100
        fast_k = ((close - lowest_low) / denominator) * 100

        # 2. SlowK = SMA of FastK over d_period
        slow_k = fast_k.rolling(window=self.d_period).mean()

        # 3. SlowD = SMA of SlowK over d_period
        slow_d = slow_k.rolling(window=self.d_period).mean()

        # Create result DataFrame (same keys as calculate())
        result = pd.DataFrame({
            'k': slow_k,
            'd': slow_d
        }, index=data.index)

        # Set first period values to NaN (warmup)
        warmup = self.k_period + self.d_period * 2 - 1
        result.iloc[:warmup] = np.nan

        return result


if __name__ == "__main__":
    data = pd.DataFrame({
        'timestamp': [i * 60000 for i in range(30)],
        'close': [100 + i for i in range(30)],
        'high': [102 + i for i in range(30)],
        'low': [98 + i for i in range(30)],
        'open': [100 + i for i in range(30)],
        'volume': [1000] * 30
    })
    
    stoch = Stochastic()
    result = stoch.calculate(data)
    print(f"Stochastic: %K={result.value['k']}, %D={result.value['d']}")