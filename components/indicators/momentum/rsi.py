"""
indicators/momentum/rsi.py - Relative Strength Index

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    RSI (Relative Strength Index) - Momentum oscillator
    Range: 0-100
    Overbought: > 70
    Oversold: < 30
    
Formula:
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
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


def calculate_rsi_values(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI hesapla - Wilder's smoothed method (TA-Lib uyumlu)

    Bu fonksiyon StochasticRSI gibi diğer indikatörler tarafından kullanılabilir.

    Args:
        close: Kapanış fiyatları (numpy array)
        period: RSI periyodu (varsayılan: 14)

    Returns:
        RSI değerleri (numpy array)
    """
    if len(close) < period + 1:
        return np.full(len(close), np.nan)

    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)

    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    rsi_values = np.zeros_like(close, dtype=float)

    # First RSI value uses SMA
    first_avg_gain = np.mean(gains[1:period + 1])
    first_avg_loss = np.mean(losses[1:period + 1])

    avg_gain = first_avg_gain
    avg_loss = first_avg_loss

    # Wilder's smoothed RSI (exponential moving average)
    for i in range(period, len(close)):
        if i == period:
            avg_gain = first_avg_gain
            avg_loss = first_avg_loss
        else:
            # Wilder's smoothing: new_avg = (prev_avg * (period-1) + current) / period
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values


class RSI(BaseIndicator):
    """
    Relative Strength Index
    
    Args:
        period: RSI period (default: 14)
        overbought: Overbought level (default: 70)
        oversold: Oversold level (default: 30)
    """
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name='rsi',
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
        return self.period + 1

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Override warmup to also initialize RSI state for incremental updates.

        This ensures update() can use Wilder's smoothing correctly.
        """
        # Call parent warmup_buffer
        super().warmup_buffer(data, symbol)

        # Initialize state directly (don't call calculate to avoid recursion)
        if len(data) >= self.period + 1:
            close = data['close'].values
            delta = np.diff(close)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)

            alpha = 1.0 / self.period
            avg_gain = np.mean(gains[:self.period])
            avg_loss = np.mean(losses[:self.period])

            for i in range(self.period, len(gains)):
                avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
                avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            buffer_key = symbol if symbol else 'default'
            if not hasattr(self, '_rsi_state'):
                self._rsi_state = {}
            self._rsi_state[buffer_key] = {
                'avg_gain': avg_gain,
                'avg_loss': avg_loss,
                'last_close': close[-1],
                'rsi': rsi
            }
    
    def validate_params(self) -> bool:
        if self.period < 1:
            raise InvalidParameterError(self.name, 'period', self.period, "Must be positive")
        if not 0 <= self.oversold < self.overbought <= 100:
            raise InvalidParameterError(self.name, 'levels', 
                                       f"oversold={self.oversold}, overbought={self.overbought}",
                                       "Invalid levels")
        return True
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate RSI (single value - for realtime)

        Uses Wilder's smoothing (RMA) - same as TA-Lib and TradingView.
        """
        close = data['close'].values

        if len(close) < self.period + 1:
            # Not enough data
            timestamp = int(data.iloc[-1]['timestamp']) if len(data) > 0 else 0
            return IndicatorResult(
                value=50.0,
                timestamp=timestamp,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period}
            )

        # Price changes
        delta = np.diff(close)

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Wilder's smoothing (RMA) - same as TA-Lib
        alpha = 1.0 / self.period

        # Initialize with SMA for first period
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        # Apply Wilder's smoothing for remaining periods
        for i in range(self.period, len(gains)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

        # Calculate RSI
        if avg_loss == 0:
            if avg_gain == 0:
                rsi = 50.0  # No movement = neutral
            else:
                rsi = 100.0  # All gains
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        timestamp = int(data.iloc[-1]['timestamp'])

        # Calculate strength: how far from neutral (50)
        # 0-100 scale where 100 = extreme overbought/oversold
        strength = min(100, abs(rsi - 50) * 2)

        # Save state for incremental update()
        # This allows update() to continue Wilder's smoothing
        if not hasattr(self, '_rsi_state'):
            self._rsi_state = {}
        self._rsi_state['default'] = {
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'last_close': close[-1],
            'rsi': rsi
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(rsi, 2),
            timestamp=timestamp,
            signal=self.get_signal(rsi),
            trend=self.get_trend(rsi),
            strength=round(strength, 2),
            metadata={'period': self.period}
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI for entire DataFrame (vectorized - for backtest)

        Returns pd.Series with RSI values for all bars.
        Uses Wilder's smoothing (RMA) - same as TradingView.
        """
        close = data['close'].values

        # Calculate price changes
        delta = np.diff(close)

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Wilder's smoothing (RMA)
        alpha = 1.0 / self.period
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))

        # Initialize with SMA for first period
        if len(gains) >= self.period:
            avg_gain[self.period] = np.mean(gains[:self.period])
            avg_loss[self.period] = np.mean(losses[:self.period])

            # Apply Wilder's smoothing for rest
            for i in range(self.period + 1, len(close)):
                avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gains[i-1] * alpha
                avg_loss[i] = avg_loss[i-1] * (1 - alpha) + losses[i-1] * alpha

        # Calculate RSI
        rsi = np.zeros(len(close))
        for i in range(len(close)):
            if avg_loss[i] == 0:
                if avg_gain[i] == 0:
                    rsi[i] = 50.0  # No movement = neutral
                else:
                    rsi[i] = 100.0  # All gains
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))

        return pd.Series(rsi, index=data.index, name='rsi')

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time) - Uses Wilder's smoothing state

        If calculate() was called previously, uses saved state for
        true incremental Wilder smoothing. Otherwise falls back to
        buffer-based calculation.

        Args:
            candle: Yeni mum verisi
            symbol: Symbol identifier (for multi-symbol support)

        Returns:
            IndicatorResult: Güncel RSI değeri
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle.get('close', 0)
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        buffer_key = symbol if symbol else 'default'

        # Check if we have saved state from calculate()
        if hasattr(self, '_rsi_state') and buffer_key in self._rsi_state:
            state = self._rsi_state[buffer_key]
            last_close = state['last_close']
            avg_gain = state['avg_gain']
            avg_loss = state['avg_loss']

            # Calculate price change
            delta = close_val - last_close

            # Separate gain and loss
            gain = delta if delta > 0 else 0
            loss = -delta if delta < 0 else 0

            # Wilder's smoothing
            alpha = 1.0 / self.period
            avg_gain = avg_gain * (1 - alpha) + gain * alpha
            avg_loss = avg_loss * (1 - alpha) + loss * alpha

            # Calculate RSI
            if avg_loss == 0:
                if avg_gain == 0:
                    rsi = 50.0
                else:
                    rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Update state
            self._rsi_state[buffer_key] = {
                'avg_gain': avg_gain,
                'avg_loss': avg_loss,
                'last_close': close_val,
                'rsi': rsi
            }

            strength = min(100, abs(rsi - 50) * 2)

            return IndicatorResult(
                value=round(rsi, 2),
                timestamp=timestamp_val,
                signal=self.get_signal(rsi),
                trend=self.get_trend(rsi),
                strength=round(strength, 2),
                metadata={'period': self.period}
            )

        # Fallback: use buffer-based calculation
        from collections import deque

        if not hasattr(self, '_buffers'):
            self._buffers = {}

        if buffer_key not in self._buffers:
            self._buffers[buffer_key] = deque(maxlen=max(100, self.period * 5))

        self._buffers[buffer_key].append(candle)

        if len(self._buffers[buffer_key]) < self.period + 1:
            return IndicatorResult(
                value=50.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'period': self.period}
            )

        buffer_data = pd.DataFrame(list(self._buffers[buffer_key]))
        return self.calculate(buffer_data)
    
    def get_signal(self, value: float) -> SignalType:
        """RSI signals"""
        if value < self.oversold:
            return SignalType.BUY
        elif value > self.overbought:
            return SignalType.SELL
        return SignalType.HOLD
    
    def get_trend(self, value: float) -> TrendDirection:
        """RSI trend"""
        if value > 50:
            return TrendDirection.UP
        elif value < 50:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL


if __name__ == "__main__":
    # Quick test
    data = pd.DataFrame({
        'timestamp': [i * 60000 for i in range(30)],
        'close': [100 + i + np.random.randn() for i in range(30)],
        'open': [100 + i for i in range(30)],
        'high': [102 + i for i in range(30)],
        'low': [98 + i for i in range(30)],
        'volume': [1000] * 30
    })
    
    rsi = RSI(period=14)
    result = rsi.calculate(data)
    print(f"RSI: {result.value} | Signal: {result.signal.value} | Trend: {result.trend.name}")


# Module exports
__all__ = ['RSI', 'calculate_rsi_values']
