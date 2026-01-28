#!/usr/bin/env python3
"""
components/indicators/community/pmax.py
SuperBot - PMax Indicator
Author: Kivanc Ozbilgic (@KivancOzbilgic)
Port: SuperBot Team
Date: 2026-01-28
Version: 1.0.0

PMax - ATR-based trend indicator that creates dynamic support/resistance levels.
Uses EMA and ATR to determine trend direction and generate buy/sell signals.

Original PineScript:
    //@version=4
    //developer: KivancOzbilgic
    //author: KivancOzbilgic
    strategy("PMax Explorer", shorttitle="PMEx", overlay=true)

Features:
- Dynamic support/resistance levels based on volatility
- Clear trend direction (UP/DOWN)
- Reduced noise with EMA smoothing
- ATR-based stop levels
- Realtime calculation with incremental updates

Signal:
- Trend = 1 (UP): PMax = longStop (support level)
- Trend = -1 (DOWN): PMax = shortStop (resistance level)
- Crossover/Crossunder = Buy/Sell signal

Formula:
    Source = (High + Low) / 2
    MAvg = EMA(Source, ma_period)
    ATR = ATR(atr_period)

    longStop = MAvg - (atr_multiplier × ATR)
    shortStop = MAvg + (atr_multiplier × ATR)

    Trend direction logic:
    - UP trend: when MAvg > previous shortStop
    - DOWN trend: when MAvg < previous longStop

    PMax = Trend == 1 ? longStop : shortStop

Usage:
    from components.indicators.community.pmax import PMax

    pmax = PMax(atr_period=10, atr_multiplier=3.0, ma_period=10)
    result = pmax.calculate(data)
    print(result.value)

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
    - components.indicators.trend.ema (internal)
    - components.indicators.volatility.atr (internal)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from collections import deque

from components.indicators.base_indicator import BaseIndicator
from components.indicators.trend.ema import EMA
from components.indicators.volatility.atr import ATR
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class PMax(BaseIndicator):
    """
    PMax - ATR-based Trend Indicator by Kivanc Ozbilgic

    Creates dynamic support/resistance levels using EMA and ATR.
    Provides clear trend direction and buy/sell signals.

    Args:
        atr_period: ATR calculation period (default: 10)
        atr_multiplier: ATR multiplier for stop levels (default: 3.0)
        ma_period: EMA period (default: 10)

    Outputs:
        value: Dict with 'pmax', 'long_stop', 'short_stop', 'mavg'
        signal: BUY (uptrend), SELL (downtrend), HOLD
        trend: UP (trend=1), DOWN (trend=-1)

    Required Periods:
        max(atr_period + 1, ma_period) + buffer
    """

    def __init__(
        self,
        atr_period: int = 10,
        atr_multiplier: float = 3.0,
        ma_period: int = 10,
        logger=None,
        error_handler=None
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.ma_period = ma_period

        super().__init__(
            name='pmax',
            category=IndicatorCategory.TREND,
            indicator_type=IndicatorType.MULTIPLE_VALUES,
            params={
                'atr_period': atr_period,
                'atr_multiplier': atr_multiplier,
                'ma_period': ma_period
            },
            logger=logger,
            error_handler=error_handler
        )

        # Component indicators
        self._ema = EMA(period=ma_period)
        self._atr = ATR(period=atr_period)

        # Buffers for update()
        self._data_buffer = None
        self._pmax_buffer = None
        self._trend_buffer = None

    def get_required_periods(self) -> int:
        """
        Minimum required number of periods.

        ATR needs atr_period + 1 (for previous close)
        EMA needs ma_period
        """
        return max(self.atr_period + 1, self.ma_period) + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.atr_period < 1:
            raise InvalidParameterError(
                self.name, 'atr_period', self.atr_period,
                "ATR period must be positive"
            )
        if self.ma_period < 1:
            raise InvalidParameterError(
                self.name, 'ma_period', self.ma_period,
                "MA period must be positive"
            )
        if self.atr_multiplier <= 0:
            raise InvalidParameterError(
                self.name, 'atr_multiplier', self.atr_multiplier,
                "ATR multiplier must be positive"
            )
        return True

    def _calculate_source(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate source price (hl2 = (high + low) / 2)

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.Series: Source prices
        """
        return (data['high'] + data['low']) / 2

    def _calculate_pmax_logic(
        self,
        mavg: pd.Series,
        atr: pd.Series
    ) -> tuple:
        """
        Core PMax calculation logic

        Args:
            mavg: EMA values
            atr: ATR values

        Returns:
            tuple: (pmax, long_stop, short_stop, trend_direction)
        """
        # Initialize arrays
        long_stop = mavg - (self.atr_multiplier * atr)
        short_stop = mavg + (self.atr_multiplier * atr)

        # Initialize result arrays
        long_stop_values = np.zeros(len(mavg))
        short_stop_values = np.zeros(len(mavg))
        trend = np.zeros(len(mavg), dtype=int)
        pmax = np.zeros(len(mavg))

        # First bar initialization
        long_stop_values[0] = long_stop.iloc[0] if not pd.isna(long_stop.iloc[0]) else 0
        short_stop_values[0] = short_stop.iloc[0] if not pd.isna(short_stop.iloc[0]) else 0
        trend[0] = 1  # Start with uptrend
        pmax[0] = long_stop_values[0]

        # Iterate through bars (Pine Script logic)
        for i in range(1, len(mavg)):
            # Skip if data is NaN
            if pd.isna(mavg.iloc[i]) or pd.isna(atr.iloc[i]):
                long_stop_values[i] = long_stop_values[i-1]
                short_stop_values[i] = short_stop_values[i-1]
                trend[i] = trend[i-1]
                pmax[i] = pmax[i-1]
                continue

            # Calculate current stops
            current_long = long_stop.iloc[i]
            current_short = short_stop.iloc[i]

            # Long stop logic: if MAvg > prev longStop, use max(current, prev)
            if mavg.iloc[i] > long_stop_values[i-1]:
                long_stop_values[i] = max(current_long, long_stop_values[i-1])
            else:
                long_stop_values[i] = current_long

            # Short stop logic: if MAvg < prev shortStop, use min(current, prev)
            if mavg.iloc[i] < short_stop_values[i-1]:
                short_stop_values[i] = min(current_short, short_stop_values[i-1])
            else:
                short_stop_values[i] = current_short

            # Trend direction logic
            prev_trend = trend[i-1]

            if prev_trend == -1 and mavg.iloc[i] > short_stop_values[i-1]:
                # Switch to uptrend
                trend[i] = 1
            elif prev_trend == 1 and mavg.iloc[i] < long_stop_values[i-1]:
                # Switch to downtrend
                trend[i] = -1
            else:
                # Maintain previous trend
                trend[i] = prev_trend

            # PMax value based on trend
            pmax[i] = long_stop_values[i] if trend[i] == 1 else short_stop_values[i]

        return (
            pd.Series(pmax, index=mavg.index, name='pmax'),
            pd.Series(long_stop_values, index=mavg.index, name='long_stop'),
            pd.Series(short_stop_values, index=mavg.index, name='short_stop'),
            pd.Series(trend, index=mavg.index, name='trend_direction')
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        PMax calculation - REALTIME

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: PMax values and signal
        """
        # Calculate source (hl2)
        source = self._calculate_source(data)

        # Create temporary DataFrame with source as close
        temp_data = data.copy()
        temp_data['close'] = source

        # Calculate EMA on source
        ema_series = self._ema.calculate_batch(temp_data)

        # Calculate ATR
        atr_series = self._atr.calculate_batch(data)

        # Calculate PMax logic
        pmax, long_stop, short_stop, trend_direction = self._calculate_pmax_logic(
            ema_series,
            atr_series
        )

        # Last values
        current_pmax = pmax.iloc[-1] if not pd.isna(pmax.iloc[-1]) else 0.0
        current_long = long_stop.iloc[-1] if not pd.isna(long_stop.iloc[-1]) else 0.0
        current_short = short_stop.iloc[-1] if not pd.isna(short_stop.iloc[-1]) else 0.0
        current_trend = int(trend_direction.iloc[-1]) if not pd.isna(trend_direction.iloc[-1]) else 0
        current_mavg = ema_series.iloc[-1] if not pd.isna(ema_series.iloc[-1]) else 0.0

        # Previous values for signal detection
        prev_trend = int(trend_direction.iloc[-2]) if len(trend_direction) > 1 and not pd.isna(trend_direction.iloc[-2]) else current_trend

        # Determine signal and trend
        signal = self._get_signal(current_trend, prev_trend)
        trend = TrendDirection.UP if current_trend == 1 else TrendDirection.DOWN if current_trend == -1 else TrendDirection.NEUTRAL

        # Calculate strength
        current_price = data['close'].iloc[-1]
        strength = self._calculate_strength(current_price, current_pmax, current_trend)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'pmax': round(current_pmax, 8),
                'long_stop': round(current_long, 8),
                'short_stop': round(current_short, 8),
                'mavg': round(current_mavg, 8),
                'trend_direction': current_trend
            },
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier,
                'ma_period': self.ma_period,
                'current_price': round(current_price, 8),
                'distance_pct': round(((current_price - current_pmax) / current_pmax) * 100, 4) if current_pmax != 0 else 0
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        PMax batch calculation - BACKTEST

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: Columns ['pmax', 'long_stop', 'short_stop', 'mavg', 'trend_direction']
        """
        self._validate_data(data)

        # Calculate source (hl2)
        source = self._calculate_source(data)

        # Create temporary DataFrame with source as close
        temp_data = data.copy()
        temp_data['close'] = source

        # Calculate EMA on source
        mavg = self._ema.calculate_batch(temp_data)

        # Calculate ATR
        atr = self._atr.calculate_batch(data)

        # Calculate PMax logic
        pmax, long_stop, short_stop, trend_direction = self._calculate_pmax_logic(
            mavg,
            atr
        )

        result = pd.DataFrame({
            'pmax': pmax,
            'long_stop': long_stop,
            'short_stop': short_stop,
            'mavg': mavg,
            'trend_direction': trend_direction
        }, index=data.index)

        return result

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - for update()

        Args:
            data: OHLCV DataFrame
            symbol: Symbol (optional)
        """
        super().warmup_buffer(data, symbol)

        max_len = self.get_required_periods() + 50

        # Initialize data buffer
        self._data_buffer = deque(maxlen=max_len)

        # Store recent data as dict
        for i in range(len(data.tail(max_len))):
            row = data.tail(max_len).iloc[i]
            self._data_buffer.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Calculate batch result for buffer initialization
        batch_result = self.calculate_batch(data.tail(max_len + 10))

        # PMax buffer for tracking
        self._pmax_buffer = deque(maxlen=10)
        self._trend_buffer = deque(maxlen=10)

        for i in range(len(batch_result.tail(10))):
            row = batch_result.tail(10).iloc[i]
            if not pd.isna(row['pmax']):
                self._pmax_buffer.append(row['pmax'])
                self._trend_buffer.append(row['trend_direction'])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update - REALTIME

        Args:
            candle: New kline data (dict or tuple)
            symbol: Symbol (optional)

        Returns:
            IndicatorResult: Updated PMax
        """
        if self._data_buffer is None:
            self._data_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            candle_dict = candle
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # Convert tuple/list to dict
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            candle_dict = {
                'timestamp': timestamp_val,
                'open': candle[1] if len(candle) > 1 else 0,
                'high': candle[2] if len(candle) > 2 else 0,
                'low': candle[3] if len(candle) > 3 else 0,
                'close': candle[4] if len(candle) > 4 else 0,
                'volume': candle[5] if len(candle) > 5 else 0
            }

        self._data_buffer.append(candle_dict)

        # Not enough data
        if len(self._data_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={
                    'pmax': 0.0,
                    'long_stop': 0.0,
                    'short_stop': 0.0,
                    'mavg': 0.0,
                    'trend_direction': 0
                },
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )

        # Calculate using full buffer
        buffer_data = pd.DataFrame(list(self._data_buffer))

        return self.calculate(buffer_data)

    def _get_signal(self, current_trend: int, prev_trend: int) -> SignalType:
        """
        Signal detection based on trend change

        Args:
            current_trend: Current trend direction (1 or -1)
            prev_trend: Previous trend direction

        Returns:
            SignalType: BUY (trend switch to up), SELL (trend switch to down), HOLD
        """
        # Trend reversal detection
        if prev_trend == -1 and current_trend == 1:
            return SignalType.BUY
        elif prev_trend == 1 and current_trend == -1:
            return SignalType.SELL

        # Maintain signal based on current trend
        if current_trend == 1:
            return SignalType.BUY
        elif current_trend == -1:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_strength(self, price: float, pmax: float, trend: int) -> float:
        """
        Signal strength (0-100)

        Based on:
        - Distance from PMax
        - Trend direction confidence
        """
        if pmax == 0:
            return 0.0

        # Distance component (normalized)
        distance_pct = abs((price - pmax) / pmax * 100)
        distance_strength = min(distance_pct * 10, 50)

        # Trend confidence (strong if trend is defined)
        trend_strength = 50 if trend != 0 else 0

        return min(distance_strength + trend_strength, 100.0)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'ma_period': 10
        }

    def _get_output_names(self) -> list:
        """Output names"""
        return ['pmax', 'long_stop', 'short_stop', 'mavg', 'trend_direction']

    def _requires_volume(self) -> bool:
        """Does not require volume"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['PMax']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """PMax indicator test"""

    print("\n" + "="*60)
    print("PMAX INDICATOR TEST")
    print("="*60 + "\n")

    # Create sample data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    n_bars = 150

    timestamps = [1697000000000 + i * 60000 for i in range(n_bars)]

    # Trend simulation with noise
    base_price = 100
    prices = [base_price]
    for i in range(n_bars - 1):
        trend = 0.5 * np.sin(i / 20)  # Sine wave trend
        noise = np.random.randn() * 0.8
        prices.append(prices[-1] + trend + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] Created {len(data)} candles")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    pmax = PMax(atr_period=10, atr_multiplier=3.0, ma_period=10)
    print(f"   [OK] Created: {pmax}")
    print(f"   [OK] Required periods: {pmax.get_required_periods()}")
    print(f"   [OK] Parameters: ATR={pmax.atr_period}, Mult={pmax.atr_multiplier}, MA={pmax.ma_period}")

    result = pmax.calculate(data)
    print(f"   [OK] PMax Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Strength: {result.strength:.2f}")

    # Test 2: Batch calculation
    print("\n3. Batch calculation test...")
    batch_result = pmax.calculate_batch(data)
    print(f"   [OK] Batch shape: {batch_result.shape}")
    print(f"   [OK] Columns: {batch_result.columns.tolist()}")
    print(f"   [OK] Last 5 PMax values:")
    for i in range(-5, 0):
        trend_str = "UP" if batch_result['trend_direction'].iloc[i] == 1 else "DOWN"
        print(f"       PMax: {batch_result['pmax'].iloc[i]:.4f} | Trend: {trend_str}")

    # Test 3: Update method
    print("\n4. Update method test...")
    pmax.warmup_buffer(data)

    new_candle = {
        'timestamp': timestamps[-1] + 60000,
        'open': prices[-1],
        'high': prices[-1] + 0.5,
        'low': prices[-1] - 0.5,
        'close': prices[-1] + 0.3,
        'volume': 1500
    }

    update_result = pmax.update(new_candle)
    print(f"   [OK] Update PMax: {update_result.value}")
    print(f"   [OK] Update Signal: {update_result.signal.value}")

    # Test 4: Trend statistics
    print("\n5. Trend statistics...")
    up_count = (batch_result['trend_direction'] == 1).sum()
    down_count = (batch_result['trend_direction'] == -1).sum()
    neutral_count = (batch_result['trend_direction'] == 0).sum()
    nan_count = batch_result['pmax'].isna().sum()

    print(f"   [OK] UP trends: {up_count}")
    print(f"   [OK] DOWN trends: {down_count}")
    print(f"   [OK] NEUTRAL: {neutral_count}")
    print(f"   [OK] NaN (warmup): {nan_count}")

    # Test 5: Different parameters
    print("\n6. Different parameter test...")
    params_list = [
        (10, 3.0, 10),
        (14, 2.5, 12),
        (20, 4.0, 15)
    ]
    for atr_p, mult, ma_p in params_list:
        pmax_test = PMax(atr_period=atr_p, atr_multiplier=mult, ma_period=ma_p)
        result = pmax_test.calculate(data)
        trend_str = "UP" if result.value['trend_direction'] == 1 else "DOWN"
        print(f"   [OK] ATR={atr_p}, Mult={mult}, MA={ma_p}: PMax={result.value['pmax']:.4f} | Trend={trend_str}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
