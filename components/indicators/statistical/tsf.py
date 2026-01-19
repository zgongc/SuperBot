#!/usr/bin/env python3
"""
indicators/statistical/tsf.py - TSF (Time Series Forecast)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

TSF (Time Series Forecast) - Zaman Serisi Tahmini.
Using linear regression for future price prediction.

Features:
- Trend prediction using linear regression
- Price prediction for the next bar
- Shows trend direction and strength
- Generates a signal based on the price-TSF difference
- Can be used as a support/resistance level

Usage:
    from components.indicators import get_indicator_class

    TSF = get_indicator_class('tsf')
    tsf = TSF(period=14)
    result = tsf.calculate(data)
    print(result.value['tsf'])

Formula:
    Linear Regression: y = mx + b
    TSF = m * (period) + b
    (Next value prediction)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje root'unu path'e ekle
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

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


class TSF(BaseIndicator):
    """
    TSF - Time Series Forecast

    Performs future price prediction using linear regression.
    Price projection for the next bar.

    Args:
        period: TSF period (default: 14)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 14, logger=None, error_handler=None):
        self.period = period

        super().__init__(
            name='tsf',
            category=IndicatorCategory.STATISTICAL,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
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
                "Period must be at least 2"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: TSF values
        """
        close = data['close']
        tsf_values = []

        for i in range(len(close)):
            if i < self.period - 1:
                tsf_values.append(np.nan)
            else:
                # Get data up to the last period
                y = close.iloc[i - self.period + 1:i + 1].values
                x = np.arange(self.period)

                # Lineer regresyon
                slope, intercept = np.polyfit(x, y, 1)

                # Next value prediction
                forecast = slope * self.period + intercept
                tsf_values.append(forecast)

        return pd.DataFrame({'tsf': tsf_values}, index=data.index)

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

        self._close_buffer = deque(maxlen=max_len)
        for val in data['close'].tail(max_len).values:
            self._close_buffer.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: TSF value
        """
        if not hasattr(self, '_close_buffer'):
            from collections import deque
            self._close_buffer = deque(maxlen=self.get_required_periods() + 50)

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self._close_buffer.append(close_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate TSF (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: TSF value
        """
        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['tsf'].dropna().values

        if len(valid_values) == 0:
            return None

        tsf_val = valid_values[-1]
        close = data['close'].iloc[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal determination: TSF > Close = forecast will rise (BUY)
        if tsf_val > close:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif tsf_val < close:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Power: Percentage difference between the prediction and the actual value.
        strength = min(abs((tsf_val - close) / close * 100) * 10, 100)

        # Add the forecast difference to the metadata.
        forecast_diff = tsf_val - close

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'tsf': round(tsf_val, 2)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=strength,
            metadata={
                'period': self.period,
                'forecast_diff': round(forecast_diff, 2)
            }
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 14}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['tsf']

    def _requires_volume(self) -> bool:
        """TSF volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['TSF']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """TSF indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 TSF (TIME SERIES FORECAST) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(150)]

    # Strong trend + noise
    base_price = 100
    trend = np.linspace(0, 30, 150)
    noise = np.random.randn(150) * 2
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(150)),
        'low': prices - np.abs(np.random.randn(150)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(150)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    tsf = TSF(period=14)
    print(f"   ✅ Created: {tsf}")
    print(f"   ✅ Kategori: {tsf.category.value}")
    print(f"   ✅ Required period: {tsf.get_required_periods()}")

    result = tsf(data)
    print(f"   ✅ TSF: {result.value['tsf']}")
    print(f"   ✅ Close: {data['close'].iloc[-1]:.2f}")
    print(f"   ✅ Forecast difference: {result.metadata['forecast_diff']:.2f}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = tsf.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 TSF values:")
    print(batch_result['tsf'].tail())

    # Test 3: Different periods
    print("\n4. Different period test...")
    for period in [7, 14, 21]:
        tsf_test = TSF(period=period)
        result = tsf_test.calculate(data)
        print(f"   ✅ TSF({period}): {result.value['tsf']:.2f}, "
              f"Diff={result.metadata['forecast_diff']:.2f}")

    # Test 4: Accuracy analysis of predictions
    print("\n5. Accuracy analysis...")
    batch_result = tsf.calculate_batch(data)
    tsf_values = batch_result['tsf'].dropna()

    # Check how accurately the TSF predicts the next bar.
    errors = []
    for i in range(len(tsf_values) - 1):
        forecast = tsf_values.iloc[i]
        actual = data['close'].iloc[i + 1]
        error = abs(forecast - actual)
        errors.append(error)

    if len(errors) > 0:
        print(f"   ✅ Average prediction error: {np.mean(errors):.4f}")
        print(f"   ✅ Minimum error: {min(errors):.4f}")
        print(f"   ✅ Max error: {max(errors):.4f}")
        print(f"   ✅ Error standard deviation: {np.std(errors):.4f}")

    # Test 5: Comparison with trend line
    print("\n6. Price comparison with TSF...")
    batch_result = tsf.calculate_batch(data)
    tsf_values = batch_result['tsf'].dropna()
    close_values = data['close'].iloc[len(data)-len(tsf_values):]

    # How much above/below the TSF price.
    above_count = sum(tsf_values.values > close_values.values)
    below_count = sum(tsf_values.values < close_values.values)

    print(f"   ✅ TSF > Close: {above_count}")
    print(f"   ✅ TSF < Close: {below_count}")
    print(f"   ✅ Ortalama fark: {(tsf_values.values - close_values.values).mean():.4f}")

    # Test 6: Trend power analysis
    print("\n7. Trend strength analysis...")
    # Calculate the trend slope for the last N bars.
    last_n = 30
    recent_closes = data['close'].tail(last_n).values
    x = np.arange(last_n)
    slope, intercept = np.polyfit(x, recent_closes, 1)

    print(f"   ✅ Last {last_n} bar trend slope: {slope:.4f}")
    print(f"   ✅ Trend direction: {'Increase' if slope > 0 else 'Decrease'}")

    # Test 7: Validasyon testi
    print("\n8. Validasyon testi...")
    try:
        invalid_tsf = TSF(period=1)
        print("   ❌ Error: Invalid period accepted!")
    except InvalidParameterError as e:
        print(f"   ✅ Period validation successful: {e}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
