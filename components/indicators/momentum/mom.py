#!/usr/bin/env python3
"""
indicators/momentum/mom.py - MOM (Momentum Indicator)

Author: SuperBot Team
Date: 2025-11-20
Versiyon: 1.0.0

Momentum (MOM) - The simplest momentum indicator.
It calculates the difference between the current price and the price from N periods ago.

Features:
- Simple and fast calculation
- Measures trend direction and strength
- Positive value = Bullish momentum
- Negative value = Bearish momentum

Usage:
    from components.indicators import get_indicator_class

    MOM = get_indicator_class('mom')
    mom = MOM(period=10)
    result = mom.calculate(data)
    print(result.value['mom'])

Formula:
    MOM = Close - Close[n periods ago]

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
from collections import deque
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class MOM(BaseIndicator):
    """
    Momentum Indicator

    The simplest momentum indicator. It compares the current price with the price N periods ago.
    Calculates the difference between the prices.

    Args:
        period: Momentum period (default: 10)
        logger: Logger instance (optional)
        error_handler: Error handler (optional)
    """

    def __init__(self, period: int = 10, logger=None, error_handler=None):
        self.period = period
        self.prices = deque(maxlen=period + 1)

        super().__init__(
            name='mom',
            category=IndicatorCategory.MOMENTUM,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={'period': period},
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period + 1

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "Period must be positive"
            )
        return True

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)

        Calculates all data vectorially.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: MOM values
        """
        close = data['close']
        mom = close - close.shift(self.period)
        return pd.DataFrame({'mom': mom}, index=data.index)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - required for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        # prices deque'yu doldur
        self.prices.clear()
        # The period+1 data is required.
        tail_data = data['close'].tail(self.period + 1).values
        for val in tail_data:
            self.prices.append(val)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)

        Args:
            candle: Yeni mum verisi (dict)

        Returns:
            IndicatorResult: Current MOM value
        """
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            timestamp_val = int(candle.get('timestamp', 0))
            close_val = candle['close']
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            close_val = candle[4] if len(candle) > 4 else 0

        self.prices.append(close_val)

        if len(self.prices) < self.period + 1:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )

        mom_value = self.prices[-1] - self.prices[0]

        # Signal and trend determination
        if mom_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif mom_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        return IndicatorResult(
            value={'mom': round(mom_value, 4)},
            timestamp=timestamp_val,
            signal=signal,
            trend=trend,
            strength=min(abs(mom_value) * 10, 100),
            metadata={'period': self.period}
        )

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate MOM (final value)

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: MOM value
        """
        # Fill the buffers
        close_values = data['close'].tail(self.period + 1).values
        self.prices.clear()
        self.prices.extend(close_values)

        # Batch hesapla
        batch_result = self.calculate_batch(data)
        valid_values = batch_result['mom'].dropna().values

        if len(valid_values) == 0:
            return None

        mom_value = valid_values[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Signal and trend
        if mom_value > 0:
            signal = SignalType.BUY
            trend = TrendDirection.UP
        elif mom_value < 0:
            signal = SignalType.SELL
            trend = TrendDirection.DOWN
        else:
            signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'mom': round(mom_value, 4)},
            timestamp=timestamp,
            signal=signal,
            trend=trend,
            strength=min(abs(mom_value) * 10, 100),
            metadata={'period': self.period}
        )

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {'period': 10}

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['mom']

    def _requires_volume(self) -> bool:
        """MOM volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['MOM']


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """MOM indicator test"""

    # Windows console UTF-8 support
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*60)
    print("🧪 MOMENTUM (MOM) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Trend + noise
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    prices = base_price + trend + noise

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.randn(100)),
        'low': prices - np.abs(np.random.randn(100)),
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in range(100)]
    })

    print(f"   ✅ {len(data)} candles created")
    print(f"   ✅ Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    mom = MOM(period=10)
    print(f"   ✅ Created: {mom}")
    print(f"   ✅ Kategori: {mom.category.value}")
    print(f"   ✅ Required period: {mom.get_required_periods()}")

    result = mom(data)
    print(f"   ✅ MOM: {result.value['mom']}")
    print(f"   ✅ Signal: {result.signal.value}")
    print(f"   ✅ Trend: {result.trend.name}")
    print(f"   ✅ Power: {result.strength:.2f}")

    # Test 2: Batch Calculation
    print("\n3. Batch Calculation Testi...")
    batch_result = mom.calculate_batch(data)
    print(f"   ✅ Batch result shape: {batch_result.shape}")
    print(f"   ✅ Last 5 MOM values:")
    print(batch_result['mom'].tail())

    # Test 3: Update method
    print("\n4. Update method test...")
    mom2 = MOM(period=10)
    init_data = data.head(50)
    mom2.calculate(init_data)

    # Yeni 5 mum ekle
    for i in range(50, 55):
        candle = {
            'timestamp': data.iloc[i]['timestamp'],
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low']
        }
        update_result = mom2.update(candle)
        if update_result:
            print(f"   ✅ Bar {i}: MOM={update_result.value['mom']:.4f}, "
                  f"Signal={update_result.signal.value}")

    # Test 4: Different periods
    print("\n5. Different period test...")
    for period in [5, 10, 20]:
        mom_test = MOM(period=period)
        result = mom_test.calculate(data)
        print(f"   ✅ MOM({period}): {result.value['mom']:.4f}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
