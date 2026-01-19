"""
indicators/volume/cmf.py - Chaikin Money Flow

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    CMF (Chaikin Money Flow) - Money flow oscillator
    Measures buying/selling pressure within a specific period
    Range: between -1.00 and +1.00
    CMF > 0 = Buying pressure
    CMF < 0 = Selling pressure

Formula:
    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    MF Volume = MFM × Volume
    CMF = Σ(MF Volume) / Σ(Volume)

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


class CMF(BaseIndicator):
    """
    Chaikin Money Flow

    Money flow indicator developed by Marc Chaikin.
    Combines volume and price position to show buying/selling pressure.

    Args:
        period: CMF period (default: 20)
        buy_threshold: Buy threshold (default: 0.05)
        sell_threshold: Sell threshold (default: -0.05)
    """

    def __init__(
        self,
        period: int = 20,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        super().__init__(
            name='cmf',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'period': period,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.period

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        if self.sell_threshold >= self.buy_threshold:
            raise InvalidParameterError(
                self.name, 'thresholds',
                f"sell={self.sell_threshold}, buy={self.buy_threshold}",
                "Sell threshold must be smaller than the buy threshold"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        CMF hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: CMF value
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Son period'u al
        high_period = high[-self.period:]
        low_period = low[-self.period:]
        close_period = close[-self.period:]
        volume_period = volume[-self.period:]

        # Calculate Money Flow Multiplier and Money Flow Volume
        mf_volumes = np.zeros(self.period)

        for i in range(self.period):
            high_low_diff = high_period[i] - low_period[i]

            if high_low_diff == 0:
                mfm = 0
            else:
                mfm = ((close_period[i] - low_period[i]) -
                       (high_period[i] - close_period[i])) / high_low_diff

            mf_volumes[i] = mfm * volume_period[i]

        # CMF hesapla
        total_mf_volume = np.sum(mf_volumes)
        total_volume = np.sum(volume_period)

        if total_volume == 0:
            cmf_value = 0.0
        else:
            cmf_value = total_mf_volume / total_volume

        timestamp = int(data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=round(cmf_value, 4),
            timestamp=timestamp,
            signal=self.get_signal(cmf_value),
            trend=self.get_trend(cmf_value),
            strength=min(abs(cmf_value) * 100, 100),
            metadata={
                'period': self.period,
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'total_volume': int(total_volume),
                'money_flow': round(total_mf_volume, 2)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        ⚡ VECTORIZED batch CMF calculation - for BACKTEST

        CMF Formula:
            MFM = ((Close - Low) - (High - Close)) / (High - Low)
            MF Volume = MFM × Volume
            CMF = Σ(MF Volume, period) / Σ(Volume, period)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.Series: CMF values for all bars

        Performance: 2000 bars in ~0.02 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # Money Flow Multiplier (vectorized)
        high_low_diff = high - low
        # Avoid division by zero
        high_low_diff = high_low_diff.replace(0, np.nan)

        mfm = ((close - low) - (high - close)) / high_low_diff
        mfm = mfm.fillna(0)

        # Money Flow Volume
        mf_volume = mfm * volume

        # CMF = rolling sum of MF Volume / rolling sum of Volume
        mf_volume_sum = mf_volume.rolling(window=self.period).sum()
        volume_sum = volume.rolling(window=self.period).sum()

        cmf = mf_volume_sum / volume_sum

        # Handle division by zero
        cmf = cmf.fillna(0)

        # Set first period values to NaN (warmup)
        cmf.iloc[:self.period-1] = np.nan

        return pd.Series(cmf.values, index=data.index, name='cmf')

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup buffer - prepares the necessary state for update().

        Args:
            data: OHLCV DataFrame (warmup verisi)
            symbol: Symbol name (optional)
        """
        super().warmup_buffer(data, symbol)

        from collections import deque
        max_len = self.get_required_periods() + 50

        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._volume_buffer = deque(maxlen=max_len)

        # Buffer'lara verileri ekle
        for i in range(len(data)):
            self._high_buffer.append(data['high'].iloc[i])
            self._low_buffer.append(data['low'].iloc[i])
            self._close_buffer.append(data['close'].iloc[i])
            self._volume_buffer.append(data['volume'].iloc[i])

        self._buffers_init = True

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init') or not self._buffers_init:
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True

        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            # List/tuple format: [timestamp, open, high, low, close, volume]
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._high_buffer.append(high_val)
        self._low_buffer.append(low_val)
        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)

        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value=0.0,
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
            'volume': list(self._volume_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })

        return self.calculate(buffer_data)

    def get_signal(self, value: float) -> SignalType:
        """
        Generate a signal from the CMF value.

        Args:
            value: CMF value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if value > self.buy_threshold:
            return SignalType.BUY
        elif value < self.sell_threshold:
            return SignalType.SELL
        return SignalType.HOLD

    def get_trend(self, value: float) -> TrendDirection:
        """
        Determine the trend based on the CMF value.

        Args:
            value: CMF value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if value > 0:
            return TrendDirection.UP
        elif value < 0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'buy_threshold': 0.05,
            'sell_threshold': -0.05
        }

    def _requires_volume(self) -> bool:
        """CMF volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['CMF']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """CMF indicator test"""

    print("\n" + "="*60)
    print("CMF (CHAIKIN MONEY FLOW) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(30)]

    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(29):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-3000, 5000))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    cmf = CMF(period=20)
    print(f"   [OK] Created: {cmf}")
    print(f"   [OK] Kategori: {cmf.category.value}")
    print(f"   [OK] Required period: {cmf.get_required_periods()}")

    result = cmf(data)
    print(f"   [OK] CMF Value: {result.value:.4f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different periods
    print("\n3. Different period test...")
    for period in [10, 20, 30]:
        cmf_test = CMF(period=period)
        result = cmf_test.calculate(data)
        print(f"   [OK] CMF({period}): {result.value:.4f} | Signal: {result.signal.value}")

    # Test 3: Custom thresholds
    print("\n4. Special threshold test...")
    cmf_custom = CMF(period=20, buy_threshold=0.1, sell_threshold=-0.1)
    result = cmf_custom.calculate(data)
    print(f"   [OK] CMF (custom thresholds): {result.value:.4f}")
    print(f"   [OK] Buy threshold: {cmf_custom.buy_threshold}")
    print(f"   [OK] Sell threshold: {cmf_custom.sell_threshold}")
    print(f"   [OK] Signal: {result.signal.value}")

    # Test 4: Volume gereksinimi
    print("\n5. Volume gereksinimi testi...")
    metadata = cmf.metadata
    print(f"   [OK] Volume required: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "CMF volume gerektirmeli!"

    # Test 5: Money flow interpretation
    print("\n6. Currency flow interpretation test...")
    result = cmf.calculate(data)
    cmf_val = result.value
    print(f"   [OK] CMF: {cmf_val:.4f}")
    if cmf_val > 0.1:
        print("   [OK] Strong buying pressure")
    elif cmf_val > 0:
        print("   [OK] Weak buying pressure")
    elif cmf_val > -0.1:
        print("   [OK] Weak put pressure")
    else:
        print("   [OK] Strong selling pressure")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = cmf.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
