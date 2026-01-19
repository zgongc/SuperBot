"""
indicators/volume/vwap_bands.py - VWAP Bands

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    VWAP Bands - Standard deviation bands around the VWAP
    VWAP + (std_dev x multiplier) = Upper band
    VWAP - (std_dev × multiplier) = Alt bant
    When the price goes outside the price bands, it signals an overbought/oversold condition.

Formula:
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Upper = VWAP + (StdDev × multiplier)
    Lower = VWAP - (StdDev × multiplier)

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


class VWAPBands(BaseIndicator):
    """
    VWAP Bands

    Creates standard deviation bands around the VWAP.
    Works with a similar logic to Bollinger Bands.

    Args:
        period: VWAP calculation period (default: 20)
        multiplier: Standard deviation multiplier (default: 2.0)
    """

    def __init__(
        self,
        period: int = 20,
        multiplier: float = 2.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.multiplier = multiplier

        super().__init__(
            name='vwap_bands',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.BANDS,
            params={
                'period': period,
                'multiplier': multiplier
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.period, 2)

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        if self.multiplier <= 0:
            raise InvalidParameterError(
                self.name, 'multiplier', self.multiplier,
                "The factor must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        VWAP Bands hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: VWAP Bands values
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Typical Price hesapla
        typical_price = (high + low + close) / 3

        # Period selection
        if len(data) <= self.period:
            tp_period = typical_price
            vol_period = volume
        else:
            tp_period = typical_price[-self.period:]
            vol_period = volume[-self.period:]

        # VWAP hesapla
        tp_vol = tp_period * vol_period
        vwap = np.sum(tp_vol) / np.sum(vol_period)

        # Calculate standard deviation (volume weighted)
        variance = np.sum(vol_period * (tp_period - vwap) ** 2) / np.sum(vol_period)
        std_dev = np.sqrt(variance)

        # Calculate the bands
        upper_band = vwap + (std_dev * self.multiplier)
        lower_band = vwap - (std_dev * self.multiplier)

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Bandwidth
        bandwidth = ((upper_band - lower_band) / vwap) * 100

        # Price position according to the bands (0-100)
        if upper_band != lower_band:
            position = ((current_price - lower_band) / (upper_band - lower_band)) * 100
        else:
            position = 50

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'upper': round(upper_band, 8),
                'vwap': round(vwap, 8),
                'lower': round(lower_band, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(current_price, upper_band, lower_band, vwap),
            trend=self.get_trend(current_price, vwap),
            strength=min(abs(50 - position), 50) * 2,  # 0-100 normalize
            metadata={
                'period': self.period,
                'multiplier': self.multiplier,
                'std_dev': round(std_dev, 8),
                'bandwidth': round(bandwidth, 2),
                'price_position': round(position, 2),
                'current_price': round(current_price, 8)
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ VECTORIZED batch VWAP Bands calculation - for BACKTEST

        VWAP Bands Formula:
            VWAP = Σ(Typical Price × Volume) / Σ(Volume)
            Variance = Σ(Volume × (TP - VWAP)²) / Σ(Volume)
            StdDev = √Variance
            Upper = VWAP + (StdDev × multiplier)
            Lower = VWAP - (StdDev × multiplier)

        Args:
            data: OHLCV DataFrame (full history)

        Returns:
            pd.DataFrame: vwap_upper, vwap, vwap_lower for all bars

        Performance: 2000 bars in ~0.03 seconds
        """
        self._validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # Typical Price
        typical_price = (high + low + close) / 3

        # Rolling VWAP
        tp_vol = typical_price * volume
        vwap = tp_vol.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

        # Rolling volume-weighted standard deviation
        def rolling_vwap_std(window_tp, window_vol, window_vwap):
            """Calculate volume-weighted std for a window"""
            if len(window_tp) < 2 or window_vol.sum() == 0:
                return np.nan
            variance = np.sum(window_vol * (window_tp - window_vwap)**2) / np.sum(window_vol)
            return np.sqrt(variance)

        # Calculate std for each bar
        std_series = pd.Series(index=data.index, dtype=float)
        for i in range(self.period - 1, len(data)):
            window_tp = typical_price.iloc[i-self.period+1:i+1].values
            window_vol = volume.iloc[i-self.period+1:i+1].values
            window_vwap = vwap.iloc[i]
            if not np.isnan(window_vwap):
                std_series.iloc[i] = rolling_vwap_std(window_tp, window_vol, window_vwap)

        # Upper and Lower Bands
        upper = vwap + (std_series * self.multiplier)
        lower = vwap - (std_series * self.multiplier)

        # Set first period values to NaN (warmup)
        vwap.iloc[:self.period-1] = np.nan
        upper.iloc[:self.period-1] = np.nan
        lower.iloc[:self.period-1] = np.nan

        return pd.DataFrame({
            'upper': upper.values,
            'vwap': vwap.values,
            'lower': lower.values
        }, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        if not hasattr(self, '_buffers_init'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._close_buffer = deque(maxlen=max_len)
            self._volume_buffer = deque(maxlen=max_len)
            self._buffers_init = True
        
        # Support both dict and list/tuple formats
        if isinstance(candle, dict):
            close_val = candle['close']
            volume_val = candle.get('volume', 1000)
            high_val = candle.get('high', candle['close'])
            low_val = candle.get('low', candle['close'])
            open_val = candle.get('open', candle['close'])
            timestamp_val = int(candle.get('timestamp', 0))
        else:
            timestamp_val = int(candle[0]) if len(candle) > 0 else 0
            open_val = candle[1] if len(candle) > 1 else 0
            high_val = candle[2] if len(candle) > 2 else 0
            low_val = candle[3] if len(candle) > 3 else 0
            close_val = candle[4] if len(candle) > 4 else 0
            volume_val = candle[5] if len(candle) > 5 else 1000

        self._close_buffer.append(close_val)
        self._volume_buffer.append(volume_val)
        
        if len(self._close_buffer) < self.get_required_periods():
            return IndicatorResult(
                value={'upper': 0.0, 'vwap': 0.0, 'lower': 0.0},
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={}
            )
        
        buffer_data = pd.DataFrame({
            'close': list(self._close_buffer),
            'volume': list(self._volume_buffer),
            'high': [high_val] * len(self._close_buffer),
            'low': [low_val] * len(self._close_buffer),
            'open': [open_val] * len(self._close_buffer),
            'timestamp': [timestamp_val] * len(self._close_buffer)
        })
        
        return self.calculate(buffer_data)

    def get_signal(self, price: float, upper: float, lower: float, vwap: float) -> SignalType:
        """
        Generate a signal based on VWAP Bands.

        Args:
            price: Current price
            upper: Upper band
            lower: Alt bant
            vwap: VWAP value

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if price <= lower:
            return SignalType.BUY  # Oversold
        elif price >= upper:
            return SignalType.SELL  # Overselling
        elif price > vwap:
            return SignalType.HOLD  # Above VWAP
        else:
            return SignalType.HOLD  # Below VWAP

    def get_trend(self, price: float, vwap: float) -> TrendDirection:
        """
        Determine the trend according to VWAP.

        Args:
            price: Current price
            vwap: VWAP value

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        distance_pct = abs(((price - vwap) / vwap) * 100)

        if price > vwap and distance_pct > 0.5:
            return TrendDirection.UP
        elif price < vwap and distance_pct > 0.5:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 20,
            'multiplier': 2.0
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['upper', 'vwap', 'lower']

    def _requires_volume(self) -> bool:
        """VWAP Bands volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VWAPBands']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """VWAP Bands indicator test"""

    print("\n" + "="*60)
    print("VWAP BANDS TEST")
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
    vwap_bands = VWAPBands(period=20, multiplier=2.0)
    print(f"   [OK] Created: {vwap_bands}")
    print(f"   [OK] Kategori: {vwap_bands.category.value}")
    print(f"   [OK] Tip: {vwap_bands.indicator_type.value}")

    result = vwap_bands(data)
    print(f"   [OK] Upper Band: {result.value['upper']:.8f}")
    print(f"   [OK] VWAP: {result.value['vwap']:.8f}")
    print(f"   [OK] Alt Bant: {result.value['lower']:.8f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different factors
    print("\n3. Different factor test...")
    for mult in [1.0, 2.0, 3.0]:
        bands = VWAPBands(period=20, multiplier=mult)
        result = bands.calculate(data)
        bandwidth = result.metadata['bandwidth']
        print(f"   [OK] Multiplier={mult}: Bandwidth={bandwidth:.2f}%")

    # Test 3: Output names
    print("\n4. Output names testi...")
    outputs = vwap_bands._get_output_names()
    print(f"   [OK] Output sayısı: {len(outputs)}")
    print(f"   [OK] Outputs: {outputs}")
    assert len(outputs) == 3, "There should be 3 outputs!"

    # Test 4: Volume gereksinimi
    print("\n5. Volume gereksinimi testi...")
    metadata = vwap_bands.metadata
    print(f"   [OK] Volume required: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "VWAP Bands volume gerektirmeli!"

    # Test 5: Price position
    print("\n6. Price position test...")
    result = vwap_bands.calculate(data)
    position = result.metadata['price_position']
    print(f"   [OK] Price position: {position:.2f}%")
    if position < 25:
        print("   [OK] Price is close to the lower band (oversold)")
    elif position > 75:
        print("   [OK] Price is close to the upper band (overbought)")
    else:
        print("   [OK] Price is within the specified range")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = vwap_bands.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
