"""
indicators/volume/volume_profile.py - Volume Profile

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Volume Profile - Volume profile
    Shows the volume distribution at price levels
    POC (Point of Control) - The price level with the highest volume
    VAH (Value Area High) - Upper limit of the value area (70% volume)
    VAL (Value Area Low) - Lower limit of the value area (70% volume)

Formula:
    1. Divide the price range into bins.
    2. Her bin'deki hacmi topla
    3. POC = Maximum volume bin
    4. VAH/VAL = Area limits containing 70% volume

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


class VolumeProfile(BaseIndicator):
    """
    Volume Profile

    Analyzes the volume distribution at different price levels.
    Calculates POC, VAH, and VAL values.

    Args:
        period: Analysis period (default: 50)
        bins: Number of price levels (default: 24)
        value_area: Value area percentage (default: 70)
    """

    def __init__(
        self,
        period: int = 50,
        bins: int = 24,
        value_area: float = 70.0,
        logger=None,
        error_handler=None
    ):
        self.period = period
        self.bins = bins
        self.value_area = value_area

        super().__init__(
            name='volume_profile',
            category=IndicatorCategory.VOLUME,
            indicator_type=IndicatorType.ZONES,
            params={
                'period': period,
                'bins': bins,
                'value_area': value_area
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.period, self.bins)

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.period < 1:
            raise InvalidParameterError(
                self.name, 'period', self.period,
                "The period must be positive"
            )
        if self.bins < 5:
            raise InvalidParameterError(
                self.name, 'bins', self.bins,
                "Bins must be at least 5"
            )
        if not (50 <= self.value_area <= 90):
            raise InvalidParameterError(
                self.name, 'value_area', self.value_area,
                "Value must be between 50% and 90%"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Volume Profile hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: POC, VAH, VAL values
        """
        # Period selection
        if len(data) > self.period:
            period_data = data.tail(self.period)
        else:
            period_data = data

        high = period_data['high'].values
        low = period_data['low'].values
        close = period_data['close'].values
        volume = period_data['volume'].values

        # Define the price range
        price_min = np.min(low)
        price_max = np.max(high)
        price_range = price_max - price_min

        if price_range == 0:
            price_range = price_min * 0.01  # %1 minimum range

        # Create bins
        bin_edges = np.linspace(price_min, price_max, self.bins + 1)
        bin_volumes = np.zeros(self.bins)

        # Distribute the volume of each candle to the corresponding bins.
        for i in range(len(period_data)):
            candle_low = low[i]
            candle_high = high[i]
            candle_volume = volume[i]

            # Find which bins this candle covers.
            for j in range(self.bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]

                # Overlap hesapla
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)

                if overlap_high > overlap_low:
                    # Overlap ratio
                    candle_range = candle_high - candle_low
                    if candle_range > 0:
                        overlap_ratio = (overlap_high - overlap_low) / candle_range
                        bin_volumes[j] += candle_volume * overlap_ratio

        # POC (Point of Control) - The price level with the highest volume.
        poc_index = np.argmax(bin_volumes)
        poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2

        # Calculate Value Area (expand around POC)
        total_volume = np.sum(bin_volumes)
        value_area_volume = total_volume * (self.value_area / 100)

        # Starting from the POC, expand in both directions.
        va_low_index = poc_index
        va_high_index = poc_index
        current_volume = bin_volumes[poc_index]

        while current_volume < value_area_volume:
            # Decide which direction to expand to.
            can_expand_low = va_low_index > 0
            can_expand_high = va_high_index < self.bins - 1

            if not can_expand_low and not can_expand_high:
                break

            vol_below = bin_volumes[va_low_index - 1] if can_expand_low else 0
            vol_above = bin_volumes[va_high_index + 1] if can_expand_high else 0

            if vol_below > vol_above and can_expand_low:
                va_low_index -= 1
                current_volume += bin_volumes[va_low_index]
            elif can_expand_high:
                va_high_index += 1
                current_volume += bin_volumes[va_high_index]
            elif can_expand_low:
                va_low_index -= 1
                current_volume += bin_volumes[va_low_index]
            else:
                break

        # VAH and VAL
        vah_price = bin_edges[va_high_index + 1]
        val_price = bin_edges[va_low_index]

        current_price = close[-1]
        timestamp = int(period_data.iloc[-1]['timestamp'])

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={
                'poc': round(poc_price, 8),
                'vah': round(vah_price, 8),
                'val': round(val_price, 8)
            },
            timestamp=timestamp,
            signal=self.get_signal(current_price, poc_price, vah_price, val_price),
            trend=self.get_trend(current_price, poc_price),
            strength=self._calculate_strength(current_price, poc_price, vah_price, val_price),
            metadata={
                'period': self.period,
                'bins': self.bins,
                'value_area_pct': self.value_area,
                'total_volume': int(total_volume),
                'poc_volume': int(bin_volumes[poc_index]),
                'current_price': round(current_price, 8),
                'price_range': round(price_range, 8)
            }
        )

    def _calculate_strength(self, price: float, poc: float, vah: float, val: float) -> float:
        """
        Calculate the strength of the price based on the value area.

        Args:
            price: Current price
            poc: POC price
            vah: VAH price
            val: VAL price

        Returns:
            float: Power value (0-100)
        """
        if val <= price <= vah:
            # Inside the value range - low power
            distance_from_poc = abs(price - poc)
            va_range = vah - val
            if va_range > 0:
                return (distance_from_poc / va_range) * 50  # 0-50
            return 0
        else:
            # Outside the value range - high power
            if price > vah:
                distance = price - vah
                reference = vah - poc
            else:  # price < val
                distance = val - price
                reference = poc - val

            if reference > 0:
                strength = 50 + min((distance / reference) * 50, 50)
                return strength
            return 50

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
                value=[],
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

    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation - returns DataFrame with multiple columns

        Note: This is a simple implementation for compatibility.
        For performance, consider implementing vectorized logic.

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Indicator values with columns: poc, vah, val
        """
        results = {
            'poc': [],
            'vah': [],
            'val': []
        }

        for i in range(len(data)):
            if i < self.get_required_periods() - 1:
                for key in results:
                    results[key].append(np.nan)
            else:
                window_data = data.iloc[:i+1]
                result = self.calculate(window_data)

                # Extract dict values
                if result and hasattr(result, 'value') and isinstance(result.value, dict):
                    results['poc'].append(result.value.get('poc', np.nan))
                    results['vah'].append(result.value.get('vah', np.nan))
                    results['val'].append(result.value.get('val', np.nan))
                else:
                    for key in results:
                        results[key].append(np.nan)

        return pd.DataFrame(results, index=data.index)

    def get_signal(self, price: float, poc: float, vah: float, val: float) -> SignalType:
        """
        Generate a signal based on the Volume Profile.

        Args:
            price: Current price
            poc: POC price
            vah: VAH price
            val: VAL price

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if price < val:
            return SignalType.BUY  # Below value area - potential support
        elif price > vah:
            return SignalType.SELL  # Above the value area - potential resistance
        elif price < poc:
            return SignalType.HOLD  # Below the price of change (POC)
        else:
            return SignalType.HOLD  # Above the POC

    def get_trend(self, price: float, poc: float) -> TrendDirection:
        """
        Determine the trend according to the POC.

        Args:
            price: Current price
            poc: POC price

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        distance_pct = abs(((price - poc) / poc) * 100)

        if price > poc and distance_pct > 1.0:
            return TrendDirection.UP
        elif price < poc and distance_pct > 1.0:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'period': 50,
            'bins': 24,
            'value_area': 70.0
        }

    def _get_output_names(self) -> list:
        """Output isimleri"""
        return ['poc', 'vah', 'val']

    def _requires_volume(self) -> bool:
        """Volume Profile volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['VolumeProfile']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Volume Profile indicator test"""

    print("\n" + "="*60)
    print("VOLUME PROFILE TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    base_price = 100
    prices = [base_price]
    volumes = [10000]

    for i in range(59):
        change = np.random.randn() * 2
        prices.append(prices[-1] + change)
        volumes.append(10000 + np.random.randint(-3000, 8000))

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.8 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.8 for p in prices],
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    vp = VolumeProfile(period=50, bins=24)
    print(f"   [OK] Created: {vp}")
    print(f"   [OK] Kategori: {vp.category.value}")
    print(f"   [OK] Tip: {vp.indicator_type.value}")

    result = vp(data)
    print(f"   [OK] POC (Point of Control): {result.value['poc']:.8f}")
    print(f"   [OK] VAH (Value Area High): {result.value['vah']:.8f}")
    print(f"   [OK] VAL (Value Area Low): {result.value['val']:.8f}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")

    # Test 2: Different binary numbers
    print("\n3. Different bin number test...")
    for bins in [12, 24, 48]:
        vp_test = VolumeProfile(period=50, bins=bins)
        result = vp_test.calculate(data)
        va_range = result.value['vah'] - result.value['val']
        print(f"   [OK] Bins={bins}: VA Range={va_range:.8f}")

    # Test 3: Output names
    print("\n4. Output names testi...")
    outputs = vp._get_output_names()
    print(f"   [OK] Output sayısı: {len(outputs)}")
    print(f"   [OK] Outputs: {outputs}")
    assert len(outputs) == 3, "There should be 3 outputs!"

    # Test 4: Volume gereksinimi
    print("\n5. Volume gereksinimi testi...")
    metadata = vp.metadata
    print(f"   [OK] Volume required: {metadata.requires_volume}")
    assert metadata.requires_volume == True, "Volume Profile volume gerektirmeli!"

    # Test 5: Price position analysis
    print("\n6. Price position analysis...")
    result = vp.calculate(data)
    current = result.metadata['current_price']
    poc = result.value['poc']
    vah = result.value['vah']
    val = result.value['val']

    print(f"   [OK] Current price: {current:.8f}")
    print(f"   [OK] POC: {poc:.8f}")
    print(f"   [OK] VAH: {vah:.8f}")
    print(f"   [OK] VAL: {val:.8f}")

    if current > vah:
        print("   [OK] Price is above the value area (potential resistance)")
    elif current < val:
        print("   [OK] Price is below the value area (potential support)")
    else:
        print("   [OK] Price value is within the specified range")

    # Test 6: Statistics
    print("\n7. Statistical test...")
    stats = vp.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
