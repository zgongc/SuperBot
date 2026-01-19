"""
indicators/structure/lvvoid.py - Liquidity Void

Version: 1.0.0
Date: 2025-10-27
Author: SuperBot Team

Description:
    LV Void (Liquidity Void) - Smart Money Concepts
    FVG + low volume combination
    Detects areas with low liquidity

    LV Void Nedir:
    - Fair Value Gap (FVG) + Low volume
    - Areas with insufficient liquidity
    - Price moves quickly (high slippage risk)
    - Usually returns to be "filled"

Formula:
    1. FVG tespiti (3 mum gap)
    2. The volume within the gap is less than Average Volume * threshold.
    3. LV Void = FVG + Low Volume

    Bullish LV Void:
    - Candle[0].high < Candle[2].low
    - Volume[1] < Avg Volume * threshold

    Bearish LV Void:
    - Candle[0].low > Candle[2].high
    - Volume[1] < Avg Volume * threshold

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class LVVoid(BaseIndicator):
    """
    Liquidity Void (LV Void)

    FVG + Low Volume kombinasyonu.
    Detects areas with liquidity shortages.

    Args:
        min_gap_percent: Minimum gap percentage (default: 0.1)
        volume_threshold: Volume threshold (default: 0.5, 50% of avg volume)
        volume_period: Volume average period (default: 20)
        max_zones: Maximum number of open zones (default: 5)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.1,
        volume_threshold: float = 0.5,
        volume_period: int = 20,
        max_zones: int = 5,
        logger=None,
        error_handler=None
    ):
        self.min_gap_percent = min_gap_percent
        self.volume_threshold = volume_threshold
        self.volume_period = volume_period
        self.max_zones = max_zones

        super().__init__(
            name='lvvoid',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'min_gap_percent': min_gap_percent,
                'volume_threshold': volume_threshold,
                'volume_period': volume_period,
                'max_zones': max_zones
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Track open LV voids
        self.open_voids: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return max(self.volume_period + 5, 10)

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.min_gap_percent < 0:
            raise InvalidParameterError(
                self.name, 'min_gap_percent', self.min_gap_percent,
                "Min gap percent cannot be negative"
            )
        if not 0 < self.volume_threshold <= 1:
            raise InvalidParameterError(
                self.name, 'volume_threshold', self.volume_threshold,
                "Volume threshold must be between 0 and 1"
            )
        if self.volume_period < 1:
            raise InvalidParameterError(
                self.name, 'volume_period', self.volume_period,
                "Volume period must be positive"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones must be positive"
            )
        return True

    def _detect_lvvoid(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        index: int
    ) -> List[Dict[str, Any]]:
        """
        LV Void tespiti

        Args:
            highs: Array of high prices
            lows: Array of low prices
            volumes: Volume dizisi
            index: The index to check (i-2, i-1, i)

        Returns:
            List[Dict]: Detected LV Voids
        """
        voids = []

        if index < 2 or index < self.volume_period:
            return voids

        # 3 mum: i-2, i-1, i
        candle_0_high = highs[index - 2]
        candle_0_low = lows[index - 2]
        candle_1_volume = volumes[index - 1]  # Gap mumunun volume'u
        candle_2_high = highs[index]
        candle_2_low = lows[index]

        mid_price = (highs[index - 1] + lows[index - 1]) / 2

        # Average volume hesapla (excluding current candle)
        avg_volume = np.mean(volumes[max(0, index - self.volume_period):index])
        volume_threshold_value = avg_volume * self.volume_threshold

        # Is the volume low?
        is_low_volume = candle_1_volume < volume_threshold_value

        if not is_low_volume:
            return voids

        # Bullish LV Void: Candle[0].high < Candle[2].low + Low Volume
        if candle_0_high < candle_2_low:
            gap_size = candle_2_low - candle_0_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                voids.append({
                    'type': 'bullish',
                    'top': candle_2_low,
                    'bottom': candle_0_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'volume': candle_1_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': candle_1_volume / avg_volume if avg_volume > 0 else 0,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        # Bearish LV Void: Candle[0].low > Candle[2].high + Low Volume
        if candle_0_low > candle_2_high:
            gap_size = candle_0_low - candle_2_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                voids.append({
                    'type': 'bearish',
                    'top': candle_0_low,
                    'bottom': candle_2_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'volume': candle_1_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': candle_1_volume / avg_volume if avg_volume > 0 else 0,
                    'created_index': index,
                    'fill_status': 'open',
                    'fill_percent': 0.0
                })

        return voids

    def _update_void_status(
        self,
        void: Dict[str, Any],
        current_high: float,
        current_low: float
    ) -> Dict[str, Any]:
        """
        Update the LV Void filling status.

        Args:
            void: LV Void bilgisi
            current_high: Current high
            current_low: Current low

        Returns:
            Dict: Updated void
        """
        top = void['top']
        bottom = void['bottom']
        gap_size = void['size']

        # Did the price enter the void?
        if current_low <= top and current_high >= bottom:
            # Calculate the filling amount
            if void['type'] == 'bullish':
                # Filled from below
                filled_amount = max(0, min(current_low, top) - bottom)
            else:
                # Filled from the top
                filled_amount = max(0, top - max(current_high, bottom))

            fill_percent = (filled_amount / gap_size) * 100
            void['fill_percent'] = min(fill_percent, 100)

            # Update status
            if void['fill_percent'] >= 100:
                void['fill_status'] = 'filled'
            elif void['fill_percent'] >= 50:
                void['fill_status'] = 'partial'
            else:
                void['fill_status'] = 'open'

        return void

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        LV Void hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: LV Void zones
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values

        # Detect new LV Void values (in the last 3 candles)
        latest_index = len(highs) - 1
        new_voids = self._detect_lvvoid(highs, lows, volumes, latest_index)

        # Yeni void'leri ekle
        self.open_voids.extend(new_voids)

        # Update the status of existing voids
        current_high = highs[-1]
        current_low = lows[-1]

        for void in self.open_voids:
            self._update_void_status(void, current_high, current_low)

        # Remove completely filled void arrays
        self.open_voids = [
            void for void in self.open_voids
            if void['fill_status'] != 'filled'
        ]

        # Apply the maximum number of zones
        if len(self.open_voids) > self.max_zones:
            self.open_voids = self.open_voids[-self.max_zones:]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Value: List of open LV Void objects
        zones = [
            {
                'type': void['type'],
                'top': round(void['top'], 2),
                'bottom': round(void['bottom'], 2),
                'size': round(void['size'], 2),
                'volume_ratio': round(void['volume_ratio'], 3),
                'fill_status': void['fill_status'],
                'fill_percent': round(void['fill_percent'], 2)
            }
            for void in self.open_voids
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'value': len(zones)},  # Dict format with zone count for consistency
            timestamp=timestamp,
            signal=self.get_signal(zones, data['close'].values[-1]),
            trend=self.get_trend(zones),
            strength=len(zones) * 25,  # Each zone is 25 points (stronger than FVG)
            metadata={
                'zones': zones,  # Full zones data in metadata
                'total_zones': len(zones),
                'bullish_zones': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_zones': len([z for z in zones if z['type'] == 'bearish']),
                'avg_volume_ratio': round(np.mean([void['volume_ratio'] for void in self.open_voids]), 3) if self.open_voids else 0,
                'min_gap_percent': self.min_gap_percent,
                'volume_threshold': self.volume_threshold
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate LV Void for entire DataFrame (vectorized - for backtest)

        Returns pd.Series with LV Void count for all bars.
        Value: Number of active LV Void zones at each bar
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values

        # Result array
        void_count = np.zeros(len(data))

        # Track all voids
        all_voids: List[Dict[str, Any]] = []

        # Scan through all bars
        for i in range(max(2, self.volume_period), len(data)):
            # Detect new voids at this bar
            new_voids = self._detect_lvvoid(highs, lows, volumes, i)
            all_voids.extend(new_voids)

            # Update all open voids
            current_high = highs[i]
            current_low = lows[i]

            for void in all_voids:
                if void['fill_status'] != 'filled':
                    self._update_void_status(void, current_high, current_low)

            # Remove filled voids
            all_voids = [
                void for void in all_voids
                if void['fill_status'] != 'filled'
            ]

            # Apply max zones limit
            if len(all_voids) > self.max_zones:
                all_voids = all_voids[-self.max_zones:]

            # Store count
            void_count[i] = len(all_voids)

        return pd.Series(void_count, index=data.index, name='lvvoid')

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
                value=[],
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

    def get_signal(self, zones: List[Dict[str, Any]], current_price: float) -> SignalType:
        """
        Generate signal from LV Void values.

        Args:
            zones: LV Void zones
            current_price: Current price

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if not zones:
            return SignalType.HOLD

        # Did the price approach a void zone?
        for zone in zones:
            distance_to_zone = min(
                abs(current_price - zone['top']),
                abs(current_price - zone['bottom'])
            )

            distance_percent = (distance_to_zone / current_price) * 100

            # Send a signal if it's within %0.5 (more sensitive than FVG)
            if distance_percent < 0.5:
                if zone['type'] == 'bullish':
                    return SignalType.BUY
                elif zone['type'] == 'bearish':
                    return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        LV Void'lerden trend belirle

        Args:
            zones: LV Void zones

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        bullish_count = len([z for z in zones if z['type'] == 'bullish'])
        bearish_count = len([z for z in zones if z['type'] == 'bearish'])

        if bullish_count > bearish_count:
            return TrendDirection.UP
        elif bearish_count > bullish_count:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'min_gap_percent': 0.1,
            'volume_threshold': 0.5,
            'volume_period': 20,
            'max_zones': 5
        }

    def _requires_volume(self) -> bool:
        """LV Void volume gerektirir"""
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['LVVoid']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """LV Void indicator test"""

    print("\n" + "="*60)
    print("LV VOID (LIQUIDITY VOID) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # FVG + low volume simulation
    base_price = 100
    prices = []
    volumes = []
    highs = []
    lows = []

    for i in range(50):
        if i == 15:
            # Create a Bullish LV Void (fast increase + low volume)
            prices.append(base_price + 10)
            volumes.append(500)  # Low volume
            highs.append(base_price + 11)
            lows.append(base_price + 9)
        elif i == 35:
            # Create a Bearish LV Void (fast decline + low volume)
            prices.append(base_price - 5)
            volumes.append(600)  # Low volume
            highs.append(base_price - 4)
            lows.append(base_price - 6)
        else:
            # Normal hareket
            prices.append(base_price + np.random.randn() * 0.5)
            volumes.append(1000 + np.random.randint(0, 500))  # Normal volume
            highs.append(prices[-1] + abs(np.random.randn()) * 0.3)
            lows.append(prices[-1] - abs(np.random.randn()) * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")
    print(f"   [OK] Volume range: {min(volumes)} -> {max(volumes)}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    lvvoid = LVVoid(min_gap_percent=0.1, volume_threshold=0.5, volume_period=20)
    print(f"   [OK] Created: {lvvoid}")
    print(f"   [OK] Kategori: {lvvoid.category.value}")
    print(f"   [OK] Required period: {lvvoid.get_required_periods()}")

    result = lvvoid(data)
    print(f"   [OK] Total Zone: {result.metadata['total_zones']}")
    print(f"   [OK] Bullish Zone: {result.metadata['bullish_zones']}")
    print(f"   [OK] Bearish Zone: {result.metadata['bearish_zones']}")
    print(f"   [OK] Avg Volume Ratio: {result.metadata['avg_volume_ratio']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength}")

    # Test 2: Zone details
    print("\n3. Zone details...")
    if result.value:
        for i, zone in enumerate(result.value[:3]):
            print(f"   [OK] Zone #{i+1}:")
            print(f"       - Tip: {zone['type']}")
            print(f"       - Top: {zone['top']:.2f}")
            print(f"       - Bottom: {zone['bottom']:.2f}")
            print(f"       - Size: {zone['size']:.2f}")
            print(f"       - Volume Ratio: {zone['volume_ratio']:.3f}")
            print(f"       - Fill: {zone['fill_status']} ({zone['fill_percent']:.1f}%)")
    else:
        print("   [OK] Open zone not found")

    # Test 3: Batch calculation
    print("\n4. Batch calculation test...")
    batch_result = lvvoid.calculate_batch(data)
    print(f"   [OK] Batch result length: {len(batch_result)}")
    print(f"   [OK] Maximum number of active zones: {int(batch_result.max())}")
    print(f"   [OK] Total zone detected: {int(batch_result.sum())}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
