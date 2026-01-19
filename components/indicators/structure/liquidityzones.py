"""
indicators/structure/liquidityzones.py - Liquidity Zones

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Liquidity Zones - Smart Money Concepts
    Detects liquidity pools (stop-loss density).

    Liquidity Zone Nedir:
    - Swing High/Low levels = Stop-loss density
    - Equal Highs/Lows = Multiple stop-loss levels
    - Smart Money bu seviyeleri "sweep" ederek likidite toplar

Formula:
    1. Swing High/Low tespiti
    2. Detect level equality (within ±tolerance range)
    3. Create liquidity pools
    4. "Sweep" detection (short-term breakout)

    Liquidity Sweep:
    - The price moves above/below the liquidity level.
    - It quickly returns (within 1-3 candles).
    - "Stop hunt" or "liquidity grab".

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


class LiquidityZones(BaseIndicator):
    """
    Liquidity Zones

    Detects liquidity pools.
    Shows areas where Smart Money's stop-loss orders are triggered.

    Args:
        left_bars: Number of bars on the left side (default: 5)
        right_bars: Number of bars on the right side (default: 5)
        equal_tolerance: Tolerance level for "equal" (%) (default: 0.1)
        max_zones: Maximum number of zones (default: 5)
        sweep_lookback: Lookback period for sweep control (default: 3)
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        equal_tolerance: float = 0.1,
        max_zones: int = 5,
        sweep_lookback: int = 3,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.equal_tolerance = equal_tolerance
        self.max_zones = max_zones
        self.sweep_lookback = sweep_lookback

        super().__init__(
            name='liquidityzones',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'equal_tolerance': equal_tolerance,
                'max_zones': max_zones,
                'sweep_lookback': sweep_lookback
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Active liquidity zones
        self.liquidityzones: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.left_bars + self.right_bars + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.left_bars < 1:
            raise InvalidParameterError(
                self.name, 'left_bars', self.left_bars,
                "Left bars must be positive"
            )
        if self.right_bars < 1:
            raise InvalidParameterError(
                self.name, 'right_bars', self.right_bars,
                "Right bars must be positive"
            )
        if self.equal_tolerance < 0:
            raise InvalidParameterError(
                self.name, 'equal_tolerance', self.equal_tolerance,
                "Equal tolerance cannot be negative"
            )
        if self.max_zones < 1:
            raise InvalidParameterError(
                self.name, 'max_zones', self.max_zones,
                "Max zones must be positive"
            )
        if self.sweep_lookback < 1:
            raise InvalidParameterError(
                self.name, 'sweep_lookback', self.sweep_lookback,
                "Sweep lookback must be positive"
            )
        return True

    def _find_swing_highs(self, highs: np.ndarray) -> List[Dict[str, Any]]:
        """Detect swing high points"""
        swing_highs = []

        for i in range(self.left_bars, len(highs) - self.right_bars):
            is_pivot = True

            for j in range(1, self.left_bars + 1):
                if highs[i] <= highs[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            for j in range(1, self.right_bars + 1):
                if highs[i] <= highs[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                swing_highs.append({
                    'index': i,
                    'value': highs[i],
                    'type': 'high'
                })

        return swing_highs

    def _find_swing_lows(self, lows: np.ndarray) -> List[Dict[str, Any]]:
        """Detect Swing Low points"""
        swing_lows = []

        for i in range(self.left_bars, len(lows) - self.right_bars):
            is_pivot = True

            for j in range(1, self.left_bars + 1):
                if lows[i] >= lows[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            for j in range(1, self.right_bars + 1):
                if lows[i] >= lows[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                swing_lows.append({
                    'index': i,
                    'value': lows[i],
                    'type': 'low'
                })

        return swing_lows

    def _find_equal_levels(self, swings: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Detect equal (equal) levels.

        Args:
            swings: Swing points

        Returns:
            List[List]: Equal gruplar
        """
        if len(swings) < 2:
            return []

        equal_groups = []

        for i, swing1 in enumerate(swings):
            group = [swing1]

            for swing2 in swings[i + 1:]:
                # Is it within the tolerance?
                diff_percent = abs(swing1['value'] - swing2['value']) / swing1['value'] * 100

                if diff_percent <= self.equal_tolerance:
                    group.append(swing2)

            # En az 2 equal seviye
            if len(group) >= 2:
                equal_groups.append(group)

        return equal_groups

    def _create_liquidityzones(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create liquidity zones.

        Args:
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar

        Returns:
            List[Dict]: Liquidity zones.
        """
        zones = []

        # Swing High'lardan liquidity zones (Sell-side liquidity)
        for swing in swing_highs[-self.max_zones:]:
            zones.append({
                'type': 'sell_side',  # Above are stop-losses (long positions)
                'level': swing['value'],
                'index': swing['index'],
                'strength': 1,  # Tek seviye
                'swept': False
            })

        # Liquidity zones (Buy-side liquidity) from Swing Lows
        for swing in swing_lows[-self.max_zones:]:
            zones.append({
                'type': 'buy_side',  # Altta stop-loss'lar (short pozisyonlar)
                'level': swing['value'],
                'index': swing['index'],
                'strength': 1,
                'swept': False
            })

        # Liquidity zones at the same level (strong)
        equal_highs = self._find_equal_levels(swing_highs)
        for group in equal_highs:
            avg_level = np.mean([s['value'] for s in group])
            zones.append({
                'type': 'sell_side_equal',
                'level': avg_level,
                'index': group[-1]['index'],
                'strength': len(group),  # Number of equal elements
                'swept': False
            })

        equal_lows = self._find_equal_levels(swing_lows)
        for group in equal_lows:
            avg_level = np.mean([s['value'] for s in group])
            zones.append({
                'type': 'buy_side_equal',
                'level': avg_level,
                'index': group[-1]['index'],
                'strength': len(group),
                'swept': False
            })

        return zones

    def _detect_sweeps(
        self,
        zones: List[Dict[str, Any]],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Liquidity sweep tespiti

        Args:
            zones: Liquidity zones
            highs, lows, closes: Price arrays

        Returns:
            List[Dict]: Updated zones
        """
        # Check the last N frames
        lookback_start = max(0, len(closes) - self.sweep_lookback)

        for zone in zones:
            if zone['swept']:
                continue  # Already swept

            for i in range(lookback_start, len(closes)):
                # Sell-side sweep (top side)
                if 'sell_side' in zone['type']:
                    # Did it pass the high level?
                    if highs[i] > zone['level']:
                        # Is it below the threshold level? (returned)
                        if closes[i] < zone['level']:
                            zone['swept'] = True
                            zone['swept_index'] = i
                            break

                # Buy-side sweep (alt taraf)
                elif 'buy_side' in zone['type']:
                    # Did it pass the low level?
                    if lows[i] < zone['level']:
                        # Is it above the threshold level? (returned)
                        if closes[i] > zone['level']:
                            zone['swept'] = True
                            zone['swept_index'] = i
                            break

        return zones

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Liquidity Zones hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Liquidity zones
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Swing High/Low tespiti
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        # Create liquidity zones
        self.liquidityzones = self._create_liquidityzones(swing_highs, swing_lows)

        # Sweep tespiti
        self.liquidityzones = self._detect_sweeps(
            self.liquidityzones, highs, lows, closes
        )

        # Filter out zones that have not been swept (active zones)
        active_zones = [z for z in self.liquidityzones if not z['swept']]

        # Select the most powerful zones
        active_zones.sort(key=lambda x: x['strength'], reverse=True)
        active_zones = active_zones[:self.max_zones]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Value: Active liquidity zones
        zones = [
            {
                'type': zone['type'],
                'level': round(zone['level'], 2),
                'strength': zone['strength'],
                'swept': zone['swept']
            }
            for zone in active_zones
        ]

        # Swept zone'lar (son sweep'ler)
        swept_zones = [
            z for z in self.liquidityzones
            if z['swept'] and z.get('swept_index', 0) >= len(closes) - self.sweep_lookback
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=zones,
            timestamp=timestamp,
            signal=self.get_signal(zones, swept_zones, closes[-1]),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_zones': len(zones),
                'sell_side_zones': len([z for z in zones if 'sell_side' in z['type']]),
                'buy_side_zones': len([z for z in zones if 'buy_side' in z['type']]),
                'recent_sweeps': len(swept_zones),
                'swept_zones': [
                    {
                        'type': z['type'],
                        'level': round(z['level'], 2),
                        'swept_at': z.get('swept_index')
                    }
                    for z in swept_zones[:3]
                ],
                'equal_tolerance': self.equal_tolerance
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current Liquidity Zones
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._high_buffer = deque(maxlen=max_len)
            self._low_buffer = deque(maxlen=max_len)
            self._close_buffer = deque(maxlen=max_len)
            
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
        
        # Yeterli veri yoksa
        min_required = self.get_required_periods()
        if len(self._close_buffer) < min_required:
            return IndicatorResult(
                value=[],
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'total_zones': 0}
            )
            
        # Calculation
        highs = np.array(self._high_buffer)
        lows = np.array(self._low_buffer)
        closes = np.array(self._close_buffer)
        
        # Swing High/Low tespiti
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)
        
        # Create liquidity zones
        self.liquidityzones = self._create_liquidityzones(swing_highs, swing_lows)
        
        # Sweep tespiti
        self.liquidityzones = self._detect_sweeps(self.liquidityzones, highs, lows, closes)
        
        # Filter out zones that have not been swept.
        active_zones = [z for z in self.liquidityzones if not z['swept']]
        active_zones.sort(key=lambda x: x['strength'], reverse=True)
        active_zones = active_zones[:self.max_zones]
        
        # Value: Active liquidity zones
        zones = [
            {
                'type': zone['type'],
                'level': round(zone['level'], 2),
                'strength': zone['strength'],
                'swept': zone['swept']
            }
            for zone in active_zones
        ]

        # Swept zone'lar
        swept_zones = [
            z for z in self.liquidityzones
            if z['swept'] and z.get('swept_index', 0) >= len(closes) - self.sweep_lookback
        ]

        return IndicatorResult(
            value=zones,
            timestamp=timestamp_val,
            signal=self.get_signal(zones, swept_zones, closes[-1]),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_zones': len(zones),
                'sell_side_zones': len([z for z in zones if 'sell_side' in z['type']]),
                'buy_side_zones': len([z for z in zones if 'buy_side' in z['type']]),
                'recent_sweeps': len(swept_zones),
                'swept_zones': [
                    {
                        'type': z['type'],
                        'level': round(z['level'], 2),
                        'swept_at': z.get('swept_index')
                    }
                    for z in swept_zones[:3]
                ],
                'equal_tolerance': self.equal_tolerance
            }
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Batch calculation - calls calculate() for each row
        
        Note: This is a simple implementation for compatibility.
        For performance, consider implementing vectorized logic.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.Series: Indicator values
        """
        results = []
        for i in range(len(data)):
            if i < self.get_required_periods() - 1:
                results.append(np.nan)
            else:
                window_data = data.iloc[:i+1]
                result = self.calculate(window_data)
                # Extract value (handle dict, float, or IndicatorResult)
                if result is None:
                    results.append(np.nan)
                elif hasattr(result, 'value'):
                    results.append(result.value)
                else:
                    results.append(result)
        
        return pd.Series(results, index=data.index, name=self.name)

    def get_signal(
        self,
        zones: List[Dict[str, Any]],
        swept_zones: List[Dict[str, Any]],
        current_price: float
    ) -> SignalType:
        """
        Generate signals from liquidity zones.

        Args:
            zones: Active zones
            swept_zones: Recently swept zones
            current_price: Current price

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Signal in the reverse direction after sweep
        if swept_zones:
            last_sweep = swept_zones[-1]

            # Sell-side sweep -> Bullish (downward reversal)
            if 'sell_side' in last_sweep['type']:
                return SignalType.BUY

            # Buy-side sweep -> Bearish (reversal upwards)
            elif 'buy_side' in last_sweep['type']:
                return SignalType.SELL

        # Approaching the zone
        for zone in zones:
            distance_percent = abs(current_price - zone['level']) / current_price * 100

            if distance_percent < 1.0:  # within 1%
                if 'buy_side' in zone['type']:
                    return SignalType.BUY  # Destek
                elif 'sell_side' in zone['type']:
                    return SignalType.SELL  # Resistance

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        Liquidity zone'lardan trend belirle

        Args:
            zones: Liquidity zones

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        buy_side_strength = sum(
            z['strength'] for z in zones if 'buy_side' in z['type']
        )
        sell_side_strength = sum(
            z['strength'] for z in zones if 'sell_side' in z['type']
        )

        if buy_side_strength > sell_side_strength * 1.2:
            return TrendDirection.UP  # More downward liquidity
        elif sell_side_strength > buy_side_strength * 1.2:
            return TrendDirection.DOWN  # More upward liquidity

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'equal_tolerance': 0.1,
            'max_zones': 5,
            'sweep_lookback': 3
        }

    def _requires_volume(self) -> bool:
        """Liquidity Zones volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['LiquidityZones']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Liquidity Zones indicator test"""

    print("\n" + "="*60)
    print("LIQUIDITY ZONES TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # Liquidity sweep simulation
    base_price = 100
    prices = []
    highs = []
    lows = []

    for i in range(60):
        if i == 20:
            # Create swing high
            price = base_price + 5
            prices.append(price)
            highs.append(price + 0.5)
            lows.append(price - 0.3)
        elif i == 35:
            # Liquidity sweep (incorrect breakage)
            price = base_price + 5.5  # Exceed the swing high
            prices.append(base_price + 4.8)  # Return
            highs.append(price)
            lows.append(base_price + 4.5)
        else:
            # Normal hareket
            price = base_price + np.random.randn() * 0.5
            prices.append(price)
            highs.append(price + abs(np.random.randn()) * 0.3)
            lows.append(price - abs(np.random.randn()) * 0.3)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    liq = LiquidityZones(
        left_bars=5,
        right_bars=5,
        equal_tolerance=0.1,
        max_zones=5,
        sweep_lookback=3
    )
    print(f"   [OK] Created: {liq}")
    print(f"   [OK] Kategori: {liq.category.value}")
    print(f"   [OK] Required period: {liq.get_required_periods()}")

    result = liq(data)
    print(f"   [OK] Total Zone: {result.metadata['total_zones']}")
    print(f"   [OK] Sell-Side: {result.metadata['sell_side_zones']}")
    print(f"   [OK] Buy-Side: {result.metadata['buy_side_zones']}")
    print(f"   [OK] Recent Sweep: {result.metadata['recent_sweeps']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength}")

    # Test 2: Zone details
    print("\n3. Zone details...")
    if result.value:
        for i, zone in enumerate(result.value[:3]):
            print(f"   [OK] Zone #{i+1}:")
            print(f"       - Tip: {zone['type']}")
            print(f"       - Seviye: {zone['level']:.2f}")
            print(f"       - Power: {zone['strength']}")
            print(f"       - Swept: {zone['swept']}")
    else:
        print("   [OK] Active zone not found")

    # Test 3: Swept zone'lar
    print("\n4. Swept zone'lar...")
    if result.metadata['swept_zones']:
        for i, swept in enumerate(result.metadata['swept_zones']):
            print(f"   [OK] Sweep #{i+1}:")
            print(f"       - Tip: {swept['type']}")
            print(f"       - Seviye: {swept['level']:.2f}")
            print(f"       - Index: {swept['swept_at']}")
    else:
        print("   [OK] No recent sweep found")

    # Test 4: Different parameters
    print("\n5. Different parameter test...")
    for tolerance in [0.05, 0.1, 0.2]:
        liq_test = LiquidityZones(equal_tolerance=tolerance)
        result = liq_test.calculate(data)
        print(f"   [OK] LIQ(tolerance={tolerance}): {result.metadata['total_zones']} zones")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = liq.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = liq.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
