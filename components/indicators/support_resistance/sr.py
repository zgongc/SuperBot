"""
indicators/support_resistance/support_resistance.py - Support and Resistance Levels

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Support and Resistance - Automatic detection of support and resistance levels.
    It determines important support and resistance levels by finding swing high/low points
    from historical price data.

Formula:
    - Find local maximums and minimums
    - Merge nearby levels
    - Rank by importance based on frequency
    - Return the top N levels

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


class SupportResistance(BaseIndicator):
    """
    Support and Resistance Level Detector

    Find swing high/low points by analyzing historical price data.
    automatically detects support and resistance levels.

    Args:
        lookback: Lookback search period (default: 50)
        num_levels: Number of levels to generate (default: 5)
        tolerance: Level merging tolerance percentage (default: 0.5)
    """

    def __init__(
        self,
        lookback: int = 50,
        num_levels: int = 5,
        tolerance: float = 0.5,
        logger=None,
        error_handler=None
    ):
        self.lookback = lookback
        self.num_levels = num_levels
        self.tolerance = tolerance

        super().__init__(
            name='support_resistance',
            category=IndicatorCategory.SUPPORT_RESISTANCE,
            indicator_type=IndicatorType.LEVELS,
            params={
                'lookback': lookback,
                'num_levels': num_levels,
                'tolerance': tolerance
            },
            logger=logger,
            error_handler=error_handler
        )

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.lookback

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.lookback < 10:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback must be at least 10"
            )
        if self.num_levels < 1:
            raise InvalidParameterError(
                self.name, 'num_levels', self.num_levels,
                "The number of levels must be positive"
            )
        if self.tolerance <= 0:
            raise InvalidParameterError(
                self.name, 'tolerance', self.tolerance,
                "Tolerance must be positive"
            )
        return True

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Support/Resistance seviyeleri hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Support and resistance levels
        """
        # Son lookback periyodu al
        recent_data = data.iloc[-self.lookback:]
        high = recent_data['high'].values
        low = recent_data['low'].values
        close = recent_data['close'].values
        
        # Fill the buffers (preparation for incremental update)
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.lookback)
            self._low_buffer = deque(maxlen=self.lookback)
            self._close_buffer = deque(maxlen=self.lookback)
            
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        
        self._high_buffer.extend(high)
        self._low_buffer.extend(low)
        self._close_buffer.extend(close)

        # Find swing points
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        # Merge levels
        all_levels = np.concatenate([swing_highs, swing_lows])
        clustered_levels = self._cluster_levels(all_levels)

        # Select the most powerful levels
        top_levels = self._select_top_levels(clustered_levels, close[-1])

        current_price = close[-1]
        timestamp = int(data.iloc[-1]['timestamp'])

        # Separate as support and resistance
        supports = [lvl for lvl in top_levels if lvl < current_price]
        resistances = [lvl for lvl in top_levels if lvl > current_price]

        # Create levels (always 6 keys - R1,R2,R3,S1,S2,S3)
        levels = {}
        sorted_resistances = sorted(resistances)[:3]
        sorted_supports = sorted(supports, reverse=True)[:3]

        for i in range(1, 4):
            if i <= len(sorted_resistances):
                levels[f'R{i}'] = round(sorted_resistances[i-1], 2)
            else:
                levels[f'R{i}'] = np.nan

        for i in range(1, 4):
            if i <= len(sorted_supports):
                levels[f'S{i}'] = round(sorted_supports[i-1], 2)
            else:
                levels[f'S{i}'] = np.nan

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=levels,
            timestamp=timestamp,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, supports, resistances),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'total_levels': len(top_levels),
                'num_supports': len(supports),
                'num_resistances': len(resistances)
            }
        )



    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation (for backtesting)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: R1..R3, S1..S3 seviyeleri
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)
        
        # 1. Find all swing points in advance (Vectorized)
        # Assuming Window=5
        window = 5
        
        # Swing Highs
        # high[i] > high[i-1] ... high[i-window] AND high[i] > high[i+1] ... high[i+window]
        # This is exactly the _find_swing_highs logic, but for the entire series.
        # argrelextrema can be used, but a sliding window max with numpy might be faster.
        
        # Find all swings using a simple loop (once for all data)
        # Or call the _find_swing_highs method for all data.
        all_swing_highs_idx = []
        all_swing_lows_idx = []
        
        # Note: _find_swing_highs returns a value, not an index. We need the index.
        # Therefore, let's inline the code here or create a version that returns an index.
        
        for i in range(window, n - window):
            if high[i] == np.max(high[i - window:i + window + 1]):
                all_swing_highs_idx.append(i)
            if low[i] == np.min(low[i - window:i + window + 1]):
                all_swing_lows_idx.append(i)
                
        all_swing_highs_idx = np.array(all_swing_highs_idx)
        all_swing_lows_idx = np.array(all_swing_lows_idx)
        
        # Result arrays
        results = {f'R{i}': np.full(n, np.nan) for i in range(1, 4)}
        results.update({f'S{i}': np.full(n, np.nan) for i in range(1, 4)})
        
        # Calculate for each bar (looking back up to the Lookback period)
        # This part must be a loop because the cluster and select logic is complex
        # However, we only perform operations on indices that contain swing points; for each bar, we take the swings in the current window.
        
        for i in range(self.lookback, n):
            start_idx = i - self.lookback
            end_idx = i
            
            # Filter swings within this range
            # We can quickly find the range using np.searchsorted
            
            # Highs
            h_start = np.searchsorted(all_swing_highs_idx, start_idx)
            h_end = np.searchsorted(all_swing_highs_idx, end_idx) # end_idx is not included
            current_swing_highs = high[all_swing_highs_idx[h_start:h_end]]
            
            # Lows
            l_start = np.searchsorted(all_swing_lows_idx, start_idx)
            l_end = np.searchsorted(all_swing_lows_idx, end_idx)
            current_swing_lows = low[all_swing_lows_idx[l_start:l_end]]
            
            # Merge and Cluster
            all_levels = np.concatenate([current_swing_highs, current_swing_lows])
            if len(all_levels) == 0:
                continue
                
            clustered = self._cluster_levels(all_levels)
            
            # Select
            current_price = close[i]
            top_levels = self._select_top_levels(clustered, current_price)
            
            supports = sorted([lvl for lvl in top_levels if lvl < current_price], reverse=True)
            resistances = sorted([lvl for lvl in top_levels if lvl > current_price])
            
            # Save the results
            for k, val in enumerate(resistances[:3]):
                results[f'R{k+1}'][i] = val
            for k, val in enumerate(supports[:3]):
                results[f'S{k+1}'][i] = val
                
        return pd.DataFrame(results, index=data.index)

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current support/resistance levels
        """
        # Buffer management
        if not hasattr(self, '_high_buffer'):
            from collections import deque
            self._high_buffer = deque(maxlen=self.lookback)
            self._low_buffer = deque(maxlen=self.lookback)
            self._close_buffer = deque(maxlen=self.lookback)
            
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
        
        if len(self._high_buffer) < self.lookback:
            return IndicatorResult(
                value=0.0,
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'insufficient_data': True}
            )
            
        # Calculation (Run the existing calculate logic on the buffer)
        high = np.array(self._high_buffer)
        low = np.array(self._low_buffer)
        close = np.array(self._close_buffer)
        
        # Find swing points
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)

        # Merge levels
        all_levels = np.concatenate([swing_highs, swing_lows])
        clustered_levels = self._cluster_levels(all_levels)

        # Select the most powerful levels
        current_price = close[-1]
        top_levels = self._select_top_levels(clustered_levels, current_price)
        
        # Separate as support and resistance
        supports = [lvl for lvl in top_levels if lvl < current_price]
        resistances = [lvl for lvl in top_levels if lvl > current_price]

        # Create levels (always 6 keys - R1,R2,R3,S1,S2,S3)
        levels = {}
        sorted_resistances = sorted(resistances)[:3]
        sorted_supports = sorted(supports, reverse=True)[:3]

        for i in range(1, 4):
            if i <= len(sorted_resistances):
                levels[f'R{i}'] = round(sorted_resistances[i-1], 2)
            else:
                levels[f'R{i}'] = np.nan

        for i in range(1, 4):
            if i <= len(sorted_supports):
                levels[f'S{i}'] = round(sorted_supports[i-1], 2)
            else:
                levels[f'S{i}'] = np.nan

        return IndicatorResult(
            value=levels,
            timestamp=timestamp_val,
            signal=self.get_signal(current_price, levels),
            trend=self.get_trend(current_price, supports, resistances),
            strength=self.calculate_strength(current_price, levels),
            metadata={
                'lookback': self.lookback,
                'current_price': round(current_price, 2),
                'total_swing_highs': len(swing_highs),
                'total_swing_lows': len(swing_lows),
                'total_levels': len(top_levels),
                'num_supports': len(supports),
                'num_resistances': len(resistances)
            }
        )
    def _find_swing_highs(self, high: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Find swing high points.

        Args:
            high: High prices
            window: Window size

        Returns:
            np.ndarray: Swing high seviyeleri
        """
        swing_highs = []
        for i in range(window, len(high) - window):
            if high[i] == np.max(high[i - window:i + window + 1]):
                swing_highs.append(high[i])
        return np.array(swing_highs)

    def _find_swing_lows(self, low: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Find swing low points.

        Args:
            low: Low prices
            window: Window size

        Returns:
            np.ndarray: Swing low seviyeleri
        """
        swing_lows = []
        for i in range(window, len(low) - window):
            if low[i] == np.min(low[i - window:i + window + 1]):
                swing_lows.append(low[i])
        return np.array(swing_lows)

    def _cluster_levels(self, levels: np.ndarray) -> np.ndarray:
        """
        Merge nearby levels.

        Args:
            levels: Seviye dizisi

        Returns:
            np.ndarray: Merged levels
        """
        if len(levels) == 0:
            return levels

        sorted_levels = np.sort(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Calculate the percentage difference with the previous level
            pct_diff = abs((level - current_cluster[-1]) / current_cluster[-1] * 100)

            if pct_diff <= self.tolerance:
                # Add to the same cluster
                current_cluster.append(level)
            else:
                # Start a new cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        # Add the last cluster
        if current_cluster:
            clustered.append(np.mean(current_cluster))

        return np.array(clustered)

    def _select_top_levels(self, levels: np.ndarray, current_price: float) -> list:
        """
        Select the most powerful levels.

        Args:
            levels: Seviye dizisi
            current_price: Current price

        Returns:
            list: Selected levels
        """
        if len(levels) == 0:
            return []

        # Sort and select based on proximity to the current price.
        distances = np.abs(levels - current_price)
        sorted_indices = np.argsort(distances)

        # Select the closest number of levels.
        selected = levels[sorted_indices[:self.num_levels * 2]]
        return sorted(selected.tolist())

    def get_signal(self, price: float, levels: dict) -> SignalType:
        """
        Generate a signal based on the price's S/R levels.

        Args:
            price: Current price
            levels: Support/Resistance levels

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        # Find the nearest support and resistance levels
        supports = [v for k, v in levels.items() if k.startswith('S')]
        resistances = [v for k, v in levels.items() if k.startswith('R')]

        if supports and min(abs(price - s) for s in supports) < price * 0.01:
            # Near support level
            return SignalType.BUY
        elif resistances and min(abs(price - r) for r in resistances) < price * 0.01:
            # Close to the resistance level
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, price: float, supports: list, resistances: list) -> TrendDirection:
        """
        Determine the trend based on the price levels of support/resistance.

        Args:
            price: Current price
            supports: Destek seviyeleri
            resistances: Resistance levels

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not supports and resistances:
            return TrendDirection.UP  # Remained below all levels
        elif supports and not resistances:
            return TrendDirection.DOWN  # All levels are above
        else:
            return TrendDirection.NEUTRAL

    def calculate_strength(self, price: float, levels: dict) -> float:
        """
        Calculate the strength of the price based on the levels.

        Args:
            price: Current price
            levels: Support/Resistance levels

        Returns:
            float: Power value (0-100)
        """
        if not levels:
            return 50.0

        all_values = list(levels.values())
        if not all_values:
            return 50.0

        min_level = min(all_values)
        max_level = max(all_values)

        if max_level == min_level:
            return 50.0

        # The position of the price between the levels.
        position = (price - min_level) / (max_level - min_level) * 100
        return min(max(position, 0), 100)

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'lookback': 50,
            'num_levels': 5,
            'tolerance': 0.5
        }

    def _requires_volume(self) -> bool:
        """Support/Resistance volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['SupportResistance']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Support/Resistance indicator test"""

    print("\n" + "="*60)
    print("SUPPORT AND RESISTANCE LEVELS TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(100)]

    # Simulate price movement with trend changes
    base_price = 100
    prices = [base_price]

    for i in range(99):
        # Periyodik dalgalanma ekle
        wave = 10 * np.sin(i / 10)
        noise = np.random.randn() * 1
        prices.append(base_price + wave + noise)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.5 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.5 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    sr = SupportResistance(lookback=50, num_levels=5, tolerance=0.5)
    print(f"   [OK] Created: {sr}")
    print(f"   [OK] Kategori: {sr.category.value}")
    print(f"   [OK] Tip: {sr.indicator_type.value}")
    print(f"   [OK] Required period: {sr.get_required_periods()}")

    result = sr(data)
    print(f"   [OK] Detected Levels:")
    for level, value in sorted(result.value.items(), key=lambda x: x[1], reverse=True):
        level_type = "Resistance" if level.startswith('R') else "Support"
        print(f"        {level} ({level_type}): {value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength:.2f}")
    print(f"   [OK] Metadata: {result.metadata}")

    # Test 2: Different lookback values
    print("\n3. Different lookback test...")
    for lookback in [30, 50, 80]:
        sr_test = SupportResistance(lookback=lookback, num_levels=5)
        result = sr_test.calculate(data)
        print(f"   [OK] SR(lookback={lookback}) - Seviyeler: {len(result.value)} | "
              f"Signal: {result.signal.value}")

    # Test 3: Different level numbers
    print("\n4. Test for different level count...")
    for num in [3, 5, 7]:
        sr_test = SupportResistance(lookback=50, num_levels=num)
        result = sr_test.calculate(data)
        print(f"   [OK] SR(num_levels={num}) - Detection: {len(result.value)} levels")

    # Test 4: Seviye analizi
    print("\n5. Seviye analizi...")
    result = sr.calculate(data)
    current = result.metadata['current_price']
    print(f"   [OK] Current price: {current}")

    supports = {k: v for k, v in result.value.items() if k.startswith('S')}
    resistances = {k: v for k, v in result.value.items() if k.startswith('R')}

    if supports:
        nearest_support = max(supports.values())
        distance_to_support = ((current - nearest_support) / current * 100)
        print(f"   [OK] Nearest support: {nearest_support:.2f} (Distance: {distance_to_support:.2f}%)")

    if resistances:
        nearest_resistance = min(resistances.values())
        distance_to_resistance = ((nearest_resistance - current) / current * 100)
        print(f"   [OK] Nearest resistance: {nearest_resistance:.2f} (Distance: {distance_to_resistance:.2f}%)")

    # Test 5: Statistics
    print("\n6. Statistical test...")
    stats = sr.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 6: Metadata
    print("\n7. Metadata testi...")
    metadata = sr.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
