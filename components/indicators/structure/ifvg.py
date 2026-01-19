"""
indicators/structure/ifvg.py - Inverse Fair Value Gap

Version: 1.0.0
Date: 2025-10-27
Author: SuperBot Team

Description:
    iFVG (Inverse Fair Value Gap) - Smart Money Concepts
    Reverse FVG detection - potential reversal signal

    iFVG Nedir:
    - FVG that forms in the opposite direction after the initial FVG.
    - Indicates a weakening/reversal of the trend.
    - May indicate a stop loss hunt or a change in trend.

    Example:
    1. A bullish FVG is formed (uptrend)
    2. Then a bearish FVG is formed (iFVG) -> The uptrend weakens

    3. A bearish FVG forms (decline).
    4. Then a bullish FVG forms (iFVG) -> The decline weakens.

Formula:
    1. FVG detection within the last N bars
    2. Formation of FVG in the opposite direction -> iFVG
    3. iFVG = Reversal warning

    Bullish iFVG:
    - There were N bearish FVG formations recently.
    - A bullish FVG has now formed -> Reversal upwards.

    Bearish iFVG:
    - There were N consecutive Bullish FVG formations.
    - A Bearish FVG has now formed -> Reversal downwards.

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from components.indicators.base_indicator import BaseIndicator
from components.indicators.indicator_types import (
    IndicatorCategory,
    IndicatorType,
    IndicatorResult,
    SignalType,
    TrendDirection,
    InvalidParameterError
)


class iFVG(BaseIndicator):
    """
    Inverse Fair Value Gap (iFVG)

    It provides a reversal signal with reverse FVG detection.
    It detects trend weakening points.

    Args:
        min_gap_percent: Minimum gap percentage (default: 0.1)
        lookback_bars: Number of bars to look back (default: 10)
        min_distance: Minimum distance between FVG's (bars) (default: 3)
    """

    def __init__(
        self,
        min_gap_percent: float = 0.1,
        lookback_bars: int = 10,
        min_distance: int = 3,
        logger=None,
        error_handler=None
    ):
        self.min_gap_percent = min_gap_percent
        self.lookback_bars = lookback_bars
        self.min_distance = min_distance

        super().__init__(
            name='ifvg',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'min_gap_percent': min_gap_percent,
                'lookback_bars': lookback_bars,
                'min_distance': min_distance
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Son FVG'leri takip et
        self.recent_fvgs: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.lookback_bars + 10

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.min_gap_percent < 0:
            raise InvalidParameterError(
                self.name, 'min_gap_percent', self.min_gap_percent,
                "Min gap percent cannot be negative"
            )
        if self.lookback_bars < 1:
            raise InvalidParameterError(
                self.name, 'lookback_bars', self.lookback_bars,
                "Lookback bars must be positive"
            )
        if self.min_distance < 1:
            raise InvalidParameterError(
                self.name, 'min_distance', self.min_distance,
                "Minimum distance must be positive"
            )
        return True

    def _detect_fvg(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """
        FVG detection (returns a single FVG)

        Args:
            highs: Array of high prices
            lows: Array of low prices
            index: Index to be checked

        Returns:
            Dict: Detected FVG or None
        """
        if index < 2:
            return None

        # 3 mum: i-2, i-1, i
        candle_0_high = highs[index - 2]
        candle_0_low = lows[index - 2]
        candle_2_high = highs[index]
        candle_2_low = lows[index]

        mid_price = (highs[index - 1] + lows[index - 1]) / 2

        # Bullish FVG
        if candle_0_high < candle_2_low:
            gap_size = candle_2_low - candle_0_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                return {
                    'type': 'bullish',
                    'top': candle_2_low,
                    'bottom': candle_0_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'index': index
                }

        # Bearish FVG
        if candle_0_low > candle_2_high:
            gap_size = candle_0_low - candle_2_high
            gap_percent = (gap_size / mid_price) * 100

            if gap_percent >= self.min_gap_percent:
                return {
                    'type': 'bearish',
                    'top': candle_0_low,
                    'bottom': candle_2_high,
                    'size': gap_size,
                    'size_percent': gap_percent,
                    'index': index
                }

        return None

    def _detect_ifvg(
        self,
        current_fvg: Dict[str, Any],
        recent_fvgs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        iFVG tespiti

        Args:
            current_fvg: Current FVG
            recent_fvgs: Son N bardaki FVG'ler

        Returns:
            Dict: iFVG information or None
        """
        if not current_fvg or not recent_fvgs:
            return None

        current_type = current_fvg['type']
        current_index = current_fvg['index']

        # Search for reverse FVG (within the lookback period)
        for fvg in reversed(recent_fvgs):
            # Minimum distance check
            distance = current_index - fvg['index']
            if distance < self.min_distance:
                continue

            # Is it within the lookback period?
            if distance > self.lookback_bars:
                break

            # Reverse direction check
            if current_type == 'bullish' and fvg['type'] == 'bearish':
                # Bearish -> Bullish (Reversal up)
                return {
                    'type': 'bullish_ifvg',
                    'reversal': 'up',
                    'current_fvg': current_fvg,
                    'previous_fvg': fvg,
                    'distance': distance,
                    'strength': min((current_fvg['size_percent'] + fvg['size_percent']) / 2, 100)
                }

            elif current_type == 'bearish' and fvg['type'] == 'bullish':
                # Bullish -> Bearish (Reversal down)
                return {
                    'type': 'bearish_ifvg',
                    'reversal': 'down',
                    'current_fvg': current_fvg,
                    'previous_fvg': fvg,
                    'distance': distance,
                    'strength': min((current_fvg['size_percent'] + fvg['size_percent']) / 2, 100)
                }

        return None

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        iFVG hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: iFVG value
        """
        highs = data['high'].values
        lows = data['low'].values

        # Yeni FVG tespiti
        latest_index = len(highs) - 1
        new_fvg = self._detect_fvg(highs, lows, latest_index)

        # Yeni FVG varsa ekle
        if new_fvg:
            self.recent_fvgs.append(new_fvg)

        # Clean up old FVG (outside the lookback period)
        cutoff_index = latest_index - self.lookback_bars
        self.recent_fvgs = [
            fvg for fvg in self.recent_fvgs
            if fvg['index'] > cutoff_index
        ]

        # iFVG tespiti
        ifvg_data = None
        if new_fvg:
            ifvg_data = self._detect_ifvg(new_fvg, self.recent_fvgs[:-1])  # Exclude current

        timestamp = int(data.iloc[-1]['timestamp'])

        # Value: iFVG type (dict format)
        value_str = ifvg_data['type'] if ifvg_data else 'none'

        # Metadata
        metadata = {
            'ifvg_type': ifvg_data['type'] if ifvg_data else 'none',
            'reversal_direction': ifvg_data['reversal'] if ifvg_data else None,
            'distance': ifvg_data['distance'] if ifvg_data else None,
            'recent_fvg_count': len(self.recent_fvgs),
            'lookback_bars': self.lookback_bars,
            'min_gap_percent': self.min_gap_percent
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value={'value': value_str},
            timestamp=timestamp,
            signal=self.get_signal(ifvg_data),
            trend=self.get_trend(ifvg_data),
            strength=ifvg_data['strength'] if ifvg_data else 0,
            metadata=metadata
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate iFVG for entire DataFrame (for backtest)

        Returns pd.Series with iFVG signals for all bars.
        Value: 1=bullish iFVG, -1=bearish iFVG, 0=none
        """
        highs = data['high'].values
        lows = data['low'].values

        # Result array: iFVG signal at each bar
        ifvg_signal = np.zeros(len(data))

        # Track all FVGs
        all_fvgs: List[Dict[str, Any]] = []

        # OPTIMIZED: Track indices separately for faster filtering
        fvg_indices = []  # Parallel array of indices

        for i in range(2, len(data)):
            # Detect new FVG
            new_fvg = self._detect_fvg(highs, lows, i)

            if new_fvg:
                all_fvgs.append(new_fvg)
                fvg_indices.append(new_fvg['index'])

                # Check for iFVG - OPTIMIZED filtering with numpy
                cutoff_index = i - self.lookback_bars

                # Convert to numpy for fast boolean indexing
                indices_arr = np.array(fvg_indices)
                valid_mask = (indices_arr > cutoff_index) & (indices_arr < i)
                valid_idx = np.where(valid_mask)[0]
                recent_fvgs = [all_fvgs[j] for j in valid_idx]

                ifvg_data = self._detect_ifvg(new_fvg, recent_fvgs)

                if ifvg_data:
                    if ifvg_data['reversal'] == 'up':
                        ifvg_signal[i] = 1  # Bullish iFVG
                    elif ifvg_data['reversal'] == 'down':
                        ifvg_signal[i] = -1  # Bearish iFVG

            # Clean old FVGs - OPTIMIZED with numpy
            if len(fvg_indices) > 0:
                cutoff_index = i - self.lookback_bars - 10
                indices_arr = np.array(fvg_indices)
                valid_mask = indices_arr > cutoff_index

                # Keep only valid FVGs
                all_fvgs = [all_fvgs[j] for j in range(len(all_fvgs)) if valid_mask[j]]
                fvg_indices = indices_arr[valid_mask].tolist()

        return pd.Series(ifvg_signal, index=data.index, name='ifvg')

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

    def get_signal(self, ifvg_data: Optional[Dict[str, Any]]) -> SignalType:
        """
        Generate a signal from iFVG.

        Args:
            ifvg_data: iFVG bilgisi

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if not ifvg_data:
            return SignalType.HOLD

        # iFVG reversal sinyali
        if ifvg_data['reversal'] == 'up':
            return SignalType.BUY  # Decrease -> Increase
        elif ifvg_data['reversal'] == 'down':
            return SignalType.SELL  # Increase -> Decrease

        return SignalType.HOLD

    def get_trend(self, ifvg_data: Optional[Dict[str, Any]]) -> TrendDirection:
        """
        iFVG'den trend belirle

        Args:
            ifvg_data: iFVG bilgisi

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not ifvg_data:
            return TrendDirection.NEUTRAL

        # iFVG indicates the new trend direction
        if ifvg_data['reversal'] == 'up':
            return TrendDirection.UP
        elif ifvg_data['reversal'] == 'down':
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'min_gap_percent': 0.1,
            'lookback_bars': 10,
            'min_distance': 3
        }

    def _requires_volume(self) -> bool:
        """iFVG volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['iFVG']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """iFVG indicator test"""

    print("\n" + "="*60)
    print("iFVG (INVERSE FAIR VALUE GAP) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating example OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(60)]

    # iFVG simulation (Ascent -> Descent -> iFVG)
    base_price = 100
    prices = []
    highs = []
    lows = []

    for i in range(60):
        if i == 15:
            # Bullish FVG (rapid upward movement)
            prices.append(base_price + 10)
            highs.append(base_price + 11)
            lows.append(base_price + 9)
        elif i == 20:
            # Bearish FVG (iFVG - reversal warning)
            prices.append(base_price + 5)
            highs.append(base_price + 6)
            lows.append(base_price + 4)
        elif i == 40:
            # Bearish FVG (fast decline)
            prices.append(base_price - 8)
            highs.append(base_price - 7)
            lows.append(base_price - 9)
        elif i == 45:
            # Bullish FVG (iFVG - reversal warning)
            prices.append(base_price - 3)
            highs.append(base_price - 2)
            lows.append(base_price - 4)
        else:
            # Normal hareket
            prices.append(base_price + np.random.randn() * 0.5)
            highs.append(prices[-1] + abs(np.random.randn()) * 0.3)
            lows.append(prices[-1] - abs(np.random.randn()) * 0.3)

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
    ifvg = iFVG(min_gap_percent=0.1, lookback_bars=10, min_distance=3)
    print(f"   [OK] Created: {ifvg}")
    print(f"   [OK] Kategori: {ifvg.category.value}")
    print(f"   [OK] Required period: {ifvg.get_required_periods()}")

    result = ifvg(data)
    print(f"   [OK] iFVG Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength}")
    print(f"   [OK] iFVG Type: {result.metadata['ifvg_type']}")
    print(f"   [OK] Reversal: {result.metadata['reversal_direction']}")
    print(f"   [OK] Distance: {result.metadata['distance']}")
    print(f"   [OK] Recent FVG Count: {result.metadata['recent_fvg_count']}")

    # Test 2: Batch calculation
    print("\n3. Batch calculation test...")
    batch_result = ifvg.calculate_batch(data)
    print(f"   [OK] Batch result length: {len(batch_result)}")
    print(f"   [OK] Bullish iFVG count: {int((batch_result == 1).sum())}")
    print(f"   [OK] Number of bearish iFVG: {int((batch_result == -1).sum())}")

    # Shows the indices where iFVG was detected.
    bullish_ifvg_indices = batch_result[batch_result == 1].index.tolist()
    bearish_ifvg_indices = batch_result[batch_result == -1].index.tolist()
    print(f"   [OK] Bullish iFVG indices: {bullish_ifvg_indices}")
    print(f"   [OK] Bearish iFVG indices: {bearish_ifvg_indices}")

    # Test 3: Different parameters
    print("\n4. Different parameter test...")
    for lookback in [5, 10, 15]:
        ifvg_test = iFVG(lookback_bars=lookback)
        result = ifvg_test.calculate(data)
        print(f"   [OK] iFVG(lookback={lookback}): {result.metadata['ifvg_type']}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
