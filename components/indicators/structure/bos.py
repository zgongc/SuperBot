"""
indicators/structure/bos.py - Break of Structure

Version: 3.0.0
Date: 2025-12-24
Author: SuperBot Team

Description:
    BOS (Break of Structure) - Smart Money Concepts
    Detects the breaking points of the market structure.

    BOS is:
    - Uprising trend: Breaking the previous swing high.
    - Declining trend: Breaking the previous swing low.
    - Indicates trend continuation (strong movement).

Formula:
    1. Swing High/Low detection (uses SwingPoints - compatible with TradingView)
    2. Son swing high/low seviyelerini takip et
    3. If the price exceeds the level -> BOS

    Bullish BOS: Close > Previous Swing High
    Bearish BOS: Close < Previous Swing Low

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - SwingPoints (../support_resistance/swing_points.py)
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
# Use SwingPoints with lazy import (to prevent circular import)


class BOS(BaseIndicator):
    """
    Break of Structure (BOS)

    Detects the breaking points of the market structure.
    Provides a signal for trend continuation.

    IMPORTANT: BOS and CHoCH are MUTUALLY EXCLUSIVE for the same break event!
    - BOS: Break in the SAME direction as current trend (continuation)
    - CHoCH: Break in OPPOSITE direction of current trend (reversal)

    For example:
    - Uptrend + bullish break (above swing high) = BOS (trend continuation)
    - Downtrend + bullish break (above swing high) = CHoCH (trend reversal)
    - Downtrend + bearish break (below swing low) = BOS (trend continuation)
    - Uptrend + bearish break (below swing low) = CHoCH (trend reversal)

    Args:
        left_bars: Number of bars on the left side (default: 5)
        right_bars: Number of bars on the right side (default: 5)
        max_levels: Maximum number of levels (default: 3)
        trend_strength: Trend strength threshold (default: 3) - for trend detection
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        max_levels: int = 3,
        trend_strength: int = 3,
        logger=None,
        error_handler=None
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.max_levels = max_levels
        self.trend_strength = trend_strength

        # Create SwingPoints with lazy import
        self._swing_points = None

        super().__init__(
            name='bos',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.SINGLE_VALUE,
            params={
                'left_bars': left_bars,
                'right_bars': right_bars,
                'max_levels': max_levels,
                'trend_strength': trend_strength
            },
            logger=logger,
            error_handler=error_handler
        )

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
        if self.max_levels < 1:
            raise InvalidParameterError(
                self.name, 'max_levels', self.max_levels,
                "Max levels must be positive"
            )
        return True

    def _get_swing_points(self):
        """Lazy load the SwingPoints instance."""
        if self._swing_points is None:
            import sys
            import os
            # Add 'components/indicators' as 'indicators' (for old import compatibility)
            components_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if components_path not in sys.path:
                sys.path.insert(0, components_path)

            from indicators.support_resistance.swing_points import SwingPoints
            self._swing_points = SwingPoints(
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                lookback=50
            )
        return self._swing_points

    def _find_swings(self, data: pd.DataFrame) -> tuple:
        """
        Detect swing high/low points using SwingPoints (with alternating filter).

        SwingPoints uses a TradingView compatible pivot algorithm:
        - Left side: strictly greater/less
        - Right side: greater/less or equal (the first pivot created wins)
        - Alternating filter: High-Low-High-Low pattern is mandatory

        Args:
            data: OHLCV DataFrame

        Returns:
            tuple: (swing_highs, swing_lows) - Each is in List[Dict] format
        """
        swing_points = self._get_swing_points()
        swing_df = swing_points.calculate_batch(data)

        # First, collect all swings (index, type, value)
        raw_swings = []
        for i in range(len(swing_df)):
            if not np.isnan(swing_df['swing_high'].iloc[i]):
                raw_swings.append((i, 'high', swing_df['swing_high'].iloc[i]))
            if not np.isnan(swing_df['swing_low'].iloc[i]):
                raw_swings.append((i, 'low', swing_df['swing_low'].iloc[i]))

        # Sort by index (chronological)
        raw_swings.sort(key=lambda x: x[0])

        # Alternating swing'leri filtrele (High-Low-High-Low pattern)
        filtered_swings = []
        for idx, swing_type, value in raw_swings:
            if not filtered_swings:
                filtered_swings.append((idx, swing_type, value))
            elif filtered_swings[-1][1] != swing_type:
                # Different type - add
                filtered_swings.append((idx, swing_type, value))
            else:
                # Same type - keep the more extreme one
                last_idx, last_type, last_value = filtered_swings[-1]
                if swing_type == 'high' and value > last_value:
                    filtered_swings[-1] = (idx, swing_type, value)
                elif swing_type == 'low' and value < last_value:
                    filtered_swings[-1] = (idx, swing_type, value)

        # Create separate lists from the filtered swings
        swing_highs = []
        swing_lows = []
        for idx, swing_type, value in filtered_swings:
            if swing_type == 'high':
                swing_highs.append({'index': idx, 'value': value})
            else:
                swing_lows.append({'index': idx, 'value': value})

        return swing_highs, swing_lows

    def _detect_trend(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> str:
        """
        Detect the current trend direction.

        SMC Approach (same logic as structure_detector.py):
        - Uptrend: Higher High OR Higher Low (any bullish structure)
        - Downtrend: Lower Low OR Lower High (any bearish structure)
        - More flexible - only looks at the last 2 swings.

        Args:
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar

        Returns:
            str: 'uptrend', 'downtrend', 'ranging'
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'ranging'

        # Look at the last 2-3 swings (like in structure_detector.py)
        last_highs = [h['value'] for h in swing_highs[-3:]]
        last_lows = [l['value'] for l in swing_lows[-3:]]

        # Check the latest swing direction
        last_high_direction = 'higher' if last_highs[-1] > last_highs[-2] else 'lower'
        last_low_direction = 'higher' if last_lows[-1] > last_lows[-2] else 'lower'

        # Uptrend: HH and HL (classic) OR just HH with equal/higher low
        # Downtrend: LL and LH (classic) OR just LL with equal/lower high
        if last_high_direction == 'higher' and last_low_direction == 'higher':
            return 'uptrend'
        elif last_high_direction == 'lower' and last_low_direction == 'lower':
            return 'downtrend'
        elif last_high_direction == 'higher':
            # HH but not HL - still bullish bias
            return 'uptrend'
        elif last_low_direction == 'lower':
            # LL but not LH - still bearish bias
            return 'downtrend'

        return 'ranging'

    def _detect_bos_at_index(
        self,
        index: int,
        closes: np.ndarray,
        recent_highs: List[Dict[str, Any]],
        recent_lows: List[Dict[str, Any]],
        current_trend: str,
        broken_highs: set,
        broken_lows: set
    ) -> Optional[Dict[str, Any]]:
        """
        CORE DETECTION FUNCTION - Used by calculate(), calculate_batch(), update()

        Empty detection - Trend check for SINGLE BAR.

        IMPORTANT: BOS and CHoCH are MUTUALLY EXCLUSIVE!
        - BOS: Break in SAME direction as trend (continuation)
        - CHoCH: Break in OPPOSITE direction of trend (reversal)

        Args:
            index: Current bar index
            closes: Close price array
            recent_highs: Recent swing highs (number of max_levels)
            recent_lows: Recent swing lows (number of max_levels)
            current_trend: 'uptrend', 'downtrend', 'ranging'
            broken_highs: Set of already broken swing high indices
            broken_lows: Set of already broken swing low indices

        Returns:
            Dict: Empty dictionary or None

        Side Effects:
            Updates broken_highs and broken_lows sets
        """
        if index < 1 or len(closes) < 2:
            return None

        current_close = closes[index]
        prev_close = closes[index - 1]

        # ===== BULLISH BREAK CHECK =====
        # Bullish BOS: NOT downtrend + break above swing high
        # (downtrend + bullish = CHoCH, handled by choch.py)
        for swing in reversed(recent_highs[-self.max_levels:]):
            swing_idx = swing['index']
            swing_val = swing['value']

            if swing_idx in broken_highs:
                continue

            if swing_idx >= index:
                continue

            # Check if this is a NEW break (crossover)
            if current_close > swing_val and prev_close <= swing_val:
                broken_highs.add(swing_idx)
                # Only signal BOS if NOT in downtrend (downtrend = CHoCH)
                if current_trend != 'downtrend':
                    return {
                        'type': 'bullish',
                        'level': swing_val,
                        'index': swing_idx,
                        'broken_at': index
                    }
                else:
                    # It's a break but CHoCH, not BOS
                    return None
            elif current_close > swing_val:
                # Already broken before, mark it
                broken_highs.add(swing_idx)

        # ===== BEARISH BREAK CHECK =====
        # Bearish BOS: NOT uptrend + break below swing low
        # (uptrend + bearish = CHoCH, handled by choch.py)
        for swing in reversed(recent_lows[-self.max_levels:]):
            swing_idx = swing['index']
            swing_val = swing['value']

            if swing_idx in broken_lows:
                continue

            if swing_idx >= index:
                continue

            # Check if this is a NEW break (crossunder)
            if current_close < swing_val and prev_close >= swing_val:
                broken_lows.add(swing_idx)
                # Only signal BOS if NOT in uptrend (uptrend = CHoCH)
                if current_trend != 'uptrend':
                    return {
                        'type': 'bearish',
                        'level': swing_val,
                        'index': swing_idx,
                        'broken_at': index
                    }
                else:
                    # It's a break but CHoCH, not BOS
                    return None
            elif current_close < swing_val:
                # Already broken before, mark it
                broken_lows.add(swing_idx)

        return None

    def _detect_bos(
        self,
        closes: np.ndarray,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Empty detection - For the last bar (wrapper for calculate())

        Uses _detect_bos_at_index() with trend detection.

        Args:
            closes: Close price array
            swing_highs: Swing high'lar
            swing_lows: Swing low'lar

        Returns:
            Dict: Empty dictionary or None
        """
        if len(closes) < 2:
            return None

        index = len(closes) - 1

        # Detect trend
        current_trend = self._detect_trend(swing_highs, swing_lows)

        # Build broken sets by scanning history
        broken_highs = set()
        broken_lows = set()

        # Recent swings for detection
        recent_highs = swing_highs[-self.max_levels:] if swing_highs else []
        recent_lows = swing_lows[-self.max_levels:] if swing_lows else []

        # Scan history to find already broken swings
        for i in range(1, index):
            current_close = closes[i]
            prev_close = closes[i - 1]

            for swing in recent_highs:
                if swing['index'] < i and swing['index'] not in broken_highs:
                    if current_close > swing['value']:
                        broken_highs.add(swing['index'])

            for swing in recent_lows:
                if swing['index'] < i and swing['index'] not in broken_lows:
                    if current_close < swing['value']:
                        broken_lows.add(swing['index'])

        # Detect BOS at current index
        bos_result = self._detect_bos_at_index(
            index=index,
            closes=closes,
            recent_highs=recent_highs,
            recent_lows=recent_lows,
            current_trend=current_trend,
            broken_highs=broken_highs,
            broken_lows=broken_lows
        )

        if bos_result:
            return bos_result

        # If BOS is not present (no breakout) - return the pending last swing level.
        if swing_highs and swing_lows:
            current_close = closes[-1]
            last_high = swing_highs[-1]['value']
            last_low = swing_lows[-1]['value']

            # Which level is closer?
            if abs(current_close - last_high) < abs(current_close - last_low):
                return {
                    'type': 'pending_bullish',
                    'level': last_high,
                    'index': swing_highs[-1]['index'],
                    'broken_at': None
                }
            else:
                return {
                    'type': 'pending_bearish',
                    'level': last_low,
                    'index': swing_lows[-1]['index'],
                    'broken_at': None
                }

        return None

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        BOS hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: BOS value
                - value: Signal value (1=bullish, -1=bearish, 0=none)
                - metadata: BOS level and details
        """
        closes = data['close'].values

        # Swing High/Low tespiti (SwingPoints kullanarak)
        swing_highs, swing_lows = self._find_swings(data)

        # BOS tespiti
        bos_data = self._detect_bos(closes, swing_highs, swing_lows)

        timestamp = int(data.iloc[-1]['timestamp'])

        # Value: Signal (1=bullish, -1=bearish, 0=none/pending)
        # Consistent with calculate_batch()!
        bos_type = bos_data.get('type', 'none') if bos_data else 'none'
        if bos_type == 'bullish':
            value = 1
        elif bos_type == 'bearish':
            value = -1
        else:
            value = 0  # 'none', 'pending_bullish', 'pending_bearish'

        # Metadata: All BOS levels and price information
        metadata = {
            'swing_highs': [{'level': s['value'], 'index': s['index']} for s in swing_highs[-self.max_levels:]],
            'swing_lows': [{'level': s['value'], 'index': s['index']} for s in swing_lows[-self.max_levels:]],
            'bos_type': bos_type,
            'bos_level': bos_data['level'] if bos_data else None,
            'left_bars': self.left_bars,
            'right_bars': self.right_bars
        }

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=value,
            timestamp=timestamp,
            signal=self.get_signal(bos_data),
            trend=self.get_trend(bos_data),
            strength=self._calculate_strength(bos_data, closes[-1]) if bos_data else 0,
            metadata=metadata
        )

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate BOS for entire DataFrame (for backtest)

        Uses _detect_bos_at_index() core function for each bar.

        Returns pd.Series with BOS signals for all bars.
        Value: 1=bullish BOS, -1=bearish BOS, 0=none

        IMPORTANT: BOS and CHoCH are MUTUALLY EXCLUSIVE!
        - BOS: Break in SAME direction as trend (continuation)
          - Uptrend + bullish break (above swing high) = BOS
          - Downtrend + bearish break (below swing low) = BOS
          - Ranging + any break = BOS (no trend to reverse)
        - CHoCH: Break in OPPOSITE direction of trend (reversal)
          - Downtrend + bullish break = CHoCH (not BOS!)
          - Uptrend + bearish break = CHoCH (not BOS!)
        """
        closes = data['close'].values

        # Result array: store BOS signal at each bar (1=bullish, -1=bearish, 0=none)
        bos_signal = np.zeros(len(data))

        # Find all swing highs/lows upfront (SwingPoints kullanarak)
        swing_highs, swing_lows = self._find_swings(data)

        # Convert to dict for fast lookup by index
        swing_high_dict = {s['index']: s['value'] for s in swing_highs}
        swing_low_dict = {s['index']: s['value'] for s in swing_lows}

        # Track recent swing levels
        recent_highs = []
        recent_lows = []

        # Track which swing levels have been broken (to detect NEW breaks only)
        broken_highs = set()  # swing indices that have been broken
        broken_lows = set()   # swing indices that have been broken

        # State: keep previous trend
        current_trend = 'ranging'

        start_idx = self.left_bars + self.right_bars

        for i in range(start_idx, len(data)):
            # Update recent swings
            if i in swing_high_dict:
                recent_highs.append({'index': i, 'value': swing_high_dict[i]})
                if len(recent_highs) > self.max_levels:
                    recent_highs = recent_highs[-self.max_levels:]

            if i in swing_low_dict:
                recent_lows.append({'index': i, 'value': swing_low_dict[i]})
                if len(recent_lows) > self.max_levels:
                    recent_lows = recent_lows[-self.max_levels:]

            # Detect trend (with state preservation)
            new_trend = self._detect_trend(recent_highs, recent_lows)
            if new_trend != 'ranging':
                current_trend = new_trend

            # Use core detection function
            bos_result = self._detect_bos_at_index(
                index=i,
                closes=closes,
                recent_highs=recent_highs,
                recent_lows=recent_lows,
                current_trend=current_trend,
                broken_highs=broken_highs,
                broken_lows=broken_lows
            )

            if bos_result:
                if bos_result['type'] == 'bullish':
                    bos_signal[i] = 1
                elif bos_result['type'] == 'bearish':
                    bos_signal[i] = -1

        return pd.Series(bos_signal, index=data.index, name='bos')

    def _calculate_strength(self, bos_data: Dict[str, Any], current_close: float) -> float:
        """
        Calculate the BOS power (0-100).

        Args:
            bos_data: BOS bilgisi
            current_close: The current close price.

        Returns:
            float: Power score
        """
        if not bos_data or not bos_data.get('broken_at'):
            return 0.0

        level = bos_data['level']
        distance = abs(current_close - level)

        # Calculate power based on breaking distance.
        strength = min((distance / level) * 1000, 100)

        return round(strength, 2)

    def warmup_buffer(self, data: pd.DataFrame, symbol: str = None) -> None:
        """
        Warmup BOS buffers with historical data

        CRITICAL: BOS uses its own buffers (_high_buffer, _low_buffer, _close_buffer)
        not BaseIndicator's _buffers. This override ensures they're properly filled.

        Args:
            data: Historical OHLCV DataFrame
            symbol: Symbol identifier (unused, for interface compatibility)
        """
        from collections import deque

        max_len = self.get_required_periods() + 50
        self._high_buffer = deque(maxlen=max_len)
        self._low_buffer = deque(maxlen=max_len)
        self._close_buffer = deque(maxlen=max_len)
        self._buffers_init = True

        # Fill buffers with historical data
        for _, row in data.tail(max_len).iterrows():
            self._high_buffer.append(row['high'])
            self._low_buffer.append(row['low'])
            self._close_buffer.append(row['close'])

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """Incremental update (Real-time)"""
        # Initialize buffers if warmup wasn't called
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
                value=0,  # No signal (consistent with calculate())
                timestamp=timestamp_val,
                signal=SignalType.HOLD,
                trend=TrendDirection.NEUTRAL,
                strength=0.0,
                metadata={'bos_type': 'none', 'bos_level': None}
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

    def get_signal(self, bos_data: Optional[Dict[str, Any]]) -> SignalType:
        """
        Generate a signal from BOS.

        Args:
            bos_data: BOS bilgisi

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if not bos_data:
            return SignalType.HOLD

        bos_type = bos_data.get('type', 'none')

        if bos_type == 'bullish':
            return SignalType.BUY
        elif bos_type == 'bearish':
            return SignalType.SELL

        return SignalType.HOLD

    def get_trend(self, bos_data: Optional[Dict[str, Any]]) -> TrendDirection:
        """
        BOS'tan trend belirle

        Args:
            bos_data: BOS bilgisi

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not bos_data:
            return TrendDirection.NEUTRAL

        bos_type = bos_data.get('type', 'none')

        if 'bullish' in bos_type:
            return TrendDirection.UP
        elif 'bearish' in bos_type:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'left_bars': 5,
            'right_bars': 5,
            'max_levels': 3,
            'trend_strength': 3
        }

    def _requires_volume(self) -> bool:
        """BOS volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['BOS']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """BOS indicator test"""

    print("\n" + "="*60)
    print("BOS (BREAK OF STRUCTURE) TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Trend simulation (rise -> fall -> rise)
    base_price = 100
    prices = []
    for i in range(50):
        if i < 15:
            # Ascent
            prices.append(base_price + i * 0.5 + np.random.randn() * 0.3)
        elif i < 35:
            # Fall
            prices.append(base_price + 7.5 - (i - 15) * 0.4 + np.random.randn() * 0.3)
        else:
            # Ascent
            prices.append(base_price - (i - 35) * 0.6 + np.random.randn() * 0.3)

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
    bos = BOS(left_bars=5, right_bars=5, max_levels=3)
    print(f"   [OK] Created: {bos}")
    print(f"   [OK] Kategori: {bos.category.value}")
    print(f"   [OK] Required period: {bos.get_required_periods()}")

    result = bos(data)
    print(f"   [OK] BOS Value: {result.value}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength}")
    print(f"   [OK] BOS Type: {result.metadata['bos_type']}")
    print(f"   [OK] Swing Highs: {len(result.metadata['swing_highs'])} items")
    print(f"   [OK] Swing Lows: {len(result.metadata['swing_lows'])} items")

    # Test 2: Show swing levels
    print("\n3. Swing seviyeleri...")
    for i, high in enumerate(result.metadata['swing_highs'][-3:]):
        print(f"   [OK] Swing High #{i+1}: {high['level']:.2f} @ index {high['index']}")
    for i, low in enumerate(result.metadata['swing_lows'][-3:]):
        print(f"   [OK] Swing Low #{i+1}: {low['level']:.2f} @ index {low['index']}")

    # Test 3: Different parameters
    print("\n4. Different parameter test...")
    for left, right in [(3, 3), (5, 5), (7, 7)]:
        bos_test = BOS(left_bars=left, right_bars=right)
        result = bos_test.calculate(data)
        print(f"   [OK] BOS({left},{right}): {result.value} | Tip: {result.metadata['bos_type']}")

    # Test 4: Statistics
    print("\n5. Statistical test...")
    stats = bos.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = bos.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
