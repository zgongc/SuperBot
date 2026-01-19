"""
indicators/structure/orderblocks.py - Order Blocks

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Order Blocks - Smart Money Concepts
    Detects the traces left by corporate orders.

    Order Block Nedir:
    - The region of the last opposing candlestick before a strong price movement.
    - The area where smart money places orders.
    - Creates strong support/resistance.

Formula:
    Bullish Order Block:
    1. Detect a strong upward movement (above the threshold)
    2. Find the last downward candle before this movement
    3. The low-high range of that candle = Order Block

    Bearish Order Block:
    1. Detect a strong downward movement.
    2. Find the last upward candle before this movement.
    3. The high-low range of that candle = Order Block.

    Test & Validation:
    - When the price returns to the OB, it usually reacts.
    - If it is broken, it loses its validity.

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


class OrderBlocks(BaseIndicator):
    """
    Order Blocks (OB)

    It identifies the areas affected by institutional orders.
    It is used as strong support/resistance levels.

    Args:
        strength_threshold: Strength threshold for strong movements (% change) (default: 1.0)
        max_blocks: Maximum number of active blocks (default: 5)
        lookback: Lookback period (default: 20)
    """

    def __init__(
        self,
        strength_threshold: float = 1.0,
        max_blocks: int = 5,
        lookback: int = 20,
        logger=None,
        error_handler=None
    ):
        self.strength_threshold = strength_threshold
        self.max_blocks = max_blocks
        self.lookback = lookback

        super().__init__(
            name='orderblocks',
            category=IndicatorCategory.STRUCTURE,
            indicator_type=IndicatorType.ZONES,
            params={
                'strength_threshold': strength_threshold,
                'max_blocks': max_blocks,
                'lookback': lookback
            },
            logger=logger,
            error_handler=error_handler
        )

        # State: Follow active Order Blocks
        self.active_blocks: List[Dict[str, Any]] = []

    def get_required_periods(self) -> int:
        """Minimum required number of periods"""
        return self.lookback + 5

    def validate_params(self) -> bool:
        """Validate parameters"""
        if self.strength_threshold <= 0:
            raise InvalidParameterError(
                self.name, 'strength_threshold', self.strength_threshold,
                "Strength threshold must be positive"
            )
        if self.max_blocks < 1:
            raise InvalidParameterError(
                self.name, 'max_blocks', self.max_blocks,
                "Max blocks must be positive"
            )
        if self.lookback < 5:
            raise InvalidParameterError(
                self.name, 'lookback', self.lookback,
                "Lookback must be at least 5"
            )
        return True

    def _is_bullish_candle(self, open_price: float, close_price: float) -> bool:
        """Mum bullish mi?"""
        return close_price > open_price

    def _is_bearish_candle(self, open_price: float, close_price: float) -> bool:
        """Mum bearish mi?"""
        return close_price < open_price

    def _calculate_candle_change(self, open_price: float, close_price: float) -> float:
        """Candle change percentage"""
        if open_price == 0:
            return 0.0
        return ((close_price - open_price) / open_price) * 100

    def _detect_strong_moves(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect strong price movements.

        Args:
            opens: Array of open prices
            closes: Array of close prices
            highs: Array of high prices
            lows: Array of low prices

        Returns:
            List[Dict]: Powerful movements
        """
        strong_moves = []

        # Filter by users who have interacted within the last lookback period.
        start_idx = max(0, len(closes) - self.lookback)

        for i in range(start_idx, len(closes)):
            change_percent = self._calculate_candle_change(opens[i], closes[i])

            # Strong upward trend
            if change_percent >= self.strength_threshold:
                strong_moves.append({
                    'type': 'bullish',
                    'index': i,
                    'change_percent': change_percent,
                    'high': highs[i],
                    'low': lows[i]
                })

            # Strong decline
            elif change_percent <= -self.strength_threshold:
                strong_moves.append({
                    'type': 'bearish',
                    'index': i,
                    'change_percent': change_percent,
                    'high': highs[i],
                    'low': lows[i]
                })

        return strong_moves

    def _find_order_block(
        self,
        move: Dict[str, Any],
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts the order block from a strong movement.

        Args:
            move: Strong movement information
            opens, closes, highs, lows: Price sequences

        Returns:
            Dict: Order block bilgisi
        """
        move_index = move['index']
        move_type = move['type']

        # Go backward
        for i in range(move_index - 1, max(0, move_index - 10), -1):
            is_bullish = self._is_bullish_candle(opens[i], closes[i])
            is_bearish = self._is_bearish_candle(opens[i], closes[i])

            # Last bearish candle for a bullish move
            if move_type == 'bullish' and is_bearish:
                return {
                    'type': 'bullish',
                    'top': highs[i],
                    'bottom': lows[i],
                    'index': i,
                    'move_index': move_index,
                    'strength': move['change_percent'],
                    'status': 'active',
                    'test_count': 0
                }

            # Last bullish candle for a bearish move
            if move_type == 'bearish' and is_bullish:
                return {
                    'type': 'bearish',
                    'top': highs[i],
                    'bottom': lows[i],
                    'index': i,
                    'move_index': move_index,
                    'strength': abs(move['change_percent']),
                    'status': 'active',
                    'test_count': 0
                }

        return None

    def _update_block_status(
        self,
        block: Dict[str, Any],
        current_high: float,
        current_low: float,
        current_close: float
    ) -> Dict[str, Any]:
        """
        Update the order block status.

        Args:
            block: Order block bilgisi
            current_high, current_low, current_close: Current prices

        Returns:
            Dict: Updated block
        """
        # Did it enter the price block?
        in_block = (current_low <= block['top'] and current_high >= block['bottom'])

        if in_block:
            block['test_count'] += 1

        # Is the block broken?
        if block['type'] == 'bullish':
            # Bullish OB below close -> broken
            if current_close < block['bottom']:
                block['status'] = 'broken'
        else:
            # Bearish closing above OB -> broken
            if current_close > block['top']:
                block['status'] = 'broken'

        return block

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Order Blocks hesapla

        Args:
            data: OHLCV DataFrame

        Returns:
            IndicatorResult: Order Block zones
        """
        opens = data['open'].values
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values

        # Detect strong movements
        strong_moves = self._detect_strong_moves(opens, closes, highs, lows)

        # Extract order block from each strong movement
        new_blocks = []
        for move in strong_moves:
            ob = self._find_order_block(move, opens, closes, highs, lows)
            if ob:
                # Duplicate check
                is_duplicate = any(
                    existing['index'] == ob['index']
                    for existing in self.active_blocks
                )
                if not is_duplicate:
                    new_blocks.append(ob)

        # Add new blocks
        self.active_blocks.extend(new_blocks)

        # Update the status of existing blocks
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]

        for block in self.active_blocks:
            self._update_block_status(block, current_high, current_low, current_close)

        # Remove broken blocks
        self.active_blocks = [
            block for block in self.active_blocks
            if block['status'] == 'active'
        ]

        # Apply the maximum number of blocks (keep the most powerful ones)
        if len(self.active_blocks) > self.max_blocks:
            self.active_blocks.sort(key=lambda x: x['strength'], reverse=True)
            self.active_blocks = self.active_blocks[:self.max_blocks]

        timestamp = int(data.iloc[-1]['timestamp'])

        # Value: List of active order blocks
        zones = [
            {
                'type': block['type'],
                'top': round(block['top'], 2),
                'bottom': round(block['bottom'], 2),
                'strength': round(block['strength'], 2),
                'test_count': block['test_count'],
                'status': block['status']
            }
            for block in self.active_blocks
        ]

        # Warmup buffer for update() method
        self.warmup_buffer(data)

        return IndicatorResult(
            value=zones,
            timestamp=timestamp,
            signal=self.get_signal(zones, current_close),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),  # Her block 20 puan
            metadata={
                'total_blocks': len(zones),
                'bullish_blocks': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_blocks': len([z for z in zones if z['type'] == 'bearish']),
                'strength_threshold': self.strength_threshold,
                'max_blocks': self.max_blocks
            }
        )

    def update(self, candle: dict, symbol: str = None) -> IndicatorResult:
        """
        Incremental update (Real-time)
        
        Args:
            candle: Yeni mum verisi (dict)
            
        Returns:
            IndicatorResult: Current Order Blocks
        """
        # Buffer management
        if not hasattr(self, '_open_buffer'):
            from collections import deque
            max_len = self.get_required_periods() + 50
            self._open_buffer = deque(maxlen=max_len)
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

        self._open_buffer.append(open_val)
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
                metadata={'total_blocks': 0}
            )
            
        # Calculation
        opens = np.array(self._open_buffer)
        closes = np.array(self._close_buffer)
        highs = np.array(self._high_buffer)
        lows = np.array(self._low_buffer)
        
        # Detect strong movements
        strong_moves = self._detect_strong_moves(opens, closes, highs, lows)
        
        # Extract order block from each strong movement
        new_blocks = []
        for move in strong_moves:
            ob = self._find_order_block(move, opens, closes, highs, lows)
            if ob:
                # Duplicate check
                is_duplicate = any(
                    existing['index'] == ob['index']
                    for existing in self.active_blocks
                )
                if not is_duplicate:
                    new_blocks.append(ob)
        
        # Add new blocks
        self.active_blocks.extend(new_blocks)
        
        # Update the status of existing blocks
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        
        for block in self.active_blocks:
            self._update_block_status(block, current_high, current_low, current_close)
        
        # Remove broken blocks
        self.active_blocks = [
            block for block in self.active_blocks
            if block['status'] == 'active'
        ]
        
        # Apply the maximum number of blocks
        if len(self.active_blocks) > self.max_blocks:
            self.active_blocks.sort(key=lambda x: x['strength'], reverse=True)
            self.active_blocks = self.active_blocks[:self.max_blocks]
        
        # Value: Active order blocks
        zones = [
            {
                'type': block['type'],
                'top': round(block['top'], 2),
                'bottom': round(block['bottom'], 2),
                'strength': round(block['strength'], 2),
                'test_count': block['test_count'],
                'status': block['status']
            }
            for block in self.active_blocks
        ]

        return IndicatorResult(
            value=zones,
            timestamp=timestamp_val,
            signal=self.get_signal(zones, current_close),
            trend=self.get_trend(zones),
            strength=min(len(zones) * 20, 100),
            metadata={
                'total_blocks': len(zones),
                'bullish_blocks': len([z for z in zones if z['type'] == 'bullish']),
                'bearish_blocks': len([z for z in zones if z['type'] == 'bearish']),
                'strength_threshold': self.strength_threshold,
                'max_blocks': self.max_blocks
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

    def get_signal(self, zones: List[Dict[str, Any]], current_price: float) -> SignalType:
        """
        Generate signals from Order Blocks.

        Args:
            zones: Order block zones
            current_price: Current price

        Returns:
            SignalType: BUY, SELL or HOLD
        """
        if not zones:
            return SignalType.HOLD

        # Is the price close to an Order Block zone?
        for zone in zones:
            # Zone center
            zone_mid = (zone['top'] + zone['bottom']) / 2
            distance = abs(current_price - zone_mid)
            zone_size = zone['top'] - zone['bottom']

            # If it is within the zone or within 2%
            if distance <= zone_size * 0.5 or (distance / current_price) * 100 < 2.0:
                if zone['type'] == 'bullish':
                    return SignalType.BUY  # Bullish OB'de destek bekle
                elif zone['type'] == 'bearish':
                    return SignalType.SELL  # Expect resistance in a bearish Order Block

        return SignalType.HOLD

    def get_trend(self, zones: List[Dict[str, Any]]) -> TrendDirection:
        """
        Order Block'lardan trend belirle

        Args:
            zones: Order block zones

        Returns:
            TrendDirection: UP, DOWN or NEUTRAL
        """
        if not zones:
            return TrendDirection.NEUTRAL

        bullish_count = len([z for z in zones if z['type'] == 'bullish'])
        bearish_count = len([z for z in zones if z['type'] == 'bearish'])

        # Strong blocks carry more weight
        bullish_strength = sum(
            z['strength'] for z in zones if z['type'] == 'bullish'
        )
        bearish_strength = sum(
            z['strength'] for z in zones if z['type'] == 'bearish'
        )

        if bullish_strength > bearish_strength * 1.2:
            return TrendDirection.UP
        elif bearish_strength > bullish_strength * 1.2:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _get_default_params(self) -> dict:
        """Default parameters"""
        return {
            'strength_threshold': 1.0,
            'max_blocks': 5,
            'lookback': 20
        }

    def _requires_volume(self) -> bool:
        """Order Blocks volume gerektirmez"""
        return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ['OrderBlocks']


# ============================================================================
# USAGE EXAMPLE (TEST)
# ============================================================================

if __name__ == "__main__":
    """Order Blocks indicator test"""

    print("\n" + "="*60)
    print("ORDER BLOCKS TEST")
    print("="*60 + "\n")

    # Create example data
    print("1. Creating sample OHLCV data...")
    np.random.seed(42)
    timestamps = [1697000000000 + i * 60000 for i in range(50)]

    # Powerful moving price simulation
    base_price = 100
    prices = []
    opens = []
    highs = []
    lows = []

    for i in range(50):
        if i == 15:
            # Strong uptrend (creates an Order Block)
            open_p = base_price
            close_p = base_price + 2.5  # %2.5 increase
            opens.append(open_p)
            prices.append(close_p)
            highs.append(close_p + 0.2)
            lows.append(open_p - 0.1)
            base_price = close_p
        elif i == 35:
            # Strong downtrend (creates an Order Block)
            open_p = base_price
            close_p = base_price - 2.0  # %2 decrease
            opens.append(open_p)
            prices.append(close_p)
            highs.append(open_p + 0.1)
            lows.append(close_p - 0.2)
            base_price = close_p
        else:
            # Normal hareket
            open_p = base_price
            close_p = base_price + np.random.randn() * 0.3
            opens.append(open_p)
            prices.append(close_p)
            highs.append(max(open_p, close_p) + abs(np.random.randn()) * 0.2)
            lows.append(min(open_p, close_p) - abs(np.random.randn()) * 0.2)
            base_price = close_p

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000 + np.random.randint(0, 500) for _ in prices]
    })

    print(f"   [OK] {len(data)} candles created")
    print(f"   [OK] Price range: {min(prices):.2f} -> {max(prices):.2f}")

    # Test 1: Basic calculation
    print("\n2. Basic calculation test...")
    ob = OrderBlocks(strength_threshold=1.0, max_blocks=5, lookback=20)
    print(f"   [OK] Created: {ob}")
    print(f"   [OK] Kategori: {ob.category.value}")
    print(f"   [OK] Required period: {ob.get_required_periods()}")

    result = ob(data)
    print(f"   [OK] Total Blocks: {result.metadata['total_blocks']}")
    print(f"   [OK] Bullish Block: {result.metadata['bullish_blocks']}")
    print(f"   [OK] Bearish Block: {result.metadata['bearish_blocks']}")
    print(f"   [OK] Signal: {result.signal.value}")
    print(f"   [OK] Trend: {result.trend.name}")
    print(f"   [OK] Power: {result.strength}")

    # Test 2: Block details
    print("\n3. Block details...")
    if result.value:
        for i, block in enumerate(result.value[:3]):
            print(f"   [OK] Block #{i+1}:")
            print(f"       - Tip: {block['type']}")
            print(f"       - Top: {block['top']:.2f}")
            print(f"       - Bottom: {block['bottom']:.2f}")
            print(f"       - Power: {block['strength']:.2f}%")
            print(f"       - Test: {block['test_count']} kez")
            print(f"       - Status: {block['status']}")
    else:
        print("   [OK] Active block not found")

    # Test 3: Different parameters
    print("\n4. Different parameter test...")
    for threshold in [0.5, 1.0, 1.5]:
        ob_test = OrderBlocks(strength_threshold=threshold)
        result = ob_test.calculate(data)
        print(f"   [OK] OB(threshold={threshold}): {result.metadata['total_blocks']} blocks")

    # Test 4: Statistics
    print("\n5. Statistical test...")
    stats = ob.statistics
    print(f"   [OK] Calculation count: {stats['calculation_count']}")
    print(f"   [OK] Error count: {stats['error_count']}")

    # Test 5: Metadata
    print("\n6. Metadata testi...")
    metadata = ob.metadata
    print(f"   [OK] Name: {metadata.name}")
    print(f"   [OK] Kategori: {metadata.category.value}")
    print(f"   [OK] Tip: {metadata.indicator_type.value}")
    print(f"   [OK] Min periyot: {metadata.min_periods}")
    print(f"   [OK] Volume required: {metadata.requires_volume}")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60 + "\n")
