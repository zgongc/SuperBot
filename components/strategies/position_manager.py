#!/usr/bin/env python3
"""
components/strategies/position_manager.py
SuperBot - Position Manager (NEW DESIGN)
Yazar: SuperBot Team
Tarih: 2025-12-07
Versiyon: 2.0.0

Position Manager - Manages position management and processes strategy template parameters.

IMPORTANT: This new design replaces the old, unused position_manager.py file (819 lines).
Purpose: To move duplicate inline code from BacktestEngine and TradingEngine to a central location.

Parameters (from simple_rsi.py PositionManagement):
- max_positions_per_symbol: Maximum number of positions per symbol.
- pyramiding_enabled: Is it possible to open multiple positions in the same direction?
- pyramiding_max_entries: Maximum pyramiding entry count.
- pyramiding_scale_factor: Size multiplier for subsequent entries.
- allow_hedging: Is it possible to open positions in the opposite direction (hedge)?
- position_timeout_enabled: Is position timeout active?
- position_timeout: Timeout duration (minutes).

Usage:
    pm = PositionManager(strategy, logger)

    # Check if the position can be opened.
    can_open, reason, scale = pm.can_open_position(
        symbol='BTCUSDT',
        side='LONG',
        positions=current_positions
    )

    # Timeout control
    should_close, reason = pm.check_position_timeout(position, current_timestamp)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class PositionOpenResult:
    """Position opening check result"""
    can_open: bool
    reason: str
    pyramiding_scale: float = 1.0  # Size scale factor for pyramiding
    should_close_opposite: bool = False  # Should opposite positions be closed (one-way mode)
    opposite_positions: List[Dict] = None  # List of opposite positions to be closed

    def __post_init__(self):
        if self.opposite_positions is None:
            self.opposite_positions = []


class PositionManager:
    """
    Position Manager - Handles strategy template parameters.

    Workflow:
    1. can_open_position() - Check if a position can be opened
    2. check_position_timeout() - Check timeout
    3. get_positions_for_symbol() - Get positions for a symbol

    NOTE: This class only makes decisions; it does not perform the opening/closing of positions; the engine handles that.
    """

    def __init__(
        self,
        strategy: Any,
        logger: Optional[Any] = None
    ):
        """
        Args:
            strategy: BaseStrategy instance (for position_management parameters)
            logger: Logger instance
        """
        self.strategy = strategy
        self.logger = logger

        # Cache strategy parameters
        pm = strategy.position_management
        self.max_positions_per_symbol = pm.max_positions_per_symbol
        self.pyramiding_enabled = pm.pyramiding_enabled
        self.pyramiding_max_entries = pm.pyramiding_max_entries
        self.pyramiding_scale_factor = pm.pyramiding_scale_factor
        self.allow_hedging = pm.allow_hedging
        self.position_timeout_enabled = pm.position_timeout_enabled
        self.position_timeout = pm.position_timeout  # Minutes

        if self.logger:
            self.logger.debug(
                f"PositionManager initialized: max_per_symbol={self.max_positions_per_symbol}, "
                f"pyramiding={self.pyramiding_enabled}, hedging={self.allow_hedging}"
            )

    # ========================================================================
    # MAIN METHODS - Engines use these methods
    # ========================================================================

    def can_open_position(
        self,
        symbol: str,
        side: str,
        positions: List[Dict]
    ) -> PositionOpenResult:
        """
        Check if the position can be opened.

        This method uses BacktestEngine (line 825-939) and TradingEngine (line 822-864).
        It replaces the duplicate code inside.

        Args:
            symbol: Symbol (e.g., 'BTCUSDT')
            side: 'LONG' or 'SHORT'
            positions: List of current open positions
                      Her pozisyon dict: {'symbol': str, 'side': str, ...}

        Returns:
            PositionOpenResult:
                - can_open: Can the position be opened?
                - reason: Explanation
                - pyramiding_scale: Pyramiding size multiplier (1.0 = normal)
                - should_close_opposite: Should opposite positions be closed?
                - opposite_positions: List of opposite positions to be closed

        Example:
            >>> result = pm.can_open_position('BTCUSDT', 'LONG', current_positions)
            >>> if result.can_open:
            ...     if result.should_close_opposite:
            ...         for pos in result.opposite_positions:
            ...             engine.close_position(pos)
            ...     quantity = base_quantity * result.pyramiding_scale
            ...     engine.open_position(symbol, side, quantity)
        """
        # 1. Filter existing positions for the symbol.
        positions_for_symbol = self._get_positions_for_symbol(symbol, positions)
        same_side_positions = [p for p in positions_for_symbol if p.get('side') == side]
        opposite_side = 'SHORT' if side == 'LONG' else 'LONG'
        opposite_positions = [p for p in positions_for_symbol if p.get('side') == opposite_side]

        # 2. Check the maximum positions per symbol.
        if len(positions_for_symbol) >= self.max_positions_per_symbol:
            return PositionOpenResult(
                can_open=False,
                reason=f"Max positions reached for {symbol} ({len(positions_for_symbol)}/{self.max_positions_per_symbol})"
            )

        # 3. Pyramiding control (multiple positions in the same direction)
        can_open_same_side = False
        pyramiding_scale = 1.0

        if self.pyramiding_enabled:
            # Pyramiding is active: positions can be opened in the same direction up to max_entries.
            if len(same_side_positions) < self.pyramiding_max_entries:
                can_open_same_side = True
                # Scale factor hesapla: 1st=1.0, 2nd=factor, 3rd=factor^2, ...
                if len(same_side_positions) > 0:
                    pyramiding_scale = self.pyramiding_scale_factor ** len(same_side_positions)
        else:
            # Pyramiding is disabled: only 1 position can be opened.
            if not same_side_positions:
                can_open_same_side = True

        # 4. Hedging mode check
        if self.allow_hedging:
            # HEDGE MODE: Opposite positions can also remain open at the same time.
            if can_open_same_side:
                return PositionOpenResult(
                    can_open=True,
                    reason="OK (hedge mode)",
                    pyramiding_scale=pyramiding_scale,
                    should_close_opposite=False
                )
            else:
                reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                return PositionOpenResult(can_open=False, reason=reason)
        else:
            # ONE-WAY MODE: Close opposite positions first
            if opposite_positions:
                # There are opposite positions - first close, then open
                if can_open_same_side:
                    return PositionOpenResult(
                        can_open=True,
                        reason="OK (close opposite first)",
                        pyramiding_scale=pyramiding_scale,
                        should_close_opposite=True,
                        opposite_positions=opposite_positions
                    )
                else:
                    # Pyramiding limit reached
                    reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                    return PositionOpenResult(can_open=False, reason=reason)
            else:
                # No opposite position
                if can_open_same_side:
                    return PositionOpenResult(
                        can_open=True,
                        reason="OK",
                        pyramiding_scale=pyramiding_scale,
                        should_close_opposite=False
                    )
                else:
                    reason = f"Pyramiding limit reached ({len(same_side_positions)}/{self.pyramiding_max_entries})"
                    return PositionOpenResult(can_open=False, reason=reason)

    def check_position_timeout(
        self,
        position: Dict,
        current_timestamp: Any
    ) -> Tuple[bool, str]:
        """
        Position timeout check.

        This method uses BacktestEngine (line 664-665) and TradingEngine (line 997-1004).
        It replaces the duplicate code inside.

        Args:
            position: Pozisyon dict {'opened_at': datetime/timestamp, ...}
            current_timestamp: The current time (datetime, pd.Timestamp, or ms int)

        Returns:
            (should_close: bool, reason: str)

        Example:
            >>> should_close, reason = pm.check_position_timeout(pos, current_time)
            >>> if should_close:
            ...     engine.close_position(pos, reason=reason)
        """
        if not self.position_timeout_enabled:
            return False, ""

        # Get the opening time of the position
        opened_at = position.get('opened_at') or position.get('entry_time') or position.get('open_time')
        if opened_at is None:
            return False, ""

        # Timestamp'leri normalize et
        opened_at_dt = self._normalize_timestamp(opened_at)
        current_dt = self._normalize_timestamp(current_timestamp)

        if opened_at_dt is None or current_dt is None:
            return False, ""

        # Calculate the elapsed time (in minutes)
        time_elapsed_minutes = (current_dt - opened_at_dt).total_seconds() / 60

        if time_elapsed_minutes >= self.position_timeout:
            reason = f"Position timeout ({self.position_timeout} min)"
            return True, reason

        return False, ""

    def get_positions_for_symbol(
        self,
        symbol: str,
        positions: List[Dict],
        side: Optional[str] = None
    ) -> List[Dict]:
        """
        Gets positions for a symbol (and optional side).

        Args:
            symbol: Sembol
            positions: All positions
            side: 'LONG' or 'SHORT' (None = all directions)

        Returns:
            Filtered position list
        """
        result = self._get_positions_for_symbol(symbol, positions)

        if side:
            result = [p for p in result if p.get('side') == side]

        return result

    def get_position_count(
        self,
        symbol: str,
        positions: List[Dict],
        side: Optional[str] = None
    ) -> int:
        """
        Get the number of positions for a symbol.

        Args:
            symbol: Sembol
            positions: All positions
            side: 'LONG' or 'SHORT' (None = all directions)

        Returns:
            Number of positions
        """
        return len(self.get_positions_for_symbol(symbol, positions, side))

    def can_pyramid(
        self,
        symbol: str,
        side: str,
        positions: List[Dict]
    ) -> Tuple[bool, int, float]:
        """
        Check if pyramiding is possible.

        Args:
            symbol: Sembol
            side: 'LONG' or 'SHORT'
            positions: All positions

        Returns:
            (can_pyramid: bool, current_entries: int, scale_factor: float)
        """
        if not self.pyramiding_enabled:
            return False, 0, 1.0

        same_side = self.get_positions_for_symbol(symbol, positions, side)
        current_entries = len(same_side)

        if current_entries >= self.pyramiding_max_entries:
            return False, current_entries, 1.0

        # Scale factor: each subsequent entry is smaller
        scale = self.pyramiding_scale_factor ** current_entries if current_entries > 0 else 1.0

        return True, current_entries, scale

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_positions_for_symbol(self, symbol: str, positions: List[Dict]) -> List[Dict]:
        """Filter positions for a symbol"""
        return [p for p in positions if p.get('symbol') == symbol]

    def _normalize_timestamp(self, ts: Any) -> Optional[datetime]:
        """Convert timestamp to datetime"""
        if ts is None:
            return None

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()

        if isinstance(ts, (int, float)):
            # Milliseconds timestamp
            try:
                return pd.Timestamp(ts, unit='ms').to_pydatetime()
            except:
                return None

        if isinstance(ts, str):
            try:
                return pd.Timestamp(ts).to_pydatetime()
            except:
                return None

        return None

    # ========================================================================
    # POSITION CREATION & UPDATE
    # ========================================================================

    @staticmethod
    def create_position(
        position_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        order_id: Optional[str] = None,
        entry_time: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Create a standard position dictionary.

        This method replaces the manual dictionary
        construction in TradingEngine and BacktestEngine.
        The position format is checked from a single location.

        Args:
            position_id: Pozisyon ID
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            quantity: Quantity
            sl_price: Stop loss price (optional)
            tp_price: Take profit price (optional)
            order_id: Broker order ID (optional)
            entry_time: Entry time (optional, default=now)

        Returns:
            Dict: Standart position dict

        Example:
            >>> pos = PositionManager.create_position(
            ...     position_id=1,
            ...     symbol='BTCUSDT',
            ...     side='LONG',
            ...     entry_price=95000.0,
            ...     quantity=0.1,
            ...     sl_price=94000.0,
            ...     tp_price=97000.0
            ... )
        """
        from datetime import datetime

        if entry_time is None:
            entry_time = datetime.now()

        return {
            'id': position_id,
            'symbol': symbol,
            'side': side.upper(),
            'entry_time': entry_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'original_quantity': quantity,  # For partial exit tracking
            'sl_price': sl_price,
            'tp_price': tp_price,
            'stop_loss': sl_price,      # Alias (stop_loss is used in some places)
            'take_profit': tp_price,    # Alias
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'order_id': order_id,
            'completed_partial_exits': 0,
        }

    @staticmethod
    def update_extreme_prices(
        position: Dict[str, Any],
        current_price: float
    ) -> bool:
        """
        Update the highest/lowest price for the position.

        Required for trailing stop and break-even calculations.

        Args:
            position: Position dictionary (mutable - updated in place)
            current_price: Current price

        Returns:
            bool: True = update performed

        Example:
            >>> updated = PositionManager.update_extreme_prices(position, current_price)
            >>> if updated:
            ...     print(f"New high/low: {position['highest_price']}/{position['lowest_price']}")
        """
        updated = False
        side = position.get('side', '').upper()

        if side == 'LONG':
            if current_price > position.get('highest_price', current_price):
                position['highest_price'] = current_price
                updated = True
        elif side == 'SHORT':
            if current_price < position.get('lowest_price', current_price):
                position['lowest_price'] = current_price
                updated = True

        return updated

    # ========================================================================
    # DISPLAY / DEBUG
    # ========================================================================

    def get_config_summary(self) -> Dict:
        """Get the configuration summary"""
        return {
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'pyramiding_enabled': self.pyramiding_enabled,
            'pyramiding_max_entries': self.pyramiding_max_entries,
            'pyramiding_scale_factor': self.pyramiding_scale_factor,
            'allow_hedging': self.allow_hedging,
            'position_timeout_enabled': self.position_timeout_enabled,
            'position_timeout': self.position_timeout
        }

    def __repr__(self) -> str:
        return (
            f"PositionManager("
            f"max_per_symbol={self.max_positions_per_symbol}, "
            f"pyramiding={self.pyramiding_enabled}, "
            f"hedging={self.allow_hedging})"
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    """Test Position Manager"""
    from dataclasses import dataclass

    @dataclass
    class MockPositionManagement:
        max_positions_per_symbol: int = 2
        pyramiding_enabled: bool = True
        pyramiding_max_entries: int = 3
        pyramiding_scale_factor: float = 0.5
        allow_hedging: bool = False
        position_timeout_enabled: bool = True
        position_timeout: int = 60  # minutes

    class MockStrategy:
        position_management = MockPositionManagement()

    print("=" * 60)
    print("PositionManager v2.0 - Test")
    print("=" * 60)

    pm = PositionManager(MockStrategy())

    # Test 1: Empty position list
    print("\n1. Empty positions - can open LONG?")
    result = pm.can_open_position('BTCUSDT', 'LONG', [])
    print(f"   can_open: {result.can_open}, reason: {result.reason}, scale: {result.pyramiding_scale}")

    # Test 2: There is a LONG position, can a second LONG be opened?
    print("\n2. One LONG exists - can open another LONG?")
    positions = [{'symbol': 'BTCUSDT', 'side': 'LONG'}]
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}, scale: {result.pyramiding_scale:.2f}")

    # Test 3: Is it possible to open a SHORT position if there is a LONG position open? (one-way mode)
    print("\n3. One LONG exists - can open SHORT? (one-way mode)")
    result = pm.can_open_position('BTCUSDT', 'SHORT', positions)
    print(f"   can_open: {result.can_open}, should_close_opposite: {result.should_close_opposite}")
    print(f"   opposite_positions: {len(result.opposite_positions)}")

    # Test 4: Max positions
    print("\n4. Max positions reached?")
    positions = [
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'}
    ]
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}")

    # Test 5: Pyramiding limit
    print("\n5. Pyramiding limit (max_entries=3)?")
    positions = [
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'},
        {'symbol': 'BTCUSDT', 'side': 'LONG'}
    ]
    # Temporarily set max_positions higher to test pyramiding limit
    pm.max_positions_per_symbol = 5
    result = pm.can_open_position('BTCUSDT', 'LONG', positions)
    print(f"   can_open: {result.can_open}, reason: {result.reason}")

    # Test 6: Timeout check
    print("\n6. Position timeout check")
    import time
    pos = {'opened_at': datetime.now()}
    should_close, reason = pm.check_position_timeout(pos, datetime.now())
    print(f"   should_close: {should_close} (just opened)")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
