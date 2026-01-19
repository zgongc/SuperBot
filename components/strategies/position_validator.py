#!/usr/bin/env python3
"""
Position Validator - Checks position rules.

It validates whether a position can be opened based on the strategy parameters.
This file is in the strategies/ folder because it is related to STRATEGY RULES.

Responsibilities:
- Check pyramiding rules
- Check hedging rules
- Check position limits
- Check position timeout rules

NOTE: Execution (actual position opening/closing) is handled in managers/position_manager.py.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Position validation result
    
    Attributes:
        accepted: Was the position accepted?
        reason: Red/acceptance reason (Turkish explanation)
        action: Action to be performed ('open_new', 'pyramid', 'reject', 'close_opposite')
    """
    accepted: bool
    reason: str
    action: Optional[str] = None


class PositionValidator:
    """
    Position Validation Logic (Strategy Rules)
    
    Sorumluluklar:
    - Check if the signal can open a position.
    - Manage the pyramiding logic.
    - Manage the hedging logic.
    - Pozisyon limitlerini zorla
    
    Sorumlu OLMADIKLARI:
    - Opening real positions (PositionManager)
    - Risk calculation (RiskManager)
    - Order execution (OrderManager)
    
    Mimari:
    - Strategy rules are located here (strategies/).
    - Execution logic manager'da (managers/)
    - DRY: Live trading and backtesting use the same code.
    - SOLID: Single Responsibility Principle
    - Testable: Unit tests can be written.
    """
    
    def __init__(self, position_management_config: Any, logger: Optional[Any] = None):
        """
        Initialize the Validator with the strategy's position_management config.
        
        Args:
            position_management_config: The PositionManagement object for the strategy.
            logger: Optional logger.
        """
        self.config = position_management_config
        self.logger = logger
        
        # Extract settings
        self.max_positions_per_symbol = getattr(position_management_config, 'max_positions_per_symbol', 1)
        self.max_total_positions = getattr(position_management_config, 'max_total_positions', 2)
        self.allow_hedging = getattr(position_management_config, 'allow_hedging', False)
        
        # Pyramiding settings
        self.pyramiding_enabled = getattr(position_management_config, 'pyramiding_enabled', False)
        self.pyramiding_max_entries = getattr(position_management_config, 'pyramiding_max_entries', 1)
        self.pyramiding_scale_factor = getattr(position_management_config, 'pyramiding_scale_factor', 1.0)
        
        # Timeout settings
        self.timeout_enabled = getattr(position_management_config, 'position_timeout_enabled', False)
        self.timeout_minutes = getattr(position_management_config, 'position_timeout', 1800)
        
        if self.logger:
            self.logger.debug(f"PositionValidator started:")
            self.logger.debug(f"  - Max pozisyon/sembol: {self.max_positions_per_symbol}")
            self.logger.debug(f"  - Max total positions: {self.max_total_positions}")
            self.logger.debug(f"  - Hedging permission: {self.allow_hedging}")
            self.logger.debug(f"  - Pyramiding: {self.pyramiding_enabled} (max: {self.pyramiding_max_entries})")
            self.logger.debug(f"  - Timeout: {self.timeout_enabled} ({self.timeout_minutes} minutes)")
    
    def can_open_position(
        self,
        signal_side: str,
        open_positions: Dict[str, Any],
        total_positions_count: Optional[int] = None
    ) -> ValidationResult:
        """
        Checks if the signal can be positioned.
        
        Args:
            signal_side: 'long' or 'short'
            open_positions: A dictionary of open positions {side -> Position or List[Position]}
            total_positions_count: The total number of open positions across all symbols (optional)
        
        Returns:
            ValidationResult (acceptance/rejection decision)
        """
        # Step 1: Check the total position limit
        if total_positions_count is None:
            total_positions_count = self._count_total_positions(open_positions)
        
        if total_positions_count >= self.max_total_positions:
            return ValidationResult(
                accepted=False,
                reason=f"Maximum total position limit reached ({total_positions_count}/{self.max_total_positions})"
            )
        
        # Step 2: Check if there is a position on this side
        same_side_positions = self._get_positions_by_side(open_positions, signal_side)
        opposite_side = 'short' if signal_side == 'long' else 'long'
        opposite_side_positions = self._get_positions_by_side(open_positions, opposite_side)
        
        # Step 2.5: max_positions_per_symbol check (for multi-symbol support)
        # Count all positions for this symbol (both sides)
        symbol_position_count = 0
        if same_side_positions:
            symbol_position_count += len(same_side_positions) if isinstance(same_side_positions, list) else 1
        if opposite_side_positions:
            symbol_position_count += len(opposite_side_positions) if isinstance(opposite_side_positions, list) else 1
        
        if symbol_position_count >= self.max_positions_per_symbol:
            return ValidationResult(
                accepted=False,
                reason=f"Maximum position limit reached per symbol ({symbol_position_count}/{self.max_positions_per_symbol})"
            )
        
        # Step 3: PYRAMIDING logic
        if same_side_positions:
            if self.pyramiding_enabled:
                # Can another entry be added?
                num_entries = len(same_side_positions) if isinstance(same_side_positions, list) else 1
                
                if num_entries < self.pyramiding_max_entries:
                    return ValidationResult(
                        accepted=True,
                        reason=f"Pyramiding: Entry {num_entries + 1}/{self.pyramiding_max_entries} ekleniyor",
                        action='pyramid'
                    )
                else:
                    return ValidationResult(
                        accepted=False,
                        reason=f"Reached pyramiding limit ({num_entries}/{self.pyramiding_max_entries})"
                    )
            else:
                # Pyramiding is disabled - cannot be opened on the same side.
                return ValidationResult(
                    accepted=False,
                    reason=f"There is already a {signal_side} position (pyramiding is disabled)"
                )
        
        # Step 4: HEDGING logic
        if opposite_side_positions:
            if self.allow_hedging:
                # The reverse side can be opened (hedging)
                return ValidationResult(
                    accepted=True,
                    reason=f"Hedging: A {signal_side} position is being opened while there is an {opposite_side} position",
                    action='open_new'
                )
            else:
                # Hedging is disabled - the opposite side cannot be opened.
                return ValidationResult(
                    accepted=False,
                    reason=f"It already has the {opposite_side} position (hedging is disabled)"
                )
        
        # Step 5: Normal entry (no collision)
        return ValidationResult(
            accepted=True,
            reason=f"A new {signal_side} position is being opened",
            action='open_new'
        )
    
    def _count_total_positions(self, open_positions: Dict[str, Any]) -> int:
        """
        Calculate the total number of open positions.
        
        Args:
            open_positions: Dictionary of open positions.
        
        Returns:
            Total number
        """
        count = 0
        for side, pos in open_positions.items():
            if isinstance(pos, list):
                count += len(pos)
            elif pos is not None:
                count += 1
        return count
    
    def _get_positions_by_side(self, open_positions: Dict[str, Any], side: str) -> Any:
        """
        Retrieves positions for a specific side.
        
        Args:
            open_positions: Dictionary of open positions
            side: 'long' or 'short'
        
        Returns:
            Position, List[Position], or None
        """
        return open_positions.get(side)
    
    def calculate_pyramid_size(self, base_size: float, entry_number: int) -> float:
        """
        Calculate the position size for pyramiding entry.
        
        Args:
            base_size: Base position size
            entry_number: Entry number (1, 2, 3, ...)
        
        Returns:
            Scaled position size.
        
        Example:
            base_size = 1.0, scale_factor = 0.5
            Entry 1: 1.0 (100%)
            Entry 2: 0.5 (50%)
            Entry 3: 0.25 (25%)
        """
        if entry_number == 1:
            return base_size
        
        # Scale each entry by the pyramiding_scale_factor
        scaled_size = base_size * (self.pyramiding_scale_factor ** (entry_number - 1))
        return scaled_size
    
    def should_close_opposite_for_entry(
        self,
        signal_side: str,
        open_positions: Dict[str, Any]
    ) -> bool:
        """
        Check if the reverse position should be closed before entering.
        (For strategies without hedging)
        
        Args:
            signal_side: 'long' or 'short'
            open_positions: Dictionary of open positions
        
        Returns:
            If True, the reverse position should be closed.
        """
        if self.allow_hedging:
            return False  # Hedging is allowed, no need to close
        
        opposite_side = 'short' if signal_side == 'long' else 'long'
        opposite_pos = self._get_positions_by_side(open_positions, opposite_side)
        
        return opposite_pos is not None

