#!/usr/bin/env python3
"""
Strategy Validation - Wrapper Module (For backward compatibility)

This module is now just a wrapper.
The actual validation logic is located in components/strategies/strategy_validator.py.

This file is kept only for backward compatibility.

New code should directly use strategy_validator.py:
    from components.strategies.strategy_validator import validate_strategy
"""

from typing import Any, Optional

# All items that can be imported
from components.strategies.strategy_validator import (
    StrategyValidator,
    ValidationError,
    StrategyValidationResult,
    ValidationException,
    validate_strategy,
    validate_strategy_strict,
)


# ==================== LEGACY WRAPPER FUNCTIONS ====================
# (Backward compatibility for old code - calls the new validation)


def validate_condition(condition: Any, context: str = "") -> None:
    """
    Validate a single condition (LEGACY)
    
    Now this function is inside the StrategyValidator.
    # Kept for backward compatibility.
    
    Args:
        condition: Condition (list/tuple)
        context: The context of the error message.
    
    Raises:
        ValidationException: Raised when an invalid condition is encountered.
    """
    # This function is now deprecated, but kept for backward compatibility.
    # Let's do a minimal validation
    if not isinstance(condition, (list, tuple)):
        raise ValidationException(f"{context}: The condition must be a list or a tuple")
    
    if len(condition) < 3:
        raise ValidationException(f"{context}: The condition must contain at least 3 elements")


def validate_entry_conditions(entry_conditions: Any) -> None:
    """
    Validate entry conditions (LEGACY)
    
    Now this function is inside the StrategyValidator.
    
    Args:
        entry_conditions: Entry conditions dict
    
    Raises:
        ValidationException: Raised when an invalid condition is encountered.
    """
    if not isinstance(entry_conditions, dict):
        raise ValidationException("entry_conditions must be a dictionary")
    
    if 'long' not in entry_conditions and 'short' not in entry_conditions:
        raise ValidationException("At least one of 'long' or 'short' condition is required")
    
    for side, conditions in entry_conditions.items():
        if not isinstance(conditions, list):
            raise ValidationException(f"The {side} must be a list")
        
        for idx, condition in enumerate(conditions):
            validate_condition(condition, f"entry_conditions.{side}[{idx}]")


def validate_exit_conditions(exit_conditions: Any) -> None:
    """
    Validate exit conditions (LEGACY)
    
    This function is now inside the StrategyValidator.
    
    Args:
        exit_conditions: Exit conditions dict
    
    Raises:
        ValidationException: Raised when an invalid condition is encountered.
    """
    if not isinstance(exit_conditions, dict):
        raise ValidationException("exit_conditions must be a dictionary")
    
    for side, conditions in exit_conditions.items():
        if not isinstance(conditions, list):
            raise ValidationException(f"The {side} must be a list")
        
        for idx, condition in enumerate(conditions):
            validate_condition(condition, f"exit_conditions.{side}[{idx}]")


def validate_risk_management(risk_management: Any) -> None:
    """
    Risk management config'ini validate et (LEGACY)
    
    This function is now inside the StrategyValidator.
    
    Args:
        risk_management: RiskManagement dataclass
    
    Raises:
        ValidationException: If the config is invalid.
    """
    if not hasattr(risk_management, 'size_value'):
        raise ValidationException("size_value is required in risk management")
    
    if risk_management.size_value <= 0:
        raise ValidationException("size_value must be positive")


def validate_exit_strategy(exit_strategy: Any) -> None:
    """
    Exit strategy config'ini validate et (LEGACY)
    
    Now this function is inside the StrategyValidator.
    
    Args:
        exit_strategy: ExitStrategy dataclass
    
    Raises:
        ValidationException: If the config is invalid.
    """
    if hasattr(exit_strategy, 'take_profit_value'):
        if exit_strategy.take_profit_value <= 0:
            raise ValidationException("take_profit_value must be positive")
    
    if hasattr(exit_strategy, 'stop_loss_value'):
        if exit_strategy.stop_loss_value <= 0:
            raise ValidationException("stop_loss_value must be positive")


def validate_position_management(position_management: Any) -> None:
    """
    Position management config'ini validate et (LEGACY)
    
    Now this function is inside the StrategyValidator.
    
    Args:
        position_management: PositionManagement dataclass
    
    Raises:
        ValidationException: If the config is invalid.
    """
    if hasattr(position_management, 'max_total_positions'):
        if position_management.max_total_positions <= 0:
            raise ValidationException("max_total_positions must be positive")


def validate_complete_strategy(strategy: Any) -> None:
    """
    Validate the entire strategy (LEGACY)
    
    Now this function is inside the StrategyValidator.
    This wrapper calls the new validator.
    
    Args:
        strategy: Strategy object
    
    Raises:
        ValidationException: If the config is invalid.
    """
    # Use the new validator
    validate_strategy_strict(strategy)


# ==================== EXPORT ====================

__all__ = [
    # Ana validator (yeni)
    'StrategyValidator',
    'ValidationError',
    'StrategyValidationResult',
    'ValidationException',
    'validate_strategy',
    'validate_strategy_strict',
    
    # Legacy functions (geriye uyumluluk)
    'validate_condition',
    'validate_entry_conditions',
    'validate_exit_conditions',
    'validate_risk_management',
    'validate_exit_strategy',
    'validate_position_management',
    'validate_complete_strategy',
]
