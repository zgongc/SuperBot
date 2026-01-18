#!/usr/bin/env python3
"""
components/strategies/helpers/condition_parser.py
SuperBot - Condition Parser

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Parses the entry/exit condition arrays.
    
    Format: ['indicator', 'operator', 'value', 'timeframe']
    Example: ['ema_50', '>', 'ema_200', '15m']
           ['rsi_14', 'crossover', 30]
           ['close', 'rising', 3, '5m']

Usage:
    from components.strategies.helpers.condition_parser import ConditionParser
    
    parser = ConditionParser()
    parsed = parser.parse(['ema_50', '>', 'ema_200', '15m'])
    # {
    #     'left': 'ema_50',
    #     'operator': '>',
    #     'right': 'ema_200',
    #     'timeframe': '15m'
    # }
"""

from typing import Dict, Any, List, Union, Optional
import re


class ConditionParser:
    """
    Parses condition arrays.
    
    Desteklenen formatlar:
        3-element: ['indicator', 'operator', 'value']
        4-element: ['indicator', 'operator', 'value', 'timeframe']
        
    Examples:
        ['rsi_14', '>', 70]
        ['ema_50', 'crossover', 'ema_200', '15m']
        ['close', 'rising', 3, '5m']
    """
    
    # Valid operators
    VALID_OPERATORS = {
        # Comparison
        '>',  '<',  '>=',  '<=',  '==',  '!=',
        
        # Crossover/Crossunder
        'crossover', 'crossunder',
        
        # Trend
        'rising', 'falling',
        
        # Range
        'between', 'outside', 'near',
        
        # Boolean
        'is', 'is_not',
    }
    
    # Special keywords (price/volume data)
    PRICE_KEYWORDS = {'open', 'high', 'low', 'close', 'volume'}
    
    def __init__(self):
        """Initialize parser"""
        pass
    
    def parse(self, condition: List) -> Dict[str, Any]:
        """
        Parses the condition array.
        
        Args:
            condition: The condition array.
                Format: ['left', 'operator', 'right'] or
                        ['left', 'operator', 'right', 'timeframe']
        
        Returns:
            Dict: Parsed condition
                {
                    'left': str,
                    'operator': str,
                    'right': Union[str, int, float, list],
                    'timeframe': Optional[str]
                }
        
        Raises:
            ValueError: Invalid condition format
        """
        if not isinstance(condition, (list, tuple)):
            raise ValueError(f"The condition must be a list or tuple, {type(condition)} was given")
        
        if len(condition) < 3:
            raise ValueError(
                f"The condition must have at least 3 elements (left, operator, right), "
                f"{len(condition)} elements exist: {condition}"
            )
        
        if len(condition) > 4:
            raise ValueError(
                f"The condition should have at most 4 elements (left, operator, right, timeframe), "
                f"{len(condition)} elements exist: {condition}"
            )
        
        # Parse elements
        left = condition[0]
        operator = condition[1]
        right = condition[2]
        timeframe = condition[3] if len(condition) == 4 else None
        
        # Validate
        self._validate_operator(operator)
        
        # Normalize
        left = self._normalize_operand(left)
        right = self._normalize_operand(right)
        timeframe = self._normalize_timeframe(timeframe)
        
        return {
            'left': left,
            'operator': operator,
            'right': right,
            'timeframe': timeframe,
            'raw': condition
        }
    
    def parse_batch(self, conditions: List[List]) -> List[Dict[str, Any]]:
        """
        Parses multiple conditions.
        
        Args:
            conditions: A list of condition arrays.
        
        Returns:
            List[Dict]: Parsed conditions
        """
        return [self.parse(cond) for cond in conditions]
    
    def _validate_operator(self, operator: str) -> None:
        """
        Checks the validity of the operator.

        Args:
            operator: Operator string
        
        Raises:
            ValueError: Invalid operator
        """
        if operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator: '{operator}'\n"
                f"Valid operators: {sorted(self.VALID_OPERATORS)}"
            )
    
    def _normalize_operand(self, operand: Any) -> Any:
        """
        Normalize the operand.
        
        Args:
            operand: The left or right operand.
        
        Returns:
            Normalized operand
        """
        # String ise lowercase yap
        if isinstance(operand, str):
            return operand.lower().strip()
        
        # If it's a number, return it as is
        if isinstance(operand, (int, float)):
            return operand
        
        # If it's a list/tuple, return it as is (for between, outside)
        if isinstance(operand, (list, tuple)):
            return operand
        
        # If it's a boolean, return it as is
        if isinstance(operand, bool):
            return operand
        
        # If it is None, return it as is
        if operand is None:
            return operand
        
        # Convert to string for other types
        return str(operand)
    
    def _normalize_timeframe(self, timeframe: Optional[str]) -> Optional[str]:
        """
        Timeframe'i normalize et
        
        Args:
            timeframe: Timeframe string
        
        Returns:
            Normalized timeframe (lowercase) or None
        """
        if timeframe is None:
            return None
        
        if not isinstance(timeframe, str):
            raise ValueError(f"Timeframe must be a string, but {type(timeframe)} was provided")
        
        return timeframe.lower().strip()
    
    # String literal keywords (not indicators)
    STRING_LITERALS = {
        # Signal types
        'buy', 'sell', 'hold', 'strong_buy', 'strong_sell',
        # Trend directions
        'up', 'down', 'neutral', 'bullish', 'bearish',
        # Boolean-like
        'true', 'false', 'yes', 'no',
    }

    def extract_indicators(self, condition: Dict[str, Any]) -> List[str]:
        """
        Extracts the indicators used in the condition.

        Args:
            condition: Parsed condition

        Returns:
            List[str]: Indicator names
        """
        indicators = []
        operator = condition.get('operator', '')

        # Sol operand
        left = condition['left']
        if isinstance(left, str) and left not in self.PRICE_KEYWORDS:
            indicators.append(left)

        # Right operand
        right = condition['right']
        if isinstance(right, str) and right not in self.PRICE_KEYWORDS:
            # The right side of the == or != operators can be a string literal.
            if operator in ('==', '!='):
                # Known string literals are not indicators
                if right.lower() in self.STRING_LITERALS:
                    pass  # Skip - string literal
                elif not self._is_numeric_string(right):
                    indicators.append(right)
            else:
                # If it's not a numeric string, it's an indicator
                if not self._is_numeric_string(right):
                    indicators.append(right)

        return indicators
    
    def _is_numeric_string(self, value: str) -> bool:
        """
        Check if the string is numeric.
        
        Args:
            value: String value
        
        Returns:
            True if numeric string
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def is_price_keyword(self, operand: str) -> bool:
        """
        Check if the operand is a price keyword.
        
        Args:
            operand: Operand string
        
        Returns:
            True if price keyword (open, high, low, close, volume)
        """
        return operand.lower() in self.PRICE_KEYWORDS
    
    def requires_historical_data(self, operator: str) -> bool:
        """
        Does the operator require historical data?
        
        Args:
            operator: Operator string
        
        Returns:
            True if historical data needed (crossover, rising, falling)
        """
        return operator in {'crossover', 'crossunder', 'rising', 'falling'}
    
    def get_lookback_period(self, condition: Dict[str, Any]) -> int:
        """
        The lookback period required for the condition.
        
        Args:
            condition: Parsed condition
        
        Returns:
            Lookback period (number of bars)
        """
        operator = condition['operator']
        
        # Crossover/Crossunder: 2 bar
        if operator in {'crossover', 'crossunder'}:
            return 2
        
        # Rising/Falling: up to the value of the right operand (e.g., rising 3 -> 3 bars)
        if operator in {'rising', 'falling'}:
            right = condition['right']
            if isinstance(right, int):
                return right + 1  # +1 because the previous bar is needed for comparison
            return 2  # Default
        
        # Others: 1 bar (available)
        return 1
    
    def __repr__(self) -> str:
        return f"<ConditionParser operators={len(self.VALID_OPERATORS)}>"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_condition(condition: List) -> Dict[str, Any]:
    """
    Convenience function - parses the condition
    
    Args:
        condition: The condition array.
    
    Returns:
        Parsed condition dict
    """
    parser = ConditionParser()
    return parser.parse(condition)


def parse_conditions(conditions: List[List]) -> List[Dict[str, Any]]:
    """
    Convenience function - parses multiple conditions.
    
    Args:
        conditions: A list of condition arrays.
    
    Returns:
        List of parsed conditions
    """
    parser = ConditionParser()
    return parser.parse_batch(conditions)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ConditionParser',
    'parse_condition',
    'parse_conditions',
]

