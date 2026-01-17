#!/usr/bin/env python3
"""
components/strategies/helpers/__init__.py
SuperBot - Strategy Helpers

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Helper utilities for strategy development:
    - condition_parser: Parse condition arrays
    - operators: Evaluate condition operators
    - validation: Validate strategy configs
    - timeframe_utils: MTF utilities
"""

from components.strategies.helpers.condition_parser import (
    ConditionParser,
    parse_condition,
    parse_conditions,
)

from components.strategies.helpers.operators import (
    # Comparison
    compare_gt,
    compare_lt,
    compare_gte,
    compare_lte,
    compare_eq,
    compare_neq,
    
    # Crossover/Crossunder
    evaluate_crossover,
    evaluate_crossunder,
    
    # Trend
    evaluate_rising,
    evaluate_falling,
    
    # Range
    evaluate_between,
    evaluate_outside,
    evaluate_near,
    
    # Unified
    evaluate_condition,
    get_required_history,
)

from components.strategies.helpers.validation import (
    StrategyValidator,
    ValidationError,
    validate_strategy,
)

from components.strategies.helpers.timeframe_utils import (
    TIMEFRAME_MINUTES,
    timeframe_to_minutes,
    timeframe_to_seconds,
    timeframe_to_milliseconds,
    compare_timeframes,
    is_higher_timeframe,
    is_lower_timeframe,
    sort_timeframes,
    get_higher_timeframes,
    get_lower_timeframes,
    calculate_bars_needed,
    is_valid_timeframe,
    get_all_timeframes,
)


__all__ = [
    # Condition Parser
    'ConditionParser',
    'parse_condition',
    'parse_conditions',
    
    # Operators
    'compare_gt',
    'compare_lt',
    'compare_gte',
    'compare_lte',
    'compare_eq',
    'compare_neq',
    'evaluate_crossover',
    'evaluate_crossunder',
    'evaluate_rising',
    'evaluate_falling',
    'evaluate_between',
    'evaluate_outside',
    'evaluate_near',
    'evaluate_condition',
    'get_required_history',
    
    # Validation
    'StrategyValidator',
    'ValidationError',
    'validate_strategy',
    
    # Timeframe Utils
    'TIMEFRAME_MINUTES',
    'timeframe_to_minutes',
    'timeframe_to_seconds',
    'timeframe_to_milliseconds',
    'compare_timeframes',
    'is_higher_timeframe',
    'is_lower_timeframe',
    'sort_timeframes',
    'get_higher_timeframes',
    'get_lower_timeframes',
    'calculate_bars_needed',
    'is_valid_timeframe',
    'get_all_timeframes',
]
