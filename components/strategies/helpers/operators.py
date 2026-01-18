#!/usr/bin/env python3
"""
components/strategies/helpers/operators.py
SuperBot - Condition Operators

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Implementation of conditional operators.
    
    Supported operators:
    - Comparison: >, <, >=, <=, ==, !=
    - Crossover/Crossunder: crossover, crossunder
    - Trend: rising, falling
    - Range: between, outside, near

Usage:
    from components.strategies.helpers.operators import evaluate_condition
    
    # Comparison
    result = evaluate_condition(45.5, '>', 30)  # True
    
    # Crossover
    result = evaluate_crossover([29, 31], 30)  # True (crossover is above)
    
    # Rising
    result = evaluate_rising([40, 42, 45], 3)  # True (3 bar rising)
"""

from typing import Any, List, Union, Optional
import numpy as np
import pandas as pd


# ============================================================================
# COMPARISON OPERATORS
# ============================================================================

def compare_gt(left: Union[int, float], right: Union[int, float]) -> bool:
    """Greater than (>)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) > float(right)


def compare_lt(left: Union[int, float], right: Union[int, float]) -> bool:
    """Less than (<)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) < float(right)


def compare_gte(left: Union[int, float], right: Union[int, float]) -> bool:
    """Greater than or equal to (>=)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) >= float(right)


def compare_lte(left: Union[int, float], right: Union[int, float]) -> bool:
    """Less than or equal to (<=)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) <= float(right)


def compare_eq(left: Any, right: Any) -> bool:
    """Equals (==)"""
    # Boolean comparison
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    
    # Numeric comparison
    try:
        return float(left) == float(right)
    except (ValueError, TypeError):
        # String comparison
        return str(left) == str(right)


def compare_neq(left: Any, right: Any) -> bool:
    """Not equal (!=)"""
    return not compare_eq(left, right)


# ============================================================================
# CROSSOVER / CROSSUNDER
# ============================================================================

def evaluate_crossover(
    series: Union[List, pd.Series, np.ndarray],
    threshold: Union[float, List, pd.Series, np.ndarray]
) -> bool:
    """
    Crossover detection (upper cut)
    
    Args:
        series: Indicator series (last 2 values are minimum)
        threshold: The value or series used as a threshold.
    
    Returns:
        True if crossover occurred (below the previous bar, now above)
    
    Example:
        series = [29, 31], threshold = 30 -> True (crossover)
        series = [31, 32], threshold = 30 -> False (already above)
    """
    # Convert to numpy array
    if isinstance(series, (list, tuple)):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values
    
    if isinstance(threshold, (list, tuple)):
        threshold = np.array(threshold)
    elif isinstance(threshold, pd.Series):
        threshold = threshold.values
    
    # Need at least 2 values
    if len(series) < 2:
        return False

    # Scalar threshold
    if np.isscalar(threshold):
        # NaN check: if any value is NaN, return False
        if pd.isna(series[-2]) or pd.isna(series[-1]) or pd.isna(threshold):
            return False
        prev_below = series[-2] < threshold
        current_above = series[-1] > threshold
        return prev_below and current_above

    # Series threshold (same length as series)
    if len(threshold) < 2:
        return False

    # NaN check: if any value is NaN, return False
    if pd.isna(series[-2]) or pd.isna(series[-1]) or pd.isna(threshold[-2]) or pd.isna(threshold[-1]):
        return False
    prev_below = series[-2] < threshold[-2]
    current_above = series[-1] > threshold[-1]
    return prev_below and current_above


def evaluate_crossunder(
    series: Union[List, pd.Series, np.ndarray],
    threshold: Union[float, List, pd.Series, np.ndarray]
) -> bool:
    """
    Crossunder detection (undershoot)
    
    Args:
        series: Indicator series (last 2 values are minimum)
        threshold: The value or series used for thresholding.
    
    Returns:
        True if crossunder occurred (previous bar was above, now it is below)
    
    Example:
        series = [31, 29], threshold = 30 -> True (crossunder)
        series = [29, 28], threshold = 30 -> False (already below)
    """
    # Convert to numpy array
    if isinstance(series, (list, tuple)):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values
    
    if isinstance(threshold, (list, tuple)):
        threshold = np.array(threshold)
    elif isinstance(threshold, pd.Series):
        threshold = threshold.values
    
    # Need at least 2 values
    if len(series) < 2:
        return False

    # Scalar threshold
    if np.isscalar(threshold):
        # NaN check: if any value is NaN, return False
        if pd.isna(series[-2]) or pd.isna(series[-1]) or pd.isna(threshold):
            return False
        prev_above = series[-2] > threshold
        current_below = series[-1] < threshold
        return prev_above and current_below

    # Series threshold (same length as series)
    if len(threshold) < 2:
        return False

    # NaN check: if any value is NaN, return False
    if pd.isna(series[-2]) or pd.isna(series[-1]) or pd.isna(threshold[-2]) or pd.isna(threshold[-1]):
        return False
    prev_above = series[-2] > threshold[-2]
    current_below = series[-1] < threshold[-1]
    return prev_above and current_below


# ============================================================================
# TREND OPERATORS
# ============================================================================

def evaluate_rising(
    series: Union[List, pd.Series, np.ndarray],
    periods: int
) -> bool:
    """
    Rising detection (is it rising?)
    
    Args:
        series: Indicator serisi
        periods: The number of bars to check.
    
    Returns:
        True if rising for N periods
    
    Example:
        series = [40, 42, 45], periods = 3 -> True
        series = [40, 42, 41], periods = 3 -> False
    """
    # Convert to numpy array
    if isinstance(series, (list, tuple)):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values
    
    # Need at least N values
    if len(series) < periods:
        return False
    
    # Check last N values
    last_n = series[-periods:]
    
    # All consecutive values should be increasing
    for i in range(1, len(last_n)):
        if last_n[i] <= last_n[i-1]:
            return False
    
    return True


def evaluate_falling(
    series: Union[List, pd.Series, np.ndarray],
    periods: int
) -> bool:
    """
    Falling detection (is it falling?)
    
    Args:
        series: Indicator serisi
        periods: The number of bars to check.
    
    Returns:
        True if falling for N periods
    
    Example:
        series = [45, 42, 40], periods = 3 -> True
        series = [45, 42, 43], periods = 3 -> False
    """
    # Convert to numpy array
    if isinstance(series, (list, tuple)):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values
    
    # Need at least N values
    if len(series) < periods:
        return False
    
    # Check last N values
    last_n = series[-periods:]
    
    # All consecutive values should be decreasing
    for i in range(1, len(last_n)):
        if last_n[i] >= last_n[i-1]:
            return False
    
    return True


# ============================================================================
# RANGE OPERATORS
# ============================================================================

def evaluate_between(
    value: Union[int, float],
    range_values: List[Union[int, float]]
) -> bool:
    """
    Check if the value is within the specified range.
    
    Args:
        value: The value to be checked
        range_values: [min, max]
    
    Returns:
        True if min <= value <= max
    
    Example:
        value = 50, range_values = [30, 70] -> True
        value = 80, range_values = [30, 70] -> False
    """
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        raise ValueError(f"The [min, max] values are required for the 'between' operator, but {range_values} was provided.")
    
    min_val, max_val = range_values
    return float(min_val) <= float(value) <= float(max_val)


def evaluate_outside(
    value: Union[int, float],
    range_values: List[Union[int, float]]
) -> bool:
    """
    Outside detection (is it outside the range?)
    
    Args:
        value: The value to be checked
        range_values: [min, max]
    
    Returns:
        True if value < min OR value > max
    
    Example:
        value = 80, range_values = [30, 70] -> True
        value = 50, range_values = [30, 70] -> False
    """
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        raise ValueError(f"The [min, max] parameters are required for the 'outside' operator, but {range_values} was provided.")
    
    min_val, max_val = range_values
    return float(value) < float(min_val) or float(value) > float(max_val)


def evaluate_near(
    value: Union[int, float],
    target_and_threshold: List[Union[int, float]]
) -> bool:
    """
    Near detection (is it near?)
    
    Args:
        value: The value to be checked
        target_and_threshold: [target, percent_threshold]
    
    Returns:
        True if value is within percent_threshold of target
    
    Example:
        value = 101, target_and_threshold = [100, 1] -> True (within %1)
        value = 105, target_and_threshold = [100, 1] -> False
    """
    if not isinstance(target_and_threshold, (list, tuple)) or len(target_and_threshold) != 2:
        raise ValueError(
            f"The 'near' operator requires [target, percent_threshold], "
            f"{target_and_threshold} verildi"
        )
    
    target, percent_threshold = target_and_threshold
    
    # Calculate threshold range
    threshold_amount = abs(float(target) * float(percent_threshold) / 100.0)
    lower_bound = float(target) - threshold_amount
    upper_bound = float(target) + threshold_amount
    
    return lower_bound <= float(value) <= upper_bound


# ============================================================================
# UNIFIED EVALUATOR
# ============================================================================

def evaluate_condition(
    left: Any,
    operator: str,
    right: Any
) -> bool:
    """
    Evaluate the condition (unified interface)
    
    Args:
        left: Left operand (value or series)
        operator: Operator string
        right: The right operand (value or series)
    
    Returns:
        bool: The result of the condition.
    
    Raises:
        ValueError: Invalid operator
    
    Example:
        evaluate_condition(45, '>', 30)  # True
        evaluate_condition([29, 31], 'crossover', 30)  # True
        evaluate_condition([40, 42, 45], 'rising', 3)  # True
    """
    # Comparison operators
    if operator == '>':
        return compare_gt(left, right)
    elif operator == '<':
        return compare_lt(left, right)
    elif operator == '>=':
        return compare_gte(left, right)
    elif operator == '<=':
        return compare_lte(left, right)
    elif operator == '==':
        return compare_eq(left, right)
    elif operator == '!=':
        return compare_neq(left, right)
    
    # Crossover/Crossunder
    elif operator == 'crossover':
        return evaluate_crossover(left, right)
    elif operator == 'crossunder':
        return evaluate_crossunder(left, right)
    
    # Trend
    elif operator == 'rising':
        return evaluate_rising(left, right)
    elif operator == 'falling':
        return evaluate_falling(left, right)
    
    # Range
    elif operator == 'between':
        return evaluate_between(left, right)
    elif operator == 'outside':
        return evaluate_outside(left, right)
    elif operator == 'near':
        return evaluate_near(left, right)
    
    # Boolean (alias for ==)
    elif operator in ('is', 'is_not'):
        if operator == 'is':
            return compare_eq(left, right)
        else:  # is_not
            return compare_neq(left, right)
    
    else:
        raise ValueError(f"Unsupported operator: '{operator}'")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_required_history(operator: str, right_operand: Any) -> int:
    """
    The number of history bars required for the operator.
    
    Args:
        operator: Operator string
        right_operand: The right operand.
    
    Returns:
        int: The required number of bars.
    """
    if operator in ('crossover', 'crossunder'):
        return 2
    
    if operator in ('rising', 'falling'):
        if isinstance(right_operand, int):
            return right_operand
        return 2  # Default
    
    # Other operators: 1 bar (existing)
    return 1


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Comparison
    'compare_gt',
    'compare_lt',
    'compare_gte',
    'compare_lte',
    'compare_eq',
    'compare_neq',
    
    # Crossover/Crossunder
    'evaluate_crossover',
    'evaluate_crossunder',
    
    # Trend
    'evaluate_rising',
    'evaluate_falling',
    
    # Range
    'evaluate_between',
    'evaluate_outside',
    'evaluate_near',
    
    # Unified
    'evaluate_condition',
    'get_required_history',
]

