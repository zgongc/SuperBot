#!/usr/bin/env python3
"""
modules/backtest/vectorized_conditions.py
SuperBot - Vectorized Condition Evaluation

Fast backtesting vectorized condition evaluation helpers.
Loop without using loop checks all candles for condition.

Author: SuperBot Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from core.logger_engine import get_logger

logger = get_logger("modules.backtest.vectorized_conditions")


def evaluate_condition_vectorized(
    left_data: pd.Series,
    operator: str,
    right_data: Any,
    warmup: int = 0
) -> pd.Series:
    """
    Vectorized condition evaluation

    Args:
        left_data: Left operand series (e.g., ema_21)
        operator: Comparison operator
        right_data: Right operand (series, number, or list)
        warmup: Warmup period (signals before this are False)

    Returns:
        Boolean series (True where condition met)
    """
    # Initialize result with False
    result = pd.Series(False, index=left_data.index)

    # Warmup period: no signals
    if warmup > 0:
        result.iloc[:warmup] = False

    try:
        if operator == '>':
            # Slice both sides to avoid index misalignment
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] > right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] > right_data).values

        elif operator == '<':
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] < right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] < right_data).values

        elif operator == '>=':
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] >= right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] >= right_data).values

        elif operator == '<=':
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] <= right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] <= right_data).values

        elif operator == '==':
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] == right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] == right_data).values

        elif operator == '!=':
            if isinstance(right_data, pd.Series):
                result.iloc[warmup:] = (left_data.iloc[warmup:] != right_data.iloc[warmup:]).values
            else:
                result.iloc[warmup:] = (left_data.iloc[warmup:] != right_data).values

        elif operator == 'crossover':
            # left crosses above right
            # Condition: left[i-1] <= right[i-1] AND left[i] > right[i]
            if warmup + 1 < len(left_data):
                # Use shift() for proper alignment
                left_prev = left_data.shift(1)
                left_curr = left_data

                if isinstance(right_data, pd.Series):
                    right_prev = right_data.shift(1)
                    right_curr = right_data
                else:
                    right_prev = right_data
                    right_curr = right_data

                # Crossover: prev below AND curr above
                crossover = (left_prev <= right_prev) & (left_curr > right_curr)
                result.iloc[warmup+1:] = crossover.iloc[warmup+1:].values

        elif operator == 'crossunder':
            # left crosses below right
            # Condition: left[i-1] >= right[i-1] AND left[i] < right[i]
            if warmup + 1 < len(left_data):
                # Use shift() for proper alignment
                left_prev = left_data.shift(1)
                left_curr = left_data

                if isinstance(right_data, pd.Series):
                    right_prev = right_data.shift(1)
                    right_curr = right_data
                else:
                    right_prev = right_data
                    right_curr = right_data

                # Crossunder: prev above AND curr below
                crossunder = (left_prev >= right_prev) & (left_curr < right_curr)
                result.iloc[warmup+1:] = crossunder.iloc[warmup+1:].values

        elif operator == 'rising':
            # Trend is rising for N periods
            # right_data should be period count (int)
            period = int(right_data) if isinstance(right_data, (int, float)) else 2
            if warmup + period < len(left_data):
                # Check if current > previous for last N candles
                is_rising = pd.Series(False, index=left_data.index)
                for i in range(warmup + period, len(left_data)):
                    is_rising.iloc[i] = all(
                        left_data.iloc[i - j] > left_data.iloc[i - j - 1]
                        for j in range(period)
                    )
                result = is_rising

        elif operator == 'falling':
            # Trend is falling for N periods
            period = int(right_data) if isinstance(right_data, (int, float)) else 2
            if warmup + period < len(left_data):
                is_falling = pd.Series(False, index=left_data.index)
                for i in range(warmup + period, len(left_data)):
                    is_falling.iloc[i] = all(
                        left_data.iloc[i - j] < left_data.iloc[i - j - 1]
                        for j in range(period)
                    )
                result = is_falling

        elif operator == 'between':
            # Value between two bounds
            if isinstance(right_data, (list, tuple)) and len(right_data) == 2:
                lower, upper = right_data
                result.iloc[warmup:] = (
                    (left_data.iloc[warmup:] >= lower) &
                    (left_data.iloc[warmup:] <= upper)
                ).values

        else:
            # Unknown operator - return all False
            pass

    except Exception as e:
        # On error, return all False
        pass

    return result


def build_condition_mask(
    condition: List[Any],
    df: pd.DataFrame,
    warmup: int = 0,
    debug: bool = False,
    indicators_mtf: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> pd.Series:
    """
    Build boolean mask for a single condition with MTF support

    Args:
        condition: Condition list [left, operator, right] or [left, operator, right, timeframe]
        df: DataFrame with indicator data (primary timeframe)
        warmup: Warmup period
        debug: Print debug info
        indicators_mtf: Multi-timeframe indicators {timeframe: {indicator: values}}

    Returns:
        Boolean series
    """
    # Parse condition
    left = condition[0]
    operator = condition[1]
    right = condition[2]
    timeframe = condition[3] if len(condition) > 3 else None  # MTF support

    # Get left operand data
    if isinstance(left, str):
        # Check if MTF indicator (timeframe specified)
        if timeframe and indicators_mtf and timeframe in indicators_mtf:
            # MTF indicator - get from indicators_mtf
            if left in indicators_mtf[timeframe]:
                left_data = pd.Series(indicators_mtf[timeframe][left], index=df.index)
                if debug:
                    logger.debug(f"   📊 MTF: {left} ({timeframe})")
            else:
                if debug:
                    logger.warning(f"   ⚠️  MTF Indicator '{left}' NOT FOUND in {timeframe}!")
                return pd.Series(False, index=df.index)
        else:
            # Primary timeframe indicator - get from df
            if left not in df.columns:
                if debug:
                    logger.warning(f"   ⚠️  Indicator '{left}' NOT FOUND in columns!")
                return pd.Series(False, index=df.index)
            left_data = df[left]
    else:
        # Constant value
        left_data = pd.Series(left, index=df.index)

    # Get right operand data
    if isinstance(right, str):
        if right in df.columns:
            # Indicator column
            right_data = df[right]
        elif operator in ('==', '!='):
            # String literal for equality comparison (e.g., 'buy', 'sell', 'UP', 'DOWN')
            right_data = right
            if debug:
                logger.debug(f"   📝 String literal: '{right}' (for {operator} comparison)")
        else:
            if debug:
                logger.warning(f"   ⚠️  Indicator '{right}' NOT FOUND in columns!")
            return pd.Series(False, index=df.index)
    elif isinstance(right, (list, tuple)):
        # Keep as-is for 'between' operator
        right_data = right
    else:
        # Constant value
        right_data = right

    # Evaluate condition
    result = evaluate_condition_vectorized(left_data, operator, right_data, warmup)
    return result


def build_conditions_mask(
    conditions: List[List[Any]],
    df: pd.DataFrame,
    warmup: int = 0,
    logic: str = 'AND',
    debug: bool = False,
    indicators_mtf: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> pd.Series:
    """
    Build combined boolean mask for multiple conditions with MTF support

    Args:
        conditions: List of conditions
        df: DataFrame with indicator data (primary timeframe)
        warmup: Warmup period
        logic: 'AND' or 'OR' (default: AND - all conditions must be met)
        debug: Print debug info
        indicators_mtf: Multi-timeframe indicators {timeframe: {indicator: values}}

    Returns:
        Combined boolean series
    """
    if not conditions:
        return pd.Series(False, index=df.index)

    # Build mask for first condition
    combined_mask = build_condition_mask(conditions[0], df, warmup, debug=debug, indicators_mtf=indicators_mtf)

    # Combine with remaining conditions
    for condition in conditions[1:]:
        mask = build_condition_mask(condition, df, warmup, debug=debug, indicators_mtf=indicators_mtf)

        if logic == 'AND':
            combined_mask = combined_mask & mask
        elif logic == 'OR':
            combined_mask = combined_mask | mask

    if debug:
        logger.debug(f"✅ Combined result: {combined_mask.sum()} / {len(combined_mask)} candles match")

    return combined_mask


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'evaluate_condition_vectorized',
    'build_condition_mask',
    'build_conditions_mask',
]
