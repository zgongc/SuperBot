#!/usr/bin/env python3
"""
components/strategies/helpers/operators.py
SuperBot - Condition Operators

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Koşul operatörlerinin implementasyonları.
    
    Desteklenen operatörler:
    - Karşılaştırma: >, <, >=, <=, ==, !=
    - Crossover/Crossunder: crossover, crossunder
    - Trend: rising, falling
    - Range: between, outside, near

Kullanım:
    from components.strategies.helpers.operators import evaluate_condition
    
    # Karşılaştırma
    result = evaluate_condition(45.5, '>', 30)  # True
    
    # Crossover
    result = evaluate_crossover([29, 31], 30)  # True (crossover yukarı)
    
    # Rising
    result = evaluate_rising([40, 42, 45], 3)  # True (3 bar rising)
"""

from typing import Any, List, Union, Optional
import numpy as np
import pandas as pd


# ============================================================================
# KARŞILAŞTIRMA OPERATÖRLERI
# ============================================================================

def compare_gt(left: Union[int, float], right: Union[int, float]) -> bool:
    """Büyüktür (>)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) > float(right)


def compare_lt(left: Union[int, float], right: Union[int, float]) -> bool:
    """Küçüktür (<)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) < float(right)


def compare_gte(left: Union[int, float], right: Union[int, float]) -> bool:
    """Büyük eşittir (>=)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) >= float(right)


def compare_lte(left: Union[int, float], right: Union[int, float]) -> bool:
    """Küçük eşittir (<=)"""
    # NaN check: if either value is NaN, return False
    if pd.isna(left) or pd.isna(right):
        return False
    return float(left) <= float(right)


def compare_eq(left: Any, right: Any) -> bool:
    """Eşittir (==)"""
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
    """Eşit değil (!=)"""
    return not compare_eq(left, right)


# ============================================================================
# CROSSOVER / CROSSUNDER
# ============================================================================

def evaluate_crossover(
    series: Union[List, pd.Series, np.ndarray],
    threshold: Union[float, List, pd.Series, np.ndarray]
) -> bool:
    """
    Crossover tespiti (yukarı kesme)
    
    Args:
        series: Indicator serisi (son 2 değer minimum)
        threshold: Kesilen değer veya seri
    
    Returns:
        True if crossover occurred (önceki bar altında, şimdi üstte)
    
    Örnek:
        series = [29, 31], threshold = 30 -> True (crossover)
        series = [31, 32], threshold = 30 -> False (zaten üstte)
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
    Crossunder tespiti (aşağı kesme)
    
    Args:
        series: Indicator serisi (son 2 değer minimum)
        threshold: Kesilen değer veya seri
    
    Returns:
        True if crossunder occurred (önceki bar üstte, şimdi altta)
    
    Örnek:
        series = [31, 29], threshold = 30 -> True (crossunder)
        series = [29, 28], threshold = 30 -> False (zaten altta)
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
# TREND OPERATÖRLERI
# ============================================================================

def evaluate_rising(
    series: Union[List, pd.Series, np.ndarray],
    periods: int
) -> bool:
    """
    Rising tespiti (yükseliyor mu?)
    
    Args:
        series: Indicator serisi
        periods: Kaç bar kontrol edilecek
    
    Returns:
        True if rising for N periods
    
    Örnek:
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
    Falling tespiti (düşüyor mu?)
    
    Args:
        series: Indicator serisi
        periods: Kaç bar kontrol edilecek
    
    Returns:
        True if falling for N periods
    
    Örnek:
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
# RANGE OPERATÖRLERI
# ============================================================================

def evaluate_between(
    value: Union[int, float],
    range_values: List[Union[int, float]]
) -> bool:
    """
    Between tespiti (aralıkta mı?)
    
    Args:
        value: Kontrol edilecek değer
        range_values: [min, max]
    
    Returns:
        True if min <= value <= max
    
    Örnek:
        value = 50, range_values = [30, 70] -> True
        value = 80, range_values = [30, 70] -> False
    """
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        raise ValueError(f"between operatörü için [min, max] gerekli, {range_values} verildi")
    
    min_val, max_val = range_values
    return float(min_val) <= float(value) <= float(max_val)


def evaluate_outside(
    value: Union[int, float],
    range_values: List[Union[int, float]]
) -> bool:
    """
    Outside tespiti (aralık dışında mı?)
    
    Args:
        value: Kontrol edilecek değer
        range_values: [min, max]
    
    Returns:
        True if value < min OR value > max
    
    Örnek:
        value = 80, range_values = [30, 70] -> True
        value = 50, range_values = [30, 70] -> False
    """
    if not isinstance(range_values, (list, tuple)) or len(range_values) != 2:
        raise ValueError(f"outside operatörü için [min, max] gerekli, {range_values} verildi")
    
    min_val, max_val = range_values
    return float(value) < float(min_val) or float(value) > float(max_val)


def evaluate_near(
    value: Union[int, float],
    target_and_threshold: List[Union[int, float]]
) -> bool:
    """
    Near tespiti (yakın mı?)
    
    Args:
        value: Kontrol edilecek değer
        target_and_threshold: [target, percent_threshold]
    
    Returns:
        True if value is within percent_threshold of target
    
    Örnek:
        value = 101, target_and_threshold = [100, 1] -> True (%1 içinde)
        value = 105, target_and_threshold = [100, 1] -> False
    """
    if not isinstance(target_and_threshold, (list, tuple)) or len(target_and_threshold) != 2:
        raise ValueError(
            f"near operatörü için [target, percent_threshold] gerekli, "
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
    Koşulu değerlendir (unified interface)
    
    Args:
        left: Sol operand (değer veya seri)
        operator: Operatör string
        right: Sağ operand (değer veya seri)
    
    Returns:
        bool: Koşul sonucu
    
    Raises:
        ValueError: Geçersiz operatör
    
    Örnek:
        evaluate_condition(45, '>', 30)  # True
        evaluate_condition([29, 31], 'crossover', 30)  # True
        evaluate_condition([40, 42, 45], 'rising', 3)  # True
    """
    # Karşılaştırma operatörleri
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
        raise ValueError(f"Desteklenmeyen operatör: '{operator}'")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_required_history(operator: str, right_operand: Any) -> int:
    """
    Operatör için gereken history bar sayısı
    
    Args:
        operator: Operatör string
        right_operand: Sağ operand
    
    Returns:
        int: Gereken bar sayısı
    """
    if operator in ('crossover', 'crossunder'):
        return 2
    
    if operator in ('rising', 'falling'):
        if isinstance(right_operand, int):
            return right_operand
        return 2  # Default
    
    # Diğer operatörler: 1 bar (mevcut)
    return 1


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Karşılaştırma
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

