#!/usr/bin/env python3
"""
components/strategies/helpers/timeframe_utils.py
SuperBot - Timeframe Utilities

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Multi-timeframe helper functions.
    
    - Timeframe conversion
    - Timeframe comparison
    - Bar alignment

Usage:
    from components.strategies.helpers.timeframe_utils import timeframe_to_minutes
    
    minutes = timeframe_to_minutes('15m')  # 15
    minutes = timeframe_to_minutes('1h')   # 60
"""

from typing import List, Optional


# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '6h': 360,
    '12h': 720,
    '1d': 1440,
    '3d': 4320,
    '1w': 10080,
    '1M': 43200,  # Approximate (30 days)
}


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert the timeframe to minutes.
    
    Args:
        timeframe: Timeframe string ('1m', '5m', '1h', etc.)
    
    Returns:
        int: In minutes.
    
    Raises:
        ValueError: Invalid timeframe
    """
    timeframe = timeframe.lower().strip()
    
    if timeframe in TIMEFRAME_MINUTES:
        return TIMEFRAME_MINUTES[timeframe]
    
    raise ValueError(
        f"Invalid timeframe: '{timeframe}'\n"
        f"Valid timeframes: {sorted(TIMEFRAME_MINUTES.keys())}"
    )


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert the timeframe to seconds.
    
    Args:
        timeframe: Timeframe string
    
    Returns:
        int: Saniye cinsinden
    """
    return timeframe_to_minutes(timeframe) * 60


def timeframe_to_milliseconds(timeframe: str) -> int:
    """
    Convert the timeframe to milliseconds.
    
    Args:
        timeframe: Timeframe string
    
    Returns:
        int: Millisaniye cinsinden
    """
    return timeframe_to_minutes(timeframe) * 60 * 1000


def compare_timeframes(tf1: str, tf2: str) -> int:
    """
    Compare two timeframes.
    
    Args:
        tf1: First timeframe
        tf2: Second timeframe
    
    Returns:
        -1 if tf1 < tf2
         0 if tf1 == tf2
         1 if tf1 > tf2
    """
    minutes1 = timeframe_to_minutes(tf1)
    minutes2 = timeframe_to_minutes(tf2)
    
    if minutes1 < minutes2:
        return -1
    elif minutes1 > minutes2:
        return 1
    else:
        return 0


def is_higher_timeframe(tf: str, base_tf: str) -> bool:
    """
    tf is greater than base_tf?
    
    Args:
        tf: The timeframe to be checked.
        base_tf: Base timeframe
    
    Returns:
        True if tf > base_tf
    """
    return compare_timeframes(tf, base_tf) > 0


def is_lower_timeframe(tf: str, base_tf: str) -> bool:
    """
    tf is less than base_tf?
    
    Args:
        tf: The timeframe to be checked.
        base_tf: Base timeframe
    
    Returns:
        True if tf < base_tf
    """
    return compare_timeframes(tf, base_tf) < 0


def sort_timeframes(timeframes: List[str], reverse: bool = False) -> List[str]:
    """
    Sort the timeframes.
    
    Args:
        timeframes: List of timeframes
        reverse: True if sorting in descending order.
    
    Returns:
        Sorted timeframe list
    """
    return sorted(
        timeframes,
        key=lambda tf: timeframe_to_minutes(tf),
        reverse=reverse
    )


def get_higher_timeframes(base_tf: str, all_timeframes: List[str]) -> List[str]:
    """
    Get higher timeframes than the base timeframe.
    
    Args:
        base_tf: Base timeframe
        all_timeframes: All timeframes
    
    Returns:
        Higher timeframe'ler (sorted)
    """
    base_minutes = timeframe_to_minutes(base_tf)
    higher = [
        tf for tf in all_timeframes
        if timeframe_to_minutes(tf) > base_minutes
    ]
    return sort_timeframes(higher)


def get_lower_timeframes(base_tf: str, all_timeframes: List[str]) -> List[str]:
    """
    Get timeframes lower than the base timeframe.
    
    Args:
        base_tf: Base timeframe
        all_timeframes: All timeframes
    
    Returns:
        Lower timeframe'ler (sorted)
    """
    base_minutes = timeframe_to_minutes(base_tf)
    lower = [
        tf for tf in all_timeframes
        if timeframe_to_minutes(tf) < base_minutes
    ]
    return sort_timeframes(lower)


def calculate_bars_needed(target_tf: str, source_tf: str) -> int:
    """
    How many source bars are required for the target timeframe?
    
    Args:
        target_tf: Target timeframe (e.g., '1h')
        source_tf: Source timeframe (e.g., '15m')
    
    Returns:
        int: The required number of bars.
    
    Example:
        calculate_bars_needed('1h', '15m')  # 4 (1h = 4x15m)
    """
    target_minutes = timeframe_to_minutes(target_tf)
    source_minutes = timeframe_to_minutes(source_tf)
    
    if target_minutes < source_minutes:
        raise ValueError(
            f"Target timeframe ({target_tf}) source'dan ({source_tf}) cannot be smaller"
        )
    
    return target_minutes // source_minutes


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Is the timeframe valid?
    
    Args:
        timeframe: Timeframe string
    
    Returns:
        True if valid
    """
    return timeframe.lower().strip() in TIMEFRAME_MINUTES


def get_all_timeframes() -> List[str]:
    """
    Returns all valid timeframes.
    
    Returns:
        Sorted timeframe list
    """
    return sort_timeframes(list(TIMEFRAME_MINUTES.keys()))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
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

