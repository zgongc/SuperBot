"""
indicators/support_resistance/__init__.py - Support/Resistance Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Support/Resistance indicator package.
    This package contains indicators that identify support and resistance levels.

Contents:
    - PivotPoints: Classic pivot seviyeleri (P, R1-R3, S1-S3)
    - FibonacciPivot: Pivot levels based on Fibonacci ratios
    - Camarilla: Camarilla pivot formula
    - Woodie: Woodie pivot formula
    - ZigZag: Swing high/low points
    - SupportResistance: Automatic support/resistance detection
    - FibonacciRetracement: Fibonacci retracement levels
    - SwingPoints: Swing high and low levels

Usage:
    from indicators.support_resistance import (
        PivotPoints,
        FibonacciPivot,
        Camarilla,
        Woodie,
        ZigZag,
        SupportResistance,
        FibonacciRetracement,
        SwingPoints
    )

    # Pivot Points example
    pivot = PivotPoints(period=1)
    result = pivot(data)
    print(f"Pivot seviyeleri: {result.value}")

    # Fibonacci Retracement example
    fib = FibonacciRetracement(lookback=50)
    result = fib(data)
    print(f"Fibonacci seviyeleri: {result.value}")

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from indicators.support_resistance.pivotpoints import PivotPoints
from indicators.support_resistance.fibonacci_pivot import FibonacciPivot
from indicators.support_resistance.camarilla import Camarilla
from indicators.support_resistance.woodie import Woodie
from indicators.support_resistance.zigzag import ZigZag
from indicators.support_resistance.sr import SupportResistance
from indicators.support_resistance.fib_retracement import FibonacciRetracement
from indicators.support_resistance.swingpoints import SwingPoints

# Paket versiyonu
__version__ = '2.0.0'

# Exported classes
__all__ = [
    # Pivot-based indicators
    'PivotPoints',
    'FibonacciPivot',
    'Camarilla',
    'Woodie',

    # Swing-based indicators
    'ZigZag',
    'SwingPoints',

    # Level detection indicators
    'SupportResistance',
    'FibonacciRetracement',
]

# Indicator category information
CATEGORY = 'support_resistance'
CATEGORY_NAME = 'Support/Resistance'
CATEGORY_DESCRIPTION = 'Indicators that identify support and resistance levels'

# List of indicators and their descriptions
INDICATORS = {
    'PivotPoints': {
        'name': 'Pivot Points',
        'description': 'Classic pivot seviyeleri (P, R1-R3, S1-S3)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'FibonacciPivot': {
        'name': 'Fibonacci Pivot Points',
        'description': 'Pivot levels based on Fibonacci ratios',
        'type': 'LEVELS',
        'params': ['period']
    },
    'Camarilla': {
        'name': 'Camarilla Pivot Points',
        'description': 'Camarilla pivot formula (R1-R4, S1-S4)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'Woodie': {
        'name': 'Woodie Pivot Points',
        'description': 'Woodie pivot formula (Close weighted)',
        'type': 'LEVELS',
        'params': ['period']
    },
    'ZigZag': {
        'name': 'ZigZag',
        'description': 'Swing high/low points',
        'type': 'SINGLE_VALUE',
        'params': ['deviation', 'depth']
    },
    'SwingPoints': {
        'name': 'Swing Points',
        'description': 'Swing high and low levels',
        'type': 'LEVELS',
        'params': ['left_bars', 'right_bars', 'lookback']
    },
    'SupportResistance': {
        'name': 'Support/Resistance',
        'description': 'Automatic support/resistance detection',
        'type': 'LEVELS',
        'params': ['lookback', 'num_levels', 'tolerance']
    },
    'FibonacciRetracement': {
        'name': 'Fibonacci Retracement',
        'description': 'Fibonacci retracement levels (0-100%)',
        'type': 'LEVELS',
        'params': ['lookback']
    },
}


def get_indicator_list():
    """Returns a list of all indicators in the category."""
    return list(INDICATORS.keys())


def get_indicator_info(indicator_name: str) -> dict:
    """
    Returns the information of the specified indicator.

    Args:
        indicator_name: Indicator name

    Returns:
        dict: Indicator information
    """
    return INDICATORS.get(indicator_name, None)


def get_category_info() -> dict:
    """Returns category information"""
    return {
        'category': CATEGORY,
        'name': CATEGORY_NAME,
        'description': CATEGORY_DESCRIPTION,
        'indicator_count': len(INDICATORS),
        'indicators': list(INDICATORS.keys())
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Support/Resistance package test and information"""

    print("\n" + "="*70)
    print("SUPPORT/RESISTANCE INDICATORS PACKAGE")
    print("="*70 + "\n")

    # Kategori bilgisi
    print("1. Kategori Bilgisi:")
    cat_info = get_category_info()
    print(f"   Kategori: {cat_info['name']}")
    print(f"   Description: {cat_info['description']}")
    print(f"   Indicator Count: {cat_info['indicator_count']}")

    # Indicator list
    print("\n2. Current Indicators:")
    for i, name in enumerate(get_indicator_list(), 1):
        info = get_indicator_info(name)
        print(f"   {i}. {info['name']}")
        print(f"      - Description: {info['description']}")
        print(f"      - Tip: {info['type']}")
        print(f"      - Parameters: {', '.join(info['params'])}")

    # Indicator grouping
    print("\n3. Indicator Groups:")

    pivot_indicators = ['PivotPoints', 'FibonacciPivot', 'Camarilla', 'Woodie']
    print(f"\n   a) Pivot Based ({len(pivot_indicators)}):")
    for name in pivot_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    swing_indicators = ['ZigZag', 'SwingPoints']
    print(f"\n   b) Swing-based ({len(swing_indicators)}):")
    for name in swing_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    level_indicators = ['SupportResistance', 'FibonacciRetracement']
    print(f"\n   c) Level Detection ({len(level_indicators)}):")
    for name in level_indicators:
        info = get_indicator_info(name)
        print(f"      - {info['name']}: {info['description']}")

    # Usage examples
    print("\n4. Quick Usage Examples:")

    print("\n   a) Pivot Points:")
    print("      from indicators.support_resistance import PivotPoints")
    print("      pivot = PivotPoints(period=1)")
    print("      result = pivot(data)")
    print("      print(result.value)  # {'R3': 105.5, 'R2': 103.2, ...}")

    print("\n   b) Fibonacci Retracement:")
    print("      from indicators.support_resistance import FibonacciRetracement")
    print("      fib = FibonacciRetracement(lookback=50)")
    print("      result = fib(data)")
    print("      print(result.value)  # {'Fib_0.0': 110, 'Fib_23.6': 108.2, ...}")

    print("\n   c) ZigZag:")
    print("      from indicators.support_resistance import ZigZag")
    print("      zigzag = ZigZag(deviation=5.0, depth=12)")
    print("      result = zigzag(data)")
    print("      print(result.value)  # Print the last pivot value")

    print("\n   d) Support/Resistance:")
    print("      from indicators.support_resistance import SupportResistance")
    print("      sr = SupportResistance(lookback=50, num_levels=5)")
    print("      result = sr(data)")
    print("      print(result.value)  # {'R1': 105, 'R2': 107, 'S1': 98, ...}")

    # Tips
    print("\n5. Usage Tips:")
    print("   - Pivot indicators are generally used with daily data")
    print("   - Suitable for Fibonacci Retracement trend movements")
    print("   - Used to filter ZigZag noise")
    print("   - Ideal for automatic support/resistance level detection")
    print("   - Swing Points are used for local max/min points")
    print("   - Accuracy increases by using multiple indicators together")

    print("\n" + "="*70)
    print(f"Total {len(INDICATORS)} Support/Resistance indicator ready!")
    print("="*70 + "\n")
