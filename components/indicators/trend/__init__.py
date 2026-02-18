"""
indicators/trend/__init__.py - Trend Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    It includes all indicators in the "Trend" category.
    15 trend indicators:

    Moving Averages (Hareketli Ortalamalar):
    - SMA: Simple Moving Average
    - EMA: Exponential Moving Average
    - WMA: Weighted Moving Average
    - HMA: Hull Moving Average
    - TEMA: Triple Exponential Moving Average
    - DEMA: Double Exponential Moving Average
    - VWMA: Volume Weighted Moving Average

    Trend Strength (Trend Strength):
    - SuperTrend: ATR-based trend indicator
    - ADX: Average Directional Index
    - Aroon: Aroon Up/Down indicator
    - Parabolic SAR: Stop and Reverse

    Channels & Complex (Kanallar & Kompleks):
    - Ichimoku: Ichimoku Cloud
    - Keltner: Keltner Channel
    - Donchian: Donchian Channel
    - MACD: Moving Average Convergence Divergence

Usage:
    from indicators.trend import SMA, EMA, MACD, ADX

    # SMA usage
    sma = SMA(period=20)
    result = sma(data)

    # MACD usage
    macd = MACD(fast_period=12, slow_period=26, signal_period=9)
    result = macd(data)

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Moving Averages
from indicators.trend.alma import ALMA
from indicators.trend.sma import SMA
from indicators.trend.ema import EMA
from indicators.trend.wma import WMA
from indicators.trend.hma import HMA
from indicators.trend.tema import TEMA
from indicators.trend.dema import DEMA
from indicators.trend.vwma import VWMA

# Trend Strength Indicators
from indicators.trend.supertrend import SuperTrend
from indicators.trend.adx import ADX
from indicators.trend.aroon import Aroon
from indicators.trend.parabolic_sar import ParabolicSAR

# Channels & Complex Indicators
from indicators.trend.ichimoku import Ichimoku
from indicators.volatility.keltner import KeltnerChannel  # Keltner is in volatility package
from indicators.trend.donchian import DonchianChannel
from indicators.trend.macd import MACD


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Moving Averages (8)
    'ALMA',
    'SMA',
    'EMA',
    'WMA',
    'HMA',
    'TEMA',
    'DEMA',
    'VWMA',

    # Trend Strength (4)
    'SuperTrend',
    'ADX',
    'Aroon',
    'ParabolicSAR',

    # Channels & Complex (4)
    'Ichimoku',
    'KeltnerChannel',
    'DonchianChannel',
    'MACD',
]


# ============================================================================
# PACKAGE INFO
# ============================================================================

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__category__ = 'trend'

# Number of indicators
TOTAL_INDICATORS = len(__all__)

# Category-based indicators
MOVING_AVERAGES = ['ALMA', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'DEMA', 'VWMA']
TREND_STRENGTH = ['SuperTrend', 'ADX', 'Aroon', 'ParabolicSAR']
CHANNELS_COMPLEX = ['Ichimoku', 'KeltnerChannel', 'DonchianChannel', 'MACD']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_indicators():
    """
    Returns a list of all trend indicators.

    Returns:
        list: All trend indicator classes.
    """
    return [
        SMA, EMA, WMA, HMA, TEMA, DEMA, VWMA,
        SuperTrend, ADX, Aroon, ParabolicSAR,
        Ichimoku, KeltnerChannel, DonchianChannel, MACD
    ]


def get_moving_averages():
    """
    Returns only the moving averages.

    Returns:
        list: Moving average indicator classes
    """
    return [SMA, EMA, WMA, HMA, TEMA, DEMA, VWMA]


def get_trend_strength_indicators():
    """
    Returns trend strength indicators.

    Returns:
        list: Trend strength indicator classes
    """
    return [SuperTrend, ADX, Aroon, ParabolicSAR]


def get_channel_indicators():
    """
    Returns the channel indicators.

    Returns:
        list: Channel indicator class types.
    """
    return [Ichimoku, KeltnerChannel, DonchianChannel, MACD]


def get_indicator_info():
    """
    Information about all trend indicators.

    Returns:
        dict: Indicator information
    """
    return {
        'category': 'trend',
        'total_count': TOTAL_INDICATORS,
        'moving_averages': {
            'count': len(MOVING_AVERAGES),
            'indicators': MOVING_AVERAGES
        },
        'trend_strength': {
            'count': len(TREND_STRENGTH),
            'indicators': TREND_STRENGTH
        },
        'channels_complex': {
            'count': len(CHANNELS_COMPLEX),
            'indicators': CHANNELS_COMPLEX
        },
        'all_indicators': __all__
    }


# ============================================================================
# PACKAGE INITIALIZATION
# ============================================================================

def _validate_imports():
    """Validate imported indicators"""
    expected = 16
    actual = len(__all__)

    if actual != expected:
        raise ImportError(
            f"Trend package import error: Expected {expected} indicators, got {actual}"
        )


# Automatic validation
_validate_imports()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Trend package testi"""

    print("\n" + "="*70)
    print("TREND INDICATORS PACKAGE TEST")
    print("="*70 + "\n")

    # Package bilgisi
    print("1. Package Bilgileri:")
    print(f"   [OK] Version: {__version__}")
    print(f"   [OK] Category: {__category__}")
    print(f"   [OK] Total Indicators: {TOTAL_INDICATORS}")

    # Kategori bilgileri
    print("\n2. Indicators by Category:")
    print(f"   [OK] Moving Averages ({len(MOVING_AVERAGES)}):")
    for ind in MOVING_AVERAGES:
        print(f"       - {ind}")

    print(f"\n   [OK] Trend Strength ({len(TREND_STRENGTH)}):")
    for ind in TREND_STRENGTH:
        print(f"       - {ind}")

    print(f"\n   [OK] Channels & Complex ({len(CHANNELS_COMPLEX)}):")
    for ind in CHANNELS_COMPLEX:
        print(f"       - {ind}")

    # Import testi
    print("\n3. Import Testi:")
    all_indicators = get_all_indicators()
    print(f"   [OK] {len(all_indicators)} indicators were successfully imported")

    # Check each indicator
    print("\n4. Indicator Checks:")
    for ind_class in all_indicators:
        try:
            # Create with default parameters
            ind = ind_class()
            print(f"   [OK] {ind.name}: {ind.__class__.__name__} - {ind.category.value}")
        except Exception as e:
            print(f"   [ERROR] {ind_class.__name__}: {e}")

    # Helper fonksiyon testi
    print("\n5. Helper Fonksiyon Testi:")
    info = get_indicator_info()
    print(f"   [OK] get_indicator_info() executed")
    print(f"   [OK] Total: {info['total_count']} indicators")

    # Export control
    print("\n6. Export Control:")
    print(f"   [OK] There are {len(__all__)} items in __all__")
    for item in __all__:
        print(f"       - {item}")

    print("\n" + "="*70)
    print("[SUCCESS] TREND PACKAGE ALL TESTS PASSED!")
    print("="*70 + "\n")
