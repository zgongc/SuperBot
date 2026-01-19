"""
indicators/combo/__init__.py - Combo (Combined) Indicator Module

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Combining multiple indicators to generate strong signals.
    A collection of combined indicators.

    This category combines different indicator types (trend, momentum, volume)
    to provide more reliable buy/sell signals.

Available Indicators:
    1. RSIBollinger - RSI + Bollinger Bands kombinasyonu
    2. MACDRSICombo - MACD + RSI kombinasyonu
    3. EMARibbon - Multiple EMA bands (5, 10, 20, 50, 100, 200)
    4. TripleScreen - Elder's famous 3-screen system
    5. SmartMoney - Smart Money Concept (SMC) analizi

Usage Examples:
    >>> from indicators.combo import RSIBollinger, MACDRSICombo
    >>>
    >>> # RSI + Bollinger kombinasyonu
    >>> combo1 = RSIBollinger(rsi_period=14, bb_period=20)
    >>> result1 = combo1.calculate(data)
    >>>
    >>> # MACD + RSI kombinasyonu
    >>> combo2 = MACDRSICombo(macd_fast=12, rsi_period=14)
    >>> result2 = combo2.calculate(data)
    >>>
    >>> # EMA Ribbon
    >>> ribbon = EMARibbon(ema_periods=[5, 10, 20, 50, 100, 200])
    >>> result3 = ribbon.calculate(data)
    >>>
    >>> # Triple Screen
    >>> ts = TripleScreen(use_macd=True)
    >>> result4 = ts.calculate(data)
    >>>
    >>> # Smart Money
    >>> smc = SmartMoney(adx_threshold=25)
    >>> result5 = smc.calculate(data)

Features:
    - Multiple indicator confirmation
    - Powerful signal filtering
    - Risk management integration
    - High accuracy rate

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
    - indicators.momentum
    - indicators.trend
    - indicators.volatility
    - indicators.volume
"""

from indicators.combo.rsi_bollinger import RSIBollinger
from indicators.combo.macd_rsi import MACDRSICombo
from indicators.combo.ema_ribbon import EMARibbon
from indicators.combo.triple_screen import TripleScreen
from indicators.combo.smart_money import SmartMoney
from indicators.combo.smart_grok import SmartGrok


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Combo indicators
    'RSIBollinger',      # RSI + Bollinger Bands
    'MACDRSICombo',      # MACD + RSI
    'EMARibbon',         # Multiple EMA bands
    'TripleScreen',      # Elder's 3-screen system
    'SmartMoney',        # Smart Money Concept
    'SmartGrok',         # Smart Money Concept - Improved
]


# ============================================================================
# METADATA
# ============================================================================

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__category__ = 'combo'
__description__ = 'Combined technical indicators - Multiple indicator combinations'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_combo_indicators():
    """
    Returns the list of all combo indicators.

    Returns:
        list: A list of Combo indicator classes.
    """
    return [
        RSIBollinger,
        MACDRSICombo,
        EMARibbon,
        TripleScreen,
        SmartMoney,
        SmartGrok,
    ]


def get_combo_indicator_names():
    """
    Returns all combo indicator names.

    Returns:
        list: A list of indicator names.
    """
    return [cls.__name__ for cls in get_combo_indicators()]


def get_combo_indicator_info():
    """
    Returns detailed information about all combo indicators.

    Returns:
        dict: Indicator names and descriptions
    """
    return {
        'RSIBollinger': {
            'description': 'RSI + Bollinger Bands kombinasyonu',
            'components': ['RSI', 'Bollinger Bands'],
            'category': 'Momentum + Volatility',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'MACDRSICombo': {
            'description': 'MACD + RSI kombinasyonu',
            'components': ['MACD', 'RSI'],
            'category': 'Trend + Momentum',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'EMARibbon': {
            'description': 'Multiple EMA bands (5, 10, 20, 50, 100, 200)',
            'components': ['EMA (multiple)'],
            'category': 'Trend',
            'output_type': 'LINES',
            'requires_volume': False
        },
        'TripleScreen': {
            'description': "Elder's 3-screen trading system",
            'components': ['MACD/EMA', 'RSI', 'Price Action'],
            'category': 'Multi-timeframe System',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': False
        },
        'SmartMoney': {
            'description': 'Smart Money Concept (SMC) analizi',
            'components': ['OBV', 'RSI', 'ADX', 'Market Structure'],
            'category': 'Volume + Momentum + Trend',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': True
        },
        'SmartGrok': {
            'description': 'Smart Money Concept (SMC) - Improved',
            'components': ['FVG', 'Order Blocks', 'BOS/CHoCH', 'Market Structure'],
            'category': 'Volume + Structure',
            'output_type': 'MULTIPLE_VALUES',
            'requires_volume': True
        }
    }


def create_combo_indicator(name: str, **params):
    """
    Create a combo indicator based on the name.

    Args:
        name: Indicator name
        **params: Indicator parameters

    Returns:
        BaseIndicator: The created indicator.

    Raises:
        ValueError: Invalid indicator name

    Example:
        >>> indicator = create_combo_indicator('RSIBollinger', rsi_period=14)
        >>> result = indicator.calculate(data)
    """
    indicators_map = {
        'RSIBollinger': RSIBollinger,
        'MACDRSICombo': MACDRSICombo,
        'EMARibbon': EMARibbon,
        'TripleScreen': TripleScreen,
        'SmartMoney': SmartMoney,
        'SmartGrok': SmartGrok,
    }

    if name not in indicators_map:
        available = ', '.join(indicators_map.keys())
        raise ValueError(
            f"Invalid combo indicator name: '{name}'. "
            f"Available indicators: {available}"
        )

    return indicators_map[name](**params)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Module initialization message (development mode)
import os
if os.getenv('SUPERBOT_DEBUG'):
    print(f"[COMBO] Loaded {len(__all__)} combo indicators")
