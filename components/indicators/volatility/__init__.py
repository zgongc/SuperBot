"""
indicators/volatility/__init__.py - Volatility Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Description:
    Volatility indicators package
    Indicators that measure the magnitude and fluctuation of price movements

Indicators:
    - ATR: Average True Range - Average true range
    - BollingerBands: Bollinger Bands - Volatility bands (SMA + StdDev)
    - KeltnerChannel: Keltner Channel - Volatility bands based on ATR
    - StandardDeviation: Standard Deviation - Standart sapma
    - TrueRange: True Range - Actual price range
    - NATR: Normalized ATR - Normalized ATR (percentage)
    - ChandelierExit: Chandelier Exit - ATR-based trailing stop
    - TTMSqueeze: TTM Squeeze - Volatility squeeze indicator

Dependencies:
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from indicators.volatility.atr import ATR
from indicators.volatility.bollinger import BollingerBands
from indicators.volatility.keltner import KeltnerChannel
from indicators.volatility.standard_dev import StandardDeviation
from indicators.volatility.true_range import TrueRange
from indicators.volatility.natr import NATR
from indicators.volatility.chandelier import ChandelierExit
from indicators.volatility.squeeze import TTMSqueeze

__all__ = [
    'ATR',
    'BollingerBands',
    'KeltnerChannel',
    'StandardDeviation',
    'TrueRange',
    'NATR',
    'ChandelierExit',
    'TTMSqueeze'
]

__version__ = '2.0.0'
__author__ = 'SuperBot Team'
__date__ = '2025-10-14'
