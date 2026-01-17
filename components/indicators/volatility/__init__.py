"""
indicators/volatility/__init__.py - Volatility Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Volatilite indikatörleri paketi
    Fiyat hareketlerinin genişliğini ve dalgalanmasını ölçen indikatörler

İndikatörler:
    - ATR: Average True Range - Ortalama gerçek aralık
    - BollingerBands: Bollinger Bands - Volatilite bantları (SMA + StdDev)
    - KeltnerChannel: Keltner Channel - ATR tabanlı volatilite bantları
    - StandardDeviation: Standard Deviation - Standart sapma
    - TrueRange: True Range - Gerçek fiyat aralığı
    - NATR: Normalized ATR - Normalleştirilmiş ATR (yüzde)
    - ChandelierExit: Chandelier Exit - ATR tabanlı trailing stop
    - TTMSqueeze: TTM Squeeze - Volatilite sıkışması göstergesi

Bağımlılıklar:
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
