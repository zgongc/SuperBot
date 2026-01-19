"""
indicators/community - TradingView Community Indicators

Version: 1.0.0
Date: 2025-12-12

Description:
    Community indicators ported from TradingView.

    The indicators in this folder:
    - Are translated from the original PineScript code to Python
    - Are integrated into the SuperBot indicator system
    - The calculate(), calculate_batch() and update() methods are compatible

Available Indicators:
    - MavilimW: A trend indicator based on WMA by Kivanc Ozbilgic.
"""

from .mavilimw import MavilimW

__all__ = [
    'MavilimW',
]
