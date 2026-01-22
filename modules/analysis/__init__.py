"""
modules/analysis - Market Structure Analysis Module

Detects SMC formations from the given candle data:
- BOS (Break of Structure)
- CHoCH (Change of Character)
- FVG (Fair Value Gap)
- Order Blocks
- Swing High/Low
- Liquidity Zones

Usage:
    from modules.analysis import AnalysisEngine

    engine = AnalysisEngine()
    result = engine.analyze(df)
"""

from .analysis_engine import AnalysisEngine
from .models import (
    BOSFormation,
    CHoCHFormation,
    FVGFormation,
    SwingPoint,
    OrderBlockFormation,
    AnalysisResult
)

__all__ = [
    'AnalysisEngine',
    'BOSFormation',
    'CHoCHFormation',
    'FVGFormation',
    'SwingPoint',
    'OrderBlockFormation',
    'AnalysisResult'
]

__version__ = '1.0.0'
