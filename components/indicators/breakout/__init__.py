"""
indicators/breakout/__init__.py - Breakout Indicators Package

Version: 2.0.0
Date: 2025-10-14
Author: SuperBot Team

Açıklama:
    Breakout kategorisi indikatörleri
    Konsolidasyon tespiti, volatilite breakout'ları ve range analizi

    İçerik (5 indikatör):
    - SqueezeMomentum: TTM Squeeze + momentum histogram
    - VolatilityBreakout: Bollinger genişlemesi ile breakout tespiti
    - BreakoutScanner: Multi-timeframe breakout analizi
    - RangeBreakout: Konsolidasyon range'i + breakout seviyeleri
    - Consolidation: Konsolidasyon tespiti ve puanı

Kullanım:
    from indicators.breakout import (
        SqueezeMomentum,
        VolatilityBreakout,
        BreakoutScanner,
        RangeBreakout,
        Consolidation
    )

    # TTM Squeeze
    squeeze = SqueezeMomentum(bb_period=20, kc_period=20)
    result = squeeze(data)

    # Volatility Breakout
    vb = VolatilityBreakout(std_dev=2.0, width_threshold=4.0)
    result = vb(data)

    # Breakout Scanner
    scanner = BreakoutScanner(lookback=20, confirmation=2)
    result = scanner(data)

    # Range Breakout
    rb = RangeBreakout(period=20, consolidation_threshold=3.0)
    result = rb(data)

    # Consolidation
    consol = Consolidation(period=20, atr_period=14)
    result = consol(data)
"""

from indicators.breakout.squeeze_momentum import SqueezeMomentum
from indicators.breakout.volatility_breakout import VolatilityBreakout
from indicators.breakout.breakout_scanner import BreakoutScanner
from indicators.breakout.range_breakout import RangeBreakout
from indicators.breakout.consolidation import Consolidation


__all__ = [
    'SqueezeMomentum',
    'VolatilityBreakout',
    'BreakoutScanner',
    'RangeBreakout',
    'Consolidation',
]


# Kategori bilgisi
CATEGORY = 'breakout'
INDICATORS = [
    'squeeze_momentum',
    'volatility_breakout',
    'breakout_scanner',
    'range_breakout',
    'consolidation',
]
