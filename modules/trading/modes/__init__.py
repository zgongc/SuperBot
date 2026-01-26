#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/trading/modes/__init__.py

SuperBot - Trading Modes Package
Date: 2025-11-27

MODES:
    PAPER  - Real data, virtual money, simulated order
    DEMO   - Real data, virtual money, testnet order
    LIVE   - Real data, real money, production order
    REPLAY - Past data (parquet), virtual money, simulated order

Usage:
    from modules.trading.modes import select_mode, PaperMode

    # With factory
    mode = select_mode("paper", config)

    # Using direct import
    mode = PaperMode(config)
"""

from modules.trading.modes.base_mode import (
    BaseMode,
    ModeType,
    OrderSide,
    OrderType,
    OrderStatus,
    Candle,
    Order,
    OrderResult,
    Position,
    Balance,
    select_mode
)

from modules.trading.modes.paper_mode import PaperMode
from modules.trading.modes.demo_mode import DemoMode
from modules.trading.modes.live_mode import LiveMode
from modules.trading.modes.replay_mode import ReplayMode

__all__ = [
    # Base
    "BaseMode",
    "ModeType",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Candle",
    "Order",
    "OrderResult",
    "Position",
    "Balance",
    "select_mode",
    # Modes
    "PaperMode",
    "DemoMode",
    "LiveMode",
    "ReplayMode"
]
