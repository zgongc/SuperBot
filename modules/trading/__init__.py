#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/trading/__init__.py

SuperBot - Trading Module Package
Date: 2025-01-26

Main trading engine and supporting components for strategy execution.

Components:
    - TradingEngine: Main orchestrator for trading operations
    - TierManager: Symbol tier-based processing management
    - PriceFeed: Real-time price data handling
    - DisplayInfo: Trading display/UI information
    - TradeLogger: Trade logging and persistence

Modes:
    - PAPER: Real data, virtual money, simulated orders
    - DEMO: Real data, virtual money, testnet orders
    - LIVE: Real data, real money, production orders
    - REPLAY: Historical data, virtual money, simulated orders

Usage:
    from modules.trading import TradingEngine
    from modules.trading.modes import PaperMode, select_mode

    engine = TradingEngine(mode="paper", strategy_path="my_strategy.py")
    await engine.start()
"""

from modules.trading.trading_engine import TradingEngine
from modules.trading.tier_manager import TierManager, TierLevel
from modules.trading.price_feed import PriceFeed
from modules.trading.display_info import DisplayInfo
from modules.trading.trade_logger import TradeLogger

__all__ = [
    "TradingEngine",
    "TierManager",
    "TierLevel",
    "PriceFeed",
    "DisplayInfo",
    "TradeLogger",
]
