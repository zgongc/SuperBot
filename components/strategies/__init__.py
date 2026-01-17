#!/usr/bin/env python3
"""
components/strategies/__init__.py
SuperBot - Strategy Package

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Portable, modular strategy system for backtesting and live trading.
"""

from components.strategies.base_strategy import (
    BaseStrategy,
    TradingSide,
    PositionSizeMethod,
    ExitMethod,
    StopLossMethod,
    SymbolConfig,
    TechnicalParameters,
    RiskManagement,
    ExitStrategy,
    PositionManagement,
)

from components.strategies.signal_validator import SignalValidator
from components.strategies.exit_manager import ExitManager
from components.strategies.market_validator import MarketValidator
from components.strategies.pattern_generator import PatternGenerator
from components.strategies.portfolio_coordinator import PortfolioCoordinator
from components.strategies.strategy_executor import StrategyExecutor
from components.strategies.strategy_manager import StrategyManager

__all__ = [
    # Base
    'BaseStrategy',
    
    # Enums
    'TradingSide',
    'PositionSizeMethod',
    'ExitMethod',
    'StopLossMethod',
    
    # Config Types
    'SymbolConfig',
    'TechnicalParameters',
    'RiskManagement',
    'ExitStrategy',
    'PositionManagement',
    
    # Managers
    'SignalValidator',
    'ExitManager',
    'MarketValidator',
    'PatternGenerator',
    'PortfolioCoordinator',
    
    # Orchestration
    'StrategyExecutor',
    'StrategyManager',
]

__version__ = '1.0.0'

