#!/usr/bin/env python3
"""
components/datamanager/services/__init__.py
SuperBot - Services Package Exports
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Central exports for all datamanager services.

Usage:
    from components.datamanager.services import (
        TradingService, Trade,
        PositionService, Position,
        PortfolioService, Portfolio,
        # ... etc
    )
"""

# =============================================================================
# Trading Service
# =============================================================================
from components.datamanager.services.trading import (
    Candle,
    Trade,
    Order,
    PnL,
    Balance,
    TradingService,
)

# =============================================================================
# Position Service
# =============================================================================
from components.datamanager.services.position import (
    Position,
    PositionService,
)

# =============================================================================
# Favorites Service
# =============================================================================
from components.datamanager.services.favorites import (
    SymbolFavorite,
    FavoritesService,
)

# =============================================================================
# Symbols Service
# =============================================================================
from components.datamanager.services.symbols import (
    SymbolMetadata,
    Watchlist,
    ExchangeSymbol,
    SymbolsService,
)

# =============================================================================
# Analysis Service
# =============================================================================
from components.datamanager.services.analysis import (
    AnalysisQueue,
    AnalysisResult,
    AnalysisAlert,
    AlertNotification,
    AnalysisService,
)

# =============================================================================
# Categories Service
# =============================================================================
from components.datamanager.services.categories import (
    Category,
    CategorySymbol,
    CategoriesService,
)

# =============================================================================
# Account Service
# =============================================================================
from components.datamanager.services.account import (
    AccountInfo,
    AccountService,
)

# =============================================================================
# Exchange Service
# =============================================================================
from components.datamanager.services.exchange import (
    ExchangeAccount,
    ExchangeService,
)

# =============================================================================
# Portfolio Service
# =============================================================================
from components.datamanager.services.portfolio import (
    PortfolioHolding,
    Portfolio,
    PortfolioPosition,
    PortfolioPositionEntry,
    PortfolioExitTarget,
    PositionTrade,
    PortfolioTransaction,
    PortfolioService,
)

# =============================================================================
# Strategy Service
# =============================================================================
from components.datamanager.services.strategy import (
    Strategy,
    StrategyComponent,
    BacktestRun,
    BacktestTrade,
    LiveTrade,
    TradingSession,
    StrategyService,
)

# =============================================================================
# Storage Service
# =============================================================================
from components.datamanager.services.storage import (
    KlineDataRegistry,
    LiveKlineBuffer,
    StorageService,
)

# =============================================================================
# Config Service
# =============================================================================
from components.datamanager.services.config import (
    SystemSetting,
    ConfigFile,
    ConfigService,
)

# =============================================================================
# Utils Service
# =============================================================================
from components.datamanager.services.utils import (
    UtilsService,
)

# =============================================================================
# All exports
# =============================================================================
__all__ = [
    # Trading
    "Candle",
    "Trade",
    "Order",
    "PnL",
    "Balance",
    "TradingService",
    # Position
    "Position",
    "PositionService",
    # Favorites
    "SymbolFavorite",
    "FavoritesService",
    # Symbols
    "SymbolMetadata",
    "Watchlist",
    "ExchangeSymbol",
    "SymbolsService",
    # Analysis
    "AnalysisQueue",
    "AnalysisResult",
    "AnalysisAlert",
    "AlertNotification",
    "AnalysisService",
    # Categories
    "Category",
    "CategorySymbol",
    "CategoriesService",
    # Account
    "AccountInfo",
    "AccountService",
    # Exchange
    "ExchangeAccount",
    "ExchangeService",
    # Portfolio
    "PortfolioHolding",
    "Portfolio",
    "PortfolioPosition",
    "PortfolioPositionEntry",
    "PortfolioExitTarget",
    "PositionTrade",
    "PortfolioTransaction",
    "PortfolioService",
    # Strategy
    "Strategy",
    "StrategyComponent",
    "BacktestRun",
    "BacktestTrade",
    "LiveTrade",
    "TradingSession",
    "StrategyService",
    # Storage
    "KlineDataRegistry",
    "LiveKlineBuffer",
    "StorageService",
    # Config
    "SystemSetting",
    "ConfigFile",
    "ConfigService",
    # Utils
    "UtilsService",
]
