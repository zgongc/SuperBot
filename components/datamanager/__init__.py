#!/usr/bin/env python3
"""
components/datamanager/__init__.py
SuperBot - DataManager Package
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Modular data management system with SQLAlchemy async backend.

Architecture:
    DataManager (facade)
    ├── TradingService      - Trade management
    ├── PositionService     - Position tracking
    ├── FavoritesService    - Favorites management
    ├── SymbolsService      - Symbol and timeframe management
    ├── AnalysisService     - Analysis results
    ├── CategoriesService   - Category management
    ├── AccountService      - Account snapshots
    ├── ExchangeService     - Exchange connections
    ├── PortfolioService    - Portfolio management
    ├── StrategyService     - Strategy and backtest management
    ├── StorageService      - Kline data storage
    └── ConfigService       - System settings

Usage:
    from components.datamanager import DataManager

    dm = DataManager({"backend": "sqlite", "path": "data/db/superbot.db"})
    await dm.start()

    # Access via facade
    await dm.trading.save_trade(...)
    await dm.portfolio.add_position(...)

    # Or import services directly
    from components.datamanager.services import TradingService, Trade

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
    - aiosqlite
"""

# Main facade
from components.datamanager.manager import DataManager

# Base classes
from components.datamanager.base import Base, DatabaseManager, BaseService

# All services and models
from components.datamanager.services import (
    # Trading
    Candle,
    Trade,
    Order,
    PnL,
    Balance,
    TradingService,
    # Position
    Position,
    PositionService,
    # Favorites
    SymbolFavorite,
    FavoritesService,
    # Symbols
    SymbolMetadata,
    Watchlist,
    ExchangeSymbol,
    SymbolsService,
    # Analysis
    AnalysisQueue,
    AnalysisResult,
    AnalysisAlert,
    AlertNotification,
    AnalysisService,
    # Categories
    Category,
    CategorySymbol,
    CategoriesService,
    # Account
    AccountInfo,
    AccountService,
    # Exchange
    ExchangeAccount,
    ExchangeService,
    # Portfolio
    PortfolioHolding,
    Portfolio,
    PortfolioPosition,
    PortfolioPositionEntry,
    PortfolioExitTarget,
    PositionTrade,
    PortfolioTransaction,
    PortfolioService,
    # Strategy
    Strategy,
    StrategyComponent,
    BacktestRun,
    BacktestTrade,
    LiveTrade,
    TradingSession,
    StrategyService,
    # Storage
    KlineDataRegistry,
    LiveKlineBuffer,
    StorageService,
    # Config
    SystemSetting,
    ConfigFile,
    ConfigService,
    # Utils
    UtilsService,
)

__all__ = [
    # Facade
    "DataManager",
    # Base
    "Base",
    "DatabaseManager",
    "BaseService",
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

__version__ = "1.0.0"
