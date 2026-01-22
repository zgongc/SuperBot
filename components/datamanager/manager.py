#!/usr/bin/env python3
"""
components/datamanager/manager.py
SuperBot - DataManager Facade
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Unified facade for all datamanager services.
Provides backward-compatible interface while delegating to modular services.

Features:
- Single entry point for all data operations
- Lazy service initialization
- Backward compatibility with old DataManager API

Usage:
    from components.datamanager import DataManager

    dm = DataManager(config)
    await dm.start()

    # Access services directly
    trades = await dm.trading.get_trades(symbol="BTCUSDT")
    await dm.portfolio.add_position(...)

    # Or use convenience methods
    trades = await dm.get_trades(symbol="BTCUSDT")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, DatabaseManager
from components.datamanager.services import (
    TradingService,
    PositionService,
    FavoritesService,
    SymbolsService,
    AnalysisService,
    CategoriesService,
    AccountService,
    ExchangeService,
    PortfolioService,
    StrategyService,
    StorageService,
    ConfigService,
    UtilsService,
)
from core.logger_engine import get_logger

logger = get_logger("components.datamanager.manager")


class DataManager:
    """
    Unified facade for all datamanager services.

    This class provides a single entry point for all data operations,
    delegating to specialized service classes internally.

    Attributes:
        trading: TradingService - Trade management
        position: PositionService - Position tracking
        favorites: FavoritesService - Favorites management
        symbols: SymbolsService - Symbol and timeframe management
        analysis: AnalysisService - Analysis results
        categories: CategoriesService - Category management
        account: AccountService - Account snapshots
        exchange: ExchangeService - Exchange connections
        portfolio: PortfolioService - Portfolio management
        strategy: StrategyService - Strategy and backtest management
        storage: StorageService - Kline data storage
        config: ConfigService - System settings
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataManager

        Args:
            config: Database configuration dict
                - backend: "sqlite" or "postgresql"
                - path: SQLite database path
                - postgresql: PostgreSQL config dict
        """
        self.config = config or {}
        self._db = DatabaseManager(config)
        self._started = False

        # Services (lazy initialized)
        self._trading: Optional[TradingService] = None
        self._position: Optional[PositionService] = None
        self._favorites: Optional[FavoritesService] = None
        self._symbols: Optional[SymbolsService] = None
        self._analysis: Optional[AnalysisService] = None
        self._categories: Optional[CategoriesService] = None
        self._account: Optional[AccountService] = None
        self._exchange: Optional[ExchangeService] = None
        self._portfolio: Optional[PortfolioService] = None
        self._strategy: Optional[StrategyService] = None
        self._storage: Optional[StorageService] = None
        self._config: Optional[ConfigService] = None
        self._utils: Optional[UtilsService] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start database and initialize all services"""
        if self._started:
            return

        try:
            await self._db.start()
            self._init_services()
            self._started = True
            logger.info("✅ DataManager started")
        except Exception as e:
            logger.error(f"❌ DataManager start error: {e}")
            raise

    async def stop(self):
        """Stop database connection"""
        if not self._started:
            return

        try:
            await self._db.stop()
            self._started = False
            logger.info("🛑 DataManager stopped")
        except Exception as e:
            logger.error(f"❌ DataManager stop error: {e}")

    def _init_services(self):
        """Initialize all service instances"""
        self._trading = TradingService(self._db)
        self._position = PositionService(self._db)
        self._favorites = FavoritesService(self._db)
        self._symbols = SymbolsService(self._db)
        self._analysis = AnalysisService(self._db)
        self._categories = CategoriesService(self._db)
        self._account = AccountService(self._db)
        self._exchange = ExchangeService(self._db)
        self._portfolio = PortfolioService(self._db)
        self._strategy = StrategyService(self._db)
        self._storage = StorageService(self._db)
        self._config = ConfigService(self._db)
        self._utils = UtilsService(self._db)

    # =========================================================================
    # Service Properties
    # =========================================================================

    @property
    def trading(self) -> TradingService:
        """Get trading service"""
        if not self._trading:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._trading

    @property
    def position(self) -> PositionService:
        """Get position service"""
        if not self._position:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._position

    @property
    def favorites(self) -> FavoritesService:
        """Get favorites service"""
        if not self._favorites:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._favorites

    @property
    def symbols(self) -> SymbolsService:
        """Get symbols service"""
        if not self._symbols:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._symbols

    @property
    def analysis(self) -> AnalysisService:
        """Get analysis service"""
        if not self._analysis:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._analysis

    @property
    def categories(self) -> CategoriesService:
        """Get categories service"""
        if not self._categories:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._categories

    @property
    def account(self) -> AccountService:
        """Get account service"""
        if not self._account:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._account

    @property
    def exchange(self) -> ExchangeService:
        """Get exchange service"""
        if not self._exchange:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._exchange

    @property
    def portfolio(self) -> PortfolioService:
        """Get portfolio service"""
        if not self._portfolio:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._portfolio

    @property
    def strategy(self) -> StrategyService:
        """Get strategy service"""
        if not self._strategy:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._strategy

    @property
    def storage(self) -> StorageService:
        """Get storage service"""
        if not self._storage:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._storage

    @property
    def config_service(self) -> ConfigService:
        """Get config service (named config_service to avoid conflict with config dict)"""
        if not self._config:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._config

    @property
    def utils(self) -> UtilsService:
        """Get utils service"""
        if not self._utils:
            raise RuntimeError("DataManager not started. Call start() first.")
        return self._utils

    # =========================================================================
    # Database Access
    # =========================================================================

    @property
    def db(self) -> DatabaseManager:
        """Get underlying database manager"""
        return self._db

    @property
    def engine(self):
        """Get SQLAlchemy engine"""
        return self._db.engine

    def session(self):
        """Get async session context manager"""
        return self._db.async_session()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "backend": self._db.backend,
            "database_url": self._db.database_url.split("@")[-1] if "@" in self._db.database_url else self._db.database_url,
            "started": self._started
        }

    # =========================================================================
    # Convenience Methods (Backward Compatibility)
    # =========================================================================

    # --- Trading ---
    async def save_trade(self, **kwargs) -> bool:
        """Save a trade"""
        return await self.trading.save_trade(**kwargs)

    async def save_order(self, **kwargs) -> bool:
        """Save an order"""
        return await self.trading.save_order(**kwargs)

    async def get_orders(self, **kwargs) -> List[Dict[str, Any]]:
        """Get orders"""
        return await self.trading.get_orders(**kwargs)

    # --- Favorites ---
    async def add_favorite(self, **kwargs) -> Optional[int]:
        """Add a favorite"""
        return await self.favorites.add_favorite(**kwargs)

    async def get_favorites(self, **kwargs) -> List[Dict[str, Any]]:
        """Get favorites"""
        return await self.favorites.get_favorites(**kwargs)

    async def update_favorite(self, favorite_id: int, **kwargs) -> bool:
        """Update favorite"""
        return await self.favorites.update_favorite(favorite_id, **kwargs)

    async def delete_favorite(self, favorite_id: int) -> bool:
        """Delete favorite"""
        return await self.favorites.delete_favorite(favorite_id)

    # --- Symbols ---
    async def get_exchange_symbols(self, **kwargs) -> List[Dict[str, Any]]:
        """Get exchange symbols"""
        return await self.symbols.get_exchange_symbols(**kwargs)

    async def add_to_watchlist(self, **kwargs) -> bool:
        """Add symbol to watchlist"""
        return await self.symbols.add_to_watchlist(**kwargs)

    async def get_watchlist(self, **kwargs) -> List[Dict[str, Any]]:
        """Get watchlist"""
        return await self.symbols.get_watchlist(**kwargs)

    async def get_last_sync_time(self, market_type: str = 'SPOT'):
        """Get last sync time for symbols"""
        return await self.symbols.get_last_sync_time(market_type)

    async def bulk_upsert_symbols(self, symbols: list) -> int:
        """Bulk upsert symbols"""
        return await self.symbols.bulk_upsert_symbols(symbols)

    async def save_exchange_symbols(self, symbols: list) -> int:
        """Save exchange symbols to database"""
        return await self.symbols.save_exchange_symbols(symbols)

    # --- Analysis ---
    async def save_analysis_result(self, **kwargs) -> Optional[int]:
        """Save analysis result"""
        return await self.analysis.save_analysis_result(**kwargs)

    async def get_latest_result(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get latest analysis result"""
        return await self.analysis.get_latest_result(symbol, timeframe)

    async def get_analysis_results(self, **kwargs) -> List[Dict[str, Any]]:
        """Get analysis results"""
        return await self.analysis.get_analysis_results(**kwargs)

    async def queue_analysis(self, **kwargs) -> Optional[int]:
        """Queue analysis"""
        return await self.analysis.queue_analysis(**kwargs)

    async def get_analysis_queue(self, **kwargs) -> List[Dict[str, Any]]:
        """Get analysis queue"""
        return await self.analysis.get_analysis_queue(**kwargs)

    # --- Settings ---
    async def get_setting(self, category: str, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a system setting"""
        return await self.config_service.get_setting(category, key, default)

    async def set_setting(self, category: str, key: str, value: str, **kwargs) -> bool:
        """Set a system setting"""
        return await self.config_service.set_setting(category, key, value, **kwargs)

    # --- Exchange ---
    async def get_exchange_accounts(self) -> List[Dict[str, Any]]:
        """Get all exchange accounts"""
        return await self.exchange.get_all_exchange_accounts()

    async def get_all_exchange_accounts(self) -> List[Dict[str, Any]]:
        """Get all exchange accounts (alias)"""
        return await self.exchange.get_all_exchange_accounts()

    async def get_exchange_account_by_id(self, account_id: int) -> Optional[Dict[str, Any]]:
        """Get exchange account by ID"""
        return await self.exchange.get_exchange_account_by_id(account_id)

    async def create_exchange_account(self, **kwargs) -> Optional[int]:
        """Create exchange account"""
        return await self.exchange.create_exchange_account(**kwargs)

    async def update_exchange_account(self, account_id: int, **kwargs) -> bool:
        """Update exchange account"""
        return await self.exchange.update_exchange_account(account_id, **kwargs)

    async def delete_exchange_account(self, account_id: int) -> bool:
        """Delete exchange account"""
        return await self.exchange.delete_exchange_account(account_id)

    async def test_exchange_connection(self, account_id: int) -> Dict[str, Any]:
        """Test exchange connection"""
        return await self.exchange.test_exchange_connection(account_id)

    # --- Categories ---
    async def get_categories(self) -> List[Dict[str, Any]]:
        """Get all categories"""
        return await self.categories.get_categories()

    async def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """Get category by ID"""
        return await self.categories.get_category(category_id)

    async def create_category(self, **kwargs) -> Optional[int]:
        """Create category"""
        return await self.categories.create_category(**kwargs)

    async def update_category(self, category_id: int, **kwargs) -> bool:
        """Update category"""
        return await self.categories.update_category(category_id, **kwargs)

    async def delete_category(self, category_id: int) -> bool:
        """Delete category"""
        return await self.categories.delete_category(category_id)

    async def get_category_symbols(self, category_id: int) -> List[Dict[str, Any]]:
        """Get category symbols"""
        return await self.categories.get_category_symbols(category_id)

    async def add_symbols_to_category(self, category_id: int, symbol_ids: List[int], **kwargs) -> bool:
        """Add symbols to category"""
        return await self.categories.add_symbols_to_category(category_id, symbol_ids, **kwargs)

    # --- Portfolio ---
    async def create_portfolio(self, **kwargs) -> Optional[int]:
        """Create portfolio"""
        return await self.portfolio.create_portfolio(**kwargs)

    async def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolios"""
        return await self.portfolio.get_all_portfolios()

    async def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Get portfolio by ID"""
        return await self.portfolio.get_portfolio_by_id(portfolio_id)

    async def update_portfolio(self, portfolio_id: int, **kwargs) -> bool:
        """Update portfolio"""
        return await self.portfolio.update_portfolio(portfolio_id, **kwargs)

    async def delete_portfolio(self, portfolio_id: int) -> bool:
        """Delete portfolio"""
        return await self.portfolio.delete_portfolio(portfolio_id)

    async def get_portfolio_positions(self, portfolio_id: int, **kwargs) -> List[Dict[str, Any]]:
        """Get portfolio positions"""
        return await self.portfolio.get_portfolio_positions(portfolio_id, **kwargs)

    async def update_portfolio_sync_time(self, portfolio_id: int) -> bool:
        """Update portfolio sync time"""
        return await self.portfolio.update_portfolio_sync_time(portfolio_id)

    async def get_portfolio_holdings(self, **kwargs) -> List[Dict[str, Any]]:
        """Get portfolio holdings"""
        return await self.portfolio.get_portfolio_holdings(**kwargs)

    # --- Position ---
    async def create_position(self, **kwargs) -> Optional[int]:
        """Create position"""
        return await self.portfolio.create_position(**kwargs)

    async def update_position(self, position_id: int, **kwargs) -> bool:
        """Update position"""
        return await self.portfolio.update_position(position_id, **kwargs)

    async def update_position_price(self, position_id: int, current_price: float) -> bool:
        """Update position price"""
        return await self.portfolio.update_position_price(position_id, current_price)

    async def close_position(self, position_id: int, **kwargs) -> bool:
        """Close position"""
        return await self.portfolio.close_position(position_id, **kwargs)

    async def delete_position(self, position_id: int) -> bool:
        """Delete position"""
        return await self.portfolio.delete_position(position_id)

    async def add_position_entry(self, position_id: int, **kwargs) -> Optional[int]:
        """Add position entry"""
        return await self.portfolio.add_position_entry(position_id, **kwargs)

    async def get_position_entries(self, position_id: int) -> List[Dict[str, Any]]:
        """Get position entries"""
        return await self.portfolio.get_position_entries(position_id)

    async def update_position_entry(self, entry_id: int, **kwargs) -> bool:
        """Update position entry"""
        return await self.portfolio.update_position_entry(entry_id, **kwargs)

    async def delete_position_entry(self, entry_id: int) -> bool:
        """Delete position entry"""
        return await self.portfolio.delete_position_entry(entry_id)

    async def get_position_exit_targets(self, position_id: int) -> List[Dict[str, Any]]:
        """Get position exit targets"""
        return await self.portfolio.get_position_exit_targets(position_id)

    async def add_exit_target(self, position_id: int, **kwargs) -> Optional[int]:
        """Add exit target"""
        return await self.portfolio.add_exit_target(position_id, **kwargs)

    async def trigger_exit_target(self, target_id: int, **kwargs) -> bool:
        """Trigger exit target"""
        return await self.portfolio.trigger_exit_target(target_id, **kwargs)

    # --- Utils (Raw SQL, Notifications) ---
    async def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute raw SQL and fetch all results"""
        return await self.utils.fetch_all(query, params)

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute raw SQL and fetch one result"""
        return await self.utils.fetch_one(query, params)

    async def execute(self, query: str, params: tuple = ()) -> Optional[int]:
        """Execute raw SQL (INSERT/UPDATE/DELETE)"""
        return await self.utils.execute(query, params)

    async def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        return await self.utils.mark_notification_read(notification_id)

    async def mark_all_notifications_read(self) -> bool:
        """Mark all notifications as read"""
        return await self.utils.mark_all_notifications_read()

    async def delete_notification(self, notification_id: int) -> bool:
        """Delete notification"""
        return await self.utils.delete_notification(notification_id)

    async def get_unread_notification_count(self) -> int:
        """Get unread notification count"""
        return await self.utils.get_unread_notification_count()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("DataManager Facade Test")
    print("=" * 70)

    async def test():
        dm = DataManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await dm.start()

        # Test 1: Access services
        print("\n1. Service Access")
        print(f"   trading: {type(dm.trading).__name__}")
        print(f"   position: {type(dm.position).__name__}")
        print(f"   favorites: {type(dm.favorites).__name__}")
        print(f"   symbols: {type(dm.symbols).__name__}")
        print(f"   analysis: {type(dm.analysis).__name__}")
        print(f"   categories: {type(dm.categories).__name__}")
        print(f"   account: {type(dm.account).__name__}")
        print(f"   exchange: {type(dm.exchange).__name__}")
        print(f"   portfolio: {type(dm.portfolio).__name__}")
        print(f"   strategy: {type(dm.strategy).__name__}")
        print(f"   storage: {type(dm.storage).__name__}")
        print(f"   config_service: {type(dm.config_service).__name__}")

        # Test 2: Convenience methods
        print("\n2. Convenience Methods")

        # Settings
        await dm.set_setting("test", "facade_key", "facade_value")
        value = await dm.get_setting("test", "facade_key")
        print(f"   Setting: test.facade_key = {value}")

        # Watchlist
        success = await dm.add_to_watchlist(
            symbol="FACADETEST",
            is_favorite=True
        )
        print(f"   Watchlist added: {success}")

        watchlist = await dm.get_watchlist()
        print(f"   Watchlist count: {len(watchlist)}")

        # Favorites
        fav_id = await dm.add_favorite(
            symbol="FACADETEST",
            base_asset="FACADE",
            quote_asset="TEST"
        )
        print(f"   Favorite added: {fav_id}")

        favorites = await dm.get_favorites()
        print(f"   Favorites count: {len(favorites)}")

        # Test 3: Direct service usage
        print("\n3. Direct Service Usage")
        orders = await dm.trading.get_orders(limit=5)
        print(f"   Recent orders: {len(orders)}")

        portfolios = await dm.portfolio.get_all_portfolios()
        print(f"   Portfolios: {len(portfolios)}")

        strategies = await dm.strategy.get_all_strategies()
        print(f"   Strategies: {len(strategies)}")

        await dm.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
