#!/usr/bin/env python3
"""
components/datamanager/services/symbols.py
SuperBot - Symbols Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Symbol management - models and CRUD operations

Features:
- SymbolMetadata, Watchlist, ExchangeSymbol models
- Exchange symbols sync and management
- Watchlist operations
- Symbol filtering and search

Usage:
    from components.datamanager.services.symbols import SymbolsService, ExchangeSymbol

    service = SymbolsService(db_manager)
    await service.save_exchange_symbols([...])
    symbols = await service.get_exchange_symbols(market_type="SPOT")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Index
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.symbols")


# ============================================
# MODELS
# ============================================

class SymbolMetadata(Base):
    """
    Symbol metadata

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "symbol_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    tier = Column(String(10))  # TIER_1, TIER_2, TIER_3
    is_active = Column(Boolean, default=True)
    base_asset = Column(String(10))
    quote_asset = Column(String(10))
    min_quantity = Column(Float)
    max_quantity = Column(Float)
    step_size = Column(Float)
    tick_size = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)


class Watchlist(Base):
    """
    Watchlist - Trading symbols to monitor

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    is_trading = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    notes = Column(String(200))
    tags = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ExchangeSymbol(Base):
    """
    Exchange symbols - raw data from Binance/exchanges

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "exchange_symbols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(20), nullable=False, index=True, default='binance')
    symbol = Column(String(20), nullable=False, index=True)
    base_asset = Column(String(10), nullable=False, index=True)
    quote_asset = Column(String(10), nullable=False, index=True)

    # Market type
    market_type = Column(String(10), nullable=False, index=True)  # 'SPOT' or 'FUTURES'

    # Status
    status = Column(String(20), index=True)  # 'TRADING', 'BREAK', 'HALT'
    is_active = Column(Boolean, default=True)

    # Spot specific
    spot_enabled = Column(Boolean, default=False)
    spot_trading = Column(Boolean, default=False)

    # Futures specific
    futures_enabled = Column(Boolean, default=False)
    contract_type = Column(String(20))  # 'PERPETUAL', 'CURRENT_QUARTER', etc.

    # Trading specifications
    price_precision = Column(Integer)
    quantity_precision = Column(Integer)
    base_asset_precision = Column(Integer)
    quote_precision = Column(Integer)

    # Filters
    min_price = Column(Float)
    max_price = Column(Float)
    tick_size = Column(Float)
    min_qty = Column(Float)
    max_qty = Column(Float)
    step_size = Column(Float)
    min_notional = Column(Float)

    # Additional metadata
    extra_data = Column(JSON)

    # Sync tracking
    last_synced_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_exchange_symbol_market', 'exchange', 'symbol', 'market_type', unique=True),
        Index('idx_market_quote', 'market_type', 'quote_asset'),
        Index('idx_base_quote', 'base_asset', 'quote_asset'),
        Index('idx_status_active', 'status', 'is_active'),
        Index('idx_exchange', 'exchange'),
    )


# ============================================
# SERVICE
# ============================================

class SymbolsService(BaseService):
    """Symbols management service"""

    # ============================================
    # Watchlist Operations
    # ============================================

    async def add_to_watchlist(
        self,
        symbol: str,
        is_trading: bool = False,
        is_favorite: bool = False,
        notes: Optional[str] = None,
        tags: Optional[str] = None
    ) -> bool:
        """
        Add or update symbol in watchlist

        Args:
            symbol: Trading pair
            is_trading: Is active for trading
            is_favorite: Is favorite
            notes: Notes
            tags: Tags (comma-separated)

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 211-264 ===
        try:
            async with self.session() as session:
                # Check if symbol already exists
                query = select(Watchlist).where(Watchlist.symbol == symbol)
                result = await session.execute(query)
                existing = result.scalars().first()

                if existing:
                    # Update existing entry
                    existing.is_trading = is_trading
                    existing.is_favorite = is_favorite
                    if notes:
                        existing.notes = notes
                    if tags:
                        existing.tags = tags
                    existing.updated_at = get_utc_now()
                else:
                    # Create new entry
                    watchlist_entry = Watchlist(
                        symbol=symbol,
                        is_trading=is_trading,
                        is_favorite=is_favorite,
                        notes=notes,
                        tags=tags
                    )
                    session.add(watchlist_entry)

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"❌ Watchlist add error: {e}")
            return False

    async def get_watchlist(
        self,
        is_trading: Optional[bool] = None,
        is_favorite: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get watchlist

        Args:
            is_trading: Filter by trading status
            is_favorite: Filter by favorite status

        Returns:
            List: Watchlist entries
        """
        # === PORTED FROM data_manager.py lines 266-309 ===
        try:
            async with self.session() as session:
                query = select(Watchlist)

                if is_trading is not None:
                    query = query.where(Watchlist.is_trading == is_trading)

                if is_favorite is not None:
                    query = query.where(Watchlist.is_favorite == is_favorite)

                result = await session.execute(query)
                entries = result.scalars().all()

                return [
                    {
                        "symbol": e.symbol,
                        "is_trading": e.is_trading,
                        "is_favorite": e.is_favorite,
                        "notes": e.notes,
                        "tags": e.tags,
                        "created_at": e.created_at,
                        "updated_at": e.updated_at
                    }
                    for e in entries
                ]

        except Exception as e:
            logger.error(f"❌ Watchlist get error: {e}")
            return []

    # ============================================
    # Exchange Symbols Operations
    # ============================================

    async def save_exchange_symbols(self, symbols: List[Dict[str, Any]]) -> int:
        """
        Save or update exchange symbols in bulk

        Args:
            symbols: List of symbol dictionaries with exchange data

        Returns:
            int: Number of symbols saved/updated
        """
        # === PORTED FROM data_manager.py lines 2075-2120 ===
        try:
            logger.debug(f"🔧 save_exchange_symbols() - Saving {len(symbols)} symbols")

            async with self.session() as session:
                saved_count = 0

                for symbol_data in symbols:
                    # Check if symbol exists
                    query = select(ExchangeSymbol).where(
                        ExchangeSymbol.symbol == symbol_data['symbol'],
                        ExchangeSymbol.market_type == symbol_data['market_type']
                    )
                    result = await session.execute(query)
                    existing = result.scalars().first()

                    if existing:
                        # Update existing
                        for key, value in symbol_data.items():
                            if hasattr(existing, key):
                                setattr(existing, key, value)
                        existing.last_synced_at = get_utc_now()
                        existing.updated_at = get_utc_now()
                    else:
                        # Create new
                        new_symbol = ExchangeSymbol(**symbol_data)
                        session.add(new_symbol)

                    saved_count += 1

                await session.commit()
                logger.info(f"✅ {saved_count} exchange symbols saved/updated")
                return saved_count

        except Exception as e:
            logger.error(f"❌ Exchange symbols save error: {e}")
            return 0

    async def get_exchange_symbols(
        self,
        market_type: Optional[str] = None,
        quote_asset: Optional[str] = None,
        is_active: bool = True,
        search: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get exchange symbols with filters

        Args:
            market_type: Filter by market type ('SPOT' or 'FUTURES')
            quote_asset: Filter by quote asset (e.g., 'USDT')
            is_active: Only active symbols
            search: Search in symbol name
            limit: Max results

        Returns:
            List: Exchange symbols
        """
        # === PORTED FROM data_manager.py lines 2122-2191 ===
        try:
            async with self.session() as session:
                query = select(ExchangeSymbol)

                # Filters
                if market_type:
                    query = query.where(ExchangeSymbol.market_type == market_type.upper())
                    # For FUTURES, only show PERPETUAL contracts
                    if market_type.upper() == 'FUTURES':
                        query = query.where(ExchangeSymbol.contract_type == 'PERPETUAL')
                if quote_asset:
                    query = query.where(ExchangeSymbol.quote_asset == quote_asset)
                if is_active:
                    query = query.where(
                        ExchangeSymbol.is_active == True,
                        ExchangeSymbol.status.in_(['TRADING', 'MANUAL'])
                    )
                if search:
                    query = query.where(ExchangeSymbol.symbol.like(f'%{search.upper()}%'))

                # Order and limit
                query = query.order_by(ExchangeSymbol.symbol).limit(limit)

                result = await session.execute(query)
                symbols = result.scalars().all()

                return [
                    {
                        'id': s.id,
                        'symbol': s.symbol,
                        'base_asset': s.base_asset,
                        'quote_asset': s.quote_asset,
                        'market_type': s.market_type,
                        'status': s.status,
                        'is_active': s.is_active,
                        'spot_enabled': s.spot_enabled,
                        'futures_enabled': s.futures_enabled,
                        'contract_type': s.contract_type,
                        'price_precision': s.price_precision,
                        'quantity_precision': s.quantity_precision,
                        'min_notional': s.min_notional,
                        'last_synced_at': s.last_synced_at
                    }
                    for s in symbols
                ]

        except Exception as e:
            logger.error(f"❌ Exchange symbols get error: {e}")
            return []

    async def get_last_sync_time(self, market_type: str = 'SPOT') -> Optional[datetime]:
        """
        Get last sync time for a market type

        Args:
            market_type: Market type to check

        Returns:
            datetime: Last sync time or None
        """
        # === PORTED FROM data_manager.py lines 2193-2216 ===
        try:
            async with self.session() as session:
                query = select(ExchangeSymbol).where(
                    ExchangeSymbol.market_type == market_type
                ).order_by(ExchangeSymbol.last_synced_at.desc()).limit(1)

                result = await session.execute(query)
                symbol = result.scalars().first()

                return symbol.last_synced_at if symbol else None

        except Exception as e:
            logger.error(f"❌ Last sync time get error: {e}")
            return None

    async def get_symbol_by_id(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        """
        Get exchange symbol by ID

        Args:
            symbol_id: Symbol ID

        Returns:
            Dict: Symbol data or None
        """
        try:
            async with self.session() as session:
                query = select(ExchangeSymbol).where(ExchangeSymbol.id == symbol_id)
                result = await session.execute(query)
                s = result.scalars().first()

                if s:
                    return {
                        'id': s.id,
                        'symbol': s.symbol,
                        'base_asset': s.base_asset,
                        'quote_asset': s.quote_asset,
                        'market_type': s.market_type,
                        'status': s.status,
                        'is_active': s.is_active
                    }

                return None

        except Exception as e:
            logger.error(f"❌ Symbol get error: {e}")
            return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 SymbolsService Test")
    print("=" * 60)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = SymbolsService(db)

        # Test 1: Add to watchlist
        print("\nTest 1: Add to watchlist")
        success = await service.add_to_watchlist(
            symbol="BTCUSDT",
            is_trading=True,
            is_favorite=True,
            notes="Main trading pair"
        )
        print(f"   ✅ Added to watchlist: {success}")

        # Test 2: Get watchlist
        print("\nTest 2: Get watchlist")
        watchlist = await service.get_watchlist()
        print(f"   ✅ Got {len(watchlist)} watchlist entries")

        # Test 3: Save exchange symbols
        print("\nTest 3: Save exchange symbols")
        count = await service.save_exchange_symbols([
            {
                "symbol": "BTCUSDT",
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "market_type": "SPOT",
                "status": "TRADING",
                "is_active": True,
                "spot_enabled": True,
                "price_precision": 2,
                "quantity_precision": 5,
                "min_notional": 10.0
            },
            {
                "symbol": "ETHUSDT",
                "base_asset": "ETH",
                "quote_asset": "USDT",
                "market_type": "SPOT",
                "status": "TRADING",
                "is_active": True,
                "spot_enabled": True,
                "price_precision": 2,
                "quantity_precision": 4,
                "min_notional": 10.0
            }
        ])
        print(f"   ✅ Saved {count} exchange symbols")

        # Test 4: Get exchange symbols
        print("\nTest 4: Get exchange symbols")
        symbols = await service.get_exchange_symbols(market_type="SPOT", quote_asset="USDT")
        print(f"   ✅ Got {len(symbols)} SPOT/USDT symbols")

        # Test 5: Get last sync time
        print("\nTest 5: Get last sync time")
        sync_time = await service.get_last_sync_time("SPOT")
        print(f"   ✅ Last sync time: {sync_time}")

        # Test 6: Search symbols
        print("\nTest 6: Search symbols")
        results = await service.get_exchange_symbols(search="BTC")
        print(f"   ✅ Found {len(results)} symbols matching 'BTC'")

        await db.stop()

    asyncio.run(test())
    print("\n✅ All tests completed!")
    print("=" * 60)
