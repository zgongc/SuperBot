#!/usr/bin/env python3
"""
components/datamanager/services/favorites.py
SuperBot - Favorites Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Favorites management - models and CRUD operations

Features:
- SymbolFavorite model
- Add/remove favorites
- Update favorite properties
- Get favorites with sorting/filtering
- View count tracking

Usage:
    from components.datamanager.services.favorites import FavoritesService, SymbolFavorite

    service = FavoritesService(db_manager)
    await service.add_favorite("BTCUSDT", "BTC", "USDT")
    favorites = await service.get_favorites()

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Index
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.favorites")


# ============================================
# MODELS
# ============================================

class SymbolFavorite(Base):
    """
    User favorite symbols with enhanced metadata

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "symbol_favorites"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False, index=True)
    market_type = Column(String(10), nullable=False, default="SPOT", index=True)  # 'SPOT' or 'FUTURES'

    # User management (future: multi-user support)
    user_id = Column(String(50), default="default", index=True)

    # Organization
    tags = Column(JSON)  # ["scalping", "swing", "breakout"]
    notes = Column(Text)
    priority = Column(Integer, default=5)  # 1-10 (10 = highest)
    color = Column(String(7))  # #FF5733 for UI highlighting

    # Tracking
    added_at = Column(DateTime, default=datetime.utcnow)
    last_viewed = Column(DateTime)
    view_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_user_priority', 'user_id', 'priority'),
        Index('idx_added_at', 'added_at'),
        Index('idx_market_type', 'market_type'),
        Index('idx_symbol_user_market', 'symbol', 'user_id', 'market_type', unique=True),
    )


# ============================================
# SERVICE
# ============================================

class FavoritesService(BaseService):
    """Favorites management service"""

    async def add_favorite(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        market_type: str = "SPOT",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        priority: int = 5,
        color: Optional[str] = None,
        user_id: str = "default"
    ) -> Optional[int]:
        """
        Add symbol to favorites

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            base_asset: Base asset (e.g., BTC)
            quote_asset: Quote asset (e.g., USDT)
            market_type: Market type (SPOT or FUTURES)
            tags: List of tags
            notes: User notes
            priority: Priority 1-10 (10 = highest)
            color: Hex color for UI
            user_id: User identifier

        Returns:
            int: Favorite ID if successful, None otherwise
        """
        # === PORTED FROM data_manager.py lines 315-382 ===
        try:
            async with self.session() as session:
                # Check if already exists
                query = select(SymbolFavorite).where(
                    SymbolFavorite.symbol == symbol,
                    SymbolFavorite.user_id == user_id,
                    SymbolFavorite.market_type == market_type
                )
                result = await session.execute(query)
                existing = result.scalars().first()

                if existing:
                    logger.warning(f"⚠️ Symbol {symbol} already in favorites (user: {user_id})")
                    return existing.id

                # Create new favorite
                favorite = SymbolFavorite(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    market_type=market_type,
                    user_id=user_id,
                    tags=tags,
                    notes=notes,
                    priority=priority,
                    color=color
                )
                session.add(favorite)
                await session.commit()
                await session.refresh(favorite)

                logger.info(f"✅ Favorite added: {symbol} (priority: {priority})")
                return favorite.id

        except Exception as e:
            logger.error(f"❌ Favorite add error: {e}")
            return None

    async def get_favorites(
        self,
        user_id: str = "default",
        sort_by: str = "priority",
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user favorites

        Args:
            user_id: User identifier
            sort_by: Sort field (priority/added_at/last_viewed/view_count)
            tags: Filter by tags

        Returns:
            List: Favorite symbols
        """
        # === PORTED FROM data_manager.py lines 384-445 ===
        try:
            async with self.session() as session:
                query = select(SymbolFavorite).where(SymbolFavorite.user_id == user_id)

                # Sort
                if sort_by == "priority":
                    query = query.order_by(SymbolFavorite.priority.desc())
                elif sort_by == "added_at":
                    query = query.order_by(SymbolFavorite.added_at.desc())
                elif sort_by == "last_viewed":
                    query = query.order_by(SymbolFavorite.last_viewed.desc())
                elif sort_by == "view_count":
                    query = query.order_by(SymbolFavorite.view_count.desc())

                result = await session.execute(query)
                favorites = result.scalars().all()

                # Filter by tags if provided
                if tags:
                    favorites = [
                        f for f in favorites
                        if f.tags and any(tag in f.tags for tag in tags)
                    ]

                return [
                    {
                        "id": f.id,
                        "symbol": f.symbol,
                        "base_asset": f.base_asset,
                        "quote_asset": f.quote_asset,
                        "market_type": f.market_type,
                        "tags": f.tags or [],
                        "notes": f.notes,
                        "priority": f.priority,
                        "color": f.color,
                        "added_at": f.added_at,
                        "last_viewed": f.last_viewed,
                        "view_count": f.view_count
                    }
                    for f in favorites
                ]

        except Exception as e:
            logger.error(f"❌ Favorites get error: {e}")
            return []

    async def update_favorite(
        self,
        favorite_id: int,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        priority: Optional[int] = None,
        color: Optional[str] = None
    ) -> bool:
        """
        Update favorite

        Args:
            favorite_id: Favorite ID
            tags: New tags
            notes: New notes
            priority: New priority
            color: New color

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 447-496 ===
        try:
            async with self.session() as session:
                query = select(SymbolFavorite).where(SymbolFavorite.id == favorite_id)
                result = await session.execute(query)
                favorite = result.scalars().first()

                if not favorite:
                    logger.warning(f"⚠️ Favorite not found: ID {favorite_id}")
                    return False

                # Update fields
                if tags is not None:
                    favorite.tags = tags
                if notes is not None:
                    favorite.notes = notes
                if priority is not None:
                    favorite.priority = priority
                if color is not None:
                    favorite.color = color

                favorite.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Favorite updated: {favorite.symbol} (ID: {favorite_id})")
                return True

        except Exception as e:
            logger.error(f"❌ Favorite update error: {e}")
            return False

    async def delete_favorite(self, favorite_id: int) -> bool:
        """
        Delete favorite

        Args:
            favorite_id: Favorite ID

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 498-527 ===
        try:
            async with self.session() as session:
                query = select(SymbolFavorite).where(SymbolFavorite.id == favorite_id)
                result = await session.execute(query)
                favorite = result.scalars().first()

                if not favorite:
                    logger.warning(f"⚠️ Favorite not found: ID {favorite_id}")
                    return False

                symbol = favorite.symbol
                await session.delete(favorite)
                await session.commit()

                logger.info(f"✅ Favorite deleted: {symbol} (ID: {favorite_id})")
                return True

        except Exception as e:
            logger.error(f"❌ Favorite delete error: {e}")
            return False

    async def increment_favorite_view(self, favorite_id: int) -> bool:
        """
        Increment view count and update last viewed

        Args:
            favorite_id: Favorite ID

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 529-547 ===
        try:
            async with self.session() as session:
                query = select(SymbolFavorite).where(SymbolFavorite.id == favorite_id)
                result = await session.execute(query)
                favorite = result.scalars().first()

                if favorite:
                    favorite.view_count += 1
                    favorite.last_viewed = get_utc_now()
                    await session.commit()
                    return True

                return False

        except Exception as e:
            logger.error(f"❌ View count update error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 FavoritesService Test")
    print("=" * 60)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = FavoritesService(db)

        # Test 1: Add favorite
        print("\nTest 1: Add favorite")
        fav_id = await service.add_favorite(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            market_type="SPOT",
            tags=["layer1", "store-of-value"],
            priority=10
        )
        print(f"   ✅ Favorite added: ID {fav_id}")

        # Test 2: Get favorites
        print("\nTest 2: Get favorites")
        favorites = await service.get_favorites()
        print(f"   ✅ Got {len(favorites)} favorites")

        # Test 3: Update favorite
        print("\nTest 3: Update favorite")
        if fav_id:
            success = await service.update_favorite(
                favorite_id=fav_id,
                priority=8,
                notes="Main trading pair"
            )
            print(f"   ✅ Updated: {success}")

        # Test 4: Increment view
        print("\nTest 4: Increment view")
        if fav_id:
            success = await service.increment_favorite_view(fav_id)
            print(f"   ✅ View incremented: {success}")

        # Test 5: Get favorites sorted by view_count
        print("\nTest 5: Get favorites sorted by view_count")
        favorites = await service.get_favorites(sort_by="view_count")
        if favorites:
            print(f"   ✅ Top viewed: {favorites[0]['symbol']} ({favorites[0]['view_count']} views)")

        # Test 6: Delete favorite
        print("\nTest 6: Delete favorite")
        if fav_id:
            success = await service.delete_favorite(fav_id)
            print(f"   ✅ Deleted: {success}")

        await db.stop()

    asyncio.run(test())
    print("\n✅ All tests completed!")
    print("=" * 60)
