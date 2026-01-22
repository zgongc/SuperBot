#!/usr/bin/env python3
"""
components/datamanager/services/categories.py
SuperBot - Categories Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Category management - models and CRUD operations

Features:
- Category, CategorySymbol models
- CRUD operations for categories
- Symbol-category relationship management

Usage:
    from components.datamanager.services.categories import CategoriesService, Category

    service = CategoriesService(db_manager)
    cat_id = await service.create_category("Layer 1", "Major blockchains")
    await service.add_symbols_to_category(cat_id, [1, 2, 3])

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.categories")


# ============================================
# MODELS
# ============================================

class Category(Base):
    """
    Category model - Symbol groups

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Default settings
    default_priority = Column(Integer, default=5)  # 1-10
    default_color = Column(String(7), default='#666666')  # Hex color

    # Stats
    symbol_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CategorySymbol(Base):
    """
    Category-Symbol relationship model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "category_symbols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, nullable=False, index=True)
    symbol_id = Column(Integer, nullable=False, index=True)

    # User preferences
    priority = Column(Integer, default=5)  # 1-10
    color = Column(String(7), default='#666666')  # Hex color

    # Timestamps
    added_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_category_symbol', 'category_id', 'symbol_id', unique=True),
    )


# ============================================
# SERVICE
# ============================================

class CategoriesService(BaseService):
    """Categories management service"""

    async def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories with accurate symbol counts

        Returns:
            List: Category list with real-time symbol counts
        """
        # === PORTED FROM data_manager.py lines 1865-1889 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(Category).order_by(Category.name)
                )
                categories = result.scalars().all()

                # Build result with accurate symbol counts
                categories_list = []
                for cat in categories:
                    # Count actual symbols in category_symbols table
                    count_result = await session.execute(
                        select(CategorySymbol).where(CategorySymbol.category_id == cat.id)
                    )
                    actual_count = len(count_result.scalars().all())

                    # Update cached count if different
                    if cat.symbol_count != actual_count:
                        cat.symbol_count = actual_count

                    categories_list.append({
                        'id': cat.id,
                        'name': cat.name,
                        'description': cat.description,
                        'default_priority': cat.default_priority,
                        'default_color': cat.default_color,
                        'symbol_count': actual_count,
                        'created_at': cat.created_at.isoformat() if cat.created_at else None,
                        'updated_at': cat.updated_at.isoformat() if cat.updated_at else None
                    })

                await session.commit()  # Save any count corrections
                return categories_list
        except Exception as e:
            logger.error(f"❌ Get categories error: {e}")
            return []

    async def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get single category by ID

        Args:
            category_id: Category ID

        Returns:
            Dict: Category data or None
        """
        # === PORTED FROM data_manager.py lines 1891-1915 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(Category).where(Category.id == category_id)
                )
                category = result.scalar_one_or_none()

                if not category:
                    return None

                return {
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'default_priority': category.default_priority,
                    'default_color': category.default_color,
                    'symbol_count': category.symbol_count or 0,
                    'created_at': category.created_at.isoformat() if category.created_at else None,
                    'updated_at': category.updated_at.isoformat() if category.updated_at else None
                }
        except Exception as e:
            logger.error(f"❌ Get category error: {e}")
            return None

    async def create_category(
        self,
        name: str,
        description: Optional[str] = None,
        default_priority: int = 5,
        default_color: str = '#666666'
    ) -> Optional[int]:
        """
        Create new category

        Args:
            name: Category name
            description: Category description
            default_priority: Default priority for symbols (1-10)
            default_color: Default hex color

        Returns:
            int: Category ID if successful
        """
        # === PORTED FROM data_manager.py lines 1917-1942 ===
        try:
            async with self.session() as session:
                category = Category(
                    name=name,
                    description=description,
                    default_priority=default_priority,
                    default_color=default_color,
                    symbol_count=0
                )
                session.add(category)
                await session.commit()
                await session.refresh(category)

                logger.info(f"✅ Created category: {name} (ID: {category.id})")
                return category.id
        except Exception as e:
            logger.error(f"❌ Create category error: {e}")
            return None

    async def update_category(
        self,
        category_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default_priority: Optional[int] = None,
        default_color: Optional[str] = None
    ) -> bool:
        """
        Update category

        Args:
            category_id: Category ID
            name: New name
            description: New description
            default_priority: New default priority
            default_color: New default color

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 1944-1979 ===
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(Category).where(Category.id == category_id)
                )
                category = result.scalar_one_or_none()

                if not category:
                    return False

                if name is not None:
                    category.name = name
                if description is not None:
                    category.description = description
                if default_priority is not None:
                    category.default_priority = default_priority
                if default_color is not None:
                    category.default_color = default_color

                category.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Updated category: {category_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Update category error: {e}")
            return False

    async def delete_category(self, category_id: int) -> bool:
        """
        Delete category and its symbol relationships

        Args:
            category_id: Category ID

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 1981-2008 ===
        try:
            async with self.session() as session:
                # Delete category_symbols first
                await session.execute(
                    CategorySymbol.__table__.delete().where(
                        CategorySymbol.category_id == category_id
                    )
                )

                # Delete category
                result = await session.execute(
                    select(Category).where(Category.id == category_id)
                )
                category = result.scalar_one_or_none()

                if not category:
                    return False

                await session.delete(category)
                await session.commit()

                logger.info(f"✅ Deleted category: {category_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Delete category error: {e}")
            return False

    async def get_category_symbols(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get all symbols in a category with full symbol information

        Args:
            category_id: Category ID

        Returns:
            List: Symbol list with category-specific data and full symbol info
        """
        # === PORTED FROM data_manager.py lines 2010-2037 ===
        # Updated to JOIN with ExchangeSymbol for full symbol data
        try:
            from components.datamanager.services.symbols import ExchangeSymbol

            async with self.session() as session:
                # Join CategorySymbol with ExchangeSymbol to get full data
                result = await session.execute(
                    select(CategorySymbol, ExchangeSymbol)
                    .join(ExchangeSymbol, CategorySymbol.symbol_id == ExchangeSymbol.id, isouter=True)
                    .where(CategorySymbol.category_id == category_id)
                )
                rows = result.all()

                symbols = []
                for cs, es in rows:
                    symbol_data = {
                        'id': cs.symbol_id,  # Use symbol_id for consistency
                        'category_symbol_id': cs.id,
                        'category_id': cs.category_id,
                        'symbol_id': cs.symbol_id,
                        'priority': cs.priority,
                        'color': cs.color,
                        'added_at': cs.added_at.isoformat() if cs.added_at else None
                    }

                    # Add ExchangeSymbol data if available
                    if es:
                        symbol_data.update({
                            'symbol': es.symbol,
                            'base_asset': es.base_asset,
                            'quote_asset': es.quote_asset,
                            'market_type': es.market_type,
                            'status': es.status,
                            'exchange': es.exchange,
                            'created_at': es.created_at.isoformat() if es.created_at else None
                        })
                    else:
                        # Symbol not found in exchange_symbols table
                        symbol_data.update({
                            'symbol': f'Unknown ({cs.symbol_id})',
                            'base_asset': '',
                            'quote_asset': '',
                            'market_type': 'SPOT',
                            'status': 'UNKNOWN',
                            'exchange': 'unknown',
                            'created_at': None
                        })

                    symbols.append(symbol_data)

                return symbols
        except Exception as e:
            logger.error(f"❌ Get category symbols error: {e}")
            return []

    async def add_symbols_to_category(
        self,
        category_id: int,
        symbol_ids: List[int],
        priority: int = 5,
        color: str = '#666666'
    ) -> bool:
        """
        Add multiple symbols to category

        Args:
            category_id: Category ID
            symbol_ids: List of symbol IDs
            priority: Priority for symbols (1-10)
            color: Hex color for symbols

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 2039-2093 ===
        try:
            async with self.session() as session:
                # Check category exists
                result = await session.execute(
                    select(Category).where(Category.id == category_id)
                )
                category = result.scalar_one_or_none()

                if not category:
                    return False

                # Get existing symbol IDs
                result = await session.execute(
                    select(CategorySymbol.symbol_id).where(
                        CategorySymbol.category_id == category_id
                    )
                )
                existing_ids = set(row[0] for row in result.fetchall())

                # Add only new symbols
                added = 0
                for symbol_id in symbol_ids:
                    if symbol_id not in existing_ids:
                        cat_sym = CategorySymbol(
                            category_id=category_id,
                            symbol_id=symbol_id,
                            priority=priority,
                            color=color
                        )
                        session.add(cat_sym)
                        added += 1

                # Update symbol_count (existing + newly added)
                category.symbol_count = len(existing_ids) + added

                await session.commit()

                logger.info(f"✅ Added {added} symbols to category {category_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Add symbols to category error: {e}")
            return False

    async def remove_symbols_from_category(
        self,
        category_id: int,
        symbol_ids: List[int]
    ) -> bool:
        """
        Remove multiple symbols from category

        Args:
            category_id: Category ID
            symbol_ids: List of symbol IDs to remove

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 2095-2131 ===
        try:
            async with self.session() as session:
                # Delete category_symbols
                await session.execute(
                    CategorySymbol.__table__.delete().where(
                        (CategorySymbol.category_id == category_id) &
                        (CategorySymbol.symbol_id.in_(symbol_ids))
                    )
                )

                # Update symbol_count
                result = await session.execute(
                    select(Category).where(Category.id == category_id)
                )
                category = result.scalar_one_or_none()

                if category:
                    result = await session.execute(
                        select(CategorySymbol).where(CategorySymbol.category_id == category_id)
                    )
                    category.symbol_count = len(result.scalars().all())

                await session.commit()

                logger.info(f"✅ Removed {len(symbol_ids)} symbols from category {category_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Remove symbols from category error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 CategoriesService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = CategoriesService(db)

        # Test 1: Create categories
        print("\n1️⃣ Create categories")
        cat1_id = await service.create_category(
            name="Layer 1",
            description="Major blockchain platforms",
            default_priority=9,
            default_color="#3498db"
        )
        print(f"   ✅ Layer 1: ID {cat1_id}")

        cat2_id = await service.create_category(
            name="DeFi",
            description="Decentralized finance protocols",
            default_priority=7,
            default_color="#2ecc71"
        )
        print(f"   ✅ DeFi: ID {cat2_id}")

        cat3_id = await service.create_category(
            name="Meme Coins",
            description="Community-driven tokens",
            default_priority=3,
            default_color="#e74c3c"
        )
        print(f"   ✅ Meme Coins: ID {cat3_id}")

        # Test 2: Get all categories
        print("\n2️⃣ Get all categories")
        categories = await service.get_categories()
        print(f"   ✅ Total: {len(categories)} categories")
        for cat in categories[:5]:
            print(f"      - {cat['name']}: {cat['symbol_count']} symbols (priority: {cat['default_priority']})")

        # Test 3: Get single category
        print("\n3️⃣ Get single category")
        if cat1_id:
            cat = await service.get_category(cat1_id)
            if cat:
                print(f"   ✅ {cat['name']}: {cat['description']}")

        # Test 4: Update category
        print("\n4️⃣ Update category")
        if cat1_id:
            success = await service.update_category(
                cat1_id,
                description="Major L1 blockchain platforms",
                default_priority=10
            )
            print(f"   ✅ Updated: {success}")

        # Test 5: Add symbols to category
        print("\n5️⃣ Add symbols to category")
        if cat1_id:
            success = await service.add_symbols_to_category(
                cat1_id,
                symbol_ids=[1, 2, 3, 4, 5],
                priority=8,
                color="#3498db"
            )
            print(f"   ✅ Added symbols: {success}")

        # Test 6: Get category symbols
        print("\n6️⃣ Get category symbols")
        if cat1_id:
            symbols = await service.get_category_symbols(cat1_id)
            print(f"   ✅ Symbols in category: {len(symbols)}")

        # Test 7: Remove symbols from category
        print("\n7️⃣ Remove symbols from category")
        if cat1_id:
            success = await service.remove_symbols_from_category(cat1_id, [4, 5])
            print(f"   ✅ Removed: {success}")
            symbols = await service.get_category_symbols(cat1_id)
            print(f"   ✅ Remaining: {len(symbols)} symbols")

        # Test 8: Delete category
        print("\n8️⃣ Delete category")
        if cat3_id:
            success = await service.delete_category(cat3_id)
            print(f"   ✅ Deleted Meme Coins: {success}")
            categories = await service.get_categories()
            print(f"   ✅ Remaining: {len(categories)} categories")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
