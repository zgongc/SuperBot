#!/usr/bin/env python3
"""
components/datamanager/base.py
SuperBot - DataManager Base Classes
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Database connection management and base service class.

Features:
- SQLAlchemy async engine management
- Session factory
- Base service class for all services

Usage:
    from components.datamanager.base import Base, DatabaseManager, BaseService

    db = DatabaseManager(config)
    await db.start()

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
    - aiosqlite
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Any

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from core.logger_engine import get_logger

logger = get_logger("components.datamanager.base")

# SQLAlchemy Base - all models use this
Base = declarative_base()


class DatabaseManager:
    """
    Database connection and engine management

    Supports SQLite (default) and PostgreSQL backends.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DatabaseManager

        Args:
            config: Database configuration dict
                - backend: "sqlite" or "postgresql"
                - path: SQLite database path (for sqlite backend)
                - postgresql: PostgreSQL config dict (for postgresql backend)
        """
        self.config = config or {}
        self.backend = self.config.get("backend", "sqlite")
        self.engine = None
        self.async_session = None
        self._build_database_url()

    def _build_database_url(self):
        """Build database URL from config"""
        if self.backend == "sqlite":
            db_path = self.config.get("path", "data/db/superbot.db")
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite+aiosqlite:///{self.db_path}"
        else:  # postgresql
            pg = self.config.get("postgresql", {})
            host = pg.get("host", "localhost")
            port = pg.get("port", 5432)
            user = pg.get("user", "superbot")
            password = pg.get("password", "")
            database = pg.get("database", "superbot")
            self.database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

    async def start(self):
        """Initialize database engine and session factory"""
        try:
            logger.debug(f"🔧 Creating async engine for: {self.database_url}")

            # =================================================================
            # HYBRID STARTUP: Windows Ctrl+C Fix
            # Using sync engine for table creation to avoid thread-lock
            # that blocks Windows signal handling (WinError 10038)
            # =================================================================
            if self.backend == "sqlite":
                sync_url = self.database_url.replace("+aiosqlite", "")
            else:
                sync_url = self.database_url.replace("+asyncpg", "")

            sync_engine = create_engine(sync_url)
            Base.metadata.create_all(sync_engine)
            sync_engine.dispose()
            logger.debug("✅ Tables created via sync engine")

            # Now create async engine for runtime operations
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                future=True
            )

            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info(f"✅ Database started: {self.backend}")

        except Exception as e:
            logger.error(f"❌ Database start error: {e}")
            raise

    async def stop(self):
        """Close database connection"""
        try:
            if self.engine:
                await self.engine.dispose()
            logger.info("🛑 Database closed")

        except Exception as e:
            logger.error(f"❌ Database close error: {e}")


class BaseService:
    """
    Base service class with session management

    All domain services extend this class.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize BaseService

        Args:
            db_manager: DatabaseManager instance
        """
        self._db = db_manager

    def session(self):
        """Get async session context manager"""
        return self._db.async_session()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🧪 DatabaseManager Test")
    print("=" * 60)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()
        print("   ✅ Database started")

        # Test session
        async with db.async_session() as session:
            print(f"   ✅ Session created: {type(session)}")

        await db.stop()
        print("   ✅ Database stopped")

    asyncio.run(test())
    print("\n✅ All tests completed!")
    print("=" * 60)
