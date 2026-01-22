#!/usr/bin/env python3
"""
components/datamanager/services/utils.py
SuperBot - Utility Functions Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Database utility operations - raw SQL, export/import

Features:
- Raw SQL execution (fetch_all, fetch_one, execute)
- Parquet export
- Notification helpers

Usage:
    from components.datamanager.services.utils import UtilsService

    service = UtilsService(db_manager)
    rows = await service.fetch_all("SELECT * FROM trades WHERE symbol = ?", ("BTCUSDT",))

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
    - pandas (optional, for parquet)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import text

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.utils")


class UtilsService(BaseService):
    """Database utility operations service"""

    # ============================================
    # Raw SQL Query Methods
    # ============================================

    async def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query and fetch all results

        Args:
            query: SQL query string (use ? for positional params)
            params: Query parameters tuple

        Returns:
            List of dictionaries (row results)
        """
        # === PORTED FROM data_manager.py lines 1336-1369 ===
        try:
            async with self.session() as session:
                # Convert positional params to dict for SQLAlchemy text()
                if params:
                    # Replace ? with :param0, :param1, etc.
                    param_dict = {}
                    modified_query = query
                    for i, p in enumerate(params):
                        param_key = f'param{i}'
                        param_dict[param_key] = p
                        modified_query = modified_query.replace('?', f':{param_key}', 1)
                    result = await session.execute(text(modified_query), param_dict)
                else:
                    result = await session.execute(text(query))

                rows = result.fetchall()

                # Convert Row objects to dictionaries
                return [dict(row._mapping) for row in rows]

        except Exception as e:
            logger.error(f"❌ fetch_all error: {e}")
            return []

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """
        Execute raw SQL query and fetch one result

        Args:
            query: SQL query string (use ? for positional params)
            params: Query parameters tuple

        Returns:
            Dictionary (row result) or None
        """
        # === PORTED FROM data_manager.py lines 1371-1403 ===
        try:
            async with self.session() as session:
                # Convert positional params to dict
                if params:
                    param_dict = {}
                    modified_query = query
                    for i, p in enumerate(params):
                        param_key = f'param{i}'
                        param_dict[param_key] = p
                        modified_query = modified_query.replace('?', f':{param_key}', 1)
                    result = await session.execute(text(modified_query), param_dict)
                else:
                    result = await session.execute(text(query))

                row = result.fetchone()

                # Convert Row object to dictionary
                return dict(row._mapping) if row else None

        except Exception as e:
            logger.error(f"❌ fetch_one error: {e}")
            return None

    async def execute(self, query: str, params: tuple = ()) -> Optional[int]:
        """
        Execute raw SQL query (INSERT/UPDATE/DELETE)

        Args:
            query: SQL query string (use ? for positional params)
            params: Query parameters tuple

        Returns:
            Last inserted row ID for INSERT, None for UPDATE/DELETE
        """
        # === PORTED FROM data_manager.py lines 1405-1439 ===
        try:
            async with self.session() as session:
                # Convert positional params to dict
                if params:
                    param_dict = {}
                    modified_query = query
                    for i, p in enumerate(params):
                        param_key = f'param{i}'
                        param_dict[param_key] = p
                        modified_query = modified_query.replace('?', f':{param_key}', 1)
                    result = await session.execute(text(modified_query), param_dict)
                else:
                    result = await session.execute(text(query))

                await session.commit()

                # Return last inserted ID for INSERT statements
                if query.strip().upper().startswith('INSERT'):
                    return result.lastrowid
                return None

        except Exception as e:
            logger.error(f"❌ execute error: {e}")
            return None

    # ============================================
    # Parquet Export
    # ============================================

    async def export_to_parquet(
        self,
        query: str,
        output_path: str,
        params: tuple = ()
    ) -> bool:
        """
        Export query results to Parquet file

        Args:
            query: SQL query string
            output_path: Output parquet file path
            params: Query parameters

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 1298-1322 ===
        try:
            import pandas as pd

            rows = await self.fetch_all(query, params)

            if not rows:
                logger.warning(f"No data found for parquet export")
                return False

            df = pd.DataFrame(rows)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(output_path, compression='snappy')

            logger.info(f"Parquet export: {len(rows)} rows -> {output_path}")
            return True

        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas pyarrow")
            return False
        except Exception as e:
            logger.error(f"Parquet export error: {e}")
            return False

    # ============================================
    # Notification Helpers
    # ============================================

    async def mark_notification_read(self, notification_id: int) -> bool:
        """Mark a notification as read"""
        # === PORTED FROM data_manager.py lines 1445-1459 ===
        try:
            async with self.session() as session:
                query = """
                    UPDATE alert_notifications
                    SET is_read = 1, read_at = :read_at
                    WHERE id = :notification_id
                """
                await session.execute(text(query), {'read_at': get_utc_now(), 'notification_id': notification_id})
                await session.commit()
            logger.info(f"Notification {notification_id} marked as read")
            return True
        except Exception as e:
            logger.error(f"mark_notification_read error: {e}")
            return False

    async def mark_all_notifications_read(self) -> bool:
        """Mark all notifications as read"""
        # === PORTED FROM data_manager.py lines 1461-1475 ===
        try:
            async with self.session() as session:
                query = """
                    UPDATE alert_notifications
                    SET is_read = 1, read_at = :read_at
                    WHERE is_read = 0
                """
                await session.execute(text(query), {'read_at': get_utc_now()})
                await session.commit()
            logger.info("All notifications marked as read")
            return True
        except Exception as e:
            logger.error(f"mark_all_notifications_read error: {e}")
            return False

    async def delete_notification(self, notification_id: int) -> bool:
        """Delete a notification"""
        # === PORTED FROM data_manager.py lines 1477-1487 ===
        try:
            async with self.session() as session:
                query = "DELETE FROM alert_notifications WHERE id = :notification_id"
                await session.execute(text(query), {'notification_id': notification_id})
                await session.commit()
            logger.info(f"Notification {notification_id} deleted")
            return True
        except Exception as e:
            logger.error(f"delete_notification error: {e}")
            return False

    async def get_unread_notification_count(self) -> int:
        """Get count of unread notifications"""
        # === PORTED FROM data_manager.py lines 1489-1494 ===
        try:
            query = "SELECT COUNT(*) as count FROM alert_notifications WHERE is_read = 0"
            result = await self.fetch_one(query)
            return result['count'] if result else 0
        except Exception as e:
            logger.error(f"get_unread_notification_count error: {e}")
            return 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("UtilsService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = UtilsService(db)

        # Test 1: fetch_all
        print("\n1. fetch_all")
        rows = await service.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"   Found {len(rows)} tables")
        for row in rows[:5]:
            print(f"      - {row.get('name', row)}")

        # Test 2: fetch_one
        print("\n2. fetch_one")
        row = await service.fetch_one("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'")
        print(f"   Total tables: {row['count'] if row else 0}")

        # Test 3: execute with params
        print("\n3. execute with params")
        # Create a test table
        await service.execute("CREATE TABLE IF NOT EXISTS utils_test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)")
        print("   Created utils_test table")

        # Insert with positional params
        result = await service.execute("INSERT INTO utils_test (name, value) VALUES (?, ?)", ("test1", 100))
        print(f"   Inserted row ID: {result}")

        # Test 4: fetch_all with params
        print("\n4. fetch_all with params")
        rows = await service.fetch_all("SELECT * FROM utils_test WHERE value > ?", (50,))
        print(f"   Found {len(rows)} rows with value > 50")

        # Test 5: fetch_one with params
        print("\n5. fetch_one with params")
        row = await service.fetch_one("SELECT * FROM utils_test WHERE name = ?", ("test1",))
        print(f"   Found: {row}")

        # Test 6: Notification count (table may not exist)
        print("\n6. get_unread_notification_count")
        try:
            count = await service.get_unread_notification_count()
            print(f"   Unread notifications: {count}")
        except Exception as e:
            print(f"   (Table not exists - expected): {e}")

        # Cleanup
        await service.execute("DROP TABLE IF EXISTS utils_test")
        print("\n7. Cleanup")
        print("   Dropped utils_test table")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
