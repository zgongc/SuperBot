#!/usr/bin/env python3
"""
components/datamanager/services/position.py
SuperBot - Position Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Position tracking - models and CRUD operations

Features:
- Position model
- Save/update positions
- Get open positions
- Position status tracking

Usage:
    from components.datamanager.services.position import PositionService, Position

    service = PositionService(db_manager)
    await service.save_position({...})
    positions = await service.get_open_positions()

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, BigInteger, DateTime
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger

logger = get_logger("components.datamanager.services.position")


# ============================================
# MODELS
# ============================================

class Position(Base):
    """
    Position tracking model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # LONG/SHORT
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    leverage = Column(Integer, default=1)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    status = Column(String(20), nullable=False)  # OPEN/CLOSED
    entry_time = Column(BigInteger, nullable=False)
    exit_time = Column(BigInteger)
    exit_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================
# SERVICE
# ============================================

class PositionService(BaseService):
    """Position management service"""

    async def save_position(self, position_data: Dict[str, Any]) -> bool:
        """
        Save or update position

        Args:
            position_data: Position data dict with position_id, symbol, side, etc.

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 211-234 ===
        try:
            async with self.session() as session:
                # Check if position already exists
                query = select(Position).where(Position.position_id == position_data.get("position_id"))
                result = await session.execute(query)
                existing = result.scalars().first()

                if existing:
                    # Update existing position
                    for key, value in position_data.items():
                        setattr(existing, key, value)
                else:
                    # Create new position
                    position = Position(**position_data)
                    session.add(position)

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"❌ Position save error: {e}")
            return False

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions

        Returns:
            List: Open position list
        """
        # === PORTED FROM data_manager.py lines 236-268 ===
        try:
            async with self.session() as session:
                query = select(Position).where(Position.status == "OPEN")
                result = await session.execute(query)
                positions = result.scalars().all()

                return [
                    {
                        "position_id": p.position_id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "leverage": p.leverage,
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "unrealized_pnl": p.unrealized_pnl,
                        "entry_time": p.entry_time
                    }
                    for p in positions
                ]

        except Exception as e:
            logger.error(f"❌ Open positions get error: {e}")
            return []

    async def get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position by ID

        Args:
            position_id: Position ID

        Returns:
            Dict: Position data or None
        """
        try:
            async with self.session() as session:
                query = select(Position).where(Position.position_id == position_id)
                result = await session.execute(query)
                p = result.scalars().first()

                if p:
                    return {
                        "id": p.id,
                        "position_id": p.position_id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "leverage": p.leverage,
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "unrealized_pnl": p.unrealized_pnl,
                        "status": p.status,
                        "entry_time": p.entry_time,
                        "exit_time": p.exit_time,
                        "exit_price": p.exit_price
                    }

                return None

        except Exception as e:
            logger.error(f"❌ Position get error: {e}")
            return None

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: int
    ) -> bool:
        """
        Close position

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_time: Exit timestamp (ms)

        Returns:
            bool: True if successful
        """
        try:
            async with self.session() as session:
                query = select(Position).where(Position.position_id == position_id)
                result = await session.execute(query)
                position = result.scalars().first()

                if not position:
                    logger.warning(f"⚠️ Position not found: {position_id}")
                    return False

                position.status = "CLOSED"
                position.exit_price = exit_price
                position.exit_time = exit_time

                # Calculate final PnL
                if position.side == "LONG":
                    pnl = (exit_price - position.entry_price) * position.quantity
                else:  # SHORT
                    pnl = (position.entry_price - exit_price) * position.quantity

                position.unrealized_pnl = pnl

                await session.commit()
                logger.info(f"✅ Position closed: {position_id} (PnL: {pnl:.2f})")
                return True

        except Exception as e:
            logger.error(f"❌ Position close error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import time
    import uuid

    print("=" * 60)
    print("🧪 PositionService Test")
    print("=" * 60)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = PositionService(db)

        # Test 1: Save position
        print("\nTest 1: Save position")
        pos_id = f"TEST_{uuid.uuid4().hex[:8]}"
        success = await service.save_position({
            "position_id": pos_id,
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.1,
            "entry_price": 50000.0,
            "current_price": 50500.0,
            "leverage": 10,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "unrealized_pnl": 50.0,
            "status": "OPEN",
            "entry_time": int(time.time() * 1000)
        })
        print(f"   ✅ Position saved: {success}")

        # Test 2: Get open positions
        print("\nTest 2: Get open positions")
        positions = await service.get_open_positions()
        print(f"   ✅ Got {len(positions)} open positions")

        # Test 3: Get position by ID
        print("\nTest 3: Get position by ID")
        position = await service.get_position_by_id(pos_id)
        print(f"   ✅ Position: {position['symbol'] if position else 'None'}")

        # Test 4: Update position
        print("\nTest 4: Update position")
        success = await service.save_position({
            "position_id": pos_id,
            "current_price": 51000.0,
            "unrealized_pnl": 100.0
        })
        print(f"   ✅ Position updated: {success}")

        # Test 5: Close position
        print("\nTest 5: Close position")
        success = await service.close_position(
            position_id=pos_id,
            exit_price=51500.0,
            exit_time=int(time.time() * 1000)
        )
        print(f"   ✅ Position closed: {success}")

        await db.stop()

    asyncio.run(test())
    print("\n✅ All tests completed!")
    print("=" * 60)
