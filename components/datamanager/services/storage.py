#!/usr/bin/env python3
"""
components/datamanager/services/storage.py
SuperBot - Data Storage Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Data storage management - Kline registry and live buffer

Features:
- KlineDataRegistry model for parquet file tracking
- LiveKlineBuffer model for live candle buffering
- Registry CRUD operations

Usage:
    from components.datamanager.services.storage import StorageService, KlineDataRegistry

    service = StorageService(db_manager)
    await service.register_kline_file(symbol_id=1, timeframe_id=1, file_path="...")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, BigInteger, Boolean, DateTime, Index
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.storage")


# ============================================
# MODELS
# ============================================

class KlineDataRegistry(Base):
    """Parquet kline data file registry"""
    __tablename__ = "kline_data_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, nullable=False, index=True)
    timeframe_id = Column(Integer, nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_candles = Column(Integer)
    is_complete = Column(Boolean, default=True)
    last_updated_at = Column(DateTime)
    next_update_due = Column(DateTime)
    data_hash = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_kline_registry_symbol', 'symbol_id', 'timeframe_id'),
        Index('idx_kline_registry_dates', 'start_date', 'end_date'),
    )


class LiveKlineBuffer(Base):
    """Live candle buffer before parquet archiving"""
    __tablename__ = "live_kline_buffer"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, nullable=False, index=True)
    timeframe_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(BigInteger)
    quote_volume = Column(Float)
    trades = Column(Integer)
    taker_buy_base = Column(Float)
    taker_buy_quote = Column(Float)
    is_closed = Column(Boolean, default=False)
    archived_to_parquet = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_live_kline_symbol_time', 'symbol_id', 'timeframe_id', 'timestamp'),
        Index('idx_live_kline_archive', 'archived_to_parquet', 'is_closed'),
    )


# ============================================
# SERVICE
# ============================================

class StorageService(BaseService):
    """Data storage management service"""

    # ============================================
    # Kline Registry
    # ============================================

    async def register_kline_file(
        self,
        symbol_id: int,
        timeframe_id: int,
        file_path: str,
        start_date: datetime,
        end_date: datetime,
        total_candles: int,
        file_size_bytes: Optional[int] = None,
        data_hash: Optional[str] = None
    ) -> Optional[int]:
        """Register a parquet kline file"""
        try:
            async with self.session() as session:
                registry = KlineDataRegistry(
                    symbol_id=symbol_id,
                    timeframe_id=timeframe_id,
                    file_path=file_path,
                    start_date=start_date,
                    end_date=end_date,
                    total_candles=total_candles,
                    file_size_bytes=file_size_bytes,
                    data_hash=data_hash,
                    is_complete=True,
                    last_updated_at=get_utc_now()
                )
                session.add(registry)
                await session.commit()
                await session.refresh(registry)

                logger.info(f"✅ Kline file registered: {file_path}")
                return registry.id
        except Exception as e:
            logger.error(f"❌ Register kline file error: {e}")
            return None

    async def get_kline_registry(
        self,
        symbol_id: Optional[int] = None,
        timeframe_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get kline registry entries"""
        try:
            async with self.session() as session:
                query = select(KlineDataRegistry)

                if symbol_id:
                    query = query.where(KlineDataRegistry.symbol_id == symbol_id)
                if timeframe_id:
                    query = query.where(KlineDataRegistry.timeframe_id == timeframe_id)

                query = query.order_by(KlineDataRegistry.start_date)
                result = await session.execute(query)
                entries = result.scalars().all()

                return [
                    {
                        'id': e.id,
                        'symbol_id': e.symbol_id,
                        'timeframe_id': e.timeframe_id,
                        'file_path': e.file_path,
                        'file_size_bytes': e.file_size_bytes,
                        'start_date': e.start_date.isoformat() if e.start_date else None,
                        'end_date': e.end_date.isoformat() if e.end_date else None,
                        'total_candles': e.total_candles,
                        'is_complete': e.is_complete,
                        'last_updated_at': e.last_updated_at.isoformat() if e.last_updated_at else None,
                        'data_hash': e.data_hash
                    }
                    for e in entries
                ]
        except Exception as e:
            logger.error(f"❌ Get kline registry error: {e}")
            return []

    async def update_kline_registry(
        self,
        registry_id: int,
        **kwargs
    ) -> bool:
        """Update kline registry entry"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(KlineDataRegistry).where(KlineDataRegistry.id == registry_id)
                )
                entry = result.scalar_one_or_none()

                if not entry:
                    return False

                allowed_fields = [
                    'file_path', 'file_size_bytes', 'start_date', 'end_date',
                    'total_candles', 'is_complete', 'data_hash', 'next_update_due'
                ]

                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(entry, key, value)

                entry.last_updated_at = get_utc_now()
                entry.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Kline registry updated: {registry_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Update kline registry error: {e}")
            return False

    async def delete_kline_registry(self, registry_id: int) -> bool:
        """Delete kline registry entry"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(KlineDataRegistry).where(KlineDataRegistry.id == registry_id)
                )
                entry = result.scalar_one_or_none()

                if entry:
                    await session.delete(entry)
                    await session.commit()
                    logger.info(f"✅ Kline registry deleted: {registry_id}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete kline registry error: {e}")
            return False

    # ============================================
    # Live Kline Buffer
    # ============================================

    async def save_live_kline(
        self,
        symbol_id: int,
        timeframe_id: int,
        timestamp: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        is_closed: bool = False
    ) -> Optional[int]:
        """Save live kline to buffer"""
        try:
            async with self.session() as session:
                # Check if exists
                result = await session.execute(
                    select(LiveKlineBuffer).where(
                        LiveKlineBuffer.symbol_id == symbol_id,
                        LiveKlineBuffer.timeframe_id == timeframe_id,
                        LiveKlineBuffer.timestamp == timestamp
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update
                    existing.open = open_price
                    existing.high = high
                    existing.low = low
                    existing.close = close
                    existing.volume = volume
                    existing.is_closed = is_closed
                    await session.commit()
                    return existing.id
                else:
                    # Create
                    kline = LiveKlineBuffer(
                        symbol_id=symbol_id,
                        timeframe_id=timeframe_id,
                        timestamp=timestamp,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        is_closed=is_closed
                    )
                    session.add(kline)
                    await session.commit()
                    await session.refresh(kline)
                    return kline.id

        except Exception as e:
            logger.error(f"❌ Save live kline error: {e}")
            return None

    async def get_live_klines(
        self,
        symbol_id: int,
        timeframe_id: int,
        limit: int = 100,
        closed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get live klines from buffer"""
        try:
            async with self.session() as session:
                query = select(LiveKlineBuffer).where(
                    LiveKlineBuffer.symbol_id == symbol_id,
                    LiveKlineBuffer.timeframe_id == timeframe_id
                )

                if closed_only:
                    query = query.where(LiveKlineBuffer.is_closed == True)

                query = query.order_by(LiveKlineBuffer.timestamp.desc()).limit(limit)
                result = await session.execute(query)
                klines = result.scalars().all()

                return [
                    {
                        'id': k.id,
                        'timestamp': k.timestamp,
                        'open': k.open,
                        'high': k.high,
                        'low': k.low,
                        'close': k.close,
                        'volume': k.volume,
                        'is_closed': k.is_closed,
                        'archived': k.archived_to_parquet
                    }
                    for k in reversed(klines)
                ]
        except Exception as e:
            logger.error(f"❌ Get live klines error: {e}")
            return []

    async def mark_klines_archived(
        self,
        symbol_id: int,
        timeframe_id: int,
        before_timestamp: int
    ) -> int:
        """Mark klines as archived to parquet"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(LiveKlineBuffer).where(
                        LiveKlineBuffer.symbol_id == symbol_id,
                        LiveKlineBuffer.timeframe_id == timeframe_id,
                        LiveKlineBuffer.timestamp < before_timestamp,
                        LiveKlineBuffer.is_closed == True,
                        LiveKlineBuffer.archived_to_parquet == False
                    )
                )
                klines = result.scalars().all()

                count = 0
                for k in klines:
                    k.archived_to_parquet = True
                    count += 1

                await session.commit()
                logger.info(f"✅ Marked {count} klines as archived")
                return count
        except Exception as e:
            logger.error(f"❌ Mark klines archived error: {e}")
            return 0

    async def cleanup_archived_klines(
        self,
        symbol_id: int,
        timeframe_id: int
    ) -> int:
        """Delete archived klines from buffer"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(LiveKlineBuffer).where(
                        LiveKlineBuffer.symbol_id == symbol_id,
                        LiveKlineBuffer.timeframe_id == timeframe_id,
                        LiveKlineBuffer.archived_to_parquet == True
                    )
                )
                klines = result.scalars().all()

                count = len(klines)
                for k in klines:
                    await session.delete(k)

                await session.commit()
                logger.info(f"✅ Cleaned up {count} archived klines")
                return count
        except Exception as e:
            logger.error(f"❌ Cleanup archived klines error: {e}")
            return 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 StorageService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = StorageService(db)

        # Test 1: Register kline file
        print("\n1️⃣ Register kline file")
        reg_id = await service.register_kline_file(
            symbol_id=1,
            timeframe_id=1,
            file_path="data/parquet/BTCUSDT_1m_2024.parquet",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            total_candles=525600,
            file_size_bytes=50_000_000
        )
        print(f"   ✅ Registry ID: {reg_id}")

        # Test 2: Get kline registry
        print("\n2️⃣ Get kline registry")
        registry = await service.get_kline_registry(symbol_id=1)
        print(f"   ✅ Found {len(registry)} entries")
        for r in registry:
            print(f"      - {r['file_path']}: {r['total_candles']} candles")

        # Test 3: Update registry
        print("\n3️⃣ Update registry")
        success = await service.update_kline_registry(
            reg_id,
            total_candles=530000,
            end_date=datetime(2025, 1, 15)
        )
        print(f"   ✅ Updated: {success}")

        # Test 4: Save live klines
        print("\n4️⃣ Save live klines")
        base_ts = 1705900000000
        for i in range(5):
            await service.save_live_kline(
                symbol_id=1,
                timeframe_id=1,
                timestamp=base_ts + (i * 60000),
                open_price=65000 + i * 100,
                high=65200 + i * 100,
                low=64900 + i * 100,
                close=65100 + i * 100,
                volume=100.5 + i * 10,
                is_closed=True
            )
        print("   ✅ 5 klines saved")

        # Test 5: Get live klines
        print("\n5️⃣ Get live klines")
        klines = await service.get_live_klines(symbol_id=1, timeframe_id=1)
        print(f"   ✅ Found {len(klines)} klines")
        for k in klines[:3]:
            print(f"      - ts={k['timestamp']}: O={k['open']}, C={k['close']}")

        # Test 6: Mark as archived
        print("\n6️⃣ Mark klines as archived")
        archived = await service.mark_klines_archived(
            symbol_id=1,
            timeframe_id=1,
            before_timestamp=base_ts + 180000
        )
        print(f"   ✅ Archived: {archived} klines")

        # Test 7: Cleanup archived
        print("\n7️⃣ Cleanup archived klines")
        cleaned = await service.cleanup_archived_klines(symbol_id=1, timeframe_id=1)
        print(f"   ✅ Cleaned: {cleaned} klines")

        klines = await service.get_live_klines(symbol_id=1, timeframe_id=1)
        print(f"   ✅ Remaining: {len(klines)} klines")

        # Test 8: Delete registry
        print("\n8️⃣ Delete registry")
        success = await service.delete_kline_registry(reg_id)
        print(f"   ✅ Deleted: {success}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
