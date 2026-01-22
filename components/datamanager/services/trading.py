#!/usr/bin/env python3
"""
components/datamanager/services/trading.py
SuperBot - Trading Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Trading data management - models and CRUD operations

Features:
- Candle, Trade, Order, PnL, Balance models
- OHLCV candle storage and retrieval
- Trade history
- Order history
- PnL tracking
- Balance snapshots

Usage:
    from components.datamanager.services.trading import TradingService, Candle, Trade

    service = TradingService(db_manager)
    await service.save_candle({...})
    candles = await service.get_candles("BTCUSDT", "1h")

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

logger = get_logger("components.datamanager.services.trading")


# ============================================
# MODELS
# ============================================

class Candle(Base):
    """
    OHLCV Candle model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    is_closed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite unique constraint
    __table_args__ = (
        Index('idx_symbol_tf_ts', 'symbol', 'timeframe', 'timestamp', unique=True),
    )


class Trade(Base):
    """
    Trade history model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    quote_qty = Column(Float, nullable=False)
    commission = Column(Float, default=0)
    commission_asset = Column(String(10))
    trade_id = Column(BigInteger, index=True)
    order_id = Column(BigInteger, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    is_maker = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    """
    Order history model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    order_id = Column(BigInteger, unique=True, index=True)
    client_order_id = Column(String(50), index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    order_type = Column(String(20), nullable=False)  # LIMIT/MARKET
    price = Column(Float)
    quantity = Column(Float, nullable=False)
    executed_qty = Column(Float, default=0)
    status = Column(String(20), nullable=False)  # NEW/FILLED/CANCELED
    time_in_force = Column(String(10))  # GTC/IOC/FOK
    timestamp = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PnL(Base):
    """
    PnL (Profit and Loss) tracking model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), index=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)  # LONG/SHORT
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    commission = Column(Float, default=0)
    net_pnl = Column(Float, nullable=False)
    entry_time = Column(BigInteger, nullable=False)
    exit_time = Column(BigInteger, nullable=False)
    duration_minutes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Balance(Base):
    """
    Balance/Equity tracking model

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "balance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_equity = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    used_margin = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    timestamp = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================
# SERVICE
# ============================================

class TradingService(BaseService):
    """Trading data management service"""

    # Stats tracking
    stats = {
        "total_candles": 0,
        "total_trades": 0,
        "total_orders": 0,
        "total_pnl_records": 0
    }

    # ============================================
    # Candle Operations
    # ============================================

    async def save_candle(self, candle_data: Dict[str, Any]) -> bool:
        """
        Save candle

        Args:
            candle_data: Candle data dict with symbol, timeframe, timestamp, OHLCV

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 211-244 ===
        try:
            async with self.session() as session:
                candle = Candle(
                    symbol=candle_data["symbol"],
                    timeframe=candle_data["timeframe"],
                    timestamp=candle_data["timestamp"],
                    open=candle_data["open"],
                    high=candle_data["high"],
                    low=candle_data["low"],
                    close=candle_data["close"],
                    volume=candle_data["volume"],
                    is_closed=candle_data.get("is_closed", True)
                )

                session.add(candle)
                await session.commit()

                self.stats["total_candles"] += 1

                return True

        except Exception as e:
            logger.error(f"❌ Candle save error: {e}")
            return False

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get candles

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            limit: Max candle count

        Returns:
            List: Candle list
        """
        # === PORTED FROM data_manager.py lines 246-302 ===
        try:
            async with self.session() as session:
                query = select(Candle).where(
                    Candle.symbol == symbol,
                    Candle.timeframe == timeframe
                )

                if start_time:
                    query = query.where(Candle.timestamp >= start_time)

                if end_time:
                    query = query.where(Candle.timestamp <= end_time)

                query = query.order_by(Candle.timestamp.desc()).limit(limit)

                result = await session.execute(query)
                candles = result.scalars().all()

                return [
                    {
                        "symbol": c.symbol,
                        "timeframe": c.timeframe,
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                        "is_closed": c.is_closed
                    }
                    for c in candles
                ]

        except Exception as e:
            logger.error(f"❌ Candle get error: {e}")
            return []

    # ============================================
    # Trade Operations
    # ============================================

    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Save trade

        Args:
            trade_data: Trade data dict

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 308-321 ===
        try:
            async with self.session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                await session.commit()

                self.stats["total_trades"] += 1
                return True

        except Exception as e:
            logger.error(f"❌ Trade save error: {e}")
            return False

    # ============================================
    # Order Operations
    # ============================================

    async def save_order(self, order_data: Dict[str, Any]) -> bool:
        """
        Save order

        Args:
            order_data: Order data dict

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 327-340 ===
        try:
            async with self.session() as session:
                order = Order(**order_data)
                session.add(order)
                await session.commit()

                self.stats["total_orders"] += 1
                return True

        except Exception as e:
            logger.error(f"❌ Order save error: {e}")
            return False

    async def get_orders(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get orders

        Args:
            status: Order status (NEW/FILLED/CANCELED/PENDING)
            symbol: Trading pair
            limit: Max order count

        Returns:
            List: Order list
        """
        # === PORTED FROM data_manager.py lines 342-393 ===
        try:
            async with self.session() as session:
                query = select(Order)

                if status:
                    query = query.where(Order.status == status)

                if symbol:
                    query = query.where(Order.symbol == symbol)

                query = query.order_by(Order.timestamp.desc()).limit(limit)

                result = await session.execute(query)
                orders = result.scalars().all()

                return [
                    {
                        "order_id": o.order_id,
                        "client_order_id": o.client_order_id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "order_type": o.order_type,
                        "price": o.price,
                        "quantity": o.quantity,
                        "executed_qty": o.executed_qty,
                        "status": o.status,
                        "time_in_force": o.time_in_force,
                        "timestamp": o.timestamp
                    }
                    for o in orders
                ]

        except Exception as e:
            logger.error(f"❌ Order get error: {e}")
            return []

    # ============================================
    # PnL Operations
    # ============================================

    async def save_pnl(self, pnl_data: Dict[str, Any]) -> bool:
        """
        Save PnL record

        Args:
            pnl_data: PnL data dict

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 399-412 ===
        try:
            async with self.session() as session:
                pnl = PnL(**pnl_data)
                session.add(pnl)
                await session.commit()

                self.stats["total_pnl_records"] += 1
                return True

        except Exception as e:
            logger.error(f"❌ PnL save error: {e}")
            return False

    async def get_total_pnl(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate total PnL

        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy

        Returns:
            Dict: PnL summary (total_pnl, total_trades, winning_trades, win_rate)
        """
        # === PORTED FROM data_manager.py lines 414-446 ===
        try:
            async with self.session() as session:
                query = select(PnL)

                if symbol:
                    query = query.where(PnL.symbol == symbol)

                if strategy:
                    query = query.where(PnL.strategy == strategy)

                result = await session.execute(query)
                pnl_records = result.scalars().all()

                total_pnl = sum(p.net_pnl for p in pnl_records)
                total_trades = len(pnl_records)
                winning_trades = len([p for p in pnl_records if p.net_pnl > 0])

                return {
                    "total_pnl": total_pnl,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0
                }

        except Exception as e:
            logger.error(f"❌ PnL calculation error: {e}")
            return {}

    # ============================================
    # Balance Operations
    # ============================================

    async def save_balance(self, balance_data: Dict[str, Any]) -> bool:
        """
        Save balance snapshot

        Args:
            balance_data: Balance data dict

        Returns:
            bool: True if successful
        """
        # === PORTED FROM data_manager.py lines 452-463 ===
        try:
            async with self.session() as session:
                balance = Balance(**balance_data)
                session.add(balance)
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"❌ Balance save error: {e}")
            return False

    async def get_latest_balance(self) -> Optional[Dict[str, Any]]:
        """
        Get latest balance

        Returns:
            Dict: Balance data (total_equity, available_balance, etc.)
        """
        # === PORTED FROM data_manager.py lines 465-491 ===
        try:
            async with self.session() as session:
                query = select(Balance).order_by(Balance.timestamp.desc()).limit(1)
                result = await session.execute(query)
                balance = result.scalars().first()

                if balance:
                    return {
                        "total_equity": balance.total_equity,
                        "available_balance": balance.available_balance,
                        "used_margin": balance.used_margin,
                        "unrealized_pnl": balance.unrealized_pnl,
                        "timestamp": balance.timestamp
                    }

                return None

        except Exception as e:
            logger.error(f"❌ Balance get error: {e}")
            return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import time

    print("=" * 60)
    print("🧪 TradingService Test")
    print("=" * 60)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = TradingService(db)

        # Test 1: Save candle
        print("\nTest 1: Save candle")
        success = await service.save_candle({
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "timestamp": int(time.time() * 1000),
            "open": 50000.0,
            "high": 50500.0,
            "low": 49500.0,
            "close": 50200.0,
            "volume": 1234.56
        })
        print(f"   ✅ Candle saved: {success}")

        # Test 2: Get candles
        print("\nTest 2: Get candles")
        candles = await service.get_candles("BTCUSDT", "1h", limit=10)
        print(f"   ✅ Got {len(candles)} candles")

        # Test 3: Save order
        print("\nTest 3: Save order")
        success = await service.save_order({
            "symbol": "BTCUSDT",
            "order_id": 123456789,
            "client_order_id": "test_order_1",
            "side": "BUY",
            "order_type": "LIMIT",
            "price": 50000.0,
            "quantity": 0.1,
            "executed_qty": 0.1,
            "status": "FILLED",
            "timestamp": int(time.time() * 1000)
        })
        print(f"   ✅ Order saved: {success}")

        # Test 4: Get orders
        print("\nTest 4: Get orders")
        orders = await service.get_orders(limit=10)
        print(f"   ✅ Got {len(orders)} orders")

        # Test 5: Save PnL
        print("\nTest 5: Save PnL")
        success = await service.save_pnl({
            "symbol": "BTCUSDT",
            "strategy": "test_strategy",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 0.1,
            "side": "LONG",
            "pnl": 100.0,
            "pnl_percent": 2.0,
            "commission": 1.0,
            "net_pnl": 99.0,
            "entry_time": int(time.time() * 1000) - 3600000,
            "exit_time": int(time.time() * 1000),
            "duration_minutes": 60
        })
        print(f"   ✅ PnL saved: {success}")

        # Test 6: Get total PnL
        print("\nTest 6: Get total PnL")
        pnl_summary = await service.get_total_pnl()
        print(f"   ✅ PnL summary: {pnl_summary}")

        # Test 7: Save balance
        print("\nTest 7: Save balance")
        success = await service.save_balance({
            "total_equity": 10000.0,
            "available_balance": 9000.0,
            "used_margin": 1000.0,
            "unrealized_pnl": 50.0,
            "timestamp": int(time.time() * 1000)
        })
        print(f"   ✅ Balance saved: {success}")

        # Test 8: Get latest balance
        print("\nTest 8: Get latest balance")
        balance = await service.get_latest_balance()
        print(f"   ✅ Latest balance: {balance}")

        await db.stop()

    asyncio.run(test())
    print("\n✅ All tests completed!")
    print("=" * 60)
