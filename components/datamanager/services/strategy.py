#!/usr/bin/env python3
"""
components/datamanager/services/strategy.py
SuperBot - Strategy & Backtest Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Strategy and backtesting management - models and CRUD operations

Features:
- Strategy, StrategyComponent, BacktestRun, BacktestTrade models
- LiveTrade, TradingSession models
- Strategy CRUD operations
- Backtest management

Usage:
    from components.datamanager.services.strategy import StrategyService, Strategy

    service = StrategyService(db_manager)
    strategy_id = await service.create_strategy("RSI Divergence", "indicator")

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, Text
from sqlalchemy.future import select

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.strategy")


# ============================================
# MODELS
# ============================================

class Strategy(Base):
    """Trading strategy definitions"""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(150))
    description = Column(Text)
    strategy_type = Column(String(20))
    code = Column(Text)
    config = Column(Text)
    entry_rules = Column(Text)
    exit_rules = Column(Text)
    risk_management = Column(Text)
    category_id = Column(Integer)
    is_active = Column(Boolean, default=True)
    is_validated = Column(Boolean, default=False)
    backtest_count = Column(Integer, default=0)
    avg_win_rate = Column(Float)
    avg_profit_factor = Column(Float)
    last_backtest_at = Column(DateTime)
    created_by = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_strategies_active', 'is_active'),
        Index('idx_strategies_category', 'category_id'),
    )


class StrategyComponent(Base):
    """Strategy component definitions (indicators, scanners, patterns)"""
    __tablename__ = "strategy_components"

    id = Column(Integer, primary_key=True, autoincrement=True)
    component_type = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    display_name = Column(String(150))
    category = Column(String(50))
    module_path = Column(String(200))
    class_name = Column(String(100))
    parameters = Column(Text)
    description = Column(Text)
    usage_example = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestRun(Base):
    """Backtest execution records"""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, unique=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    strategy_snapshot = Column(Text)
    category_id = Column(Integer)
    symbol_ids = Column(Text)
    timeframe_id = Column(Integer, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, default=10000)
    position_size_pct = Column(Float, default=10)
    max_positions = Column(Integer, default=5)
    commission_pct = Column(Float, default=0.1)
    slippage_pct = Column(Float, default=0.05)
    status = Column(String(20), default='pending')
    progress = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    final_capital = Column(Float)
    total_profit = Column(Float)
    total_profit_pct = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    parquet_file = Column(String(500))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_backtest_strategy', 'strategy_id'),
        Index('idx_backtest_status', 'status'),
        Index('idx_backtest_dates', 'start_date', 'end_date'),
    )


class BacktestTrade(Base):
    """Backtest trade records"""
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id = Column(Integer, nullable=False, index=True)
    symbol_id = Column(Integer, nullable=False, index=True)
    side = Column(String(10))
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    commission = Column(Float)
    net_pnl = Column(Float)
    entry_signal = Column(String(50))
    exit_signal = Column(String(50))
    duration_minutes = Column(Integer)
    mae = Column(Float)  # Maximum Adverse Excursion
    mfe = Column(Float)  # Maximum Favorable Excursion

    __table_args__ = (
        Index('idx_backtest_trades_run', 'backtest_run_id'),
        Index('idx_backtest_trades_symbol', 'symbol_id'),
    )


class LiveTrade(Base):
    """Live trading records"""
    __tablename__ = "live_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), nullable=False, unique=True)
    strategy_id = Column(Integer, index=True)
    symbol_id = Column(Integer, nullable=False, index=True)
    portfolio_id = Column(Integer)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20))
    entry_time = Column(DateTime, nullable=False)
    entry_order_id = Column(String(100))
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_commission = Column(Float, default=0)
    exit_time = Column(DateTime)
    exit_order_id = Column(String(100))
    exit_price = Column(Float)
    exit_commission = Column(Float, default=0)
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    trailing_stop = Column(Boolean, default=False)
    realized_pnl = Column(Float)
    realized_pnl_pct = Column(Float)
    status = Column(String(20), default='open')
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    risk_amount = Column(Float)
    risk_reward_ratio = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_live_trades_status', 'status'),
        Index('idx_live_trades_symbol', 'symbol_id'),
        Index('idx_live_trades_strategy', 'strategy_id'),
        Index('idx_live_trades_entry_time', 'entry_time'),
    )


class TradingSession(Base):
    """Trading session tracking"""
    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False, unique=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    category_id = Column(Integer)
    max_positions = Column(Integer, default=5)
    position_size_pct = Column(Float, default=10)
    risk_per_trade_pct = Column(Float, default=2)
    max_daily_loss_pct = Column(Float, default=5)
    initial_capital = Column(Float, nullable=False)
    current_capital = Column(Float)
    status = Column(String(20), default='active')
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    total_pnl = Column(Float)
    total_pnl_pct = Column(Float)
    daily_loss = Column(Float, default=0)
    daily_profit = Column(Float, default=0)
    emergency_stop = Column(Boolean, default=False)
    emergency_reason = Column(Text)
    started_at = Column(DateTime, nullable=False)
    stopped_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_trading_sessions_status', 'status'),
        Index('idx_trading_sessions_strategy', 'strategy_id'),
    )


# ============================================
# SERVICE
# ============================================

class StrategyService(BaseService):
    """Strategy and backtesting management service"""

    # ============================================
    # Strategies
    # ============================================

    async def get_all_strategies(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Get all strategies"""
        try:
            async with self.session() as session:
                query = select(Strategy).order_by(Strategy.name)
                if active_only:
                    query = query.where(Strategy.is_active == True)

                result = await session.execute(query)
                strategies = result.scalars().all()

                return [
                    {
                        'id': s.id,
                        'name': s.name,
                        'display_name': s.display_name,
                        'description': s.description,
                        'strategy_type': s.strategy_type,
                        'is_active': s.is_active,
                        'is_validated': s.is_validated,
                        'backtest_count': s.backtest_count,
                        'avg_win_rate': s.avg_win_rate,
                        'avg_profit_factor': s.avg_profit_factor,
                        'last_backtest_at': s.last_backtest_at.isoformat() if s.last_backtest_at else None,
                        'created_by': s.created_by,
                        'created_at': s.created_at.isoformat() if s.created_at else None
                    }
                    for s in strategies
                ]
        except Exception as e:
            logger.error(f"❌ Get all strategies error: {e}")
            return []

    async def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get strategy by ID with full details"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(Strategy).where(Strategy.id == strategy_id)
                )
                s = result.scalar_one_or_none()

                if not s:
                    return None

                return {
                    'id': s.id,
                    'name': s.name,
                    'display_name': s.display_name,
                    'description': s.description,
                    'strategy_type': s.strategy_type,
                    'code': s.code,
                    'config': s.config,
                    'entry_rules': s.entry_rules,
                    'exit_rules': s.exit_rules,
                    'risk_management': s.risk_management,
                    'category_id': s.category_id,
                    'is_active': s.is_active,
                    'is_validated': s.is_validated,
                    'backtest_count': s.backtest_count,
                    'avg_win_rate': s.avg_win_rate,
                    'avg_profit_factor': s.avg_profit_factor,
                    'last_backtest_at': s.last_backtest_at.isoformat() if s.last_backtest_at else None,
                    'created_by': s.created_by,
                    'created_at': s.created_at.isoformat() if s.created_at else None,
                    'updated_at': s.updated_at.isoformat() if s.updated_at else None
                }
        except Exception as e:
            logger.error(f"❌ Get strategy by ID error: {e}")
            return None

    async def create_strategy(
        self,
        name: str,
        strategy_type: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[str] = None,
        entry_rules: Optional[str] = None,
        exit_rules: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Optional[int]:
        """Create new strategy"""
        try:
            async with self.session() as session:
                strategy = Strategy(
                    name=name,
                    display_name=display_name or name,
                    description=description,
                    strategy_type=strategy_type,
                    config=config,
                    entry_rules=entry_rules,
                    exit_rules=exit_rules,
                    created_by=created_by,
                    is_active=True,
                    is_validated=False,
                    backtest_count=0
                )
                session.add(strategy)
                await session.commit()
                await session.refresh(strategy)

                logger.info(f"✅ Strategy created: {name} (ID: {strategy.id})")
                return strategy.id
        except Exception as e:
            logger.error(f"❌ Create strategy error: {e}")
            return None

    async def update_strategy(self, strategy_id: int, **kwargs) -> bool:
        """Update strategy fields"""
        try:
            if not kwargs:
                return True

            async with self.session() as session:
                result = await session.execute(
                    select(Strategy).where(Strategy.id == strategy_id)
                )
                strategy = result.scalar_one_or_none()

                if not strategy:
                    return False

                allowed_fields = [
                    'name', 'display_name', 'description', 'strategy_type',
                    'code', 'config', 'entry_rules', 'exit_rules', 'risk_management',
                    'category_id', 'is_active', 'is_validated'
                ]

                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(strategy, key, value)

                strategy.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Strategy updated: {strategy_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Update strategy error: {e}")
            return False

    async def delete_strategy(self, strategy_id: int) -> bool:
        """Delete strategy"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(Strategy).where(Strategy.id == strategy_id)
                )
                strategy = result.scalar_one_or_none()

                if strategy:
                    await session.delete(strategy)
                    await session.commit()
                    logger.info(f"✅ Strategy deleted: {strategy_id}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete strategy error: {e}")
            return False

    # ============================================
    # Backtest Runs
    # ============================================

    async def create_backtest_run(
        self,
        strategy_id: int,
        timeframe_id: int,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        symbol_ids: Optional[str] = None
    ) -> Optional[int]:
        """Create new backtest run"""
        try:
            async with self.session() as session:
                run_id = f"BT-{uuid.uuid4().hex[:8].upper()}"

                backtest = BacktestRun(
                    run_id=run_id,
                    strategy_id=strategy_id,
                    timeframe_id=timeframe_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    symbol_ids=symbol_ids,
                    status='pending'
                )
                session.add(backtest)
                await session.commit()
                await session.refresh(backtest)

                logger.info(f"✅ Backtest run created: {run_id}")
                return backtest.id
        except Exception as e:
            logger.error(f"❌ Create backtest run error: {e}")
            return None

    async def get_backtest_runs(
        self,
        strategy_id: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get backtest runs with optional filters"""
        try:
            async with self.session() as session:
                query = select(BacktestRun).order_by(BacktestRun.created_at.desc())

                if strategy_id:
                    query = query.where(BacktestRun.strategy_id == strategy_id)
                if status:
                    query = query.where(BacktestRun.status == status)

                result = await session.execute(query)
                runs = result.scalars().all()

                return [
                    {
                        'id': r.id,
                        'run_id': r.run_id,
                        'strategy_id': r.strategy_id,
                        'timeframe_id': r.timeframe_id,
                        'start_date': r.start_date.isoformat() if r.start_date else None,
                        'end_date': r.end_date.isoformat() if r.end_date else None,
                        'initial_capital': r.initial_capital,
                        'status': r.status,
                        'progress': r.progress,
                        'total_trades': r.total_trades,
                        'win_rate': r.win_rate,
                        'total_profit_pct': r.total_profit_pct,
                        'profit_factor': r.profit_factor,
                        'max_drawdown': r.max_drawdown,
                        'created_at': r.created_at.isoformat() if r.created_at else None
                    }
                    for r in runs
                ]
        except Exception as e:
            logger.error(f"❌ Get backtest runs error: {e}")
            return []

    async def update_backtest_status(
        self,
        backtest_id: int,
        status: str,
        progress: Optional[int] = None,
        results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update backtest run status and results"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(BacktestRun).where(BacktestRun.id == backtest_id)
                )
                backtest = result.scalar_one_or_none()

                if not backtest:
                    return False

                backtest.status = status
                if progress is not None:
                    backtest.progress = progress

                if status == 'running' and not backtest.started_at:
                    backtest.started_at = get_utc_now()

                if status == 'completed' and results:
                    backtest.completed_at = get_utc_now()
                    backtest.total_trades = results.get('total_trades', 0)
                    backtest.winning_trades = results.get('winning_trades', 0)
                    backtest.losing_trades = results.get('losing_trades', 0)
                    backtest.win_rate = results.get('win_rate')
                    backtest.final_capital = results.get('final_capital')
                    backtest.total_profit = results.get('total_profit')
                    backtest.total_profit_pct = results.get('total_profit_pct')
                    backtest.profit_factor = results.get('profit_factor')
                    backtest.sharpe_ratio = results.get('sharpe_ratio')
                    backtest.max_drawdown = results.get('max_drawdown')

                await session.commit()
                logger.info(f"✅ Backtest status updated: {backtest_id} -> {status}")
                return True
        except Exception as e:
            logger.error(f"❌ Update backtest status error: {e}")
            return False

    # ============================================
    # Live Trades
    # ============================================

    async def create_live_trade(
        self,
        symbol_id: int,
        side: str,
        entry_price: float,
        quantity: float,
        strategy_id: Optional[int] = None,
        portfolio_id: Optional[int] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[int]:
        """Create new live trade"""
        try:
            async with self.session() as session:
                trade_id = f"LT-{uuid.uuid4().hex[:8].upper()}"

                trade = LiveTrade(
                    trade_id=trade_id,
                    strategy_id=strategy_id,
                    symbol_id=symbol_id,
                    portfolio_id=portfolio_id,
                    side=side,
                    entry_time=get_utc_now(),
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    status='open',
                    notes=notes
                )
                session.add(trade)
                await session.commit()
                await session.refresh(trade)

                logger.info(f"✅ Live trade created: {trade_id}")
                return trade.id
        except Exception as e:
            logger.error(f"❌ Create live trade error: {e}")
            return None

    async def get_live_trades(
        self,
        status: Optional[str] = None,
        strategy_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get live trades with optional filters"""
        try:
            async with self.session() as session:
                query = select(LiveTrade).order_by(LiveTrade.entry_time.desc())

                if status:
                    query = query.where(LiveTrade.status == status)
                if strategy_id:
                    query = query.where(LiveTrade.strategy_id == strategy_id)

                result = await session.execute(query)
                trades = result.scalars().all()

                return [
                    {
                        'id': t.id,
                        'trade_id': t.trade_id,
                        'strategy_id': t.strategy_id,
                        'symbol_id': t.symbol_id,
                        'side': t.side,
                        'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                        'entry_price': t.entry_price,
                        'quantity': t.quantity,
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'exit_price': t.exit_price,
                        'stop_loss_price': t.stop_loss_price,
                        'take_profit_price': t.take_profit_price,
                        'realized_pnl': t.realized_pnl,
                        'realized_pnl_pct': t.realized_pnl_pct,
                        'status': t.status,
                        'notes': t.notes
                    }
                    for t in trades
                ]
        except Exception as e:
            logger.error(f"❌ Get live trades error: {e}")
            return []

    async def close_live_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: Optional[str] = None
    ) -> bool:
        """Close a live trade"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(LiveTrade).where(LiveTrade.id == trade_id)
                )
                trade = result.scalar_one_or_none()

                if not trade:
                    return False

                trade.exit_time = get_utc_now()
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.status = 'closed'

                # Calculate P&L
                if trade.side == 'BUY':
                    pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - exit_price) * trade.quantity

                trade.realized_pnl = pnl
                trade.realized_pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100

                await session.commit()
                logger.info(f"✅ Live trade closed: {trade.trade_id}, P&L: ${pnl:.2f}")
                return True
        except Exception as e:
            logger.error(f"❌ Close live trade error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 StrategyService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = StrategyService(db)

        # Test 1: Create strategies
        print("\n1️⃣ Create strategies")
        s1_id = await service.create_strategy(
            name="RSI_Divergence",
            strategy_type="indicator",
            display_name="RSI Divergence Strategy",
            description="Buy when RSI divergence detected",
            entry_rules="RSI < 30 and price making lower lows",
            exit_rules="RSI > 70 or TP/SL hit",
            created_by="admin"
        )
        print(f"   ✅ Strategy 1 ID: {s1_id}")

        s2_id = await service.create_strategy(
            name="MA_Crossover",
            strategy_type="indicator",
            display_name="Moving Average Crossover",
            description="Classic MA crossover strategy"
        )
        print(f"   ✅ Strategy 2 ID: {s2_id}")

        # Test 2: Get all strategies
        print("\n2️⃣ Get all strategies")
        strategies = await service.get_all_strategies()
        print(f"   ✅ Found {len(strategies)} strategies")
        for s in strategies:
            print(f"      - {s['name']}: {s['strategy_type']}")

        # Test 3: Get strategy by ID
        print("\n3️⃣ Get strategy by ID")
        strategy = await service.get_strategy_by_id(s1_id)
        if strategy:
            print(f"   ✅ Name: {strategy['name']}")
            print(f"   ✅ Entry rules: {strategy['entry_rules']}")

        # Test 4: Update strategy
        print("\n4️⃣ Update strategy")
        success = await service.update_strategy(
            s1_id,
            description="Enhanced RSI divergence with volume confirmation",
            is_validated=True
        )
        print(f"   ✅ Updated: {success}")

        # Test 5: Create backtest run
        print("\n5️⃣ Create backtest run")
        bt_id = await service.create_backtest_run(
            strategy_id=s1_id,
            timeframe_id=1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            initial_capital=10000.0,
            symbol_ids="1,2,3"
        )
        print(f"   ✅ Backtest ID: {bt_id}")

        # Test 6: Update backtest status
        print("\n6️⃣ Update backtest status")
        await service.update_backtest_status(bt_id, 'running', progress=50)
        await service.update_backtest_status(bt_id, 'completed', results={
            'total_trades': 150,
            'winning_trades': 90,
            'losing_trades': 60,
            'win_rate': 60.0,
            'final_capital': 15000.0,
            'total_profit': 5000.0,
            'total_profit_pct': 50.0,
            'profit_factor': 1.8,
            'sharpe_ratio': 1.5,
            'max_drawdown': 12.5
        })
        print("   ✅ Backtest completed")

        # Test 7: Get backtest runs
        print("\n7️⃣ Get backtest runs")
        runs = await service.get_backtest_runs(strategy_id=s1_id)
        print(f"   ✅ Found {len(runs)} runs")
        for r in runs:
            print(f"      - {r['run_id']}: {r['status']}, WR: {r['win_rate']}%")

        # Test 8: Create live trade
        print("\n8️⃣ Create live trade")
        trade_id = await service.create_live_trade(
            symbol_id=1,
            side='BUY',
            entry_price=65000.0,
            quantity=0.1,
            strategy_id=s1_id,
            stop_loss_price=63000.0,
            take_profit_price=70000.0,
            notes="RSI divergence signal"
        )
        print(f"   ✅ Trade ID: {trade_id}")

        # Test 9: Get live trades
        print("\n9️⃣ Get live trades")
        trades = await service.get_live_trades(status='open')
        print(f"   ✅ Found {len(trades)} open trades")

        # Test 10: Close live trade
        print("\n🔟 Close live trade")
        success = await service.close_live_trade(
            trade_id=trade_id,
            exit_price=68000.0,
            exit_reason="TP1 hit"
        )
        print(f"   ✅ Closed: {success}")

        trades = await service.get_live_trades(status='closed')
        if trades:
            print(f"   ✅ P&L: ${trades[0]['realized_pnl']:.2f}")

        # Test 11: Delete strategy
        print("\n1️⃣1️⃣ Delete strategy")
        success = await service.delete_strategy(s2_id)
        print(f"   ✅ Deleted: {success}")

        strategies = await service.get_all_strategies()
        print(f"   ✅ Remaining strategies: {len(strategies)}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
