#!/usr/bin/env python3
"""
components/datamanager/services/portfolio.py
SuperBot - Portfolio Models & Service
Author: SuperBot Team
Date: 2026-01-22
Version: 1.0.0

Portfolio management - models and CRUD operations

Features:
- Portfolio, PortfolioPosition, PortfolioPositionEntry, PortfolioExitTarget models
- Portfolio holdings management
- Position tracking with DCA support
- Take Profit / Stop Loss targets
- P&L calculations

Usage:
    from components.datamanager.services.portfolio import PortfolioService, Portfolio

    service = PortfolioService(db_manager)
    portfolio_id = await service.create_portfolio("BTC Portfolio")
    position_id = await service.create_position(portfolio_id, symbol_id=1, quantity=1.0)

Dependencies:
    - python>=3.10
    - sqlalchemy>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, Text
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from components.datamanager.base import Base, BaseService, DatabaseManager
from core.logger_engine import get_logger
from core.timezone_utils import get_utc_now

logger = get_logger("components.datamanager.services.portfolio")


# ============================================
# MODELS
# ============================================

class PortfolioHolding(Base):
    """
    Portfolio holdings with performance tracking

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)

    # Holdings
    total_quantity = Column(Float, default=0)
    average_entry_price = Column(Float)
    current_price = Column(Float)

    # Valuation
    cost_basis = Column(Float)  # Total invested
    current_value = Column(Float)
    unrealized_pnl = Column(Float, default=0)
    unrealized_pnl_pct = Column(Float, default=0)

    # Performance
    realized_pnl = Column(Float, default=0)  # From closed trades
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)

    # Allocation
    portfolio_weight = Column(Float)  # Percentage of total portfolio

    # Tracking
    first_purchase = Column(DateTime)
    last_purchase = Column(DateTime)
    last_sale = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_current_value', 'current_value'),
        Index('idx_unrealized_pnl', 'unrealized_pnl'),
    )


class Portfolio(Base):
    """
    Portfolio definitions - Links to ExchangeAccount

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)

    # Exchange Reference (no FK constraint for modular services)
    exchange_account_id = Column(Integer, nullable=True, index=True)

    # Settings
    is_active = Column(Boolean, default=True)
    is_primary = Column(Boolean, default=False)
    auto_sync = Column(Boolean, default=False)
    last_synced_at = Column(DateTime)

    # Summary Fields
    total_value = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    position_count = Column(Integer, default=0)

    # Performance Metrics
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    monthly_pnl = Column(Float, default=0.0)
    monthly_pnl_pct = Column(Float, default=0.0)

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_portfolios_exchange_account', 'exchange_account_id'),
    )


class PortfolioPosition(Base):
    """
    Portfolio position tracking

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, nullable=False, index=True)
    symbol_id = Column(Integer, nullable=True, index=True)  # Nullable for manual positions
    symbol = Column(String(50), nullable=True, index=True)  # Symbol string (e.g., "BTC/USDT")

    # Position Details
    side = Column(String(10), nullable=False)  # LONG, SHORT
    position_type = Column(String(20))  # SPOT, FUTURES
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)

    # Cost & P&L
    total_cost = Column(Float)
    average_price = Column(Float)
    unrealized_pnl = Column(Float)
    unrealized_pnl_pct = Column(Float)
    realized_pnl = Column(Float)
    exit_price = Column(Float)

    # Source Tracking
    source = Column(String(30), default='manual')
    external_id = Column(String(100))

    # Status
    is_open = Column(Boolean, default=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_positions_portfolio', 'portfolio_id', 'is_open'),
        Index('idx_positions_symbol', 'symbol_id'),
    )


class PortfolioPositionEntry(Base):
    """
    Multi-entry support for positions (DCA)

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolio_position_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, nullable=False, index=True)

    # Entry Details
    entry_number = Column(Integer, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)

    # Status
    status = Column(String(20), default='filled')
    entry_time = Column(DateTime, default=datetime.utcnow)

    # Source Tracking
    source = Column(String(30), default='manual')
    external_order_id = Column(String(100))

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_position_entries_position', 'position_id', 'entry_number'),
    )


class PortfolioExitTarget(Base):
    """
    Take Profit & Stop Loss levels for positions

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolio_exit_targets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, nullable=False, index=True)

    # Target Details
    target_type = Column(String(10), nullable=False)  # TP or SL
    target_number = Column(Integer, nullable=False)
    target_price = Column(Float, nullable=False)
    quantity_percentage = Column(Float, nullable=False)

    # Execution Status
    status = Column(String(20), default='pending')
    triggered_at = Column(DateTime)
    triggered_price = Column(Float)
    quantity_closed = Column(Float)
    pnl = Column(Float)

    # Order Tracking
    external_order_id = Column(String(100))

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_exit_targets_position', 'position_id', 'target_type'),
        Index('idx_exit_targets_status', 'status'),
    )


class PositionTrade(Base):
    """
    Individual trades for a position (DCA entries/exits)

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "position_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, nullable=False, index=True)

    # Trade Details
    trade_type = Column(String(10), nullable=False, default='ENTRY')  # ENTRY or EXIT
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    cost = Column(Float)
    fee = Column(Float, default=0.0)

    # Metadata
    traded_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    __table_args__ = (
        Index('idx_position_trades_position_id', 'position_id'),
        Index('idx_position_trades_type', 'trade_type'),
        Index('idx_position_trades_date', 'traded_at'),
    )


class PortfolioTransaction(Base):
    """
    Portfolio transaction history

    === PORTED FROM dm_models.py ===
    """
    __tablename__ = "portfolio_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, nullable=False, index=True)
    transaction_type = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    fee = Column(Float, default=0)
    source = Column(String(20))
    external_id = Column(String(100))
    executed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_transactions_position', 'position_id'),
        Index('idx_transactions_executed', 'executed_at'),
    )


# ============================================
# SERVICE
# ============================================

class PortfolioService(BaseService):
    """Portfolio management service"""

    # ============================================
    # Portfolio Holdings
    # ============================================

    async def update_portfolio_holding(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        quantity: float,
        entry_price: float,
        current_price: float
    ) -> bool:
        """
        Update or create portfolio holding

        === PORTED FROM data_manager.py lines 211-283 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioHolding).where(PortfolioHolding.symbol == symbol)
                )
                holding = result.scalars().first()

                cost_basis = quantity * entry_price
                current_value = quantity * current_price
                unrealized_pnl = current_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

                if holding:
                    holding.total_quantity = quantity
                    holding.average_entry_price = entry_price
                    holding.current_price = current_price
                    holding.cost_basis = cost_basis
                    holding.current_value = current_value
                    holding.unrealized_pnl = unrealized_pnl
                    holding.unrealized_pnl_pct = unrealized_pnl_pct
                    holding.last_purchase = get_utc_now()
                    holding.updated_at = get_utc_now()
                else:
                    holding = PortfolioHolding(
                        symbol=symbol,
                        base_asset=base_asset,
                        quote_asset=quote_asset,
                        total_quantity=quantity,
                        average_entry_price=entry_price,
                        current_price=current_price,
                        cost_basis=cost_basis,
                        current_value=current_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        first_purchase=get_utc_now(),
                        last_purchase=get_utc_now()
                    )
                    session.add(holding)

                await session.commit()
                logger.info(f"✅ Portfolio holding updated: {symbol}")
                return True

        except Exception as e:
            logger.error(f"❌ Update portfolio holding error: {e}")
            return False

    async def get_portfolio_holdings(self) -> List[Dict[str, Any]]:
        """
        Get all portfolio holdings

        === PORTED FROM data_manager.py lines 285-325 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioHolding).order_by(PortfolioHolding.current_value.desc())
                )
                holdings = result.scalars().all()

                return [
                    {
                        'id': h.id,
                        'symbol': h.symbol,
                        'base_asset': h.base_asset,
                        'quote_asset': h.quote_asset,
                        'total_quantity': h.total_quantity,
                        'average_entry_price': h.average_entry_price,
                        'current_price': h.current_price,
                        'cost_basis': h.cost_basis,
                        'current_value': h.current_value,
                        'unrealized_pnl': h.unrealized_pnl,
                        'unrealized_pnl_pct': h.unrealized_pnl_pct,
                        'realized_pnl': h.realized_pnl,
                        'total_trades': h.total_trades,
                        'portfolio_weight': h.portfolio_weight,
                        'first_purchase': h.first_purchase.isoformat() if h.first_purchase else None,
                        'last_purchase': h.last_purchase.isoformat() if h.last_purchase else None
                    }
                    for h in holdings
                ]

        except Exception as e:
            logger.error(f"❌ Get portfolio holdings error: {e}")
            return []

    # ============================================
    # Portfolios
    # ============================================

    async def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """
        Get all portfolios with exchange account info

        === PORTED FROM data_manager.py lines 331-371 ===
        """
        try:
            # Import here to avoid circular imports
            from components.datamanager.services.exchange import ExchangeAccount

            async with self.session() as session:
                # Left join with exchange_accounts to get exchange info
                from sqlalchemy import outerjoin
                result = await session.execute(
                    select(Portfolio, ExchangeAccount)
                    .outerjoin(ExchangeAccount, Portfolio.exchange_account_id == ExchangeAccount.id)
                    .order_by(
                        Portfolio.is_primary.desc(),
                        Portfolio.name
                    )
                )
                rows = result.all()

                return [
                    {
                        'id': p.id,
                        'name': p.name,
                        'exchange_account_id': p.exchange_account_id,
                        'is_active': p.is_active,
                        'is_primary': p.is_primary,
                        'auto_sync': p.auto_sync,
                        'last_synced_at': p.last_synced_at.isoformat() if p.last_synced_at else None,
                        'total_value': p.total_value,
                        'total_cost': p.total_cost,
                        'total_pnl': p.total_pnl,
                        'pnl_percentage': p.pnl_percentage,
                        'position_count': p.position_count,
                        'notes': p.notes,
                        'created_at': p.created_at.isoformat() if p.created_at else None,
                        'updated_at': p.updated_at.isoformat() if p.updated_at else None,
                        # Exchange account fields
                        'exchange': ea.exchange if ea else None,
                        'environment': ea.environment if ea else None,
                        'account_type': ea.account_type if ea else None
                    }
                    for p, ea in rows
                ]
        except Exception as e:
            logger.error(f"❌ Get all portfolios error: {e}")
            return []

    async def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """
        Get portfolio by ID with exchange account info

        === PORTED FROM data_manager.py lines 373-412 ===
        """
        try:
            # Import here to avoid circular imports
            from components.datamanager.services.exchange import ExchangeAccount

            async with self.session() as session:
                # Left join with exchange_accounts to get exchange info
                result = await session.execute(
                    select(Portfolio, ExchangeAccount)
                    .outerjoin(ExchangeAccount, Portfolio.exchange_account_id == ExchangeAccount.id)
                    .where(Portfolio.id == portfolio_id)
                )
                row = result.first()

                if not row:
                    return None

                p, ea = row

                return {
                    'id': p.id,
                    'name': p.name,
                    'exchange_account_id': p.exchange_account_id,
                    'is_active': p.is_active,
                    'is_primary': p.is_primary,
                    'auto_sync': p.auto_sync,
                    'last_synced_at': p.last_synced_at.isoformat() if p.last_synced_at else None,
                    'total_value': p.total_value,
                    'total_cost': p.total_cost,
                    'total_pnl': p.total_pnl,
                    'pnl_percentage': p.pnl_percentage,
                    'position_count': p.position_count,
                    'notes': p.notes,
                    'created_at': p.created_at.isoformat() if p.created_at else None,
                    'updated_at': p.updated_at.isoformat() if p.updated_at else None,
                    # Exchange account fields
                    'exchange': ea.exchange if ea else None,
                    'environment': ea.environment if ea else None,
                    'account_type': ea.account_type if ea else None
                }
        except Exception as e:
            logger.error(f"❌ Get portfolio by ID error: {e}")
            return None

    async def create_portfolio(
        self,
        name: str,
        exchange_account_id: Optional[int] = None,
        notes: Optional[str] = None,
        is_primary: bool = False
    ) -> Optional[int]:
        """
        Create new portfolio

        === PORTED FROM data_manager.py lines 414-449 ===
        """
        try:
            async with self.session() as session:
                if is_primary:
                    # Unset existing primary
                    existing = await session.execute(
                        select(Portfolio).where(Portfolio.is_primary == True)
                    )
                    for p in existing.scalars().all():
                        p.is_primary = False

                portfolio = Portfolio(
                    name=name,
                    exchange_account_id=exchange_account_id,
                    is_active=True,
                    is_primary=is_primary,
                    auto_sync=False,
                    notes=notes
                )
                session.add(portfolio)
                await session.commit()
                await session.refresh(portfolio)

                logger.info(f"✅ Portfolio created: {name} (ID: {portfolio.id})")
                return portfolio.id
        except Exception as e:
            logger.error(f"❌ Create portfolio error: {e}")
            return None

    async def update_portfolio(self, portfolio_id: int, **kwargs) -> bool:
        """
        Update portfolio fields

        === PORTED FROM data_manager.py lines 451-479 ===
        """
        try:
            if not kwargs:
                return True

            async with self.session() as session:
                result = await session.execute(
                    select(Portfolio).where(Portfolio.id == portfolio_id)
                )
                portfolio = result.scalar_one_or_none()

                if not portfolio:
                    return False

                allowed_fields = [
                    'name', 'exchange_account_id', 'is_active', 'is_primary',
                    'auto_sync', 'notes', 'total_value', 'total_cost',
                    'total_pnl', 'pnl_percentage', 'position_count'
                ]

                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(portfolio, key, value)

                portfolio.updated_at = get_utc_now()
                await session.commit()

                logger.info(f"✅ Portfolio updated: {portfolio_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Update portfolio error: {e}")
            return False

    async def delete_portfolio(self, portfolio_id: int) -> bool:
        """
        Delete portfolio and all its positions

        === PORTED FROM data_manager.py lines 481-500 ===
        """
        try:
            async with self.session() as session:
                # Delete positions first
                positions = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.portfolio_id == portfolio_id)
                )
                for pos in positions.scalars().all():
                    await session.delete(pos)

                # Delete portfolio
                result = await session.execute(
                    select(Portfolio).where(Portfolio.id == portfolio_id)
                )
                portfolio = result.scalar_one_or_none()

                if portfolio:
                    await session.delete(portfolio)
                    await session.commit()
                    logger.info(f"✅ Portfolio deleted: {portfolio_id}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete portfolio error: {e}")
            return False

    async def update_portfolio_sync_time(self, portfolio_id: int) -> bool:
        """Update portfolio last sync time"""
        try:
            return await self.update_portfolio(portfolio_id, last_synced_at=get_utc_now())
        except Exception as e:
            logger.error(f"❌ Update sync time error: {e}")
            return False

    # ============================================
    # Positions
    # ============================================

    async def get_portfolio_positions(
        self,
        portfolio_id: int,
        is_open: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get portfolio positions with optional filter

        === PORTED FROM data_manager.py lines 523-604 ===
        """
        try:
            async with self.session() as session:
                query = select(PortfolioPosition).where(
                    PortfolioPosition.portfolio_id == portfolio_id
                )

                if is_open is not None:
                    query = query.where(PortfolioPosition.is_open == is_open)

                query = query.order_by(PortfolioPosition.opened_at.desc())
                result = await session.execute(query)
                positions = result.scalars().all()

                position_list = []
                for p in positions:
                    # Get entries total quantity
                    entries_result = await session.execute(
                        select(PortfolioPositionEntry).where(
                            PortfolioPositionEntry.position_id == p.id,
                            PortfolioPositionEntry.status == 'filled'
                        )
                    )
                    entries = entries_result.scalars().all()
                    entries_total_quantity = sum(e.quantity for e in entries)

                    position_list.append({
                        'id': p.id,
                        'portfolio_id': p.portfolio_id,
                        'symbol_id': p.symbol_id,
                        'symbol': p.symbol,
                        'side': p.side,
                        'position_type': p.position_type,
                        'quantity': p.quantity,
                        'entries_total_quantity': entries_total_quantity,
                        'total_quantity': p.quantity + entries_total_quantity,
                        'entry_price': p.entry_price,
                        'current_price': p.current_price,
                        'total_cost': p.total_cost,
                        'average_price': p.average_price,
                        'unrealized_pnl': p.unrealized_pnl,
                        'unrealized_pnl_pct': p.unrealized_pnl_pct,
                        'realized_pnl': p.realized_pnl,
                        'exit_price': p.exit_price,
                        'source': p.source,
                        'external_id': p.external_id,
                        'is_open': p.is_open,
                        'opened_at': p.opened_at.isoformat() if p.opened_at else None,
                        'closed_at': p.closed_at.isoformat() if p.closed_at else None,
                        'notes': p.notes
                    })

                return position_list
        except Exception as e:
            logger.error(f"❌ Get portfolio positions error: {e}")
            return []

    async def get_position_by_id(self, position_id: int) -> Optional[Dict[str, Any]]:
        """
        Get single position by ID

        === PORTED FROM data_manager.py lines 606-649 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.id == position_id)
                )
                pos = result.scalar_one_or_none()

                if not pos:
                    return None

                return {
                    'id': pos.id,
                    'portfolio_id': pos.portfolio_id,
                    'symbol_id': pos.symbol_id,
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'position_type': pos.position_type,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'total_cost': pos.total_cost,
                    'average_price': pos.average_price,
                    'market_value': pos.quantity * pos.current_price if pos.current_price else 0,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'is_open': pos.is_open,
                    'opened_at': pos.opened_at.isoformat() if pos.opened_at else None,
                    'closed_at': pos.closed_at.isoformat() if pos.closed_at else None,
                    'source': pos.source,
                    'external_id': pos.external_id,
                    'notes': pos.notes
                }
        except Exception as e:
            logger.error(f"❌ Get position by ID error: {e}")
            return None

    async def create_position(
        self,
        portfolio_id: int,
        quantity: float,
        entry_price: float,
        symbol: Optional[str] = None,
        symbol_id: Optional[int] = None,
        side: str = 'LONG',
        position_type: str = 'SPOT',
        opened_at: Optional[str] = None,
        source: str = 'manual',
        external_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[int]:
        """
        Create new portfolio position

        Args:
            portfolio_id: Portfolio ID
            quantity: Position quantity
            entry_price: Entry price
            symbol: Symbol string (e.g., "BTC/USDT") - for manual positions
            symbol_id: Symbol ID from exchange_symbols table (optional)
            side: LONG or SHORT (default: LONG)
            position_type: SPOT or FUTURES (default: SPOT)
            opened_at: ISO datetime string (default: now)
            source: Position source (default: manual)
            external_id: External order ID
            notes: Optional notes

        === PORTED FROM data_manager.py lines 651-775 ===
        """
        try:
            if not symbol and not symbol_id:
                logger.error("❌ Either symbol or symbol_id is required")
                return None

            async with self.session() as session:
                total_cost = quantity * entry_price

                # Parse opened_at datetime
                opened_at_dt = get_utc_now()
                if opened_at:
                    try:
                        opened_at_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                    except Exception:
                        pass

                position = PortfolioPosition(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    symbol_id=symbol_id,
                    side=side,
                    position_type=position_type,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=entry_price,
                    total_cost=total_cost,
                    average_price=entry_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    source=source,
                    external_id=external_id,
                    is_open=True,
                    opened_at=opened_at_dt,
                    notes=notes
                )
                session.add(position)
                await session.commit()
                await session.refresh(position)

                symbol_info = symbol or f"symbol_id={symbol_id}"
                logger.info(f"✅ Position created: {symbol_info} x{quantity} @ {entry_price}")
                return position.id
        except Exception as e:
            logger.error(f"❌ Create position error: {e}")
            return None

    async def update_position_price(self, position_id: int, current_price: float) -> bool:
        """
        Update position current price and calculate P&L

        === PORTED FROM data_manager.py lines 777-803 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.id == position_id)
                )
                position = result.scalars().first()

                if not position:
                    return False

                position.current_price = current_price
                current_value = position.quantity * current_price
                cost = position.quantity * position.entry_price
                position.unrealized_pnl = current_value - cost
                position.unrealized_pnl_pct = (position.unrealized_pnl / cost * 100) if cost > 0 else 0
                position.updated_at = get_utc_now()

                await session.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Update position price error: {e}")
            return False

    async def update_position(
        self,
        position_id: int,
        quantity: Optional[float] = None,
        entry_price: Optional[float] = None,
        opened_at: Optional[str] = None,
        notes: Optional[str] = None,
        is_open: Optional[bool] = None,
        exit_price: Optional[float] = None
    ) -> bool:
        """
        Update position details

        === PORTED FROM data_manager.py lines 805-879 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.id == position_id)
                )
                position = result.scalar_one_or_none()

                if not position:
                    return False

                was_open = position.is_open

                if quantity is not None:
                    position.quantity = quantity
                if entry_price is not None:
                    position.entry_price = entry_price
                if opened_at is not None:
                    # Parse datetime string to datetime object
                    from datetime import datetime
                    if isinstance(opened_at, str):
                        try:
                            position.opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                        except ValueError:
                            position.opened_at = datetime.strptime(opened_at, '%Y-%m-%dT%H:%M')
                    else:
                        position.opened_at = opened_at
                if notes is not None:
                    position.notes = notes
                if is_open is not None:
                    position.is_open = is_open

                # Handle position closing
                if was_open and is_open == False and exit_price is not None:
                    realized_pnl = (exit_price - position.entry_price) * position.quantity
                    position.realized_pnl = realized_pnl
                    position.exit_price = exit_price
                    position.closed_at = get_utc_now()

                    # Create EXIT trade record
                    exit_trade = PositionTrade(
                        position_id=position_id,
                        trade_type='EXIT',
                        quantity=position.quantity,
                        price=exit_price,
                        cost=exit_price * position.quantity,
                        fee=0.0,
                        traded_at=get_utc_now(),
                        notes=f"Position closed with P&L: ${realized_pnl:.2f}"
                    )
                    session.add(exit_trade)
                    logger.info(f"💰 Position #{position_id} closed with P&L: ${realized_pnl:.2f}")

                # Recalculate averages
                if quantity is not None or entry_price is not None:
                    await self._recalculate_position_averages(session, position_id)

                await session.commit()
                logger.info(f"✅ Position updated: #{position_id}")
                return True

        except Exception as e:
            logger.error(f"❌ Update position error: {e}")
            return False

    async def close_position(self, position_id: int, close_price: Optional[float] = None) -> bool:
        """
        Close a position

        === PORTED FROM data_manager.py lines 881-907 ===
        """
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.id == position_id)
                )
                position = result.scalars().first()

                if not position:
                    return False

                position.is_open = False
                position.closed_at = get_utc_now()

                if close_price:
                    position.current_price = close_price
                    current_value = position.quantity * close_price
                    cost = position.quantity * position.entry_price
                    position.unrealized_pnl = current_value - cost
                    position.unrealized_pnl_pct = (position.unrealized_pnl / cost * 100) if cost > 0 else 0

                await session.commit()
                logger.info(f"✅ Position closed: {position_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Close position error: {e}")
            return False

    async def delete_position(self, position_id: int) -> bool:
        """Delete a position"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPosition).where(PortfolioPosition.id == position_id)
                )
                position = result.scalar_one_or_none()

                if position:
                    await session.delete(position)
                    await session.commit()
                    logger.info(f"✅ Position deleted: {position_id}")
                    return True

                return False
        except Exception as e:
            logger.error(f"❌ Delete position error: {e}")
            return False

    # ============================================
    # Position Entries (DCA)
    # ============================================

    async def add_position_entry(
        self,
        position_id: int,
        entry_number: int,
        quantity: float,
        entry_price: float,
        status: str = 'filled',
        source: str = 'manual',
        traded_at: Optional[str] = None
    ) -> Optional[int]:
        """
        Add entry to a position (multi-entry / DCA)

        === PORTED FROM data_manager.py lines 926-973 ===
        """
        try:
            async with self.session() as session:
                cost = quantity * entry_price

                entry_time = get_utc_now()
                if traded_at:
                    entry_time = datetime.fromisoformat(traded_at.replace('Z', '+00:00'))

                entry = PortfolioPositionEntry(
                    position_id=position_id,
                    entry_number=entry_number,
                    quantity=quantity,
                    entry_price=entry_price,
                    cost=cost,
                    status=status,
                    entry_time=entry_time,
                    source=source
                )

                session.add(entry)
                await session.commit()
                await session.refresh(entry)

                if status == 'filled':
                    await self._recalculate_position_averages(session, position_id)

                logger.info(f"✅ Position entry added: Position #{position_id}, Entry #{entry_number}")
                return entry.id

        except Exception as e:
            logger.error(f"❌ Add position entry error: {e}")
            return None

    async def get_position_entries(self, position_id: int) -> List[Dict[str, Any]]:
        """Get all entries for a position"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPositionEntry)
                    .where(PortfolioPositionEntry.position_id == position_id)
                    .order_by(PortfolioPositionEntry.entry_number)
                )
                entries = result.scalars().all()

                return [
                    {
                        'id': e.id,
                        'position_id': e.position_id,
                        'entry_number': e.entry_number,
                        'quantity': e.quantity,
                        'entry_price': e.entry_price,
                        'cost': e.cost,
                        'status': e.status,
                        'entry_time': e.entry_time.isoformat() if e.entry_time else None,
                        'source': e.source,
                        'notes': e.notes
                    }
                    for e in entries
                ]

        except Exception as e:
            logger.error(f"❌ Get position entries error: {e}")
            return []

    async def delete_position_entry(self, entry_id: int) -> bool:
        """Delete a position entry"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioPositionEntry).where(PortfolioPositionEntry.id == entry_id)
                )
                entry = result.scalar_one_or_none()

                if not entry:
                    return False

                position_id = entry.position_id
                was_filled = entry.status == 'filled'

                await session.delete(entry)

                if was_filled:
                    await self._recalculate_position_averages(session, position_id)

                await session.commit()
                logger.info(f"✅ Position entry deleted: #{entry_id}")
                return True

        except Exception as e:
            logger.error(f"❌ Delete position entry error: {e}")
            return False

    async def _recalculate_position_averages(self, session, position_id: int):
        """Recalculate average price from position + entries"""
        try:
            result = await session.execute(
                select(PortfolioPosition).where(PortfolioPosition.id == position_id)
            )
            position = result.scalar_one_or_none()

            if not position:
                return

            entries_result = await session.execute(
                select(PortfolioPositionEntry)
                .where(PortfolioPositionEntry.position_id == position_id)
                .where(PortfolioPositionEntry.status == 'filled')
            )
            entries = entries_result.scalars().all()

            position_cost = position.quantity * position.entry_price
            entries_quantity = sum(e.quantity for e in entries)
            entries_cost = sum(e.cost for e in entries)

            total_quantity = position.quantity + entries_quantity
            total_cost = position_cost + entries_cost
            average_price = total_cost / total_quantity if total_quantity > 0 else position.entry_price

            position.average_price = average_price
            position.total_cost = total_cost
            await session.commit()

        except Exception as e:
            logger.error(f"❌ Recalculate position averages error: {e}")

    # ============================================
    # Exit Targets (TP/SL)
    # ============================================

    async def add_exit_target(
        self,
        position_id: int,
        target_type: str,
        target_number: int,
        target_price: float,
        quantity_percentage: float
    ) -> Optional[int]:
        """Add Take Profit or Stop Loss target"""
        try:
            async with self.session() as session:
                target = PortfolioExitTarget(
                    position_id=position_id,
                    target_type=target_type,
                    target_number=target_number,
                    target_price=target_price,
                    quantity_percentage=quantity_percentage,
                    status='pending'
                )

                session.add(target)
                await session.commit()
                await session.refresh(target)

                logger.info(f"✅ Exit target added: {target_type}{target_number} @ {target_price}")
                return target.id

        except Exception as e:
            logger.error(f"❌ Add exit target error: {e}")
            return None

    async def get_exit_targets(
        self,
        position_id: int,
        target_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get exit targets for a position"""
        try:
            async with self.session() as session:
                query = select(PortfolioExitTarget).where(
                    PortfolioExitTarget.position_id == position_id
                )

                if target_type:
                    query = query.where(PortfolioExitTarget.target_type == target_type)

                query = query.order_by(
                    PortfolioExitTarget.target_type,
                    PortfolioExitTarget.target_number
                )

                result = await session.execute(query)
                targets = result.scalars().all()

                return [
                    {
                        'id': t.id,
                        'position_id': t.position_id,
                        'target_type': t.target_type,
                        'target_number': t.target_number,
                        'target_price': t.target_price,
                        'quantity_percentage': t.quantity_percentage,
                        'status': t.status,
                        'triggered_at': t.triggered_at.isoformat() if t.triggered_at else None,
                        'triggered_price': t.triggered_price,
                        'pnl': t.pnl,
                        'notes': t.notes
                    }
                    for t in targets
                ]

        except Exception as e:
            logger.error(f"❌ Get exit targets error: {e}")
            return []

    async def trigger_exit_target(
        self,
        target_id: int,
        triggered_price: float,
        quantity_closed: float,
        pnl: float
    ) -> bool:
        """Mark exit target as triggered"""
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(PortfolioExitTarget).where(PortfolioExitTarget.id == target_id)
                )
                target = result.scalar_one_or_none()

                if not target:
                    return False

                target.status = 'triggered'
                target.triggered_at = get_utc_now()
                target.triggered_price = triggered_price
                target.quantity_closed = quantity_closed
                target.pnl = pnl

                await session.commit()
                logger.info(f"✅ Exit target triggered: {target.target_type}{target.target_number}")
                return True

        except Exception as e:
            logger.error(f"❌ Trigger exit target error: {e}")
            return False

    # ============================================
    # Summary
    # ============================================

    async def update_portfolio_summary(self, portfolio_id: int) -> bool:
        """Recalculate and update portfolio summary"""
        try:
            positions = await self.get_portfolio_positions(portfolio_id, is_open=True)

            async with self.session() as session:
                result = await session.execute(
                    select(Portfolio).where(Portfolio.id == portfolio_id)
                )
                portfolio = result.scalar_one_or_none()

                if not portfolio:
                    return False

                if not positions:
                    portfolio.total_value = 0.0
                    portfolio.total_cost = 0.0
                    portfolio.total_pnl = 0.0
                    portfolio.pnl_percentage = 0.0
                    portfolio.position_count = 0
                else:
                    total_cost = sum(p.get('total_cost', 0) or 0 for p in positions)
                    total_pnl = sum(p.get('unrealized_pnl', 0) or 0 for p in positions)
                    total_value = total_cost + total_pnl

                    portfolio.total_value = total_value
                    portfolio.total_cost = total_cost
                    portfolio.total_pnl = total_pnl
                    portfolio.pnl_percentage = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
                    portfolio.position_count = len(positions)

                await session.commit()
                logger.info(f"✅ Portfolio summary updated: #{portfolio_id}")
                return True

        except Exception as e:
            logger.error(f"❌ Update portfolio summary error: {e}")
            return False


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("🧪 PortfolioService Comprehensive Test")
    print("=" * 70)

    async def test():
        db = DatabaseManager({"backend": "sqlite", "path": "data/db/test_services.db"})
        await db.start()

        service = PortfolioService(db)

        # Test 1: Create portfolio
        print("\n1️⃣ Create portfolio")
        portfolio_id = await service.create_portfolio(
            name="BTC Long Portfolio",
            notes="Test portfolio",
            is_primary=True
        )
        print(f"   ✅ Portfolio ID: {portfolio_id}")

        # Test 2: Create second portfolio
        print("\n2️⃣ Create second portfolio")
        portfolio2_id = await service.create_portfolio(
            name="Altcoin Portfolio",
            notes="Altcoin test"
        )
        print(f"   ✅ Portfolio 2 ID: {portfolio2_id}")

        # Test 3: Get all portfolios
        print("\n3️⃣ Get all portfolios")
        portfolios = await service.get_all_portfolios()
        print(f"   ✅ Found {len(portfolios)} portfolios")
        for p in portfolios:
            primary = " (PRIMARY)" if p['is_primary'] else ""
            print(f"      - {p['name']}{primary}")

        # Test 4: Create position
        print("\n4️⃣ Create position")
        position_id = await service.create_position(
            portfolio_id=portfolio_id,
            symbol_id=1,
            quantity=0.5,
            entry_price=65000.0,
            side='LONG',
            position_type='SPOT',
            notes="First BTC position"
        )
        print(f"   ✅ Position ID: {position_id}")

        # Test 5: Add DCA entries
        print("\n5️⃣ Add DCA entries")
        entry1_id = await service.add_position_entry(
            position_id=position_id,
            entry_number=1,
            quantity=0.25,
            entry_price=64000.0
        )
        print(f"   ✅ Entry 1 ID: {entry1_id}")

        entry2_id = await service.add_position_entry(
            position_id=position_id,
            entry_number=2,
            quantity=0.25,
            entry_price=63000.0
        )
        print(f"   ✅ Entry 2 ID: {entry2_id}")

        # Test 6: Get position with entries
        print("\n6️⃣ Get position details")
        position = await service.get_position_by_id(position_id)
        if position:
            print(f"   ✅ Quantity: {position['quantity']}")
            print(f"   ✅ Entry price: ${position['entry_price']}")
            print(f"   ✅ Average price: ${position['average_price']}")

        # Test 7: Get entries
        print("\n7️⃣ Get position entries")
        entries = await service.get_position_entries(position_id)
        print(f"   ✅ Found {len(entries)} entries")
        for e in entries:
            print(f"      - Entry #{e['entry_number']}: {e['quantity']} @ ${e['entry_price']}")

        # Test 8: Add exit targets
        print("\n8️⃣ Add exit targets (TP/SL)")
        tp1_id = await service.add_exit_target(
            position_id=position_id,
            target_type='TP',
            target_number=1,
            target_price=70000.0,
            quantity_percentage=50.0
        )
        print(f"   ✅ TP1 ID: {tp1_id}")

        sl_id = await service.add_exit_target(
            position_id=position_id,
            target_type='SL',
            target_number=1,
            target_price=60000.0,
            quantity_percentage=100.0
        )
        print(f"   ✅ SL1 ID: {sl_id}")

        # Test 9: Get exit targets
        print("\n9️⃣ Get exit targets")
        targets = await service.get_exit_targets(position_id)
        print(f"   ✅ Found {len(targets)} targets")
        for t in targets:
            print(f"      - {t['target_type']}{t['target_number']}: ${t['target_price']} ({t['quantity_percentage']}%)")

        # Test 10: Update position price
        print("\n🔟 Update position price")
        await service.update_position_price(position_id, 67000.0)
        position = await service.get_position_by_id(position_id)
        if position:
            print(f"   ✅ Current price: ${position['current_price']}")
            print(f"   ✅ Unrealized P&L: ${position['unrealized_pnl']:.2f}")

        # Test 11: Update portfolio summary
        print("\n1️⃣1️⃣ Update portfolio summary")
        await service.update_portfolio_summary(portfolio_id)
        portfolio = await service.get_portfolio_by_id(portfolio_id)
        if portfolio:
            print(f"   ✅ Total cost: ${portfolio['total_cost']:.2f}")
            print(f"   ✅ Total P&L: ${portfolio['total_pnl']:.2f}")
            print(f"   ✅ Position count: {portfolio['position_count']}")

        # Test 12: Trigger TP
        print("\n1️⃣2️⃣ Trigger TP1")
        await service.trigger_exit_target(
            target_id=tp1_id,
            triggered_price=70500.0,
            quantity_closed=0.5,
            pnl=2750.0
        )
        targets = await service.get_exit_targets(position_id, target_type='TP')
        for t in targets:
            print(f"   ✅ TP1 status: {t['status']}, P&L: ${t['pnl']}")

        # Test 13: Close position
        print("\n1️⃣3️⃣ Close position")
        await service.close_position(position_id, close_price=68000.0)
        position = await service.get_position_by_id(position_id)
        if position:
            print(f"   ✅ is_open: {position['is_open']}")

        # Test 14: Portfolio holdings
        print("\n1️⃣4️⃣ Portfolio holdings")
        await service.update_portfolio_holding(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            quantity=1.5,
            entry_price=64000.0,
            current_price=68000.0
        )
        holdings = await service.get_portfolio_holdings()
        print(f"   ✅ Found {len(holdings)} holdings")
        for h in holdings:
            print(f"      - {h['symbol']}: {h['total_quantity']} (P&L: ${h['unrealized_pnl']:.2f})")

        # Test 15: Delete operations
        print("\n1️⃣5️⃣ Delete operations")
        await service.delete_position_entry(entry2_id)
        entries = await service.get_position_entries(position_id)
        print(f"   ✅ Remaining entries: {len(entries)}")

        await service.delete_portfolio(portfolio2_id)
        portfolios = await service.get_all_portfolios()
        print(f"   ✅ Remaining portfolios: {len(portfolios)}")

        await db.stop()

    asyncio.run(test())
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
