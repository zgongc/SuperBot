#!/usr/bin/env python3
"""
Portfolio Analytics Service
Business logic layer for 3commas-style portfolio analytics

Combines DataManager + PortfolioCalculator to provide:
- Portfolio statistics
- Token allocation (pie chart data)
- Performance metrics (Sharpe, Sortino, Drawdown)
- 24h/30d P&L breakdown
- Position-level analytics with multi-entry support
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add SuperBot to path
base_dir = Path(__file__).parent.parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from components.datamanager import DataManager
from modules.webui.managers.portfolio_calculator import PortfolioCalculator
from core.logger_engine import LoggerEngine

logger_engine = LoggerEngine()
logger = logger_engine.get_logger(__name__)


class PortfolioAnalyticsService:
    """Service for portfolio analytics and statistics"""

    def __init__(self, data_manager: DataManager):
        """
        Initialize service

        Args:
            data_manager: DataManager instance
        """
        self.dm = data_manager
        self.calculator = PortfolioCalculator()

    async def get_portfolio_statistics(
        self,
        portfolio_id: int,
        include_closed_positions: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive portfolio statistics

        Args:
            portfolio_id: Portfolio ID
            include_closed_positions: Include closed positions for performance metrics

        Returns:
            Dict with all portfolio statistics
        """
        try:
            # Get portfolio
            portfolio = await self.dm.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                logger.error(f"Portfolio not found: {portfolio_id}")
                return {}

            # Get open positions
            open_positions = await self.dm.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=True
            )

            # Get closed positions (for performance metrics)
            closed_positions = []
            if include_closed_positions:
                closed_positions = await self.dm.get_portfolio_positions(
                    portfolio_id=portfolio_id,
                    is_open=False
                )

            # Calculate basic metrics
            total_value = self.calculator.calculate_total_value(open_positions)
            total_cost = self.calculator.calculate_total_cost(open_positions)
            total_pnl, pnl_pct = self.calculator.calculate_pnl(open_positions)

            # Token allocation (for pie chart)
            token_allocation = self.calculator.calculate_token_allocation(open_positions)

            # Trade performance metrics
            win_rate = self.calculator.calculate_win_rate(closed_positions)
            profit_factor = self.calculator.calculate_profit_factor(closed_positions)
            avg_win, avg_loss = self.calculator.calculate_average_win_loss(closed_positions)
            risk_reward = self.calculator.calculate_risk_reward_ratio(closed_positions)

            # Build statistics
            stats = {
                'portfolio_id': portfolio_id,
                'portfolio_name': portfolio['name'],
                'timestamp': datetime.utcnow().isoformat(),

                # Portfolio Overview
                'overview': {
                    'total_value': total_value,
                    'total_cost': total_cost,
                    'total_pnl': total_pnl,
                    'pnl_percentage': pnl_pct,
                    'position_count': len(open_positions),
                    'daily_pnl': portfolio.get('daily_pnl', 0.0),
                    'daily_pnl_pct': portfolio.get('daily_pnl_pct', 0.0),
                    'monthly_pnl': portfolio.get('monthly_pnl', 0.0),
                    'monthly_pnl_pct': portfolio.get('monthly_pnl_pct', 0.0)
                },

                # Token Allocation (Pie Chart)
                'allocation': token_allocation,

                # Trade Performance
                'performance': {
                    'total_trades': len(closed_positions),
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'risk_reward_ratio': risk_reward
                },

                # Risk Metrics (will be calculated if equity curve available)
                'risk_metrics': {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'max_drawdown_value': 0.0
                }
            }

            logger.info(f"✅ Portfolio statistics calculated: {portfolio['name']}")
            return stats

        except Exception as e:
            logger.error(f"❌ Get portfolio statistics error: {e}", exc_info=True)
            return {}

    async def get_position_details(
        self,
        position_id: int,
        include_entries: bool = True,
        include_exit_targets: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed position information with entries and exit targets

        Args:
            position_id: Position ID
            include_entries: Include position entries (multi-entry)
            include_exit_targets: Include TP/SL targets

        Returns:
            Dict with position details
        """
        try:
            # Get position by ID
            position = await self.dm.get_position_by_id(position_id)

            if not position:
                logger.error(f"Position not found: {position_id}")
                return {}

            result = {
                'position': position,
                'entries': [],
                'exit_targets': [],
                'metrics': {}
            }

            # Get entries
            if include_entries:
                entries = await self.dm.get_position_entries(position_id)
                result['entries'] = entries

                # Calculate metrics from entries
                if entries:
                    current_price = position.get('current_price', 0)
                    metrics = self.calculator.calculate_position_metrics(
                        entries=entries,
                        current_price=current_price
                    )
                    result['metrics'] = metrics

            # Get exit targets
            if include_exit_targets:
                exit_targets = await self.dm.get_exit_targets(position_id)
                result['exit_targets'] = exit_targets

                # Separate TP and SL
                result['take_profits'] = [t for t in exit_targets if t['target_type'] == 'TP']
                result['stop_losses'] = [t for t in exit_targets if t['target_type'] == 'SL']

            logger.debug(f"Position details retrieved: {position_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Get position details error: {e}", exc_info=True)
            return {}

    async def get_portfolio_positions_with_details(
        self,
        portfolio_id: int,
        is_open: Optional[bool] = True
    ) -> List[Dict[str, Any]]:
        """
        Get all positions with entries and exit targets

        Args:
            portfolio_id: Portfolio ID
            is_open: Filter by open/closed status

        Returns:
            List of positions with details
        """
        try:
            # Get positions
            positions = await self.dm.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=is_open
            )

            detailed_positions = []

            for position in positions:
                position_id = position['id']

                # Get entries
                entries = await self.dm.get_position_entries(position_id)

                # Get exit targets
                exit_targets = await self.dm.get_exit_targets(position_id)

                # Calculate metrics from entries
                metrics = {}
                if entries:
                    current_price = position.get('current_price', 0)
                    metrics = self.calculator.calculate_position_metrics(
                        entries=entries,
                        current_price=current_price
                    )

                detailed_positions.append({
                    **position,
                    'entries': entries,
                    'exit_targets': exit_targets,
                    'entry_count': len(entries),
                    'tp_count': len([t for t in exit_targets if t['target_type'] == 'TP']),
                    'sl_count': len([t for t in exit_targets if t['target_type'] == 'SL']),
                    'metrics': metrics
                })

            logger.info(f"✅ Retrieved {len(detailed_positions)} positions with details")
            return detailed_positions

        except Exception as e:
            logger.error(f"❌ Get portfolio positions with details error: {e}", exc_info=True)
            return []

    async def calculate_daily_pnl(
        self,
        portfolio_id: int
    ) -> Dict[str, float]:
        """
        Calculate 24h P&L for portfolio

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dict with daily_pnl and daily_pnl_pct
        """
        try:
            # This is a simplified version - in production you'd compare
            # current positions value with value from 24h ago

            # Get current positions
            positions = await self.dm.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=True
            )

            # For now, return zeros (would need historical data)
            # In production: Query transactions/snapshots from last 24h
            daily_pnl = 0.0
            daily_pnl_pct = 0.0

            logger.debug(f"Daily P&L calculated: {portfolio_id}")
            return {
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct
            }

        except Exception as e:
            logger.error(f"❌ Calculate daily P&L error: {e}", exc_info=True)
            return {'daily_pnl': 0.0, 'daily_pnl_pct': 0.0}

    async def calculate_monthly_pnl(
        self,
        portfolio_id: int
    ) -> Dict[str, float]:
        """
        Calculate 30d P&L for portfolio

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dict with monthly_pnl and monthly_pnl_pct
        """
        try:
            # Simplified version - would need historical data
            # Get closed positions from last 30 days
            all_closed = await self.dm.get_portfolio_positions(
                portfolio_id=portfolio_id,
                is_open=False
            )

            # Filter positions closed in last 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            recent_closed = [
                p for p in all_closed
                if p.get('closed_at') and
                datetime.fromisoformat(p['closed_at'].replace('Z', '+00:00')) > cutoff_date
            ]

            # Sum realized P&L
            monthly_pnl = sum(p.get('realized_pnl', 0) or 0 for p in recent_closed)

            # Calculate percentage (simplified)
            total_cost = sum(p.get('total_cost', 0) or 0 for p in recent_closed)
            monthly_pnl_pct = (monthly_pnl / total_cost * 100) if total_cost > 0 else 0.0

            logger.debug(f"Monthly P&L calculated: {portfolio_id}")
            return {
                'monthly_pnl': monthly_pnl,
                'monthly_pnl_pct': monthly_pnl_pct
            }

        except Exception as e:
            logger.error(f"❌ Calculate monthly P&L error: {e}", exc_info=True)
            return {'monthly_pnl': 0.0, 'monthly_pnl_pct': 0.0}

    async def update_portfolio_metrics(
        self,
        portfolio_id: int
    ) -> bool:
        """
        Update all portfolio metrics (summary, daily/monthly P&L)

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Success status
        """
        try:
            # Update summary (total_value, total_cost, total_pnl, etc.)
            await self.dm.update_portfolio_summary(portfolio_id)

            # Calculate daily P&L
            daily = await self.calculate_daily_pnl(portfolio_id)

            # Calculate monthly P&L
            monthly = await self.calculate_monthly_pnl(portfolio_id)

            # Update portfolio with new P&L values
            await self.dm.update_portfolio(
                portfolio_id=portfolio_id,
                daily_pnl=daily['daily_pnl'],
                daily_pnl_pct=daily['daily_pnl_pct'],
                monthly_pnl=monthly['monthly_pnl'],
                monthly_pnl_pct=monthly['monthly_pnl_pct']
            )

            logger.info(f"✅ Portfolio metrics updated: {portfolio_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Update portfolio metrics error: {e}", exc_info=True)
            return False

    async def add_position_with_entries(
        self,
        portfolio_id: int,
        symbol: str,
        side: str,
        entries: List[Dict[str, Any]],
        exit_targets: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """
        Create position with multiple entries and exit targets

        Args:
            portfolio_id: Portfolio ID
            symbol: Trading symbol
            side: LONG or SHORT
            entries: List of entry dicts with quantity, price
            exit_targets: Optional list of TP/SL dicts

        Returns:
            Position ID if successful
        """
        try:
            if not entries:
                logger.error("No entries provided")
                return None

            # Calculate totals
            total_qty = sum(e['quantity'] for e in entries)
            total_cost = sum(e['quantity'] * e['price'] for e in entries)
            avg_price = total_cost / total_qty if total_qty > 0 else 0.0

            # Create position
            position_id = await self.dm.create_position(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=total_qty,
                entry_price=avg_price,
                side=side,
                position_type='SPOT',
                source='manual'
            )

            if not position_id:
                logger.error("Failed to create position")
                return None

            # Add entries
            for i, entry in enumerate(entries, 1):
                await self.dm.add_position_entry(
                    position_id=position_id,
                    entry_number=i,
                    quantity=entry['quantity'],
                    entry_price=entry['price'],
                    status=entry.get('status', 'filled'),
                    source=entry.get('source', 'manual')
                )

            # Add exit targets
            if exit_targets:
                for target in exit_targets:
                    await self.dm.add_exit_target(
                        position_id=position_id,
                        target_type=target['type'],
                        target_number=target['number'],
                        target_price=target['price'],
                        quantity_percentage=target['percentage']
                    )

            # Update portfolio summary
            await self.dm.update_portfolio_summary(portfolio_id)

            logger.info(f"✅ Position created with {len(entries)} entries: {symbol}")
            return position_id

        except Exception as e:
            logger.error(f"❌ Add position with entries error: {e}", exc_info=True)
            return None
