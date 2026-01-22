#!/usr/bin/env python3
"""
Portfolio Calculator - Pure Calculation Functions
Stateless functions for portfolio performance metrics

3commas-style metrics:
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk-adjusted returns
- Max Drawdown: Largest peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / Gross loss
- Average Win/Loss: Mean profit/loss per trade
- Risk/Reward Ratio: Average win / Average loss
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math


class PortfolioCalculator:
    """Pure calculation functions for portfolio metrics"""

    @staticmethod
    def calculate_total_value(positions: List[Dict[str, Any]]) -> float:
        """
        Calculate total portfolio value

        Args:
            positions: List of position dicts with 'market_value' or 'quantity' * 'current_price'

        Returns:
            Total portfolio value
        """
        total = 0.0
        for pos in positions:
            if 'market_value' in pos and pos['market_value']:
                total += pos['market_value']
            elif 'quantity' in pos and 'current_price' in pos:
                total += pos['quantity'] * pos['current_price']

        return total

    @staticmethod
    def calculate_total_cost(positions: List[Dict[str, Any]]) -> float:
        """
        Calculate total cost basis

        Args:
            positions: List of position dicts with 'total_cost' or 'quantity' * 'entry_price'

        Returns:
            Total cost basis
        """
        total = 0.0
        for pos in positions:
            if 'total_cost' in pos and pos['total_cost']:
                total += pos['total_cost']
            elif 'quantity' in pos and 'entry_price' in pos:
                total += pos['quantity'] * pos['entry_price']

        return total

    @staticmethod
    def calculate_pnl(
        positions: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Calculate total P&L and percentage

        Args:
            positions: List of position dicts

        Returns:
            (total_pnl, pnl_percentage)
        """
        total_value = PortfolioCalculator.calculate_total_value(positions)
        total_cost = PortfolioCalculator.calculate_total_cost(positions)

        total_pnl = total_value - total_cost
        pnl_percentage = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        return total_pnl, pnl_percentage

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted returns)

        Formula: (Mean Return - Risk Free Rate) / StdDev of Returns

        Args:
            returns: List of daily returns (as decimals, e.g., 0.02 for 2%)
            risk_free_rate: Annual risk-free rate (default 0%)
            periods_per_year: Number of periods in a year (365 for daily)

        Returns:
            Annualized Sharpe Ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        # Calculate mean return
        mean_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Daily risk-free rate
        daily_rfr = risk_free_rate / periods_per_year

        # Sharpe ratio
        sharpe = (mean_return - daily_rfr) / std_dev

        # Annualize
        annualized_sharpe = sharpe * math.sqrt(periods_per_year)

        return annualized_sharpe

    @staticmethod
    def calculate_sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted returns)

        Similar to Sharpe but only considers downside volatility

        Args:
            returns: List of daily returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sortino Ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        # Calculate mean return
        mean_return = sum(returns) / len(returns)

        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_variance = sum(r ** 2 for r in downside_returns) / len(returns)
        downside_dev = math.sqrt(downside_variance)

        if downside_dev == 0:
            return 0.0

        # Daily risk-free rate
        daily_rfr = risk_free_rate / periods_per_year

        # Sortino ratio
        sortino = (mean_return - daily_rfr) / downside_dev

        # Annualize
        annualized_sortino = sortino * math.sqrt(periods_per_year)

        return annualized_sortino

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: List[Tuple[datetime, float]]
    ) -> Tuple[float, float]:
        """
        Calculate Maximum Drawdown

        Args:
            equity_curve: List of (timestamp, portfolio_value) tuples

        Returns:
            (max_drawdown_percentage, max_drawdown_value)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0.0

        max_drawdown_pct = 0.0
        max_drawdown_val = 0.0
        peak_value = equity_curve[0][1]

        for timestamp, value in equity_curve:
            # Update peak
            if value > peak_value:
                peak_value = value

            # Calculate drawdown from peak
            if peak_value > 0:
                drawdown_pct = (peak_value - value) / peak_value * 100
                drawdown_val = peak_value - value

                if drawdown_pct > max_drawdown_pct:
                    max_drawdown_pct = drawdown_pct
                    max_drawdown_val = drawdown_val

        return max_drawdown_pct, max_drawdown_val

    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate Win Rate

        Args:
            trades: List of closed trade dicts with 'pnl' or 'realized_pnl'

        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0

        winning_trades = 0

        for trade in trades:
            pnl = trade.get('pnl')
            if pnl is None:
                pnl = trade.get('realized_pnl', 0)
            if pnl is not None and pnl > 0:
                winning_trades += 1

        win_rate = (winning_trades / len(trades)) * 100
        return win_rate

    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate Profit Factor

        Formula: Gross Profit / Gross Loss

        Args:
            trades: List of closed trade dicts with 'pnl' or 'realized_pnl'

        Returns:
            Profit factor (>1 is profitable, <1 is losing)
        """
        if not trades:
            return 0.0

        gross_profit = 0.0
        gross_loss = 0.0

        for trade in trades:
            pnl = trade.get('pnl')
            if pnl is None:
                pnl = trade.get('realized_pnl', 0)
            if pnl is not None:
                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

        if gross_loss == 0:
            # Return None (null in JSON) when there are no losses
            # This avoids JSON serialization errors with Infinity
            return None if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return profit_factor

    @staticmethod
    def calculate_average_win_loss(
        trades: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Calculate Average Win and Average Loss

        Args:
            trades: List of closed trade dicts with 'pnl' or 'realized_pnl'

        Returns:
            (average_win, average_loss)
        """
        if not trades:
            return 0.0, 0.0

        wins = []
        losses = []

        for trade in trades:
            pnl = trade.get('pnl')
            if pnl is None:
                pnl = trade.get('realized_pnl', 0)
            if pnl is not None:
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return avg_win, avg_loss

    @staticmethod
    def calculate_risk_reward_ratio(
        trades: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Risk/Reward Ratio

        Formula: Average Win / Average Loss

        Args:
            trades: List of closed trade dicts

        Returns:
            Risk/Reward ratio (>1 means avg win > avg loss)
        """
        avg_win, avg_loss = PortfolioCalculator.calculate_average_win_loss(trades)

        if avg_loss == 0:
            # Return None (null in JSON) when there are no losses
            # This avoids JSON serialization errors with Infinity
            return None if avg_win > 0 else 0.0

        return avg_win / avg_loss

    @staticmethod
    def calculate_daily_returns(
        equity_curve: List[Tuple[datetime, float]]
    ) -> List[float]:
        """
        Calculate daily returns from equity curve

        Args:
            equity_curve: List of (timestamp, portfolio_value) tuples (sorted by time)

        Returns:
            List of daily returns as decimals (0.02 = 2% gain)
        """
        if not equity_curve or len(equity_curve) < 2:
            return []

        returns = []

        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i - 1][1]
            curr_value = equity_curve[i][1]

            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        return returns

    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        """
        Extract base asset from symbol string

        Examples:
            "ETHUSDT" -> "ETH"
            "ETH/USDT" -> "ETH"
            "BTCUSDC" -> "BTC"
            "BTC/USD" -> "BTC"
        """
        if not symbol:
            return 'UNKNOWN'

        # Handle slash format (ETH/USDT)
        if '/' in symbol:
            return symbol.split('/')[0].upper()

        # Handle common quote assets (longest first to avoid partial matches)
        quote_assets = ['USDT', 'USDC', 'BUSD', 'TUSD', 'FDUSD', 'USD', 'BTC', 'ETH', 'BNB']
        symbol_upper = symbol.upper()

        for quote in quote_assets:
            if symbol_upper.endswith(quote):
                base = symbol_upper[:-len(quote)]
                if base:  # Make sure we have something left
                    return base

        # Fallback: return the symbol itself
        return symbol_upper

    @staticmethod
    def calculate_token_allocation(
        positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate token allocation breakdown

        Args:
            positions: List of position dicts with 'base_asset' or 'symbol' and 'market_value'

        Returns:
            List of {token, value, percentage} sorted by value descending
        """
        if not positions:
            return []

        # Aggregate by token
        token_totals = {}
        total_value = 0.0

        for pos in positions:
            # Try base_asset first, then parse from symbol
            token = pos.get('base_asset')
            if not token:
                symbol = pos.get('symbol', '')
                token = PortfolioCalculator._extract_base_asset(symbol)

            value = pos.get('market_value', 0) or (pos.get('quantity', 0) * pos.get('current_price', 0))

            if token not in token_totals:
                token_totals[token] = 0.0

            token_totals[token] += value
            total_value += value

        # Calculate percentages
        allocation = []

        for token, value in token_totals.items():
            percentage = (value / total_value * 100) if total_value > 0 else 0.0
            allocation.append({
                'token': token,
                'value': value,
                'percentage': percentage
            })

        # Sort by value descending
        allocation.sort(key=lambda x: x['value'], reverse=True)

        return allocation

    @staticmethod
    def calculate_position_metrics(
        entries: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a multi-entry position

        Args:
            entries: List of position entries with 'quantity', 'entry_price', 'cost'
            current_price: Current market price

        Returns:
            Dict with total_qty, avg_entry_price, total_cost, market_value, unrealized_pnl, pnl_pct
        """
        if not entries:
            return {
                'total_quantity': 0.0,
                'average_entry_price': 0.0,
                'total_cost': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'pnl_percentage': 0.0
            }

        # Only count filled entries
        filled_entries = [e for e in entries if e.get('status') == 'filled']

        if not filled_entries:
            return {
                'total_quantity': 0.0,
                'average_entry_price': 0.0,
                'total_cost': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'pnl_percentage': 0.0
            }

        total_qty = sum(e['quantity'] for e in filled_entries)
        total_cost = sum(e['cost'] for e in filled_entries)

        avg_price = total_cost / total_qty if total_qty > 0 else 0.0
        market_value = total_qty * current_price
        unrealized_pnl = market_value - total_cost
        pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0

        return {
            'total_quantity': total_qty,
            'average_entry_price': avg_price,
            'total_cost': total_cost,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            'pnl_percentage': pnl_pct
        }

    @staticmethod
    def calculate_comprehensive_stats(
        positions: List[Dict[str, Any]],
        closed_trades: List[Dict[str, Any]],
        equity_curve: Optional[List[Tuple[datetime, float]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio statistics

        Args:
            positions: List of current positions
            closed_trades: List of closed trades
            equity_curve: Optional equity curve for advanced metrics

        Returns:
            Dict with all portfolio statistics
        """
        # Basic metrics
        total_value = PortfolioCalculator.calculate_total_value(positions)
        total_cost = PortfolioCalculator.calculate_total_cost(positions)
        total_pnl, pnl_pct = PortfolioCalculator.calculate_pnl(positions)

        # Trade metrics
        win_rate = PortfolioCalculator.calculate_win_rate(closed_trades)
        profit_factor = PortfolioCalculator.calculate_profit_factor(closed_trades)
        avg_win, avg_loss = PortfolioCalculator.calculate_average_win_loss(closed_trades)
        risk_reward = PortfolioCalculator.calculate_risk_reward_ratio(closed_trades)

        # Token allocation
        allocation = PortfolioCalculator.calculate_token_allocation(positions)

        stats = {
            # Portfolio Overview
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'pnl_percentage': pnl_pct,
            'position_count': len([p for p in positions if p.get('is_open', True)]),

            # Trade Performance
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'risk_reward_ratio': risk_reward,

            # Token Allocation
            'token_allocation': allocation,

            # Risk Metrics (if equity curve provided)
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_value': 0.0
        }

        # Advanced metrics (requires equity curve)
        if equity_curve and len(equity_curve) >= 2:
            returns = PortfolioCalculator.calculate_daily_returns(equity_curve)

            if returns:
                stats['sharpe_ratio'] = PortfolioCalculator.calculate_sharpe_ratio(returns)
                stats['sortino_ratio'] = PortfolioCalculator.calculate_sortino_ratio(returns)

            max_dd_pct, max_dd_val = PortfolioCalculator.calculate_max_drawdown(equity_curve)
            stats['max_drawdown'] = max_dd_pct
            stats['max_drawdown_value'] = max_dd_val

        return stats
