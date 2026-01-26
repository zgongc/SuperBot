#!/usr/bin/env python3
"""
components/optimizer/metrics.py
SuperBot - Optimizer Metric Calculator
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Calculates comprehensive metrics from backtest results.
Supports 30+ metrics: Sharpe ratio, Profit Factor, SQN, expectancy, etc.

Features:
- 30+ comprehensive metrics (Return, Risk, Risk-Adjusted, Trade, Profit, etc.)
- BacktestMetrics dataclass (all metrics in a single object)
- MetricsCalculator class (calculation logic)
- Van Tharp SQN, Kelly Criterion support
- Annualized Sharpe/Sortino/Calmar ratios

Usage:
    from components.optimizer.v2.metrics import MetricsCalculator

    calculator = MetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(
        trades=trades,
        initial_balance=10000,
        backtest_days=365
    )

    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"SQN: {metrics.sqn:.2f}")

Dependencies:
    - python>=3.10
    - numpy>=1.24.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from datetime import datetime, timedelta


@dataclass
class BacktestMetrics:
    """All metrics for the backtest result"""

    # ========== RETURN METRICS ==========
    total_return: float              # Total return %
    annualized_return: float         # Annualized return %
    cagr: float                      # Compound annual growth rate

    # ========== RISK METRICS ==========
    max_drawdown: float              # Maximum drawdown percentage
    max_drawdown_duration: int       # Maximum drawdown duration (days)
    volatility: float                # Return volatility (standard deviation)
    downside_deviation: float        # Downside volatility

    # ========== RISK-ADJUSTED METRICS ==========
    sharpe_ratio: float              # (Return - RFR) / Volatility
    sortino_ratio: float             # (Return - RFR) / Downside deviation
    calmar_ratio: float              # Return / Max drawdown
    omega_ratio: float               # Profit/loss probability weighted ratio

    # ========== TRADE METRICS ==========
    total_trades: int                # Total number of trades
    winning_trades: int              # Number of winning trades
    losing_trades: int               # Number of losing trades
    win_rate: float                  # Winning rate %

    # ========== PROFIT METRICS ==========
    gross_profit: float              # Gross profit (from winning transactions)
    gross_loss: float                # Gross loss (from losing transactions)
    net_profit: float                # Net profit (gross profit - gross loss)
    profit_factor: float             # Gross profit / Gross loss

    # ========== AVERAGE METRICS ==========
    avg_trade: float                 # Average P&L per trade
    avg_win: float                   # Average winning transaction
    avg_loss: float                  # Average loss operation
    avg_win_loss_ratio: float        # Average win / Average loss

    # ========== STREAK METRICS ==========
    max_consecutive_wins: int        # Maximum consecutive wins
    max_consecutive_losses: int      # Maximum consecutive loss

    # ========== POSITION METRICS ==========
    avg_holding_time: float          # Average holding time (hours)
    max_holding_time: float          # Maximum holding time (hours)
    min_holding_time: float          # Minimum holding time (hours)

    # ========== EXPECTANCY METRICS ==========
    expectancy: float                # Expected value per operation
    expectancy_ratio: float          # Expectancy / Average loss

    # ========== SYSTEM QUALITY ==========
    sqn: float                       # System Quality Number (Van Tharp)
    kelly_criterion: float           # Optimal position size percentage


class MetricsCalculator:
    """Calculates comprehensive metrics from backtesting operations"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the metric calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        trades: List[dict],
        initial_balance: float,
        backtest_days: int
    ) -> BacktestMetrics:
        """
        Calculate comprehensive metrics from the operation list.

        Args:
            trades: List of Trade objects (from BacktestEngine)
            initial_balance: Initial balance
            backtest_days: Backtest duration (in days)

        Returns:
            BacktestMetrics: All metrics
        """

        if not trades:
            return self._empty_metrics()

        # Extract trade data
        pnls = [getattr(t, 'net_pnl_usd', 0) for t in trades]
        returns = [getattr(t, 'net_pnl_pct', 0) for t in trades]

        # Calculate holding times (in hours)
        holding_times = []
        for t in trades:
            exit_time = getattr(t, 'exit_time', None)
            entry_time = getattr(t, 'entry_time', None)
            if exit_time and entry_time:
                duration = (exit_time - entry_time).total_seconds() / 3600
                holding_times.append(duration)

        # Winning/losing trades
        winning_trades = [t for t in trades if getattr(t, 'net_pnl_usd', 0) > 0]
        losing_trades = [t for t in trades if getattr(t, 'net_pnl_usd', 0) <= 0]

        # Calculate metrics
        metrics = BacktestMetrics(
            # Return metrics
            total_return=self._total_return(pnls, initial_balance),
            annualized_return=self._annualized_return(pnls, initial_balance, backtest_days),
            cagr=self._cagr(pnls, initial_balance, backtest_days),

            # Risk metrics
            max_drawdown=self._max_drawdown(trades, initial_balance),
            max_drawdown_duration=self._max_dd_duration(trades),
            volatility=np.std(returns) if returns else 0,
            downside_deviation=self._downside_deviation(returns),

            # Risk-adjusted metrics
            sharpe_ratio=self._sharpe_ratio(returns, backtest_days),
            sortino_ratio=self._sortino_ratio(returns, backtest_days),
            calmar_ratio=self._calmar_ratio(pnls, initial_balance, backtest_days),
            omega_ratio=self._omega_ratio(returns),

            # Trade metrics
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(trades) * 100) if trades else 0,

            # Profit metrics
            gross_profit=sum(getattr(t, 'net_pnl_usd', 0) for t in winning_trades),
            gross_loss=abs(sum(getattr(t, 'net_pnl_usd', 0) for t in losing_trades)),
            net_profit=sum(pnls),
            profit_factor=self._profit_factor(winning_trades, losing_trades),

            # Average metrics
            avg_trade=np.mean(pnls) if pnls else 0,
            avg_win=np.mean([getattr(t, 'net_pnl_usd', 0) for t in winning_trades]) if winning_trades else 0,
            avg_loss=np.mean([getattr(t, 'net_pnl_usd', 0) for t in losing_trades]) if losing_trades else 0,
            avg_win_loss_ratio=self._avg_win_loss_ratio(winning_trades, losing_trades),

            # Streak metrics
            max_consecutive_wins=self._max_consecutive_wins(trades),
            max_consecutive_losses=self._max_consecutive_losses(trades),

            # Position metrics
            avg_holding_time=np.mean(holding_times) if holding_times else 0,
            max_holding_time=max(holding_times) if holding_times else 0,
            min_holding_time=min(holding_times) if holding_times else 0,

            # Expectancy metrics
            expectancy=self._expectancy(winning_trades, losing_trades, len(trades)),
            expectancy_ratio=self._expectancy_ratio(winning_trades, losing_trades, len(trades)),

            # System quality
            sqn=self._sqn(pnls),
            kelly_criterion=self._kelly_criterion(winning_trades, losing_trades, len(trades)),
        )

        return metrics

    # ========================================================================
    # RETURN METRICS
    # ========================================================================

    def _total_return(self, pnls: List[float], initial_balance: float) -> float:
        """Total return %"""
        if not pnls or initial_balance == 0:
            return 0.0
        return (sum(pnls) / initial_balance) * 100

    def _annualized_return(self, pnls: List[float], initial_balance: float, days: int) -> float:
        """Annualized return %"""
        if not pnls or days == 0 or initial_balance == 0:
            return 0.0

        total_return = sum(pnls) / initial_balance
        years = days / 365

        if years == 0:
            return 0.0

        return (total_return / years) * 100

    def _cagr(self, pnls: List[float], initial_balance: float, days: int) -> float:
        """Compound Annual Growth Rate"""
        if not pnls or days == 0 or initial_balance == 0:
            return 0.0

        final_balance = initial_balance + sum(pnls)
        years = days / 365

        if years == 0 or final_balance <= 0:
            return 0.0

        cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
        return cagr

    # ========================================================================
    # RISK METRICS
    # ========================================================================

    def _max_drawdown(self, trades: List[dict], initial_balance: float) -> float:
        """Maximum drawdown %"""
        if not trades:
            return 0.0

        # Calculate equity curve
        equity = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for trade in trades:
            equity += getattr(trade, 'net_pnl_usd', 0)
            if equity > peak:
                peak = equity

            dd = ((peak - equity) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _max_dd_duration(self, trades: List[dict]) -> int:
        """Maximum drawdown duration in days"""
        # Simplified: return 0 for now (requires equity curve timestamps)
        return 0

    def _downside_deviation(self, returns: List[float]) -> float:
        """Downside deviation (only negative returns)"""
        if not returns:
            return 0.0

        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0.0

        return np.std(negative_returns)

    # ========================================================================
    # RISK-ADJUSTED METRICS
    # ========================================================================

    def _sharpe_ratio(self, returns: List[float], days: int) -> float:
        """
        Sharpe Ratio = (Mean return - Risk-free rate) / Std dev of returns

        Annualized for fair comparison
        """
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rfr = (1 + self.risk_free_rate) ** (1/365) - 1

        # Sharpe ratio
        sharpe = (mean_return - daily_rfr) / std_return

        # Annualize (sqrt of trading days per year)
        # Crypto: 365 days, Stocks: 252 days
        trading_days_per_year = 365
        sharpe_annual = sharpe * np.sqrt(trading_days_per_year)

        return sharpe_annual

    def _sortino_ratio(self, returns: List[float], days: int) -> float:
        """Sortino ratio (downside risk-adjusted)"""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        downside_dev = self._downside_deviation(returns)

        if downside_dev == 0:
            return 0.0

        daily_rfr = (1 + self.risk_free_rate) ** (1/365) - 1
        sortino = (mean_return - daily_rfr) / downside_dev

        # Annualize
        trading_days_per_year = 365
        sortino_annual = sortino * np.sqrt(trading_days_per_year)

        return sortino_annual

    def _calmar_ratio(self, pnls: List[float], initial_balance: float, days: int) -> float:
        """Calmar ratio = Annualized return / Max drawdown"""
        if not pnls or days == 0:
            return 0.0

        ann_return = self._annualized_return(pnls, initial_balance, days)
        max_dd = self._max_drawdown([{'net_pnl_usd': p} for p in pnls], initial_balance)

        if max_dd == 0:
            return 0.0

        return ann_return / max_dd

    def _omega_ratio(self, returns: List[float], threshold: float = 0) -> float:
        """Omega ratio (probability-weighted gains vs losses)"""
        if not returns:
            return 0.0

        gains = sum(r - threshold for r in returns if r > threshold)
        losses = sum(threshold - r for r in returns if r < threshold)

        if losses == 0:
            return float('inf') if gains > 0 else 0.0

        return gains / losses

    # ========================================================================
    # PROFIT METRICS
    # ========================================================================

    def _profit_factor(self, winners: List[dict], losers: List[dict]) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss

        > 1.0 = Profitable
        > 1.5 = Good
        > 2.0 = Excellent
        """
        if not losers:
            return float('inf') if winners else 0.0

        gross_profit = sum(getattr(t, 'net_pnl_usd', 0) for t in winners)
        gross_loss = abs(sum(getattr(t, 'net_pnl_usd', 0) for t in losers))

        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def _avg_win_loss_ratio(self, winners: List[dict], losers: List[dict]) -> float:
        """Average win / Average loss ratio"""
        if not winners or not losers:
            return 0.0

        avg_win = np.mean([getattr(t, 'net_pnl_usd', 0) for t in winners])
        avg_loss = abs(np.mean([getattr(t, 'net_pnl_usd', 0) for t in losers]))

        return avg_win / avg_loss if avg_loss > 0 else 0.0

    # ========================================================================
    # EXPECTANCY METRICS
    # ========================================================================

    def _expectancy(self, winners: List[dict], losers: List[dict], total: int) -> float:
        """
        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)

        Expected profit per trade
        """
        if total == 0:
            return 0.0

        win_rate = len(winners) / total
        loss_rate = len(losers) / total

        avg_win = np.mean([getattr(t, 'net_pnl_usd', 0) for t in winners]) if winners else 0
        avg_loss = abs(np.mean([getattr(t, 'net_pnl_usd', 0) for t in losers])) if losers else 0

        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def _expectancy_ratio(self, winners: List[dict], losers: List[dict], total: int) -> float:
        """Expectancy / Average loss"""
        expectancy = self._expectancy(winners, losers, total)
        avg_loss = abs(np.mean([getattr(t, 'net_pnl_usd', 0) for t in losers])) if losers else 0

        return expectancy / avg_loss if avg_loss > 0 else 0.0

    # ========================================================================
    # SYSTEM QUALITY
    # ========================================================================

    def _sqn(self, pnls: List[float]) -> float:
        """
        System Quality Number (Van Tharp)

        SQN = sqrt(N) × (Mean / Std dev)

        1.6-1.9 = Below average
        2.0-2.4 = Average
        2.5-2.9 = Good
        3.0-5.0 = Excellent
        > 5.0 = Superb
        """
        if not pnls or len(pnls) < 2:
            return 0.0

        mean = np.mean(pnls)
        std = np.std(pnls)

        if std == 0:
            return 0.0

        return np.sqrt(len(pnls)) * (mean / std)

    def _kelly_criterion(self, winners: List[dict], losers: List[dict], total: int) -> float:
        """
        Kelly Criterion = W - (1-W)/R

        W = Win rate
        R = Avg win / Avg loss

        Returns optimal position size %
        """
        if total == 0 or not winners or not losers:
            return 0.0

        win_rate = len(winners) / total
        avg_win = np.mean([getattr(t, 'net_pnl_usd', 0) for t in winners])
        avg_loss = abs(np.mean([getattr(t, 'net_pnl_usd', 0) for t in losers]))

        if avg_loss == 0:
            return 0.0

        r = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / r)

        # Cap at 25% (full Kelly is too aggressive)
        return min(kelly * 100, 25.0)

    # ========================================================================
    # STREAK METRICS
    # ========================================================================

    def _max_consecutive_wins(self, trades: List[dict]) -> int:
        """Maximum consecutive winning trades"""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if getattr(trade, 'net_pnl_usd', 0) > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _max_consecutive_losses(self, trades: List[dict]) -> int:
        """Maximum consecutive losing trades"""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if getattr(trade, 'net_pnl_usd', 0) <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics (all zeros)"""
        return BacktestMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            volatility=0.0,
            downside_deviation=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            profit_factor=0.0,
            avg_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_win_loss_ratio=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            avg_holding_time=0.0,
            max_holding_time=0.0,
            min_holding_time=0.0,
            expectancy=0.0,
            expectancy_ratio=0.0,
            sqn=0.0,
            kelly_criterion=0.0,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BacktestMetrics',
    'MetricsCalculator',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta

    print("=" * 60)
    print("🧪 MetricsCalculator Test")
    print("=" * 60)

    # Test 1: Empty trade list
    print("\n📊 Test 1: Empty trade list")
    calculator = MetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(
        trades=[],
        initial_balance=10000,
        backtest_days=365
    )
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Total Trades: {metrics.total_trades}")
    print("   ✅ Test successful")

    # Test 2: Example trade list (profitable trades)
    print("\n📊 Test 2: Example trade list (3 winners, 2 losers)")
    sample_trades = [
        {'net_pnl_usd': 100, 'return_pct': 1.0, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=2)},
        {'net_pnl_usd': 150, 'return_pct': 1.5, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=3)},
        {'net_pnl_usd': -50, 'return_pct': -0.5, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=1)},
        {'net_pnl_usd': 200, 'return_pct': 2.0, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=4)},
        {'net_pnl_usd': -30, 'return_pct': -0.3, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=1)},
    ]

    metrics = calculator.calculate_all_metrics(
        trades=sample_trades,
        initial_balance=10000,
        backtest_days=365
    )

    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Winning Trades: {metrics.winning_trades}")
    print(f"   Losing Trades: {metrics.losing_trades}")
    print(f"   Win Rate: {metrics.win_rate:.1f}%")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   SQN: {metrics.sqn:.2f}")
    print(f"   Net Profit: ${metrics.net_profit:.2f}")
    print(f"   Total Return: {metrics.total_return:.2f}%")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"   Avg Holding Time: {metrics.avg_holding_time:.1f} hours")
    print("   ✅ Test successful")

    # Test 3: Metric value checks
    print("\n📊 Test 3: Metric value checks")
    assert metrics.total_trades == 5, "Total number of trades is incorrect"
    assert metrics.winning_trades == 3, "The number of winning trades is incorrect"
    assert metrics.losing_trades == 2, "The number of losing trades is incorrect"
    assert metrics.win_rate == 60.0, "Win rate is incorrect"
    assert metrics.net_profit == 370, "Net profit is incorrect"
    assert metrics.gross_profit == 450, "Gross profit is incorrect"
    assert metrics.gross_loss == 80, "Gross loss is incorrect"
    print("   ✅ All assertions passed")

    print("\n✅ All tests completed!")
    print("=" * 60)
