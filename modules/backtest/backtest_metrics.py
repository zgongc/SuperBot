#!/usr/bin/env python3
"""
modules/backtest/backtest_metrics.py
SuperBot - Backtest Performance Metrics Calculator
Yazar: SuperBot Team
Tarih: 2025-11-16
Versiyon: 3.0.0

Backtest sonuçlarının performans metriklerini hesaplar.

Özellikler:
- Comprehensive metrics calculation
- BacktestMetrics dataclass support
- Sharpe, Sortino, Calmar ratios
- Drawdown analysis
- Trade statistics

Kullanım:
    from modules.backtest.backtest_metrics import calculate_metrics

    metrics = calculate_metrics(trades, config)

Bağımlılıklar:
    - python>=3.10
    - numpy>=1.24.0
    - pandas>=2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from modules.backtest.backtest_types import BacktestMetrics, Trade, BacktestConfig

if TYPE_CHECKING:
    pass


def calculate_metrics(
    trades: List[Trade],
    config: BacktestConfig
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics

    Args:
        trades: List of Trade objects
        config: Backtest config

    Returns:
        BacktestMetrics: Comprehensive metrics
    """
    if not trades:
        return _empty_metrics()

    # Calculate balances
    initial_balance = config.initial_balance
    final_balance = initial_balance + sum(t.net_pnl_usd for t in trades)

    # Delegate to detailed calculation
    return _calculate_detailed_metrics(trades, initial_balance, final_balance)


def calculate_performance_metrics(
    trades: List[Dict[str, Any]],
    initial_balance: float,
    final_balance: float,
    equity_curve: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    DEPRECATED: Use calculate_metrics() instead

    Legacy function for backward compatibility
    """
    """
    Calculate comprehensive performance metrics

    Args:
        trades: List of completed trades
        initial_balance: Starting balance
        final_balance: Ending balance
        equity_curve: Optional equity curve data

    Returns:
        Dict with performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'total_profit': 0.0,
            'total_profit_pct': 0.0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss': 0.0,
            'avg_loss_pct': 0.0,
            'win_loss_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'total_costs': 0.0,
        }

    # Basic stats
    total_trades = len(trades)
    total_profit = final_balance - initial_balance
    total_profit_pct = (total_profit / initial_balance) * 100

    # Win/Loss stats
    winners = [t for t in trades if t.get('net_pnl_usd', 0) > 0]
    losers = [t for t in trades if t.get('net_pnl_usd', 0) <= 0]

    num_winners = len(winners)
    num_losers = len(losers)
    win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0.0

    # Average win/loss
    avg_win = np.mean([t['net_pnl_usd'] for t in winners]) if winners else 0.0
    avg_win_pct = np.mean([t['net_pnl_pct'] for t in winners]) if winners else 0.0
    avg_loss = np.mean([t['net_pnl_usd'] for t in losers]) if losers else 0.0
    avg_loss_pct = np.mean([t['net_pnl_pct'] for t in losers]) if losers else 0.0

    # Win/Loss ratio
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # Profit factor
    total_wins = sum([t['net_pnl_usd'] for t in winners])
    total_losses = abs(sum([t['net_pnl_usd'] for t in losers]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    # Trading costs
    total_commission = sum([t.get('commission', 0) for t in trades])
    total_slippage = sum([t.get('slippage', 0) for t in trades])
    total_costs = total_commission + total_slippage

    # Drawdown (from equity curve if available)
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    if equity_curve and len(equity_curve) > 0:
        equity_values = [e.get('balance', initial_balance) for e in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdown = running_max - equity_values
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = (max_drawdown / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0.0

    # Sharpe Ratio (simplified - using trade returns)
    if len(trades) > 1:
        returns = [t.get('net_pnl_pct', 0) / 100 for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(len(trades)) if std_return > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    return {
        # Basic
        'total_trades': total_trades,
        'total_profit': total_profit,
        'total_profit_pct': total_profit_pct,

        # Win/Loss
        'winners': num_winners,
        'losers': num_losers,
        'win_rate': win_rate,

        # Averages
        'avg_win': avg_win,
        'avg_win_pct': avg_win_pct,
        'avg_loss': avg_loss,
        'avg_loss_pct': avg_loss_pct,
        'win_loss_ratio': win_loss_ratio,

        # Risk metrics
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,

        # Costs
        'total_commission': total_commission,
        'total_slippage': total_slippage,
        'total_costs': total_costs,
    }


def print_performance_summary(
    metrics: Dict[str, Any],
    initial_balance: float,
    final_balance: float,
    logger
):
    """
    Print performance summary to logger

    Args:
        metrics: Performance metrics dict
        initial_balance: Starting balance
        final_balance: Ending balance
        logger: Logger instance
    """
    logger.info("\n" + "━" * 60)
    logger.info("📊 PERFORMANCE SUMMARY")
    logger.info("━" * 60)

    # Returns
    logger.info("\n💰 Returns:")
    logger.info(f"   Initial Capital:        ${initial_balance:,.2f}")
    logger.info(f"   Final Balance:          ${final_balance:,.2f}")

    total_profit = metrics['total_profit']
    total_profit_pct = metrics['total_profit_pct']
    logger.info(f"   Total Profit:           ${total_profit:,.2f} ({total_profit_pct:+.1f}%)")

    if metrics['total_costs'] > 0:
        logger.info(f"   Total Commission:       ${metrics['total_commission']:,.2f}")
        logger.info(f"   Total Slippage:         ${metrics['total_slippage']:,.2f}")
        logger.info(f"   Total Trading Costs:    ${metrics['total_costs']:,.2f}")

    # Trade Statistics
    logger.info("\n📈 Trade Statistics:")
    logger.info(f"   Total Trades:           {metrics['total_trades']}")
    logger.info(f"   Winners:                {metrics['winners']} ({metrics['win_rate']:.1f}%)")
    logger.info(f"   Losers:                 {metrics['losers']} ({100 - metrics['win_rate']:.1f}%)")

    logger.info("")
    avg_win = metrics['avg_win']
    avg_win_pct = metrics['avg_win_pct']
    avg_loss = metrics['avg_loss']
    avg_loss_pct = metrics['avg_loss_pct']

    logger.info(f"   Avg Win:                ${avg_win:,.2f} ({avg_win_pct:+.1f}%)")
    logger.info(f"   Avg Loss:               ${avg_loss:,.2f} ({avg_loss_pct:+.1f}%)")

    if metrics['win_loss_ratio'] > 0:
        logger.info(f"   Win/Loss Ratio:         {metrics['win_loss_ratio']:.2f}")

    # Risk Metrics
    logger.info("\n📊 Risk Metrics:")
    if metrics['profit_factor'] > 0:
        logger.info(f"   Profit Factor:          {metrics['profit_factor']:.2f}")

    max_dd_pct = metrics['max_drawdown_pct']
    max_dd = metrics['max_drawdown']
    logger.info(f"   Max Drawdown:           -{max_dd_pct:.1f}% (${max_dd:,.0f})")

    if metrics['sharpe_ratio'] != 0:
        logger.info(f"   Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    else:
        logger.info(f"   Sharpe Ratio:           N/A")


# ============================================================================
# HELPER FUNCTIONS (NEW - V3)
# ============================================================================

def _empty_metrics() -> BacktestMetrics:
    """Empty metrics for no trades"""
    return BacktestMetrics(
        total_return_usd=0.0,
        total_return_pct=0.0,
        total_trades=0,
        winners=0,
        losers=0,
        win_rate=0.0,
        avg_win_usd=0.0,
        avg_win_pct=0.0,
        avg_loss_usd=0.0,
        avg_loss_pct=0.0,
        largest_win_usd=0.0,
        largest_loss_usd=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown_usd=0.0,
        max_drawdown_pct=0.0,
        avg_drawdown_pct=0.0,
        recovery_factor=0.0,
        total_commission=0.0,
        total_slippage=0.0,
        total_spread=0.0,
        total_costs=0.0,
        avg_trade_duration_minutes=0.0,
        max_consecutive_wins=0,
        max_consecutive_losses=0,
    )


def _calculate_detailed_metrics(
    trades: List[Trade],
    initial_balance: float,
    final_balance: float
) -> BacktestMetrics:
    """
    Calculate detailed metrics from Trade objects

    Args:
        trades: List of Trade objects
        initial_balance: Starting balance
        final_balance: Ending balance

    Returns:
        BacktestMetrics: Complete metrics
    """
    # Basic
    total_trades = len(trades)
    total_return_usd = final_balance - initial_balance
    total_return_pct = (total_return_usd / initial_balance) * 100

    # Win/Loss
    winners = [t for t in trades if t.net_pnl_usd > 0]
    losers = [t for t in trades if t.net_pnl_usd <= 0]

    num_winners = len(winners)
    num_losers = len(losers)
    win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0.0

    # Averages
    avg_win_usd = np.mean([t.net_pnl_usd for t in winners]) if winners else 0.0
    avg_win_pct = np.mean([t.net_pnl_pct for t in winners]) if winners else 0.0
    avg_loss_usd = np.mean([t.net_pnl_usd for t in losers]) if losers else 0.0
    avg_loss_pct = np.mean([t.net_pnl_pct for t in losers]) if losers else 0.0

    # Largest
    largest_win_usd = max([t.net_pnl_usd for t in winners]) if winners else 0.0
    largest_loss_usd = min([t.net_pnl_usd for t in losers]) if losers else 0.0

    # Profit Factor
    total_wins = sum([t.net_pnl_usd for t in winners])
    total_losses = abs(sum([t.net_pnl_usd for t in losers]))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    # Sharpe Ratio
    if len(trades) > 1:
        returns = [t.net_pnl_pct / 100 for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(len(trades)) if std_return > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (downside deviation only)
    if len(trades) > 1:
        returns = [t.net_pnl_pct / 100 for t in trades]
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino_ratio = (np.mean(returns) / downside_std) * np.sqrt(len(trades)) if downside_std > 0 else 0.0
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Drawdown (simplified - from trades)
    equity = initial_balance
    peak = initial_balance
    max_dd_usd = 0.0
    drawdowns = []

    for trade in trades:
        equity += trade.net_pnl_usd
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd_usd:
            max_dd_usd = dd
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        drawdowns.append(dd_pct)

    # Max drawdown percentage - peak'e göre hesaplanmalı (leverage-safe)
    # NOT: initial_balance'a göre hesaplamak leverage ile yanlış sonuç verir
    max_dd_pct = -(max_dd_usd / peak * 100) if peak > 0 else 0.0  # Negative!
    avg_dd_pct = -np.mean(drawdowns) if drawdowns else 0.0  # Negative!

    # Calmar Ratio (return / max DD) - use abs because DD is negative
    calmar_ratio = (total_return_pct / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0

    # Recovery Factor - use abs because DD is negative
    recovery_factor = (total_return_usd / max_dd_usd) if max_dd_usd > 0 else 0.0

    # Costs
    total_commission = sum([t.commission for t in trades])
    total_slippage = sum([t.slippage for t in trades])
    total_spread = sum([t.spread for t in trades])
    total_costs = total_commission + total_slippage + total_spread

    # Duration
    avg_duration = np.mean([t.duration_minutes for t in trades]) if trades else 0.0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.net_pnl_usd > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    return BacktestMetrics(
        total_return_usd=total_return_usd,
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        winners=num_winners,
        losers=num_losers,
        win_rate=win_rate,
        avg_win_usd=avg_win_usd,
        avg_win_pct=avg_win_pct,
        avg_loss_usd=avg_loss_usd,
        avg_loss_pct=avg_loss_pct,
        largest_win_usd=largest_win_usd,
        largest_loss_usd=largest_loss_usd,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        avg_drawdown_pct=avg_dd_pct,
        recovery_factor=recovery_factor,
        total_commission=total_commission,
        total_slippage=total_slippage,
        total_spread=total_spread,
        total_costs=total_costs,
        avg_trade_duration_minutes=avg_duration,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'calculate_metrics',  # NEW - V3
    'calculate_performance_metrics',  # DEPRECATED
    'print_performance_summary',
]
