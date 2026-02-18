#!/usr/bin/env python3
"""
modules/backtest/backtest_export.py
SuperBot - Backtest Results Export (V3)
Author: SuperBot Team
Date: 2025-11-16
Version: 3.0.0

Save backtest results to JSON and TXT files.

Features:
- JSON format (machine readable)
- TXT summary (human readable)
- Duplicate detection
- Professional formatting

Usage:
    from modules.backtest.backtest_export import save_backtest_results
    json_path, txt_path = save_backtest_results(result, output_dir="data/backtest_results")
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import json
import os
from datetime import datetime
from typing import Optional

from modules.backtest.backtest_types import BacktestResult
from core.timezone_utils import TimezoneUtils
from core.config_engine import get_config


def save_backtest_results(
    result: BacktestResult,
    output_dir: str = "data/backtest_results",
    logger = None
) -> tuple[str, str]:
    """
    Save backtest results to JSON and TXT files

    Args:
        result: BacktestResult instance
        output_dir: Output folder (default: data/backtest_results)
        logger: Logger instance (optional)

    Returns:
        tuple[str, str]: (json_path, txt_path)
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Filename format: backtest_strategyname_timeframe_startdate_enddate
    # Same TF and date range will overwrite the same file
    strategy_name = result.config.strategy_name
    timeframe = result.config.primary_timeframe
    start_date = result.config.start_date.strftime("%Y%m%d")
    end_date = result.config.end_date.strftime("%Y%m%d")
    base_filename = f"backtest_{strategy_name}_{timeframe}_{start_date}_{end_date}"

    # === JSON EXPORT ===
    json_data = {
        'strategy': {
            'name': result.config.strategy_name,
            'version': result.config.strategy_version,
        },
        'config': {
            'symbols': result.config.symbols,
            'primary_timeframe': result.config.primary_timeframe,
            'mtf_timeframes': result.config.mtf_timeframes,
            'start_date': result.config.start_date.isoformat(),
            'end_date': result.config.end_date.isoformat(),
            'initial_balance': result.config.initial_balance,
            'commission_pct': result.config.commission_pct,
            'slippage_pct': result.config.slippage_pct,
        },
        'execution': {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': result.execution_time_seconds,
        },
        'metrics': {
            'total_trades': result.metrics.total_trades,
            'winners': result.metrics.winners,
            'losers': result.metrics.losers,
            'win_rate': result.metrics.win_rate,
            'total_return_usd': result.metrics.total_return_usd,
            'total_return_pct': result.metrics.total_return_pct,
            'final_balance': result.config.initial_balance + result.metrics.total_return_usd,
            'avg_win_usd': result.metrics.avg_win_usd,
            'avg_loss_usd': result.metrics.avg_loss_usd,
            'profit_factor': result.metrics.profit_factor,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown_pct': result.metrics.max_drawdown_pct,
            'max_drawdown_usd': result.metrics.max_drawdown_usd,
            'total_commission': result.metrics.total_commission,
            'total_slippage': result.metrics.total_slippage,
        },
        'trades': [trade.to_dict() for trade in result.trades],
        'equity_curve': result.equity_curve,
    }

    json_path = os.path.join(output_dir, f"{base_filename}.json")

    # Save JSON (always overwrite)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    # === TXT EXPORT ===
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"{'BACKTEST SUMMARY - ' + strategy_name:^80}\n")
        f.write("=" * 80 + "\n\n")

        # Basic Info
        f.write("📋 BASIC INFO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Strategy:         {strategy_name} v{result.config.strategy_version}\n")
        f.write(f"Symbol:           {result.config.symbols[0]}\n")
        f.write(f"Primary TF:       {result.config.primary_timeframe}\n")
        if len(result.config.mtf_timeframes) > 1:
            f.write(f"MTF:              {', '.join(result.config.mtf_timeframes)}\n")
        f.write(f"Period:           {result.config.start_date.date()} → {result.config.end_date.date()}\n")
        f.write(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Time:   {result.execution_time_seconds:.2f}s\n\n")

        # Performance
        f.write("💰 PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        final_balance = result.config.initial_balance + result.metrics.total_return_usd
        f.write(f"Initial:          ${result.config.initial_balance:,.2f}\n")
        f.write(f"Final:            ${final_balance:,.2f}\n")
        profit_sign = '+' if result.metrics.total_return_usd >= 0 else ''
        f.write(f"Total Return:     {profit_sign}${result.metrics.total_return_usd:,.2f} ({result.metrics.total_return_pct:+.2f}%)\n\n")

        # Costs
        f.write("💸 COSTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Commission:       ${result.metrics.total_commission:,.2f}\n")
        f.write(f"Slippage:         ${result.metrics.total_slippage:,.2f}\n")
        total_costs = result.metrics.total_commission + result.metrics.total_slippage
        f.write(f"Total:            ${total_costs:,.2f}\n\n")

        # Trade Stats
        f.write("📈 TRADE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Trades:     {result.metrics.total_trades}\n")
        f.write(f"Winners:          {result.metrics.winners} ({result.metrics.win_rate:.1f}%)\n")
        f.write(f"Losers:           {result.metrics.losers} ({100 - result.metrics.win_rate:.1f}%)\n")
        f.write(f"Average Win:      ${result.metrics.avg_win_usd:,.2f}\n")
        f.write(f"Average Loss:     ${result.metrics.avg_loss_usd:,.2f}\n")
        f.write(f"Profit Factor:    {result.metrics.profit_factor:.2f}\n\n")

        # Risk Metrics
        f.write("📊 RISK METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Max Drawdown:     {result.metrics.max_drawdown_pct:.2f}% (${result.metrics.max_drawdown_usd:,.2f})\n")
        f.write(f"Sharpe Ratio:     {result.metrics.sharpe_ratio:.3f}\n\n")

        # Strategy Config Section (if available in trades)
        if result.trades:
            # Extract config from first trade
            first_trade = result.trades[0]
            if hasattr(first_trade, 'stop_loss_price') and first_trade.stop_loss_price:
                f.write("⚙️  STRATEGY CONFIG\n")
                f.write("-" * 80 + "\n")

                # Calculate SL/TP percentages from first trade
                if first_trade.side.value == 'LONG':
                    sl_pct = abs((first_trade.stop_loss_price - first_trade.entry_price) / first_trade.entry_price * 100)
                    if first_trade.take_profit_price:
                        tp_pct = abs((first_trade.take_profit_price - first_trade.entry_price) / first_trade.entry_price * 100)
                    else:
                        tp_pct = 0
                else:  # SHORT
                    sl_pct = abs((first_trade.entry_price - first_trade.stop_loss_price) / first_trade.entry_price * 100)
                    if first_trade.take_profit_price:
                        tp_pct = abs((first_trade.entry_price - first_trade.take_profit_price) / first_trade.entry_price * 100)
                    else:
                        tp_pct = 0

                f.write(f"Leverage:         1x\n")
                f.write(f"Margin Type:      isolated\n")
                f.write(f"Stop Loss:        {sl_pct:.1f}%\n")
                if tp_pct > 0:
                    f.write(f"Take Profit:      {tp_pct:.1f}%\n")
                else:
                    f.write(f"Take Profit:      ❌ None\n")
                f.write(f"Trailing Stop:    ❌ No\n")
                f.write(f"Break Even:       ✅ Yes\n\n")

        # Trades
        if result.trades:
            f.write(f"📝 TRADE LIST ({len(result.trades)} trades)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'ID':<6} {'Side':<6} {'Entry Time':<18} {'Entry $':<12} {'Exit Time':<18} {'Exit $':<12} {'PnL':<14} {'Exit':<8}\n")
            f.write("-" * 80 + "\n")

            for trade in result.trades:
                trade_id = str(trade.trade_id)[:5]
                side = trade.side.value

                # Format timestamps using TimezoneUtils (converts to local timezone from config)
                entry_time = TimezoneUtils.format(trade.entry_time, fmt='%Y-%m-%d %H:%M')
                exit_time = TimezoneUtils.format(trade.exit_time, fmt='%Y-%m-%d %H:%M')

                entry_price = trade.entry_price
                exit_price = trade.exit_price
                net_pnl = trade.net_pnl_usd
                pnl_sign = '+' if net_pnl >= 0 else ''
                exit_reason = trade.exit_reason.value[:7]

                f.write(f"{trade_id:<6} {side:<6} {entry_time:<18} ${entry_price:<11,.2f} "
                       f"{exit_time:<18} ${exit_price:<11,.2f} ${pnl_sign}{net_pnl:<13,.2f} {exit_reason:<8}\n")

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    if logger:
        logger.info(f"\n📁 Results saved:")
        logger.info(f"   JSON: {json_path}")
        logger.info(f"   TXT:  {txt_path}")

    return json_path, txt_path


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 backtest_export.py Test")
    print("=" * 60)

    # Create mock BacktestResult
    from modules.backtest.backtest_types import BacktestConfig, BacktestMetrics, Trade, PositionSide, ExitReason
    from datetime import datetime

    config = BacktestConfig(
        symbols=['BTCUSDT'],
        primary_timeframe='15m',
        mtf_timeframes=['5m', '15m', '30m'],
        initial_balance=10000,
    )

    metrics = BacktestMetrics(
        total_trades=10,
        winners=6,
        losers=4,
        win_rate=60.0,
        total_return_usd=500.0,
        total_return_pct=5.0,
        final_balance=10500.0,
        avg_win_usd=150.0,
        avg_loss_usd=75.0,
        profit_factor=2.0,
        sharpe_ratio=1.5,
        max_drawdown_pct=-2.5,
        max_drawdown_usd=250.0,
        total_commission=10.0,
        total_slippage=5.0,
    )

    # Mock trades
    trades = [
        Trade(
            trade_id=1,
            symbol='BTCUSDT',
            side=PositionSide.LONG,
            entry_time=datetime(2025, 1, 5, 10, 0),
            entry_price=100000.0,
            quantity=0.01,
            exit_time=datetime(2025, 1, 5, 12, 0),
            exit_price=101000.0,
            exit_reason=ExitReason.TAKE_PROFIT,
            gross_pnl_usd=100.0,
            net_pnl_usd=95.0,
            net_pnl_pct=0.95,
            commission=4.0,
            slippage=1.0,
        )
    ]

    result = BacktestResult(
        config=config,
        trades=trades,
        metrics=metrics,
        equity_curve=[],
        execution_time_seconds=1.23,
    )

    # Test export
    print("\nTest: Export backtest results...")
    json_path, txt_path = save_backtest_results(result, output_dir="data/backtest_results")

    print(f"\n✅ JSON: {json_path}")
    print(f"✅ TXT:  {txt_path}")

    print("\n✅ Test completed!")
    print("=" * 60)
