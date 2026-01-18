#!/usr/bin/env python3
"""
modules/backtest/bulk_backtest.py
SuperBot - Bulk Backtest Runner
Yazar: SuperBot Team
Tarih: 2025-11-23
Versiyon: 1.0.0

Automatically tests multiple symbol and timeframe combinations.

Usage:
    python -m modules.backtest.bulk_backtest --strategy base_template2.py --symbol "BTCUSDT,ETHUSDT,BNBUSDT" --timeframe "1d,1w,4h"

Features:
- Test all symbol x timeframe combinations.
- Detailed results for each test.
- Best/worst performance comparison.
- Genel istatistikler

Dependencies:
    - python>=3.10
    - modules.backtest.backtest_engine
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import asyncio
import argparse
import itertools
import logging
from datetime import datetime
from typing import List

from modules.backtest.backtest_engine import BacktestEngine, run_backtest_cli
from modules.backtest.backtest_types import BacktestResult
from components.strategies.strategy_manager import StrategyManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# BULK BACKTEST
# ============================================================================

async def bulk_backtest(
    strategy_path: str,
    symbols: List[str],
    timeframes: List[str],
    start_date: str = None,
    end_date: str = None,
    initial_balance: float = None,
    verbose: bool = False
) -> List[BacktestResult]:
    """
    Bulk backtest - Test multiple symbol and timeframe combinations.

    Args:
        strategy_path: Path to the strategy file
        symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
        timeframes: List of timeframes (e.g., ['1d', '1w', '4h'])
        start_date: Start date (optional, YYYY-MM-DD)
        end_date: End date (optional, YYYY-MM-DD)
        initial_balance: Initial balance (optional)
        verbose: Detailed output

    Returns:
        List[BacktestResult]: The results of all combinations.
    """
    logger.info("=" * 80)
    logger.info("🚀 BULK BACKTEST STARTED")
    logger.info("=" * 80)
    logger.info(f"Strategy: {strategy_path}")
    logger.info(f"Symbols: {', '.join(symbols)} ({len(symbols)} items)")
    logger.info(f"Timeframes: {', '.join(timeframes)} ({len(timeframes)} adet)")
    if start_date:
        logger.info(f"Period: {start_date} → {end_date}")
    if initial_balance:
        logger.info(f"Balance: ${initial_balance:,.0f}")

    # Create all combinations
    combinations = list(itertools.product(symbols, timeframes))
    total_tests = len(combinations)

    logger.info(f"Total Number of Tests: {total_tests}")
    logger.info("=" * 80)

    results = []
    start_time = datetime.now()

    for idx, (symbol, timeframe) in enumerate(combinations, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Test {idx}/{total_tests}: {symbol} @ {timeframe}")
        logger.info(f"{'='*80}")

        try:
            # Run backtest for each combination
            result = await run_backtest_cli(
                strategy_path=strategy_path,
                verbose=verbose,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance
            )
            results.append(result)

            # Short summary
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate

            logger.info(f"\n✅ {symbol} @ {timeframe} - Completed!")
            logger.info(f"   Trades: {total_trades}, Win Rate: {win_rate:.1f}%, Return: {total_return:+.2f}%")

        except Exception as e:
            logger.error(f"❌ {symbol} @ {timeframe} - ERROR: {e}")
            import traceback
            if verbose:
                logger.debug(traceback.format_exc())
            continue

    # General summary
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("📊 BULK BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Success: {len(results)}")
    logger.info(f"Error: {total_tests - len(results)}")
    logger.info(f"Total Time: {elapsed:.1f}s")
    logger.info("")

    # Show the best results
    if results:
        # Sort by return
        sorted_by_return = sorted(
            results,
            key=lambda r: r.metrics.total_return_pct,
            reverse=True
        )

        logger.info("🏆 TOP 5 RESULTS (Return):")
        for i, result in enumerate(sorted_by_return[:5], 1):
            symbol = result.config.symbols[0]
            timeframe = result.config.primary_timeframe
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate

            logger.info(
                f"   {i}. {symbol} @ {timeframe}: "
                f"{total_return:+.2f}% ({total_trades} trades, WR: {win_rate:.1f}%)"
            )

        logger.info("")
        logger.info("📉 WORST 5 RESULTS (Return):")
        for i, result in enumerate(sorted_by_return[-5:][::-1], 1):
            symbol = result.config.symbols[0]
            timeframe = result.config.primary_timeframe
            total_trades = len(result.trades)
            total_return = result.metrics.total_return_pct
            win_rate = result.metrics.win_rate

            logger.info(
                f"   {i}. {symbol} @ {timeframe}: "
                f"{total_return:+.2f}% ({total_trades} trades, WR: {win_rate:.1f}%)"
            )

        # Genel istatistikler
        avg_return = sum(r.metrics.total_return_pct for r in results) / len(results)
        avg_trades = sum(len(r.trades) for r in results) / len(results)
        avg_winrate = sum(r.metrics.win_rate for r in results) / len(results)

        logger.info("")
        logger.info("📈 AVERAGE VALUES:")
        logger.info(f"   Return: {avg_return:+.2f}%")
        logger.info(f"   Number of Trades: {avg_trades:.1f}")
        logger.info(f"   Win Rate: {avg_winrate:.1f}%")

        # Sharpe ratio comparison
        logger.info("")
        logger.info("📊 BEST SHARPE RATIO:")
        sorted_by_sharpe = sorted(
            results,
            key=lambda r: r.metrics.sharpe_ratio,
            reverse=True
        )
        for i, result in enumerate(sorted_by_sharpe[:3], 1):
            symbol = result.config.symbols[0]
            timeframe = result.config.primary_timeframe
            sharpe = result.metrics.sharpe_ratio
            total_return = result.metrics.total_return_pct

            logger.info(
                f"   {i}. {symbol} @ {timeframe}: "
                f"Sharpe {sharpe:.3f} (Return: {total_return:+.2f}%)"
            )

    logger.info("=" * 80)

    return results


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Bulk Backtest - Test multiple symbols and timeframes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # 3 symbols x 2 timeframe = 6 tests
        python -m modules.backtest.bulk_backtest --strategy base_template2.py --symbol "BTCUSDT,ETHUSDT,BNBUSDT" --timeframe "1d,4h"

        # Specify the date range.
        python -m modules.backtest.bulk_backtest --strategy base_template2.py --symbol "BTCUSDT,ETHUSDT" --timeframe "1d,1w,4h" --start 2025-01-01 --end 2025-02-01

        # Custom balance
        python -m modules.backtest.bulk_backtest --strategy base_template2.py --symbol "BTCUSDT" --timeframe "1d,4h,1h,15m,5m" --balance 5000
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy file path (short name or full path)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Comma-separated list of symbols (e.g., "BTCUSDT,ETHUSDT,BNBUSDT")'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        required=True,
        help='Comma-separated list of timeframes (e.g., "1d,1w,4h")',
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD) - Inherited from Strategy by default'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD) - Inherited from Strategy by default'
    )

    parser.add_argument(
        '--balance',
        type=float,
        help='Initial balance - Inherited from Strategy'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Parse comma-separated lists
    symbols = [s.strip() for s in args.symbol.split(',')]
    timeframes = [tf.strip() for tf in args.timeframe.split(',')]

    # Run bulk backtest
    await bulk_backtest(
        strategy_path=args.strategy,
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        verbose=args.verbose
    )


if __name__ == "__main__":
    asyncio.run(main())
