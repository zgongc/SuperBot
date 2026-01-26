#!/usr/bin/env python3
"""
components/optimizer/cli.py
SuperBot - Optimizer CLI Interface
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Command-line interface for the Optimizer.
Stage-by-stage optimization or full automatic mode.

Features:
- CLI argument parsing (argparse)
- Strategy loading (dosyadan import)
- Tek stage optimizasyon
- Optimize all stages automatically (--auto)
- Progress tracking
- Results summary

Usage:
    # Tek stage optimize et
    python -m components.optimizer.v2.cli \
        --strategy templates/TradingView_Dashboard.py \
        --stage risk_management \
        --method grid \
        --trials 100 \
        --symbol BTCUSDT \
        --timeframe 30m \
        --start 2024-01-01 \
        --end 2025-01-01

    # Automatically optimize all stages
    python -m components.optimizer.v2.cli \
        --strategy templates/TradingView_Dashboard.py \
        --auto \
        --symbol BTCUSDT \
        --timeframe 30m \
        --start 2024-01-01 \
        --end 2025-01-01

    # Custom optimizer config
    python -m components.optimizer.v2.cli \
        --strategy templates/TradingView_Dashboard.py \
        --auto \
        --symbol BTCUSDT \
        --timeframe 30m \
        --start 2024-01-01 \
        --end 2025-01-01 \
        --metric sharpe_ratio \
        --parallel 16 \
        --beam-width 10

Dependencies:
    - python>=3.10
    - argparse
    - asyncio
"""

from __future__ import annotations

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib.util

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from modules.optimizer import Optimizer, OptimizerConfig


# ============================================================================
# STRATEGY LOADING & CONFIG
# ============================================================================

def extract_backtest_config_from_strategy(strategy: Any) -> Dict[str, Any]:
    """
    Extract the backtest configuration from the Strategy.

    Args:
        strategy: Strategy instance

    Returns:
        Backtest config dict
    """
    from datetime import datetime, timedelta

    config = {}

    # Symbol (get the first enabled symbol)
    if hasattr(strategy, 'symbols'):
        for symbol_config in strategy.symbols:
            if symbol_config.enabled and symbol_config.symbol:
                # Get the first symbol
                first_symbol = symbol_config.symbol[0] if isinstance(symbol_config.symbol, list) else symbol_config.symbol
                config['symbol'] = f"{first_symbol}{symbol_config.quote}"
                break

    # If the symbol is not found, use the default.
    if 'symbol' not in config:
        config['symbol'] = 'BTCUSDT'

    # Timeframe
    if hasattr(strategy, 'primary_timeframe'):
        config['timeframe'] = strategy.primary_timeframe
    else:
        config['timeframe'] = '15m'

    # Initial balance
    if hasattr(strategy, 'initial_balance'):
        config['initial_balance'] = strategy.initial_balance
    else:
        config['initial_balance'] = 10000

    # Dates (take from strategy, otherwise last 1 year)
    if hasattr(strategy, 'backtest_start_date') and strategy.backtest_start_date:
        # The strategy contains a date in YYYY-MM-DDTHH:MM format, extract YYYY-MM-DD from it.
        start_str = strategy.backtest_start_date
        config['start_date'] = start_str[:10] if 'T' in start_str else start_str
    else:
        # Otherwise, the last 1 year
        start_date = datetime.now() - timedelta(days=365)
        config['start_date'] = start_date.strftime('%Y-%m-%d')

    if hasattr(strategy, 'backtest_end_date') and strategy.backtest_end_date:
        # There is a date in the Strategy.
        end_str = strategy.backtest_end_date
        config['end_date'] = end_str[:10] if 'T' in end_str else end_str
    else:
        # If not, today
        end_date = datetime.now()
        config['end_date'] = end_date.strftime('%Y-%m-%d')

    return config


def load_strategy_from_file(strategy_path: str) -> Any:
    """
    Load the strategy file and create an instance.

    Args:
        strategy_path: Path to the strategy file (e.g., 'templates/TradingView_Dashboard.py')

    Returns:
        Strategy instance

    Raises:
        ValueError: If the file is not found or the strategy class does not exist.
    """
    path = Path(strategy_path)

    if not path.exists():
        # Try relative to components/strategies/
        path = Path('components/strategies') / strategy_path
        if not path.exists():
            raise ValueError(f"Strategy file not found: {strategy_path}")

    print(f"📂 Loading strategy: {path}")

    # Import module
    spec = importlib.util.spec_from_file_location("strategy_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Module could not be loaded: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find strategy class (priority order: 'Strategy' -> 'strategy' containing -> any class)
    strategy_class = None
    candidates = []

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name not in ['BaseStrategy', 'Path']:
            # Skip utility classes (RiskManagement, ExitMethod, etc.)
            if any(x in name for x in ['Method', 'Config', 'Management', 'Parameters']):
                continue
            candidates.append((name, obj))

    # Priority 1: A class named exactly 'Strategy'
    for name, obj in candidates:
        if name == 'Strategy':
            strategy_class = obj
            break

    # Priority 2: Class containing 'Strategy' (but not Method or Config)
    if strategy_class is None:
        for name, obj in candidates:
            if 'Strategy' in name:
                strategy_class = obj
                break

    # Priority 3: The first class we find
    if strategy_class is None and candidates:
        strategy_class = candidates[0][1]

    if strategy_class is None:
        raise ValueError(f"Strategy class not found: {path}")

    print(f"✅ Strategy class found: {strategy_class.__name__}")

    # Create instance
    strategy_instance = strategy_class()

    return strategy_instance


# ============================================================================
# CLI FUNCTIONS
# ============================================================================

async def run_single_stage(
    strategy: Any,
    stage_name: str,
    method: str,
    trials: int,
    backtest_config: Dict[str, Any],
    optimizer_config: OptimizerConfig
):
    """
    Tek bir stage'i optimize et

    Args:
        strategy: Strategy instance
        stage_name: Stage name
        method: Optimization method
        trials: Maximum number of trials
        backtest_config: Backtest settings
        optimizer_config: Optimizer settings
    """
    print("\n" + "=" * 60)
    print("🚀 Optimizer - Single Stage Mode")
    print("=" * 60)
    print(f"📊 Stage: {stage_name}")
    print(f"🔧 Method: {method}")
    print(f"🔢 Trials: {trials}")
    print(f"💹 Symbol: {backtest_config['symbol']}")
    print(f"⏰ Timeframe: {backtest_config['timeframe']}")
    print(f"📅 Period: {backtest_config['start_date']} → {backtest_config['end_date']}")

    # Create optimizer
    optimizer = Optimizer(
        strategy=strategy,
        backtest_config=backtest_config,
        optimizer_config=optimizer_config
    )

    # Run optimization
    results = await optimizer.optimize_stage(
        stage_name=stage_name,
        stage_number=1,  # Single stage mode
        method=method,
        max_trials=trials
    )

    print(f"\n✅ Optimization completed!")
    print(f"📁 Results: {optimizer.stage_manager.run_dir}")


async def run_auto_mode(
    strategy: Any,
    backtest_config: Dict[str, Any],
    optimizer_config: OptimizerConfig,
    custom_stages: Optional[List[str]] = None,
    custom_trials: Optional[List[int]] = None
):
    """
    Automatically optimize all stages.

    Args:
        strategy: Strategy instance
        backtest_config: Backtest settings
        optimizer_config: Optimizer settings
        custom_stages: List of custom stages (if None, taken from optimizer_parameters)
        custom_trials: Number of trials per stage (if None, uses default)
    """
    print("\n" + "=" * 60)
    print("🚀 Optimizer - Auto Mode")
    print("=" * 60)
    print(f"💹 Symbol: {backtest_config['symbol']}")
    print(f"⏰ Timeframe: {backtest_config['timeframe']}")
    print(f"📅 Period: {backtest_config['start_date']} → {backtest_config['end_date']}")

    # Create optimizer
    optimizer = Optimizer(
        strategy=strategy,
        backtest_config=backtest_config,
        optimizer_config=optimizer_config
    )

    # Get stages from optimizer_parameters
    if custom_stages is None:
        optimizer_params = getattr(strategy, 'optimizer_parameters', {})

        # Exclude special keys and filter only enabled stages
        excluded_keys = {'optimization_settings', 'constraints'}
        stages = []
        skipped_stages = []

        for stage_name, stage_config in optimizer_params.items():
            if stage_name in excluded_keys:
                continue
            # Check if stage is enabled (default: False if not specified)
            if isinstance(stage_config, dict) and stage_config.get('enabled', False):
                stages.append(stage_name)
            else:
                skipped_stages.append(stage_name)

        if not stages:
            print("⚠️ No stage has been marked as 'enabled: True'!")
            print("    In the strategy file, activate at least one stage within optimizer_parameters:")
            print("    'risk_management': {'enabled': True, ...}")
            return

        print(f"\n📋 Optimize edilecek stage'ler ({len(stages)}):")
        for i, stage in enumerate(stages, 1):
            print(f"   ✅ {i}. {stage}")

        if skipped_stages:
            print(f"\n⏭️  Atlanan stage'ler (enabled: False):")
            for stage in skipped_stages:
                print(f"   ⬚ {stage}")
    else:
        stages = custom_stages

    # Default trial counts per stage
    if custom_trials is None:
        default_trials = {
            'risk_management': 105,
            'exit_strategy': 500,
            'indicators': 200,
            'position_management': 100,
            'market_filters': 150,
        }
        trials_per_stage = [default_trials.get(stage, 100) for stage in stages]
    else:
        trials_per_stage = custom_trials

    # Run each stage sequentially
    total_start = datetime.now()

    for stage_number, (stage_name, max_trials) in enumerate(zip(stages, trials_per_stage), 1):
        print(f"\n{'='*60}")
        print(f"📍 Stage {stage_number}/{len(stages)}: {stage_name}")
        print(f"{'='*60}")

        await optimizer.optimize_stage(
            stage_name=stage_name,
            stage_number=stage_number,
            method='grid',  # Default to grid search
            max_trials=max_trials
        )

    # Final summary
    total_time = (datetime.now() - total_start).total_seconds()

    print("\n" + "=" * 60)
    print("🎉 ALL STAGES COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"🔢 Total Backtests: {optimizer.total_backtests_run}")
    print(f"📁 Results: {optimizer.stage_manager.run_dir}")

    # Load and show final optimized parameters
    all_results = optimizer.stage_manager.load_all_results()

    print(f"\n📊 Optimized Parameters:")
    for result in all_results:
        print(f"\n   Stage: {result.stage}")
        print(f"   Best Params: {result.best_params}")


# ============================================================================
# MAIN CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create a CLI argument parser"""

    parser = argparse.ArgumentParser(
        description='SuperBot Optimizer - Stage-by-stage optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Disable prefix matching (e.g., --verbo won't work)
        epilog="""
Examples:
  # Auto mode (strategy defaults kullanarak)
  python -m components.optimizer.cli --strategy templates/TradingView_Dashboard.py --auto

  # Auto mode (custom symbol and timeframe)
  python -m components.optimizer.cli --strategy templates/TradingView_Dashboard.py \\
      --auto --symbol ETHUSDT --timeframe 1h

  # Single stage optimization
  python -m components.optimizer.cli --strategy templates/TradingView_Dashboard.py \\
      --stage risk_management --method grid --trials 100

  # Full custom config
  python -m components.optimizer.cli --strategy templates/TradingView_Dashboard.py \\
      --auto --symbol BTCUSDT --timeframe 30m --start 2024-01-01 --end 2025-01-01 --balance 50000
        """
    )

    # Required arguments
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy file path (e.g., templates/TradingView_Dashboard.py)'
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--auto',
        action='store_true',
        help='Automatically optimize all stages'
    )
    mode_group.add_argument(
        '--stage',
        type=str,
        help='Optimize a single stage (e.g., risk_management)'
    )

    # Backtest config (optional - strategy'den okunur, CLI override eder)
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT) - Taken from default in Strategy')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 30m, 1h) - Taken from the default in the Strategy')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD) - Default: last 1 year')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD) - Default: today')
    parser.add_argument('--balance', type=float, help='Initial balance - Taken from Strategy\'s default value')

    # Optimization config (single stage mode)
    parser.add_argument(
        '--method',
        type=str,
        choices=['grid', 'random', 'beam'],
        default='grid',
        help='Optimization method (default: grid)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Maximum number of trials (default: 100)'
    )

    # Optimizer config
    parser.add_argument(
        '--metric',
        type=str,
        default='sharpe_ratio',
        help=(
            'Primary metric for optimization (default: sharpe_ratio)\n'
            'Options:\n'
            '  - sharpe_ratio: Risk-adjusted return (return/risk ratio)\n'
            '  - total_return: Total return in percentage (highest gain)\n'
            '  - profit_factor: Profit/Loss ratio (>1 is profitable)\n'
            '  - calmar_ratio: Return/Max Drawdown ratio\n'
            '  - sortino_ratio: Similar to Sharpe ratio, but only considers downside risk\n'
            '  - weighted_score: A custom weighted metric combination\n'
            'Example: --metric total_return (for the highest return)'
        )
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=10,
        help='Max parallel backtests (default: 10)'
    )
    parser.add_argument(
        '--beam-width',
        type=int,
        default=10,
        help='Beam width (top N per stage) (default: 10)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.02,
        help='Risk-free rate for Sharpe/Sortino (default: 0.02)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose mode - Detailed output from BacktestEngine (indicators, conditions, signals)'
    )

    return parser


async def main_async(args):
    """Async main function"""

    try:
        # 1. Load strategy
        strategy = load_strategy_from_file(args.strategy)

        # 2. Extract backtest config from strategy (defaults)
        backtest_config = extract_backtest_config_from_strategy(strategy)

        # 3. Override with CLI arguments (if provided)
        if args.symbol:
            backtest_config['symbol'] = args.symbol
        if args.timeframe:
            backtest_config['timeframe'] = args.timeframe
        if args.start:
            backtest_config['start_date'] = args.start
        if args.end:
            backtest_config['end_date'] = args.end
        if args.balance:
            backtest_config['initial_balance'] = args.balance

        # Verbose mode
        backtest_config['verbose'] = args.verbose

        print(f"\n📋 Backtest Config (Strategy defaults + CLI overrides):")
        print(f"   Symbol: {backtest_config['symbol']}")
        print(f"   Timeframe: {backtest_config['timeframe']}")
        print(f"   Period: {backtest_config['start_date']} → {backtest_config['end_date']}")
        print(f"   Balance: ${backtest_config['initial_balance']:,.0f}")

        # 4. Create optimizer config
        optimizer_config = OptimizerConfig(
            max_parallel_backtests=args.parallel,
            primary_metric=args.metric,
            beam_width=args.beam_width,
            risk_free_rate=args.risk_free_rate,
        )

        # 4. Run optimization
        if args.auto:
            # Auto mode
            await run_auto_mode(
                strategy=strategy,
                backtest_config=backtest_config,
                optimizer_config=optimizer_config
            )
        else:
            # Single stage mode
            await run_single_stage(
                strategy=strategy,
                stage_name=args.stage,
                method=args.method,
                trials=args.trials,
                backtest_config=backtest_config,
                optimizer_config=optimizer_config
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Verbose mode - set the logger level to DEBUG
    if args.verbose:
        import logging
        from core.logger_engine import get_logger
        # Set the root logger to the DEBUG level
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set the optimizer loggers to DEBUG.
        for logger_name in ['components.optimizer', 'modules.backtest']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
        print("🔍 Debug mode is active - detailed logs will be displayed\n")

    # Run async main
    asyncio.run(main_async(args))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
