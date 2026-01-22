#!/usr/bin/env python3
"""
modules/analysis/cli.py

Analysis Module CLI - Test and analysis tool

Usage:
    python -m modules.analysis.cli --help
    python -m modules.analysis.cli analyze --symbol BTCUSDT --timeframe 5m
    python -m modules.analysis.cli test
    python -m modules.analysis.cli interactive
"""

import sys
import io

# Windows console encoding fix (emoji support)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import yaml

# SuperBot path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.analysis import AnalysisEngine


def load_analysis_config() -> dict:
    """Load analysis config from config/analysis.yaml"""
    config_path = BASE_DIR / "config" / "analysis.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('analysis', {})
        except Exception as e:
            print(f"⚠️  Configuration could not be loaded: {e}")
    return {}


# ============================================================================
# COLORS & FORMATTING
# ============================================================================

class Colors:
    """ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Backgrounds
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def colored(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{color}{text}{Colors.RESET}"


def print_header(title: str):
    """Print section header"""
    width = 70
    print()
    print(colored("=" * width, Colors.CYAN))
    print(colored(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(colored("=" * width, Colors.CYAN))


def print_subheader(title: str):
    """Print subsection header"""
    print()
    print(colored(f"── {title} ", Colors.YELLOW) + colored("─" * (50 - len(title)), Colors.DIM))


def print_formation(formation_type: str, formation, index: int = None):
    """Print a single formation"""
    idx_str = f"[{index:3d}]" if index is not None else ""

    if formation_type == 'bos':
        color = Colors.GREEN if formation.type == 'bullish' else Colors.RED
        arrow = "↑" if formation.type == 'bullish' else "↓"
        print(f"  {idx_str} {colored(f'{arrow} BOS', color)} "
              f"Level: {formation.broken_level:.4f} → {formation.break_price:.4f} "
              f"(Strength: {formation.strength:.0f})")

    elif formation_type == 'choch':
        color = Colors.GREEN if formation.type == 'bullish' else Colors.RED
        arrow = "⇈" if formation.type == 'bullish' else "⇊"
        print(f"  {idx_str} {colored(f'{arrow} CHoCH', Colors.YELLOW + Colors.BOLD)} "
              f"[{formation.previous_trend}] Level: {formation.broken_level:.4f} "
              f"(Significance: {formation.significance:.0f})")

    elif formation_type == 'fvg':
        color = Colors.GREEN if formation.type == 'bullish' else Colors.RED
        status = "FILLED" if formation.filled else f"{formation.filled_percent:.0f}%"
        print(f"  {idx_str} {colored(f'□ FVG', color)} "
              f"Range: {formation.bottom:.4f} - {formation.top:.4f} "
              f"({formation.size_pct:.2f}%) Age: {formation.age} [{status}]")

    elif formation_type == 'swing':
        color = Colors.GREEN if formation.type == 'high' else Colors.RED
        arrow = "▲" if formation.type == 'high' else "▼"
        status = "BROKEN" if formation.broken else "ACTIVE"
        print(f"  {idx_str} {colored(f'{arrow} Swing {formation.type.upper()}', color)} "
              f"Price: {formation.price:.4f} [{status}]")

    elif formation_type == 'ob':
        color = Colors.GREEN if formation.type == 'bullish' else Colors.RED
        arrow = "█" if formation.type == 'bullish' else "█"
        status = formation.status.upper()
        print(f"  {idx_str} {colored(f'{arrow} OB {formation.type.upper()}', color)} "
              f"Range: {formation.bottom:.4f} - {formation.top:.4f} "
              f"(Strength: {formation.strength:.2f}) [{status}]")

    elif formation_type == 'liquidity':
        color = Colors.MAGENTA
        status = "SWEPT" if formation.swept else "ACTIVE"
        print(f"  {idx_str} {colored(f'◆ LIQ {formation.type.upper()}', color)} "
              f"Level: {formation.level:.4f} (Strength: {formation.strength}) [{status}]")

    elif formation_type == 'qml':
        is_bullish = 'bullish' in formation.type
        color = Colors.GREEN if is_bullish else Colors.RED
        print(f"  {idx_str} {colored(f'◇ QML {formation.type.upper()}', color)} "
              f"Break: {formation.break_level:.4f} Head: {formation.head:.4f}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_parquet_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500
) -> Optional[pd.DataFrame]:
    """
    Load data from parquet files

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        timeframe: Timeframe (e.g., 5m, 1h)
        start_date: Start date (YYYY-MM-DD) - if set without end_date, end=now()
        end_date: End date (YYYY-MM-DD)
        limit: Max bars to load (only used if start_date is not set)

    Returns:
        DataFrame or None

    Logic:
        - start + end -> use date range (ignore limit)
        - start only -> from start to now (ignore limit)
        - neither -> last `limit` bar
    """
    # Yeni format: data/parquets/{symbol}/
    parquet_dir = BASE_DIR / "data" / "parquets"
    symbol_dir = parquet_dir / symbol

    if not symbol_dir.exists():
        print(colored(f"❌ Symbol directory not found: {symbol_dir}", Colors.RED))
        return None

    # Find parquet files - find all years
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(symbol_dir.glob(pattern))

    if not files:
        print(colored(f"❌ Parquet not found: {pattern}", Colors.RED))
        return None

    # Merge all files (multi-year support)
    all_dfs = []
    for parquet_file in sorted(files):
        print(colored(f"📂 Loading: {parquet_file.name}", Colors.DIM))
        try:
            df = pd.read_parquet(parquet_file)
            all_dfs.append(df)
        except Exception as e:
            print(colored(f"⚠️ File could not be read: {parquet_file.name} - {e}", Colors.YELLOW))

    if not all_dfs:
        print(colored(f"❌ No parquet files could be read", Colors.RED))
        return None

    # Merge
    df = pd.concat(all_dfs, ignore_index=True)

    # Column mapping (parquet format → analysis format)
    if 'open_time' in df.columns and 'timestamp' not in df.columns:
        # Convert datetime to timestamp ms
        df['timestamp'] = pd.to_datetime(df['open_time']).astype('int64') // 10**6

    # Ensure required columns
    required = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        print(colored(f"❌ Missing columns. Required: {required}", Colors.RED))
        print(colored(f"   Available: {list(df.columns)}", Colors.DIM))
        return None

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Date filtering logic
    use_date_range = start_date is not None

    if use_date_range:
        # start_date is provided - date range mode
        start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
        df = df[df['timestamp'] >= start_ts]

        if end_date:
            # end_date is provided.
            end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
            df = df[df['timestamp'] <= end_ts]
        else:
            # end_date is not provided - use now()
            now_ts = int(datetime.now().timestamp() * 1000)
            df = df[df['timestamp'] <= now_ts]
            print(colored(f"ℹ️  End date not specified, using now()", Colors.DIM))
    else:
        # start_date is not provided - last N bars mode
        if limit and len(df) > limit:
            df = df.tail(limit)

    df = df.reset_index(drop=True)
    return df


def generate_test_data(bars: int = 200) -> pd.DataFrame:
    """
    Generate synthetic test data with trends

    Args:
        bars: Number of bars

    Returns:
        DataFrame
    """
    np.random.seed(42)

    base_price = 100
    prices = [base_price]

    # Create trend patterns
    for i in range(bars - 1):
        if i < bars * 0.25:
            trend = 0.3 + np.random.randn() * 0.3  # Uptrend
        elif i < bars * 0.5:
            trend = -0.25 + np.random.randn() * 0.3  # Downtrend
        elif i < bars * 0.75:
            trend = 0.35 + np.random.randn() * 0.3  # Strong uptrend
        else:
            trend = -0.2 + np.random.randn() * 0.3  # Downtrend

        prices.append(prices[-1] + trend)

    prices = np.array(prices)
    highs = prices + np.abs(np.random.randn(bars)) * 0.5
    lows = prices - np.abs(np.random.randn(bars)) * 0.5
    opens = prices + np.random.randn(bars) * 0.2

    # Create timestamps (5m intervals)
    base_time = int(datetime(2024, 1, 1).timestamp() * 1000)

    return pd.DataFrame({
        'timestamp': [base_time + i * 300000 for i in range(bars)],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.randint(1000, 10000, bars)
    })


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def build_engine_config(file_config: dict = None) -> dict:
    """Build engine config from analysis.yaml config"""
    config = file_config or {}

    return {
        'swing': {
            'left_bars': config.get('swing', {}).get('left_bars', 5),
            'right_bars': config.get('swing', {}).get('right_bars', 5),
        },
        'structure': {
            'max_levels': config.get('bos', {}).get('max_levels', 5),
            'trend_strength': config.get('bos', {}).get('trend_strength', 2),
        },
        'fvg': {
            'min_size_pct': config.get('fvg', {}).get('min_size_pct', 0.1),
            'max_age': config.get('fvg', {}).get('max_age', 50),
        },
        'patterns': {'enabled': False},
        'orderblocks': {
            'enabled': config.get('orderblocks', {}).get('enabled', False),
            'strength_threshold': config.get('orderblocks', {}).get('strength_threshold', 1.0),
            'max_blocks': config.get('orderblocks', {}).get('max_blocks', 3),
            'lookback': config.get('orderblocks', {}).get('lookback', 20),
        },
        'liquidity': {
            'enabled': config.get('liquidity', {}).get('enabled', False),
            'equal_tolerance': config.get('liquidity', {}).get('equal_tolerance', 0.1),
            'max_zones': config.get('liquidity', {}).get('max_zones', 5),
            'sweep_lookback': config.get('liquidity', {}).get('sweep_lookback', 3),
        },
        'qml': {
            'enabled': config.get('qml', {}).get('enabled', False),
            'lookback_bars': config.get('qml', {}).get('lookback_bars', 30),
            'break_threshold': config.get('qml', {}).get('break_threshold', 0.1),
        },
    }


def run_analysis(
    data: pd.DataFrame,
    config: dict = None,
    verbose: bool = True
) -> None:
    """
    Run analysis and print results

    Args:
        data: OHLCV DataFrame
        config: Engine config (if None, loads from config/analysis.yaml)
        verbose: Print detailed output
    """
    # Load config from file if not provided
    if config is None:
        file_config = load_analysis_config()
        config = build_engine_config(file_config)

    engine = AnalysisEngine(config)

    print_header("MARKET STRUCTURE ANALYSIS")

    # Data info
    start_time = datetime.fromtimestamp(data['timestamp'].iloc[0] / 1000)
    end_time = datetime.fromtimestamp(data['timestamp'].iloc[-1] / 1000)

    print(f"\n  📊 Data:  {len(data)} bars")
    print(f"  📅 Range: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  💰 Price: {data['close'].iloc[0]:.4f} → {data['close'].iloc[-1]:.4f}")

    # Run analysis
    print(colored("\n  ⏳ Analyzing...", Colors.DIM))
    result = engine.analyze(data)

    # Summary
    summary = engine.get_summary()
    print_subheader("SUMMARY")
    print(f"  Current Trend: {colored(summary['current_trend'].upper(), Colors.CYAN)}")
    print(f"  BOS Count:     {summary['bos_count']}")
    print(f"  CHoCH Count:   {summary['choch_count']}")
    print(f"  FVG Count:     {summary['fvg_count']} (Active: {summary['active_fvg_count']})")
    print(f"  Swing Count:   {summary['swing_count']}")

    # Optional detector counts
    if 'ob_count' in summary:
        print(f"  OB Count:      {summary['ob_count']} (Active: {summary.get('active_ob_count', 0)})")
    if 'liquidity_count' in summary:
        print(f"  Liquidity:     {summary['liquidity_count']} (Active: {summary.get('active_liquidity_count', 0)}, Swept: {summary.get('swept_count', 0)})")
    if 'qml_count' in summary:
        print(f"  QML Count:     {summary['qml_count']} (Bull: {summary.get('bullish_qml_count', 0)}, Bear: {summary.get('bearish_qml_count', 0)})")

    # Current levels
    levels = engine.get_current_levels()
    print_subheader("CURRENT LEVELS")
    if levels['swing_high']:
        sh = levels['swing_high']
        print(f"  Swing High: {colored(f'{sh:.4f}', Colors.GREEN)}")
    if levels['swing_low']:
        sl = levels['swing_low']
        print(f"  Swing Low:  {colored(f'{sl:.4f}', Colors.RED)}")

    # BOS formations
    bos_list = engine.get_formations('bos')
    if bos_list:
        print_subheader(f"BOS FORMATIONS ({len(bos_list)})")
        for f in bos_list[-10:]:  # Last 10
            print_formation('bos', f, f.break_index)

    # CHoCH formations
    choch_list = engine.get_formations('choch')
    if choch_list:
        print_subheader(f"CHoCH FORMATIONS ({len(choch_list)})")
        for f in choch_list[-5:]:  # Last 5
            print_formation('choch', f, f.break_index)

    # Active FVGs
    active_fvgs = engine.get_formations('fvg', active_only=True)
    if active_fvgs:
        print_subheader(f"ACTIVE FVGs ({len(active_fvgs)})")
        for f in active_fvgs[-10:]:  # Last 10
            print_formation('fvg', f, f.created_index)

    # Order Blocks
    ob_list = engine.get_formations('ob')
    if ob_list:
        print_subheader(f"ORDER BLOCKS ({len(ob_list)})")
        for f in ob_list[-10:]:  # Last 10
            print_formation('ob', f, f.index)

    # Liquidity Zones
    liq_list = engine.get_formations('liquidity')
    if liq_list:
        print_subheader(f"LIQUIDITY ZONES ({len(liq_list)})")
        for f in liq_list[-10:]:  # Last 10
            print_formation('liquidity', f, f.index)

    # QML Patterns
    qml_list = engine.get_formations('qml')
    if qml_list:
        print_subheader(f"QML PATTERNS ({len(qml_list)})")
        for f in qml_list[-10:]:  # Last 10
            print_formation('qml', f, f.index)

    # Last bar analysis
    last = result[-1]
    print_subheader("LAST BAR ANALYSIS")
    print(f"  Bar Index:  {last.bar_index}")
    print(f"  Timestamp:  {datetime.fromtimestamp(last.timestamp / 1000).strftime('%Y-%m-%d %H:%M')}")
    print(f"  Trend:      {colored(last.trend.upper(), Colors.CYAN)}")
    print(f"  Bias:       {colored(last.market_bias.upper(), Colors.GREEN if last.market_bias == 'bullish' else Colors.RED if last.market_bias == 'bearish' else Colors.YELLOW)}")
    print(f"  Structure:  {last.structure}")

    if last.new_bos:
        print(f"  New BOS:    {colored('YES', Colors.GREEN)} - {last.new_bos.type}")
    if last.new_choch:
        print(f"  New CHoCH:  {colored('YES', Colors.YELLOW)} - {last.new_choch.type}")
    if last.new_fvg:
        print(f"  New FVG:    {colored('YES', Colors.BLUE)} - {last.new_fvg.type}")

    print()


def run_bar_by_bar(data: pd.DataFrame, start: int = 0, count: int = 20) -> None:
    """
    Run bar-by-bar analysis

    Args:
        data: OHLCV DataFrame
        start: Start bar index
        count: Number of bars to show
    """
    config = {
        'swing': {'left_bars': 3, 'right_bars': 3},
        'fvg': {'min_size_pct': 0.05},
        'patterns': {'enabled': False}
    }

    engine = AnalysisEngine(config)
    result = engine.analyze(data)

    print_header("BAR-BY-BAR ANALYSIS")

    end = min(start + count, len(result))

    for i in range(start, end):
        r = result[i]
        time_str = datetime.fromtimestamp(r.timestamp / 1000).strftime('%m-%d %H:%M')
        close = data['close'].iloc[i]

        # Build formation string
        formations = []
        if r.new_bos:
            color = Colors.GREEN if r.new_bos.type == 'bullish' else Colors.RED
            formations.append(colored(f"BOS({r.new_bos.type[0].upper()})", color))
        if r.new_choch:
            formations.append(colored(f"CHoCH({r.new_choch.type[0].upper()})", Colors.YELLOW))
        if r.new_fvg:
            color = Colors.GREEN if r.new_fvg.type == 'bullish' else Colors.RED
            formations.append(colored(f"FVG({r.new_fvg.type[0].upper()})", color))
        if r.new_swing:
            color = Colors.GREEN if r.new_swing.type == 'high' else Colors.RED
            formations.append(colored(f"SW({r.new_swing.type[0].upper()})", color))

        formation_str = " ".join(formations) if formations else colored("─", Colors.DIM)

        # Trend color
        trend_color = Colors.GREEN if r.trend == 'uptrend' else Colors.RED if r.trend == 'downtrend' else Colors.DIM
        trend_str = r.trend[:5].ljust(5)

        print(f"  [{i:3d}] {time_str} | {close:10.4f} | {colored(trend_str, trend_color)} | {formation_str}")

    print()


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_analyze(args):
    """Analyze command"""
    if args.symbol and args.timeframe:
        data = load_parquet_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit
        )
    else:
        print(colored("ℹ️  Using generated test data", Colors.BLUE))
        data = generate_test_data(args.limit or 200)

    if data is None:
        return

    run_analysis(data, verbose=args.verbose)


def cmd_test(args):
    """Test command with synthetic data"""
    print_header("ANALYSIS MODULE TEST")

    bars = args.bars or 200
    print(f"\n  Generating {bars} bars of synthetic data...")

    data = generate_test_data(bars)
    run_analysis(data)

    if args.bar_by_bar:
        run_bar_by_bar(data, start=50, count=30)


def cmd_interactive(args):
    """Interactive mode"""
    print_header("INTERACTIVE MODE")
    print("\n  Commands:")
    print("    analyze <symbol> <timeframe>  - Analyze parquet data")
    print("    test [bars]                   - Test with synthetic data")
    print("    bars [start] [count]          - Show bar-by-bar analysis")
    print("    quit / exit                   - Exit")
    print()

    data = None
    engine = None

    while True:
        try:
            cmd = input(colored("analysis> ", Colors.CYAN)).strip().lower()

            if not cmd:
                continue

            parts = cmd.split()
            command = parts[0]

            if command in ('quit', 'exit', 'q'):
                print("Bye!")
                break

            elif command == 'test':
                bars = int(parts[1]) if len(parts) > 1 else 200
                data = generate_test_data(bars)
                run_analysis(data)

            elif command == 'analyze':
                if len(parts) < 3:
                    print("Usage: analyze <symbol> <timeframe>")
                    continue
                data = load_parquet_data(parts[1].upper(), parts[2])
                if data is not None:
                    run_analysis(data)

            elif command == 'bars':
                if data is None:
                    print("No data loaded. Run 'test' or 'analyze' first.")
                    continue
                start = int(parts[1]) if len(parts) > 1 else 0
                count = int(parts[2]) if len(parts) > 2 else 20
                run_bar_by_bar(data, start, count)

            else:
                print(f"Unknown command: {command}")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(colored(f"Error: {e}", Colors.RED))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Market Structure Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.analysis.cli test
  python -m modules.analysis.cli test --bars 500 --bar-by-bar
  python -m modules.analysis.cli analyze --symbol BTCUSDT --timeframe 5m
  python -m modules.analysis.cli interactive
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze market data')
    analyze_parser.add_argument('--symbol', '-s', help='Trading symbol (e.g., BTCUSDT)')
    analyze_parser.add_argument('--timeframe', '-t', help='Timeframe (e.g., 5m, 1h)')
    analyze_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    analyze_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    analyze_parser.add_argument('--limit', '-l', type=int, default=500, help='Max bars')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test with synthetic data')
    test_parser.add_argument('--bars', '-b', type=int, default=200, help='Number of bars')
    test_parser.add_argument('--bar-by-bar', action='store_true', help='Show bar-by-bar analysis')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
