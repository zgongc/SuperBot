#!/usr/bin/env python3
"""
modules/backtest/test_backtest_engine.py
SuperBot - Backtest Engine Integration Test
Yazar: SuperBot Team
Tarih: 2025-11-16
Versiyon: 3.0.0

BacktestEngine V3 test suite:
Using existing strategy templates for testing.
- Position sizing tests (CRITICAL!)
- Metrics validation tests
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
import os
from typing import Dict

from modules.backtest.backtest_engine import BacktestEngine
from components.strategies.strategy_manager import StrategyManager


# ============================================================================
# TESTS
# ============================================================================

async def test_basic_backtest(strategy_path: str):
    """Test 1: Basic backtest"""
    print("\n" + "="*60)
    print(f"Test 1: Basic Backtest - {strategy_path}")
    print("="*60)

    # Load strategy
    strategy_manager = StrategyManager()

    Loading strategy: {strategy_path}
    strategy, _ = strategy_manager.load_strategy(strategy_path, validate=True)

    print(f"   ✅ Strategy: {strategy.strategy_name} v{strategy.strategy_version}")
    print(f"   ✅ Timeframe: {strategy.primary_timeframe}")
    print(f"   ✅ Initial Balance: ${strategy.initial_balance:,.0f}")

    # Backtest engine
    engine = BacktestEngine()

    Backtest is starting...
    result = await engine.run(strategy, use_cache=True)

    RESULTS:
    Trade count: {result.metrics.total_trades}
    print(f"   Total Return: ${result.metrics.total_return_usd:,.2f} ({result.metrics.total_return_pct:.2f}%)")
    print(f"   Win Rate: {result.metrics.win_rate:.2f}%")
    print(f"   Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
    print(f"   Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"   Execution Time: {result.execution_time_seconds:.2f}s")

    # Validation
    assert result.metrics.total_trades >= 0, "Total trades cannot be negative"
    assert result.execution_time_seconds > 0, "Execution time must be positive"

    print("\n   ✅ Test 1 PASSED!")
    return result


async def test_position_sizing(strategy_path: str):
    """Test 2: Position sizing - CRITICAL TEST!"""
    print("\n" + "="*60)
    print("Test 2: Position Sizing - Different values should yield different results!")
    print("="*60)

    # Load strategy
    strategy_manager = StrategyManager()

    engine = BacktestEngine()
    results = {}

    # Test 1: %5 position size
    print(f"\n   📊 Test 1: FIXED_PERCENT = 5%")
    strategy1, _ = strategy_manager.load_strategy(strategy_path, validate=True)
    strategy1.risk_management.position_percent_size = 5.0  # ← Use position_percent_size, not size_value!
    result1 = await engine.run(strategy1, use_cache=True)
    results['5%'] = result1.metrics
    print(f"      Trades: {result1.metrics.total_trades}")
    print(f"      Total Return: ${result1.metrics.total_return_usd:,.2f} ({result1.metrics.total_return_pct:.2f}%)")

    # Test 2: %20 position size
    print(f"\n   📊 Test 2: FIXED_PERCENT = 20%")
    strategy2, _ = strategy_manager.load_strategy(strategy_path, validate=True)
    strategy2.risk_management.position_percent_size = 20.0  # ← Use position_percent_size, not size_value!
    result2 = await engine.run(strategy2, use_cache=True)
    results['20%'] = result2.metrics
    print(f"      Trades: {result2.metrics.total_trades}")
    print(f"      Total Return: ${result2.metrics.total_return_usd:,.2f} ({result2.metrics.total_return_pct:.2f}%)")

    # CRITICAL VALIDATION
    print(f"\n   🔍 Verification:")
    return1_pct = results['5%'].total_return_pct
    return2_pct = results['20%'].total_return_pct
    return1_usd = results['5%'].total_return_usd
    return2_usd = results['20%'].total_return_usd

    print(f"   5% position → {return1_pct:+.2f}% (${return1_usd:+.2f})")
    print(f"   20% position → {return2_pct:+.2f}% (${return2_usd:+.2f})")

    Position sizing should be clearly defined in USD-based.
    # 20% position 5% is four times its value, so USD return should be approximately 4x.
    ratio_expected = 20.0 / 5.0  # 4.0x
    ratio_actual = abs(return2_usd / return1_usd) if return1_usd != 0 else 0

    print(f"   Expected ratio: {ratio_expected:.1f}x")
    print(f"   Actual ratio: {ratio_actual:.1f}x")

    Error rate control: 20% error penalty
    if abs(ratio_actual - ratio_expected) / ratio_expected < 0.2:  # ±20% tolerance
        print(f"   ✔️ Position sizing is working correctly!")
    else:
        print(f"   ❌ ERROR: Position sizing ratio is incorrect!")
        raise AssertionError(
            f"Position sizing ratio mismatch! Expected {ratio_expected:.1f}x but got {ratio_actual:.1f}x"
        )

    print("\n   ✅ Test 2 PASSED!")
    return results


async def test_metrics_calculation(strategy_path: str):
    "Are metric calculations correct?"
    print("\n" + "="*60)
    print("Test 3: Metrics Calculation Validation")
    print("="*60)

    # Load strategy
    strategy_manager = StrategyManager()
    strategy, _ = strategy_manager.load_strategy(strategy_path, validate=True)

    engine = BacktestEngine()
    result = await engine.run(strategy, use_cache=True)
    metrics = result.metrics

    print(f"\n   📊 METRICS:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Winners: {metrics.winners}")
    print(f"   Losers: {metrics.losers}")
    print(f"   Win Rate: {metrics.win_rate:.2f}%")
    print(f"   Total Return USD: ${metrics.total_return_usd:,.2f}")
    print(f"   Total Return %: {metrics.total_return_pct:.2f}%")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")

    Verification
    print(f"\n   🔍 Verification:")

    # 1. Winners + Losers = Total Trades
    if metrics.total_trades > 0:
        assert metrics.winners + metrics.losers == metrics.total_trades, \
            f"Winners ({metrics.winners}) + Losers ({metrics.losers}) != Total Trades ({metrics.total_trades})"
        print(f"   ✅ Winners + Losers = Total Trades")

    # 2. Win Rate Calculation
    if metrics.total_trades > 0:
        expected_win_rate = (metrics.winners / metrics.total_trades) * 100
        assert abs(metrics.win_rate - expected_win_rate) < 0.01, \
            f"Win rate mismatch: {metrics.win_rate} != {expected_win_rate}"
        print(f"   ✔️ Win Rate Calculated")

    # 3. Total Return USD Calculation (trades)
    if len(result.trades) > 0:
        manual_return = sum(t.net_pnl_usd for t in result.trades)
        assert abs(metrics.total_return_usd - manual_return) < 0.01, \
            f"Total return mismatch: {metrics.total_return_usd} != {manual_return}"
        print(f"   ✔️ Total Return USD Correctly Calculated")

    # 4. Profit Factor > 0 (if trade exists)
    if metrics.total_trades > 0 and metrics.losers > 0:
        assert metrics.profit_factor >= 0, "Profit factor cannot be negative"
        Profit Factor is valid

    # 5. Maximum drawdown ≤ 0 (maximum drawdown negative should be)
    assert metrics.max_drawdown_pct <= 0, "Max drawdown cannot be positive"
    Max Drawdown is valid

    print("\n   ✅ Test 3 PASSED!")
    return metrics


async def test_caching(strategy_path: str):
    """Is data caching working?"""
    print("\n" + "="*60)
    print("Test 4: Data Caching")
    print("="*60)

    # Load strategy
    strategy_manager = StrategyManager()
    strategy, _ = strategy_manager.load_strategy(strategy_path, validate=True)

    engine = BacktestEngine()

    # First run - to be cached
    First run (will be cached)…
    result1 = await engine.run(strategy, use_cache=True)
    time1 = result1.execution_time_seconds
    print(f"   Execution time: {time1:.2f}s")

    # Second run - to be retrieved from cache
    print("\n   Second run (from cache)...)")
    result2 = await engine.run(strategy, use_cache=True)
    time2 = result2.execution_time_seconds
    print(f"   Execution time: {time2:.2f}s")

    # Third run - cache disabled
    Third run (cache disabled)...
    result3 = await engine.run(strategy, use_cache=False)
    time3 = result3.execution_time_seconds
    print(f"   Execution time: {time3:.2f}s")

    RESULTS:
    First run: {time1:.2f}s
    print(f"   Cache'li run: {time2:.2f}s")
    print(f"   Cache'siz run: {time3:.2f}s")

    Results should be the same.
    assert result1.metrics.total_trades == result2.metrics.total_trades == result3.metrics.total_trades, \
        "Don't change the cache results!"
    print(f"   ✔️ Cache results are not changing")

    print("\n   ✅ Test 4 PASSED!")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='Backtest Engine Tests')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy file path (e.g. simple_rsi.py)')
    args = parser.parse_args()

    # Resolve strategy path
    strategy_path = args.strategy
    if not os.path.exists(strategy_path):
        # Try finding it in templates folder
        possible_path = os.path.join(project_root, "components", "strategies", "templates", strategy_path)
        if os.path.exists(possible_path):
            strategy_path = possible_path
        else:
            print(f"❌ Strategy file not found: {strategy_path}")
            return False

    print("\n" + "="*60)
    print("🧪 BACKTEST ENGINE V3 - INTEGRATION TESTS")
    print(f"Strategy: {strategy_path}")
    print("="*60)

    try:
        # Test 1: Basic backtest
        await test_basic_backtest(strategy_path)

        # Test 2: Position sizing (CRITICAL!)
        await test_position_sizing(strategy_path)

        # Test 3: Metrics calculation
        await test_metrics_calculation(strategy_path)

        # Test 4: Caching
        await test_caching(strategy_path)

        print("\n" + "="*60)
        print("All tests are successful!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        TEST FAILED: {e}
        print("="*60)
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
