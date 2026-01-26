#!/usr/bin/env python3
"""
tests/test_tier_manager.py
SuperBot - TierManager Test

Tests the Minimal TierManager:
1. Initialization
2. Tier transitions
3. Sorgulama
4. Summary/display

Usage:
    python tests/test_tier_manager.py
"""

from __future__ import annotations

import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.trading.tier_manager import TierManager, TierLevel, SymbolTierState

from core.logger_engine import get_logger
logger = get_logger("modules.trading.trading_engine")
# ============================================================================
# HELPERS
# ============================================================================

def load_symbols_from_json(limit: int = 0) -> list:
    """Load symbols from exchange_futures.json"""
    json_path = project_root / "data" / "json" / "exchange_futures.json"

    if not json_path.exists():
        logger.info(f"⚠️ {json_path} not found, test symbols will be used")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    symbols = []
    for item in data:
        if item.get("is_active") and item.get("status") == "TRADING":
            symbols.append(item.get("symbol"))

    if limit > 0:
        symbols = symbols[:limit]

    logger.info(f"📂 {len(symbols)} symbols loaded")
    return symbols


class MockLogger:
    """Test for mock logger"""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose


    def info(self, msg):
        if self.verbose:
            logger.info(f"INFO: {msg}")

    def debug(self, msg):
        if self.verbose:
            logger.info(f"DEBUG: {msg}")

    def warning(self, msg):
        logger.info(f"WARNING: {msg}")

    def error(self, msg):
        logger.info(f"ERROR: {msg}")


# ============================================================================
# TESTS
# ============================================================================

def test_initialization():
    """Test 1: Initialization"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 1: Initialization")
    logger.info("=" * 60)

    symbols = load_symbols_from_json(limit=50)
    

    # Create TierManager
    tm = TierManager(logger=logger)

    # Initialize
    tm.initialize(symbols)

    # Kontroller
    assert len(tm) == len(symbols), f"❌ Incorrect number of symbols: {len(tm)} != {len(symbols)}"
    assert tm.symbol_count == len(symbols), f"❌ symbol_count is incorrect"

    # All should start in ANALYSIS
    analysis_count = tm.count_by_tier(TierLevel.ANALYSIS)
    assert analysis_count == len(symbols), f"❌ All should be ANALYSIS: {analysis_count} != {len(symbols)}"

    # Summary kontrol
    summary = tm.get_summary()
    assert summary['total'] == len(symbols), f"❌ Total incorrect"
    assert summary['counts'][3] == len(symbols), f"❌ T3 count is incorrect"

    logger.info(f"✅ {len(symbols)} symbols started in TIER 3")
    logger.info(f"✅ Summary: {summary['display']}")

    return tm, symbols


def test_tier_transitions(tm: TierManager, symbols: list):
    """Test 2: Tier transitions"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 2: Tier Transitions")
    logger.info("=" * 60)

    # Select test symbols
    btc = symbols[0] if "BTCUSDT" in symbols else symbols[0]
    eth = symbols[1] if len(symbols) > 1 else btc

    # ANALYSIS → MONITORING
    logger.info(f"\n📍 {btc}: ANALYSIS → MONITORING")
    changed = tm.set_tier(btc, TierLevel.MONITORING)
    assert changed == True, "❌ The tier should be changed"
    assert tm.get_tier(btc) == TierLevel.MONITORING, "❌ It should be MONITORING"
    logger.info(f"✅ {btc} → TIER 2 (MONITORING)")

    # MONITORING → DECISION
    logger.info(f"\n📍 {btc}: MONITORING → DECISION")
    changed = tm.set_tier(btc, TierLevel.DECISION)
    assert changed == True, "❌ The tier should be changed"
    assert tm.get_tier(btc) == TierLevel.DECISION, "❌ It should be DECISION"

    state = tm.get_state(btc)
    assert state.consecutive_candles == 1, f"❌ consecutive_candles should be 1: {state.consecutive_candles}"
    logger.info(f"✅ {btc} → TIER 1 (DECISION) [#1]")

    # Stay in DECISION (consecutive candle)
    logger.info(f"\n📍 {btc}: Stay in DECISION (consecutive candle)")
    changed = tm.set_tier(btc, TierLevel.DECISION)
    assert changed == False, "❌ The tier should not change"

    state = tm.get_state(btc)
    assert state.consecutive_candles == 2, f"❌ consecutive_candles should be 2: {state.consecutive_candles}"
    logger.info(f"✅ {btc} remained in TIER 1 [#2]")

    # DECISION → POSITION
    logger.info(f"\n📍 {btc}: DECISION → POSITION")
    changed = tm.set_tier(btc, TierLevel.POSITION)
    assert changed == True, "❌ The tier should be changed"
    assert tm.get_tier(btc) == TierLevel.POSITION, "❌ It should be POSITION"

    state = tm.get_state(btc)
    assert state.consecutive_candles == 1, f"❌ consecutive_candles should be reset: {state.consecutive_candles}"
    logger.info(f"✅ {btc} → TIER 0 (POSITION) [#1]")

    # POSITION -> ANALYSIS (position closed)
    logger.info(f"\n📍 {btc}: POSITION -> ANALYSIS (position closed)")
    changed = tm.set_tier(btc, TierLevel.ANALYSIS)
    assert changed == True, "❌ The tier should be changed"
    assert tm.get_tier(btc) == TierLevel.ANALYSIS, "❌ It should be ANALYSIS"

    state = tm.get_state(btc)
    assert state.consecutive_candles == 0, f"❌ consecutive_candles must be 0: {state.consecutive_candles}"
    assert state.previous_tier == TierLevel.POSITION, "❌ previous_tier must be POSITION"
    logger.info(f"✅ {btc} → TIER 3 (ANALYSIS)")

    # Summary
    logger.info(f"\n📊 Summary: {tm.get_summary()['display']}")


def test_query_methods(tm: TierManager, symbols: list):
    """Test 3: Query methods"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 3: Query Methods")
    logger.info("=" * 60)

    # Assign some symbols to different tiers
    test_symbols = symbols[:10]

    # 2 POSITION, 3 DECISION, 2 MONITORING, 3 ANALYSIS
    tm.set_tier(test_symbols[0], TierLevel.POSITION)
    tm.set_tier(test_symbols[1], TierLevel.POSITION)
    tm.set_tier(test_symbols[2], TierLevel.DECISION)
    tm.set_tier(test_symbols[3], TierLevel.DECISION)
    tm.set_tier(test_symbols[4], TierLevel.DECISION)
    tm.set_tier(test_symbols[5], TierLevel.MONITORING)
    tm.set_tier(test_symbols[6], TierLevel.MONITORING)
    # 7, 8, 9 are already ANALYSIS

    # Query
    positions = tm.get_positions()
    decisions = tm.get_decisions()
    monitoring = tm.get_monitoring()

    logger.info(f"\n💼 POSITION ({len(positions)}): {positions}")
    logger.info(f"🎯 DECISION ({len(decisions)}): {decisions}")
    logger.info(f"👀 MONITORING ({len(monitoring)}): {monitoring[:5]}...")

    assert len(positions) == 2, f"❌ It should be POSITION=2: {len(positions)}"
    assert len(decisions) == 3, f"❌ It should be DECISION=3: {len(decisions)}"
    assert len(monitoring) == 2, f"❌ MONITORING must be 2: {len(monitoring)}"

    logger.info(f"\n✅ Query methods are working correctly")

    # __contains__ test
    assert test_symbols[0] in tm, "❌ __contains__ should work"
    assert "INVALID_SYMBOL" not in tm, "❌ __contains__ should work"
    logger.info(f"✅ __contains__ is working")


def test_display(tm: TierManager):
    """Test 4: Display format"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 4: Display Format")
    logger.info("=" * 60)

    # Simple display
    simple = tm.format_display(verbose=False)
    logger.info(f"\n📊 Simple: {simple}")

    # Verbose display
    verbose = tm.format_display(verbose=True, limit=5)
    logger.info(f"\n📊 Verbose:\n{verbose}")

    logger.info(f"\n✅ Display format is working")


def test_reset(tm: TierManager):
    """Test 5: Reset"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 5: Reset")
    logger.info("=" * 60)

    symbols = tm.symbols

    # Single symbol reset
    first_sym = symbols[0]
    old_tier = tm.get_tier(first_sym)
    tm.reset(first_sym)
    new_tier = tm.get_tier(first_sym)

    logger.info(f"📍 {first_sym}: {old_tier.name} → {new_tier.name} (reset)")
    assert new_tier == TierLevel.ANALYSIS, "❌ It should be ANALYSIS after the reset"
    logger.info(f"✅ Single symbol reset is working")

    # Reset all symbols
    tm.reset()

    for sym in symbols[:5]:
        tier = tm.get_tier(sym)
        assert tier == TierLevel.ANALYSIS, f"❌ {sym} should be ANALYSIS: {tier}"

    summary = tm.get_summary()
    assert summary['counts'][3] == len(symbols), "❌ All should be ANALYSIS"
    logger.info(f"✅ All symbols have been reset")
    logger.info(f"📊 Summary: {summary['display']}")


def test_simulation():
    """Test 6: Simulation (real-life scenario)"""
    logger.info("=" * 60)
    logger.info("🧪 TEST 6: Simulation")
    logger.info("=" * 60)

    # Load all symbols
    symbols = load_symbols_from_json()

    tm = TierManager(logger=logger)
    tm.initialize(symbols)

    logger.info(f"\n📊 Starting: {tm.get_summary()['display']}")

    # Simulation: 100 ticks
    logger.info(f"\n🔄 Starting 100 tick simulation...")

    for tick in range(100):
        # Select random symbols in each tick
        sample_size = min(20, len(symbols))
        random_symbols = random.sample(symbols, sample_size)

        for sym in random_symbols:
            current_tier = tm.get_tier(sym)

            # Random tier change (realistic probabilities)
            rand = random.random()

            if current_tier == TierLevel.ANALYSIS:
                if rand < 0.05:  # Switch to %5 MONITORING
                    tm.set_tier(sym, TierLevel.MONITORING)

            elif current_tier == TierLevel.MONITORING:
                if rand < 0.03:  # Go to %3 DECISION
                    tm.set_tier(sym, TierLevel.DECISION)
                elif rand < 0.15:  # Go to %12 ANALYSIS
                    tm.set_tier(sym, TierLevel.ANALYSIS)

            elif current_tier == TierLevel.DECISION:
                if rand < 0.02:  # Go to %2 POSITION (open trade)
                    tm.set_tier(sym, TierLevel.POSITION)
                elif rand < 0.20:  # Go back to %18 MONITORING
                    tm.set_tier(sym, TierLevel.MONITORING)

            elif current_tier == TierLevel.POSITION:
                if rand < 0.05:  # Switch to %5 ANALYSIS (close position)
                    tm.set_tier(sym, TierLevel.ANALYSIS)

        # Summary every 20 ticks
        if (tick + 1) % 20 == 0:
            summary = tm.get_summary()
            logger.info(f"   Tick {tick+1}: {summary['display']} | Changes: {summary['changes']}")

    # Final
    logger.info(f"\n📊 Final: {tm.get_summary()['display']}")
    logger.info(f"📈 Total tier changes: {tm.get_summary()['changes']}")

    # Verbose display
    logger.info(f"\n{tm.format_display(verbose=True, limit=5)}")

    logger.info(f"\n✅ Simulation completed")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 TIER MANAGER V5 TESTS")
    logger.info("=" * 70)

    try:
        # Test 1-4 (dependent)
        tm, symbols = test_initialization()
        test_tier_transitions(tm, symbols)
        test_query_methods(tm, symbols)
        test_display(tm)
        test_reset(tm)

        # Test 6 (independent)
        test_simulation()

        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 70)

        return True

    except AssertionError as e:
        logger.info(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        logger.info(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Windows encoding fix
    sys.stdout.reconfigure(encoding='utf-8')

    success = run_all_tests()
    sys.exit(0 if success else 1)
