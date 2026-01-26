#!/usr/bin/env python3
"""
modules/trading/test_tier_standalone.py
SuperBot - TierManager Standalone Test

TierManager + DisplayInfo main loop testi.

Usage:
    python modules/trading/test_tier_standalone.py
    python modules/trading/test_tier_standalone.py --verbose
    python modules/trading/test_tier_standalone.py --loop 60
"""

from __future__ import annotations

import sys
import json
import time
import random
import argparse
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# IMPORTS
# ============================================================================

from core.logger_engine import get_logger
from core.config_engine import get_config_engine
from core.event_bus import EventBus
from core.cache_manager import CacheManager

from modules.trading.tier_manager import TierManager, TierLevel
from modules.trading.display_info import DisplayInfo

# Logger
logger = get_logger("test.tier_manager")


# ============================================================================
# HELPERS
# ============================================================================

def load_symbols_from_json(limit: int = 0) -> list:
    """Load symbols from exchange_futures.json"""
    json_path = project_root / "data" / "json" / "exchange_futures.json"

    if not json_path.exists():
        logger.warning(f"⚠️ {json_path} not found, test symbols will be used")
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


def generate_mock_conditions(direction: str, target_score: float = None) -> tuple:
    """
    Create mock condition data.

    Args:
        direction: "LONG" or "SHORT"
        target_score: Hedef skor (0.0-1.0). None ise random.
                      1.0 = all conditions met
                      0.67 = 2/3 condition met
                      0.33 = 1/3 condition met
                      0.0 = no condition met

    Returns:
        (conditions_long, conditions_short, conditions_met, conditions_total)
    """
    total_conditions = 3

    # If a target score exists, determine the number of attempts accordingly.
    if target_score is not None:
        target_met = round(target_score * total_conditions)
        # Determine boolean values (the first N are True)
        met_values = [i < target_met for i in range(total_conditions)]
        random.shuffle(met_values)  # Shuffle
    else:
        # Random boolean values (more likely to be True - for test dynamism)
        met_values = [random.random() > 0.35 for _ in range(total_conditions)]

    # Example conditions (taken from the actual strategy format)
    long_conditions = [
        {'condition': ['close', '>', 'ema_5'], 'met': met_values[0], 'left_value': round(random.uniform(90, 110), 2), 'right_value': round(random.uniform(90, 110), 2)},
        {'condition': ['adx_14_adx', '>', 23], 'met': met_values[1], 'left_value': round(random.uniform(5, 40), 2), 'right_value': 23.0},
        {'condition': ['rsi_14', '<', 70], 'met': met_values[2], 'left_value': round(random.uniform(30, 80), 2), 'right_value': 70.0},
    ]
    short_conditions = [
        {'condition': ['close', '<', 'ema_5'], 'met': met_values[0], 'left_value': round(random.uniform(90, 110), 2), 'right_value': round(random.uniform(90, 110), 2)},
        {'condition': ['adx_14_adx', '>', 23], 'met': met_values[1], 'left_value': round(random.uniform(5, 40), 2), 'right_value': 23.0},
        {'condition': ['rsi_14', '>', 30], 'met': met_values[2], 'left_value': round(random.uniform(30, 80), 2), 'right_value': 30.0},
    ]

    if direction == "LONG":
        met = sum(1 for c in long_conditions if c['met'])
        total = len(long_conditions)
        return long_conditions, [], met, total
    else:
        met = sum(1 for c in short_conditions if c['met'])
        total = len(short_conditions)
        return [], short_conditions, met, total


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_main_loop(verbose: bool = False, duration: int = 0) -> None:
    """
    Main loop - Stops with Ctrl+C

    Args:
        verbose: Detailed output
        duration: Execution time (0 = infinite)
    """
    logger.info("=" * 60)
    logger.info("🔧 Loading dependencies...")
    logger.info("=" * 60)

    # Config (singleton with pre-loaded configs)
    config = get_config_engine()
    logger.info("✅ ConfigEngine loaded")

    # EventBus
    event_bus = EventBus()
    logger.info("✅ EventBus loaded")

    # CacheManager
    cache_manager = CacheManager()
    logger.info("✅ CacheManager loaded")

    # TierManager
    tier_manager = TierManager(
        logger=logger,
        config=config,
        event_bus=event_bus,
        cache_manager=cache_manager,
        verbose=verbose
    )
    logger.info("✅ TierManager loaded")

    # Symbols
    symbols = load_symbols_from_json(limit=50)
    tier_manager.initialize(symbols)

    # DisplayInfo
    display = DisplayInfo(
        tier_manager=tier_manager,
        logger=logger,
        config=config
    )
    logger.info("✅ DisplayInfo loaded")

    # Status interval (config'den - no prefix, merged to root)
    status_interval = config.get('status_display.status_interval', 15)
    logger.info(f"📊 Status interval: {status_interval}s")

    # TierManager interval bilgileri
    logger.info(f"📊 Intervals enabled: {tier_manager.intervals_enabled}")
    logger.info(f"📊 Tier intervals: {tier_manager.intervals}")
    logger.info(f"📊 Return to tier: {tier_manager.get_return_tier().name}")

    # TEST: Assign some symbols to different tiers initially (without waiting for the interval)
    # This allows the simulation to start immediately.
    test_symbols = symbols[:20]  # First 20 symbols
    for i, sym in enumerate(test_symbols):
        if i < 2:  # 2 POSITION
            tier_manager.set_tier(sym, TierLevel.POSITION, score=1.0, direction="LONG")
        elif i < 5:  # 3 DECISION - %100 score (all conditions met)
            direction = "LONG" if i % 2 == 0 else "SHORT"
            cond_long, cond_short, met, total = generate_mock_conditions(direction, target_score=1.0)
            tier_manager.set_tier(
                sym, TierLevel.DECISION, score=1.0, direction=direction,
                conditions_long=cond_long, conditions_short=cond_short,
                conditions_met=met, conditions_total=total,
                candle_close_pending=True
            )
        elif i < 12:  # 7 MONITORING - 67% score (2/3 condition met)
            direction = "LONG" if i % 2 == 0 else "SHORT"
            cond_long, cond_short, met, total = generate_mock_conditions(direction, target_score=0.67)
            score = met / total if total > 0 else 0.67
            tier_manager.set_tier(
                sym, TierLevel.MONITORING, score=score, direction=direction,
                conditions_long=cond_long, conditions_short=cond_short,
                conditions_met=met, conditions_total=total
            )
        # The rest remains in ANALYSIS

    logger.info(f"📊 Initial tier distribution: {tier_manager.get_summary()['display']}")

    last_status_time = 0

    start_time = time.time()

    logger.info("\n" + "=" * 60)
    if duration > 0:
        logger.info(f"🔄 MAIN LOOP ({duration} saniye)")
    else:
        logger.info("🔄 MAIN LOOP (Stops with Ctrl+C)")
    logger.info("=" * 60)

    try:
        while True:
            current_time = time.time()

            # Duration check
            if duration > 0 and (current_time - start_time) >= duration:
                break

            # ════════════════════════════════════════════════════════════
            # Periodic Status Display
            # ════════════════════════════════════════════════════════════
            if current_time - last_status_time >= status_interval:
                last_status_time = current_time

                logger.info("─" * 60)
                # In verbose mode, display the condition details first.
                if verbose:
                    for line in display.format_conditions_verbose():
                        logger.info(line)
                # Status line
                logger.info(display.format_status_line())
                # Tier summary
                for line in display.format_tier_summary(verbose=verbose):
                    logger.info(line)
                logger.info("─" * 60)

            # ════════════════════════════════════════════════════════════
            # Tier-based scanning simulation (TierManager interval management)
            # ════════════════════════════════════════════════════════════

            # TierManager'dan kontrol edilmesi gereken sembolleri al
            symbols_to_check = tier_manager.get_symbols_to_check()

            # TIER 0 (POSITION): Simulate trade closure
            if TierLevel.POSITION in symbols_to_check:
                for sym in symbols_to_check[TierLevel.POSITION]:
                    # Trade closing simulation (2% chance)
                    if random.random() < 0.02:
                        return_tier = tier_manager.get_return_tier()
                        tier_manager.set_tier(sym, return_tier)

            # TIER 1 (DECISION): Check for candle closure.
            # In a real system: conditions are checked again when the candle closes
            # If 100% is still valid -> go to POSITION (entry)
            # If not 100%, go back to MONITORING.
            if TierLevel.DECISION in symbols_to_check:
                for sym in symbols_to_check[TierLevel.DECISION]:
                    state = tier_manager.get_state(sym)

                    # Candle closing simulation (assuming a candle closes every 5 seconds)
                    # If candle_close_pending=True, it means the candle closing is being waited for.
                    if state and state.candle_close_pending:
                        # The candle is closed! Check the conditions again.
                        direction = state.direction or random.choice(["LONG", "SHORT"])
                        cond_long, cond_short, met, total = generate_mock_conditions(direction)
                        score = met / total if total > 0 else 0.0

                        if score >= 1.0:
                            # 100% still valid -> Make an entry! Go to POSITION.
                            tier_manager.set_tier(
                                sym, TierLevel.POSITION, score=1.0, direction=direction,
                                conditions_long=cond_long, conditions_short=cond_short,
                                conditions_met=met, conditions_total=total,
                                candle_close_pending=False,
                                ready_for_entry=True
                            )
                        else:
                            # Conditions have changed -> Fall back to MONITORING
                            tier_manager.set_tier(
                                sym, TierLevel.MONITORING, score=score, direction=direction,
                                conditions_long=cond_long, conditions_short=cond_short,
                                conditions_met=met, conditions_total=total,
                                candle_close_pending=False
                            )

            # TIER 2 (MONITORING): Tier transition based on score
            if TierLevel.MONITORING in symbols_to_check:
                for sym in symbols_to_check[TierLevel.MONITORING]:
                    direction = random.choice(["LONG", "SHORT"])
                    cond_long, cond_short, met, total = generate_mock_conditions(direction)
                    score = met / total if total > 0 else 0.5

                    # If 100% of the conditions are met -> go to DECISION
                    if score >= 1.0:
                        tier_manager.set_tier(
                            sym, TierLevel.DECISION, score=1.0, direction=direction,
                            conditions_long=cond_long, conditions_short=cond_short,
                            conditions_met=met, conditions_total=total,
                            candle_close_pending=True
                        )
                    # If it falls below 50% -> go back to ANALYSIS
                    elif score < 0.5:
                        tier_manager.set_tier(sym, TierLevel.ANALYSIS)
                    else:
                        # Stay in MONITORING, update with conditions
                        tier_manager.set_tier(
                            sym, TierLevel.MONITORING, score=score, direction=direction,
                            conditions_long=cond_long, conditions_short=cond_short,
                            conditions_met=met, conditions_total=total
                        )

            # TIER 3 (ANALYSIS): Tier transition based on score
            if TierLevel.ANALYSIS in symbols_to_check:
                analysis_symbols = symbols_to_check[TierLevel.ANALYSIS]
                # Check a few random symbols
                for sym in random.sample(analysis_symbols, min(5, len(analysis_symbols))):
                    direction = random.choice(["LONG", "SHORT"])
                    cond_long, cond_short, met, total = generate_mock_conditions(direction)
                    score = met / total if total > 0 else 0.0

                    # If 100% of the conditions are met -> proceed directly to DECISION.
                    if score >= 1.0:
                        tier_manager.set_tier(
                            sym, TierLevel.DECISION, score=1.0, direction=direction,
                            conditions_long=cond_long, conditions_short=cond_short,
                            conditions_met=met, conditions_total=total,
                            candle_close_pending=True
                        )
                    # If the condition is met (50%+), move to MONITORING.
                    elif score >= 0.5:
                        tier_manager.set_tier(
                            sym, TierLevel.MONITORING, score=score, direction=direction,
                            conditions_long=cond_long, conditions_short=cond_short,
                            conditions_met=met, conditions_total=total
                        )
                    # Less than 50% -> Stay in ANALYSIS (do nothing)

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Main loop durduruldu (Ctrl+C)")

    # Final status
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL STATUS")
    logger.info("=" * 60)
    logger.info(display.format_full_display(verbose=True))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TierManager Standalone Test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--loop', '-l', type=int, default=0, help='Run for N seconds (0 = infinite)')
    args = parser.parse_args()

    # Windows encoding fix
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    run_main_loop(verbose=args.verbose, duration=args.loop)


if __name__ == "__main__":
    main()
