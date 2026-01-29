#!/usr/bin/env python3
"""
modules/trading/tier_manager.py
SuperBot - Tier Manager with Config/EventBus/Cache Integration

Version: 5.1.0
Date: 2025-12-02

4-Tier symbol management system.
Only performs symbol -> tier mapping.
DOES NOT make decisions, does not evaluate conditions, only keeps records.

TIER Seviyeleri:
    TIER 0: POSITION   -> Active position exists
    TIER 1: DECISION   -> 100% condition, waiting for candle close
    TIER 2: MONITORING -> 50%+ condition, being monitored
    TIER 3: ANALYSIS   -> Initial pool

Entegrasyonlar:
    - Config: trading.yaml'dan threshold/interval okur
    - EventBus: Publishes tier changes (optional)
    - CacheManager: Caches tier states (optional)

Usage:
    # Basit (standalone)
    tier_manager = TierManager(logger=logger)
    tier_manager.initialize(symbols)

    # Integrated (for daemon/webui access)
    tier_manager = TierManager(
        logger=logger,
        config=config_engine,
        event_bus=event_bus,
        cache_manager=cache_manager
    )
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import time


# ============================================================================
# TIER ENUM
# ============================================================================

class TierLevel(IntEnum):
    """
    Tier levels - lower value = higher priority
    """
    POSITION = 0    # Active position management
    DECISION = 1    # Signal confirmed, waiting for candle close
    MONITORING = 2  # Potential candidate (>=50% condition)
    ANALYSIS = 3    # Initial pool

    @property
    def emoji(self) -> str:
        return {
            TierLevel.POSITION: "💼",
            TierLevel.DECISION: "🎯",
            TierLevel.MONITORING: "👀",
            TierLevel.ANALYSIS: "🔍"
        }.get(self, "❓")

    @property
    def name_tr(self) -> str:
        return {
            TierLevel.POSITION: "POSITION",
            TierLevel.DECISION: "DECISION",
            TierLevel.MONITORING: "MONITORING",
            TierLevel.ANALYSIS: "ANALYSIS"
        }.get(self, "?")

    @property
    def name_en(self) -> str:
        return {
            TierLevel.POSITION: "POSITION",
            TierLevel.DECISION: "DECISION",
            TierLevel.MONITORING: "MONITORING",
            TierLevel.ANALYSIS: "ANALYSIS"
        }.get(self, "?")


# ============================================================================
# SYMBOL STATE
# ============================================================================

@dataclass
class SymbolTierState:
    """Symbol tier status"""
    symbol: str
    tier: TierLevel = TierLevel.ANALYSIS
    previous_tier: Optional[TierLevel] = None
    tier_changed_at: Optional[datetime] = None
    consecutive_candles: int = 0  # Number of consecutive DECISION/POSITION candles

    # Optional metadata (set by StrategyExecutor)
    score: float = 0.0              # Condition score (0.0-1.0)
    direction: Optional[str] = None  # LONG/SHORT

    # Condition tracking (SignalValidator'dan)
    conditions_met: int = 0         # How many conditions were met
    conditions_total: int = 0       # Total number of conditions
    # Format: [{'condition': ['close', '>', 'ema_5'], 'met': True, 'left_value': 100.5, 'right_value': 99.2}, ...]
    conditions_long: List[Dict[str, Any]] = field(default_factory=list)
    conditions_short: List[Dict[str, Any]] = field(default_factory=list)

    # Entry flow control
    candle_close_pending: bool = False  # Is the candle closing?
    ready_for_entry: bool = False       # Is entry ready? (candle_close + %100)
    last_check_at: Optional[datetime] = None  # Last check time

    def to_dict(self) -> Dict[str, Any]:
        """Serialization for Cache/EventBus"""
        return {
            'symbol': self.symbol,
            'tier': self.tier.value,
            'tier_name': self.tier.name_en,
            'previous_tier': self.previous_tier.value if self.previous_tier else None,
            'tier_changed_at': self.tier_changed_at.isoformat() if self.tier_changed_at else None,
            'consecutive_candles': self.consecutive_candles,
            'score': self.score,
            'direction': self.direction,
            'conditions_met': self.conditions_met,
            'conditions_total': self.conditions_total,
            'conditions_long': self.conditions_long,
            'conditions_short': self.conditions_short,
            'candle_close_pending': self.candle_close_pending,
            'ready_for_entry': self.ready_for_entry,
            'last_check_at': self.last_check_at.isoformat() if self.last_check_at else None
        }


# ============================================================================
# TIER MANAGER
# ============================================================================

class TierManager:
    """
    Tier Manager with Config/EventBus/Cache Integration

    Sorumluluklar:
        ✅ Symbol -> Tier mapping
        ✅ Tier change tracking
        ✅ Summary/statistics
        ✅ Reading parameters from config
        ✅ Tier change broadcasting with EventBus (optional)
        ✅ State sharing with CacheManager (optional)

    OUT OF SCOPE:
        ❌ Condition evaluation (handled by StrategyExecutor)
        ❌ Score calculation (handled by StrategyExecutor)
        ❌ Trade decision (handled by TradingEngine)
        ❌ Position bilgisi (PositionManager yapar)
    """

    # Default config values
    DEFAULT_THRESHOLDS = {
        'monitoring': 0.50,
        'decision': 1.0
    }

    DEFAULT_INTERVALS = {
        'position': 1,
        'decision': 5,
        'monitoring': 15,
        'analysis': 60
    }

    def __init__(
        self,
        logger: Optional[Any] = None,
        config: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        on_tier_change: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Args:
            logger: Logger instance
            config: ConfigEngine instance (trading.yaml okur)
            event_bus: EventBus instance (tier.change broadcasts)
            cache_manager: CacheManager instance (tier:summary cache'ler)
            on_tier_change: Callback function(symbol, old_tier, new_tier)
        """
        self.logger = logger
        self.config = config
        self.event_bus = event_bus
        self.cache = cache_manager
        self.on_tier_change = on_tier_change
        self.verbose = verbose

        # Symbol states: symbol → SymbolTierState
        self._states: Dict[str, SymbolTierState] = {}

        # Statistics
        self._tier_change_count: int = 0
        self._initialized: bool = False

        # Interval tracking: tier_value → last_check_time
        self._last_tier_check: Dict[int, float] = {
            TierLevel.POSITION.value: 0.0,
            TierLevel.DECISION.value: 0.0,
            TierLevel.MONITORING.value: 0.0,
            TierLevel.ANALYSIS.value: 0.0,
        }

        # Load config
        self._load_config()

    def _load_config(self) -> None:
        """Load parameters from config"""
        if not self.config:
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()
            self.intervals = self.DEFAULT_INTERVALS.copy()
            self.intervals_enabled = False
            self.show_tier_changes = True
            return

        # trading.yaml → tiers section (no prefix, merged to root)
        tiers_config = self.config.get('tiers', {})

        # Thresholds
        self.thresholds = tiers_config.get('thresholds', self.DEFAULT_THRESHOLDS)

        # Intervals
        intervals_config = tiers_config.get('intervals', {})
        self.intervals_enabled = intervals_config.get('enabled', False)
        self.intervals = {
            'position': intervals_config.get('position', self.DEFAULT_INTERVALS['position']),
            'decision': intervals_config.get('decision', self.DEFAULT_INTERVALS['decision']),
            'monitoring': intervals_config.get('monitoring', self.DEFAULT_INTERVALS['monitoring']),
            'analysis': intervals_config.get('analysis', self.DEFAULT_INTERVALS['analysis']),
        }

        # Post position config
        post_position = tiers_config.get('post_position', {})
        self.return_to_tier = post_position.get('return_to_tier', 3)

        # Status display config (no prefix, merged to root)
        self.show_tiers = self.config.get('status_display.show_conditions_for_tiers', [0, 1, 2, 3])
        self.max_display = self.config.get('status_display.max_symbols_display', 20)
        self.status_interval = self.config.get('status_display.status_interval', 15)
        self.show_tier_changes = self.config.get('status_display.show_tier_changes', True)

        if self.verbose and self.logger:
            self.logger.info(
                f"📊 TierManager config: intervals_enabled={self.intervals_enabled}, "
                f"intervals={self.intervals}, return_to_tier={self.return_to_tier}"
            )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self, symbols: List[str]) -> None:
        """
        Initialize symbols with the ANALYSIS tier.

        Args:
            symbols: List of symbols
        """
        for symbol in symbols:
            if symbol not in self._states:
                self._states[symbol] = SymbolTierState(
                    symbol=symbol,
                    tier=TierLevel.ANALYSIS,
                    tier_changed_at=datetime.now()
                )

        self._initialized = True

        if self.verbose:
            self.logger.info(f"📊 TierManager: {len(symbols)} symbols started in TIER 3")

        # Save to cache
        self._cache_summary()

    def add_symbol(self, symbol: str) -> None:
        """Add a single symbol"""
        if symbol not in self._states:
            self._states[symbol] = SymbolTierState(
                symbol=symbol,
                tier=TierLevel.ANALYSIS,
                tier_changed_at=datetime.now()
            )

    def remove_symbol(self, symbol: str) -> None:
        """Extract symbol"""
        self._states.pop(symbol, None)

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    def set_tier(
        self,
        symbol: str,
        tier: TierLevel,
        score: float = 0.0,
        direction: Optional[str] = None,
        conditions_long: Optional[List[Dict[str, Any]]] = None,
        conditions_short: Optional[List[Dict[str, Any]]] = None,
        conditions_met: int = 0,
        conditions_total: int = 0,
        candle_close_pending: bool = False,
        ready_for_entry: bool = False
    ) -> bool:
        """
        Assign the tier of the symbol.

        Args:
            symbol: Symbol name
            tier: New tier
            score: Condition score (optional, from StrategyExecutor)
            direction: LONG/SHORT (optional)
            conditions_long: Details for LONG condition
                Format: [{'condition': ['close', '>', 'ema_5'], 'met': True, 'left_value': 100.5, 'right_value': 99.2}, ...]
            conditions_short: SHORT condition details
            conditions_met: How many conditions are met
            conditions_total: Total number of conditions
            candle_close_pending: Is the candle closing?
            ready_for_entry: Is entry ready?

        Returns:
            bool: Did the tier change?
        """
        # If the state does not exist, create it.
        if symbol not in self._states:
            self._states[symbol] = SymbolTierState(symbol=symbol)

        state = self._states[symbol]
        old_tier = state.tier

        # Update score and direction
        state.score = score
        state.direction = direction

        # Update condition tracking
        state.conditions_met = conditions_met
        state.conditions_total = conditions_total
        state.candle_close_pending = candle_close_pending
        state.ready_for_entry = ready_for_entry
        state.last_check_at = datetime.now()

        # Update condition details
        if conditions_long is not None:
            state.conditions_long = conditions_long
        if conditions_short is not None:
            state.conditions_short = conditions_short

        # No changes
        if old_tier == tier:
            # But if it remains in DECISION/POSITION, increment the counter.
            if tier in (TierLevel.DECISION, TierLevel.POSITION):
                state.consecutive_candles += 1
            return False

        # Tier changed
        state.previous_tier = old_tier
        state.tier = tier
        state.tier_changed_at = datetime.now()
        self._tier_change_count += 1

        # Counter reset (when transitioning to a new tier)
        if tier in (TierLevel.DECISION, TierLevel.POSITION):
            state.consecutive_candles = 1
        else:
            state.consecutive_candles = 0

        # Log (show_tier_changes check)
        if self.logger and self.show_tier_changes:
            self.logger.info(
                f"{old_tier.emoji}→{tier.emoji} {symbol}: "
                f"TIER {old_tier.value} → TIER {tier.value}"
            )

        # Callback
        if self.on_tier_change:
            try:
                self.on_tier_change(symbol, old_tier, tier)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"on_tier_change callback error: {e}")

        # EventBus publish
        self._publish_tier_change(symbol, old_tier, tier, state)

        return True

    def get_tier(self, symbol: str) -> TierLevel:
        """Get the symbol's tier"""
        state = self._states.get(symbol)
        return state.tier if state else TierLevel.ANALYSIS

    def get_state(self, symbol: str) -> Optional[SymbolTierState]:
        """Get the complete state of the symbol"""
        return self._states.get(symbol)

    def update_symbol_state(
        self,
        symbol: str,
        score: float,
        conditions_met: int = 0,
        conditions_total: int = 0,
        conditions_long: Optional[List[Dict[str, Any]]] = None,
        conditions_short: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update the symbol state and determine the tier based on the score.

        Score thresholds (config'den):
        - score >= 1.0 (decision_threshold) → DECISION
        - score >= 0.5 (monitoring_threshold) → MONITORING
        - score < 0.5 → ANALYSIS

        Args:
            symbol: Symbol name
            score: Condition score (0.0-1.0)
            conditions_met: Number of conditions met
            conditions_total: Total number of conditions
            conditions_long: LONG condition details
            conditions_short: SHORT condition details
        """
        # Determine direction (which side is better)
        long_score = 0.0
        short_score = 0.0

        if conditions_long:
            met = sum(1 for c in conditions_long if c.get('met', False))
            total = len(conditions_long)
            long_score = met / total if total > 0 else 0.0

        if conditions_short:
            met = sum(1 for c in conditions_short if c.get('met', False))
            total = len(conditions_short)
            short_score = met / total if total > 0 else 0.0

        direction = "LONG" if long_score >= short_score else "SHORT"

        # Determine the tier based on the score
        if score >= self.thresholds['decision']:
            new_tier = TierLevel.DECISION
        elif score >= self.thresholds['monitoring']:
            new_tier = TierLevel.MONITORING
        else:
            new_tier = TierLevel.ANALYSIS

        # Do not modify if the current tier is POSITION (position management is separate)
        current_state = self._states.get(symbol)
        if current_state and current_state.tier == TierLevel.POSITION:
            # Only update conditions, do not change the tier.
            current_state.score = score
            current_state.direction = direction
            current_state.conditions_met = conditions_met
            current_state.conditions_total = conditions_total
            current_state.conditions_long = conditions_long or []
            current_state.conditions_short = conditions_short or []
            current_state.last_check_at = datetime.now()
            return

        # update with set_tier
        self.set_tier(
            symbol=symbol,
            tier=new_tier,
            score=score,
            direction=direction,
            conditions_long=conditions_long,
            conditions_short=conditions_short,
            conditions_met=conditions_met,
            conditions_total=conditions_total
        )

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_by_tier(self, tier: TierLevel) -> List[str]:
        """Belirli tier'daki sembolleri al"""
        return [s for s, state in self._states.items() if state.tier == tier]

    def get_positions(self) -> List[str]:
        """POSITION tier'daki sembolleri al"""
        return self.get_by_tier(TierLevel.POSITION)

    def get_decisions(self) -> List[str]:
        """DECISION tier'daki sembolleri al"""
        return self.get_by_tier(TierLevel.DECISION)

    def get_monitoring(self) -> List[str]:
        """MONITORING tier'daki sembolleri al"""
        return self.get_by_tier(TierLevel.MONITORING)

    def get_analysis(self) -> List[str]:
        """ANALYSIS tier'daki sembolleri al"""
        return self.get_by_tier(TierLevel.ANALYSIS)

    def count_by_tier(self, tier: TierLevel) -> int:
        """The number of symbols in the tier"""
        return len(self.get_by_tier(tier))

    def get_check_interval(self, tier: TierLevel) -> int:
        """Check interval for the tier (seconds)"""
        tier_name = tier.name.lower()
        return self.intervals.get(tier_name, 60)

    # ========================================================================
    # INTERVAL CHECKING
    # ========================================================================

    def should_check_tier(self, tier: TierLevel) -> bool:
        """
        Should this tier be checked right now?

        If intervals.enabled is False, it always returns True.
        If intervals.enabled is True, it decides based on the interval.

        Args:
            tier: The tier to be checked.

        Returns:
            bool: Kontrol edilmeli mi?
        """
        # If intervals are disabled, always check.
        if not self.intervals_enabled:
            return True

        current_time = time.time()
        tier_value = tier.value
        tier_name = tier.name.lower()

        interval = self.intervals.get(tier_name, 60)
        last_check = self._last_tier_check.get(tier_value, 0.0)

        return (current_time - last_check) >= interval

    def mark_tier_checked(self, tier: TierLevel) -> None:
        """
        Mark as checked.

        Args:
            tier: The tier to be marked.
        """
        self._last_tier_check[tier.value] = time.time()

    def get_symbols_to_check(self) -> Dict[TierLevel, List[str]]:
        """
        Group the symbols to be checked according to the tier.

        If intervals.enabled is False, it iterates over all symbols.
        If intervals.enabled is True, it iterates over the symbols of only the tiers that have completed the interval.

        Returns:
            {TierLevel.POSITION: [...], TierLevel.DECISION: [...], ...}
        """
        result: Dict[TierLevel, List[str]] = {}

        for tier in TierLevel:
            if self.should_check_tier(tier):
                symbols = self.get_by_tier(tier)
                if symbols:
                    result[tier] = symbols
                    self.mark_tier_checked(tier)

        return result

    def get_return_tier(self) -> TierLevel:
        """
        The tier to return to after the position is closed.

        Returns:
            TierLevel: The tier to be transitioned to (from config).
        """
        return TierLevel(getattr(self, 'return_to_tier', 3))

    # ========================================================================
    # SUMMARY / STATS
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Tier summary

        Returns:
            {
                'counts': {0: 2, 1: 5, 2: 15, 3: 378},
                'total': 400,
                'display': 'T0:2 | T1:5 | T2:15 | T3:378',
                'changes': 42,
                'thresholds': {...},
                'intervals': {...}
            }
        """
        counts = {tier.value: 0 for tier in TierLevel}

        for state in self._states.values():
            counts[state.tier.value] += 1

        total = sum(counts.values())

        display = " | ".join([
            f"T{tier.value}:{counts[tier.value]}"
            for tier in TierLevel
        ])

        return {
            'counts': counts,
            'total': total,
            'display': display,
            'changes': self._tier_change_count,
            'thresholds': self.thresholds,
            'intervals': self.intervals,
            'timestamp': datetime.now().isoformat()
        }

    def format_display(self, verbose: bool = False, limit: Optional[int] = None) -> str:
        """
        Terminal display format.

        Args:
            verbose: Detailed display
            limit: Maximum number of symbols for each tier (None = from config)
        """
        if limit is None:
            limit = getattr(self, 'max_display', 10)

        summary = self.get_summary()
        lines = [f"📊 TIERS: {summary['display']}"]

        if not verbose:
            return lines[0]

        lines.append("")

        for tier in TierLevel:
            # Filter according to the show_tiers configuration.
            if hasattr(self, 'show_tiers') and tier.value not in self.show_tiers:
                continue

            symbols = self.get_by_tier(tier)
            count = len(symbols)

            if count == 0:
                continue

            lines.append(f"{tier.emoji} TIER {tier.value} - {tier.name_tr} ({count}):")

            for sym in symbols[:limit]:
                state = self._states[sym]
                parts = [f"   • {sym}"]

                if state.score > 0:
                    parts.append(f"({state.score*100:.0f}%)")

                if state.direction:
                    parts.append(f"[{state.direction}]")

                if state.consecutive_candles > 0:
                    parts.append(f"#{state.consecutive_candles}")

                lines.append(" ".join(parts))

            if count > limit:
                lines.append(f"   ... and {count - limit} more symbols")

        return "\n".join(lines)

    # ========================================================================
    # EVENTBUS INTEGRATION
    # ========================================================================

    def _publish_tier_change(
        self,
        symbol: str,
        old_tier: TierLevel,
        new_tier: TierLevel,
        state: SymbolTierState
    ) -> None:
        """Publish a tier change to the EventBus"""
        if not self.event_bus:
            return

        try:
            event_data = {
                'symbol': symbol,
                'old_tier': old_tier.value,
                'new_tier': new_tier.value,
                'old_tier_name': old_tier.name_en,
                'new_tier_name': new_tier.name_en,
                'score': state.score,
                'direction': state.direction,
                'consecutive_candles': state.consecutive_candles,
                'timestamp': datetime.now().isoformat()
            }

            # Genel tier change event
            self.event_bus.publish(
                topic='tier.change',
                data=event_data,
                source='TierManager'
            )

            # Symbol special event
            self.event_bus.publish(
                topic=f'tier.{symbol}.change',
                data=event_data,
                source='TierManager'
            )

        except Exception as e:
            if self.logger:
                self.logger.error(f"EventBus publish error: {e}")

    def publish_status_report(self) -> None:
        """Publish status report (called periodically)"""
        if not self.event_bus:
            return

        try:
            summary = self.get_summary()

            # Detailed report
            report_data = {
                'summary': summary,
                'positions': self.get_positions(),
                'decisions': self.get_decisions(),
                'monitoring_count': self.count_by_tier(TierLevel.MONITORING),
                'analysis_count': self.count_by_tier(TierLevel.ANALYSIS),
                'top_candidates': self._get_top_candidates(limit=10),
                'timestamp': datetime.now().isoformat()
            }

            self.event_bus.publish(
                topic='tier.status.report',
                data=report_data,
                source='TierManager'
            )

            # Also save to the cache
            self._cache_summary()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Status report publish error: {e}")

    def _get_top_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Highest scoring MONITORING/DECISION symbols"""
        candidates = []

        for symbol, state in self._states.items():
            if state.tier in (TierLevel.MONITORING, TierLevel.DECISION):
                candidates.append({
                    'symbol': symbol,
                    'tier': state.tier.value,
                    'score': state.score,
                    'direction': state.direction
                })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:limit]

    # ========================================================================
    # CACHE INTEGRATION
    # ========================================================================

    def _cache_summary(self) -> None:
        """Save the summary to the cache"""
        if not self.cache:
            return

        try:
            summary = self.get_summary()
            self.cache.set('tier:summary', summary, ttl=60)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Cache set error: {e}")

    def _cache_state(self, symbol: str, state: SymbolTierState) -> None:
        """Save the single state to the cache"""
        if not self.cache:
            return

        try:
            self.cache.set(f'tier:state:{symbol}', state.to_dict(), ttl=60)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Cache set error: {e}")

    # ========================================================================
    # UTILITY
    # ========================================================================

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        State resetle

        Args:
            symbol: Specific symbol (None = all)
        """
        if symbol:
            if symbol in self._states:
                self._states[symbol] = SymbolTierState(
                    symbol=symbol,
                    tier=TierLevel.ANALYSIS,
                    tier_changed_at=datetime.now()
                )
        else:
            symbols = list(self._states.keys())
            self._states.clear()
            self._tier_change_count = 0
            for sym in symbols:
                self._states[sym] = SymbolTierState(
                    symbol=sym,
                    tier=TierLevel.ANALYSIS,
                    tier_changed_at=datetime.now()
                )

        self._cache_summary()

    def reload_config(self) -> None:
        """Reload the configuration"""
        self._load_config()
        if self.logger:
            self.logger.info("📊 TierManager config reloaded")

    @property
    def symbols(self) -> List[str]:
        """All symbols"""
        return list(self._states.keys())

    @property
    def symbol_count(self) -> int:
        """Total number of symbols"""
        return len(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._states


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'TierLevel',
    'SymbolTierState',
    'TierManager',
]
