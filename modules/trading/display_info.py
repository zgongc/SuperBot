#!/usr/bin/env python3
"""
modules/trading/display_info.py
SuperBot - Display Info Manager

Retrieves data from TierManager, formats it, and displays it.
TierManager only holds data; the display logic is here.

Usage:
    display = DisplayInfo(tier_manager, logger)

    # Status line (her saniye)
    print(display.format_status_line())

    # Tier details (verbose)
    print(display.format_tier_details())
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta, timezone
import time

from core.timezone_utils import TimezoneUtils, get_utc_now, get_config_timezone_now
from core.config_engine import get_config
from core.cache_manager import get_cache


class DisplayInfo:
    """
    Display information formatter.

    Retrieves data from TierManager, formats for the terminal.
    Optional: connector (for server time), strategy (for balance)

    Replay Mode: Displays timestamp from Parquet data (instead of system time).
    """

    def __init__(
        self,
        tier_manager: Any,
        logger: Optional[Any] = None,
        config: Optional[Any] = None,
        connector: Optional[Any] = None,
        strategy: Optional[Any] = None,
        positions: Optional[Dict[str, Dict[str, Any]]] = None,
        mode: Optional[Any] = None,
        indicator_manager: Optional[Any] = None
    ):
        """
        Args:
            tier_manager: TierManager instance
            logger: Logger instance
            config: ConfigEngine instance
            connector: BinanceAPI instance (for server time)
            strategy: Strategy instance (for balance, timeframe)
            positions: Dictionary of open positions {symbol: position_dict}
            mode: Trading mode instance (PaperMode, ReplayMode, etc.)
            indicator_manager: IndicatorManager instance (for metadata)
        """
        self.tier_manager = tier_manager
        self.logger = logger
        self.config = config
        self.connector = connector
        self.strategy = strategy
        self._positions = positions  # Reference to TradingEngine._positions
        self.mode = mode  # To get the balance
        self.indicator_manager = indicator_manager  # For BOS/CHoCH metadata

        # Uptime tracking
        self._start_time = time.time()

        # Replay mode tracking
        self._replay_start_index = 0  # Replay start index

        # Fallback values (used if connector/strategy is not available)
        self._balance: float = 10000.00
        self._primary_timeframe: str = "5m"

        # Config
        self._load_config()

    def _load_config(self) -> None:
        """Load settings from config"""
        if not self.config:
            self.status_interval = 15
            self.show_tiers = [0, 1, 2, 3]
            self.verbose_tiers = [0, 1, 2, 3]
            self.max_symbols_per_tier = 10
            return

        # trading.yaml → status_display section (no prefix, merged to root)
        status_config = self.config.get('status_display', {})
        self.status_interval = status_config.get('status_interval', 15)
        self.show_tiers = status_config.get('show_sembols_for_tiers', [0, 1, 2, 3])
        self.verbose_tiers = status_config.get('display_verbose_conditions', [0, 1, 2, 3])
        self.max_symbols_per_tier = status_config.get('max_symbols_display', 10)

    # ========================================================================
    # UPTIME
    # ========================================================================

    @property
    def uptime(self) -> timedelta:
        """Uptime as timedelta"""
        return timedelta(seconds=time.time() - self._start_time)

    @property
    def uptime_str(self) -> str:
        """Formatted uptime string (H:MM:SS)"""
        total_seconds = int(time.time() - self._start_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    @property
    def is_replay_mode(self) -> bool:
        """Check if we're in replay mode"""
        if not self.mode:
            return False
        # Check mode_type or class name
        if hasattr(self.mode, 'mode_type'):
            from modules.trading.modes.base_mode import ModeType
            return self.mode.mode_type == ModeType.REPLAY
        return self.mode.__class__.__name__ == 'ReplayMode'

    def get_replay_candle_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the current candle in replay mode.

        Returns:
            Candle timestamp (datetime) or None
        """
        if not self.is_replay_mode:
            return None

        # ReplayMode._current_candle'dan al
        if hasattr(self.mode, '_current_candle') and self.mode._current_candle:
            # Get the candle of the first symbol
            for symbol, candle in self.mode._current_candle.items():
                if candle and hasattr(candle, 'timestamp'):
                    ts_ms = candle.timestamp
                    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return None

    def get_replay_progress_str(self) -> str:
        """
        Replay mode ilerleme bilgisi

        Returns:
            "String in the format 'Bar 150/5000 (3.0%)'"
        """
        if not self.is_replay_mode:
            return ""

        if hasattr(self.mode, '_current_index') and hasattr(self.mode, '_data'):
            total = sum(len(c) for c in self.mode._data.values())
            current = sum(self.mode._current_index.values())
            pct = (current / total * 100) if total > 0 else 0
            return f"Bar {current}/{total} ({pct:.1f}%)"
        return ""

    def get_replay_speed_str(self) -> str:
        """Replay mode speed information"""
        if not self.is_replay_mode:
            return ""

        if hasattr(self.mode, '_speed'):
            return f"{self.mode._speed}x"
        return "1x"

    # ========================================================================
    # TIME HELPERS (Uses TimezoneUtils - gets UTC offset from config)
    # ========================================================================

    def get_server_time_utc(self) -> datetime:
        """
        Get the server time (UTC, timezone-aware).
        Connector varsa senkronize server time, yoksa local UTC.
        """
        if self.connector and hasattr(self.connector, 'get_synced_server_time'):
            return self.connector.get_synced_server_time()
        return get_utc_now()

    def get_local_time(self) -> datetime:
        """
        Get the local time (get the UTC offset from the config using TimezoneUtils).
        Connector varsa server time + offset, yoksa TimezoneUtils.now()
        """
        if self.connector and hasattr(self.connector, 'get_synced_server_time'):
            server_utc = self.connector.get_synced_server_time()
            return TimezoneUtils.to_timezone(server_utc)
        return TimezoneUtils.now()

    def get_next_candle_time(self, timeframe: str = "5m") -> datetime:
        """
        Next candle closing time (Binance epoch-aligned).

        Binance candles starting from the epoch:
        - 1m: Beginning of each minute (00:00, 00:01, 00:02...)
        - 5m: Every 5 minutes (00:00, 00:05, 00:10...)
        """
        server_utc = self.get_server_time_utc()
        tf_seconds = self._timeframe_to_seconds(timeframe)

        if tf_seconds <= 0:
            return TimezoneUtils.to_timezone(server_utc + timedelta(minutes=5))

        # Binance epoch-aligned calculation
        current_ts = server_utc.timestamp()
        current_candle_start = (int(current_ts) // tf_seconds) * tf_seconds
        next_candle_close_ts = current_candle_start + tf_seconds

        # Convert back to datetime (UTC) and apply timezone via TimezoneUtils
        next_close_utc = datetime.fromtimestamp(next_candle_close_ts, tz=timezone.utc)
        return TimezoneUtils.to_timezone(next_close_utc)

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert the timeframe string to seconds"""
        try:
            if timeframe.endswith('m'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 3600
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 86400
            elif timeframe.endswith('w'):
                return int(timeframe[:-1]) * 604800
            else:
                return 300  # Default 5m
        except (ValueError, IndexError):
            return 300

    # ========================================================================
    # SETTERS (to update fallback values)
    # ========================================================================

    def set_balance(self, balance: float) -> None:
        """Update balance (fallback)"""
        self._balance = balance

    def set_timeframe(self, timeframe: str) -> None:
        """Update primary timeframe (fallback)"""
        self._primary_timeframe = timeframe

    # ========================================================================
    # PROPERTY GETTERS (strategy/connector varsa oradan, yoksa fallback)
    # ========================================================================

    @property
    def balance(self) -> float:
        """
        Get the balance.
        If a mode exists, get it from there (updated), otherwise from the strategy, otherwise from the fallback.

        V5 FIX: Show available balance (after deducting margin from position)
        """
        # From Mode, get the current available balance (after deducting the margin in the position)
        if self.mode and hasattr(self.mode, '_available_balance'):
            return self.mode._available_balance

        if self.strategy:
            # Paper mode balance
            if hasattr(self.strategy, 'paper_balance'):
                return self.strategy.paper_balance
            # Initial balance (strategy template)
            if hasattr(self.strategy, 'initial_balance'):
                return self.strategy.initial_balance
            # Or, general balance
            if hasattr(self.strategy, 'balance'):
                return self.strategy.balance
        return self._balance

    @property
    def primary_timeframe(self) -> str:
        """
        Primary timeframe al.
        Strategy varsa oradan, yoksa fallback.
        """
        if self.strategy and hasattr(self.strategy, 'primary_timeframe'):
            return self.strategy.primary_timeframe or self._primary_timeframe
        return self._primary_timeframe

    # ========================================================================
    # FORMAT METHODS
    # ========================================================================

    def format_status_line(self) -> str:
        """
        Main status line format.

        Live/Paper mode:
            "⏱️ Uptime: 0:06:01 │ UTC+3: 22:41:41 │ Next 5m: 22:45:00 │ 💰 Balance: $10,000.00"

        Replay mode:
            "🎬 REPLAY │ 2024-01-15 14:30 │ Bar 150/5000 (3.0%) │ Speed: 2x │ 💰 Balance: $10,000.00"
        """
        # Different format for replay mode
        if self.is_replay_mode:
            return self._format_replay_status_line()

        # Normal mode (live/paper)
        local_time = self.get_local_time()
        tf = self.primary_timeframe
        next_candle = self.get_next_candle_time(tf)

        # UTC offset'i TimezoneUtils'den al (config'den okur)
        utc_offset = self._get_utc_offset()

        parts = [
            f"⏱️ Uptime: {self.uptime_str}",
            f"UTC+{utc_offset}: {local_time.strftime('%H:%M:%S')}",
            f"Next {tf}: {next_candle.strftime('%H:%M:%S')}",
            f"💰 Balance: ${self.balance:,.2f}"
        ]

        return " │ ".join(parts)

    def _format_replay_status_line(self) -> str:
        """
        Status line for replay mode.

        Returns:
            "🎬 REPLAY │ 2024-01-15 14:30 │ Bar 150/5000 (3.0%) │ Speed: 2x │ 💰 Balance: $10,000.00"
        """
        candle_time = self.get_replay_candle_time()
        progress = self.get_replay_progress_str()
        speed = self.get_replay_speed_str()

        # Candle time (from parquet)
        if candle_time:
            # UTC offset uygula
            utc_offset = self._get_utc_offset()
            local_candle_time = candle_time + timedelta(hours=utc_offset)
            time_str = local_candle_time.strftime('%Y-%m-%d %H:%M')
        else:
            time_str = "Loading..."

        # Pause state
        paused = ""
        if hasattr(self.mode, '_paused') and self.mode._paused:
            paused = " ⏸️"

        parts = [
            f"🎬 REPLAY{paused}",
            time_str,
            progress,
            f"Speed: {speed}",
            f"💰 Balance: ${self.balance:,.2f}"
        ]

        return " │ ".join(parts)

    def _get_utc_offset(self) -> int:
        """Config'den UTC offset al (system.utc_offset)"""
        try:
            cfg = get_config()
            return cfg.get("system", {}).get("utc_offset", 3)
        except Exception:
            return 3  # Default UTC+3

    def format_conditions_verbose(self) -> List[str]:
        """
        Verbose condition details - for tiers in the config.

        Config: status_display.display_verbose_conditions = [0, 1, 2, 3]
        0 = POSITION, 1 = DECISION, 2 = MONITORING, 3 = ANALYSIS

        Returns:
            List of condition detail lines
        """
        if not self.tier_manager:
            return []

        lines = []
        symbols_to_show = []

        # Get which tiers to display from the config
        # verbose_tiers = [0, 1, 2, 3] like (TierLevel.value values)
        from modules.trading.tier_manager import TierLevel

        # Tier value → (getter_method, tier_name) mapping
        tier_map = {
            0: (self.tier_manager.get_positions, 'POSITION'),
            1: (self.tier_manager.get_decisions, 'DECISION'),
            2: (self.tier_manager.get_monitoring, 'MONITORING'),
            3: (self.tier_manager.get_analysis, 'ANALYSIS'),
        }

        # Process the tiers in the config in order (priority: lower value = higher priority)
        for tier_value in sorted(self.verbose_tiers):
            if tier_value not in tier_map:
                continue

            getter, tier_name = tier_map[tier_value]
            symbols = getter()

            for sym in symbols:
                state = self.tier_manager.get_state(sym)
                if state and (state.conditions_long or state.conditions_short):
                    symbols_to_show.append((tier_name, state))

        if not symbols_to_show:
            return []

        lines.append("=" * 60)
        lines.append("📋 VERBOSE: Condition Details")
        lines.append("=" * 60)
        lines.append("")

        for tier_name, state in symbols_to_show[:self.max_symbols_per_tier]:
            # Header line
            direction = state.direction or "N/A"
            score_pct = int(state.score * 100)

            # Count met conditions
            long_met = sum(1 for c in state.conditions_long if c.get('met', False)) if state.conditions_long else 0
            long_total = len(state.conditions_long) if state.conditions_long else 0
            short_met = sum(1 for c in state.conditions_short if c.get('met', False)) if state.conditions_short else 0
            short_total = len(state.conditions_short) if state.conditions_short else 0

            # Use the direction's count for display
            if direction == "LONG":
                met_count, total_count = long_met, long_total
            elif direction == "SHORT":
                met_count, total_count = short_met, short_total
            else:
                met_count = max(long_met, short_met)
                total_count = max(long_total, short_total)

            lines.append(f"    🔍 {state.symbol} ({tier_name}) - {direction} - {met_count}/{total_count} ({score_pct}%)")

            # Show only the conditions of the active direction
            # Current close = current price from cache (open candle)
            conditions = state.conditions_long if direction == "LONG" else state.conditions_short
            current_close = self._get_current_price(state.symbol)

            if conditions:
                for cond in conditions:
                    lines.append(self._format_condition_line(cond, symbol=state.symbol, current_close=current_close))

            lines.append("")

        return lines

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for the symbol (from cache - open candle).

        Args:
            symbol: Symbol name

        Returns:
            Current price or None
        """
        if not symbol:
            return None

        try:
            cache = get_cache()
            cached_price = cache.get(f"price:{symbol}")
            if cached_price:
                return float(cached_price)
        except Exception:
            pass

        return None

    def _format_condition_line(self, cond: Dict[str, Any], symbol: str = None, current_close: float = None) -> str:
        """
        Format the single condition line.

        Args:
            cond: Condition dictionary - SignalValidator format:
                  {'condition': ['close', '<', 'ema_5'], 'met': True, 'left_value': 899.51, 'right_value': 899.09}
            symbol: Symbol name (for BOS/CHoCH metadata)
            current_close: Current close price (for BOS hint)

        Returns:
            Formatted condition line like:
            "      ✅ ['close', '<', 'ema_5']: 899.510000 < 899.090000"
            "      ❌ ['bos', '==', 1]: 0 == 1 (350.30 > 352.50 for bullish)"
        """
        met = cond.get('met', False)
        condition = cond.get('condition', [])
        left_value = cond.get('left_value')
        right_value = cond.get('right_value')
        error = cond.get('error')

        # Emoji based on met status
        emoji = "✅" if met else "❌"

        # In case of an error, use a custom format
        if error:
            return f"      ⚠️ {condition}: ERROR - {error}"

        # None value check
        if left_value is None or right_value is None:
            return f"      ⚠️ {condition}: N/A (indicator data missing)"

        # Operator from condition
        operator = condition[1] if len(condition) >= 2 else "?"

        # Use the current price under certain conditions (from cache)
        display_left = left_value
        if len(condition) >= 1 and str(condition[0]).lower() == 'close' and current_close:
            display_left = current_close

        # Base format
        line = f"      {emoji} {condition}: {display_left:.6f} {operator} {right_value:.6f}"

        # Add hint for conditions
        hint = self._get_condition_hint(condition, left_value, right_value, symbol=symbol, met=met, current_close=current_close)
        if hint:
            line += f"  ({hint})"

        return line

    def _get_condition_hint(self, condition: list, left_value: float, right_value: float, symbol: str = None, met: bool = False, current_close: float = None) -> str:
        """
        Generate a hint for the condition.

        Shows the target/broken level for signals like BOS, CHoCH.
        Shows the difference for numeric conditions that are not met.

        Args:
            condition: A list of conditions like ['bos', '==', 1]
            left_value: Left value (indicator value)
            right_value: Right value (target value)
            symbol: Symbol name (for metadata)
            met: Whether the condition is met
            current_close: Current close price (extracted from conditions)

        Returns:
            A hint string or an empty string.
        """
        if len(condition) < 3:
            return ""

        left_name = str(condition[0]).lower()
        operator = condition[1]
        target = condition[2]

        # BOS/CHoCH signal conditions - show swing levels
        if left_name in ('bos', 'choch') and operator == '==' and target in (1, -1, '1', '-1'):
            target_val = int(target)
            swing_level = self._get_swing_level(symbol, left_name, target_val)

            if met:
                # Condition met - shows the broken level
                if swing_level and current_close:
                    if target_val == 1:
                        return f"{current_close:.2f} > {swing_level:.2f} ✓"
                    elif target_val == -1:
                        return f"{current_close:.2f} < {swing_level:.2f} ✓"
                elif swing_level:
                    # There is no close, but there is swing.
                    return f"broke {swing_level:.2f} ✓"
            else:
                # Condition not met - show the target level
                if swing_level and current_close:
                    if target_val == 1:
                        return f"{current_close:.2f} > {swing_level:.2f} for bullish"
                    elif target_val == -1:
                        return f"{current_close:.2f} < {swing_level:.2f} for bearish"
                elif swing_level:
                    # There is no close, but there is swing.
                    if target_val == 1:
                        return f"need close > {swing_level:.2f}"
                    elif target_val == -1:
                        return f"need close < {swing_level:.2f}"
                else:
                    if target_val == 1:
                        return "close > swing_high for bullish"
                    elif target_val == -1:
                        return "close < swing_low for bearish"
            return ""

        # Numeric comparison hints (only when not met)
        if not met:
            if operator in ('>', '>=') and isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                if left_value < right_value:
                    diff = right_value - left_value
                    pct = (diff / right_value * 100) if right_value != 0 else 0
                    return f"need +{diff:.2f} ({pct:.1f}%)"

            if operator in ('<', '<=') and isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                if left_value > right_value:
                    diff = left_value - right_value
                    pct = (diff / left_value * 100) if left_value != 0 else 0
                    return f"need -{diff:.2f} ({pct:.1f}%)"

        return ""

    def _get_swing_level(self, symbol: str, indicator_name: str, target: int) -> Optional[float]:
        """
        Get the swing level for BOS/CHoCH.

        Args:
            symbol: Symbol name
            indicator_name: 'bos' or 'choch'
            target: 1 (bullish) or -1 (bearish)

        Returns:
            Swing level or None
        """
        if not self.indicator_manager or not symbol:
            return None

        try:
            calculators = self.indicator_manager.calculators.get(symbol, {})
            if not calculators:
                # DEBUG: No calculators for symbol
                if self.logger:
                    self.logger.debug(f"_get_swing_level: No calculators for {symbol}")
                return None

            calculator = calculators.get(indicator_name)
            if not calculator:
                # DEBUG: No calculator for indicator
                if self.logger:
                    self.logger.debug(f"_get_swing_level: No calculator '{indicator_name}' for {symbol}, available: {list(calculators.keys())}")
                return None

            if not calculator.last_result or not calculator.last_result.metadata:
                # DEBUG: No result/metadata
                if self.logger:
                    self.logger.debug(f"_get_swing_level: No last_result/metadata for {indicator_name}@{symbol}")
                return None

            metadata = calculator.last_result.metadata

            if target == 1:  # Bullish - need swing_high
                swing_highs = metadata.get('swing_highs', [])
                if swing_highs:
                    return swing_highs[-1].get('level')
                elif self.logger:
                    self.logger.debug(f"_get_swing_level: No swing_highs in metadata for {indicator_name}@{symbol}")
            elif target == -1:  # Bearish - need swing_low
                swing_lows = metadata.get('swing_lows', [])
                if swing_lows:
                    return swing_lows[-1].get('level')
                elif self.logger:
                    self.logger.debug(f"_get_swing_level: No swing_lows in metadata for {indicator_name}@{symbol}")
        except Exception as e:
            if self.logger:
                self.logger.debug(f"_get_swing_level error: {e}")

        return None

    def format_tier_summary(self, verbose: bool = False) -> List[str]:
        """
        Tier summary - each tier is a line (returned as a list)

        Args:
            verbose: If True, also display symbols (according to the show_sembols_for_tiers config).

        Returns:
            List of tier summary lines
        """
        if not self.tier_manager:
            return ["📊 TIERS: N/A"]

        lines = []

        summary = self.tier_manager.get_summary()
        counts = summary['counts']

        # Tier descriptions
        tier_info = [
            (0, "💼", "POSITION", "There is an active position"),
            (1, "🎯", "   DECISION", "100% condition, waiting for candle close"),
            (2, "👀", "  WATCH", "%50+ condition, being watched"),
            (3, "🔍", "  ANALYSIS", "symbol is being analyzed"),
        ]

        for tier_val, emoji, name, desc in tier_info:
            count = counts.get(tier_val, 0)

            # TIER 3 specific format (no symbol list)
            if tier_val == 3:
                lines.append(f"{emoji} TIER {tier_val} - {name} - {count} {desc}")
                continue

            # Other tiers: if count exists, include it in parentheses.
            if count > 0:
                lines.append(f"{emoji} TIER {tier_val} - {name} ({count}): {desc}")
            else:
                lines.append(f"{emoji} TIER {tier_val} - {name} ({count}) {desc}")

            # Show symbols in verbose mode
            if verbose and count > 0 and tier_val in self.show_tiers:
                tier_symbols = self.tier_manager.get_by_tier(tier_val)
                for sym in tier_symbols[:self.max_symbols_per_tier]:
                    state = self.tier_manager.get_state(sym)
                    if state:
                        symbol_lines = self._format_symbol_line(tier_val, state)
                        lines.extend(symbol_lines)

                if count > self.max_symbols_per_tier:
                    lines.append(f"   ... and {count - self.max_symbols_per_tier} more symbols")

        # Total line
        lines.append(f"📈 Total: {summary['total']} symbols │ Tier change: {summary['changes']}")

        return lines

    def _format_symbol_line(self, tier: int, state) -> List[str]:
        """
        Format the symbol line according to the tier.

        Args:
            tier: Tier value (0-3)
            state: SymbolTierState

        Returns:
            List of formatted lines (TIER 0 has multiple lines per symbol)
        """
        symbol = state.symbol
        score = state.score
        direction = state.direction or ""

        if tier == 0:  # POSITION - Multi-line format
            return self._format_position_lines(state)

        elif tier == 1:  # DECISION
            # ⏳ ETHUSDT (100%) - SHORT, expecting a mum close.
            return [f"   ⏳ {symbol} ({score*100:.0f}%) - {direction} candle closing is expected"]

        elif tier == 2:  # MONITORING
            # 📊 AVAXUSDT (66%) - The conditions for a SHORT position are becoming ripe.
            return [f"   📊 {symbol} ({score*100:.0f}%) - {direction} conditions are maturing"]

        else:
            # Generic format
            return [f"   • {symbol} ({score*100:.0f}%) [{direction}]"]

    def _format_position_lines(self, state) -> List[str]:
        """
        Detailed position information for TIER 0 (POSITION)

        Args:
            state: SymbolTierState

        Returns:
            List of formatted lines for position display
        """
        from datetime import datetime

        symbol = state.symbol
        direction = state.direction or "LONG"

        # Get real data from TradingEngine._positions
        position = self._positions.get(symbol) if self._positions else None

        if position:
            # Actual position data
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)
            sl_price = position.get('sl_price', 0)
            tp_price = position.get('tp_price', 0)
            entry_time = position.get('entry_time')
            side = position.get('side', direction)
            leverage = getattr(self.strategy, 'leverage', 10) if self.strategy else 10

            # Current price - Get from the global cache (updated by the TradingEngine)
            current_price = entry_price  # Default fallback

            cache = get_cache()
            cached_price = cache.get(f"price:{symbol}")
            if cached_price:
                current_price = float(cached_price)

            # PnL hesapla
            if side == 'LONG':
                pnl_usd = (current_price - entry_price) * quantity
            else:
                pnl_usd = (entry_price - current_price) * quantity

            notional = entry_price * quantity
            margin = notional / leverage
            pnl_percent = (pnl_usd / notional) * 100 if notional > 0 else 0

            # SL/TP percentages
            sl_percent = ((sl_price - entry_price) / entry_price) * 100 if sl_price and entry_price else 0
            tp_percent = ((tp_price - entry_price) / entry_price) * 100 if tp_price and entry_price else 0

            # Duration
            if entry_time:
                duration_delta = datetime.now() - entry_time
                total_seconds = int(duration_delta.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                duration = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                duration = "0:00:00"
        else:
            # Fallback - use direction from state
            return [f"   ⏳ {symbol} ({direction}) - position data is being retrieved..."]

        # Direction emoji
        dir_emoji = "📉" if side == "SHORT" else "📈"

        # PnL emoji
        pnl_emoji = "🔴" if pnl_percent < 0 else "🟢"

        # Dynamic price format (for coins with small prices)
        def fmt_price(price: float) -> str:
            """Decimal format suitable for price"""
            if price >= 1000:
                return f"{price:,.2f}"
            elif price >= 1:
                return f"{price:.4f}"
            elif price >= 0.01:
                return f"{price:.6f}"
            else:
                return f"{price:.8f}"

        entry_fmt = fmt_price(entry_price)
        now_fmt = fmt_price(current_price)
        sl_fmt = fmt_price(sl_price) if sl_price else "N/A"
        tp_fmt = fmt_price(tp_price) if tp_price else "N/A"

        lines = [
            f"   {dir_emoji} {symbol:<10} │ {side:<5} │ Entry: ${entry_fmt:<12} │ Now: ${now_fmt:<12} │ {pnl_emoji} PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})",
            f"      💰 Size: {quantity:.4f} ({leverage}x) │ Margin: ${margin:.2f} │ Notional: ${notional:.2f}",
            f"      🛑 SL: ${sl_fmt} ({sl_percent:+.2f}%) │ 🎯 TP: ${tp_fmt} ({tp_percent:+.2f}%) │ ⏱️ Duration: {duration}"
        ]

        return lines

    def format_tier_details(self, limit: Optional[int] = None) -> str:
        """
        Detailed tier display

        Args:
            limit: Maximum number of symbols for each tier (None = from config)

        Returns:
            Multi-line tier details
        """
        if not self.tier_manager:
            return "📊 TierManager not available"

        if limit is None:
            limit = self.max_symbols_per_tier

        lines = []

        # For each tier
        from modules.trading.tier_manager import TierLevel

        for tier in TierLevel:
            # Tiers to be displayed in the config
            if tier.value not in self.show_tiers:
                continue

            symbols = self.tier_manager.get_by_tier(tier)
            count = len(symbols)

            if count == 0:
                continue

            lines.append(f"{tier.emoji} TIER {tier.value} - {tier.name_tr} ({count}):")

            for sym in symbols[:limit]:
                state = self.tier_manager.get_state(sym)
                if not state:
                    continue

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

        return "\n".join(lines) if lines else "📊 All symbols are in ANALYSIS"

    def format_full_display(self, verbose: bool = False) -> str:
        """
        Tam display (status line + tier info)

        Args:
            verbose: Detailed tier display

        Returns:
            Multi-line full display
        """
        lines = [
            "─" * 60,
            self.format_status_line(),
        ]

        if verbose:
            lines.append("")
            lines.append(self.format_tier_details())
        else:
            lines.append(self.format_tier_summary())

        return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ['DisplayInfo']
