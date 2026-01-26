#!/usr/bin/env python3
"""
modules/trading/trading_engine.py
SuperBot - Trading Engine V5
Author: SuperBot Team
Date: 2025-12-01
Versiyon: 5.0.0

Trading Engine - Main Orchestrator

Features:
- Minimal, temiz mimari
- 4 Core singleton (logger, config, event_bus, cache)
- Step-by-step initialization
- Tier-based processing

Usage:
    from modules.trading.trading_engine import TradingEngine

    engine = TradingEngine(mode="paper", strategy_path="ema5_bb_adx.py")
    await engine.start()

Dependencies:
    - python>=3.12
    - core/logger_engine
    - core/config_engine
    - core/event_bus
    - core/cache_manager
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

# ════════════════════════════════════════════════════════════════════════════
# CORE IMPORTS (Singleton pattern)
# ════════════════════════════════════════════════════════════════════════════
from core.logger_engine import get_logger
from core.config_engine import get_config
from core.event_bus import get_event_bus
from core.cache_manager import get_cache

# Mode base class (for type hints)
from modules.trading.modes.base_mode import BaseMode

# ════════════════════════════════════════════════════════════════════════════
# COMPONENT IMPORTS
# ════════════════════════════════════════════════════════════════════════════
from core.timezone_utils import TimezoneUtils

# Strategy
from components.strategies.strategy_manager import StrategyManager

# Managers
from components.managers.risk_manager import RiskManager
from components.managers.symbols_manager import SymbolsManager

# Indicators (bridge helper will be used)
from components.indicators.indicator_manager import IndicatorManager
from components.strategies.helpers.strategy_indicator_bridge import create_indicator_manager_from_strategy

# Exchange
from components.exchanges.binance_api import BinanceAPI

# Trading V5 Components
from modules.trading.tier_manager import TierManager, TierLevel
from modules.trading.price_feed import PriceFeed
from modules.trading.display_info import DisplayInfo

# Exit management
from components.strategies.exit_manager import ExitManager


class TradingEngine:
    """
    Trading Engine V5 - Main Orchestrator

    SORUMLULUKLAR:
    ✅ Component orchestration
    ✅ Mode management (paper/live)
    ✅ Tier-based processing
    ✅ Event-driven candle processing

    KULLANILAN 4 CORE SINGLETON:
    - logger: Logging
    - config: Configuration
    - event_bus: Event system
    - cache: CacheManager
    """

    def __init__(
        self,
        mode: str = "paper",
        strategy_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """
        Create a Trading Engine.

        Args:
            mode: Trading mode (paper/live)
            strategy_path: Strategy template path
            config: Override config (optional)
            verbose: Detailed output
        """
        # ════════════════════════════════════════════════════════════════════
        # 4 CORE SINGLETONS
        # ════════════════════════════════════════════════════════════════════
        self.logger = get_logger("modules.trading.trading_engine")
        self.config = config or get_config()
        self.event_bus = get_event_bus()
        self.cache = get_cache()

        # ════════════════════════════════════════════════════════════════════
        # PARAMETERS
        # ════════════════════════════════════════════════════════════════════
        self.mode_name = mode
        self.strategy_path = strategy_path
        self.verbose = verbose

        # ════════════════════════════════════════════════════════════════════
        # STATE
        # ════════════════════════════════════════════════════════════════════
        self._running = False
        self._initialized = False

        # ════════════════════════════════════════════════════════════════════
        # COMPONENTS (lazy init)
        # ════════════════════════════════════════════════════════════════════
        self.current_mode: Optional[BaseMode] = None
        self.connector: Optional[BinanceAPI] = None
        self.strategy = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.indicator_manager: Optional[IndicatorManager] = None
        self.symbols_manager: Optional[SymbolsManager] = None
        self.symbols: List[str] = []  # Trading sembolleri

        # Data feeds (lazy init in _start_data_feeds)
        self.websocket_engine = None
        self.mtf_engines: Dict[str, Any] = {}

        # V5 Components (lazy init)
        self.tier_manager: Optional[TierManager] = None
        self.price_feed: Optional[PriceFeed] = None
        self.display_info: Optional[DisplayInfo] = None

        # Position Management (Paper Mode)
        self._positions: Dict[str, Dict[str, Any]] = {}  # {symbol: position_dict}
        self._trade_counter = 0
        self._exit_manager: Optional[ExitManager] = None

        # Candle close tracking (last processed candle timestamp for each symbol)
        self._last_processed_candle: Dict[str, int] = {}  # {symbol: timestamp}

        # V5 FIX: Pending decisions cache - Store data in the signal for 100%, use it when the candle closes.
        # This prevents momentary signals, such as those from BOS, from being lost.
        self._pending_decisions: Dict[str, Dict[str, Any]] = {}  # {symbol: {side, data, timestamp}}

        self.logger.info(f"🤖 TradingEngine V5 created (mode: {mode}, verbose: {verbose})")

    # ════════════════════════════════════════════════════════════════════════════
    # STRATEGY LOADING
    # ════════════════════════════════════════════════════════════════════════════

    async def _load_strategy(self) -> Any:
        """
        Load strategy template

        StrategyManager path normalization yapar:
        - "grok_scalp" → components/strategies/templates/grok_scalp.py
        - "grok_scalp.py" → components/strategies/templates/grok_scalp.py
        - Full path → as-is

        BaseStrategy automatically:
        - primary_timeframe'i mtf_timeframes'e ekler

        Returns:
            Strategy: The loaded strategy instance.

        Raises:
            ValueError: If the strategy path is not specified.
        """
        if not self.strategy_path:
            raise ValueError("Strategy path belirtilmedi!")

        # Load with StrategyManager (including path normalization)
        self.strategy_manager = StrategyManager(logger=self.logger)
        strategy, executor = self.strategy_manager.load_strategy(self.strategy_path)

        # Store the executor
        self._strategy_executor = executor

        self.logger.info(f"✅ Strategy loaded: {strategy.strategy_name} v{strategy.strategy_version}")

        return strategy

    # ════════════════════════════════════════════════════════════════════════════
    # SYMBOL LOADING
    # ════════════════════════════════════════════════════════════════════════════

    async def _load_symbols(self) -> List[str]:
        """
        Trading sembollerini SymbolsManager'dan al

        Creates and initializes the SymbolsManager if it doesn't exist.
        Source information is retrieved from strategy.symbol_source.

        SymbolsManager manages all sources:
        - config: From the Config file.
        - strategy: Strategy template'den
        - file: from a JSON file
        - exchange: Exchange API'den

        Normalization is performed in the SymbolsManager:
        - ETH → ETHUSDT
        - ETHUSDT → ETHUSDT
        - ETHUSDTUSDT -> ETHUSDT (fix for double quotes)

        Returns:
            List[str]: Normalized symbol list.
        """
        # If SymbolsManager does not exist, create it.
        if not self.symbols_manager:
            symbol_source = getattr(self.strategy, 'symbol_source', 'strategy')

            # Config'e source'u yaz
            if hasattr(self.config, 'set'):
                self.config.set('symbols.source', symbol_source)

            self.symbols_manager = SymbolsManager(
                config=self.config,
                exchange_client=self.connector,
                cache_manager=self.cache,
                logger=self.logger,
                strategy=self.strategy,
                strategy_file=self.strategy_path  # File name (e.g., grok_scalp_king_5m_v2.py)
            )
            await self.symbols_manager.initialize()

        symbols = self.symbols_manager.get_active_symbols()

        if not symbols:
            self.logger.warning("⚠️ Could not retrieve symbol from SymbolsManager, using fallback")
            return ["BTCUSDT", "ETHUSDT"]

        self.logger.info(f"📊 {len(symbols)} symbols loaded: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
        return symbols

    # ════════════════════════════════════════════════════════════════════════════
    # MODE LOADING
    # ════════════════════════════════════════════════════════════════════════════

    AVAILABLE_MODES = ['paper', 'demo', 'live', 'replay']

    async def _load_mode(self, mode_name: str) -> BaseMode:
        """
        Load mode

        Connector and Strategy must be loaded FIRST!
        Mode uses them, it doesn't create them itself.

        Connector selection:
        - replay: None (works from parquet)
        - demo: testnet=True
        - paper/live: testnet=False (production)

        Args:
            mode_name: Mode name (paper/demo/live/replay)

        Returns:
            BaseMode: Mode instance

        Raises:
            ValueError: Invalid mode
        """
        if mode_name not in self.AVAILABLE_MODES:
            raise ValueError(
                f"Invalid mode: {mode_name}. "
                f"Valid modes: {', '.join(self.AVAILABLE_MODES)}"
            )

        self.logger.info(f"🎮 Loading mode: {mode_name}")

        # Prepare mode configuration (connector, strategy, etc.)
        mode_config = {
            "connector": self.connector,
            "strategy": self.strategy,
            "cache_manager": self.cache,
            "event_bus": self.event_bus,
            "indicator_manager": self.indicator_manager,
            "risk_manager": self.risk_manager,
            "symbols_manager": self.symbols_manager,
            **self.config.get(f"trading.modes.{mode_name}", {})
        }

        # Dynamic import and instantiation
        if mode_name == 'paper':
            from modules.trading.modes.paper_mode import PaperMode
            mode = PaperMode(mode_config, self.logger)

        elif mode_name == 'demo':
            from modules.trading.modes.demo_mode import DemoMode
            mode = DemoMode(mode_config, self.logger)

        elif mode_name == 'live':
            from modules.trading.modes.live_mode import LiveMode
            mode = LiveMode(mode_config, self.logger)

        elif mode_name == 'replay':
            from modules.trading.modes.replay_mode import ReplayMode
            mode = ReplayMode(mode_config, self.logger)

        # Initialize mode
        await mode.initialize()

        return mode

    # ════════════════════════════════════════════════════════════════════════════
    # V5 COMPONENTS INITIALIZATION
    # ════════════════════════════════════════════════════════════════════════════

    async def _initialize_v5_components(self) -> None:
        """
        V5 Components: TierManager + PriceFeed + DisplayInfo

        For basic visibility:
        - PriceFeed: Cached price access (EventBus subscription)
        - TierManager: Symbol tier tracking
        - DisplayInfo: Terminal output formatting
        """
        self.logger.info("🔧 V5 Components are being started...")

        # ════════════════════════════════════════════════════════════════
        # 1. PriceFeed (Automatically subscribes to the EventBus)
        # ════════════════════════════════════════════════════════════════
        self.price_feed = PriceFeed(
            cache_manager=self.cache,
            event_bus=self.event_bus,
            logger=get_logger("modules.trading.price_feed")
        )
        self.logger.info("   ✅ PriceFeed is ready")

        # ════════════════════════════════════════════════════════════════
        # 2. TierManager (initialize all symbols in the ANALYSIS tier)
        # ════════════════════════════════════════════════════════════════
        # Pass the full config (for status_display.show_tier_changes)
        self.tier_manager = TierManager(
            config=self.config,
            cache_manager=self.cache,
            event_bus=self.event_bus,
            logger=get_logger("modules.trading.tier_manager")
        )

        # Add symbols to the TierManager (starts in the ANALYSIS tier)
        for symbol in self.symbols:
            self.tier_manager.set_tier(symbol, TierLevel.ANALYSIS)

        self.logger.info(f"   ✅ TierManager ready ({len(self.symbols)} symbols)")

        # ════════════════════════════════════════════════════════════════
        # 3. DisplayInfo (TierManager + Strategy + Connector + Positions)
        # ════════════════════════════════════════════════════════════════
        self.display_info = DisplayInfo(
            tier_manager=self.tier_manager,
            logger=get_logger("modules.trading.display_info"),
            config=self.config,
            connector=self.connector,
            strategy=self.strategy,
            positions=self._positions,  # Live position tracking
            mode=self.current_mode,  # For balance
            indicator_manager=self.indicator_manager  # For BOS/CHoCH swing levels
        )
        self.logger.info("   ✅ DisplayInfo ready")

    # ════════════════════════════════════════════════════════════════════════════
    # DATA FEEDS
    # ════════════════════════════════════════════════════════════════════════════

    async def _start_data_feeds(self, symbols: List[str]) -> None:
        """
        Start WebSocket + MTF + Indicator subscription

        Args:
            symbols: Trading symbols (loaded in initialize)

        Flow:
        1. Create a WebSocketEngine.
        2. Subscribe to symbols (bulk).
        3. Subscribe the IndicatorManager to the EventBus.
        4. Create MTF engines (per symbol).
        5. Start the MTF engines (parallel warmup).
        """
        from components.managers.websocket_engine import WebSocketEngine
        from components.managers.multi_timeframe_engine import MultiTimeframeEngine

        self.logger.info(f"📡 Starting data feeds: {len(symbols)} symbols")

        # ════════════════════════════════════════════════════════════════
        # 1. Create a WebSocketEngine
        # ════════════════════════════════════════════════════════════════
        ws_config = self.config.get("websocket", {})
        ws_config["testnet"] = (self.mode_name == 'demo')

        self.websocket_engine = WebSocketEngine(
            config=ws_config,
            event_bus=self.event_bus
        )
        await self.websocket_engine.start()
        self.logger.info("   ✅ WebSocketEngine is ready")

        # ════════════════════════════════════════════════════════════════
        # 2. Subscribe symbols (bulk - ALL timeframes!)
        # ════════════════════════════════════════════════════════════════
        mtf_timeframes = self.strategy.mtf_timeframes  # ['1m', '5m']
        channels = [f"kline_{tf}" for tf in mtf_timeframes]  # ['kline_1m', 'kline_5m']

        await self.websocket_engine.subscribe(
            symbols=symbols,
            channels=channels
        )
        self.logger.info(f"   ✅ {len(symbols)} symbols subscribed ({', '.join(mtf_timeframes)})")

        # ════════════════════════════════════════════════════════════════
        # 3. IndicatorManager EventBus subscription
        # ════════════════════════════════════════════════════════════════
        if self.indicator_manager and hasattr(self.indicator_manager, 'subscribe_to_symbol'):
            for symbol in symbols:
                await self.indicator_manager.subscribe_to_symbol(
                    symbol=symbol,
                    connector=self.connector,
                    auto_warmup=False  # MTF Engine handles warmup
                )
            self.logger.info(f"   ✅ IndicatorManager subscribed ({len(symbols)} symbols)")

        # ════════════════════════════════════════════════════════════════
        # 4. Create MTF Engines (per symbol)
        # ════════════════════════════════════════════════════════════════
        # V7: Use the Strategy's warmup_period value (prioritize over auto-calculate)
        # Max of: strategy.warmup_period, auto-calculated, config default
        strategy_warmup = getattr(self.strategy, 'warmup_period', 200)
        config_warmup = self.config.get("trading.strategy.warmup_period", 200)
        auto_warmup = 200  # default
        if self.indicator_manager and hasattr(self.indicator_manager, '_calculate_warmup_period'):
            auto_warmup = self.indicator_manager._calculate_warmup_period()

        warmup_period = max(strategy_warmup, auto_warmup, config_warmup)

        self.mtf_engines: Dict[str, MultiTimeframeEngine] = {}

        for symbol in symbols:
            mtf_engine = MultiTimeframeEngine(
                symbol=symbol,
                timeframes=self.strategy.mtf_timeframes,
                primary_timeframe=self.strategy.primary_timeframe,
                websocket_manager=self.websocket_engine,
                event_bus=self.event_bus,
                connector_engine=self.connector,
                indicator_manager=self.indicator_manager,
                on_candle_closed=self._on_candle_closed,
                warmup_period=warmup_period,
                verbose=self.verbose
            )
            self.mtf_engines[symbol] = mtf_engine

        self.logger.info(f"   ✅ {len(self.mtf_engines)} MTF Engine ready")

        # ════════════════════════════════════════════════════════════════
        # 5. MTF Engines start (parallel warmup)
        # ════════════════════════════════════════════════════════════════
        tasks = [mtf.start() for mtf in self.mtf_engines.values()]
        await asyncio.gather(*tasks)
        self.logger.info(f"   ✅ All MTF Engines started (warmup: {warmup_period} candles)")

        # V4: Initialize _last_processed_candle with warmup's last candle timestamp
        # This way, the first check after warmup is not interpreted as a "new candle".
        for symbol, mtf_engine in self.mtf_engines.items():
            primary_tf = self.strategy.primary_timeframe
            tf_info = mtf_engine.tf_info.get(primary_tf)
            if tf_info and tf_info.candles:
                self._last_processed_candle[symbol] = tf_info.candles[-1].timestamp

    async def _on_candle_closed(self, symbol: str, timeframe: str) -> None:
        """
        Called when a candle closes (MTF Engine -> TradingEngine -> Mode)

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "5m")
        """
        if not self.current_mode:
            return

        try:
            # Route to the mode (for strategy evaluation)
            if hasattr(self.current_mode, 'on_candle_closed'):
                await self.current_mode.on_candle_closed(symbol, timeframe)
        except Exception as e:
            self.logger.error(f"❌ {symbol}: Candle callback error: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # TIER PROCESSING
    # ════════════════════════════════════════════════════════════════════════════

    async def _process_tiers(self, loop_count: int) -> None:
        """
        Tier-based condition evaluation

        It runs every second and performs condition checks at different intervals depending on the tier.

        Args:
            loop_count: Main loop counter (for interval calculation)
        """
        if not self.tier_manager:
            return

        # Warmup check - Are MTF engines ready?
        if not self._is_warmup_complete():
            # DEBUG: Show detailed warmup status
            if loop_count % 5 == 0:
                self._debug_warmup_status(loop_count)
            return

        # Get interval values from the config
        intervals_enabled = self.config.get("trading.tiers.intervals.enabled", False)

        if intervals_enabled:
            # Different interval based on tier (optimized for 100+ symbols)
            tier_intervals = {
                TierLevel.POSITION: 1,    # Her saniye (SL/TP kritik)
                TierLevel.DECISION: 5,    # 5 saniyede bir
                TierLevel.MONITORING: 15, # 15 saniyede bir
                TierLevel.ANALYSIS: 60    # 60 saniyede bir
            }
        else:
            # All tiers are processed in each loop (for the small symbol list).
            tier_intervals = {
                TierLevel.POSITION: 1,
                TierLevel.DECISION: 1,
                TierLevel.MONITORING: 1,
                TierLevel.ANALYSIS: 1
            }

        # Get which symbols we need to check from TierManager
        symbols_to_check = self.tier_manager.get_symbols_to_check()

        # exit check for POSITION tier (every second)
        if self._positions:
            await self._check_position_exits()

        # Condition check for each tier
        for tier_level, interval in tier_intervals.items():
            if loop_count % interval == 0:
                symbols = symbols_to_check.get(tier_level, [])
                for symbol in symbols:
                    await self._evaluate_symbol(symbol, tier_level)

    def _debug_warmup_status(self, loop_count: int) -> None:
        """
        DEBUG: Show detailed warmup status.
        """
        return  # <-- CLOSED: Uncomment this line

        self.logger.info(f"⏳ WARMUP STATUS (loop: {loop_count})")

        if not self.mtf_engines:
            self.logger.info("   ❌ No MTF Engines!")
            return

        for symbol, mtf_engine in self.mtf_engines.items():
            running = getattr(mtf_engine, 'running', False)
            primary_tf = getattr(mtf_engine, 'primary_timeframe', '?')
            tf_info = getattr(mtf_engine, 'tf_info', {})

            if primary_tf in tf_info:
                candle_count = len(tf_info[primary_tf].candles)
            else:
                candle_count = 0

            status = "✅" if running and candle_count >= 10 else "❌"
            self.logger.info(f"   {status} {symbol}: running={running}, candles={candle_count}/10, tf={primary_tf}")

        self.logger.info("")

    def _debug_conditions(self, symbol: str, side: str, summary: dict) -> None:
        """
        DEBUG: Log the details of the condition.

        Usage: enable/disable with comment out.
        """
        return  # <-- CLOSED: Uncomment this line

        # DEBUG: summary'nin kendisini logla
        self.logger.info(f"🔍 DEBUG: {symbol} summary keys: {list(summary.keys())}")

        details = summary.get('details', [])
        if not details:
            self.logger.info(f"🔍 {symbol} ({side}): No condition details! summary={summary}")
            return

        met = summary.get('conditions_met', 0)
        total = summary.get('conditions_total', 0)
        score_pct = int(summary.get('score', 0) * 100)

        self.logger.info(f"🔍 {symbol} - {side} - {met}/{total} ({score_pct}%)")

        for cond in details:
            condition = cond.get('condition', '?')
            is_met = cond.get('met', False)
            left_val = cond.get('left_value')
            right_val = cond.get('right_value')
            error = cond.get('error')

            emoji = "✅" if is_met else "❌"

            # condition list ise string yap
            cond_str = str(condition) if isinstance(condition, list) else condition

            if error:
                self.logger.info(f"   {emoji} {cond_str}: ERROR - {error}")
            elif left_val is None or right_val is None:
                left_str = f"{left_val:.6f}" if isinstance(left_val, (int, float)) else str(left_val)
                right_str = f"{right_val:.6f}" if isinstance(right_val, (int, float)) else str(right_val)
                self.logger.info(f"   {emoji} {cond_str}: {left_str} vs {right_str}")
            else:
                # Format the values
                left_str = f"{left_val:.6f}" if isinstance(left_val, (int, float)) else str(left_val)
                right_str = f"{right_val:.6f}" if isinstance(right_val, (int, float)) else str(right_val)
                op = condition[1] if isinstance(condition, list) and len(condition) >= 2 else "?"
                self.logger.info(f"   {emoji} {cond_str}: {left_str} {op} {right_str}")

        self.logger.info("")  # Empty line

    def _is_warmup_complete(self) -> bool:
        """
        Have all MTF engines completed warmup?

        MTF Engine is running and candles exist for the primary timeframe,
        it means the warmup is complete.

        Returns:
            bool: True = warmup is complete, tier processing can begin
        """
        if not self.mtf_engines:
            return False

        for symbol, mtf_engine in self.mtf_engines.items():
            # MTF Engine running check
            if not getattr(mtf_engine, 'running', False):
                return False

            # Candle check for the primary timeframe
            primary_tf = getattr(mtf_engine, 'primary_timeframe', None)
            tf_info = getattr(mtf_engine, 'tf_info', {})

            if primary_tf and primary_tf in tf_info:
                candles = tf_info[primary_tf].candles
                if len(candles) < 10:  # There must be a minimum of 10 candles
                    return False

        return True

    async def _evaluate_symbol(self, symbol: str, current_tier: TierLevel) -> None:
        """
        Condition evaluation for a single symbol.

        Evaluates entry conditions using the SignalValidator.
        Updates the tier based on the score.
        Opens a position if there is a 100% signal in the DECISION tier.

        V5 FIX: %100 sinyalde veriyi _pending_decisions'a cache'le.
        Open a trade using cached data when the candle closes.
        This prevents the loss of momentary signals, such as those from BOS.

        Args:
            symbol: Symbol name
            current_tier: Current tier level
        """
        try:
            # 1. Create DataFrame from MTF Engine
            mtf_engine = self.mtf_engines.get(symbol)
            if not mtf_engine:
                self.logger.debug(f"⚠️ {symbol}: MTF Engine not found")
                return

            # Calculate with the latest data (always use last_result)
            data = self._build_dataframe_from_mtf(mtf_engine, use_previous_result=False)
            if data is None:
                self.logger.debug(f"⚠️ {symbol}: DataFrame could not be created")
                return

            # 2. Evaluate the conditions for both sides using SignalValidator.
            signal_validator = self._strategy_executor.signal_manager
            long_summary = signal_validator.get_conditions_summary(data, 'long')
            short_summary = signal_validator.get_conditions_summary(data, 'short')

            # Determine the best side (for the tier)
            if long_summary['score'] >= short_summary['score']:
                best_summary = long_summary
                best_side = 'LONG'
            else:
                best_summary = short_summary
                best_side = 'SHORT'

            score = best_summary.get('score', 0.0)
            conditions_met = best_summary.get('conditions_met', 0)
            conditions_total = best_summary.get('conditions_total', 0)

            # DEBUG: Show condition details (enable/disable with comment out)
            self._debug_conditions(symbol, best_side, best_summary)

            # 3. Update TierManager (with details of both sides)
            self.tier_manager.update_symbol_state(
                symbol=symbol,
                score=score,
                conditions_met=conditions_met,
                conditions_total=conditions_total,
                conditions_long=long_summary.get('details', []),
                conditions_short=short_summary.get('details', [])
            )

            # 4. Check for new candle closing.
            is_new_candle = self._is_new_candle_closed(symbol, mtf_engine)

            # 5. Entry Signal Check
            new_tier = self.tier_manager.get_tier(symbol)

            # If the position already exists, skip it.
            if symbol in self._positions:
                # If there is a pending decision, clear it.
                if symbol in self._pending_decisions:
                    del self._pending_decisions[symbol]
                return

            # V5 FIX: %100 sinyalde veriyi cache'le
            if score >= 1.0:
                # Get the current candle timestamp
                primary_tf = self.strategy.primary_timeframe
                tf_info = mtf_engine.tf_info.get(primary_tf)
                current_ts = tf_info.candles[-1].timestamp if tf_info and tf_info.candles else 0

                # If the cache does not exist or the parameters are different, create a new cache.
                pending = self._pending_decisions.get(symbol)
                if not pending or pending.get('timestamp') != current_ts:
                    self._pending_decisions[symbol] = {
                        'side': best_side,
                        'data': data,
                        'timestamp': current_ts,
                        'score': score
                    }
                    self.logger.debug(f"📌 {symbol}: 100% signal cached (ts={current_ts}, side={best_side})")

            # V5 FIX: Use cached data when the candle closes.
            if is_new_candle and symbol in self._pending_decisions:
                pending = self._pending_decisions[symbol]
                cached_side = pending['side']
                cached_data = pending['data']

                self.logger.info(f"🎯 {symbol}: Candlestick closed + Cached 100% signal! {cached_side} is opening...")
                await self._open_position(symbol, cached_side, cached_data)

                # Clear cache
                del self._pending_decisions[symbol]

            # If the score drops below 100%, clear the cache.
            elif score < 1.0 and symbol in self._pending_decisions:
                self.logger.debug(f"🗑️ {symbol}: Score decreased, pending decision cleared")
                del self._pending_decisions[symbol]

            # 6. Verbose log
            if self.verbose:
                tier_name = new_tier.name
                pending_mark = " [PENDING]" if symbol in self._pending_decisions else ""
                self.logger.debug(
                    f"📊 {symbol}: L={long_summary['conditions_met']}/{long_summary['conditions_total']} "
                    f"S={short_summary['conditions_met']}/{short_summary['conditions_total']} → {tier_name}{pending_mark}"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} evaluation error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    # ════════════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT (Paper Mode)
    # ════════════════════════════════════════════════════════════════════════════

    async def _open_position(self, symbol: str, side: str, data: Dict[str, Any]) -> None:
        """
        Open position (Paper Mode)

        Follows the BacktestEngine._open_position() pattern:
        - Calculate SL/TP with ExitManager
        - Calculate position size with RiskManager
        - Execute the order with PaperMode.execute_order()

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            data: DataFrame dict (indicators dahil)
        """
        from datetime import datetime
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            # 1. Get the current price
            current_price = await self.current_mode.get_current_price(symbol)
            if not current_price or current_price <= 0:
                self.logger.warning(f"⚠️ {symbol}: Price could not be retrieved, position cannot be opened")
                return

            # 2. Calculate Stop Loss/Take Profit with ExitManager
            if not self._exit_manager:
                self._exit_manager = ExitManager(self.strategy, logger=self.logger)

            # Primary timeframe DataFrame'i al
            primary_tf = self.strategy.primary_timeframe
            primary_df = data.get(primary_tf) if data else None

            sl_price = self._exit_manager.calculate_stop_loss(
                current_price, side, data=primary_df
            )
            tp_price = self._exit_manager.calculate_take_profit(
                current_price, side, stop_loss_price=sl_price, data=primary_df
            )

            # 3. Calculate position size with RiskManager
            balance = await self.current_mode.get_balance()
            portfolio_value = balance.available if balance else 10000.0

            quantity = self.risk_manager.calculate_position_size_from_strategy(
                strategy=self.strategy,
                risk_management=self.strategy.risk_management,
                entry_price=current_price,
                portfolio_value=portfolio_value,
                stop_loss_price=sl_price
            )

            if quantity <= 0:
                self.logger.warning(f"⚠️ {symbol}: Position size 0, position cannot be opened")
                return

            # 4. Create and execute the order
            order_side = OrderSide.BUY if side == 'LONG' else OrderSide.SELL
            order = Order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                stop_loss=sl_price,
                take_profit=tp_price
            )

            result = await self.current_mode.execute_order(order)

            if result.status.value == 'FILLED':
                # 5. Create a position dictionary (for tracking)
                self._trade_counter += 1
                entry_time = datetime.now()

                position = {
                    'id': self._trade_counter,
                    'symbol': symbol,
                    'side': side,
                    'entry_time': entry_time,
                    'entry_price': result.price,
                    'quantity': result.filled_quantity,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'highest_price': result.price,
                    'lowest_price': result.price,
                    'order_id': result.order_id,
                }

                self._positions[symbol] = position

                # 6. Move to the POSITION tier in TierManager.
                self.tier_manager.set_tier(symbol, TierLevel.POSITION)

                # 7. Log entry (kompakt format)
                sl_pct = abs((sl_price - result.price) / result.price) * 100 if sl_price else 0
                tp_pct = abs((tp_price - result.price) / result.price) * 100 if tp_price else 0
                leverage = getattr(self.strategy, 'leverage', 1)
                notional = result.price * result.filled_quantity
                margin = notional / leverage

                side_emoji = "📈" if side == 'LONG' else "📉"
                self.logger.info(f"\n🎯 {side_emoji} {symbol:<10} │ {side} Entry │ @ ${result.price:,.2f}")
                self.logger.info(f"   💰 Size: {result.filled_quantity:.4f} ({leverage}x) │ Margin: ${margin:,.2f} │ Notional: ${notional:,.2f}")
                sl_str = f"🛑 SL: ${sl_price:,.2f} ({sl_pct:.2f}%)" if sl_price else ""
                tp_str = f"🎯 TP: ${tp_price:,.2f} ({tp_pct:.2f}%)" if tp_price else ""
                self.logger.info(f"   {sl_str} │ {tp_str}")

        except Exception as e:
            import traceback
            self.logger.error(f"❌ {symbol}: Error opening position: {e}")
            self.logger.error(traceback.format_exc())

    async def _check_position_exits(self) -> None:
        """
        Check all open positions (SL/TP/Trailing)

        For each POSITION tier symbol:
        - Check exit conditions with StrategyExecutor.evaluate_exit()
        - If SL/TP is hit, close the position
        - Update SL for Trailing/Break-even
        """
        if not self._positions:
            return

        for symbol, position in list(self._positions.items()):
            try:
                # 1. Get the current price
                current_price = await self.current_mode.get_current_price(symbol)
                if not current_price:
                    continue

                # Update highest/lowest (for trailing)
                if position['side'] == 'LONG':
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                else:
                    if current_price < position['lowest_price']:
                        position['lowest_price'] = current_price

                # 2. MTF Engine'den data al
                mtf_engine = self.mtf_engines.get(symbol)
                if not mtf_engine:
                    continue

                data = self._build_dataframe_from_mtf(mtf_engine)

                # 3. Check with StrategyExecutor.evaluate_exit()
                exit_result = self._strategy_executor.evaluate_exit(
                    symbol=symbol,
                    position=position,
                    data=data,
                    current_price=current_price
                )

                # 4. SL update (trailing/break-even)
                if exit_result.get('updated_sl') and exit_result['updated_sl'] != position.get('sl_price'):
                    old_sl = position.get('sl_price')
                    new_sl = exit_result['updated_sl']
                    position['sl_price'] = new_sl
                    position['stop_loss'] = new_sl

                    if exit_result.get('break_even_moved'):
                        self.logger.info(f"🔄 {symbol}: Break-even SL moved: ${old_sl:,.2f} → ${new_sl:,.2f}")
                    else:
                        self.logger.info(f"🔄 {symbol}: Trailing SL updated: ${old_sl:,.2f} → ${new_sl:,.2f}")

                # 5. Exit check
                if exit_result.get('should_exit'):
                    exit_type = exit_result.get('exit_type', 'UNKNOWN')
                    await self._close_position(symbol, position, current_price, exit_type)

            except Exception as e:
                self.logger.debug(f"⚠️ {symbol} exit check error: {e}")

    async def _close_position(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        exit_type: str
    ) -> None:
        """
        Close position (Paper Mode)

        Args:
            symbol: Trading symbol
            position: Position dict
            current_price: Current price (at the time of the exit call)
            exit_type: 'SL', 'TP', 'SIGNAL', etc.
        """
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            # 1. Create order
            close_side = OrderSide.SELL if position['side'] == 'LONG' else OrderSide.BUY
            order = Order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position['quantity']
            )

            result = await self.current_mode.execute_order(order)

            if result.status.value == 'FILLED':
                # 2. PnL hesapla
                entry_price = position['entry_price']
                quantity = position['quantity']

                if position['side'] == 'LONG':
                    gross_pnl = (result.price - entry_price) * quantity
                else:
                    gross_pnl = (entry_price - result.price) * quantity

                position_value = entry_price * quantity
                net_pnl = gross_pnl - result.fee
                net_pnl_pct = (net_pnl / position_value) * 100

                # 3. Clear position
                del self._positions[symbol]

                # 4. Move to the ANALYSIS tier in TierManager.
                self.tier_manager.set_tier(symbol, TierLevel.ANALYSIS)

                # 5. Log exit (kompakt format)
                pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                exit_emoji = '✅' if net_pnl >= 0 else '❌'

                self.logger.info(f"\n{exit_emoji} {symbol:<10} │ {exit_type} Exit │ Entry: ${entry_price:,.2f} → Exit: ${result.price:,.2f}")
                self.logger.info(f"   {pnl_emoji} PnL: {net_pnl_pct:+.2f}% (${net_pnl:+,.2f}) │ Fee: ${result.fee:.2f}")

        except Exception as e:
            self.logger.error(f"❌ {symbol}: Error closing position: {e}")

    def _is_new_candle_closed(self, symbol: str, mtf_engine) -> bool:
        """
        Has a new candle closed? (different from the last processed one)

        WebSocket continuously sends open candles. Entry should only be made when a
        NEW candle closes (like in backtesting).

        Logic:
        - candles[-1] is the last closed candle.
        - If the timestamp has changed, it means a new candle has closed.
        - Do not create duplicate entries for the same candle.

        Args:
            symbol: Symbol name
            mtf_engine: MultiTimeframeEngine instance

        Returns:
            bool: True = a new candle has closed, an entry can be made.
        """
        try:
            primary_tf = self.strategy.primary_timeframe
            tf_info = mtf_engine.tf_info.get(primary_tf)

            if not tf_info or not tf_info.candles:
                return False

            # The timestamp of the last closed candle.
            last_closed_ts = tf_info.candles[-1].timestamp

            # Has this candle been processed before?
            last_processed_ts = self._last_processed_candle.get(symbol, 0)

            if last_closed_ts > last_processed_ts:
                # A new candle has closed! Save the timestamp.
                self._last_processed_candle[symbol] = last_closed_ts
                return True

            # The same candle, it has already been processed.
            return False

        except Exception as e:
            self.logger.debug(f"⚠️ Candle close check error: {e}")
            return False

    def _update_price_cache(self) -> None:
        """
        Write the current prices of all symbols to the cache.

        Gets the price from the current candle or the last candle in the MTF Engine.
        Required for DisplayInfo (P&L calculation).
        """
        if not self.mtf_engines or not self.cache:
            return

        for symbol, mtf_engine in self.mtf_engines.items():
            try:
                primary_tf = self.strategy.primary_timeframe
                tf_info = mtf_engine.tf_info.get(primary_tf)

                if not tf_info:
                    continue

                # If there is a current candle, take it from there, otherwise take it from the last closed candle.
                if tf_info.current_candle:
                    price = tf_info.current_candle.close
                elif tf_info.candles and len(tf_info.candles) > 0:
                    price = tf_info.candles[-1].close
                else:
                    continue

                # Cache'e yaz (60 saniye TTL)
                self.cache.set(f"price:{symbol}", price, ttl=60)

            except Exception:
                pass  # Silent fail - not critical

    def _build_dataframe_from_mtf(
        self,
        mtf_engine,
        use_previous_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Create a DataFrame dictionary for SignalValidator from the MTF Engine.

        V7: The Trading Engine does NOT perform calculations!
        All indicator values come from calculators.
        Calculator key format: "ema_50_5m", "ema_50_15m" (HER ZAMAN suffix)

        Args:
            mtf_engine: MultiTimeframeEngine instance
            use_previous_result: True = use the values of the closed candle (for entry)
                                False = use the current values (for monitoring)

        Returns:
            Dict[str, pd.DataFrame]: {timeframe: DataFrame} or None
        """
        import pandas as pd

        try:
            result = {}
            symbol = mtf_engine.symbol
            primary_tf = mtf_engine.primary_timeframe

            # Known timeframe suffixes for stripping
            known_tf_suffixes = ["_1m", "_3m", "_5m", "_15m", "_30m", "_1h", "_2h", "_4h", "_6h", "_12h", "_1d"]

            # First, create the base DataFrame for all timeframes.
            for tf, tf_info in mtf_engine.tf_info.items():
                candles = list(tf_info.candles)
                if not candles:
                    continue

                # Convert candles to a list of dictionaries
                candle_dicts = []
                for candle in candles:
                    candle_dicts.append({
                        'timestamp': candle.timestamp,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    })

                df = pd.DataFrame(candle_dicts)
                result[tf] = df

            # If the primary DataFrame does not exist, return
            if primary_tf not in result:
                return None

            primary_df = result[primary_tf]

            # V8: Separate DataFrame for each timeframe - with its own OHLCV + indicators.
            # Calculator key format: "ema_50_5m", "ema_50_15m"
            # Each timeframe takes its own indicators (without suffix)
            #
            # Result:
            #   result['5m']  = {close, ema_50, rsi_14, ...}  ← 5m values
            #   result['15m'] = {close, ema_50, rsi_14, ...}  ← 15m values
            #
            # SignalValidator:
            #   ['close', '<', 'ema_50']        → result['5m']['close'] < result['5m']['ema_50']
            #   ['close', '<', 'ema_50', '15m'] → result['15m']['close'] < result['15m']['ema_50']

            if self.indicator_manager:
                calculators = self.indicator_manager.calculators.get(symbol, {})

                if not calculators:
                    self.logger.warning(f"⚠️ {symbol}: calculators is empty! Warmup may not have been performed.")

                # Add each calculator to the DataFrame of its corresponding timeframe.
                for calc_key, calculator in calculators.items():
                    # calc_key: "ema_50_5m", "rsi_14_15m", etc.
                    if use_previous_result and calculator.previous_result:
                        value = calculator.previous_result.value
                    elif calculator.last_result:
                        value = calculator.last_result.value
                    else:
                        continue

                    # Separate the timeframe and indicator name from calc_key
                    # "ema_50_5m" → indicator="ema_50", tf="5m"
                    # "adx_14_15m" → indicator="adx_14", tf="15m"
                    target_tf = None
                    indicator_name = calc_key

                    for tf_suffix in known_tf_suffixes:
                        if calc_key.endswith(tf_suffix):
                            target_tf = tf_suffix[1:]  # "_5m" → "5m"
                            indicator_name = calc_key[:-len(tf_suffix)]  # "ema_50_5m" → "ema_50"
                            break

                    # Find the target DataFrame
                    if target_tf and target_tf in result:
                        target_df = result[target_tf]
                    else:
                        # If there is no suffix or the suffix is unknown, add it to the primary.
                        target_df = result.get(primary_tf)
                        if target_df is None:
                            continue

                    # Add the value to the DataFrame (without suffix!)
                    if isinstance(value, dict):
                        # Multi-output indicators: {'lower': 100, 'middle': 105, 'upper': 110}
                        # Create a separate column for each key: bollinger_lower, bollinger_upper, etc.
                        for key, val in value.items():
                            col_name = f"{indicator_name}_{key}"
                            if isinstance(val, (int, float)):
                                target_df[col_name] = val
                            elif hasattr(val, 'iloc') and len(val) > 0:
                                target_df[col_name] = val.iloc[-1]
                            elif isinstance(val, (list, tuple)) and len(val) > 0:
                                target_df[col_name] = val[-1]
                            else:
                                target_df[col_name] = val
                        # Add the main indicator name as well (either the initial value or the dictionary itself)
                        if value:
                            first_val = list(value.values())[0]
                            if isinstance(first_val, (int, float)):
                                target_df[indicator_name] = first_val
                            elif hasattr(first_val, 'iloc') and len(first_val) > 0:
                                target_df[indicator_name] = first_val.iloc[-1]
                    elif isinstance(value, (int, float)):
                        target_df[indicator_name] = value
                    elif hasattr(value, 'iloc'):
                        target_df[indicator_name] = value.iloc[-1] if len(value) > 0 else None
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        target_df[indicator_name] = value[-1]
                    else:
                        target_df[indicator_name] = value

            # Also add the primary timeframe by default (backward compat)
            result['default'] = result.get(primary_tf)

            # Log the keys and values when the DataFrame is created for the first time.
            if result and self.verbose and not hasattr(self, '_df_keys_logged'):
                self._df_keys_logged = True
                for tf, df in result.items():
                    if tf != 'default' and df is not None:
                        keys = list(df.columns)
                        self.logger.info(f"📋 DataFrame Keys [{tf}]: {keys}")
                        # DEBUG: Show the close and ema_50 values for each TF.
                        close_val = df['close'].iloc[-1] if 'close' in df.columns else 'N/A'
                        ema_val = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else 'N/A'
                        self.logger.info(f"   📊 {tf}: close={close_val}, ema_50={ema_val}")

            return result if result else None

        except Exception as e:
            self.logger.info(f"❌ DataFrame build error: {e}")
            import traceback
            self.logger.info(traceback.format_exc())
            return None

    # ════════════════════════════════════════════════════════════════════════════
    # LIFECYCLE: initialize() + start() + stop()
    # ════════════════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """
        Initialize the Trading Engine.

        Sequence:
        1. Load strategy (from config source)
        2. Exchange connector (testnet/production based on mode)
        3. Create managers (get config from strategy)
        4. Load mode (with connector, strategy, managers)
        5. Start data feeds (TODO)
        6. Tier system kur (TODO)

        Raises:
            RuntimeError: Already initialized.
        """
        if self._initialized:
            self.logger.warning("⚠️ TradingEngine is already initialized")
            return

        self.logger.info("=" * 60)
        self.logger.info("🚀 TradingEngine V5 is starting...")
        self.logger.info("=" * 60)

        try:
            # ════════════════════════════════════════════════════════════════
            # Load strategy (from config source)
            # ════════════════════════════════════════════════════════════════
            self.strategy = await self._load_strategy()

            # ════════════════════════════════════════════════════════════════
            # Load symbols (immediately after the strategy)
            # ════════════════════════════════════════════════════════════════
            self.symbols = await self._load_symbols()

            # ════════════════════════════════════════════════════════════════
            # Exchange connector (replay: None, demo: testnet, paper/live: production)
            # ════════════════════════════════════════════════════════════════
            if self.mode_name == 'replay':
                self.connector = None
                self.logger.info("📡 Connector: None (replay mode)")
            else:
                binance_config = self.config.get("connectors.binance", {}).copy()
                binance_config["testnet"] = (self.mode_name == 'demo')
                self.connector = BinanceAPI(
                    config=binance_config,
                    cache_manager=self.cache
                )

            # ════════════════════════════════════════════════════════════════
            # RiskManager (strategy'den config)
            # ════════════════════════════════════════════════════════════════
            risk_config = getattr(self.strategy, 'risk_management', {})
            self.risk_manager = RiskManager(config=risk_config, logger=self.logger)
            self.logger.info("🛡️ RiskManager is ready")

            # ════════════════════════════════════════════════════════════════
            # IndicatorManager (with bridge helper - same pattern as backtest)
            # ════════════════════════════════════════════════════════════════
            self.indicator_manager = create_indicator_manager_from_strategy(
                strategy=self.strategy,
                cache_manager=self.cache,
                logger=self.logger,
                event_bus=self.event_bus,
                verbose=self.verbose
            )

            # ════════════════════════════════════════════════════════════════
            # Load mode (with connector, strategy, and managers)
            # ════════════════════════════════════════════════════════════════
            self.current_mode = await self._load_mode(self.mode_name)

            # ════════════════════════════════════════════════════════════════
            # Start data feeds (WebSocket + Indicator subscription)
            # ════════════════════════════════════════════════════════════════
            if self.mode_name != 'replay':
                await self._start_data_feeds(self.symbols)

            # ════════════════════════════════════════════════════════════════
            # V5 Components: TierManager + PriceFeed + DisplayInfo
            # ════════════════════════════════════════════════════════════════
            await self._initialize_v5_components()

            self._initialized = True

            self.logger.info("=" * 60)
            self.logger.info("✅ TradingEngine V5 HAZIR")
            self.logger.info("=" * 60)

        except FileNotFoundError as e:
            # Strategy not found - clean error message (no traceback)
            self.logger.error(str(e))
            raise SystemExit(1)

        except Exception as e:
            self.logger.error(f"❌ Initialization error: {e}")
            raise

    async def start(self) -> None:
        """
        Start trading (blocking - stops with Ctrl+C)

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("TradingEngine has not been initialized! Call initialize() first.")

        if self._running:
            self.logger.warning("⚠️ TradingEngine is already running")
            return

        self._running = True
        self.logger.info("🚀 TradingEngine started")
        self.logger.info("💡 Press Ctrl+C to stop")

        # Main loop (stops with Ctrl+C)
        try:
            # Config'den status interval al (trading.yaml → status_display.status_interval)
            display_interval = self.config.get("status_display.status_interval", 15)
            loop_count = 0

            while self._running:
                loop_count += 1

                # ════════════════════════════════════════════════════════════
                # Price Cache Update (MTF Engine'den)
                # ════════════════════════════════════════════════════════════
                self._update_price_cache()

                # ════════════════════════════════════════════════════════════
                # Tier Processing (condition evaluation)
                # ════════════════════════════════════════════════════════════
                await self._process_tiers(loop_count)

                # ════════════════════════════════════════════════════════════
                # Display status periodically (same format as test_tier_standalone.py)
                # ════════════════════════════════════════════════════════════
                if self.display_info and loop_count % display_interval == 0:
                    self.logger.info("─" * 60)
                    # Show detailed conditions in verbose mode
                    if self.verbose:
                        for line in self.display_info.format_conditions_verbose():
                            self.logger.info(line)
                    # Status line
                    self.logger.info(self.display_info.format_status_line())
                    # Tier summary
                    for line in self.display_info.format_tier_summary(verbose=self.verbose):
                        self.logger.info(line)
                    self.logger.info("─" * 60)

                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("🛑 Main loop cancelled")

    async def stop(self) -> None:
        """
        Stop trading.
        """
        # If it hasn't started, exit silently.
        if not self._running and not self._initialized:
            return

        if not self._running:
            self.logger.warning("⚠️ TradingEngine is already stopped")
            return

        self.logger.info("🛑 TradingEngine durduruluyor...")
        self._running = False

        # ════════════════════════════════════════════════════════════════
        # Disable MTF Engines
        # ════════════════════════════════════════════════════════════════
        if self.mtf_engines:
            self.logger.info("   🛑 MTF Engines are being shut down...")
            for symbol, mtf in self.mtf_engines.items():
                try:
                    await mtf.stop()
                except Exception as e:
                    self.logger.error(f"   ❌ MTF stop error ({symbol}): {e}")
            self.mtf_engines.clear()

        # ════════════════════════════════════════════════════════════════
        # Close WebSocket
        # ════════════════════════════════════════════════════════════════
        if self.websocket_engine:
            self.logger.info("   🛑 WebSocket is being closed...")
            try:
                await self.websocket_engine.stop()
            except Exception as e:
                self.logger.error(f"   ❌ WebSocket stop error: {e}")
            self.websocket_engine = None

        # ════════════════════════════════════════════════════════════════
        # Disable mode
        # ════════════════════════════════════════════════════════════════
        if self.current_mode:
            await self.current_mode.shutdown()

        # ════════════════════════════════════════════════════════════════
        # Close the connector
        # ════════════════════════════════════════════════════════════════
        if self.connector:
            self.logger.info("   🛑 BinanceAPI is being closed...")
            try:
                if hasattr(self.connector, 'close'):
                    await self.connector.close()
                elif hasattr(self.connector, 'shutdown'):
                    await self.connector.shutdown()
            except Exception as e:
                self.logger.error(f"   ❌ Connector close error: {e}")
            self.connector = None

        self.logger.info("✅ TradingEngine durduruldu")

    @property
    def is_running(self) -> bool:
        """Is the engine running?"""
        return self._running

    @property
    def is_initialized(self) -> bool:
        """Is the engine initialized?"""
        return self._initialized


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Engine V5")
    parser.add_argument("--mode", "-m", default="paper", choices=["paper", "demo", "live", "replay"])
    parser.add_argument("--strategy", "-s", required=True, help="Strategy template (e.g., ema5_bb_adx.py)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    async def main():
        engine = TradingEngine(
            mode=args.mode,
            strategy_path=args.strategy,
            verbose=args.verbose
        )

        try:
            await engine.initialize()
            await engine.start()
        except KeyboardInterrupt:
            print("\n⚠️ Ctrl+C detected")
        finally:
            await engine.stop()

    asyncio.run(main())
