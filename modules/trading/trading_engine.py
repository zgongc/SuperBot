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
from components.strategies.risk_manager import RiskManager
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

# Position management
from components.strategies.position_manager import PositionManager
from components.strategies.pnl_calculator import PnLCalculator

# Trade logging
from modules.trading.trade_logger import TradeLogger


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
        self._position_manager: Optional[PositionManager] = None

        # Trade Logger (JSON history)
        self._trade_logger: Optional[TradeLogger] = None

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

        StrategyManager performs path normalization:
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
            raise ValueError("Strategy path was not specified!")

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
        Get trading symbols from SymbolsManager.

        Creates and initializes the SymbolsManager if it doesn't exist.
        Source information is retrieved from strategy.symbol_source.

        SymbolsManager manages all sources:
        - config: From the Config file.
        - strategy: Strategy template'den
        - file: from a JSON file
        - exchange: Exchange API'den

        The normalization process is performed in the SymbolsManager:
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

    # ════════════════════════════════════════════════════════════════════════════
    # REPLAY DATA LOADING
    # ════════════════════════════════════════════════════════════════════════════

    async def _load_replay_data(self) -> None:
        """
        Load parquet data for replay mode.

        The strategy loads data from the data/parquets/ directory within the
        backtest_start_date and backtest_end_date range.
        """
        from modules.trading.modes.replay_mode import ReplayMode
        from pathlib import Path
        import pandas as pd
        from datetime import datetime

        if not isinstance(self.current_mode, ReplayMode):
            self.logger.error("❌ current_mode is not ReplayMode!")
            return

        self.logger.info("📂 Loading replay data...")

        # Parquet dizini
        parquet_dir = Path("data/parquets")
        if not parquet_dir.exists():
            self.logger.error(f"❌ Parquet directory not found: {parquet_dir}")
            return

        # Strategy'den date range al
        start_date_str = getattr(self.strategy, 'backtest_start_date', None)
        end_date_str = getattr(self.strategy, 'backtest_end_date', None)

        start_date = None
        end_date = None

        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('T', ' '))
            except:
                pass

        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('T', ' '))
            except:
                pass

        # Primary timeframe
        primary_tf = self.strategy.primary_timeframe

        # Load parquet for each symbol
        loaded_count = 0
        for symbol in self.symbols:
            # Search for the Parquet file - new format: data/parquets/{symbol}/{symbol}_{tf}_*.parquet
            symbol_dir = parquet_dir / symbol
            parquet_files = list(symbol_dir.glob(f"{symbol}_{primary_tf}_*.parquet")) if symbol_dir.exists() else []

            if not parquet_files:
                self.logger.warning(f"   ⚠️ {symbol}: Parquet file not found")
                continue

            # Get the most recent file
            parquet_file = sorted(parquet_files)[-1]

            try:
                # Load to ReplayMode
                await self.current_mode.load_parquet(
                    filepath=str(parquet_file),
                    symbol=symbol,
                    timeframe=primary_tf
                )

                # Date range filter (varsa)
                key = f"{symbol}_{primary_tf}"
                if key in self.current_mode._data and (start_date or end_date):
                    original_count = len(self.current_mode._data[key])
                    filtered = []

                    for candle in self.current_mode._data[key]:
                        candle_time = datetime.fromtimestamp(candle.timestamp / 1000)
                        if start_date and candle_time < start_date:
                            continue
                        if end_date and candle_time > end_date:
                            continue
                        filtered.append(candle)

                    self.current_mode._data[key] = filtered
                    self.current_mode._stats["total_candles"] = sum(
                        len(c) for c in self.current_mode._data.values()
                    )
                    self.logger.info(
                        f"   📅 {symbol}: {original_count} → {len(filtered)} candles (date filter)"
                    )

                loaded_count += 1

            except Exception as e:
                self.logger.error(f"   ❌ {symbol}: Parquet loading error: {e}")

        if loaded_count == 0:
            self.logger.error("❌ No parquet files could be loaded!")
        else:
            total_candles = self.current_mode._stats.get("total_candles", 0)
            self.logger.info(f"✅ {loaded_count} symbols loaded ({total_candles} candle)")

        # Register the replay callback.
        self.current_mode.on_candle(self._on_replay_candle)

    async def _on_replay_candle(self, candle) -> None:
        """
        Called for each candle in replay mode.

        Args:
            candle: Candle object
        """
        try:
            symbol = candle.symbol
            timeframe = candle.timeframe

            # Send the candle to IndicatorManager
            if self.indicator_manager:
                # Create a DataFrame (single candle)
                import pandas as pd
                df = pd.DataFrame([{
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                }])

                # Indicator hesapla
                await self.indicator_manager.update_indicators(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=df
                )

            # Strategy evaluation (more data is needed for a full evaluation)
            # TODO: Evaluate after the warmup period is complete.

        except Exception as e:
            self.logger.debug(f"Replay candle error: {e}")

    async def _start_data_feeds(self, symbols: List[str]) -> None:
        """
        Start WebSocket + MTF + Indicator subscription.

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
                TierLevel.POSITION: 1,    # Every second (SL/TP critical)
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
            self.logger.info("   ❌ MTF Engines not found!")
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

        # DEBUG: Log the summary itself.
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
        Have all MTF engines completed the warmup?

        MTF Engine is running=True and candles exist for the primary timeframe,
        which means the warmup is complete.

        Returns:
            bool: True = warmup is complete, tier processing can begin
        """
        # Replay mode - MTF engines are not available, warmup is not required
        if self.mode_name == 'replay':
            return True

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

            # Calculate with the latest data (use MTFEngine.build_dataframe)
            data = mtf_engine.build_dataframe(self.indicator_manager, use_previous_candle=False)
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

            # 4. Check for new candle closing (use MTFEngine.is_new_candle_closed)
            last_ts = self._last_processed_candle.get(symbol, 0)
            is_new_candle, current_ts = mtf_engine.is_new_candle_closed(last_ts)
            if is_new_candle:
                self._last_processed_candle[symbol] = current_ts

            # 5. Entry Signal Check
            new_tier = self.tier_manager.get_tier(symbol)

            # If the position already exists, skip it.
            if symbol in self._positions:
                # If there is a pending decision, clear it.
                if symbol in self._pending_decisions:
                    del self._pending_decisions[symbol]
                return

            # V5 FIX: %100 signal + candle close = open trade immediately
            if score >= 1.0 and is_new_candle:
                # The candle closed and there is a 100% signal - open a trade immediately
                self.logger.info(f"🎯 {symbol}: Candlestick closed + 100% signal! {best_side} is opening...")
                await self._open_position(symbol, best_side, data)

                # Clear any pending items if they exist.
                if symbol in self._pending_decisions:
                    del self._pending_decisions[symbol]

            elif score >= 1.0:
                # There is a 100% signal, but the candle has not closed yet - cache it.
                primary_tf = self.strategy.primary_timeframe
                tf_info = mtf_engine.tf_info.get(primary_tf)
                current_ts = tf_info.candles[-1].timestamp if tf_info and tf_info.candles else 0

                pending = self._pending_decisions.get(symbol)
                if not pending or pending.get('timestamp') != current_ts:
                    self._pending_decisions[symbol] = {
                        'side': best_side,
                        'data': data,
                        'timestamp': current_ts,
                        'score': score
                    }
                    self.logger.debug(f"📌 {symbol}: 100% signal cached, waiting for candle close...")

            elif is_new_candle and symbol in self._pending_decisions:
                # The candle closed and there is a previously cached signal.
                pending = self._pending_decisions[symbol]
                cached_side = pending['side']
                cached_data = pending['data']

                self.logger.info(f"🎯 {symbol}: Candlestick closed + Cached signal! {cached_side} is opening...")
                await self._open_position(symbol, cached_side, cached_data)
                del self._pending_decisions[symbol]

            elif score < 1.0 and symbol in self._pending_decisions:
                # Score decreased, clear the cache
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
        - Position Management kontrolleri (max_positions, pyramiding, hedging)
        - Calculate SL/TP with ExitManager
        - Calculate position size with RiskManager
        - Execute order with PaperMode.execute_order()

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            data: DataFrame dict (indicators dahil)
        """
        from datetime import datetime
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            # ════════════════════════════════════════════════════════════════
            # 0. POSITION MANAGEMENT KONTROLLERI (via PositionManager)
            # ════════════════════════════════════════════════════════════════

            # Lazy init PositionManager
            if not self._position_manager:
                self._position_manager = PositionManager(self.strategy, logger=self.logger)

            # Convert self._positions dict to list format for PositionManager
            positions_list = list(self._positions.values())

            # Use PositionManager to check if position can be opened
            open_result = self._position_manager.can_open_position(symbol, side, positions_list)

            if not open_result.can_open:
                self.logger.info(f"⚠️ {symbol}: {open_result.reason}, skipping")
                return

            # ONE-WAY MODE: Close opposite positions first if required
            if open_result.should_close_opposite:
                for opp_pos in open_result.opposite_positions:
                    self.logger.info(f"🔄 {symbol}: Closing opposite {opp_pos['side']} position (one-way mode)")
                    opp_price = await self.current_mode.get_current_price(symbol)
                    if opp_price:
                        await self._close_position(symbol, opp_pos, opp_price, "SIGNAL_REVERSE")

            # Store pyramiding scale for later use
            pyramiding_scale = open_result.pyramiding_scale

            # ════════════════════════════════════════════════════════════════
            # 1. Get the current price
            # ════════════════════════════════════════════════════════════════
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

            # Apply pyramiding scale factor (BacktestEngine pattern)
            if pyramiding_scale < 1.0:
                original_qty = quantity
                quantity *= pyramiding_scale
                self.logger.info(f"📈 {symbol}: Pyramiding entry #{len(same_side_positions)+1}, size scaled: {original_qty:.6f} → {quantity:.6f} ({pyramiding_scale:.2f}x)")

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
                # 5. Create position dictionary (use PositionManager.create_position)
                self._trade_counter += 1

                position = PositionManager.create_position(
                    position_id=self._trade_counter,
                    symbol=symbol,
                    side=side,
                    entry_price=result.price,
                    quantity=result.filled_quantity,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    order_id=result.order_id
                )

                self._positions[symbol] = position

                # 6. Move to the POSITION tier in TierManager.
                self.tier_manager.set_tier(symbol, TierLevel.POSITION)

                # 7. Save TradeLogger entry (AI Training: indicator snapshot + market context)
                if self._trade_logger:
                    indicator_snapshot = self._extract_indicator_snapshot(symbol, data)
                    market_context = self._extract_market_context(symbol, data)

                    self._trade_logger.log_entry(
                        trade_id=self._trade_counter,
                        symbol=symbol,
                        side=side,
                        entry_price=result.price,
                        quantity=result.filled_quantity,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        indicators=indicator_snapshot,
                        market_context=market_context
                    )

                # 8. Log entry (kompakt format)
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
        Check all open positions (SL/TP/Trailing/Timeout)

        For each POSITION tier symbol:
        - Check the position timeout (BacktestEngine pattern)
        - Check the exit conditions using StrategyExecutor.evaluate_exit()
        - Close the position if SL/TP is hit
        - Update the SL for Trailing/Break-even
        """
        from datetime import datetime

        if not self._positions:
            return

        # Lazy init PositionManager
        if not self._position_manager:
            self._position_manager = PositionManager(self.strategy, logger=self.logger)

        for symbol, position in list(self._positions.items()):
            try:
                # 1. Get the current price
                current_price = await self.current_mode.get_current_price(symbol)
                if not current_price:
                    continue

                # ════════════════════════════════════════════════════════════════
                # 1b. POSITION TIMEOUT CHECK (via PositionManager)
                # ════════════════════════════════════════════════════════════════
                should_timeout, timeout_reason = self._position_manager.check_position_timeout(
                    position, datetime.now()
                )
                if should_timeout:
                    self.logger.info(f"⏰ {symbol}: {timeout_reason}, closing...")
                    await self._close_position(symbol, position, current_price, "TIMEOUT")
                    continue  # Skip to next position

                # Update highest/lowest (for trailing) - Use PositionManager
                extreme_updated = PositionManager.update_extreme_prices(position, current_price)

                # Update also in TradeLogger (for recovery)
                if extreme_updated and self._trade_logger:
                    trade_id = position.get('position_id', position.get('id', 0))
                    self._trade_logger.update_extreme_prices(
                        trade_id=trade_id,
                        highest_price=position.get('highest_price'),
                        lowest_price=position.get('lowest_price')
                    )

                # 2. Get data from the MTF Engine (use MTFEngine.build_dataframe)
                mtf_engine = self.mtf_engines.get(symbol)
                if not mtf_engine:
                    continue

                data = mtf_engine.build_dataframe(self.indicator_manager, use_previous_candle=False)

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

                    trade_id = position.get('position_id', 0)

                    if exit_result.get('break_even_moved'):
                        self.logger.info(f"🔄 {symbol}: Break-even SL moved: ${old_sl:,.2f} → ${new_sl:,.2f}")
                        # TradeLogger break-even record
                        if self._trade_logger:
                            self._trade_logger.log_break_even(
                                trade_id=trade_id,
                                old_sl=old_sl,
                                new_sl=new_sl,
                                current_price=current_price
                            )
                    else:
                        self.logger.info(f"🔄 {symbol}: Trailing SL updated: ${old_sl:,.2f} → ${new_sl:,.2f}")
                        # TradeLogger trailing record
                        if self._trade_logger:
                            self._trade_logger.log_trailing_update(
                                trade_id=trade_id,
                                old_sl=old_sl,
                                new_sl=new_sl,
                                current_price=current_price
                            )

                # 4b. Partial Exit control
                if exit_result.get('partial_exit_size'):
                    partial_size = exit_result['partial_exit_size']
                    partial_level = exit_result.get('partial_exit_level', 1)
                    await self._partial_close_position(symbol, position, current_price, partial_size, partial_level)
                    continue  # Don't check full exit after partial

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
            current_price: Current price (when called)
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
                # 2. Calculate PnL (use PnLCalculator)
                entry_price = position['entry_price']
                quantity = position['quantity']
                gross_pnl, net_pnl, net_pnl_pct = PnLCalculator.calculate(
                    entry_price=entry_price,
                    exit_price=result.price,
                    quantity=quantity,
                    side=position['side'],
                    fee=result.fee
                )

                # 3. Save TradeLogger exit (AI Training: indicator snapshot + market context)
                trade_id = position.get('position_id', 0)
                if self._trade_logger:
                    # Indicator and market context at the exit.
                    mtf_engine = self.mtf_engines.get(symbol)
                    data = mtf_engine.build_dataframe(self.indicator_manager, use_previous_candle=False) if mtf_engine else None
                    indicator_snapshot = self._extract_indicator_snapshot(symbol, data) if data else {}
                    market_context = self._extract_market_context(symbol, data) if data else {}

                    self._trade_logger.log_exit(
                        trade_id=trade_id,
                        exit_price=result.price,
                        exit_reason=exit_type,
                        pnl=net_pnl,
                        pnl_pct=net_pnl_pct,
                        fee=result.fee,
                        indicators=indicator_snapshot,
                        market_context=market_context
                    )

                # 4. Clear position
                del self._positions[symbol]

                # 5. Move to the ANALYSIS tier in TierManager.
                self.tier_manager.set_tier(symbol, TierLevel.ANALYSIS)

                # 6. Log exit (kompakt format)
                pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                exit_emoji = '✅' if net_pnl >= 0 else '❌'

                self.logger.info(f"\n{exit_emoji} {symbol:<10} │ {exit_type} Exit │ Entry: ${entry_price:,.2f} → Exit: ${result.price:,.2f}")
                self.logger.info(f"   {pnl_emoji} PnL: {net_pnl_pct:+.2f}% (${net_pnl:+,.2f}) │ Fee: ${result.fee:.2f}")

        except Exception as e:
            self.logger.error(f"❌ {symbol}: Error closing position: {e}")

    async def _partial_close_position(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        exit_size: float,
        exit_level: int
    ) -> None:
        """
        Partially close the position (Partial Exit)

        Args:
            symbol: Trading symbol
            position: Position dict
            current_price: Current price
            exit_size: Exit percentage (0.40 = 40%)
            exit_level: Which partial exit (1, 2, 3...)
        """
        from modules.trading.modes.base_mode import Order, OrderSide, OrderType

        try:
            # 1. Calculate the amount to be closed
            close_quantity = position['quantity'] * exit_size

            # 2. Create order confirmation
            close_side = OrderSide.SELL if position['side'] == 'LONG' else OrderSide.BUY
            order = Order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=close_quantity
            )

            result = await self.current_mode.execute_order(order)

            if result.status.value == 'FILLED':
                # 3. Calculate PnL (use PnLCalculator)
                gross_pnl, net_pnl = PnLCalculator.calculate_partial(
                    entry_price=position['entry_price'],
                    exit_price=result.price,
                    close_quantity=close_quantity,
                    side=position['side'],
                    fee=result.fee
                )

                # 4. Update position
                position['quantity'] -= close_quantity
                completed_exits = position.get('completed_partial_exits', 0)
                position['completed_partial_exits'] = completed_exits + 1

                # 5. Remaining percentage
                original_qty = position.get('original_quantity', position['quantity'] + close_quantity)
                if 'original_quantity' not in position:
                    position['original_quantity'] = original_qty
                remaining_pct = (position['quantity'] / original_qty) * 100

                # 6. Save partial exit in TradeLogger
                trade_id = position.get('position_id', 0)
                if self._trade_logger:
                    self._trade_logger.log_partial_exit(
                        trade_id=trade_id,
                        level=exit_level,
                        exit_price=result.price,
                        size_pct=exit_size * 100,
                        remaining_pct=remaining_pct,
                        pnl=net_pnl,
                        fee=result.fee
                    )

                # 7. Log
                pnl_emoji = '🟢' if net_pnl >= 0 else '🔴'
                self.logger.info(f"📤 {symbol}: Partial exit {exit_level}: Closed {exit_size*100:.0f}% @ ${result.price:,.2f}")
                self.logger.info(f"   → Remaining: {remaining_pct:.0f}% of original position")
                self.logger.info(f"   → {pnl_emoji} Profit: ${net_pnl:+,.2f}")

        except Exception as e:
            self.logger.error(f"❌ {symbol}: Partial close error: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # RECOVERY & AI TRAINING HELPERS
    # ════════════════════════════════════════════════════════════════════════════

    async def _recover_positions(self) -> None:
        """
        Recover open positions from the previous session.

        Loads open trades from TradeLogger and adds them to the _positions dictionary.
        This ensures that trades continue even when the TradingEngine restarts.
        """
        if not self._trade_logger:
            return

        if not self._trade_logger.has_open_trades():
            return

        open_positions = self._trade_logger.get_open_trades_for_recovery()

        if not open_positions:
            return

        self.logger.info(f"🔄 Recovery: {len(open_positions)} open positions found")

        for pos in open_positions:
            symbol = pos['symbol']

            # Check if the symbol is active
            if symbol not in self.symbols:
                self.logger.warning(f"   ⚠️ {symbol}: Not found in the active symbol list, skip")
                continue

            # Load the position
            self._positions[symbol] = pos

            # Move to the POSITION tier in TierManager.
            if self.tier_manager:
                self.tier_manager.set_tier(symbol, TierLevel.POSITION)

            sl_display = f"${pos['sl_price']:,.2f}" if pos['sl_price'] else "None"
            self.logger.info(
                f"   ✅ {symbol}: {pos['side']} @ ${pos['entry_price']:,.2f} recovered (SL: {sl_display})"
            )

        # Update the trade counter.
        last_id = self._trade_logger.get_last_trade_id()
        if last_id > self._trade_counter:
            self._trade_counter = last_id
            self.logger.info(f"   📊 Trade counter: {self._trade_counter}")

    def _extract_indicator_snapshot(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicator snapshot for AI Training.

        Returns the last indicator values from the DataFrame as a dictionary.

        Args:
            symbol: Symbol name
            data: MTF DataFrame dict

        Returns:
            Dict: Indicator values
            {
                "rsi_14": 35.2,
                "ema_50": 95000.5,
                "atr_14": 1500.0,
                "bb_upper": 96000.0,
                "bb_lower": 94000.0,
                ...
            }
        """
        snapshot = {}

        try:
            # Primary timeframe DataFrame'i al
            primary_tf = getattr(self.strategy, 'primary_timeframe', '5m')
            df = data.get(primary_tf) if data else None

            if df is None or len(df) == 0:
                return snapshot

            # Exclude OHLCV columns
            exclude_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}

            # Get all indicator values from the last line
            last_row = df.iloc[-1]

            for col in df.columns:
                if col.lower() not in exclude_cols:
                    val = last_row[col]
                    # NaN check
                    if val is not None and not (isinstance(val, float) and val != val):
                        snapshot[col] = float(val) if isinstance(val, (int, float)) else val

        except Exception as e:
            self.logger.debug(f"Indicator snapshot error: {e}")

        return snapshot

    def _extract_market_context(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract market context for AI Training.

        Returns market status information such as volatility, trend, and volume.

        Args:
            symbol: Symbol name
            data: MTF DataFrame dict

        Returns:
            Dict: Market context
            {
                "volatility": "high",
                "trend": "bullish",
                "volume_ratio": 1.5,
                "price_change_pct": 0.5,
                ...
            }
        """
        context = {}

        try:
            primary_tf = getattr(self.strategy, 'primary_timeframe', '5m')
            df = data.get(primary_tf) if data else None

            if df is None or len(df) < 20:
                return context

            # Calculate based on the last 20 candles
            recent = df.tail(20)
            last = df.iloc[-1]

            # Price change
            if len(recent) >= 2:
                price_change = (last['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close'] * 100
                context['price_change_pct'] = round(price_change, 2)

            # Trend (simple: last close vs 20-bar SMA)
            sma_20 = recent['close'].mean()
            if last['close'] > sma_20 * 1.01:
                context['trend'] = 'bullish'
            elif last['close'] < sma_20 * 0.99:
                context['trend'] = 'bearish'
            else:
                context['trend'] = 'neutral'

            # Volatility (ATR varsa)
            if 'atr' in df.columns or 'atr_14' in df.columns:
                atr_col = 'atr' if 'atr' in df.columns else 'atr_14'
                atr = last[atr_col]
                atr_pct = (atr / last['close']) * 100 if last['close'] > 0 else 0
                context['volatility_pct'] = round(atr_pct, 2)
                if atr_pct > 3:
                    context['volatility'] = 'high'
                elif atr_pct > 1.5:
                    context['volatility'] = 'medium'
                else:
                    context['volatility'] = 'low'

            # Volume ratio (son volume / average)
            avg_volume = recent['volume'].mean()
            if avg_volume > 0:
                context['volume_ratio'] = round(last['volume'] / avg_volume, 2)

        except Exception as e:
            self.logger.debug(f"Market context error: {e}")

        return context

    # ════════════════════════════════════════════════════════════════════════════
    # LIFECYCLE: initialize() + start() + stop()
    # ════════════════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """
        Initialize the Trading Engine.

        Queue:
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
            if self.mode_name == 'replay':
                # Replay mode: Load Parquet data
                await self._load_replay_data()
            else:
                await self._start_data_feeds(self.symbols)

            # ════════════════════════════════════════════════════════════════
            # V5 Components: TierManager + PriceFeed + DisplayInfo
            # ════════════════════════════════════════════════════════════════
            await self._initialize_v5_components()

            # ════════════════════════════════════════════════════════════════
            # Trade Logger (JSON trade history)
            # ════════════════════════════════════════════════════════════════
            strategy_name = getattr(self.strategy, 'strategy_name', 'unknown')
            self._trade_logger = TradeLogger(
                strategy_name=strategy_name,
                mode=self.mode_name,
                logger=get_logger("modules.trading.trade_logger")
            )
            self.logger.info("📝 TradeLogger is ready")

            # ════════════════════════════════════════════════════════════════
            # Position Recovery (load open trades)
            # ════════════════════════════════════════════════════════════════
            await self._recover_positions()

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

        # Main loop (stops with Ctrl+C)
        try:
            # ════════════════════════════════════════════════════════════════
            # REPLAY MODE: Call ReplayMode.play()
            # ════════════════════════════════════════════════════════════════
            if self.mode_name == 'replay':
                self.logger.info("🎬 Replay Mode is starting...")
                self.logger.info("💡 Controls: +/- (speed), SPACE (pause), q (quit)")

                # play() runs asynchronously and waits for it to finish.
                await self.current_mode.play()

                self.logger.info("🏁 Replay completed")
                self._running = False
                return

            # ════════════════════════════════════════════════════════════════
            # NORMAL MODE (paper/live): Tier-based processing
            # ════════════════════════════════════════════════════════════════
            self.logger.info("💡 Press Ctrl+C to stop")

            # Config'den status interval al (trading.yaml → status_display.status_interval)
            display_interval = self.config.get("status_display.status_interval", 15)
            loop_count = 0

            while self._running:
                loop_count += 1

                # ════════════════════════════════════════════════════════════
                # Price Cache Update - Skipped (PriceFeed handles this via WebSocket)
                # Will be replaced with PriceFeed.update_from_mtf_engines() in Phase 2
                # ════════════════════════════════════════════════════════════

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
