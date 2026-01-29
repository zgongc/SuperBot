# 🔍 Trading Engine V5 - TierManager Entegrasyon Analizi

> **Date:** 2025-12-03
> **Purpose:** To determine the best integration strategy for V5 by learning from the Backtest Engine architecture.

---

## 📊 CURRENT STATUS ANALYSIS

### Trading Engine V5 (639 lines)
```
TradingEngine V5 - Ultra Lean Orchestrator
├── __init__(): 4 Core singleton + lazy components
│   ├── get_logger() ✅
│   ├── get_config() ✅
│   ├── get_event_bus() ✅ (import variable, not actively used)
│   └── get_cache() ✅ (import variable, not in active use)
├── _load_strategy(): Uses StrategyManager.
├── _load_symbols(): Uses SymbolsManager
├── _load_mode(): Dynamic import + BaseMode pattern
├── _start_data_feeds(): WebSocket + MTF setup (COMMENT-OUT)
├── _on_candle_closed(): Mode'a route eder
├── initialize(): Sequential 6-step init
├── start(): Main loop (TODO - only sleep(1))
└── stop(): Clean shutdown
```

### TierManager V5.1 (784 lines)
```
TierManager V5.1 - Mature Tier System
├── TierLevel enum (POSITION=0, DECISION=1, MONITORING=2, ANALYSIS=3)
├── SymbolTierState dataclass (conditions tracking dahil)
├── Config entegrasyonu (trading.yaml'dan okur)
├── EventBus entegrasyonu (tier.change publish)
├── Cache entegrasyonu (tier:summary cache)
├── Interval checking (should_check_tier, get_symbols_to_check)
└── Status reporting (publish_status_report)
```

### DisplayInfo (577 lines)
```
DisplayInfo - Presentation Layer
├── Retrieves data from TierManager
├── format_status_line(): Uptime, time, balance
├── format_tier_summary(): Tier summary
├── format_conditions_verbose(): Condition details
└── format_position_lines(): Position details
```

---

## 🏗️ LESSONS LEARNED FROM THE BACKTEST ENGINE

### 1. Manager Composition Pattern
```python
# Backtest Engine approach
class BacktestEngine:
    def __init__(self):
        # Managers use lazy initialization, the engine only coordinates.
        self.parquets_engine = ParquetsEngine()
        self.risk_manager = RiskManager(logger=self.logger)
        self.position_manager = PositionManager(logger=self.logger)

    async def run(self, strategy):
        # Created during execution
        strategy_executor = StrategyExecutor(strategy, logger=self.logger)
        exit_manager = ExitManager(strategy, logger=self.logger)
```

**For the Trading Engine:**
```python
class TradingEngine:
    def __init__(self):
        # Lazy placeholders
        self.tier_manager: Optional[TierManager] = None
        self.display_info: Optional[DisplayInfo] = None

    async def initialize(self):
        # Create during initialization (after the strategy is loaded)
        self.tier_manager = TierManager(
            logger=self.logger,
            config=self.config,
            event_bus=self.event_bus,
            cache_manager=self.cache
        )
        self.display_info = DisplayInfo(
            tier_manager=self.tier_manager,
            logger=self.logger,
            config=self.config,
            connector=self.connector,
            strategy=self.strategy
        )
```

### 2. Sequential Pipeline Pattern
```
Backtest Flow:
1. BUILD CONFIG ← Strategy object
2. LOAD DATA ← ParquetsEngine
3. CALCULATE INDICATORS ← IndicatorManager (vectorized)
4. GENERATE SIGNALS ← VectorizedConditions
5. SIMULATE POSITIONS ← Single-pass loop
6. CALCULATE METRICS ← BacktestMetrics
7. RETURN RESULT

Trading Flow (Recommended):
1. LOAD STRATEGY ← StrategyManager
2. LOAD SYMBOLS ← SymbolsManager
3. INIT CONNECTOR ← BinanceAPI
4. INIT MANAGERS ← TierManager, RiskManager, IndicatorManager
5. INIT MODE ← BaseMode (paper/live/demo)
6. START DATA FEEDS ← WebSocket + MTF
7. START TIER LOOP ← Main processing loop
```

### 3. Single-Pass Processing (Backtest)
```python
# Backtest: Single pass for each candle
for i in range(warmup, len(data)):
    row = data.iloc[i]
    signal = signals[i]

    # 1. First, check for EXIT
    for position in positions[:]:
        exit_result = strategy_executor.evaluate_exit(...)
        if exit_result['should_exit']:
            close_position(position)

    # 2. Then check ENTRY
    if signal != 0:
        new_position = open_position(...)
        positions.append(new_position)
```

**For the Trading Engine (Tier-Based):**
```python
# Trading: Trading at different intervals depending on the tier.
async def _tier_processing_loop(self):
    while self._running:
        symbols_to_check = self.tier_manager.get_symbols_to_check()

        # TIER 0: Every second (SL/TP tick-based)
        if TierLevel.POSITION in symbols_to_check:
            await self._process_positions(symbols_to_check[TierLevel.POSITION])

        # TIER 1: 5 seconds (Decision - waiting for candle close)
        if TierLevel.DECISION in symbols_to_check:
            await self._process_decisions(symbols_to_check[TierLevel.DECISION])

        # TIER 2: 15 seconds (Monitoring - conditions are being monitored)
        if TierLevel.MONITORING in symbols_to_check:
            await self._process_monitoring(symbols_to_check[TierLevel.MONITORING])

        # TIER 3: 60 seconds (Analysis - new candidates are being scanned)
        if TierLevel.ANALYSIS in symbols_to_check:
            await self._process_analysis(symbols_to_check[TierLevel.ANALYSIS])

        await asyncio.sleep(1)  # Base interval
```

### 4. Exit-First Logic
```python
# Proven in backtesting: EXIT first, ENTRY later
# This sequence is critical - there might be both an output and an input on the same candle.

async def _on_candle_closed(self, symbol: str, timeframe: str):
    """Called when a candle closes"""

    # 1. FIRST: Check if the position exists, then exit.
    tier = self.tier_manager.get_tier(symbol)
    if tier == TierLevel.POSITION:
        await self._check_exit(symbol, timeframe)

    # 2. LATER: Entry control (if in the DECISION tier)
    if tier == TierLevel.DECISION:
        await self._check_entry(symbol, timeframe)
```

---

## 🎯 RECOMMENDED ARCHITECTURE: "LEAN COORDINATOR"

### Principle: The Engine does NOT do the work, it COORDINATES.

```
                    ┌─────────────────────────────────────┐
                    │        TradingEngine V5             │
                    │      (Lean Coordinator)             │
                    │                                     │
                    │  - Component lifecycle              │
                    │  - Event routing                    │
                    │  - Error handling                   │
                    │  - Shutdown coordination            │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │ TierManager │         │ ModeManager │         │ DataManager │
    │             │         │             │         │             │
    │ - Tier state│         │ - Paper     │         │ - WebSocket │
    │ - Intervals │         │ - Live      │         │ - MTF       │
    │ - EventBus  │         │ - Demo      │         │ - Indicators│
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           │         ┌─────────────┼─────────────┐         │
           │         │             │             │         │
           ▼         ▼             ▼             ▼         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      EventBus                               │
    │  tier.change │ candle.closed │ position.opened │ ...        │
    └─────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ DisplayInfo │
    │             │
    │ - Status    │
    │ - Tiers     │
    │ - Positions │
    └─────────────┘
```

### Responsibility Distribution

| Component | Responsibility | EventBus Events |
|-----------|------------|-----------------|
| **TradingEngine** | Lifecycle, routing, shutdown | - |
| **TierManager** | Symbol→Tier mapping, intervals | `tier.change`, `tier.status.report` |
| **Mode (Paper/Live)** | Trade execution, position tracking | `position.opened`, `position.closed` |
| **DataManager** | WS, MTF, indicator subscription | `candle.closed`, `tick.update` |
| **DisplayInfo** | Terminal output formatting | (subscriber only) |
| **StrategyExecutor** | Entry/Exit signal generation | - |

---

## 📋 ENTEGRASYON ADIMLARI

### Step 1: TierManager Integration (Priority: HIGH)

```python
# trading_engine.py changes

# Add import
from modules.trading.tier_manager import TierManager, TierLevel
from modules.trading.display_info import DisplayInfo

class TradingEngine:
    def __init__(self, ...):
        # ... existing code ...

        # Tier system (lazy init)
        self.tier_manager: Optional[TierManager] = None
        self.display_info: Optional[DisplayInfo] = None

    async def initialize(self):
        # ... existing initialization ...

        # ════════════════════════════════════════════════════════════════
        # TierManager initialization (after symbols are loaded)
        # ════════════════════════════════════════════════════════════════
        self.tier_manager = TierManager(
            logger=self.logger,
            config=self.config,
            event_bus=self.event_bus,
            cache_manager=self.cache,
            on_tier_change=self._on_tier_change,
            verbose=self.verbose
        )
        self.tier_manager.initialize(self.symbols)
        self.logger.info(f"📊 TierManager ready: {len(self.symbols)} symbols")

        # ════════════════════════════════════════════════════════════════
        # DisplayInfo init
        # ════════════════════════════════════════════════════════════════
        self.display_info = DisplayInfo(
            tier_manager=self.tier_manager,
            logger=self.logger,
            config=self.config,
            connector=self.connector,
            strategy=self.strategy
        )
        self.logger.info("📺 DisplayInfo is ready")

    def _on_tier_change(self, symbol: str, old_tier: TierLevel, new_tier: TierLevel):
        """Tier change callback"""
        # Engine'de ekstra logic gerekirse buraya
        pass
```

### Step 2: Tier-Based Processing Loop (Priority: HIGH)

```python
async def start(self):
    """Start trading"""
    if not self._initialized:
        raise RuntimeError("TradingEngine initialize edilmedi!")

    self._running = True
    self.logger.info("🚀 TradingEngine started")

    # Background tasks
    tasks = [
        asyncio.create_task(self._tier_processing_loop()),
        asyncio.create_task(self._status_display_loop()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        self.logger.info("🛑 Tasks were cancelled")

async def _tier_processing_loop(self):
    """Tier-based main processing loop"""
    while self._running:
        try:
            # Get the symbols to be checked from TierManager
            symbols_to_check = self.tier_manager.get_symbols_to_check()

            for tier, symbols in symbols_to_check.items():
                if tier == TierLevel.POSITION:
                    await self._process_tier_position(symbols)
                elif tier == TierLevel.DECISION:
                    await self._process_tier_decision(symbols)
                elif tier == TierLevel.MONITORING:
                    await self._process_tier_monitoring(symbols)
                elif tier == TierLevel.ANALYSIS:
                    await self._process_tier_analysis(symbols)

        except Exception as e:
            self.logger.error(f"❌ Tier loop error: {e}")

        await asyncio.sleep(1)  # Base interval

async def _status_display_loop(self):
    """Periodic status display"""
    interval = self.config.get('status_display.status_interval', 15)

    while self._running:
        try:
            # Status line
            status = self.display_info.format_status_line()
            self.logger.info(status)

            # Tier summary
            tier_lines = self.display_info.format_tier_summary(verbose=self.verbose)
            for line in tier_lines:
                self.logger.info(line)

            # If verbose, details of the condition
            if self.verbose:
                condition_lines = self.display_info.format_conditions_verbose()
                for line in condition_lines:
                    self.logger.info(line)

            # EventBus'a status report
            self.tier_manager.publish_status_report()

        except Exception as e:
            self.logger.error(f"❌ Error displaying status: {e}")

        await asyncio.sleep(interval)
```

### Step 3: Tier Processing Methods (Priority: MEDIUM)

```python
async def _process_tier_position(self, symbols: List[str]):
    """
    TIER 0: Active positions (1s interval)

    - SL/TP tick-based kontrol
    - Update trailing stop.
    - Break-even kontrol
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Delegate to mode (PaperMode/LiveMode)
            if hasattr(self.current_mode, 'check_position_exit'):
                await self.current_mode.check_position_exit(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 0 error: {e}")

async def _process_tier_decision(self, symbols: List[str]):
    """
    TIER 1: Decision stage (5s interval)

    - 100% condition met
    - Candle close bekleniyor
    - Send a signal to Mode if the entry is ready.
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            state = self.tier_manager.get_state(symbol)
            if state and state.ready_for_entry:
                # Send an entry signal to the mode.
                if hasattr(self.current_mode, 'execute_entry'):
                    await self.current_mode.execute_entry(
                        symbol=symbol,
                        direction=state.direction,
                        score=state.score
                    )
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 1 error: {e}")

async def _process_tier_monitoring(self, symbols: List[str]):
    """
    TIER 2: Monitoring phase (15s interval)

    - Condition met with 50% or more.
    - Re-evaluate the conditions.
    - Check for promotion to DECISION.
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Re-evaluate the conditions
            await self._evaluate_conditions(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 2 error: {e}")

async def _process_tier_analysis(self, symbols: List[str]):
    """
    TIER 3: Analysis phase (60s interval)

    - Scan for new candidates
    - Check for promotion to MONITORING
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Evaluate conditions
            await self._evaluate_conditions(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 3 error: {e}")

async def _evaluate_conditions(self, symbol: str):
    """
    Evaluate conditions for the symbol and update the tier.

    Uses StrategyExecutor, reports the result to TierManager.
    """
    if not self._strategy_executor:
        return

    # Indicator data al (IndicatorManager'dan)
    indicator_data = await self._get_indicator_data(symbol)
    if not indicator_data:
        return

    # Evaluate conditions
    result = self._strategy_executor.evaluate_entry(
        symbol=symbol,
        data=indicator_data,
        current_price=indicator_data.get('close', 0)
    )

    # Tier hesapla
    score = result.get('score', 0)
    direction = result.get('direction')
    conditions_long = result.get('conditions_long', [])
    conditions_short = result.get('conditions_short', [])

    # Determine the tier based on thresholds
    thresholds = self.tier_manager.thresholds

    if score >= thresholds.get('decision', 1.0):
        new_tier = TierLevel.DECISION
    elif score >= thresholds.get('monitoring', 0.5):
        new_tier = TierLevel.MONITORING
    else:
        new_tier = TierLevel.ANALYSIS

    # Update the TierManager
    self.tier_manager.set_tier(
        symbol=symbol,
        tier=new_tier,
        score=score,
        direction=direction,
        conditions_long=conditions_long,
        conditions_short=conditions_short,
        conditions_met=result.get('conditions_met', 0),
        conditions_total=result.get('conditions_total', 0)
    )
```

### Step 4: Candle Callback Integration (Priority: HIGH)

```python
async def _on_candle_closed(self, symbol: str, timeframe: str):
    """
    Called when a candle closes (MTF Engine -> TradingEngine)

    Exit-First Logic:
    1. FIRST: Check if there is a position, then check the exit condition.
    2. THEN: Check the entry condition.
    """
    if not self.current_mode or not self.tier_manager:
        return

    try:
        tier = self.tier_manager.get_tier(symbol)

        # 1. FIRST EXIT (POSITION tier)
        if tier == TierLevel.POSITION:
            if hasattr(self.current_mode, 'on_candle_closed'):
                await self.current_mode.on_candle_closed(symbol, timeframe)

        # 2. Re-evaluate the conditions
        await self._evaluate_conditions(symbol)

        # 3. In the DECISION tier, perform entry validation.
        tier = self.tier_manager.get_tier(symbol)  # Get the current tier
        state = self.tier_manager.get_state(symbol)

        if tier == TierLevel.DECISION and state:
            # Is the candle closed? Is the entry ready?
            state.candle_close_pending = False
            state.ready_for_entry = True

            # Entry execute
            if hasattr(self.current_mode, 'execute_entry'):
                entry_result = await self.current_mode.execute_entry(
                    symbol=symbol,
                    direction=state.direction,
                    score=state.score
                )

                if entry_result and entry_result.get('success'):
                    # Upgrade to the POSITION tier
                    self.tier_manager.set_tier(
                        symbol=symbol,
                        tier=TierLevel.POSITION,
                        direction=state.direction,
                        score=state.score
                    )

    except Exception as e:
        self.logger.error(f"❌ {symbol}: Candle callback error: {e}")
```

---

## 🔄 DATA FLOW DIAGRAM (NEW - UPDATED)

### Critical Understanding: There are two different data streams.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1️⃣ REAL-TIME DATA FLOW                               │
│                     (WebSocket → Indicator → Cache)                         │
└─────────────────────────────────────────────────────────────────────────────┘

Binance WebSocket
       │
       │ kline_1m (Updates are received every second)
       ▼
┌──────────────────┐
│ WebSocketEngine  │
│                  │
│ - Parse kline    │
│ - Emit events    │
└────────┬─────────┘
         │
         │ EventBus: "kline.update.BTCUSDT.1m"
         ▼
┌──────────────────┐
│ MTF Engine       │──────────────────────────────────────┐
│ (per symbol)     │                                      │
│                  │                                      │
│ 1m buffer        │  Aggregation:                        │
│ ┌────────────┐   │  1m × 5  → 5m candle                 │
│ │ O H L C V  │   │  1m × 15 → 15m candle                │
│ │ O H L C V  │   │  1m × 60 → 1h candle                 │
│ │ O H L C V  │   │  ...                                 │
│ │ ...        │   │                                      │
│ └────────────┘   │                                      │
└────────┬─────────┘                                      │
         │                                                │
         │ Her 1m kline update'inde                       │
         ▼                                                │
┌──────────────────┐                                      │
│IndicatorManager  │◄─────────────────────────────────────┘
│                  │  Warmup: First N candle indicator
│ - RSI            │ required for calculation
│ - EMA            │ (e.g., minimum 14 candles for RSI_14)
│ - Bollinger      │
│ - ATR            │  ┌─────────────────────────────────┐
│ - ...            │  │ WARMUP STATUS                   │
└────────┬─────────┘  │                                 │
         │            │ warmup_complete = False         │
         │            │ → Indicator hesaplanmaz         │
         │            │ -> Tier check is not performed           │
         │            │                                 │
         │            │ warmup_complete = True          │
         │            │ -> Indicator is calculated          │
         │            │ -> Tier check begins             │
         │            └─────────────────────────────────┘
         │
         │ Calculated values
         ▼
┌──────────────────┐
│ CacheManager     │
│                  │
│ indicators:      │
│   BTCUSDT:5m:    │
│     rsi_14: 45.2 │
│     ema_20: 42100│
│     bb_upper:... │
│                  │
│ TTL: 60s         │
└──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     2️⃣ TIER EVALUATION FLOW                                 │
│              (Polling-based, independent of candle_closed!)                   │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────────┐
                              │   _tier_processing_loop()   │
                              │      (main loop)            │
                              │                             │
                              │   while running:            │
                              │     check_tiers()           │
                              │     await sleep(1)          │
                              └──────────────┬──────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
    │ TIER 3: ANALYSIS│           │TIER 2: MONITOR  │           │TIER 1: DECISION │
    │   (60s interval)│           │  (15s interval) │           │   (5s interval) │
    │                 │           │                 │           │                 │
    │ All symbols | | %50+ condition | | %100 condition |
    │ scanned         │           │ provided        │           │ provided        │
    │                 │           │ symbols       │           │ symbols       │
    └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
             │                             │                             │
             │                             │                             │
             ▼                             ▼                             ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        _evaluate_conditions(symbol)                      │
    │                                                                          │
    │  1. CacheManager'dan indicator verilerini al                            │
    │     indicators = cache.get(f"indicators:{symbol}:{timeframe}")          │
    │                                                                          │
    │  2. Evaluate conditions with StrategyExecutor.
    │     result = strategy_executor.evaluate_entry(symbol, indicators)        │
    │                                                                          │
    │  3. Determine a new tier based on the score.
    │     score >= 1.0  → DECISION                                            │
    │     score >= 0.5  → MONITORING                                          │
    │     score < 0.5   → ANALYSIS                                            │
    │                                                                          │
    │  4. Update the TierManager
    │     tier_manager.set_tier(symbol, new_tier, score, direction, ...)      │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     3️⃣ TRADE EXECUTION FLOW                                 │
│                  (candle_closed ONLY IMPORTANT HERE!)                      │
└─────────────────────────────────────────────────────────────────────────────┘

MTF Engine
    │
    │ candle_closed event (5m candle closed!)
    │ (Only for the primary_timeframe)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│              _on_candle_closed(symbol, timeframe)                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ ONLY for TIER 0 (POSITION) and TIER 1 (DECISION)!        │  │
│  │                                                             │  │
│  │ TIER 0 -> Exit check (SL/TP at candle closing price?)      │  │
│  │ TIER 1 -> Entry execute (are the conditions still 100%? -> TRADE!)   │  │
│  │                                                             │  │
│  │ TIER 2/3 -> NOTHING TO DO                                  │  │
│  │           (tier_processing_loop already checks this)       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  if tier == POSITION:                                            │
│      mode.check_exit_on_candle_close(symbol)                     │
│      # Was the stop-loss/take-profit triggered? Is there a signal output?               │
│                                                                  │
│  elif tier == DECISION:                                          │
│      # The candle closed, entry time!
│      mode.execute_entry(symbol, direction, score)                │
│      if success:                                                 │
│          tier_manager.set_tier(symbol, POSITION)                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    4️⃣ POSITION MANAGEMENT FLOW                              │
│                      (TIER 0 - Tick-based SL/TP)                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────────┐
                              │   _tier_processing_loop()   │
                              │                             │
                              │   TIER 0: 1s interval       │
                              │   (HIGHEST PRIORITY)       │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │  _process_tier_position()   │
                              │                             │
                              │  for symbol in positions:   │
                              │    current_price = get()    │
                              │    check_sl_tp(price)       │
                              │    check_trailing()         │
                              │    check_breakeven()        │
                              └──────────────┬──────────────┘
                                             │
                            ┌────────────────┴────────────────┐
                            │                                 │
                            ▼                                 ▼
                   SL/TP tetiklendi?                  Trailing update?
                            │                                 │
                            ▼                                 ▼
                   mode.close_position()              mode.update_sl()
                   tier → get_return_tier()
                   (from config: ANALYSIS or MONITORING)
```

---

## 🎯 1D STRATEGY SCENARIO

**Question:** In the 1D strategy, is the tier check performed 24 hours after the candle_closed event?

**Answer:** NO! The tier check operates **independently** of 'candle_closed'.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1D STRATEGY EXAMPLE FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

Time 00:00 - Start of the day
│
├── WebSocket: 1m kline data is coming (continuously)
│
├── MTF Engine: 1m -> 1D aggregation (1440 adet 1m = 1 adet 1D)
│   └── The 1D candle has not yet CLOSED, but the OHLC is constantly updated!
│
├── IndicatorManager: 1D indicators are recalculated every 1M UPDATE.
│   └── RSI_14_1d, EMA_20_1d, BB_1d, ATR_1d...
│   └── (Calculated with the O, H, L, C values of the open candle)
│
├── CacheManager: Current indicator values are in the cache.
│
└── _tier_processing_loop():
    │
    ├── TIER 3 check (her 60s):
    │   └── BTCUSDT conditions increased to 60% -> MONITORING
    │
    ├── TIER 2 check (her 15s):
    │   └── BTCUSDT conditions 85% -> still MONITORING
    │
    ├── TIER 2 check (her 15s):
    │   └── BTCUSDT conditions increased to 100% -> upgrade to DECISION!
    │
    └── TIER 1 check (her 5s):
        └── BTCUSDT 100% condition MET but...
            ├── candle_close_pending = True (candle has not yet closed)
            └── Entry YAPILMAZ, bekle!

23:59:59 - End of day (1D candle is closing!)
│
└── MTF Engine: candle_closed event ("BTCUSDT", "1d")
    │
    └── _on_candle_closed("BTCUSDT", "1d"):
        │
        └── tier == DECISION and candle_close_pending == True
            │
            └── candle_close_pending = False
            └── ready_for_entry = True
            └── mode.execute_entry("BTCUSDT", "LONG", 1.0)
            └── tier → POSITION
```

### Summary:

| Operation | When does it happen? | Is Candle Close required? |
|-------|----------------|--------------------------|
| Indicator calculation | Every 1m update | ❌ No |
| Tier 3 to 2 transition | Every 60 seconds polling | ❌ No |
| Tier 2 to 1 transition | Every 15 seconds polling | ❌ No |
| **Entry execute** | **Candle close immediately** | **✅ YES** |
| SL/TP control | Every 1s polling | ❌ No |
| **Exit execute** | Tick-based or candle close | **Depending on the situation** |

---

## 📊 WARMUP AND INDICATOR FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WARMUP PROCESS                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Engine started
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. CALCULATE WARMUP PERIOD                                       │
│                                                                  │
│ warmup_period = max(                                             │
│     indicator.required_periods for indicator in strategy         │
│ ) + buffer                                                       │
│                                                                  │
│ Example:                                                           │
│   RSI_14      → 14 candle                                        │
│   EMA_50      → 50 candle                                        │
│   BB_20       → 20 candle                                        │
│   ATR_14      → 14 candle                                        │
│   ─────────────────────                                          │
│   warmup_period = 50 + 10 (buffer) = 60 candle                   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. LOAD HISTORICAL DATA (for warmup) │
│                                                                  │
│ for symbol in symbols:                                           │
│     for timeframe in mtf_timeframes:                             │
│         # From Parquet or from API
│         historical = connector.get_klines(                       │
│             symbol=symbol,                                       │
│             timeframe=timeframe,                                 │
│             limit=warmup_period                                  │
│         )                                                        │
│                                                                  │
│         # Load into the MTF Engine                                     │
│         mtf_engine.load_historical(historical)                   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. INDICATOR WARMUP                                              │
│                                                                  │
│ indicator_manager.warmup(historical_data)                        │
│                                                                  │
│ # Each indicator performs its own warmup.
│ # The first N values may be NaN, this is normal
│                                                                  │
│ RSI_14:   [NaN, NaN, ..., NaN, 45.2, 48.1, 52.3, ...]            │
│            ▲─── 14 candle ───▲                                   │
│                                                                  │
│ warmup_complete = True                                           │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. START WEBSOCKET                                              │
│                                                                  │
│ # Now you can receive live data
│ websocket_engine.subscribe(symbols, channels)                    │
│                                                                  │
│ # Every time a new 1m Kline arrives:                                  │
│ # - Update MTF buffer
│ # - Indicator incremental hesapla                                │
│ # - Update cache
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. START TIER PROCESSING                                        │
│                                                                  │
│ # After the warmup is complete, the tier check begins.
│ _tier_processing_loop()                                          │
│                                                                  │
│ # If warmup is NOT complete:                                       │
│ # - Tier check is not performed
│ # - No entry or exit operations are allowed.
│ # - Only data is collected
└──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    INDICATOR INCREMENTAL UPDATE                              │
└─────────────────────────────────────────────────────────────────────────────┘

WebSocket: A new 1m kline has arrived.
       │
       ▼
MTF Engine: Update 1m buffer
       │
       ├── is_candle_update? (Is the candle still open?)
       │   │
       │   └── Only update the current candle's OHLC.
       │       └── Indicators are recalculated with OPEN candle values.
       │
       └── is_candle_close? (is candle closed?)
           │
           └── Add a new candle to the buffer
               └── Mark the old candle as "closed".
               └── emit the candle_closed event


┌─────────────────────────────────────────────────────────────────────────────┐
│                       CACHE YAPISI                                           │
└─────────────────────────────────────────────────────────────────────────────┘

CacheManager
│
├── indicators:{symbol}:{timeframe}
│   │
│   └── {
│         "rsi_14": 45.234,
│         "ema_5": 42150.50,
│         "ema_20": 42089.30,
│         "bb_upper": 42500.00,
│         "bb_middle": 42100.00,
│         "bb_lower": 41700.00,
│         "atr_14": 350.25,
│         "close": 42180.00,
│         "high": 42250.00,
│         "low": 42050.00,
│         "open": 42100.00,
│         "volume": 15234.56,
│         "timestamp": 1701619200000,
│         "is_closed": false  # Is it an open candle?
│       }
│
├── tier:summary
│   │
│   └── {
│         "counts": {0: 2, 1: 5, 2: 15, 3: 378},
│         "display": "T0:2 | T1:5 | T2:15 | T3:378",
│         "timestamp": "2025-12-03T..."
│       }
│
└── tier:state:{symbol}
    │
    └── {
          "symbol": "BTCUSDT",
          "tier": 1,
          "score": 0.85,
          "direction": "LONG",
          "conditions_met": 5,
          "conditions_total": 6,
          ...
        }
```

---

## ⚠️ IMPORTANT POINTS

### 1. Thread Safety
```python
# The TierManager._states dictionary is not thread-safe.
# If multi-threading is to be used:
import threading
self._lock = threading.Lock()

def set_tier(self, ...):
    with self._lock:
        # ... tier update ...
```

### 2. Async/Await Consistency
```python
# All I/O operations should be asynchronous.
# Blocking calls should be wrapped with asyncio.to_thread()

# ❌ Wrong
result = self.connector.get_balance()  # Blocking!

# ✅ Correct
result = await self.connector.get_balance()  # or
result = await asyncio.to_thread(self.connector.get_balance)
```

### 3. Error Isolation
```python
# Every symbol operation should be within a try/except block.
# A symbol error should not affect the others.

for symbol in symbols:
    try:
        await self._process_symbol(symbol)
    except Exception as e:
        self.logger.error(f"❌ {symbol} error: {e}")
        # Continue, don't stop
```

### 4. Graceful Shutdown
```python
async def stop(self):
    """Clean shutdown"""
    self._running = False

    # 1. Stop new operations
    # 2. Save open positions (for crash recovery)
    # 3. Close WebSocket
    # 4. Mode shutdown
    # 5. Cache flush
```

---

## 📊 COMPARISON: BEFORE vs AFTER

### BEFORE (V5 exists)
```
TradingEngine V5
├── 4 Core singleton ✅
├── Strategy loading ✅
├── Symbol loading ✅
├── Mode loading ✅
├── Data feeds (COMMENT-OUT) ⚠️
├── Tier system ❌
├── Processing loop ❌ (only sleep)
├── Display ❌
└── EventBus/Cache is actively used ❌
```

### LATER (V5 + Integration)
```
TradingEngine V5 + TierManager
├── 4 Core singleton ✅
├── Strategy loading ✅
├── Symbol loading ✅
├── Mode loading ✅
├── Data feeds ✅ (active)
├── TierManager ✅ (entegre)
├── DisplayInfo ✅ (entegre)
├── Tier-based processing loop ✅
├── EventBus is actively used ✅
├── Cache is actively being used ✅
└── Exit-First logic ✅
```

---

## 🚀 APPLICATION PRIORITIES

| Priority | Step | Estimated Lines | Dependency |
|---------|------|---------------|------------|
| 1 | TierManager import & init | +30 | - |
| 2 | DisplayInfo import & init | +20 | TierManager |
| 3 | _tier_processing_loop | +60 | TierManager |
| 4 | _status_display_loop | +40 | DisplayInfo |
| 5 | Tier processing methods | +80 | Mode |
| 6 | update _on_candle_closed | +40 | StrategyExecutor |
| 7 | Enable data feeds | +10 | WebSocket, MTF |
| 8 | stop() update | +20 | - |

**Total:** ~300 lines added -> V5 will have 939 lines (still less than V4!)

---

## 🎯 RESULT

### Recommended Approach: "Incremental Integration"

1. **First, integrate TierManager + DisplayInfo** (basic visibility).
2. Add the "Next Tier Loop" (processing logic).
3. **Finally, activate Data Feeds** (for a fully functional system).

This approach:
- Can be tested at each step.
- Backward compatible
- Uses Backtest Engine's proven patterns.
- Engine maintains the "lean coordinator" role.

---

## 🔬 V1 ANALYSIS: Real-Time Evaluation

V1'de işleme şu şekilde:
In V1, the processing is as follows:

```python
# V1 _main_loop() - trading_engine_v1.py:1644-1676
async def _main_loop(self):
    while self.is_running:
        loop_count += 1

        # Real-time evaluation (10 saniyede bir)
        if loop_count % 10 == 0:
            await self._realtime_evaluation()  # For ALL symbols

        # Status log (60 saniyede bir)
        if loop_count % 60 == 0:
            self.display_trading_info()

        # Tier status (15 saniyede bir)
        elif loop_count % 15 == 0:
            self.display_live_status()

        await asyncio.sleep(1)
```

### V1's Stop Loss/Take Profit Control
```python
# V1 _evaluate_exits_for_symbol() - trading_engine_v1.py:1281-1343
async def _evaluate_exits_for_symbol(self, symbol, indicator_data):
    # Get current price from DataFrame (LAST CANDLE value)
    current_price = indicator_data[primary_tf]['close'].iloc[-1]

    for position in positions:
        exit_result = strategy_executor.evaluate_exit(
            position=position,
            current_price=current_price  # DataFrame'den!
        )

        if exit_result.get('should_exit'):
            await self._close_position(position, current_price, reason)
```

### V1 Problem: Not Tick-Based!
```
❌ V1: Get the latest close price from the DataFrame every 10 seconds.
       -> In SL $100, the price dropped to $99, but 10 seconds later it rose to $101.
       -> SL MISS! Because when checked, the price was $101.

✅ WHAT SHOULD HAPPEN: Check at every tick (every price update).
       -> In SL $100, the price dropped to $99.
       -> IMMEDIATE output (at the moment the tick arrives)
```

---

## 🎯 NEW PROPOSAL: Hybrid Model

### Two Different Price Sources

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TICK DATA vs CANDLE DATA                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. TICK DATA (WebSocket aggTrade/bookTicker)
   └── Income on every price change
   └── Used for stop-loss/take-profit control.
   └── Indicator hesaplamaz

2. CANDLE DATA (WebSocket kline)
   └── Revenue for each mom update (1s)
   └── Used for indicator calculation
   └── Used for entry/exit signal.
```

### Recommended Flow (Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID DATA FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

WebSocket
    │
    ├── kline_1m (candle updates) ───────────────────────────────────────┐
    │                                                                     │
    │   └── MTF Engine → IndicatorManager → Cache                        │
    │       └── Tier evaluation (polling, interval-based)                │
    │       └── Entry signals (candle close event)                       │
    │                                                                     │
    └── bookTicker/aggTrade (tick updates) ──────────────────────────────┤
                                                                          │
        └── PriceStream (new component)
            └── Only position check is performed at each tick.
            └── SL/TP hit check                                          │
            └── Cache'e current_price yaz                                │
                                                                          │
                                                                          ▼
                    ┌────────────────────────────────────────────────────┐
                    │              POSITION CHECK (Tick-Based)           │
                    │                                                    │
                    │  for position in active_positions:                 │
                    │      if current_price <= position.sl_price:        │
                    │          IMMEDIATE EXIT! (market order)            │
                    │      elif current_price >= position.tp_price:      │
                    │          IMMEDIATE EXIT! (market order)            │
                    │                                                    │
                    │  Trailing stop update de burada                    │
                    └────────────────────────────────────────────────────┘
```

---

## 📊 DECISION TABLE: When to Use What?

| Operation | Data Source | Check Time | Timeout |
|-------|--------------|----------------|---------|
| **Indicator calculation** | kline (candle) | Every candle update | ❌ |
| **Tier evaluation** | Cache (indicators) | Polling (interval or continuous) | ❌ |
| **Entry signal** | Cache (indicators) | Candle close event | ✅ Candle close |
| **SL/TP check** | bookTicker (tick) | Her tick | ❌ |
| **Trailing update** | bookTicker (tick) | Her tick | ❌ |
| **Signal exit** | kline (candle) | Candle close event | ✅ Candle close |

### Difference between Entry and Exit

```
ENTRY:
└── Candle CLOSE is required (for signal validation)
└── Trading with open candle data is risky.
└── "Close > EMA" condition is finalized when the candle closes.

EXIT (SL/TP):
└── Candle close BEKLEMEZ!
└── Exit when the price reaches the stop loss (SL) level.
└── Every millisecond is important (the loss can increase)

EXIT (Signal-based):
└── Candle close is required.
└── "Exit when RSI is greater than 70" -> check when the candle closes
```

---

## 🔧 APPLICATION SUGGESTION

### Option 1: Simple (Like the current V1, interval-based)
```python
# Pros: Kolay implement, az complexity
# Cons: Stop loss/take profit may be delayed (1-10 seconds)

async def _tier_processing_loop(self):
    while self._running:
        # TIER 0: Every second (SL/TP interval-based)
        await self._check_positions()

        # TIER 1-3: According to the interval
        symbols_to_check = self.tier_manager.get_symbols_to_check()
        ...
        await asyncio.sleep(1)
```

### Option 2: Hybrid (Tick + Candle)
```python
# Pros: Real-time stop loss/take profit.
# Cons: More WebSocket subscriptions, complexity

# Separate tick stream
async def _on_tick(self, symbol: str, price: float):
    """Called on each price update"""
    for position in self._get_positions(symbol):
        if self._check_sl_tp(position, price):
            await self._immediate_exit(position, price)

# Separate candle stream
async def _on_candle_closed(self, symbol: str, timeframe: str):
    """Called when the candle closes - entry and signal exit"""
    ...
```

### Option 3: Order-Based (Leave it to the exchange)
```python
# Pros: Most reliable, exchange guaranteed
# Cons: Does not work in paper mode, less control

# Send SL/TP orders along with the entry.
async def execute_entry(self, symbol, direction):
    # Main order
    entry_order = await self.connector.create_order(...)

    # SL order (stop-market)
    sl_order = await self.connector.create_order(
        type='STOP_MARKET',
        stopPrice=sl_price,
        closePosition=True
    )

    # TP order (take-profit-market)
    tp_order = await self.connector.create_order(
        type='TAKE_PROFIT_MARKET',
        stopPrice=tp_price,
        closePosition=True
    )
```

---

## 🎯 RECOMMENDED APPROACH

### Paper Mode: Option 1 (Interval-based)
- Checking the price every second is sufficient.
- Millisecond precision is unnecessary for the simulation.
- Implement edilmesi kolay

### Live Mode: Option 3 (Order-based) + Option 2 backup
- Send a SL/TP order to the exchange (OCO or separately).
- Backup as a tick-based check (for connection loss)
- The safest approach

```python
class TradingEngine:
    async def _handle_entry_success(self, symbol, position):
        """When the entry is successful"""

        if self.mode_name == 'live':
            # Send SL/TP order to the exchange
            await self._place_sl_tp_orders(symbol, position)
        else:
            # Paper mode: Add to TierManager, will be checked with polling
            self.tier_manager.set_tier(symbol, TierLevel.POSITION, ...)
```

---

## 📝 SUMMARY: What Will Happen When

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   WebSocket     │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
           ┌────────────────┐                   ┌────────────────┐
           │  kline_1m      │                   │  bookTicker    │
           │  (candle data) │                   │  (tick data)   │
           └───────┬────────┘                   └───────┬────────┘
                   │                                    │
                   ▼                                    │
           ┌────────────────┐                          │
           │  MTF Engine    │                          │
           │  + Indicators  │                          │
           └───────┬────────┘                          │
                   │                                    │
                   ▼                                    │
           ┌────────────────┐                          │
           │  CacheManager  │◄─────────────────────────┘
           │                │     current_price
           │  indicators +  │
           │  prices        │
           └───────┬────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐     ┌───────────────┐
│ Tier Loop     │     │ Position Loop │
│ (polling)     │     │ (1s interval) │
│               │     │               │
│ - Evaluate    │     │ - SL/TP check │
│ - Tier update │     │ - Trailing    │
└───────┬───────┘     └───────┬───────┘
        │                     │
        ▼                     │
┌───────────────┐             │
│ candle_closed │             │
│ event         │             │
│               │             │
│ - Entry exec  │             │
│ - Signal exit │             │
└───────────────┘             │
                              │
                              ▼
                      ┌───────────────┐
                      │ EXIT (SL/TP)  │
                      │ - Immediate   │
                      │ - Market order│
                      └───────────────┘
```

### Interval Status (`intervals.enabled`)

```yaml
# trading.yaml
tiers:
  intervals:
    enabled: false   # For the small symbol list
    # enabled: true  # Source code optimization for 100+ symbols
```

**enabled: false** -> All tiers are checked in each iteration (ideal for 20 symbols)
**enabled: true** -> Different interval based on tier (for 100+ symbols)

---

**Analiz Tarihi:** 2025-12-03
**Update:** V1 analysis, Hybrid model suggestion, SL/TP tick-based explanation.
**Analiz Eden:** Claude AI Assistant
**Reference:** TRADING_ENGINE_ANALYSIS.md (Comparison of V1-V4)
