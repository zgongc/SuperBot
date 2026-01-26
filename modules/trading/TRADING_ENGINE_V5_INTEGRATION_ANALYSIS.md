# 🔍 Trading Engine V5 - TierManager Entegrasyon Analizi

> **Tarih:** 2025-12-03
> **Amaç:** Backtest Engine mimarisinden öğrenerek V5 için en iyi entegrasyon stratejisini belirlemek

---

## 📊 MEVCUT DURUM ANALİZİ

### Trading Engine V5 (639 satır)
```
TradingEngine V5 - Ultra Lean Orchestrator
├── __init__(): 4 Core singleton + lazy components
│   ├── get_logger() ✅
│   ├── get_config() ✅
│   ├── get_event_bus() ✅ (import var, aktif kullanım YOK)
│   └── get_cache() ✅ (import var, aktif kullanım YOK)
├── _load_strategy(): StrategyManager kullanır
├── _load_symbols(): SymbolsManager kullanır
├── _load_mode(): Dynamic import + BaseMode pattern
├── _start_data_feeds(): WebSocket + MTF setup (COMMENT-OUT)
├── _on_candle_closed(): Mode'a route eder
├── initialize(): Sequential 6-step init
├── start(): Main loop (TODO - sadece sleep(1))
└── stop(): Clean shutdown
```

### TierManager V5.1 (784 satır)
```
TierManager V5.1 - Olgun Tier Sistemi
├── TierLevel enum (POSITION=0, DECISION=1, MONITORING=2, ANALYSIS=3)
├── SymbolTierState dataclass (conditions tracking dahil)
├── Config entegrasyonu (trading.yaml'dan okur)
├── EventBus entegrasyonu (tier.change publish)
├── Cache entegrasyonu (tier:summary cache)
├── Interval checking (should_check_tier, get_symbols_to_check)
└── Status reporting (publish_status_report)
```

### DisplayInfo (577 satır)
```
DisplayInfo - Presentation Layer
├── TierManager'dan veri alır
├── format_status_line(): Uptime, time, balance
├── format_tier_summary(): Tier özeti
├── format_conditions_verbose(): Koşul detayları
└── format_position_lines(): Pozisyon detayları
```

---

## 🏗️ BACKTEST ENGINE'DEN ÖĞRENECEKLER

### 1. Manager Composition Pattern
```python
# Backtest Engine yaklaşımı
class BacktestEngine:
    def __init__(self):
        # Manager'lar lazy init, engine sadece koordine eder
        self.parquets_engine = ParquetsEngine()
        self.risk_manager = RiskManager(logger=self.logger)
        self.position_manager = PositionManager(logger=self.logger)

    async def run(self, strategy):
        # Execution sırasında oluştur
        strategy_executor = StrategyExecutor(strategy, logger=self.logger)
        exit_manager = ExitManager(strategy, logger=self.logger)
```

**Trading Engine için:**
```python
class TradingEngine:
    def __init__(self):
        # Lazy placeholders
        self.tier_manager: Optional[TierManager] = None
        self.display_info: Optional[DisplayInfo] = None

    async def initialize(self):
        # Initialize sırasında oluştur (strategy yüklendikten sonra)
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

Trading Flow (Önerilen):
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
# Backtest: Her candle için tek geçiş
for i in range(warmup, len(data)):
    row = data.iloc[i]
    signal = signals[i]

    # 1. Önce EXIT kontrol
    for position in positions[:]:
        exit_result = strategy_executor.evaluate_exit(...)
        if exit_result['should_exit']:
            close_position(position)

    # 2. Sonra ENTRY kontrol
    if signal != 0:
        new_position = open_position(...)
        positions.append(new_position)
```

**Trading Engine için (Tier-Based):**
```python
# Trading: Tier'a göre farklı interval'larda işlem
async def _tier_processing_loop(self):
    while self._running:
        symbols_to_check = self.tier_manager.get_symbols_to_check()

        # TIER 0: Her saniye (SL/TP tick-based)
        if TierLevel.POSITION in symbols_to_check:
            await self._process_positions(symbols_to_check[TierLevel.POSITION])

        # TIER 1: 5 saniye (Decision - candle close bekleniyor)
        if TierLevel.DECISION in symbols_to_check:
            await self._process_decisions(symbols_to_check[TierLevel.DECISION])

        # TIER 2: 15 saniye (Monitoring - koşullar izleniyor)
        if TierLevel.MONITORING in symbols_to_check:
            await self._process_monitoring(symbols_to_check[TierLevel.MONITORING])

        # TIER 3: 60 saniye (Analysis - yeni adaylar taranıyor)
        if TierLevel.ANALYSIS in symbols_to_check:
            await self._process_analysis(symbols_to_check[TierLevel.ANALYSIS])

        await asyncio.sleep(1)  # Base interval
```

### 4. Exit-First Logic
```python
# Backtest'te kanıtlanmış: EXIT önce, ENTRY sonra
# Bu sıra kritik - aynı mumda hem çıkış hem giriş olabilir

async def _on_candle_closed(self, symbol: str, timeframe: str):
    """Candle kapandığında çağrılır"""

    # 1. ÖNCE: Pozisyon varsa exit kontrol
    tier = self.tier_manager.get_tier(symbol)
    if tier == TierLevel.POSITION:
        await self._check_exit(symbol, timeframe)

    # 2. SONRA: Entry kontrol (DECISION tier'da ise)
    if tier == TierLevel.DECISION:
        await self._check_entry(symbol, timeframe)
```

---

## 🎯 ÖNERİLEN MİMARİ: "LEAN COORDINATOR"

### Prensip: Engine İş YAPMAZ, Koordine EDER

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

### Sorumluluk Dağılımı

| Component | Sorumluluk | EventBus Events |
|-----------|------------|-----------------|
| **TradingEngine** | Lifecycle, routing, shutdown | - |
| **TierManager** | Symbol→Tier mapping, intervals | `tier.change`, `tier.status.report` |
| **Mode (Paper/Live)** | Trade execution, position tracking | `position.opened`, `position.closed` |
| **DataManager** | WS, MTF, indicator subscription | `candle.closed`, `tick.update` |
| **DisplayInfo** | Terminal output formatting | (subscriber only) |
| **StrategyExecutor** | Entry/Exit signal generation | - |

---

## 📋 ENTEGRASYON ADIMLARI

### Adım 1: TierManager Entegrasyonu (Öncelik: YÜKSEK)

```python
# trading_engine.py değişiklikleri

# Import ekle
from modules.trading.tier_manager import TierManager, TierLevel
from modules.trading.display_info import DisplayInfo

class TradingEngine:
    def __init__(self, ...):
        # ... mevcut kod ...

        # Tier system (lazy init)
        self.tier_manager: Optional[TierManager] = None
        self.display_info: Optional[DisplayInfo] = None

    async def initialize(self):
        # ... mevcut initialization ...

        # ════════════════════════════════════════════════════════════════
        # TierManager init (symbols yüklendikten sonra)
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
        self.logger.info(f"📊 TierManager hazır: {len(self.symbols)} sembol")

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
        self.logger.info("📺 DisplayInfo hazır")

    def _on_tier_change(self, symbol: str, old_tier: TierLevel, new_tier: TierLevel):
        """Tier değişiklik callback'i"""
        # Engine'de ekstra logic gerekirse buraya
        pass
```

### Adım 2: Tier-Based Processing Loop (Öncelik: YÜKSEK)

```python
async def start(self):
    """Trading başlat"""
    if not self._initialized:
        raise RuntimeError("TradingEngine initialize edilmedi!")

    self._running = True
    self.logger.info("🚀 TradingEngine başlatıldı")

    # Background tasks
    tasks = [
        asyncio.create_task(self._tier_processing_loop()),
        asyncio.create_task(self._status_display_loop()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        self.logger.info("🛑 Tasks iptal edildi")

async def _tier_processing_loop(self):
    """Tier-based ana işleme döngüsü"""
    while self._running:
        try:
            # TierManager'dan kontrol edilecek sembolleri al
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
            self.logger.error(f"❌ Tier loop hatası: {e}")

        await asyncio.sleep(1)  # Base interval

async def _status_display_loop(self):
    """Periyodik status gösterimi"""
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

            # Verbose ise koşul detayları
            if self.verbose:
                condition_lines = self.display_info.format_conditions_verbose()
                for line in condition_lines:
                    self.logger.info(line)

            # EventBus'a status report
            self.tier_manager.publish_status_report()

        except Exception as e:
            self.logger.error(f"❌ Status display hatası: {e}")

        await asyncio.sleep(interval)
```

### Adım 3: Tier İşleme Metodları (Öncelik: ORTA)

```python
async def _process_tier_position(self, symbols: List[str]):
    """
    TIER 0: Aktif pozisyonlar (1s interval)

    - SL/TP tick-based kontrol
    - Trailing stop güncelleme
    - Break-even kontrol
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Mode'a delege et (PaperMode/LiveMode)
            if hasattr(self.current_mode, 'check_position_exit'):
                await self.current_mode.check_position_exit(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 0 hatası: {e}")

async def _process_tier_decision(self, symbols: List[str]):
    """
    TIER 1: Karar aşaması (5s interval)

    - %100 koşul sağlandı
    - Candle close bekleniyor
    - Entry hazırsa Mode'a sinyal gönder
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            state = self.tier_manager.get_state(symbol)
            if state and state.ready_for_entry:
                # Mode'a entry sinyali gönder
                if hasattr(self.current_mode, 'execute_entry'):
                    await self.current_mode.execute_entry(
                        symbol=symbol,
                        direction=state.direction,
                        score=state.score
                    )
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 1 hatası: {e}")

async def _process_tier_monitoring(self, symbols: List[str]):
    """
    TIER 2: İzleme aşaması (15s interval)

    - %50+ koşul sağlanmış
    - Koşullar yeniden değerlendir
    - DECISION'a yükselme kontrolü
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Koşulları yeniden değerlendir
            await self._evaluate_conditions(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 2 hatası: {e}")

async def _process_tier_analysis(self, symbols: List[str]):
    """
    TIER 3: Analiz aşaması (60s interval)

    - Yeni adayları tara
    - MONITORING'e yükselme kontrolü
    """
    for symbol in symbols:
        if not self._running:
            break

        try:
            # Koşulları değerlendir
            await self._evaluate_conditions(symbol)
        except Exception as e:
            self.logger.error(f"❌ {symbol} TIER 3 hatası: {e}")

async def _evaluate_conditions(self, symbol: str):
    """
    Symbol için koşulları değerlendir ve tier güncelle

    StrategyExecutor kullanır, sonucu TierManager'a bildirir
    """
    if not self._strategy_executor:
        return

    # Indicator data al (IndicatorManager'dan)
    indicator_data = await self._get_indicator_data(symbol)
    if not indicator_data:
        return

    # Koşulları değerlendir
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

    # Threshold'lara göre tier belirle
    thresholds = self.tier_manager.thresholds

    if score >= thresholds.get('decision', 1.0):
        new_tier = TierLevel.DECISION
    elif score >= thresholds.get('monitoring', 0.5):
        new_tier = TierLevel.MONITORING
    else:
        new_tier = TierLevel.ANALYSIS

    # TierManager'ı güncelle
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

### Adım 4: Candle Callback Entegrasyonu (Öncelik: YÜKSEK)

```python
async def _on_candle_closed(self, symbol: str, timeframe: str):
    """
    Candle kapandığında çağrılır (MTF Engine → TradingEngine)

    Exit-First Logic:
    1. ÖNCE: Pozisyon varsa exit kontrol
    2. SONRA: Entry kontrol
    """
    if not self.current_mode or not self.tier_manager:
        return

    try:
        tier = self.tier_manager.get_tier(symbol)

        # 1. ÖNCE EXIT (POSITION tier)
        if tier == TierLevel.POSITION:
            if hasattr(self.current_mode, 'on_candle_closed'):
                await self.current_mode.on_candle_closed(symbol, timeframe)

        # 2. Koşulları yeniden değerlendir
        await self._evaluate_conditions(symbol)

        # 3. DECISION tier'da ise entry kontrolü
        tier = self.tier_manager.get_tier(symbol)  # Güncel tier'ı al
        state = self.tier_manager.get_state(symbol)

        if tier == TierLevel.DECISION and state:
            # Candle kapandı, entry hazır mı?
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
                    # POSITION tier'a yükselt
                    self.tier_manager.set_tier(
                        symbol=symbol,
                        tier=TierLevel.POSITION,
                        direction=state.direction,
                        score=state.score
                    )

    except Exception as e:
        self.logger.error(f"❌ {symbol}: Candle callback hatası: {e}")
```

---

## 🔄 DATA FLOW DİYAGRAMI (YENİ - DÜZELTİLMİŞ)

### Kritik Anlayış: İki Farklı Veri Akışı Var

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1️⃣ REAL-TIME DATA FLOW                               │
│                     (WebSocket → Indicator → Cache)                         │
└─────────────────────────────────────────────────────────────────────────────┘

Binance WebSocket
       │
       │ kline_1m (HER SANİYE güncelleme gelir)
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
│                  │  Warmup: İlk N candle indicator
│ - RSI            │  hesaplaması için gerekli
│ - EMA            │  (örn: RSI_14 için min 14 candle)
│ - Bollinger      │
│ - ATR            │  ┌─────────────────────────────────┐
│ - ...            │  │ WARMUP DURUMU                   │
└────────┬─────────┘  │                                 │
         │            │ warmup_complete = False         │
         │            │ → Indicator hesaplanmaz         │
         │            │ → Tier check yapılmaz           │
         │            │                                 │
         │            │ warmup_complete = True          │
         │            │ → Indicator hesaplanır          │
         │            │ → Tier check başlar             │
         │            └─────────────────────────────────┘
         │
         │ Hesaplanan değerler
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
│              (Polling-based, candle_closed'dan BAĞIMSIZ!)                   │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────────┐
                              │   _tier_processing_loop()   │
                              │      (ana döngü)            │
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
    │ Tüm semboller   │           │ %50+ koşul      │           │ %100 koşul      │
    │ taranır         │           │ sağlanan        │           │ sağlanan        │
    │                 │           │ semboller       │           │ semboller       │
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
    │  2. StrategyExecutor ile koşulları değerlendir                          │
    │     result = strategy_executor.evaluate_entry(symbol, indicators)        │
    │                                                                          │
    │  3. Score'a göre yeni tier belirle                                      │
    │     score >= 1.0  → DECISION                                            │
    │     score >= 0.5  → MONITORING                                          │
    │     score < 0.5   → ANALYSIS                                            │
    │                                                                          │
    │  4. TierManager'ı güncelle                                              │
    │     tier_manager.set_tier(symbol, new_tier, score, direction, ...)      │
    └─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     3️⃣ TRADE EXECUTION FLOW                                 │
│                  (candle_closed SADECE BURADA ÖNEMLİ!)                      │
└─────────────────────────────────────────────────────────────────────────────┘

MTF Engine
    │
    │ candle_closed event (5m mum kapandı!)
    │ (Sadece primary_timeframe için)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│              _on_candle_closed(symbol, timeframe)                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ SADECE TIER 0 (POSITION) ve TIER 1 (DECISION) için!        │  │
│  │                                                             │  │
│  │ TIER 0 → Exit kontrolü (mum kapanış fiyatında SL/TP?)      │  │
│  │ TIER 1 → Entry execute (koşullar hala %100 mi? → TRADE!)   │  │
│  │                                                             │  │
│  │ TIER 2/3 → YAPILACAK BİR ŞEY YOK                           │  │
│  │           (tier_processing_loop zaten kontrol ediyor)       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  if tier == POSITION:                                            │
│      mode.check_exit_on_candle_close(symbol)                     │
│      # SL/TP tetiklendi mi? Sinyal çıkışı var mı?               │
│                                                                  │
│  elif tier == DECISION:                                          │
│      # Mum kapandı, entry zamanı!                                │
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
                              │   (EN YÜKSEK ÖNCELİK)       │
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
                   (config'den: ANALYSIS veya MONITORING)
```

---

## 🎯 1D STRATEJİ SENARYOSU

**Soru:** 1D stratejide candle_closed 24 saat sonra mı tier check yapılacak?

**Cevap:** HAYIR! Tier check **candle_closed'dan BAĞIMSIZ** çalışır.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1D STRATEJİ ÖRNEK AKIŞI                             │
└─────────────────────────────────────────────────────────────────────────────┘

Saat 00:00 - Gün başı
│
├── WebSocket: 1m kline verileri geliyor (sürekli)
│
├── MTF Engine: 1m → 1D aggregation (1440 adet 1m = 1 adet 1D)
│   └── 1D candle henüz KAPANMADI ama OHLC sürekli güncelleniyor!
│
├── IndicatorManager: 1D indikatörleri HER 1M UPDATE'de yeniden hesaplanır
│   └── RSI_14_1d, EMA_20_1d, BB_1d, ATR_1d...
│   └── (Açık mumun O,H,L,C değerleri ile hesaplanır)
│
├── CacheManager: Güncel indicator değerleri cache'te
│
└── _tier_processing_loop():
    │
    ├── TIER 3 check (her 60s):
    │   └── BTCUSDT koşulları %60 → MONITORING'e yükselt
    │
    ├── TIER 2 check (her 15s):
    │   └── BTCUSDT koşulları %85 → hala MONITORING
    │
    ├── TIER 2 check (her 15s):
    │   └── BTCUSDT koşulları %100 → DECISION'a yükselt!
    │
    └── TIER 1 check (her 5s):
        └── BTCUSDT %100 koşul SAĞLANDI ama...
            ├── candle_close_pending = True (mum henüz kapanmadı)
            └── Entry YAPILMAZ, bekle!

Saat 23:59:59 - Gün sonu (1D candle kapanıyor!)
│
└── MTF Engine: candle_closed event ("BTCUSDT", "1d")
    │
    └── _on_candle_closed("BTCUSDT", "1d"):
        │
        └── tier == DECISION ve candle_close_pending == True
            │
            └── candle_close_pending = False
            └── ready_for_entry = True
            └── mode.execute_entry("BTCUSDT", "LONG", 1.0)
            └── tier → POSITION
```

### Özet:

| İşlem | Ne Zaman Olur? | Candle Close Gerekli mi? |
|-------|----------------|--------------------------|
| Indicator hesaplama | Her 1m update | ❌ Hayır |
| Tier 3→2 geçişi | Her 60s polling | ❌ Hayır |
| Tier 2→1 geçişi | Her 15s polling | ❌ Hayır |
| **Entry execute** | **Candle close anında** | **✅ EVET** |
| SL/TP kontrolü | Her 1s polling | ❌ Hayır |
| **Exit execute** | Tick-based veya candle close | **Duruma göre** |

---

## 📊 WARMUP VE INDICATOR AKIŞI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WARMUP SÜRECİ                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Engine başlatıldı
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. WARMUP PERİYODU HESAPLA                                       │
│                                                                  │
│ warmup_period = max(                                             │
│     indicator.required_periods for indicator in strategy         │
│ ) + buffer                                                       │
│                                                                  │
│ Örnek:                                                           │
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
│ 2. TARİHSEL VERİ YÜKLE (warmup için)                             │
│                                                                  │
│ for symbol in symbols:                                           │
│     for timeframe in mtf_timeframes:                             │
│         # Parquet'ten veya API'den                               │
│         historical = connector.get_klines(                       │
│             symbol=symbol,                                       │
│             timeframe=timeframe,                                 │
│             limit=warmup_period                                  │
│         )                                                        │
│                                                                  │
│         # MTF Engine'e yükle                                     │
│         mtf_engine.load_historical(historical)                   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. INDICATOR WARMUP                                              │
│                                                                  │
│ indicator_manager.warmup(historical_data)                        │
│                                                                  │
│ # Her indicator kendi warmup'ını yapar                           │
│ # İlk N değer NaN olabilir, bu normal                            │
│                                                                  │
│ RSI_14:   [NaN, NaN, ..., NaN, 45.2, 48.1, 52.3, ...]            │
│            ▲─── 14 candle ───▲                                   │
│                                                                  │
│ warmup_complete = True                                           │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. WEBSOCKET BAŞLAT                                              │
│                                                                  │
│ # Artık canlı veri alabilir                                      │
│ websocket_engine.subscribe(symbols, channels)                    │
│                                                                  │
│ # Her yeni 1m kline geldiğinde:                                  │
│ # - MTF buffer güncelle                                          │
│ # - Indicator incremental hesapla                                │
│ # - Cache güncelle                                               │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. TIER PROCESSING BAŞLAT                                        │
│                                                                  │
│ # Warmup complete olduktan sonra tier check başlar               │
│ _tier_processing_loop()                                          │
│                                                                  │
│ # Warmup complete DEĞİLSE:                                       │
│ # - Tier check yapılmaz                                          │
│ # - Entry/Exit yapılmaz                                          │
│ # - Sadece veri toplanır                                         │
└──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    INDICATOR INCREMENTAL UPDATE                              │
└─────────────────────────────────────────────────────────────────────────────┘

WebSocket: Yeni 1m kline geldi
       │
       ▼
MTF Engine: 1m buffer güncelle
       │
       ├── is_candle_update? (mum hala açık)
       │   │
       │   └── Sadece current candle OHLC güncelle
       │       └── Indicator'lar AÇIK MUM değerleriyle yeniden hesaplanır
       │
       └── is_candle_close? (mum kapandı)
           │
           └── Buffer'a yeni candle ekle
               └── Eski candle'ı "closed" olarak işaretle
               └── candle_closed event emit et


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
│         "is_closed": false  # Açık mum mu?
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

## ⚠️ DİKKAT EDİLECEK NOKTALAR

### 1. Thread Safety
```python
# TierManager._states dict'i thread-safe değil
# Eğer multi-thread kullanılacaksa:
import threading
self._lock = threading.Lock()

def set_tier(self, ...):
    with self._lock:
        # ... tier güncelleme ...
```

### 2. Async/Await Consistency
```python
# Tüm I/O operasyonları async olmalı
# Blocking calls asyncio.to_thread() ile wrap edilmeli

# ❌ Yanlış
result = self.connector.get_balance()  # Blocking!

# ✅ Doğru
result = await self.connector.get_balance()  # veya
result = await asyncio.to_thread(self.connector.get_balance)
```

### 3. Error Isolation
```python
# Her symbol işlemi try/except içinde olmalı
# Bir symbol hatası diğerlerini etkilememeli

for symbol in symbols:
    try:
        await self._process_symbol(symbol)
    except Exception as e:
        self.logger.error(f"❌ {symbol} hatası: {e}")
        # Devam et, durma
```

### 4. Graceful Shutdown
```python
async def stop(self):
    """Clean shutdown"""
    self._running = False

    # 1. Yeni işlemleri durdur
    # 2. Açık pozisyonları kaydet (crash recovery için)
    # 3. WebSocket kapat
    # 4. Mode shutdown
    # 5. Cache flush
```

---

## 📊 KARŞILAŞTIRMA: ÖNCE vs SONRA

### ÖNCE (V5 mevcut)
```
TradingEngine V5
├── 4 Core singleton ✅
├── Strategy loading ✅
├── Symbol loading ✅
├── Mode loading ✅
├── Data feeds (COMMENT-OUT) ⚠️
├── Tier system ❌
├── Processing loop ❌ (sadece sleep)
├── Display ❌
└── EventBus/Cache aktif kullanım ❌
```

### SONRA (V5 + Entegrasyon)
```
TradingEngine V5 + TierManager
├── 4 Core singleton ✅
├── Strategy loading ✅
├── Symbol loading ✅
├── Mode loading ✅
├── Data feeds ✅ (aktif)
├── TierManager ✅ (entegre)
├── DisplayInfo ✅ (entegre)
├── Tier-based processing loop ✅
├── EventBus aktif kullanım ✅
├── Cache aktif kullanım ✅
└── Exit-First logic ✅
```

---

## 🚀 UYGULAMA ÖNCELİKLERİ

| Öncelik | Adım | Tahmini Satır | Bağımlılık |
|---------|------|---------------|------------|
| 1 | TierManager import & init | +30 | - |
| 2 | DisplayInfo import & init | +20 | TierManager |
| 3 | _tier_processing_loop | +60 | TierManager |
| 4 | _status_display_loop | +40 | DisplayInfo |
| 5 | Tier işleme metodları | +80 | Mode |
| 6 | _on_candle_closed güncelleme | +40 | StrategyExecutor |
| 7 | Data feeds aktif etme | +10 | WebSocket, MTF |
| 8 | stop() güncelleme | +20 | - |

**Toplam:** ~300 satır ekleme → V5 939 satır olacak (hala V4'ten az!)

---

## 🎯 SONUÇ

### Önerilen Yaklaşım: "Incremental Integration"

1. **Önce TierManager + DisplayInfo** entegre et (temel görünürlük)
2. **Sonra Tier Loop** ekle (işleme mantığı)
3. **Son olarak Data Feeds** aktif et (tam çalışır sistem)

Bu yaklaşım:
- Her adımda test edilebilir
- Geriye dönük uyumlu
- Backtest Engine'in kanıtlanmış pattern'lerini kullanır
- Engine "lean coordinator" rolünü korur

---

## 🔬 V1 ANALİZİ: Real-Time Evaluation

V1'de işleme şu şekilde:

```python
# V1 _main_loop() - trading_engine_v1.py:1644-1676
async def _main_loop(self):
    while self.is_running:
        loop_count += 1

        # Real-time evaluation (10 saniyede bir)
        if loop_count % 10 == 0:
            await self._realtime_evaluation()  # TÜM semboller için

        # Status log (60 saniyede bir)
        if loop_count % 60 == 0:
            self.display_trading_info()

        # Tier status (15 saniyede bir)
        elif loop_count % 15 == 0:
            self.display_live_status()

        await asyncio.sleep(1)
```

### V1'in SL/TP Kontrolü
```python
# V1 _evaluate_exits_for_symbol() - trading_engine_v1.py:1281-1343
async def _evaluate_exits_for_symbol(self, symbol, indicator_data):
    # Get current price from DataFrame (SON MUM değeri)
    current_price = indicator_data[primary_tf]['close'].iloc[-1]

    for position in positions:
        exit_result = strategy_executor.evaluate_exit(
            position=position,
            current_price=current_price  # DataFrame'den!
        )

        if exit_result.get('should_exit'):
            await self._close_position(position, current_price, reason)
```

### V1 Problemi: Tick-Based DEĞİL!
```
❌ V1: Her 10 saniyede DataFrame'den son close fiyatı al
       → SL $100'da, fiyat $99'a düştü ama 10 saniye sonra $101'e çıktı
       → SL MISS! Çünkü check yapıldığında fiyat $101'di

✅ OLMASI GEREKEN: Her tick'te (her fiyat güncellemesinde) kontrol
       → SL $100'da, fiyat $99'a düştü
       → ANINDA çıkış (tick geldiği an)
```

---

## 🎯 YENİ ÖNERİ: Hybrid Model

### İki Farklı Fiyat Kaynağı

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TICK DATA vs CANDLE DATA                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. TICK DATA (WebSocket aggTrade/bookTicker)
   └── Her fiyat değişikliğinde gelir
   └── SL/TP kontrolü için kullanılır
   └── Indicator hesaplamaz

2. CANDLE DATA (WebSocket kline)
   └── Her mum güncellemesinde gelir (1s)
   └── Indicator hesaplama için kullanılır
   └── Entry/Exit sinyali için kullanılır
```

### Önerilen Akış (Hybrid)

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
        └── PriceStream (yeni component)                                 │
            └── Her tick'te SADECE pozisyon kontrolü                     │
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

## 📊 KARAR TABLOSU: Ne Zaman Ne Kullanılır?

| İşlem | Veri Kaynağı | Kontrol Zamanı | Bekleme |
|-------|--------------|----------------|---------|
| **Indicator hesaplama** | kline (candle) | Her candle update | ❌ |
| **Tier evaluation** | Cache (indicators) | Polling (interval veya sürekli) | ❌ |
| **Entry signal** | Cache (indicators) | Candle close event | ✅ Candle close |
| **SL/TP check** | bookTicker (tick) | Her tick | ❌ |
| **Trailing update** | bookTicker (tick) | Her tick | ❌ |
| **Signal exit** | kline (candle) | Candle close event | ✅ Candle close |

### Entry vs Exit Farkı

```
ENTRY:
└── Candle CLOSE gerekli (sinyal doğrulama)
└── Açık mum verileriyle trade açmak riskli
└── "Close > EMA" koşulu mum kapanınca kesinleşir

EXIT (SL/TP):
└── Candle close BEKLEMEZ!
└── Fiyat SL'e değdiği AN çıkış
└── Her millisaniye önemli (kayıp büyüyebilir)

EXIT (Signal-based):
└── Candle close gerekli
└── "RSI > 70 iken çıkış" → mum kapanınca kontrol
```

---

## 🔧 UYGULAMA ÖNERİSİ

### Seçenek 1: Basit (Mevcut V1 gibi, interval-based)
```python
# Pros: Kolay implement, az complexity
# Cons: SL/TP gecikebilir (1-10 saniye)

async def _tier_processing_loop(self):
    while self._running:
        # TIER 0: Her saniye (SL/TP interval-based)
        await self._check_positions()

        # TIER 1-3: Interval'a göre
        symbols_to_check = self.tier_manager.get_symbols_to_check()
        ...
        await asyncio.sleep(1)
```

### Seçenek 2: Hybrid (Tick + Candle)
```python
# Pros: Gerçek zamanlı SL/TP
# Cons: Daha fazla WebSocket subscription, complexity

# Ayrı tick stream
async def _on_tick(self, symbol: str, price: float):
    """Her fiyat güncellemesinde çağrılır"""
    for position in self._get_positions(symbol):
        if self._check_sl_tp(position, price):
            await self._immediate_exit(position, price)

# Ayrı candle stream
async def _on_candle_closed(self, symbol: str, timeframe: str):
    """Mum kapandığında çağrılır - entry ve signal exit"""
    ...
```

### Seçenek 3: Order-Based (Exchange'e bırak)
```python
# Pros: En güvenilir, exchange garantili
# Cons: Paper mode'da çalışmaz, daha az kontrol

# Entry ile birlikte SL/TP order'ları da gönder
async def execute_entry(self, symbol, direction):
    # Ana order
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

## 🎯 ÖNERİLEN YAKLAŞIM

### Paper Mode: Seçenek 1 (Interval-based)
- Her saniye fiyat kontrolü yeterli
- Simülasyon için milisaniye hassasiyeti gereksiz
- Implement edilmesi kolay

### Live Mode: Seçenek 3 (Order-based) + Seçenek 2 backup
- Exchange'e SL/TP order gönder (OCO veya ayrı)
- Backup olarak tick-based kontrol (bağlantı kopması için)
- En güvenli yaklaşım

```python
class TradingEngine:
    async def _handle_entry_success(self, symbol, position):
        """Entry başarılı olduğunda"""

        if self.mode_name == 'live':
            # Exchange'e SL/TP order gönder
            await self._place_sl_tp_orders(symbol, position)
        else:
            # Paper mode: TierManager'a ekle, polling ile kontrol edilecek
            self.tier_manager.set_tier(symbol, TierLevel.POSITION, ...)
```

---

## 📝 ÖZET: Neyin Ne Zaman Olacağı

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

### Interval Durumu (`intervals.enabled`)

```yaml
# trading.yaml
tiers:
  intervals:
    enabled: false   # Küçük sembol listesi için
    # enabled: true  # 100+ sembol için kaynak optimizasyonu
```

**enabled: false** → Tüm tierlar her döngüde kontrol edilir (20 sembol için ideal)
**enabled: true** → Tier'a göre farklı interval (100+ sembol için)

---

**Analiz Tarihi:** 2025-12-03
**Güncelleme:** V1 analizi, Hybrid model önerisi, SL/TP tick-based açıklama
**Analiz Eden:** Claude AI Assistant
**Referans:** TRADING_ENGINE_ANALYSIS.md (V1-V4 karşılaştırma)
