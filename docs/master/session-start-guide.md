# SuperBot - Claude Session Start Guide
**🚀 Read This File at the Start of Every New Session!**

---

## ⚡ Quick Reference (Read in 5 Minutes)

### 1. **What is the Project?**
SuperBot = Multi-exchange crypto trading platform with **daemon-based architecture**

**Modules:**
- 🤖 **AI Module:** FastAPI (uvicorn) - Strategy optimization
- 🌐 **WebUI Module:** Flask (waitress) - Web interface
- 📈 **Trading Module:** Async Python - Live/Paper trading
- 🔬 **Backtest Module:** On-demand - Strategy testing
- 🐕 **Watchdog:** Health monitoring & auto-restart

**Core Infrastructure:**
- ConfigEngine, LoggerEngine, EventBus, CacheManager, ProcessManager, GracefulShutdown

---

### 2. **CRITICAL Information (Must Know!)**

#### ❗ **"Don't Reinvent the Wheel"**
- Existing code **works**, don't break it!
- There are patterns copied from old system (`temp/` folder)
- Don't suggest "better ways", **follow existing patterns**

#### ❗ **Why Does Daemon Architecture Exist?**
```
superbot.py (Master Daemon)
    ↓
Shared Resources (Cache, EventBus, ConnectionPool)
    ↓
Modules (Trading, WebUI, AI, Backtest)
```

**Each module runs in a separate process!** Daemon handles orchestration.

#### ❗ **Why Are There 3 Exchange Files?**
1. **BinanceAPI** (`components/exchanges/binance_api.py`)
   - Direct module usage
   - BaseExchangeAPI implementation
   - Async executor pattern

2. **ConnectorEngine** (`temp/exchange_connector_engine.py`)
   - Daemon shared resource
   - Wrapper around python-binance
   - Cache, rate limiting, retry

3. **ConnectionPoolEngine** (`temp/exchange_connection_engine.py`)
   - HTTP/WebSocket connection pooling
   - Shared across all modules

**All are necessary! Don't delete!**

#### ❗ **Async Executor Pattern (SUPER CRITICAL!)**
```python
# ❌ WRONG - Blocks event loop, 10x slowdown
ticker = self.client.get_ticker(symbol="BTCUSDT")

# ✅ CORRECT - Non-blocking, 8-10x speedup
loop = asyncio.get_event_loop()
ticker = await loop.run_in_executor(
    None,
    lambda: self.client.get_ticker(symbol="BTCUSDT")
)
```

**Why:** python-binance is sync, trading engine is async → executor pattern mandatory!

**Where to use:**
- ✅ **ALL exchange API methods** (get_ticker, get_orderbook, create_order, etc.)
- ✅ Inside BinanceAPI
- ✅ Inside ConnectorEngine

---

### 3. **File Structure**

```
SuperBot/
├── superbot.py              # Master daemon orchestrator
├── superbot-cli.py          # CLI tool (IPC client)
│
├── core/                    # CORE infrastructure
│   ├── config_engine.py
│   ├── logger_engine.py
│   ├── event_bus.py         # Pub/Sub messaging
│   ├── cache_manager.py     # Shared cache
│   ├── process_manager.py
│   ├── graceful_shutdown.py
│   ├── ipc_server.py        # JSON-RPC 2.0
│   ├── module_launcher.py   # uvicorn/flask/python/thread
│   ├── watchdog.py          # Health monitoring
│   └── scheduler.py         # Cron-like tasks
│
├── components/
│   ├── exchanges/           # Exchange API implementations
│   │   ├── base_api.py      # Abstract base
│   │   └── binance_api.py   # Binance implementation
│   └── strategies/          # Trading strategies
│
├── modules/
│   ├── ai/                  # AI Module (FastAPI)
│   ├── webui/               # WebUI Module (Flask)
│   ├── trading/             # Trading Module (async)
│   └── backtest/            # Backtest Module
│
├── config/
│   ├── main.yaml            # Main config
│   ├── connectors.yaml      # Exchange configs
│   ├── infrastructure.yaml  # Cache, EventBus, ConnectionPool
│   └── daemon.yaml          # Module definitions, autostart
│
├── temp/                    # OLD system reference (DON'T DELETE!)
│   ├── binance_client.py         # Async executor pattern example
│   ├── exchange_connector_engine.py
│   └── exchange_connection_engine.py
│
└── docs/
    └── claude/
        ├── daemon-architecture-guide.md  # DETAILED GUIDE
        └── session-start-guide.md        # THIS FILE
```

---

### 4. **Event Bus (Inter-Module Communication)**

**Pub/Sub Pattern:**
```python
# Event publish
await event_bus.publish_async(
    topic='price.BTCUSDT.update',
    data={'price': 50000, 'timestamp': time.time()},
    source='trading_engine'
)

# Event subscribe
def on_price_update(event):
    print(f"Price: {event.data['price']}")

event_bus.subscribe('price.*.update', on_price_update)
```

**Topic patterns:**
- `price.{SYMBOL}.update` → Price updates
- `trade.executed` → Trade executed
- `order.filled` → Order filled
- `module.started` → Module started
- `system.ready` → System ready

**Wildcard support:** `price.*.update` listens to all symbols

---

### 5. **Cache Manager (Performance)**

```python
# Cache write (5s TTL optimal for ticker)
cache_manager.set('ticker:BTCUSDT', ticker_data, ttl=5)

# Cache read
cached = cache_manager.get('ticker:BTCUSDT')
if cached:
    return cached  # Cache hit
else:
    # Cache miss, go to API
    data = await connector.get_ticker('BTCUSDT')
    cache_manager.set('ticker:BTCUSDT', data, ttl=5)
    return data
```

**Optimal TTL values:**
- Ticker: **5s**
- Orderbook: **1s**
- Klines: **60s**
- Balance: **10s**

**Target:** >70% cache hit rate

---

### 6. **Module Lifecycle**

**Start:**
```bash
# CLI
superbot-cli module start trading

# RPC
{"jsonrpc": "2.0", "method": "module.start", "params": {"module": "trading"}, "id": 1}
```

**Stop:**
```bash
superbot-cli module stop trading
```

**Restart:**
```bash
superbot-cli module restart trading
```

**Status:**
```bash
superbot-cli module status trading
```

---

### 7. **RPC Methods (17 total)**

**Daemon:**
- `daemon.status`, `daemon.stop`, `daemon.reload_config`

**Module:**
- `module.start`, `module.stop`, `module.restart`, `module.status`, `module.list`

**Trading:**
- `trading.positions`, `trading.orders`, `trading.balance`

**Monitoring:**
- `monitoring.health`, `monitoring.metrics`, `monitoring.resources`

**Logs:**
- `logs.stream`

---

### 8. **Common Issues**

#### Issue 1: "Event loop blocking" (10x slowdown)
**Solution:** Use async executor pattern (example above)

#### Issue 2: "Module failed to start"
**Debug:**
```bash
tail -f logs/daemon.log
python -m modules.trading.engine --mode paper  # Manual test
```

#### Issue 3: "Low cache hit rate"
**Solution:** Check TTL values (ticker=5s, orderbook=1s, klines=60s)

#### Issue 4: "Connection pool exhausted"
**Solution:** `config/infrastructure.yaml` → `max_connections: 20` (increase)

---

### 9. **Checklist: When Making Code Changes**

- [ ] Did you use async executor pattern? (in exchange API)
- [ ] Did you add event bus integration? (in modules)
- [ ] Did you use cache manager? (for performance)
- [ ] Is it config-driven? (no hard-coded values)
- [ ] Did you add graceful shutdown handler?
- [ ] Did you implement health check?
- [ ] Did you add logs? (logger.info/error)
- [ ] Did you test? (manual + unit test)

---

### 10. **Don't Do List**

**❌ NEVER DO:**

1. **Block event loop**
   ```python
   # ❌ Sync call in async context
   ticker = self.client.get_ticker()
   ```

2. **Bypass shared resources**
   ```python
   # ❌ Each module creates its own connector
   connector = BinanceAPI(config)
   ```

3. **Hard-coded config**
   ```python
   # ❌
   API_KEY = "xyz123"
   ```

4. **Direct module call**
   ```python
   # ❌
   trading_engine.execute_trade()

   # ✅ Use event bus
   event_bus.publish('trade.execute', {...})
   ```

5. **Exception swallow**
   ```python
   # ❌
   try:
       something()
   except:
       pass
   ```

---

## 📚 More Information

**Read detailed guide:**
```bash
docs/claude/daemon-architecture-guide.md
```

**This file contains:**
- Daemon architecture details
- Module lifecycle details
- RPC communication details
- Performance optimization
- Troubleshooting
- Best practices
- Integration examples

---

## 🎯 Summary (30 Seconds)

1. **Daemon architecture** → Modules in separate processes
2. **Async executor pattern** → python-binance sync, prevent event loop blocking
3. **Event bus** → Inter-module communication
4. **Cache manager** → Performance (>70% hit rate target)
5. **Config-driven** → Everything in config
6. **Don't reinvent the wheel** → Don't break working code

**What to do now:**
1. Read this file ✅
2. Read `docs/claude/daemon-architecture-guide.md` (detailed)
3. Understand what user wants
4. Follow existing patterns
5. Write code, test it

---

**Good luck! 🚀**
