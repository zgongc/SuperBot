# SuperBot Daemon Architecture - Claude Guide
**Date:** 2025-11-26
**Version:** 1.0.0
**Author:** SuperBot Team

---

## 🎯 IMPORTANT: You Must Read This Document!

This document explains SuperBot's daemon architecture and critical design decisions. **Read this document before making any changes!**

---

## 📚 Core Principles

### 1. **Don't Reinvent the Wheel!**
- There IS code copied from existing working system
- This code is **tested and proven**
- When adding new features, **follow existing patterns**
- Don't suggest "better ways", **don't break working system**

### 2. **Daemon Architecture - Why Does It Exist?**
SuperBot is a **daemon-based multi-module system**:

```
superbot.py (Master Daemon)
    ↓
Shared Resources (CacheManager, EventBus, ConnectionPool)
    ↓
Modules (Trading, WebUI, AI, Backtest)
```

**Why this way?**
- A single Python process is not enough
- Each module runs in its own process (isolation)
- Communication via shared resources (event bus, cache)
- Central orchestration (start/stop/restart/health check)

---

## 🏗️ Core Architecture

### **CORE Infrastructure Layer**

#### 1. ConfigEngine
- Loads all config files (`config/*.yaml`)
- Supports hot reload (SIGHUP signal)
- Nested config access: `config.get('infrastructure.cache.backend')`

#### 2. LoggerEngine
- Unified logging system
- Ultra compact format (single line for INFO level)
- File rotation, console output

#### 3. EventBus
- **Pub/Sub messaging** (inter-module communication)
- Topic-based routing: `price.BTCUSDT.update`, `trade.executed`
- Wildcard support: `price.*.update`
- Memory & Redis backend
- **CRITICAL:** Trading engine publishes all events here

#### 4. CacheManager
- **Shared cache** (used by all modules)
- Memory & Redis backend
- TTL, LRU eviction
- Cache for ticker, orderbook, klines
- **5 second TTL** optimal for ticker

#### 5. ProcessManager
- Engine lifecycle management
- Dependency-based startup order
- Auto-restart on crash
- Health check monitoring

#### 6. GracefulShutdown
- Signal handling (SIGINT, SIGTERM, SIGBREAK)
- Callback system (pre/cleanup/post)
- State persistence
- Position close (optional)

---

### **Daemon Components Layer**

#### 1. IPCServer
- **JSON-RPC 2.0** over Unix socket (Linux/Mac)
- TCP fallback for Windows (127.0.0.1:9999)
- RPC method handler registration
- Authentication token support
- **17 RPC methods** (daemon control, module control, monitoring)

#### 2. ModuleLauncher
- **4 module type** support:
  - `uvicorn`: FastAPI apps (AI Module)
  - `flask`: Flask apps (WebUI Module) - waitress for production
  - `python`: Async Python scripts (Trading, Backtest)
  - `thread`: Background threads (Monitoring)
- PID file tracking (`.superbot/module_name.pid`)
- Health check via HTTP endpoint
- Graceful shutdown (SIGTERM → wait 10s → SIGKILL)

#### 3. ThreadPoolManager
- Thread pool management
- Resource allocation per module

#### 4. Watchdog
- **Periodic health checks** (process monitoring with psutil)
- Auto-restart on crash
- Restart limits (default: 3 attempts)
- Restart cooldown (default: 60s)
- Alert system (via event bus)
- **CRITICAL:** Checks if process is zombie

#### 5. TaskScheduler
- **Cron-like scheduling**
- Time-based: `"09:00"` format (HH:MM)
- Cron-based: `"0 2 * * *"` (simplified cron)
- Day-of-week filtering
- Timezone-aware (pytz)
- **Use cases:**
  - Trading schedule (auto start/stop at specific hours)
  - Daily backtest (2 AM)
  - Weekly reports (Sunday midnight)

---

## 🔌 Exchange API Architecture

### **Why Are There 3 Different Files?**

#### 1. **BinanceAPI** (`components/exchanges/binance_api.py`)
- **Usage:** Direct usage within modules
- **Features:**
  - BaseExchangeAPI implementation
  - Async executor pattern (sync → async wrapper)
  - Cache manager integration
  - Config-driven (testnet/production)
- **Example:** Used in `modules/trading/modes/paper_mode.py`

#### 2. **ConnectorEngine** (`temp/exchange_connector_engine.py`)
- **Usage:** Daemon shared resource
- **Features:**
  - python-binance wrapper
  - Cache manager integration
  - Rate limiting
  - Retry mechanism
  - **All modules share the same connector**
- **Why needed:** Central connector for daemon architecture

#### 3. **ConnectionPoolEngine** (`temp/exchange_connection_engine.py`)
- **Usage:** Daemon shared connection pool
- **Features:**
  - HTTP/WebSocket connection pooling (aiohttp)
  - Min/max connection management
  - Health check loop
  - Idle timeout, auto-reconnect
  - **All modules use the same pool**
- **Why needed:** Connection reuse, performance optimization

**NOTE:** `connection_pool.py` and `exchange_connection_engine.py` are the same file → one moved to `_deprecated`

---

## ⚡ Performance Critical: Async Executor Pattern

### **Problem: Event Loop Blocking**
```python
# ❌ WRONG - Blocks event loop
def get_ticker(self, symbol: str):
    return self.client.get_ticker(symbol=symbol)  # Sync call
```

### **Solution: Async Executor**
```python
# ✅ CORRECT - Non-blocking
async def get_ticker(self, symbol: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: self.client.get_ticker(symbol=symbol)
    )
    return result
```

**Why important:**
- python-binance is **sync** library
- Trading engine is **async** (asyncio event loop)
- Sync call → event loop freeze → 10x slowdown
- Async executor → non-blocking → 8-10x speedup

**Where used:**
- ✅ `BinanceAPI` (ALL methods)
- ✅ `ConnectorEngine` (ALL methods)
- ✅ `temp/binance_client.py` (OLD system - reference)

---

## 🎭 Module Lifecycle

### **Module Types and Launch Methods**

#### 1. **uvicorn** (FastAPI)
```yaml
ai:
  enabled: true
  type: uvicorn
  app: "modules.ai.app:app"
  host: "127.0.0.1"
  port: 8000
  workers: 2
  debug: false  # production: workers=2, debug=false
```

**Launch command:**
```bash
python -m uvicorn modules.ai.app:app --host 127.0.0.1 --port 8000 --workers 2
```

#### 2. **flask** (Flask/Waitress)
```yaml
webui:
  enabled: true
  type: flask
  app: "modules.webui.app:app"
  host: "0.0.0.0"
  port: 8080
  debug: false  # production: waitress, debug: flask dev server
```

**Launch command (production):**
```bash
python -m waitress --host=0.0.0.0 --port=8080 modules.webui.app:app
```

**Launch command (debug):**
```bash
FLASK_APP=modules.webui.app:app FLASK_DEBUG=1 python -m flask run --host=0.0.0.0 --port=8080
```

#### 3. **python** (Async Python)
```yaml
trading:
  enabled: true
  type: python
  module: "modules.trading.engine"
  args: ["--mode", "live"]
```

**Launch command:**
```bash
python -m modules.trading.engine --mode live
```

#### 4. **thread** (Background Thread)
```yaml
monitoring:
  enabled: false
  type: thread
  module: "modules.monitoring.health_monitor"
```

**Launch:** Runs in background with Threading.Thread

---

## 📡 RPC Communication

### **IPC Server - JSON-RPC 2.0**

**Socket path:**
- Linux/Mac: `/tmp/superbot.sock` (Unix socket)
- Windows: `127.0.0.1:9999` (TCP fallback)

**RPC Methods (17 total):**

#### Daemon Control
- `daemon.status` → Daemon status
- `daemon.stop` → Stop daemon
- `daemon.reload_config` → Config reload (SIGHUP)

#### Module Control
- `module.start` → Start module
- `module.stop` → Stop module
- `module.restart` → Restart module
- `module.status` → Module status
- `module.list` → List all modules

#### Trading (Proxy to Trading Module)
- `trading.positions` → Open positions
- `trading.orders` → Orders
- `trading.balance` → Balance

#### Monitoring
- `monitoring.health` → Health check
- `monitoring.metrics` → Metrics
- `monitoring.resources` → Resource usage (CPU, RAM, threads)

#### Logs
- `logs.stream` → Log streaming (WebSocket recommended)

**Example RPC call:**
```json
{
  "jsonrpc": "2.0",
  "method": "module.start",
  "params": {
    "module": "trading",
    "params": {"mode": "paper"}
  },
  "id": 1
}
```

---

## 🚀 Startup Sequence

### **1. Daemon Startup**
```bash
python superbot.py
```

**Sequence:**
1. Check already running (PID file check)
2. Write PID file
3. Initialize CORE infrastructure
   - ConfigEngine (load all configs)
   - LoggerEngine
   - EventBus
   - CacheManager
   - ProcessManager
   - GracefulShutdown
4. Initialize daemon components
   - ModuleLauncher
   - ThreadPoolManager
   - IPCServer (register 17 RPC handlers)
   - Watchdog
   - TaskScheduler
5. Load module definitions (from `config/daemon.yaml`)
6. Start IPC server
7. Start Watchdog
8. Start Scheduler
9. **Start autostart modules** (`config/daemon.yaml` → `autostart: [webui, trading]`)
10. Publish `system.ready` event
11. Run forever (await asyncio.sleep loop)

### **2. Module Startup (via RPC)**
```bash
# Via CLI
superbot-cli module start trading

# Via RPC
{"jsonrpc": "2.0", "method": "module.start", "params": {"module": "trading"}, "id": 1}
```

**Sequence:**
1. Check module exists
2. Check module not already running
3. Merge params with config
4. Launch module via ModuleLauncher
   - Build command (uvicorn/flask/python/thread)
   - Start process/thread
   - Write PID file
5. Update module status → `running`
6. Publish `module.started` event
7. Return PID

---

## 🛑 Shutdown Sequence

### **Graceful Shutdown**
```bash
# SIGTERM or SIGINT
kill -TERM <daemon_pid>

# Or via RPC
{"jsonrpc": "2.0", "method": "daemon.stop", "params": {}, "id": 1}
```

**Sequence:**
1. Trigger graceful shutdown (GracefulShutdown.initiate())
2. Stop scheduler
3. Stop watchdog
4. Stop all running modules (graceful)
   - Send SIGTERM
   - Wait 10 seconds
   - If still running → SIGKILL
5. Stop IPC server
6. Stop thread pools
7. Close cache connections
8. Close event bus
9. Remove PID file
10. Remove socket file
11. Publish `system.shutdown` event

---

## 🔧 Configuration Files

### **config/daemon.yaml**
```yaml
daemon:
  # Module definitions
  modules:
    webui:
      enabled: true
      type: flask
      app: "modules.webui.app:app"
      host: "0.0.0.0"
      port: 8080
      debug: false
      healthcheck_endpoint: "/health"

    trading:
      enabled: true
      type: python
      module: "modules.trading.engine"
      args: ["--mode", "paper"]

    ai:
      enabled: false
      type: uvicorn
      app: "modules.ai.app:app"
      host: "127.0.0.1"
      port: 8000
      workers: 2

  # Autostart modules
  autostart:
    - webui
    # - trading  # Manual start

  # IPC config
  ipc:
    socket_path: "/tmp/superbot.sock"
    auth_token: null  # Optional

  # Watchdog config
  watchdog:
    enabled: true
    check_interval: 30  # seconds
    auto_restart_on_crash: true
    max_restart_attempts: 3
    restart_cooldown: 60  # seconds
    alert_on_restart: true

  # Scheduler config
  schedule:
    timezone: "Europe/Istanbul"

    # Trading schedule
    trading_start: "09:00"  # HH:MM
    trading_stop: "18:00"
    trading_days: [1, 2, 3, 4, 5]  # Mon-Fri

    # Daily backtest
    daily_backtest:
      enabled: false
      cron: "0 2 * * *"  # 2 AM daily
      strategy: "default"

    # Weekly report
    weekly_report:
      enabled: false
      cron: "0 0 * * 0"  # Sunday midnight

  # Resource allocation
  resource_allocation:
    thread_pools:
      default:
        max_workers: 10
      io:
        max_workers: 20
      cpu:
        max_workers: 4
```

---

## 🐛 Common Issues & Solutions

### **Issue 1: "Daemon already running"**
**Cause:** PID file exists and process is running

**Solution:**
```bash
# Stop daemon
superbot-cli daemon stop

# Or manually kill PID
cat .superbot/daemon.pid
kill <pid>

# If stale PID file, delete it
rm .superbot/daemon.pid
```

---

### **Issue 2: "Module failed to start"**
**Cause:** Port in use, missing dependency, or config error

**Debug:**
```bash
# Check module logs
tail -f logs/daemon.log

# Test manual startup
python -m modules.trading.engine --mode paper
```

**Solution:**
- Change port (config/daemon.yaml)
- Install dependency (`pip install -r requirements.txt`)
- Fix config

---

### **Issue 3: "Event loop blocking"**
**Cause:** Sync API call in async context

**Symptoms:**
- Trading engine slow (10x)
- WebUI freezing
- Timeout errors

**Solution:**
```python
# ❌ WRONG
ticker = self.client.get_ticker(symbol="BTCUSDT")

# ✅ CORRECT
loop = asyncio.get_event_loop()
ticker = await loop.run_in_executor(
    None,
    lambda: self.client.get_ticker(symbol="BTCUSDT")
)
```

**Check:**
- `components/exchanges/binance_api.py` → Does every method have async executor?
- `temp/exchange_connector_engine.py` → Does every method have async executor?

---

### **Issue 4: "Connection pool exhausted"**
**Cause:** Connection leak, timeout, or max_connections too low

**Solution:**
```yaml
# config/infrastructure.yaml
infrastructure:
  connection_pool:
    min_connections: 2
    max_connections: 10  # Increase: 20
    connection_timeout: 30
    idle_timeout: 300
```

---

### **Issue 5: "Low cache hit rate"**
**Cause:** TTL too low, or cache disabled

**Solution:**
```python
# Optimal TTL for ticker: 5 seconds
self.cache_manager.set(cache_key, result, ttl=5)

# For orderbook: 1 second
self.cache_manager.set(cache_key, result, ttl=1)

# For klines: 60 seconds
self.cache_manager.set(cache_key, result, ttl=60)
```

**Check stats:**
```python
stats = cache_manager.get_stats()
print(f"Hit rate: {stats['cache_hit_rate']}")
# Target: >70% hit rate
```

---

## 📋 Checklist: Adding New Features

### **Adding New Exchange:**
- [ ] Add new API class in `components/exchanges/`
- [ ] Inherit `BaseExchangeAPI`
- [ ] **Use async executor pattern in ALL methods**
- [ ] Add cache manager integration (TTL: ticker=5s, orderbook=1s, klines=60s)
- [ ] Add config file (`config/connectors.yaml`)
- [ ] Implement health check
- [ ] Test (ticker, orderbook, balance, create_order)

### **Adding New Module:**
- [ ] Add new module folder in `modules/`
- [ ] Determine module type (uvicorn/flask/python/thread)
- [ ] Add module definition in `config/daemon.yaml`
- [ ] Add health check endpoint (for HTTP-based modules)
- [ ] Add event bus integration (event publish/subscribe)
- [ ] Use cache manager (shared cache access)
- [ ] Add graceful shutdown handler
- [ ] Test (start/stop/restart/crash recovery)

### **Performance Optimization:**
- [ ] Check async executor pattern (any sync calls?)
- [ ] Check cache hit rate (>70% target)
- [ ] Check connection pool usage (any leaks?)
- [ ] Check event bus overhead (too many events?)
- [ ] Check memory leak (monitoring with psutil)

---

## 🎓 Best Practices

### **1. Config-Driven Design**
- No hard-coded values, everything in config
- Environment-specific config (testnet/production)
- Hot reload support (SIGHUP signal)

### **2. Event-Driven Communication**
- NO direct calls between modules
- Pub/sub via event bus
- Loose coupling, high cohesion

### **3. Shared Resources**
- CacheManager: Used by all modules
- ConnectionPool: Used by all modules
- EventBus: Used by all modules

### **4. Graceful Degradation**
- Module crash → auto-restart (watchdog)
- Redis down → fallback to memory cache
- Connection fail → retry with exponential backoff

### **5. Monitoring & Alerting**
- Health check per module
- Metrics collection (CPU, RAM, threads)
- Alert on crash/restart
- Log aggregation

---

## 🚨 CRITICAL: Don't Do List

### **❌ NEVER DO:**

1. **Block event loop**
   ```python
   # ❌ DON'T
   def get_ticker(self):
       return self.client.get_ticker()  # Sync call in async context
   ```

2. **Bypass shared resources**
   ```python
   # ❌ DON'T - Each module creates its own connector
   connector = BinanceAPI(config)

   # ✅ DO - Use daemon's shared connector
   connector = self.daemon.connector
   ```

3. **Hard-coded config**
   ```python
   # ❌ DON'T
   API_KEY = "xyz123"

   # ✅ DO
   api_key = self.config.get('binance.endpoints.production.api_key')
   ```

4. **Direct module call**
   ```python
   # ❌ DON'T
   trading_engine.execute_trade()

   # ✅ DO - Use event bus
   self.event_bus.publish('trade.execute', {'symbol': 'BTCUSDT', 'side': 'BUY'})
   ```

5. **Exception swallow**
   ```python
   # ❌ DON'T
   try:
       something()
   except:
       pass  # Silent fail

   # ✅ DO
   try:
       something()
   except Exception as e:
       self.logger.error(f"Error: {e}")
       raise  # Or handle it
   ```

---

## 📞 Integration Points

### **Trading Engine → Exchange API**
```python
# modules/trading/engine.py
from components.exchanges.binance_api import BinanceAPI

# Get ticker with async executor
ticker = await self.connector.get_ticker("BTCUSDT")

# Reads from cache (5s TTL), goes to API on cache miss
```

### **Module → Event Bus**
```python
# Event publish
await self.event_bus.publish_async(
    topic='price.BTCUSDT.update',
    data={'price': 50000, 'timestamp': time.time()},
    source='trading_engine'
)

# Event subscribe
def on_price_update(event):
    print(f"New price: {event.data['price']}")

self.event_bus.subscribe('price.*.update', on_price_update)
```

### **Module → Cache**
```python
# Cache write
self.cache_manager.set('ticker:BTCUSDT', ticker_data, ttl=5)

# Cache read
cached = self.cache_manager.get('ticker:BTCUSDT')
if cached:
    return cached  # Cache hit
else:
    # Cache miss, fetch from API
    data = await self.connector.get_ticker('BTCUSDT')
    self.cache_manager.set('ticker:BTCUSDT', data, ttl=5)
    return data
```

---

## 🎯 Summary

**SuperBot Daemon Architecture:**
- ✅ Master daemon orchestrator (superbot.py)
- ✅ Shared resources (cache, event bus, connection pool)
- ✅ Multi-module system (AI, WebUI, Trading, Backtest)
- ✅ IPC/RPC communication (JSON-RPC 2.0)
- ✅ Health monitoring & auto-restart (watchdog)
- ✅ Task scheduling (cron-like)
- ✅ Graceful shutdown
- ✅ Config-driven design
- ✅ Event-driven communication
- ✅ Performance optimized (async executor, cache, connection pool)

**Remember:**
1. **Async executor pattern** use everywhere
2. **Event bus** for inter-module communication
3. **Cache manager** for performance optimization
4. **Config-driven** everything
5. **Graceful degradation** always

**More Information:**
- `docs/architecture/` → Architecture documents
- `docs/api/` → API references
- `config/` → All configuration files
- `temp/` → Old system reference (binance_client.py, exchange_*.py)

---

**Last Updated:** 2025-11-26
**Written by:** SuperBot Team & Claude (Session Analysis)
