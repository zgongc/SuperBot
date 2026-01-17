# SuperBot Daemon Architecture - Claude Guide
**Tarih:** 2025-11-26
**Versiyon:** 1.0.0
**Yazar:** SuperBot Team

---

## 🎯 ÖNEMLİ: Bu Belgeyi Mutlaka Oku!

Bu belge, SuperBot'un daemon mimarisini ve kritik tasarım kararlarını açıklar. **Yeni değişiklik yapmadan önce bu belgeyi oku!**

---

## 📚 Temel Prensipler

### 1. **Tekerleği Yeniden İcat Etme!**
- Mevcut çalışan sistemden kopyalanan kodlar VAR
- Bu kodlar **denenmiş ve test edilmiş**
- Yeni özellik eklerken **mevcut pattern'leri takip et**
- "Daha iyi yol" önerme, **çalışan sistemi bozma**

### 2. **Daemon Architecture - Neden Var?**
SuperBot bir **daemon-based multi-module system**:

```
superbot.py (Master Daemon)
    ↓
Shared Resources (CacheManager, EventBus, ConnectionPool)
    ↓
Modules (Trading, WebUI, AI, Backtest)
```

**Neden böyle?**
- Tek bir Python process'i yeterli değil
- Her modül kendi process'inde çalışır (isolation)
- Shared resources ile communication (event bus, cache)
- Central orchestration (start/stop/restart/health check)

---

## 🏗️ Core Architecture

### **CORE Infrastructure Layer**

#### 1. ConfigEngine
- Tüm config dosyalarını yükler (`config/*.yaml`)
- Hot reload destekler (SIGHUP signal)
- Nested config access: `config.get('infrastructure.cache.backend')`

#### 2. LoggerEngine
- Unified logging system
- Ultra compact format (INFO seviyesi için tek satır)
- File rotation, console output

#### 3. EventBus
- **Pub/Sub messaging** (modüller arası iletişim)
- Topic-based routing: `price.BTCUSDT.update`, `trade.executed`
- Wildcard support: `price.*.update`
- Memory & Redis backend
- **KRİTİK:** Trading engine tüm event'leri buraya yayınlar

#### 4. CacheManager
- **Shared cache** (tüm modüller kullanır)
- Memory & Redis backend
- TTL, LRU eviction
- Ticker, orderbook, klines için cache
- **5 saniye TTL** ticker için optimal

#### 5. ProcessManager
- Engine lifecycle management
- Dependency-based startup order
- Auto-restart on crash
- Health check monitoring

#### 6. GracefulShutdown
- Signal handling (SIGINT, SIGTERM, SIGBREAK)
- Callback system (pre/cleanup/post)
- State persistence
- Position close (opsiyonel)

---

### **Daemon Components Layer**

#### 1. IPCServer
- **JSON-RPC 2.0** over Unix socket (Linux/Mac)
- TCP fallback for Windows (127.0.0.1:9999)
- RPC method handler registration
- Authentication token support
- **17 RPC method** (daemon control, module control, monitoring)

#### 2. ModuleLauncher
- **4 module type** desteği:
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
- **Periodic health checks** (psutil ile process monitoring)
- Auto-restart on crash
- Restart limits (default: 3 attempts)
- Restart cooldown (default: 60s)
- Alert system (event bus üzerinden)
- **KRİTİK:** Process'in zombie olup olmadığını kontrol eder

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

### **Neden 3 Farklı Dosya Var?**

#### 1. **BinanceAPI** (`components/exchanges/binance_api.py`)
- **Kullanım:** Modüller içinde doğrudan kullanım
- **Özellikler:**
  - BaseExchangeAPI implementation
  - Async executor pattern (sync → async wrapper)
  - Cache manager integration
  - Config-driven (testnet/production)
- **Örnek:** `modules/trading/modes/paper_mode.py` içinde kullanılır

#### 2. **ConnectorEngine** (`temp/exchange_connector_engine.py`)
- **Kullanım:** Daemon shared resource
- **Özellikler:**
  - python-binance wrapper
  - Cache manager integration
  - Rate limiting
  - Retry mechanism
  - **Tüm modüller aynı connector'ı paylaşır**
- **Neden gerekli:** Daemon architecture için central connector

#### 3. **ConnectionPoolEngine** (`temp/exchange_connection_engine.py`)
- **Kullanım:** Daemon shared connection pool
- **Özellikler:**
  - HTTP/WebSocket connection pooling (aiohttp)
  - Min/max connection management
  - Health check loop
  - Idle timeout, auto-reconnect
  - **Tüm modüller aynı pool'u kullanır**
- **Neden gerekli:** Connection reuse, performance optimization

**NOT:** `connection_pool.py` ve `exchange_connection_engine.py` aynı dosya → biri `_deprecated`'e taşındı

---

## ⚡ Performance Critical: Async Executor Pattern

### **Problem: Event Loop Blocking**
```python
# ❌ YANLIŞ - Event loop'u bloklar
def get_ticker(self, symbol: str):
    return self.client.get_ticker(symbol=symbol)  # Sync call
```

### **Çözüm: Async Executor**
```python
# ✅ DOĞRU - Non-blocking
async def get_ticker(self, symbol: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: self.client.get_ticker(symbol=symbol)
    )
    return result
```

**Neden önemli:**
- python-binance **sync** library
- Trading engine **async** (asyncio event loop)
- Sync call → event loop freeze → 10x yavaşlama
- Async executor → non-blocking → 8-10x hızlanma

**Nerede kullanılır:**
- ✅ `BinanceAPI` (TÜM methodlarda)
- ✅ `ConnectorEngine` (TÜM methodlarda)
- ✅ `temp/binance_client.py` (OLD system - reference)

---

## 🎭 Module Lifecycle

### **Module Types ve Launch Methods**

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

**Launch:** Threading.Thread ile arka planda çalıştırılır

---

## 📡 RPC Communication

### **IPC Server - JSON-RPC 2.0**

**Socket path:**
- Linux/Mac: `/tmp/superbot.sock` (Unix socket)
- Windows: `127.0.0.1:9999` (TCP fallback)

**RPC Methods (17 adet):**

#### Daemon Control
- `daemon.status` → Daemon durumu
- `daemon.stop` → Daemon'u durdur
- `daemon.reload_config` → Config reload (SIGHUP)

#### Module Control
- `module.start` → Module başlat
- `module.stop` → Module durdur
- `module.restart` → Module restart
- `module.status` → Module durumu
- `module.list` → Tüm modülleri listele

#### Trading (Proxy to Trading Module)
- `trading.positions` → Açık pozisyonlar
- `trading.orders` → Emirler
- `trading.balance` → Bakiye

#### Monitoring
- `monitoring.health` → Health check
- `monitoring.metrics` → Metrikler
- `monitoring.resources` → Kaynak kullanımı (CPU, RAM, threads)

#### Logs
- `logs.stream` → Log streaming (WebSocket önerilir)

**Örnek RPC call:**
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

### **1. Daemon Başlatma**
```bash
python superbot.py
```

**Sequence:**
1. Check already running (PID file kontrolü)
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

### **2. Module Başlatma (via RPC)**
```bash
# CLI ile
superbot-cli module start trading

# RPC ile
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

# Ya da RPC ile
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
    # - trading  # Manuel başlatılacak

  # IPC config
  ipc:
    socket_path: "/tmp/superbot.sock"
    auth_token: null  # Opsiyonel

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

### **Issue 1: "Daemon zaten çalışıyor"**
**Sebep:** PID file mevcut ve process çalışıyor

**Çözüm:**
```bash
# Daemon'u durdur
superbot-cli daemon stop

# Ya da PID'yi manuel kill et
cat .superbot/daemon.pid
kill <pid>

# Stale PID file varsa sil
rm .superbot/daemon.pid
```

---

### **Issue 2: "Module başlatılamadı"**
**Sebep:** Port kullanımda, dependency eksik, ya da config hatası

**Debug:**
```bash
# Module logs'u kontrol et
tail -f logs/daemon.log

# Manuel başlatmayı test et
python -m modules.trading.engine --mode paper
```

**Çözüm:**
- Port değiştir (config/daemon.yaml)
- Dependency kur (`pip install -r requirements.txt`)
- Config düzelt

---

### **Issue 3: "Event loop blocking"**
**Sebep:** Sync API call async context içinde

**Belirti:**
- Trading engine yavaş (10x)
- WebUI freezing
- Timeout errors

**Çözüm:**
```python
# ❌ YANLIŞ
ticker = self.client.get_ticker(symbol="BTCUSDT")

# ✅ DOĞRU
loop = asyncio.get_event_loop()
ticker = await loop.run_in_executor(
    None,
    lambda: self.client.get_ticker(symbol="BTCUSDT")
)
```

**Kontrol et:**
- `components/exchanges/binance_api.py` → TÜM methodlarda async executor var mı?
- `temp/exchange_connector_engine.py` → TÜM methodlarda async executor var mı?

---

### **Issue 4: "Connection pool exhausted"**
**Sebep:** Connection leak, timeout, ya da max_connections düşük

**Çözüm:**
```yaml
# config/infrastructure.yaml
infrastructure:
  connection_pool:
    min_connections: 2
    max_connections: 10  # Artır: 20
    connection_timeout: 30
    idle_timeout: 300
```

---

### **Issue 5: "Cache hit rate düşük"**
**Sebep:** TTL çok düşük, ya da cache disabled

**Çözüm:**
```python
# Ticker için optimal TTL: 5 saniye
self.cache_manager.set(cache_key, result, ttl=5)

# Orderbook için: 1 saniye
self.cache_manager.set(cache_key, result, ttl=1)

# Klines için: 60 saniye
self.cache_manager.set(cache_key, result, ttl=60)
```

**Stats kontrol:**
```python
stats = cache_manager.get_stats()
print(f"Hit rate: {stats['cache_hit_rate']}")
# Target: >70% hit rate
```

---

## 📋 Checklist: Yeni Özellik Eklerken

### **Yeni Exchange Ekleme:**
- [ ] `components/exchanges/` içine yeni API class'ı ekle
- [ ] `BaseExchangeAPI` inherit et
- [ ] **TÜM methodlarda async executor pattern kullan**
- [ ] Cache manager integration ekle (TTL: ticker=5s, orderbook=1s, klines=60s)
- [ ] Config dosyası ekle (`config/connectors.yaml`)
- [ ] Health check implement et
- [ ] Test et (ticker, orderbook, balance, create_order)

### **Yeni Module Ekleme:**
- [ ] `modules/` içine yeni module klasörü ekle
- [ ] Module type belirle (uvicorn/flask/python/thread)
- [ ] `config/daemon.yaml` içine module tanımı ekle
- [ ] Health check endpoint ekle (HTTP-based modules için)
- [ ] Event bus integration ekle (event publish/subscribe)
- [ ] Cache manager kullan (shared cache access)
- [ ] Graceful shutdown handler ekle
- [ ] Test et (start/stop/restart/crash recovery)

### **Performance Optimization:**
- [ ] Async executor pattern kontrol et (sync calls var mı?)
- [ ] Cache hit rate kontrol et (>70% hedef)
- [ ] Connection pool kullanımı kontrol et (leak var mı?)
- [ ] Event bus overhead kontrol et (çok fazla event var mı?)
- [ ] Memory leak kontrol et (psutil ile monitoring)

---

## 🎓 Best Practices

### **1. Config-Driven Design**
- Hard-coded value yok, her şey config'te
- Environment-specific config (testnet/production)
- Hot reload support (SIGHUP signal)

### **2. Event-Driven Communication**
- Module'ler arası direct call YOK
- Event bus üzerinden pub/sub
- Loose coupling, high cohesion

### **3. Shared Resources**
- CacheManager: Tüm modüller kullanır
- ConnectionPool: Tüm modüller kullanır
- EventBus: Tüm modüller kullanır

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

## 🚨 KRİTİK: Yapma Listesi

### **❌ ASLA YAPMA:**

1. **Event loop'u blokla**
   ```python
   # ❌ YAPMA
   def get_ticker(self):
       return self.client.get_ticker()  # Sync call in async context
   ```

2. **Shared resource'u bypass et**
   ```python
   # ❌ YAPMA - Her module kendi connector'ını yaratır
   connector = BinanceAPI(config)

   # ✅ YAP - Daemon'un shared connector'ını kullan
   connector = self.daemon.connector
   ```

3. **Hard-coded config**
   ```python
   # ❌ YAPMA
   API_KEY = "xyz123"

   # ✅ YAP
   api_key = self.config.get('binance.endpoints.production.api_key')
   ```

4. **Direct module call**
   ```python
   # ❌ YAPMA
   trading_engine.execute_trade()

   # ✅ YAP - Event bus kullan
   self.event_bus.publish('trade.execute', {'symbol': 'BTCUSDT', 'side': 'BUY'})
   ```

5. **Exception swallow**
   ```python
   # ❌ YAPMA
   try:
       something()
   except:
       pass  # Silent fail

   # ✅ YAP
   try:
       something()
   except Exception as e:
       self.logger.error(f"Error: {e}")
       raise  # Ya da handle et
   ```

---

## 📞 Integration Points

### **Trading Engine → Exchange API**
```python
# modules/trading/engine.py
from components.exchanges.binance_api import BinanceAPI

# Async executor ile ticker al
ticker = await self.connector.get_ticker("BTCUSDT")

# Cache'den okur (5s TTL), cache miss ise API'ye gider
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

**Unutma:**
1. **Async executor pattern** her yerde kullan
2. **Event bus** ile modüller arası iletişim
3. **Cache manager** ile performance optimization
4. **Config-driven** her şey
5. **Graceful degradation** her zaman

**Daha Fazla Bilgi:**
- `docs/architecture/` → Mimari dökümanlar
- `docs/api/` → API referansları
- `config/` → Tüm configuration dosyaları
- `temp/` → Old system reference (binance_client.py, exchange_*.py)

---

**Son Güncelleme:** 2025-11-26
**Yazan:** SuperBot Team & Claude (Session Analysis)
