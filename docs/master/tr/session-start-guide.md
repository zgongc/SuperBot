# SuperBot - Claude Session Start Guide
**🚀 Her Yeni Session Başında Bu Dosyayı Oku!**

---

## ⚡ Hızlı Referans (5 Dakikada Oku)

### 1. **Proje Nedir?**
SuperBot = Multi-exchange crypto trading platform with **daemon-based architecture**

**Modüller:**
- 🤖 **AI Module:** FastAPI (uvicorn) - Strategy optimization
- 🌐 **WebUI Module:** Flask (waitress) - Web interface
- 📈 **Trading Module:** Async Python - Live/Paper trading
- 🔬 **Backtest Module:** On-demand - Strategy testing
- 🐕 **Watchdog:** Health monitoring & auto-restart

**Core Infrastructure:**
- ConfigEngine, LoggerEngine, EventBus, CacheManager, ProcessManager, GracefulShutdown

---

### 2. **KRİTİK Bilgiler (Mutlaka Bil!)**

#### ❗ **"Tekerleği Yeniden İcat Etme"**
- Mevcut kod **çalışıyor**, bozma!
- Old system'den kopyalanan pattern'ler var (`temp/` klasöründe)
- "Daha iyi yol" önerme, **mevcut pattern'i takip et**

#### ❗ **Daemon Architecture Neden Var?**
```
superbot.py (Master Daemon)
    ↓
Shared Resources (Cache, EventBus, ConnectionPool)
    ↓
Modules (Trading, WebUI, AI, Backtest)
```

**Her modül ayrı process'te çalışır!** Daemon orchestration yapıyor.

#### ❗ **Neden 3 Exchange Dosyası Var?**
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

**Hepsi gerekli! Sil!**

#### ❗ **Async Executor Pattern (SUPER KRİTİK!)**
```python
# ❌ YANLIŞ - Event loop'u bloklar, 10x yavaşlama
ticker = self.client.get_ticker(symbol="BTCUSDT")

# ✅ DOĞRU - Non-blocking, 8-10x hızlanma
loop = asyncio.get_event_loop()
ticker = await loop.run_in_executor(
    None,
    lambda: self.client.get_ticker(symbol="BTCUSDT")
)
```

**Neden:** python-binance sync, trading engine async → executor pattern zorunlu!

**Nerede kullan:**
- ✅ **TÜM exchange API methodlarında** (get_ticker, get_orderbook, create_order, etc.)
- ✅ BinanceAPI içinde
- ✅ ConnectorEngine içinde

---

### 3. **Dosya Yapısı**

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
│   ├── main.yaml            # Ana config
│   ├── connectors.yaml      # Exchange configs
│   ├── infrastructure.yaml  # Cache, EventBus, ConnectionPool
│   └── daemon.yaml          # Module definitions, autostart
│
├── temp/                    # OLD system reference (SİLME!)
│   ├── binance_client.py         # Async executor pattern örneği
│   ├── exchange_connector_engine.py
│   └── exchange_connection_engine.py
│
└── docs/
    └── claude/
        ├── daemon-architecture-guide.md  # DETAYLI REHBER
        └── session-start-guide.md        # BU DOSYA
```

---

### 4. **Event Bus (Modüller Arası İletişim)**

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

**Wildcard support:** `price.*.update` tüm symbol'leri dinler

---

### 5. **Cache Manager (Performance)**

```python
# Cache write (5s TTL ticker için optimal)
cache_manager.set('ticker:BTCUSDT', ticker_data, ttl=5)

# Cache read
cached = cache_manager.get('ticker:BTCUSDT')
if cached:
    return cached  # Cache hit
else:
    # Cache miss, API'ye git
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

### 7. **RPC Methods (17 adet)**

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

#### Issue 1: "Event loop blocking" (10x yavaşlama)
**Çözüm:** Async executor pattern kullan (yukarıdaki örnek)

#### Issue 2: "Module başlatılamadı"
**Debug:**
```bash
tail -f logs/daemon.log
python -m modules.trading.engine --mode paper  # Manuel test
```

#### Issue 3: "Cache hit rate düşük"
**Çözüm:** TTL değerlerini kontrol et (ticker=5s, orderbook=1s, klines=60s)

#### Issue 4: "Connection pool exhausted"
**Çözüm:** `config/infrastructure.yaml` → `max_connections: 20` (artır)

---

### 9. **Checklist: Kod Değişikliği Yaparken**

- [ ] Async executor pattern kullandın mı? (exchange API'de)
- [ ] Event bus entegrasyonu ekledin mi? (module'lerde)
- [ ] Cache manager kullandın mı? (performance için)
- [ ] Config-driven mı? (hard-coded value yok)
- [ ] Graceful shutdown handler ekledin mi?
- [ ] Health check implement ettin mi?
- [ ] Log ekledi mi? (logger.info/error)
- [ ] Test ettin mi? (manuel + unit test)

---

### 10. **Yapma Listesi**

**❌ ASLA YAPMA:**

1. **Event loop blokla**
   ```python
   # ❌ Sync call in async context
   ticker = self.client.get_ticker()
   ```

2. **Shared resource bypass et**
   ```python
   # ❌ Her module kendi connector'ını yapar
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

   # ✅ Event bus kullan
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

## 📚 Daha Fazla Bilgi

**Detaylı rehber oku:**
```bash
docs/claude/daemon-architecture-guide.md
```

**Bu dosya içerir:**
- Daemon architecture detayları
- Module lifecycle detayları
- RPC communication detayları
- Performance optimization
- Troubleshooting
- Best practices
- Integration examples

---

## 🎯 Özet (30 Saniyede)

1. **Daemon architecture** → Modüller ayrı process'lerde
2. **Async executor pattern** → python-binance sync, event loop blocking önleme
3. **Event bus** → Modüller arası iletişim
4. **Cache manager** → Performance (>70% hit rate hedef)
5. **Config-driven** → Her şey config'te
6. **Tekerleği yeniden icat etme** → Çalışan kodu boz

**Şimdi ne yapmalı:**
1. Bu dosyayı oku ✅
2. `docs/claude/daemon-architecture-guide.md` oku (detaylı)
3. User'ın ne istediğini anla
4. Mevcut pattern'leri takip et
5. Code yaz, test et

---

**İyi çalışmalar! 🚀**
