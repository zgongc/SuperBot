# components/data - Data Management Layer

**SuperBot Data Layer** - Data download, storage, and management components.

---

## 📋 Overview

This directory contains all of SuperBot's data management operations:
- **Database Management:** SQLite/PostgreSQL veri saklama
- **Historical Data:** Parquet-based historical OHLCV data
- **Data Download:** Exchange'lerden veri indirme
- **Timeframe Resampling:** Creating a higher timeframe from a lower timeframe.

---

## 📁 Dosyalar

### 1. **database_engine.py** → `core/database_engine.py`
**Responsibility:** Database connection management

**Features:**
- ✅ SQLite/PostgreSQL dual backend (config-driven)
- ✅ Async SQLAlchemy 2.0 engine
- ✅ Connection pooling
- ✅ Session factory
- ✅ Auto table creation
- ✅ Health check
- ✅ Graceful shutdown

**Usage:**
```python
from core.database_engine import DatabaseEngine

db = DatabaseEngine(config, logger)
await db.initialize()

async with db.get_session() as session:
    result = await session.execute(query)

await db.shutdown()
```

**Config:** `config/infrastructure.yaml` -> `database` section

---

### 2. **database_models.py**
**Responsibility:** SQLAlchemy ORM model definitions

**Features:**
- ✅ SQLAlchemy `Base` class
- ⏳ Models will be added as needed (currently 0 tables)

**Usage:**
```python
from components.data.database_models import Base
from sqlalchemy import Column, Integer, String, Float

# Yeni model ekle
class Candle(Base):
    __tablename__ = "candles"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(Integer, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
```

**Note:** The DatabaseEngine automatically creates the table after you add it.

---

### 3. **database_manager.py**
**Responsibility:** DatabaseEngine facade - unified API

**Features:**
- ✅ DatabaseEngine wrapper
- ✅ Session context manager
- ✅ Health check proxy
- ✅ Singleton pattern
- ⏳ CRUD methods will be added as needed (currently NOT available)

**Usage:**
```python
from components.data.database_manager import get_database_manager

dm = get_database_manager()
await dm.initialize()

# Custom query
async with dm.get_session() as session:
    result = await session.execute(select(Model).where(...))

# Health check
is_healthy = await dm.health_check()

await dm.shutdown()
```

**Singleton:** `get_database_manager()` returns the global instance.

**Future CRUD examples:**
```python
# To be added when needed:
await dm.save_candle(candle_data)
candles = await dm.get_candles(symbol, timeframe, limit=100)
await dm.save_trade(trade_data)
trades = await dm.get_open_trades()
```

---

### 4. **historical_data_manager.py**
**Responsibility:** Historical OHLCV data management (Parquet-based)

**Features:**
- ✅ Multi-backend: Parquet/SQLite/PostgreSQL/CSV
- ✅ Smart incremental updates (duplicate prevention)
- ✅ Date range filtering
- ✅ Data validation & cleaning
- ✅ Cache management
- ✅ Integration with data_downloader.py

**Usage:**
```python
from components.data.historical_data_manager import HistoricalDataManager

hdm = HistoricalDataManager(config, logger)

# Load historical data
df = await hdm.load_data(
    symbol='BTCUSDT',
    timeframe='1m',
    start_date='2025-01-01',
    end_date=None  # Up to now
)

# Update data (incremental)
await hdm.update_data(
    symbol='BTCUSDT',
    timeframes=['1m', '5m']
)

# Get info
info = hdm.get_data_info('BTCUSDT')
```

**Data Source:** Parquet files (`data/parquets/`)

**Use Cases:**
- Backtest Module (historical data loading)
- AI Training (feature engineering data)
- Analysis (indicator calculations)

---

### 5. **data_downloader.py**
**Responsibility:** Downloading historical data from exchanges.

**Features:**
- ✅ Binance API integration
- ✅ All timeframes support (1m → 1M)
- ✅ Smart incremental update (son timestamp'ten devam)
- ✅ Duplicate detection & removal
- ✅ Parquet save
- ✅ Progress tracking

**Usage:**
```python
from components.data.data_downloader import DataDownloader

downloader = DataDownloader()

# Initial download
await downloader.download(
    symbol='BTCUSDT',
    timeframe='1m',
    start_date='2025-01-01',
    output_dir='data/parquets'
)

# Update (incremental)
await downloader.update(
    symbol='BTCUSDT',
    timeframe='1m',
    output_dir='data/parquets'
)
```

**Output:** Parquet files
- Format: `BTCUSDT_1m_2025.parquet`

**Dependencies:** `python-binance`, `pandas`, `pyarrow`

---

### 6. **timeframe_resampler.py**
**Responsibility:** Conversion from lower timeframe to higher timeframe.

**Features:**
- ✅ Smart source selection (closest lower timeframe)
- ✅ OHLCV aggregation (pandas resample)
- ✅ File naming with `_re` suffix
- ✅ Volume summation
- ✅ Validation

**Usage:**
```python
from components.data.timeframe_resampler import TimeframeResampler

resampler = TimeframeResampler(data_dir='data/parquets')

# 1m → 2h resample
df_2h = resampler.resample(
    symbol='BTCUSDT',
    target_tf='2h',
    year=2025
)
```

**Output:** Resampled parquet files
- Format: `BTCUSDT_2h_2025_re1m.parquet` (resampled from 1m)

**Use Cases:**
- Missing timeframe data (2h, 3h, 6h, 8h, 3d)
- Backtest optimization (less data)

**Resample Hierarchy:**
- 3m → 1m
- 2h → 1h
- 6h → 4h, 2h, 1h
- 8h → 4h, 2h, 1h
- 3d → 1d

---

## 🗂️ Data Flow

### Historical Data Pipeline

```
┌─────────────────────┐
│  Exchange (Binance) │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  data_downloader.py │  ← Download/Update historical data
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Parquet Files      │  ← data/parquets/BTCUSDT_1m_2025.parquet
└──────────┬──────────┘
           │
           ├→ ┌──────────────────────────┐
           │  │ timeframe_resampler.py   │  ← 1m → 2h, 3h, etc.
           │  └──────────┬───────────────┘
           │             ↓
           ├→ ┌──────────────────────────┐
           │  │ Resampled Parquet Files  │
           │  └──────────────────────────┘
           │
           ↓
┌─────────────────────────────┐
│ historical_data_manager.py  │  ← Unified data loader
└──────────┬──────────────────┘
           │
           ├→ Backtest Module
           ├→ AI Training Module
           └→ Analysis Module
```

### Real-time Data Pipeline (Gelecek)

```
┌─────────────────────┐
│  WebSocket Stream   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Database (SQLite)  │  ← database_manager.py
│  LiveKlineBuffer    │     (live candles before archiving)
└──────────┬──────────┘
           │
           ├→ Trading Engine (real-time)
           │
           └→ Archive to Parquet (daily/weekly)
```

---

## 🎯 Database vs Parquet

### Database (SQLite/PostgreSQL)
**Purpose:** Real-time operational data

**Use Cases:**
- Live trading data (positions, orders, trades)
- WebUI dashboard (recent data)
- Real-time monitoring
- Session tracking

**Avantajlar:**
- Fast queries (indexed)
- Transactional
- Relational (foreign keys)

**Dezavantajlar:**
- Limited size (SQLite)
- Slower for bulk analytics

---

### Parquet Files
**Purpose:** Historical bulk data storage

**Use Cases:**
- Backtest data (years of OHLCV)
- AI training datasets
- Analytics & indicators
- Long-term archival

**Avantajlar:**
- Compressed (snappy)
- Fast columnar reads
- Unlimited size
- Portable

**Dezavantajlar:**
- No updates (immutable)
- No relations
- File-based queries

---

## 🚀 Usage Scenarios

### Scenario 1: Data Preparation for Backtesting
```python
# 1. Download historical data
downloader = DataDownloader()
await downloader.download('BTCUSDT', '1m', start_date='2024-01-01')

# 2. Resample to higher timeframe (if needed)
resampler = TimeframeResampler()
df_1h = resampler.resample('BTCUSDT', '1h', year=2024)

# 3. Load for backtest
hdm = HistoricalDataManager(config, logger)
df = await hdm.load_data('BTCUSDT', '1m', start_date='2024-01-01')

# 4. Run backtest
# ...
```

### Senaryo 2: Live Trading Data Kaydetme (Gelecek)
```python
# DatabaseManager will be used
dm = get_database_manager()
await dm.initialize()

# Real-time candle kaydet
await dm.save_candle({
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "timestamp": 1700000000000,
    "open": 50000,
    "high": 50100,
    "low": 49900,
    "close": 50050,
    "volume": 100.5
})

# Save trade
await dm.save_trade({
    "symbol": "BTCUSDT",
    "side": "LONG",
    "entry_price": 50000,
    "quantity": 0.1,
    # ...
})
```

### Senaryo 3: WebUI Dashboard Data (Gelecek)
```python
# Recent trades
trades = await dm.get_trades(limit=50)

# Open positions
positions = await dm.get_open_positions()

# Portfolio balance
balance = await dm.get_latest_balance()
```

---

## 📊 File Organization

```
components/data/
├── database_models.py          # SQLAlchemy Base + Models (to be added as needed)
├── database_manager.py         # DatabaseEngine facade (CRUD operations will be added as needed)
├── historical_data_manager.py  # Parquet-based historical data loader
├── data_downloader.py          # Binance historical data downloader
├── timeframe_resampler.py      # Timeframe resampling (1m → 2h, etc.)
└── README.md                   # This file

core/
└── database_engine.py          # Database connection manager
```

---

## 🔧 Configuration

**Config:** `config/infrastructure.yaml`

```yaml
database:
  backend: "sqlite"  # sqlite, postgresql

  sqlite:
    path: "data/database/superbot.db"
    timeout: 30
    check_same_thread: false
    wal_mode: true
    pool_size: 5
    max_overflow: 10

  postgresql:
    host: "${POSTGRES_HOST}"
    port: "${POSTGRES_PORT}"
    database: "${POSTGRES_DB}"
    user: "${POSTGRES_USER}"
    password: "${POSTGRES_PASSWORD}"
    pool_size: 10
    max_overflow: 20
```

---

## ⚡ First Use

### 1. Database Setup
```python
from components.data.database_manager import get_database_manager

dm = get_database_manager()
await dm.initialize()  # Database and tables are created

# Health check
is_healthy = await dm.health_check()
```

### 2. Historical Data Download
```python
from components.data.data_downloader import DataDownloader

downloader = DataDownloader()
await downloader.download('BTCUSDT', '1m', start_date='2024-01-01')
# → data/parquets/BTCUSDT_1m_2024.parquet
```

### 3. Load Historical Data
```python
from components.data.historical_data_manager import HistoricalDataManager

hdm = HistoricalDataManager(config, logger)
df = await hdm.load_data('BTCUSDT', '1m', start_date='2024-01-01')
# → pandas DataFrame (OHLCV)
```

---

## 🎯 Development Plan

### Phase 1: Base Infrastructure ✅
- [x] DatabaseEngine (core/database_engine.py)
- [x] Base class (database_models.py)
- [x] DatabaseManager facade (database_manager.py)
- [x] Historical data (historical_data_manager.py)
- [x] Data downloader (data_downloader.py)
- [x] Timeframe resampler (timeframe_resampler.py)

### Phase 2: Models (as needed)
First need: **WebUI Portfolio Module**
- [ ] ExchangeSymbol model (symbol list)
- [ ] SymbolFavorite model (user favorites)
- [ ] Portfolio model (portfolio definitions)
- [ ] PortfolioPosition model (portfolio positions)
- [ ] Corresponding CRUD methods

**Backtest Module requirement:**
- [ ] BacktestRun model
- [ ] BacktestTrade model
- [ ] Strategy model

**Live Trading requirement:**
- [ ] LiveTrade model
- [ ] Order model
- [ ] Position model

### Phase 3: Advanced Features (Long Term)
- [ ] Alembic migrations (schema versioning)
- [ ] Repository pattern (clean CRUD separation)
- [ ] Bulk operations optimization
- [ ] Query performance tuning
- [ ] Data archival (old data cleanup)

---

## 🔍 Database vs Parquet - Ne Zaman Hangisi?

### Use Database:
✅ Real-time data (live trading positions)
✅ Transactional data (orders, trades)
✅ Recent data queries (last 100 trades)
✅ Relational data (trade ↔ orders)
✅ WebUI dashboard (dynamic queries)

### Use Parquet:
✅ Historical bulk data (years of OHLCV)
✅ Backtest data (static datasets)
✅ AI training datasets (millions of rows)
✅ Analytics (indicator calculations)
✅ Long-term archival (immutable history)

---

## 📝 Development Notes

### Yeni Model Eklemek:

**1. database_models.py'ye ekle:**
```python
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10))  # LONG/SHORT
    entry_price = Column(Float)
    quantity = Column(Float)
    timestamp = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**2. DatabaseEngine automatically creates tables:**
```python
await db.initialize()  # Base.metadata.create_all() runs automatically
```

**3. Add CRUD operations to DatabaseManager (optional):**
```python
async def save_trade(self, trade_data: dict) -> bool:
    """Record a trade"""
    async with self.get_session() as session:
        trade = Trade(**trade_data)
        session.add(trade)
        await session.commit()
    return True

async def get_trades(self, symbol: str, limit: int = 100) -> List[dict]:
    """Get trades"""
    async with self.get_session() as session:
        result = await session.execute(
            select(Trade)
            .where(Trade.symbol == symbol)
            .order_by(Trade.timestamp.desc())
            .limit(limit)
        )
        trades = result.scalars().all()
        return [{"symbol": t.symbol, "side": t.side, ...} for t in trades]
```

### Test Etme:
```bash
# Model test
python components/data/database_models.py

# DatabaseManager test
python components/data/database_manager.py

# DatabaseEngine test
python core/database_engine.py
```

---

## 🚨 Important Rules

### ✅ DO:
- Make sure you **need** to add a model before doing so.
- Tables should be kept **minimal** (avoid adding unnecessary fields).
- Add **lazy** CRUD methods (add them when they are needed).
- Test (after each new model/method)

### ❌ DON'T:
- Adding 40 tables "just in case"
- Do not add unused fields.
- Move WebUI-specific logic to the DatabaseManager.
- Repository pattern is currently NOT IMPLEMENTED (over-engineering)

---

## 🔗 Related Files

- `core/database_engine.py` - Database connection layer
- `config/infrastructure.yaml` - Database config
- `docs/plans/data_manager_implementation_plan.md` - Detailed plan (IGNORE - overly detailed)

---

**Created:** 2025-11-25
**Status:** ✅ Base infrastructure is ready - Models/CRUD will be added as needed.
