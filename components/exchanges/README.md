# 🏦 Exchange API - SuperBot

Exchange bağlantıları ve API wrapper'ları

**Son Güncelleme:** 2025-11-15
**Versiyon:** 1.0.0

---

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Desteklenen Exchange'ler](#desteklenen-exchangeler)
- [Kullanım](#kullanım)
- [Konfigürasyon](#konfigürasyon)
- [API Referansı](#api-referansı)
- [Geliştirme](#geliştirme)

---

## 🎯 Genel Bakış

Bu modül, farklı exchange'lere bağlanmak için tek bir interface sağlar. Tüm exchange API'leri `BaseExchangeAPI` abstract class'ını implement eder.

### Mimari

```
BaseExchangeAPI (abstract)
    └── BinanceAPI (python-binance)
```

**Tasarım Prensipleri:**
- ✅ Tek interface (BaseExchangeAPI)
- ✅ Config-driven setup (config/connectors.yaml)
- ✅ Testnet/production desteği
- ✅ Type-safe implementation
- ✅ Comprehensive logging

---

## 🏦 Desteklenen Exchange'ler

### ✅ Binance

**Durum:** Aktif
**Kütüphane:** python-binance
**Testnet:** ✅ Destekleniyor
**Özellikler:**
- Spot trading
- Futures trading
- Market data (ticker, orderbook, klines)
- Account management
- Order management

**Kullanım:**
```python
from components.exchanges import BinanceAPI
from core.config_engine import ConfigEngine

# Config yükle
config = ConfigEngine().get_config('connectors')['binance']

# API oluştur
binance = BinanceAPI(config=config)

# Market data
ticker = await binance.get_ticker('BTCUSDT')
print(f"BTC Fiyat: {ticker['lastPrice']}")

# Order book
orderbook = await binance.get_orderbook('BTCUSDT', limit=10)
print(f"En iyi bid: {orderbook['bids'][0]}")

# Trading
order = await binance.create_order(
    symbol='BTCUSDT',
    side='BUY',
    order_type='MARKET',
    quantity=0.001
)
print(f"Order ID: {order['orderId']}")
```

---

## ⚙️ Konfigürasyon

Exchange ayarları `config/connectors.yaml` dosyasında yapılandırılır.

### Binance Konfigürasyonu

```yaml
# config/connectors.yaml
binance:
  enabled: true
  testnet: true  # false for production

  # Credentials (testnet flag'ine göre seçilir)
  endpoints:
    testnet:
      api_key: "${BINANCE_TESTNET_API_KEY}"
      secret_key: "${BINANCE_TESTNET_API_SECRET}"

    production:
      api_key: "${BINANCE_API_KEY}"
      secret_key: "${BINANCE_API_SECRET}"

  # Rate limiting
  rate_limit:
    max_requests_per_minute: 1200
    weight_limit: 1200

  # Retry settings
  retry:
    enabled: true
    max_attempts: 3
    backoff_factor: 2

  # Features
  features:
    spot_trading: true
    futures_trading: true
```

### Environment Variables

`.env` dosyasına credentials ekleyin:

```bash
# Testnet (sandbox)
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret

# Production
BINANCE_API_KEY=your_production_api_key
BINANCE_API_SECRET=your_production_api_secret
```

**⚠️ Önemli:**
- Production credentials'ları asla commit etmeyin
- `.env` dosyası `.gitignore`'da olmalı
- Testnet ile önce test edin

---

## 📖 API Referansı

### BinanceAPI

#### Market Data Methods

##### `get_ticker(symbol: str) -> Dict`
Ticker fiyat bilgisi alır.

**Parametreler:**
- `symbol` (str): Trading pair (örn: "BTCUSDT")

**Döner:**
```python
{
    "symbol": "BTCUSDT",
    "lastPrice": "45000.00",
    "volume": "123456.78",
    "priceChange": "-500.00",
    "priceChangePercent": "-1.10"
}
```

**Örnek:**
```python
ticker = await binance.get_ticker('BTCUSDT')
price = float(ticker['lastPrice'])
```

---

##### `get_orderbook(symbol: str, limit: int = 100) -> Dict`
Order book verisi alır.

**Parametreler:**
- `symbol` (str): Trading pair
- `limit` (int): Depth (5, 10, 20, 50, 100, 500, 1000, 5000)

**Döner:**
```python
{
    "bids": [[price, quantity], ...],  # Alış emirleri
    "asks": [[price, quantity], ...],  # Satış emirleri
    "lastUpdateId": 123456
}
```

**Örnek:**
```python
orderbook = await binance.get_orderbook('BTCUSDT', limit=10)
best_bid = float(orderbook['bids'][0][0])
best_ask = float(orderbook['asks'][0][0])
spread = best_ask - best_bid
```

---

##### `get_klines(symbol: str, interval: str, limit: int = 100) -> List`
Kline/Candlestick verisi alır.

**Parametreler:**
- `symbol` (str): Trading pair
- `interval` (str): Interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
- `limit` (int): Kline sayısı (max 1000)
- `start_time` (int, optional): Başlangıç zamanı (timestamp ms)
- `end_time` (int, optional): Bitiş zamanı (timestamp ms)

**Döner:**
```python
[
    [
        1499040000000,  # Open time
        "0.01634000",   # Open
        "0.80000000",   # High
        "0.01575800",   # Low
        "0.01577100",   # Close
        "148976.11427815",  # Volume
        1499644799999,  # Close time
        "2434.19055334",  # Quote asset volume
        308,  # Number of trades
        "1756.87402397",  # Taker buy base asset volume
        "28.46694368",  # Taker buy quote asset volume
        "17928899.62484339"  # Ignore
    ],
    ...
]
```

**Örnek:**
```python
klines = await binance.get_klines('BTCUSDT', '1h', limit=24)
for kline in klines:
    open_price = float(kline[1])
    close_price = float(kline[4])
    print(f"Open: {open_price}, Close: {close_price}")
```

---

#### Account Methods

##### `get_balance() -> Dict`
Hesap bakiyesi alır.

**Döner:**
```python
{
    "balances": [
        {"asset": "BTC", "free": "1.5", "locked": "0.5"},
        {"asset": "USDT", "free": "10000", "locked": "2000"}
    ]
}
```

**Örnek:**
```python
account = await binance.get_balance()
for balance in account['balances']:
    if float(balance['free']) > 0:
        print(f"{balance['asset']}: {balance['free']}")
```

---

#### Trading Methods

##### `create_order(symbol, side, order_type, quantity, price=None) -> Dict`
Order oluşturur.

**Parametreler:**
- `symbol` (str): Trading pair
- `side` (str): "BUY" veya "SELL"
- `order_type` (str): "LIMIT", "MARKET", etc.
- `quantity` (float): Miktar
- `price` (float, optional): Fiyat (LIMIT için gerekli)

**Döner:**
```python
{
    "orderId": 123456,
    "symbol": "BTCUSDT",
    "status": "FILLED",
    "executedQty": "0.001",
    "price": "45000.00"
}
```

**Örnek:**
```python
# Market order
order = await binance.create_order(
    symbol='BTCUSDT',
    side='BUY',
    order_type='MARKET',
    quantity=0.001
)

# Limit order
order = await binance.create_order(
    symbol='BTCUSDT',
    side='BUY',
    order_type='LIMIT',
    quantity=0.001,
    price=44000.00
)
```

---

##### `cancel_order(symbol: str, order_id: str) -> Dict`
Order iptal eder.

**Parametreler:**
- `symbol` (str): Trading pair
- `order_id` (str): Order ID

**Örnek:**
```python
result = await binance.cancel_order('BTCUSDT', '123456')
print(f"Status: {result['status']}")  # CANCELED
```

---

##### `get_open_orders(symbol: Optional[str] = None) -> List[Dict]`
Açık order'ları alır.

**Parametreler:**
- `symbol` (str, optional): Trading pair (None ise tümü)

**Örnek:**
```python
# Tüm açık order'lar
orders = await binance.get_open_orders()

# Specific symbol
orders = await binance.get_open_orders('BTCUSDT')

for order in orders:
    print(f"Order {order['orderId']}: {order['side']} {order['quantity']}")
```

---

#### Utility Methods

##### `get_server_time() -> Dict`
Binance server zamanı alır.

**Döner:**
```python
{"serverTime": 1234567890000}
```

**Örnek:**
```python
from datetime import datetime

server_time = await binance.get_server_time()
dt = datetime.fromtimestamp(server_time['serverTime'] / 1000)
print(f"Server time: {dt}")
```

---

##### `health_check() -> bool`
API sağlığını kontrol eder.

**Döner:** `True` ise API çalışıyor

**Örnek:**
```python
if binance.health_check():
    print("✅ Binance API çalışıyor")
else:
    print("❌ Binance API bağlantı hatası")
```

---

##### `get_stats() -> Dict`
API istatistiklerini alır.

**Döner:**
```python
{
    "total_requests": 1234,
    "total_errors": 5,
    "testnet": True,
    "enabled": True
}
```

---

## 🔧 Geliştirme

### Testnet Kullanımı

**1. Binance Testnet Account Oluştur:**
- https://testnet.binance.vision/
- API key + secret al

**2. `.env` Dosyasına Ekle:**
```bash
BINANCE_TESTNET_API_KEY=your_key
BINANCE_TESTNET_API_SECRET=your_secret
```

**3. Config'de Testnet Aktif:**
```yaml
# config/connectors.yaml
binance:
  testnet: true  # ✅ Testnet aktif
```

**4. Test Et:**
```python
from components.exchanges import BinanceAPI
from core.config_engine import ConfigEngine

config = ConfigEngine().get_config('connectors')['binance']
binance = BinanceAPI(config=config)

# Test
print(f"Testnet: {binance.testnet}")  # True
print(f"API URL: {binance.client.API_URL}")  # https://testnet.binance.vision/api

# Server time test
server_time = await binance.get_server_time()
print(f"✅ Bağlantı başarılı: {server_time}")
```

---

### Production'a Geçiş

**1. Production API Keys Al:**
- https://www.binance.com/en/my/settings/api-management
- API key + secret al
- IP whitelist ekle (güvenlik)

**2. `.env` Dosyasına Ekle:**
```bash
BINANCE_API_KEY=your_production_key
BINANCE_API_SECRET=your_production_secret
```

**3. Config'de Production Aktif:**
```yaml
# config/connectors.yaml
binance:
  testnet: false  # ✅ Production aktif
```

**⚠️ UYARI:**
- Production'da gerçek para kullanılır!
- Küçük miktarlarla test edin
- Stop-loss kullanın
- API permissions'ı minimal tutun (sadece spot trading)

---

### Test Script

`components/exchanges/binance_api.py` dosyasını direkt çalıştırarak test edebilirsiniz:

```bash
python components/exchanges/binance_api.py
```

**Output:**
```
============================================================
🧪 BinanceAPI Test
============================================================

1️⃣  Config test:
   ✅ BinanceAPI oluşturuldu
   - Testnet: True
   - API URL: https://testnet.binance.vision/api
   - Enabled: True

2️⃣  Stats:
   - Total requests: 0
   - Total errors: 0

3️⃣  Health check:
   - Health: True
   - Server time: 2025-11-15 18:07:38

✅ Tüm testler tamamlandı!
============================================================
```

---

## 📚 Kaynaklar

### Binance

**Dokümantasyon:**
- [Binance Spot API](https://binance-docs.github.io/apidocs/spot/en/)
- [python-binance Library](https://python-binance.readthedocs.io/)
- [Binance Testnet](https://testnet.binance.vision/)

**Rate Limits:**
- Requests: 1,200/min
- Weight: 1,200/min
- Orders: 100/10s

**Best Practices:**
- Her request weight'e sahiptir
- Weight limiti aşmayın (ban riski)
- Rate limit header'larını kontrol edin
- Retry mekanizması kullanın

---

## 🔐 Güvenlik

### API Key Güvenliği

**DO:**
- ✅ Environment variables kullanın
- ✅ `.env` dosyasını `.gitignore`'a ekleyin
- ✅ IP whitelist kullanın
- ✅ Minimal permissions verin
- ✅ API key'leri düzenli değiştirin

**DON'T:**
- ❌ API key'leri kod içine yazmayın
- ❌ API key'leri commit etmeyin
- ❌ API key'leri public repo'lara koymayın
- ❌ Withdrawal permission vermeyin (bot için)
- ❌ Unlimited permissions vermeyin

### Permission Settings

Binance API key için önerilen permissions:
- ✅ **Enable Reading** (Balance, orders görüntüleme)
- ✅ **Enable Spot & Margin Trading** (Order oluşturma)
- ❌ **Enable Withdrawals** (ASLA!)
- ❌ **Enable Universal Transfer** (ASLA!)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. `BinanceAPIException: Invalid API-key`
**Çözüm:**
- `.env` dosyasında API key doğru mu kontrol edin
- Config dosyasında `testnet: true` ise testnet key kullanın
- Key'in aktif olduğundan emin olun

#### 2. `Timestamp for this request is outside of the recvWindow`
**Çözüm:**
- Sistem saatiniz doğru mu kontrol edin
- NTP sync kullanın
- Server time ile local time farkı 1 saniyeden fazla olmamalı

#### 3. `APIError(code=-1021): Timestamp for this request was 1000ms ahead`
**Çözüm:**
```python
# Server time ile sync
server_time = await binance.get_server_time()
local_time = int(time.time() * 1000)
time_diff = server_time['serverTime'] - local_time
print(f"Time diff: {time_diff}ms")
```

#### 4. `Rate limit exceeded`
**Çözüm:**
- Request frequency azaltın
- Weight'leri kontrol edin
- `enableRateLimit: true` kullanın (python-binance otomatik halleder)

---

## 📞 Destek

**Sorular için:**
- GitHub Issues: [SuperBot Issues](https://github.com/your-repo/issues)
- Dokümantasyon: `docs/` klasörü

**Exchange-specific:**
- Binance: https://www.binance.com/en/support

---

**Last Updated:** 2025-11-15
**Maintainer:** SuperBot Team
