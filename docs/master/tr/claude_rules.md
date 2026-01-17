# 🤖 SuperBot - Claude Geliştirme Kuralları

> **Son Güncelleme:** 2025-10-22
> **Versiyon:** 2.0.0
> **Hedef:** VS Code Claude Asistanı

---

## 🎯 PROJE GENEL BAKIŞ

**SuperBot** modüler mimariye sahip bir kripto trading botudur:

```
KATMANLAR:
  CORE        → Altyapı (logger, config, events, cache, vb.)
  COMPONENTS  → İş mantığı (indicators, strategies, data, managers)
  MODULES     → Uygulamalar (backtest, trading, ai, webui)

KURAL: Üst katmanlar alt katmanları kullanır. Alt katmanlar bağımsızdır.
```

**Mevcut Yapı:**
```
trading-bot/
├── core/                    # Altyapı katmanı
├── components/              # Yeniden kullanılabilir iş mantığı
│   ├── engines/            # Aktif engine'ler (start/stop)
│   ├── managers/           # Pasif manager'lar (CRUD)
│   ├── analysis/           # Analiz araçları
│   ├── connectors/         # Exchange bağlantıları
│   ├── data/               # Veri yönetimi
│   ├── monitoring/         # İzleme & metrikler
│   ├── notifiers/          # Bildirim sistemi
│   ├── patterns/           # Pattern algılama
│   └── strategies/         # Strateji şablonları
├── modules/                 # Uygulama katmanı
│   ├── backtest/           # Backtesting modülü
│   ├── trading/            # Canlı trading modülü
│   ├── ai/                 # AI/ML modülü
│   └── webui/              # Web dashboard
└── config/                  # Yapılandırma dosyaları
```

---

## 🚨 KRİTİK KURALLAR - ASLA İHLAL ETME

### 1. EMOJİ KORUMA 🎨

**Hiçbir dosyadan emoji'leri ASLA silme veya değiştirme!**

```python
# ❌ YANLIŞ - Emoji'leri silme
print("Veri yükleniyor...")
logger.info("Engine başlatıldı")

# ✅ DOĞRU - Emoji'leri olduğu gibi bırak
print("📂 Veri yükleniyor...")
logger.info("🚀 Engine başlatıldı")
```

**Neden:**
- Emoji'ler kasıtlı ve okunabilirliği artırır
- Windows konsol görüntüleme sorunları sadece kozmetiktir
- Kod emoji'lerle dahili olarak mükemmel çalışır
- Konsoldaki `UnicodeEncodeError` bir kod hatası DEĞİLDİR

**Eylem:** Emoji görüntüleme hatalarını yoksay, kodu DEĞİŞTİRME

---

### 2. TÜRKÇE LOKALİZASYON 🇹🇷

**TÜM çıktılar Türkçe olmalı - loglar, yorumlar, exception'lar, print'ler!**

> **📖 Tam Rehber:** Kapsamlı çeviri sözlüğü için [docs/master/localization_guide.md](localization_guide.md) dosyasına bak

#### ✅ Türkçe Olması Gerekenler:
- Tüm yorumlar, log mesajları, exception mesajları, print ifadeleri, docstring'ler, test çıktıları

#### ❌ İngilizce Kalması Gerekenler:
- Değişken/fonksiyon/sınıf/modül isimleri, import ifadeleri, dictionary key'leri, JSON alanları, API endpoint'leri

#### Hızlı Örnekler:

```python
# ✅ DOĞRU
logger.info("🚀 Engine başlatılıyor...")
logger.error(f"❌ Bağlantı hatası: {e}")
raise ValueError("Geçersiz parametre")

def calculate_risk(self, position):
    """
    Pozisyon riskini hesaplar

    Args:
        position: Pozisyon bilgisi
    Returns:
        float: Risk yüzdesi
    """
    if not position:
        raise ValueError("Pozisyon verisi boş")
    return position['size'] * position['leverage']

# ❌ YANLIŞ
logger.info("Starting engine...")
raise ValueError("Invalid parameter")
```

#### Yaygın Türkçe Terimler:
```python
"başlatılıyor/başlatıldı"     # starting/started
"durduruluyor/durduruldu"     # stopping/stopped
"başarılı/başarısız"          # successful/failed
"hata/uyarı"                  # error/warning
"yükleniyor/yüklendi"         # loading/loaded
"bağlanıyor/bağlandı"         # connecting/connected
```

---

### 3. DOSYA YAPISI STANDARDI 📄

**Her Python modülü başlık dokümantasyonu ve test bölümü içermeli!**

#### Dosya Başlığı (Zorunlu):

```python
#!/usr/bin/env python3
"""
path/to/file.py
SuperBot - Modül Adı
Yazar: SuperBot Team
Tarih: YYYY-MM-DD
Versiyon: X.Y.Z

Modül açıklaması (kısa ve öz)

Özellikler:
- Özellik 1
- Özellik 2

Kullanım:
    from module import Class
    instance = Class()

Bağımlılıklar:
    - python>=3.10
    - package1>=1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Doğrudan çalıştırma için proje kökünü path'e ekle
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
```

#### Dosya Sonu (Kütüphaneler için zorunlu):

```python
# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ModuleName Test")
    print("=" * 60)

    print("Test 1:")
    # Test kodu buraya
    print("   ✅ Test başarılı")

    print("\n✅ Tüm testler tamamlandı!")
    print("=" * 60)
```

**Testleri Nasıl Çalıştırırsın:**

```bash
# ✅ HER İKİ YÖNTEM DE ÇALIŞIR (başlıktaki sys.path işleme sayesinde)
python -m core.logger_engine              # Modül sözdizimi
python .\core\logger_engine.py            # Doğrudan dosya çalıştırma

python -m components.managers.risk_manager
python .\components\managers\risk_manager.py
```

**Neden ikisi de çalışır?**
- Başlık şablonu `__name__ == "__main__"` olduğunda `sys.path.insert(0, project_root)` ekler
- Modül sözdizimi (`-m`) otomatik olarak proje kökünü PYTHONPATH'e ekler
- Doğrudan dosya çalıştırma başlık şablonundaki sys.path'i kullanır

**Referans:** Mükemmel örnek için `core/event_bus.py` dosyasına bak

---

## 📋 KODLAMA STANDARTLARI

### Python En İyi Uygulamaları:

```python
# ✅ Her dosyanın başına ekle (Python 3.7+)
from __future__ import annotations

# Bu şunları sağlar:
# - Forward references (henüz tanımlanmamış sınıflara referans)
# - Type hints runtime'da evaluate edilmez (performans)
# - Circular import sorunlarını önler
```

### Loglama Standartları:

```python
# ✅ DOĞRU - Türkçe + Emoji + Bağlam
logger.debug(f"🔍 Debug: {variable}")
logger.info(f"📊 İstatistik güncellendi: {count} kayıt")
logger.warning(f"⚠️  Uyarı: {message}")
logger.error(f"❌ Hata: {error_message}")
logger.critical(f"🚨 Kritik: {critical_issue}")
```

### Yaygın Emoji'ler:
- ✅ Başarılı | ❌ Başarısız | ⚠️ Uyarı | 🔍 Debug
- 📊 İstatistik | 🚀 Başlatma | 🛑 Durdurma | 🔄 Yeniden başlatma
- 💾 Veri kaydı | 🌐 Network | 🔐 Güvenlik | 💰 Para

---

## 🏗️ MİMARİ KILAVUZLAR

### Sistem Mimarisi Referansı:

**KRİTİK:** HERHANGİ bir kod yazmadan önce, `system_architecture.md` dosyasını oku ve şunları anla:
- Proje yapısı (core/components/modules)
- Component sorumlulukları
- Bağımlılık ilişkileri

### Katman Bağımlılık Kuralları:

```
✅ İZİN VERİLEN:
  MODULES     → COMPONENTS → CORE
  COMPONENTS  → CORE
  MODULES     → CORE

❌ İZİN VERİLMEYEN:
  CORE        → COMPONENTS
  CORE        → MODULES
  COMPONENTS  → MODULES
```

### 🔥 KRİTİK: Her Zaman Core Engine'leri Kullan

**ASLA özel logger veya config instance'ı oluşturma!**

```python
# ✅ DOĞRU - Core engine fonksiyonlarını kullan (singleton pattern)
from core.logger_engine import get_logger
from core.config_engine import get_config

logger = get_logger("components.managers.risk_manager")  # İsimli logger
config = get_config()  # Singleton config instance

# ❌ YANLIŞ - Özel logger'lar oluşturma
import logging
logger = logging.getLogger(__name__)

# ❌ YANLIŞ - Özel config okuyucular oluşturma
with open('config.yaml') as f:
    config = yaml.load(f)

# ❌ YANLIŞ - Doğrudan instance oluşturma
from core.logger_engine import LoggerEngine
logger = LoggerEngine()  # Her seferinde yeni instance
```

**Neden:**
- Singleton pattern - Aynı instance kullanılır (bellek verimli)
- İsimli logger'lar - Hangi modülden geldiği belli
- Session'lar arası bağlam parçalanmasını önler
- Merkezi yapılandırmayı korur
- Tutarlı loglama formatı sağlar

**Kural:** Herhangi bir yerde logger veya config gerekirse, HER ZAMAN `core/` dan `get_logger()` ve `get_config()` kullan

### Component Organizasyonu:

```
components/
├── connectors/       # Exchange API bağlantıları
├── data/            # Veri yönetimi
│   ├── websocket_engine.py
│   ├── multi_timeframe_engine.py
│   ├── data_downloader.py
│   └── historical_data_manager.py
├── managers/        # İş mantığı manager'ları
│   ├── account_manager.py
│   ├── risk_manager.py
│   ├── order_manager.py
│   ├── position_manager.py
│   ├── portfolio_manager.py
│   └── strategy_executor.py
├── indicators/      # Teknik indikatörler
└── strategies/      # Strateji şablonları
```

### KRİTİK: Component Sorumlulukları

**Kod yazmadan ÖNCE hangi component'in ne yaptığını kontrol et:**

| Component | Sorumluluk |
|-----------|------------|
| **BinanceClient** | API bağlantısı, order gönderme, balance sorgulama |
| **WebSocketEngine** | WebSocket bağlantı yönetimi, auto-reconnect |
| **MultiTimeframeEngine** | 1m → 5m, 15m, 1h aggregation |
| **DataDownloader** | Gerçek zamanlı veri orkestrasyon |
| **HistoricalDataManager** | Parquet veri yükleme |
| **AccountManager** | Bakiye, kaldıraç, margin yönetimi |
| **RiskManager** | Risk kontrolleri + pozisyon boyutlandırma hesaplama |
| **OrderManager** | Order validasyonu + gönderme |
| **PositionManager** | Pozisyon yaşam döngüsü yönetimi |
| **PortfolioManager** | Performans metrikleri, kazanma oranı, PnL, Sharpe |
| **StrategyExecutor** | Giriş/çıkış sinyal üretimi |
| **IndicatorEngine** | Teknik indikatör hesaplamaları |

### ⚠️ KAÇINILMASI GEREKEN YAYGIN HATALAR:

1. **Mevcut olanları kontrol etmeden yeni component'ler oluşturma**
   - ❌ indicators/momentum/'da varken RSI fonksiyonu yazma
   - ❌ OrderManager varken OrderExecutor oluşturma
   - ❌ RiskManager'da varken pozisyon boyutlandırma mantığı yazma

2. **Fonksiyonaliteyi çoğaltma**
   - Herhangi bir şey yazmadan önce `components/` kontrol et

3. **Bağımlılık kurallarını ihlal etme**
   - Core component'ler ASLA components/'tan import etmez
   - Components ASLA modules/'tan import etmez

### İsimlendirme Kuralları:

```python
# ✅ DOĞRU
multi_timeframe_engine.py     # Aktif component (start/stop)
order_manager.py              # Pasif component (CRUD)
correlation_analyzer.py       # Analiz aracı
binance_client.py            # Connector

# ❌ YANLIŞ
multi_timeframe_manager.py    # Manager ama engine gibi davranıyor
order_engine.py               # Engine ama manager gibi davranıyor
```

### Kod Yazmadan Önce Kontrol Listesi:

- [ ] system_architecture.md oku
- [ ] Component zaten var mı kontrol et
- [ ] Doğru component konumunu doğrula
- [ ] Bağımlılık kurallarını onayla
- [ ] Component sorumluluk tablosunu kontrol et
- [ ] Çoğaltma olmadığından emin ol

---

## 📝 SON NOTLAR

### Önemli Hatırlatmalar:

1. **Emoji'ler asla silinmez** - Display hatası görmezden gelinir
2. **Tüm çıktılar Türkçe** - Kod İngilizce olabilir ama çıktılar Türkçe
3. **Dosya yapısı standart** - Header + body + test section
4. **Layer bağımlılıkları** - Sadece yukarıdan aşağıya
5. **Naming conventions** - Engine, Manager, Analyzer farkı önemli

### Kod İnceleme Ret Kriterleri:

❌ PR reddedilir:
- İngilizce log/comment/exception içeriyorsa
- Emoji silinmişse
- Header/footer eksikse
- Layer dependency ihlali varsa

✅ PR onaylanır:
- Tüm kurallar uygulanmışsa
- Test section varsa
- Component responsibilities doğru

---

**Son Güncelleme:** 2025-10-22
**Versiyon:** 2.0.0
**Bakımcı:** SuperBot Team

**Bu rehber tüm geliştiriciler ve AI asistanları tarafından takip edilmelidir.**
