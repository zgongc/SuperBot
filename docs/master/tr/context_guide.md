# 🧠 SuperBot - Context Management Guide

> **Amaç:** Her session'da Claude'un proje bağlamını hızlıca yakalaması için

---

## 🚀 Session Başlangıcı (Her Yeni Konuşmada)

### 1. Bu Dosyayı Oku
```
docs/claude/context_guide.md  (bu dosya - quick reference)
```

### 2. 🔥 Daemon Architecture (YENİ - ÖNCE OKU!)
```
docs/claude/session-start-guide.md        # ⚡ HIZLI BAŞLANGIÇ (5 dakika)
docs/claude/daemon-architecture-guide.md  # 📚 DETAYLI REHBER (tüm architecture)
```

**KRİTİK:** Daemon architecture bilmeden kod yazma!

### 3. Proje Vizyonunu Anla
```
docs/claude/PROJECT_VISION.md      # NEDEN yapıyoruz? Başarı kriterleri
docs/plans/implementation_plan.md  # NE yapıyoruz? Teknoloji stack
docs/plans/rules.md                # NASIL yapıyoruz? Geliştirme prensipleri
```

### 4. Detaylı Kuralları Öğren
```
docs/claude/claude_rules.md        # Claude için detaylı kurallar (329 satır)
docs/master/system_architecture.md # Mimari detaylar (eğer varsa)
```

---

## 🎯 Proje Vizyonu

> **📖 Tam vizyon için:** `docs/claude/PROJECT_VISION.md` oku

### Ne Yapıyoruz?
**SuperBot**: AI destekli, multi-exchange crypto trading platformu

### Neden?
- Crypto future trading için profesyonel bot
- Solo geliştirici + 1-2 arkadaş kullanımı
- **Başarı kriteri:** Live trading'de kar

### Öncelik: Backtest Module (CRITICAL)
> "İlk Backtest biterse projenin çoğu biter"
- Strategy aynı kod: backtest + trading + optimization + AI

### Özel Özellikler:
- **Replay Mode**: TradingView-like canlı izleme
- **Multi-Timeframe (MTF)**: Cross-timeframe signals
- **Hybrid Strategy**: AI + Classical TA
- **Config-driven**: Memory/SQLite (dev) → Redis/PostgreSQL (prod)

### Temel Prensipler:
1. **Plan-Önce**: Yeni geliştirme öncesi plan güncelle
2. **Backtest-Önce**: Stratejiler önce backtest'ten geçmeli
3. **Modülerlik**: Core/components paylaşılan, modüller gevşek bağlı
4. **Observability**: Logging ve metrikler ilk günden

---

## 📋 Proje Quick Reference

### Mimari Katmanlar:
```
CORE (altyapı)
  ↑
COMPONENTS (business logic)
  ↑
MODULES (uygulamalar)
```

**Kural:** Sadece yukarıdan aşağıya import!

### 🔥 Kritik Hatırlatmalar:

#### 1. Logger & Config
```python
# ✅ HER ZAMAN
from core.logger_engine import get_logger
from core.config_engine import get_config

logger = get_logger("components.managers.risk_manager")
config = get_config()

# ❌ ASLA
import logging
logger = logging.getLogger(__name__)
```

#### 2. Emoji Preservation
```python
# ✅ Emoji'leri ASLA silme
logger.info("🚀 Engine başlatılıyor...")

# ❌ Console'da garbled görünse bile silme!
```

#### 4. File Structure
```python
#!/usr/bin/env python3
"""
path/to/file.py
SuperBot - Module Name
...docstring...
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent  # Adjust depth
    sys.path.insert(0, str(project_root))

# ... kod ...

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Test...")
```

---

## 🗺️ Component Responsibility Map

**Yeni kod yazmadan ÖNCE kontrol et:**

| Component | Ne Yapar? |
|-----------|-----------|
| **BinanceClient** | API bağlantısı, order gönderme |
| **WebSocketEngine** | WebSocket yönetimi, auto-reconnect |
| **MultiTimeframeEngine** | 1m → 5m, 15m, 1h aggregation |
| **DataDownloader** | Real-time veri orkestrasyon |
| **HistoricalDataManager** | Parquet veri yükleme |
| **AccountManager** | Balance, leverage, margin |
| **RiskManager** | Risk kontrolü + pozisyon boyutlandırma |
| **OrderManager** | Order validasyon + gönderme |
| **PositionManager** | Pozisyon lifecycle |
| **PortfolioManager** | Performance metrics (PnL, Sharpe) |
| **StrategyExecutor** | Entry/exit sinyal üretme |
| **IndicatorEngine** | Teknik indikatör hesaplama |

### ⚠️ Yaygın Hatalar:

```
❌ RSI yazmadan önce → indicators/momentum/rsi.py var mı kontrol et
❌ Position sizing logic → RiskManager'da zaten var
❌ Order execution → OrderManager kullan, yeniden yazma
```

---

## 📂 Proje Yapısı Özet

```
SuperBot/
├── core/                    # Logger, Config, EventBus, Cache, Rate Limiter
├── components/
│   ├── connectors/         # Binance, CCXT
│   ├── data/               # WebSocket, MultiTimeframe, DataDownloader
│   ├── managers/           # Account, Risk, Order, Position, Portfolio
│   ├── indicators/         # trend/, momentum/, volatility/
│   └── strategies/         # BaseStrategyTemplate, user strategies
├── modules/
│   ├── trading/           # Live/Paper/Demo/Replay
│   ├── backtest/          # Backtesting engine
│   ├── ai/                # ML models
│   └── webui/             # Flask dashboard
└── config/                # YAML configs + .env
```

---

## 🎯 Yeni Görev Başlarken Checklist

- [ ] `context_guide.md` oku (bu dosya)
- [ ] `claude_rules.md` oku
- [ ] İlgili component zaten var mı kontrol et
- [ ] Layer dependency kurallarını kontrol et
- [ ] `get_logger()` ve `get_config()` kullan
- [ ] Emoji'leri koru, Türkçe output yaz

---

## 📖 Daha Fazla Bilgi İçin

| Kategori | Dosya | Ne İçerir? |
|----------|-------|-----------|
| **⚡ Hızlı Başlangıç** | `docs/claude/session-start-guide.md` | 🔥 **İLK OKU!** Daemon architecture, async executor, event bus (5 dk) |
| **📚 Daemon Architecture** | `docs/claude/daemon-architecture-guide.md` | 🔥 **DETAYLI REHBER!** Master daemon, shared resources, IPC/RPC |
| **🌟 Vizyon & Hedefler** | `docs/claude/PROJECT_VISION.md` | Neden yapmak istiyoruz? Başarı kriterleri |
| **🎯 Master Plan** | `docs/plans/implementation_plan.md` | Teknoloji stack, modüller, roadmap |
| **📏 Prensipler** | `docs/plans/rules.md` | Genel geliştirme prensipleri, süreçler |
| **🤖 Claude Kuralları** | `docs/claude/claude_rules.md` | Detaylı geliştirme kuralları (329 satır) |
| **🏗️ Mimari** | `docs/master/system_architecture.md` | Tam mimari dokümantasyon |
| **🇹🇷 Lokalizasyon** | `docs/master/localization_guide.md` | Türkçe çeviri sözlüğü |
| **📚 Genel Bakış** | `README.md` | Proje özeti, kurulum, quick start |

---

## 💡 Context Kaybı Olursa

Eğer session uzarsa ve context kaybedilirse:

```bash
# User'a şunu söyle:
"Context yenilenmesi için lütfen şu dosyaları sırayla oku:
 1. docs/claude/context_guide.md
 2. docs/claude/session-start-guide.md
 3. docs/claude/daemon-architecture-guide.md (opsiyonel ama önerilen)"
```

## 🧠 Captain's Memory - Session Hafızası

Session'lar arası bilgi hatırlamak için SQLite tabanlı hafıza sistemi.

### Session Başında Context Al
```bash
python memory/captain_memory.py summary
```

### Kullanım (Terminal'den)
```bash
# Log ekle
python memory/captain_memory.py log "Bugün X yaptım"

# Karar kaydet
python memory/captain_memory.py decision "topic" "karar"

# Bilgi kaydet
python memory/captain_memory.py learn "topic" "öğrenilen bilgi"

# Son logları gör
python memory/captain_memory.py show

# Ara
python memory/captain_memory.py search "QML"
```

### Python'dan Kullanım
```python
from memory.captain_memory import get_memory
m = get_memory()

# Session özeti al (Claude için)
print(m.get_session_summary())

# Log ekle
m.log("QML pattern çizimi tamamlandı", category="implementation")

# Karar kaydet
m.decision("Zone Head'den başlar", topic="QML", context="SMC mantığı")

# Bilgi öğren
m.learn("BaselineSeries box çizmek için kullanılır", topic="charts")
```

---

## 🆕 Yeni Eklenenler

### 2025-12-22: Captain's Memory
- ✅ **memory/captain_memory.py** → Session hafıza sistemi
- SQLite tabanlı kalıcı hafıza
- Log, Decision, Knowledge tablolari
- CLI ve Python API

### 2025-11-26: Daemon Architecture Dökümanları
- ✅ **session-start-guide.md** → 5 dakikalık hızlı başlangıç
- ✅ **daemon-architecture-guide.md** → Tam daemon architecture rehberi

**Neden eklendi:**
- Eski session'larda daemon architecture anlaşılamamış
- Async executor pattern unutulmuş
- Exchange dosyaları silinmiş (connector_engine, connection_engine)
- "Tekerleği yeniden icat et" problem'i tekrarlamış

**Şimdi ne yapılmalı:**
- Her yeni session: `session-start-guide.md` OKU!
- Daemon ile ilgili soru: `daemon-architecture-guide.md` OKU!
- Exchange API yazarken: Async executor pattern MUTLAKA kullan!
- Session başında: `python memory/captain_memory.py summary` çalıştır!

---

**Version:** 1.2.0
**Last Updated:** 2025-12-22
**Maintainer:** SuperBot Team
