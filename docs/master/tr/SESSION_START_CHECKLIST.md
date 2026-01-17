# 🚀 Session Start Checklist

> **For AI Assistants**: Her yeni session başladığında bu listeyi takip et

---

## ✅ Checklist

### 1️⃣ Quick Context Loading (5 dakika)
- [ ] `docs/claude/context_guide.md` oku
  - Kritik kuralları öğren
  - Component map'i gör
  - Quick reference al

### 2️⃣ Project Vision Understanding (10 dakika)
- [ ] `docs/claude/PROJECT_VISION.md` oku ⭐ **ÖNEMLİ**
  - Neden yapıyoruz?
  - Başarı kriterleri neler?
  - Solo developer, backtest priority
  - Replay mode, MTF, hybrid strategy

- [ ] `docs/plans/implementation_plan.md` oku
  - Ne yapıyoruz?
  - Teknoloji stack nedir?
  - Modüller nasıl çalışıyor?

- [ ] `docs/plans/rules.md` oku
  - Plan-Önce prensibi
  - Backtest-Önce prensibi
  - Modülerlik ve observability

### 3️⃣ Detailed Rules (Gerekirse)
- [ ] `docs/claude/claude_rules.md` oku
  - Emoji preservation
  - Turkish localization
  - File structure standard
  - Core engine usage
  - Component organization

---

## 🎯 Session Başlangıç Komutları

### Minimum (Hızlı başlangıç):
```
"docs/claude/context_guide.md oku ve özet ver"
```

### Tam (Kapsamlı bağlam):
```
"Session başlatıyorum. Şu dosyaları sırayla oku:
1. docs/claude/context_guide.md
2. docs/claude/PROJECT_VISION.md
3. docs/plans/implementation_plan.md
4. docs/plans/rules.md

Sonra proje hakkında kısa özet ver."
```

### Context Yenileme (Session ortasında):
```
"Context yenile - docs/claude/context_guide.md oku"
```

---

## 📊 Context Loading Seviyeleri

| Seviye | Dosyalar | Süre | Ne Zaman? |
|--------|----------|------|-----------|
| **Quick** | context_guide.md | 2 dk | Küçük değişiklikler için |
| **Standard** | context_guide + implementation_plan | 5 dk | Normal geliştirme |
| **Full** | Tüm docs | 15 dk | Büyük feature geliştirme |

---

## 🧠 Bağlam Öncelik Sırası

1. **context_guide.md** - Quick reference (ÖNCELİK 1)
2. **PROJECT_VISION.md** - Neden yapıyoruz? Başarı kriterleri (ÖNCELİK 2)
3. **implementation_plan.md** - Ne yapıyoruz? Teknoloji stack
4. **rules.md** - Geliştirme prensipleri
5. **claude_rules.md** - Detaylı kurallar
6. **system_architecture.md** - Mimari detaylar (ihtiyaç halinde)

---

## 💡 Session İçinde Hatırlatma

Eğer Claude şunları yaparsa, context yenile:

- ❌ Custom logger oluşturma (`logging.getLogger`)
- ❌ Emoji silme
- ❌ İngilizce log/exception yazma
- ❌ Var olan component'i tekrar yazma
- ❌ Layer dependency ihlali

**Komut:**
```
"Context kaybettik. docs/master/context_guide.md oku ve kuralları hatırla"
```

---

## 🎓 Öğrenme Notları

### Kritik Kurallar (Asla Unutma):
1. ✅ Her zaman `get_logger()` ve `get_config()` kullan
2. ✅ Emoji'leri koru
3. ✅ Tüm output Türkçe
4. ✅ `from __future__ import annotations` ekle
5. ✅ Yeni kod yazmadan component map kontrol et

### Yaygın Component'ler:
- **RiskManager**: Pozisyon boyutlandırma + risk kontrolü
- **OrderManager**: Order validasyon + gönderme
- **PositionManager**: Pozisyon lifecycle
- **WebSocketEngine**: WebSocket yönetimi
- **MultiTimeframeEngine**: Timeframe aggregation

---

**Version:** 1.0.0
**Last Updated:** 2025-11-14
**Maintainer:** SuperBot Team
