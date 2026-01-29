# AI Training - Tasarım Kararları

## Soru-Cevap Özeti (1-12)

### Soru 1: Öğrenme Yaklaşımı
**Karar: C + Daha Fazlası (Hybrid+)**
- Stratejiyi taklit ET ama NEDEN'i öğren
- Alternatif aksiyonları test et (LONG yerine HOLD/SHORT daha iyi miydi?)
- Kar/zarar feedback'i ile öğren
- Aynı trade'i farklı senaryolarla binlerce kez test edebilmeli

### Soru 2: Multi-Asset Öğrenme
**Karar: C (Hybrid)**
- Base features normalized (sembol-agnostik)
- Opsiyonel sembol embedding (fine-tune için)
- Tek model, farklı sembollerle eğitilebilir
- Fine-tuning ile sembole özel adaptasyon

### Soru 3: Market Context (External Data)
**Karar: Extra Feature + Modüler Yapı**
- CRYPTOCAP:TOTAL, USDT.D, BTC.D gibi market-wide veriler
- Veri yoksa → Model normal çalışır
- Veri varsa → Extra sinyal olarak kullanır
- Regime modülü ayrı API olarak çalışabilir
- USDT RSI vs Sembol RSI ters korelasyon kullanılabilir

### Soru 4: External Context Format
**Karar: D (Hibrit - Vektör + Override)**
```python
external_context = {
    "regime_score": 0.7,      # USDT.D, BTC.D bazlı
    "sentiment_score": 0.5,   # Haber/event bazlı
    "volatility_regime": 0.8, # VIX benzeri
    "correlation_score": 0.6  # BTC ile korelasyon
    # ... 1-10 arası configurable
}
```
- Config'de her parametre `enabled: true/false`
- Kritik durumda override: "NO_TRADE", "LONG_ONLY", "SHORT_ONLY"

### Soru 5: Başarı Kriteri
**Karar: Adaptasyon Odaklı**
- Taklit = Geçmişe overfit riski
- Model prensipleri öğrenmeli, ezberlememeli
- "RSI düşükken LONG" değil, "momentum tersine dönüşünde entry"
- Dinamik eşikler, market koşuluna göre kaymalı

### Soru 6: Adaptasyon Mekanizması
**Karar: A + C (Online Learning + Meta-Learning)**
- **Online Learning**: Sürekli yeni verilerle güncelleme
- **Meta-Learning**: "Nasıl adapte olacağını" öğren
- Yeni market koşulu = birkaç örnek ile hızlı adaptasyon
- Catastrophic forgetting'e dikkat

### Soru 7: Model Çıktı Formatı
**Karar: Regression + Dynamic Exit**
- Model sadece "aç/kapat" değil, **potansiyel kazancı tahmin etsin**
- `predicted_pnl > threshold` → Trade aç
- `predicted_pnl > current_pnl` → Devam et
- TP %6 sabit değil, model "bu trade %12'ye gidebilir" diyebilmeli

### Soru 8: Dynamic Exit Management
**Karar: D (Hybrid - Entry Plan + Continuous Monitoring)**
```python
# Entry'de başlangıç exit planı
initial_plan = model.predict_exit_plan(entry_features)
# {tp_percent, be_trigger, partial_exits, trailing_start}

# Pozisyondayken plan güncelleme
while in_position:
    adjustment = model.should_adjust(current_state, initial_plan)
    # "KEEP_PLAN" / "TIGHTEN_BE" / "EXTEND_TP" / "EARLY_EXIT"
```
- Config'den exit profilleri seçilebilir (aggressive, balanced, conservative)
- Model her trade için dinamik parametreler önerebilir

### Soru 9: Model Mimarisi
**Karar: B (Modular - Ayrı Modeller)**
```yaml
models:
  entry_model:
    enabled: true
    type: "xgboost"  # veya "lstm", "transformer"
  exit_model:
    enabled: true
    type: "lstm"
  regime_model:
    enabled: false   # opsiyonel
```
- Her modül bağımsız test/güncelleme yapılabilir
- Config'den açılıp kapatılabilir

### Soru 10: Modüler Eğitim Stratejisi
**Karar: C (Iterative Refinement)**
```
Round 1: Entry v1 eğit → Exit v1 eğit
Round 2: Exit v1 ile Entry v2 eğit → Entry v2 ile Exit v2 eğit
Round 3: ... (convergence'a kadar)
```
```yaml
training:
  strategy: "iterative"
  iterative:
    max_rounds: 3
    convergence_threshold: 0.01  # PnL değişimi < %1 ise dur
    freeze_after_convergence: true
```
- Entry ve Exit birbirini iteratif olarak iyileştirir
- Max 2-3 round genelde yeterli

### Soru 11: Feature Set
**Karar: B (Strateji + Config-driven Ekstralar)**
- Strateji indikatörleri (ema_50, rsi_14) zorunlu
- Ek feature'lar `modules/simple_train/configs/features.yaml` dosyasından
- Her feature `enabled: true/false` ile açılıp kapatılabilir
- Mevcut `components/indicators` kullanılacak, yeni indikatör yazılmayacak

```yaml
# features.yaml örneği
features:
  # Strateji zorunlu (stratejiden otomatik alınır)
  strategy_indicators: auto  # ema_50, rsi_14

  # Opsiyonel ekstralar
  volume:
    volume_change:
      enabled: true
      window: 5
  volatility:
    atr_14:
      enabled: true
  momentum:
    price_momentum:
      enabled: true
      window: 10
```

### Soru 12: Feature Normalization
**Karar: D (Hybrid - Feature-Type Based + Rolling Window)**
```yaml
normalizers:
  rsi_14:
    method: "minmax"
    min: 0
    max: 100
  ema_50_distance:
    method: "rolling_zscore"
    window: 100
  atr_14:
    method: "rolling_percentile"
    window: 200
  volume_change:
    method: "log_zscore"
```
- Her feature tipine uygun normalization
- Rolling window ile dinamik adaptasyon
- Config'den değiştirilebilir

### Soru 13: Lookback/Forecast Window
**Karar: Config-driven**
```yaml
model:
  lookback_window: 100    # Model kaç bar geriye bakacak (5m'de ~8 saat)
  forecast_horizon: null  # null = trade kapanana kadar bekle
  labeling:
    method: "trade_result"  # trade_result, price_change, bucketed
```
- Lookback: Model'e verilen geçmiş bar sayısı
- Forecast: Label için bakılan gelecek bar sayısı
- `trade_result` metodu: Trade TP/SL/Timeout'a kadar bekler

### Soru 14: Model Çalışma Modu
**Karar: Config-driven Filter/Direction**
```yaml
entry_model:
  mode: "filter"           # filter, direction, both
  filter:
    enabled: true
    threshold: 0.5         # Confidence > 0.5 ise trade aç
    require_match: true    # AI ve strateji aynı yönde olmalı
  direction:
    enabled: false
    # LONG / SHORT / HOLD öner
```
- **Filter Mode**: Strateji sinyali gelir → Model "aç" veya "açma" der
- **Direction Mode**: Stratejiden bağımsız yön öner
- **Both Mode**: İkisi birden

### Soru 15: Config Dosyaları
**Karar: 2 Ana Config Dosyası**
1. `configs/training.yaml` - Eğitim/model parametreleri
   - Server settings (port: 8100)
   - Device settings (auto/cuda/cpu)
   - Data settings (parquet, timeframe, symbols)
   - Environment (commission, slippage, position sizing)
   - Entry/Exit model parametreleri
   - Training strategy (iterative)
   - Evaluation metrics
   - Inference config
   - Checkpoints & logging
   - Early stopping

2. `configs/features.yaml` - Feature tanımları
   - Strategy indicators (auto)
   - Additional features (enabled/disabled)
   - Normalizers (method + params)
   - External context (optional)

---

## Mimari Özet (Güncellenmiş)

```
┌─────────────────────────────────────────────────────────────────┐
│                      REGIME MODULE (Optional)                    │
│  (USDT.D, BTC.D, Sentiment, Volatility)                         │
│  API: /regime/score                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ external_context (configurable)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ENTRY MODEL                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Base Features│  │   Symbol    │  │  External   │             │
│  │ (RSI, EMA)  │  │  Embedding  │  │  Context    │             │
│  │ normalized  │  │ (optional)  │  │  (optional) │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┴────────────────┘                     │
│                          │                                       │
│                    ┌─────▼─────┐                                │
│                    │  Predict  │                                │
│                    │ Expected  │                                │
│                    │   PnL     │                                │
│                    └─────┬─────┘                                │
│                          │                                       │
│         predicted_pnl > threshold? ──► OPEN POSITION            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼ (pozisyon açıldı)
┌─────────────────────────────────────────────────────────────────┐
│                        EXIT MODEL                                │
│                                                                  │
│  Entry'de:  predict_exit_plan(features)                         │
│             → {tp%, be_trigger, partial_exits, trailing}        │
│                                                                  │
│  Her bar:   should_adjust(current_state, plan)                  │
│             → KEEP_PLAN / TIGHTEN_BE / EXTEND_TP / EARLY_EXIT   │
│                                                                  │
│  Config'den seçilebilir profiller:                              │
│  - aggressive:   TP %10, BE yok, trailing %8'de                 │
│  - balanced:     TP %6, BE %2'de, partial %4'te                 │
│  - conservative: TP %4, BE %1'de, hızlı çık                     │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTATION ENGINE                             │
│  - Online Learning (continuous update)                           │
│  - Meta-Learning (learn to adapt quickly)                        │
│  - Iterative Training: Entry ↔ Exit refine each other           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Eğitim Akışı (Iterative)

```
┌──────────────────────────────────────────────────────────────┐
│ ROUND 1                                                       │
│   Entry v1: Strateji sinyalleriyle eğit                      │
│   Exit v1:  Entry v1'in trade'leriyle eğit                   │
│   → Baseline PnL hesapla                                      │
├──────────────────────────────────────────────────────────────┤
│ ROUND 2                                                       │
│   Entry v2: Exit v1'in feedback'iyle güncelle                │
│             (iyi çıkış yapılan entry'ler ödüllendirilir)     │
│   Exit v2:  Entry v2'nin trade'leriyle güncelle              │
│   → PnL değişimi < threshold? STOP : CONTINUE                │
├──────────────────────────────────────────────────────────────┤
│ ROUND 3 (gerekirse)                                          │
│   ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Yapılacaklar (Soru 1-15)

### Faz 1: Temel Altyapı ✅
- [x] `configs/features.yaml` - Feature tanımları
- [x] `configs/training.yaml` - Eğitim parametreleri
- [x] `core/feature_extractor.py` - Strateji + config'den feature çıkarma
- [x] `core/normalizer.py` - Rolling window normalization
- [x] `core/data_loader.py` - Parquet data loader (ParquetsEngine wrapper)

### Faz 2: Entry Model ✅
- [x] `models/entry_model.py` - Entry model mimarisi (XGBoost/LightGBM/LSTM)
- [x] Label generator (trade_result method)
- [x] `training/entry_trainer.py` - Training pipeline
- [x] `scripts/train.py` - CLI training script
- [x] Strateji sinyalleriyle ilk eğitim (198 sinyal, %49.5 baseline)

### Faz 2.5: Rich Labels (Multi-Output Infrastructure) ✅
- [x] `models/rich_label_generator.py` - Zengin label generator
- [x] Rich label yapısı: `pnl_pct`, `exit_reason`, `max_favorable`, `max_adverse`, `bars_to_exit`, `peak_to_exit_ratio`
- [x] `prepare_data.py --multi` - Multi-output mode data preparation
- [x] `train.py --multi` - Multi-output mode training support
- [x] Parquet'te rich labels kaydediliyor (future için hazır)
- [x] Backward compatible - binary classification çalışıyor
- [x] MTF (Multi-TimeFrame) + Multi-output support
- [x] Feature filtering (rich labels array'e karışmıyor)
- [x] Klasör yapısı: `data/ai/prepared/simple_train/{symbol}/{strategy}/multi/`

**Kullanım:**
```bash
# Data hazırlama
python -m modules.simple_train.scripts.prepare_data --multi --symbol BTCUSDT --start 2024

# Model eğitimi
python -m modules.simple_train.scripts.train --from-prepared --multi --symbol BTCUSDT --years 2024

# MTF + Multi
python -m modules.simple_train.scripts.train --from-prepared --multi --timeframe 5m,15m --years 2024
```

**Not:** Rich labels şu an parquet'te mevcut ama model tarafından kullanılmıyor. Gelecekte:
- Feature engineering (signal_quality = max_favorable / max_adverse)
- Multi-output regression (expected_pnl, expected_duration prediction)
- Dynamic TP/SL optimization

### Faz 3: Exit Model
- [ ] `models/exit_model.py` - Exit model mimarisi
- [ ] Dynamic exit parameter prediction (TP%, BE, trailing)
- [ ] Exit profilleri (aggressive, balanced, conservative)
- [ ] Continuous monitoring (should_adjust)

### Faz 4: Eğitim Pipeline
- [ ] `training/trainer.py` - Ana eğitim sınıfı
- [ ] Iterative training loop (Entry ↔ Exit)
- [ ] Convergence detection
- [ ] Evaluation & metrics

### Faz 5: Adaptasyon (Gelecek)
- [ ] Online learning altyapısı
- [ ] Meta-learning araştırması (MAML, Reptile)
- [ ] Catastrophic forgetting önleme

### Faz 6: External Context (Opsiyonel)
- [ ] Regime module API
- [ ] USDT.D, BTC.D entegrasyonu
- [ ] Config-driven external parameters
- [ ] Symbol embedding

---

## Dosya Yapısı (Planlanan)

```
modules/simple_train/
├── __init__.py
├── DESIGN_DECISIONS.md        # Bu dosya
├── configs/
│   ├── training.yaml          ✅ Oluşturuldu
│   └── features.yaml          ✅ Oluşturuldu
├── core/
│   ├── __init__.py            ✅ Oluşturuldu
│   ├── feature_extractor.py   ✅ Oluşturuldu
│   ├── normalizer.py          ✅ Oluşturuldu
│   └── data_loader.py         ✅ Oluşturuldu
├── models/
│   ├── __init__.py            ✅ Oluşturuldu
│   ├── entry_model.py         ✅ Oluşturuldu
│   ├── rich_label_generator.py ✅ Oluşturuldu (2026-01-15)
│   └── exit_model.py          ⏳ Sırada
├── training/
│   ├── __init__.py            ✅ Oluşturuldu
│   ├── entry_trainer.py       ✅ Oluşturuldu
│   └── evaluator.py           ⏳ Sırada
├── scripts/
│   ├── __init__.py            ✅ Oluşturuldu
│   ├── prepare_data.py        ✅ Oluşturuldu (--multi support)
│   └── train.py               ✅ Oluşturuldu (--multi support)
└── api/
    ├── __init__.py
    └── server.py              # FastAPI server (port 8100)
```

---

*Son güncelleme: 2026-01-15 - Faz 2.5 tamamlandı: Rich Labels (Multi-output infrastructure) eklendi*
