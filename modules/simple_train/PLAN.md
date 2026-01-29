# Simple Train Modülü - Tasarım Planı

## Amaç
Mevcut başarılı stratejinin (simple_rsi_ai) trade pattern'larını öğrenen,
Supervised Learning tabanlı basit ve etkili bir training pipeline.

---

## Faz 1: Veri Toplama & Analiz

### 1.1 Trade History Extraction
```
Input:  Strateji backtest sonuçları
Output: trades.parquet (tüm trade'ler ve özellikleri)

Kolonlar:
- entry_time, exit_time, duration_minutes
- entry_price, exit_price
- side (LONG/SHORT)
- pnl_pct, pnl_usd
- exit_reason (TP/SL/TRAILING/BE)
- entry_features (73 feature o anki değerleri)
```

### 1.2 Feature Snapshot
```
Her trade entry anında tüm feature değerlerini kaydet:
- rsi_14, ema_50, atr_14, vs.
- Session bilgileri (is_london, hour_sin, vs.)
- Volume indikatörleri
```

### 1.3 Analiz Scripti
```python
# analyze_trades.py
- Kârlı vs zararlı trade karşılaştırması
- Feature distribution analizi
- Correlation matrix
- Win rate by feature range
```

---

## Faz 2: Feature Importance

### 2.1 İlk Analiz
```
- Pearson correlation (feature vs pnl)
- Random Forest feature importance
- SHAP values (model-agnostic)
```

### 2.2 Feature Selection
```
- Gereksiz feature'ları çıkar (correlation > 0.95)
- Düşük importance feature'ları çıkar
- Final feature set: ~20-30 feature
```

---

## Faz 3: Supervised Learning

### 3.1 Model Tipi
```
Binary Classification:
- Input: Selected features (20-30)
- Output: Win probability (0-1)

Model Options:
- XGBoost (recommended - fast, interpretable)
- LightGBM (alternative)
- Simple Neural Network (2-3 layer MLP)
```

### 3.2 Training Pipeline
```python
# train.py
1. Load trade history + features
2. Split: Train (70%) / Val (15%) / Test (15%)
3. Train model with cross-validation
4. Evaluate on test set
5. Save model + feature list
```

### 3.3 Hyperparameter Tuning
```
- Optuna ile otomatik tuning
- Target metric: ROC-AUC veya Profit Factor
```

---

## Faz 4: Integration

### 4.1 Predictor Module
```python
# predictor.py
class SimplePredictor:
    def predict(self, features: dict) -> float:
        """Return win probability 0-1"""

    def should_trade(self, features: dict, threshold: float = 0.6) -> bool:
        """Trade açılmalı mı?"""
```

### 4.2 Strategy Integration
```python
# simple_rsi_ai.py içinde
if simple_predictor.should_trade(features, threshold=0.6):
    open_position()
else:
    skip_signal()
```

---

## Dosya Yapısı

```
modules/simple_train/
├── PLAN.md                 # Bu dosya
├── configs/
│   └── training.yaml       # Hyperparameters
│
├── scripts/
│   ├── extract_trades.py   # Faz 1.1 - Trade history çıkar
│   ├── analyze_trades.py   # Faz 1.3 - Analiz ve görselleştirme
│   ├── feature_importance.py # Faz 2 - Feature selection
│   └── train.py            # Faz 3 - Model training
│
├── core/
│   ├── data_loader.py      # Data loading utilities
│   ├── feature_selector.py # Feature selection logic
│   └── model.py            # Model wrapper (XGBoost/LightGBM)
│
├── inference/
│   └── predictor.py        # Faz 4 - Production predictor
│
└── notebooks/              # Jupyter analiz notebooks
    └── trade_analysis.ipynb
```

---

## Başlangıç Adımları

1. [ ] `extract_trades.py` - Backtest'ten trade history çıkar
2. [ ] `analyze_trades.py` - Temel istatistikler ve görselleştirme
3. [ ] Feature importance analizi
4. [ ] XGBoost model training
5. [ ] Predictor integration

---

## Beklenen Çıktılar

### Trade History Format (parquet)
| Column | Type | Description |
|--------|------|-------------|
| trade_id | int | Unique trade ID |
| entry_time | datetime | Entry timestamp |
| exit_time | datetime | Exit timestamp |
| side | str | LONG/SHORT |
| pnl_pct | float | Profit/Loss % |
| win | bool | pnl_pct > 0 |
| rsi_14 | float | RSI at entry |
| ema_50 | float | EMA at entry |
| ... | ... | All 73 features |

### Model Output
```
models/simple_train/
├── model.joblib          # Trained XGBoost model
├── feature_list.json     # Selected features
├── scaler.joblib         # Feature scaler (if needed)
└── metrics.json          # Training metrics
```

---

## Timeline Tahmini

| Faz | Açıklama | Süre |
|-----|----------|------|
| 1 | Veri toplama & analiz | 1-2 saat |
| 2 | Feature importance | 1 saat |
| 3 | Model training | 2-3 saat |
| 4 | Integration | 1 saat |
| **Total** | | **5-7 saat** |

---

## Notlar

- RL yerine Supervised Learning çünkü:
  - Daha hızlı training
  - Daha kolay debug
  - Feature importance görülebilir
  - Strateji zaten çalışıyor, sadece filtre gerekli

- XGBoost tercih çünkü:
  - Tabular data'da SOTA performans
  - Hızlı inference
  - Built-in feature importance
  - Overfitting'e dayanıklı
