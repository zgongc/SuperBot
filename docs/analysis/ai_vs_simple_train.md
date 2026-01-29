# 🤖 modules/ai vs modules/simple_train - Comparison Analysis

> **Date:** 2026-01-29
> **Purpose:** Detailed comparison of two AI modules in SuperBot

---

## 1. Core Difference: Learning Paradigm

| Feature | `modules/ai` | `modules/simple_train` |
|---------|-------------|----------------------|
| **Approach** | **Reinforcement Learning** (PPO) | **Supervised Learning** (XGBoost/LightGBM/LSTM) |
| **Learning Method** | Trial-and-error, self-discovery | Learns WIN/LOSE from historical trades |
| **Framework** | Stable-Baselines3 + Gymnasium | scikit-learn / XGBoost / LightGBM / PyTorch |
| **Output** | HOLD/LONG/SHORT action + probabilities | WIN probability (0-1 binary classification) |

---

## 2. Architecture Comparison

| Component | `modules/ai` | `modules/simple_train` |
|---------|-------------|----------------------|
| **Environment** | `TradingEnv` (Gymnasium env, ~1040 LOC) | `TradeSimulator` (~18 KB, label generator) |
| **Model** | `PPOAgent` (SB3 wrapper, 576 LOC) | `EntryModel` (1219 LOC) + `ExitModel` (16 KB) |
| **Feature Count** | ~80 (hardcoded) | ~30-50 (config-driven, `features.yaml`) |
| **Feature Source** | Self-extracted (RSI, MACD, SMC, patterns) | Auto-reads from strategy + derived features |
| **Training** | `trainer.py` + curriculum learning | `EntryTrainer` (925 LOC) + iterative refinement |
| **Inference** | `RLPredictor` (1050 LOC) | `SimpleTrainPredictor` (166 LOC) |
| **API** | FastAPI server (remote GPU) | None (direct backtest integration) |
| **Config** | 3 YAML (`ai_config`, `features`, `training`) | 2 YAML (`features`, `training`) |

---

## 3. Directory Structure

### modules/ai
```
modules/ai/
├── __init__.py
├── README.md
├── docs/
│   ├── AI_MODULE_SPEC.md
│   └── AI_RL_MASTER_PLAN.md
├── configs/
│   ├── ai_config.yaml          # AI decision & inference config
│   ├── features.yaml           # Feature definitions
│   └── training.yaml           # Training parameters & reward tuning
├── core/
│   └── trading_env.py          # Gymnasium TradingEnv (~1040 LOC)
├── models/
│   └── ppo_agent.py            # PPO agent wrapper (~576 LOC)
├── training/
│   └── trainer.py              # Training pipeline & curriculum
├── inference/
│   ├── predictor.py            # Real-time RL predictor (~1050 LOC)
│   └── model_selector.py       # Symbol-specific model selection (~328 LOC)
├── api/
│   └── server.py               # FastAPI inference server
└── scripts/
    ├── prepare_data.py         # Data preparation & features (~1083 LOC)
    └── train.py                # Training CLI
```

### modules/simple_train
```
modules/simple_train/
├── DESIGN_DECISIONS.md
├── PLAN.md
├── inference.py                # SimpleTrainPredictor (~166 LOC)
├── configs/
│   ├── features.yaml           # Feature definitions & normalization
│   └── training.yaml           # Model training hyperparameters
├── core/
│   ├── feature_extractor.py    # Strategy-aware feature extraction (~653 LOC)
│   ├── normalizer.py           # 5 normalization methods (~497 LOC)
│   └── data_loader.py          # ParquetsEngine wrapper (~688 LOC)
├── models/
│   ├── entry_model.py          # Binary classification entry filter (~1219 LOC)
│   ├── exit_model.py           # Multi-output exit optimizer (~16 KB)
│   └── rich_label_generator.py # Multi-dimensional label generation (~18 KB)
├── training/
│   └── entry_trainer.py        # Complete training pipeline (~925 LOC)
├── scripts/
│   ├── prepare_data.py         # Data preparation (~39 KB)
│   ├── train.py                # CLI training script (~27 KB)
│   └── train_exit.py           # Exit model trainer (~8.9 KB)
├── backtest/
│   └── trade_simulator.py      # Independent trade simulation (~18 KB)
├── inference/                   # (empty, future use)
└── notebooks/                   # (empty, future use)
```

---

## 4. Operating Modes

### modules/ai - 2 Modes
1. **Standalone**: RL agent generates its own signals (HOLD/LONG/SHORT)
2. **Filtering**: Approves/rejects strategy signals (confidence threshold)

### modules/simple_train - 3 Modes
1. **Filter**: Strategy signal + AI → Open/Skip
2. **Direction**: AI independently predicts LONG/SHORT/HOLD
3. **Both**: Combination of filter + direction

---

## 5. Training Process

| Step | `modules/ai` | `modules/simple_train` |
|------|-------------|----------------------|
| **Data** | OHLCV → simulation in TradingEnv | OHLCV → TradeSimulator → WIN/LOSE labels |
| **Training** | PPO timesteps (1M-5M steps) | XGBoost fit (100 trees, binary classification) |
| **Curriculum** | 3 levels (increasing commission/slippage) | Iterative (Entry↔Exit mutual refinement) |
| **Evaluation** | Episode metrics (Sharpe, drawdown) | Classification (accuracy, F1) + trading metrics |
| **Duration** | Long (GPU, hours) | Short (CPU, minutes) |

---

## 6. Label Mechanism

### modules/ai (Indirect - Reward Function)
```
reward = realized_pnl (1.0)
       + unrealized_change (0.05)
       + sharpe (0.1)
       - drawdown_penalty (3.0)
       - trade_cost_penalty (0.5)
       + trend_alignment (0.3)
```
- Agent learns by exploring in its environment
- No explicit labels, only reward signals

### modules/simple_train (Direct - Explicit Labels)
```
RichLabelGenerator → WIN/LOSE + rich metadata:
  - pnl_pct          (actual profit/loss %)
  - exit_reason       (TP/SL/BE/PE/TS/TIMEOUT)
  - bars_to_exit      (trade duration)
  - max_favorable     (peak profit during trade)
  - max_adverse       (largest drawdown during trade)
  - peak_to_exit_ratio (profit left on table)
```
- TradeSimulator forward simulation generates labels

---

## 7. Feature Engineering

### modules/ai (~80 features, hardcoded)
- Timeframe features: 9 (one-hot + normalized)
- Price features: 8 (returns, candle structure)
- Indicator features: 30+ (RSI, MACD, EMA, ATR, Bollinger, CMF, OBV, VWAP)
- SMC features: 8 (Fair Value Gap, Break of Structure)
- Pattern features: 8 (Doji, Hammer, Engulfing, etc.)
- Portfolio features: 11 (position, PnL, drawdown, win rate)

### modules/simple_train (~30-50 features, config-driven)
- **Price**: EMA distance, price momentum, price volatility
- **Momentum**: RSI change, RSI slope, Stochastic RSI, CCI
- **Volume**: Volume change, volume spike, OBV change
- **Volatility**: ATR, ATR change, Bollinger Bands width
- **Trend**: ADX, EMA trend
- **Time**: Hour/day cyclical encoding, session flags
- **External Context**: Regime scores, sentiment (optional)

**Key Difference**: `simple_train` reads features from strategy config automatically, `ai` extracts its own.

---

## 8. Inference Architecture

### modules/ai - RLPredictor (1050 LOC)
```python
# Full-featured predictor with fallback chain
predictor = RLPredictor("BTCUSDT", "1h")
predictor.load_model()

result = predictor.predict(df, portfolio_state)
# → {action: "LONG", confidence: 0.72, probabilities: {...}}

# Additional capabilities:
predictor.optimize_tp_sl()           # AI-based TP/SL
predictor.should_activate_trailing() # Trailing stop decision
predictor.should_activate_break_even() # Break-even management
predictor.get_partial_exit_levels()  # Partial exit recommendations
predictor.detect_market_regime()     # Market condition classification
```

**Model Fallback Chain:**
1. Symbol-specific model → 2. Final model → 3. Generic fallback → 4. BTCUSDT fallback

### modules/simple_train - SimpleTrainPredictor (166 LOC)
```python
# Lightweight predictor for backtest integration
predictor = SimpleTrainPredictor(
    model_path="data/ai/checkpoints/simple_train/simple_rsi/model.pkl",
    strategy_name="simple_rsi"
)

probs = predictor.predict_batch(features_df)
# → {index: probability}  (e.g., {100: 0.73, 250: 0.45, ...})
```

---

## 9. Advantages & Disadvantages

| | `modules/ai` (RL) | `modules/simple_train` (Supervised) |
|--|-------------------|-------------------------------------|
| **Advantage** | Can discover new patterns | Fast training, explainable, stable |
| **Advantage** | Dynamic decisions (TP/SL/trailing) | Config-driven feature engineering |
| **Advantage** | Market regime detection | Iterative entry↔exit refinement |
| **Advantage** | Production-ready FastAPI server | Lightweight, CPU-only |
| **Disadvantage** | Slow training (GPU required) | Only learns from history, no discovery |
| **Disadvantage** | Black box, hard to explain | Exit model still in development |
| **Disadvantage** | Reward shaping is sensitive | Dependent on strategy quality |
| **Disadvantage** | Complex setup (SB3 + Gymnasium) | No live trading server yet |

---

## 10. Development Status

| Feature | `modules/ai` | `modules/simple_train` |
|---------|-------------|----------------------|
| **Entry Filter** | ✅ Complete | ✅ Complete |
| **Exit Optimization** | ✅ Framework ready (TP/SL/trailing/BE) | ⏳ ExitModel exists but not integrated |
| **Backtest Integration** | ✅ Yes | ✅ Yes |
| **Live Trading** | ✅ Ready (FastAPI server) | ❌ Not yet |
| **Multi-Timeframe** | ✅ Yes | ✅ Yes |
| **Multi-Symbol** | ✅ Model fallback chain | ⏳ Planned (symbol embedding) |
| **Config-Driven Features** | ❌ Hardcoded | ✅ features.yaml |
| **Rich Labels** | ❌ Reward-based | ✅ RichLabelGenerator |
| **Model Types** | PPO only | XGBoost / LightGBM / LSTM |
| **Normalization** | Basic (env internal) | 5 methods (minmax, zscore, rolling, percentile, log) |

---

## 11. Integration Points

### Both modules integrate via BacktestEngine:
```python
# modules/ai
from modules.ai.inference import RLPredictor
predictor = RLPredictor("BTCUSDT", "1h")
decision = predictor.should_enter(df, "LONG", portfolio_state)

# modules/simple_train
from modules.simple_train.inference import SimpleTrainPredictor
predictor = SimpleTrainPredictor(model_path, strategy_name)
probs = predictor.predict_batch(features_df)
```

---

## 12. Conclusion

**`modules/ai`** and **`modules/simple_train`** are **complementary**, not competing:

| Use Case | Recommended Module |
|----------|-------------------|
| Quick prototyping & strategy improvement | `simple_train` |
| Explainable, auditable decisions | `simple_train` |
| CPU-only environments | `simple_train` |
| Independent signal generation | `ai` |
| Dynamic trade management (TP/SL/trailing) | `ai` |
| Market regime detection | `ai` |
| Live trading with remote GPU | `ai` |
| Deep pattern discovery | `ai` |

**Potential Synergy**: Use `simple_train` for fast entry filtering + `ai` for dynamic exit management. Both can be used in a pipeline where `simple_train` handles initial signal quality and `ai` handles real-time trade optimization.

---

**Version:** 1.0.0
**Last Updated:** 2026-01-29
**Maintainer:** SuperBot Team
