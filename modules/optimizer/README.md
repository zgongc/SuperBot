# SuperBot Optimizer

**Version**: 2.0.0
**Date**: 2025-11-17
**Status**: ✅ Production Ready

---

## 📋 Overview

Next-generation optimizer with:
- ✅ **Asynchronous backtest execution** (10-16 x parallel)
- ✅ **Stage-by-stage optimization** (automatic chaining)
- ✅ **30+ comprehensive metrics** (Sharpe, PF, SQN, etc.)
- ✅ **AI training data export** (50K+ samples)
- ✅ **Resume capability** (interrupted optimization)
- ✅ **Multi-metric ranking** (weighted score)

---

## 🚀 Quick Start

```bash
# Tek stage optimize et (exit_strategy)
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage exit_strategy \
    --trials 50

# Optimize all stages automatically
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --auto

# With custom configuration
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage risk_management \
    --trials 100 \
    --metric profit_factor \
    --parallel 16 \
    --verbose
```

---

## 🎛️ CLI Arguments (Detailed)

### Basic Arguments

| Argument | Required | Default | Description |
|---------|---------|------------|----------|
| `--strategy` | ✅ | - | Strategy file path |
| `--stage` | ⚠️ | - | Perform single-stage optimization (cannot be used with --auto) |
| `--auto` | ⚠️ | - | Automatically optimize all stages |

⚠️ One of `--stage` or `--auto` is required.

### Backtest Configuration (Optional - Read from Strategy)

| Argument | Default | Description |
|---------|------------|----------|
| `--symbol` | From strategy | Trading symbol (e.g., BTCUSDT) |
| `--timeframe` | From Strategy | Timeframe (e.g., 5m, 15m, 1h) |
| `--start` | From Strategy | Start date (YYYY-MM-DD) |
| `--end` | From strategy | End date (YYYY-MM-DD) |
| `--balance` | From strategy | Initial balance ($) |

### Optimizasyon Config

| Argument | Default | Description |
|---------|------------|----------|
| `--method` | `grid` | Optimization method: `grid`, `random`, `beam` |
| `--trials` | `100` | Maximum number of trials |
| `--metric` | `sharpe_ratio` | Metric to be optimized (see below) |
| `--parallel` | `10` | Number of parallel backtests |
| `--beam-width` | `10` | How many top results to keep at each stage |
| `--risk-free-rate` | `0.02` | Risk-free rate for Sharpe/Sortino calculation |
| `--verbose` | `false` | Detailed output mode |

---

## 📊 `--metric` Options

Optimizer determines which metric to optimize:

### 1. `sharpe_ratio` (Default) - Risk-Adjusted Return
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric sharpe_ratio
```
- **Formula**: (Return - Risk-Free Rate) / Volatility
- **Purpose**: Maximum return per unit of risk.
- **Ideal value**: > 2.0
- **For the best:** Traders who want a balanced risk/return.

### 2. `total_return` - Maximum Profit
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric total_return
```
- **Formula**: (Final Balance - Initial Balance) / Initial Balance x 100
- **Purpose**: To achieve the highest total return.
- **Warning**: There is a high drawdown risk!
- **For the best:** Aggressive traders

### 3. `profit_factor` - Profit/Loss Ratio
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric profit_factor
```
- **Formula**: Gross Profit / Gross Loss
- **Purpose**: Consistent profit
- **Ideal value**: > 1.5
- **Best for:** Traders seeking consistency.

### 4. `calmar_ratio` - Return/Drawdown
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric calmar_ratio
```
- **Formula**: Annualized Return / Max Drawdown
- **Purpose**: Best return based on drawdown.
- **Ideal value**: > 2.0
- **Best for:** Traders who want to avoid drawdown.

### 5. `sortino_ratio` - Downside Risk Adjusted
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric sortino_ratio
```
- **Formula**: (Return - Risk-Free Rate) / Downside Deviation
- **Purpose**: Only penalize the lost volatility.
- **Ideal value**: > 2.0
- **For the best:** Asymmetric strategies

### 6. `win_rate` - Win Rate
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric win_rate
```
- **Formula**: Winning Trades / Total Trades x 100
- **Purpose**: Highest winning percentage
- **Ideal value**: > 55%
- **For the best:** Scalping strategies

### 7. `sqn` - System Quality Number
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric sqn
```
- **Formula**: √N x (Avg Trade / Std Dev of Trades)
- **Purpose**: System quality measurement
- **Ideal value**: > 2.5
- **For the best:** For those seeking systemic quality.

### 8. `weighted_score` - Custom Combination
```bash
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --metric weighted_score
```
- Uses the `metric_weights` from the Strategy.
- Combines multiple metrics with weighted averaging.

---

## ⚡ `--parallel` Description

It determines how many backtests to run concurrently:

```bash
# 16 parallel backtest (fast but CPU intensive)
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --parallel 16

# 4 parallel backtests (slow but low resource)
python modules/optimizer/cli.py --strategy ... --stage exit_strategy --parallel 4
```

**Suggestions:**
- **CPU Core count**: `--parallel` <= Your CPU core count
- **RAM status**: Each backtest uses approximately 200-500MB of RAM.
- **Default 10**: Good balance for most systems.

| CPU Cores | Recommended `--parallel` |
|-----------|----------------------|
| 4 | 4 |
| 8 | 8-10 |
| 16 | 12-16 |
| 32+ | 16-20 |

---

## 🔦 `--beam-width` Description

It determines how many of the "best" results will be passed to the next stage:

```bash
# Store the top 5 results (fewer combinations, faster)
python modules/optimizer/cli.py --strategy ... --auto --beam-width 5

# Store the top 20 results (more diversity, slower)
python modules/optimizer/cli.py --strategy ... --auto --beam-width 20
```

**How it Works:**
1. Stage 1 (risk_management) is optimized.
2. The best `beam-width` result is selected.
3. Stage 2 (exit_strategy) is optimized based on these results.
4. The best `beam-width` is selected again.
5. ...continues

**Suggestions:**
- **Quick test**: `--beam-width 5`
- **Balanced**: `--beam-width 10` (default)
- **Comprehensive**: `--beam-width 20`

---

## 📋 Stage Names

`--stage` argument can use the following stages:

| Stage | Description | Optimized |
|-------|----------|-----------------|
| `main_strategy` | Main strategy | side_method, leverage |
| `risk_management` | Risk management | sizing_method, position_size |
| `exit_strategy` | Exit strategy | SL, TP, break-even, trailing |
| `indicators` | Indicators | RSI period, EMA period, etc. |
| `entry_conditions` | Entry conditions | Not yet implemented |
| `position_management` | Position management | max_positions, pyramiding |
| `market_filters` | Market filters | Not yet implemented |

---

## 💡 Example Usage Scenarios

### Scenario 1: Fast SL/TP Optimization
```bash
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage exit_strategy \
    --trials 50 \
    --metric sharpe_ratio \
    --verbose
```

### Senaryo 2: Risk Management Optimizasyonu
```bash
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage risk_management \
    --trials 30 \
    --metric profit_factor
```

### Scenario 3: Fully Automatic (All Stages)
```bash
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --auto \
    --parallel 16 \
    --beam-width 10 \
    --metric sharpe_ratio
```

### Scenario 4: Different Symbol and Timeframe
```bash
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage exit_strategy \
    --symbol ETHUSDT \
    --timeframe 15m \
    --start 2024-06-01 \
    --end 2024-12-01 \
    --trials 100
```

### Scenario 5: Focused on Low Drawdown
```bash
python modules/optimizer/cli.py \
    --strategy components/strategies/templates/simple_rsi.py \
    --stage exit_strategy \
    --metric calmar_ratio \
    --trials 100
```

---

## 📊 Features

### 1. Metrics Calculator

Calculates 30+ metrics from backtest results:

```python
from components.optimizer import MetricsCalculator

calculator = MetricsCalculator(risk_free_rate=0.02)
metrics = calculator.calculate_all_metrics(
    trades=trades,
    initial_balance=10000,
    backtest_days=365
)

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Win Rate: {metrics.win_rate:.1f}%")
print(f"Max Drawdown: {metrics.max_drawdown:.1f}%")
```

**Available Metrics**:
- **Return**: total_return, annualized_return, cagr
- **Risk**: max_drawdown, volatility, downside_deviation
- **Risk-Adjusted**: sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio
- **Trade**: total_trades, winning_trades, losing_trades, win_rate
- **Profit**: gross_profit, gross_loss, net_profit, profit_factor
- **Average**: avg_trade, avg_win, avg_loss, avg_win_loss_ratio
- **Streak**: max_consecutive_wins, max_consecutive_losses
- **Position**: avg_holding_time, max_holding_time, min_holding_time
- **Expectancy**: expectancy, expectancy_ratio
- **System Quality**: sqn (System Quality Number), kelly_criterion

---

### 2. Stage Results System (Coming Soon)

Automatically save results after each stage:

```json
{
  "run_id": "opt_20251117_143022",
  "stage": "risk_management",
  "stage_number": 1,
  "strategy_name": "TradingView_Dashboard",
  "backtest_period": {
    "start": "2024-01-01",
    "end": "2025-01-01",
    "symbol": "BTCUSDT",
    "timeframe": "30m"
  },
  "best_params": {
    "sizing_method": "RISK_BASED",
    "max_risk_per_trade": 2.0
  },
  "top_10_results": [
    {
      "rank": 1,
      "params": {"sizing_method": "RISK_BASED", "max_risk_per_trade": 2.0},
      "sharpe": 2.45,
      "return": 45.3,
      "max_drawdown": 12.0,
      "trades": 87
    }
  ]
}
```

---

### 3. AI Training Data Export ✅

Export all optimization results as AI training dataset:

```python
from components.optimizer import DataExporter

exporter = DataExporter()

# Export training data from 100 optimization runs
dataset_path = exporter.export_training_data(
    optimization_runs=["opt_*"],  # Glob pattern
    output_format="parquet",
    include_all_trials=True
)

# Output:
# 📊 Training Dataset Info:
#    Total samples: 50,000
#    Unique strategies: 10
#    Unique symbols: 5
#    Unique timeframes: 4
# ✅ Exported to: data/training/optimizer/optimizer_dataset_50000_samples_20251117.parquet

# Prepare features for ML
X, y, feature_names, categorical_features = exporter.prepare_features_for_ml(
    dataset_path=str(dataset_path),
    target_metric='metric_sharpe_ratio'
)

# Output:
# 🔧 Features prepared for ML:
#    X shape: (50000, 52)
#    y shape: (50000,)
#    Feature count: 52
#    Categorical features: 3
```

**Output**: 50,000+ training samples ready for AI model training

---

## 🎯 Optimization Workflow

### Stage-by-Stage Pipeline

```
Stage 1: Risk Management (105 trials)
  ├─ Best params saved to stage_1_risk.json
  ├─ Top 10 param sets kept
  └─ Continue to Stage 2

Stage 2: Exit Strategy (500 trials)
  ├─ Apply Stage 1 best params
  ├─ Best params saved to stage_2_exit.json
  ├─ Top 10 param sets kept
  └─ Continue to Stage 3

Stage 3: Indicators (200 trials)
  ├─ Apply Stage 1+2 best params
  ├─ Best params saved to stage_3_indicators.json
  └─ Continue...

Final: Export optimized strategy
  └─ templates/TradingView_Dashboard_optimized_20251117.py
```

**Total Backtests**: ~800 (vs 157 billion if all at once!)

---

## 📈 Optimization Metrics Guide

### Available Metrics for `--metric` Parameter

Choose the right metric based on your optimization goal:

#### 1. `sharpe_ratio` (Default) - Risk-Adjusted Return
**Use when**: You want balanced risk/return optimization
```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric sharpe_ratio
```

**Interpretation**:
- **< 1.0**: Poor risk-adjusted return
- **1.0-2.0**: Acceptable
- **2.0-3.0**: Good
- **> 3.0**: Excellent

**Best for**: Conservative traders, portfolio management, institutional trading

---

#### 2. `total_return` - Maximum Profit
**Use when**: You want highest total profit (aggressive)
```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric total_return
```

**Interpretation**:
- Returns percentage gain/loss over period
- Ignores risk and drawdown
- Can find high-profit but high-risk strategies

**Best for**: Aggressive traders, short-term optimization, high risk tolerance

⚠️ **Warning**: Can lead to overfitting and high drawdown strategies

---

#### 3. `profit_factor` - Win/Loss Ratio
**Use when**: You want consistent winning over losing trades
```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric profit_factor
```

**Interpretation**:
- **< 1.0**: Losing strategy (more losses than wins)
- **1.0-1.5**: Marginally profitable
- **1.5-2.0**: Good
- **> 2.0**: Excellent

**Formula**: Gross Profit / Gross Loss

**Best for**: Finding robust strategies with good win/loss balance

---

#### 4. `calmar_ratio` - Return/Drawdown Ratio
**Use when**: You want to minimize drawdown while maximizing return
```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric calmar_ratio
```

**Interpretation**:
- **< 1.0**: Poor
- **1.0-3.0**: Good
- **> 3.0**: Excellent

**Formula**: Annualized Return / Max Drawdown

**Best for**: Risk-averse traders, drawdown-sensitive portfolios

---

#### 5. `sortino_ratio` - Downside Risk Adjusted
**Use when**: You only care about downside volatility (not upside)
```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric sortino_ratio
```

**Interpretation**:
- Similar to Sharpe but uses downside deviation instead of total volatility
- **> 2.0**: Good
- **> 3.0**: Excellent

**Best for**: Asymmetric strategies with large upside moves

---

#### 6. `weighted_score` - Custom Multi-Metric Combination
**Use when**: You want to optimize multiple metrics simultaneously

```bash
python -m components.optimizer.cli \
    --strategy templates/TradingView_Dashboard.py \
    --auto \
    --metric weighted_score
```

**Configuration in strategy template**:
```python
self.optimizer_parameters = {
    'optimization_settings': {
        'primary_metric': 'weighted_score',

        # Define custom weights for each metric
        'metric_weights': {
            'sharpe_ratio': 0.30,      # 30% weight on risk-adjusted return
            'profit_factor': 0.25,     # 25% weight on win/loss ratio
            'total_return': 0.20,      # 20% weight on total profit
            'win_rate': 0.15,          # 15% weight on win percentage
            'max_drawdown': -0.10,     # -10% weight (minimize drawdown)
        },
    },
}
```

**Example Configurations**:

1. **Balanced Trading** (Default):
```python
'metric_weights': {
    'sharpe_ratio': 0.30,
    'profit_factor': 0.25,
    'total_return': 0.20,
    'win_rate': 0.15,
    'max_drawdown': -0.10,
}
```

2. **Aggressive Growth**:
```python
'metric_weights': {
    'total_return': 0.50,
    'profit_factor': 0.30,
    'sharpe_ratio': 0.20,
}
```

3. **Conservative/Risk-Averse**:
```python
'metric_weights': {
    'sharpe_ratio': 0.35,
    'calmar_ratio': 0.30,
    'max_drawdown': -0.25,  # Strongly penalize drawdown
    'win_rate': 0.10,
}
```

4. **High Win Rate Focus**:
```python
'metric_weights': {
    'win_rate': 0.40,
    'profit_factor': 0.30,
    'sharpe_ratio': 0.20,
    'avg_win_loss_ratio': 0.10,
}
```

**Best for**: Advanced users, multi-objective optimization, custom trading goals

---

### Quick Metric Selection Guide

| Goal | Recommended Metric | Alternative |
|------|-------------------|-------------|
| **Balanced risk/return** | `sharpe_ratio` | `sortino_ratio` |
| **Maximum profit** | `total_return` | `cagr` |
| **Minimize drawdown** | `calmar_ratio` | `max_drawdown` (minimize) |
| **Consistent wins** | `profit_factor` | `win_rate` |
| **Custom goals** | `weighted_score` | - |

---

### Real-World Comparison: Different Metrics = Different Results

Same strategy, same data, different metrics:

```bash
# Example 1: Optimize for Sharpe Ratio (balanced)
python -m components.optimizer.cli --auto --metric sharpe_ratio
# Result: sizing_method=RISK_BASED, max_risk=2.0%
# → Return: 35%, Sharpe: 2.5, MaxDD: 12%

# Example 2: Optimize for Total Return (aggressive)
python -m components.optimizer.cli --auto --metric total_return
# Result: sizing_method=FIXED_PERCENT, position_size=20%
# → Return: 65%, Sharpe: 1.2, MaxDD: 28%

# Example 3: Optimize for Calmar Ratio (low drawdown)
python -m components.optimizer.cli --auto --metric calmar_ratio
# Result: sizing_method=RISK_BASED, max_risk=1.0%
# → Return: 22%, Sharpe: 2.1, MaxDD: 7%
```

**Key Insight**: Choose metric based on your risk tolerance and goals!

---

### Metrics Interpretation Guide

#### Sharpe Ratio
- **< 1.0**: Poor
- **1.0-2.0**: Acceptable
- **2.0-3.0**: Good
- **> 3.0**: Excellent

#### Profit Factor
- **< 1.0**: Losing strategy
- **1.0-1.5**: Marginally profitable
- **1.5-2.0**: Good
- **> 2.0**: Excellent

#### System Quality Number (SQN)
- **1.6-1.9**: Below average
- **2.0-2.4**: Average
- **2.5-2.9**: Good
- **3.0-5.0**: Excellent
- **> 5.0**: Superb

#### Win Rate
- **40-50%**: Acceptable (if avg_win > avg_loss)
- **50-60%**: Good
- **> 60%**: Excellent

#### Max Drawdown
- **< 10%**: Excellent
- **10-20%**: Good
- **20-30%**: Acceptable
- **> 30%**: High risk

---

## 🔧 Configuration

### In Strategy Template

```python
self.optimizer_parameters = {
    'optimization_settings': {
        # Primary metric to optimize
        'primary_metric': 'sharpe_ratio',  # or 'profit_factor', 'total_return', etc.

        # Weighted score (if primary_metric='weighted_score')
        'metric_weights': {
            'sharpe_ratio': 0.30,
            'profit_factor': 0.25,
            'win_rate': 0.15,
            'total_return': 0.15,
            'max_drawdown': -0.15,  # Negative = minimize
        },

        # Minimum thresholds (filter bad results)
        'min_thresholds': {
            'total_trades': 20,
            'sharpe_ratio': 0.5,
            'profit_factor': 1.0,
            'win_rate': 40.0,
            'max_drawdown': 30.0,
        },
    },
}
```

---

## 📁 Directory Structure

```
components/optimizer/
├── __init__.py               # Package exports
├── README.md                 # This file
├── metrics.py                # Metrics calculator ✅
├── stage_results.py          # Stage results manager ✅
├── optimizer.py              # Main optimizer class ✅
├── data_exporter.py          # AI training data export ✅
├── cli.py                    # CLI interface ✅
└── __main__.py               # Module entry point ✅

data/optimization_results/
├── opt_20251117_143022_stage_1_risk_management.json
├── opt_20251117_143022_stage_2_exit_strategy.json
└── opt_20251117_143022_stage_3_indicators.json

data/training/optimizer/
└── optimizer_dataset_50000_samples.parquet
```

---

## 🧪 Testing

```bash
# Test metrics calculator
python -m pytest tests/test_optimizer_v2_metrics.py

# Test optimizer (coming soon)
python -m pytest tests/test_optimizer_v2.py
```

---

## 📚 API Reference

### MetricsCalculator

```python
class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """

    def calculate_all_metrics(
        self,
        trades: List[dict],
        initial_balance: float,
        backtest_days: int
    ) -> BacktestMetrics:
        """
        Calculate comprehensive metrics from trade list

        Returns:
            BacktestMetrics with 30+ metrics
        """
```

### BacktestMetrics

```python
@dataclass
class BacktestMetrics:
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float

    # Risk metrics
    max_drawdown: float
    volatility: float
    downside_deviation: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float

    # ... and 15 more metrics
```

---

## 🚀 Roadmap

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] MetricsCalculator (30+ metrics)
- [x] StageResults manager (JSON storage)
- [x] DataExporter (AI training data)
- [x] Directory structure
- [x] Documentation

### Phase 2: Optimizer Core ✅ COMPLETE
- [x] OptimizerV2 main class
- [x] Parameter generator (grid/random/beam)
- [x] Parallel backtest runner (async)
- [x] Auto stage chaining
- [x] Multi-metric ranking
- [ ] Resume capability (TODO)

### Phase 3: CLI Interface ✅ COMPLETE
- [x] CLI commands (argparse)
- [x] Strategy loading (from file)
- [x] Single stage mode
- [x] Auto mode (all stages)
- [x] Progress tracking (built-in OptimizerV2)
- [ ] Results visualization (TODO)

### Phase 4: AI Model Training
- [ ] Baseline model (Random Forest)
- [ ] Neural network model
- [ ] Model training scripts
- [ ] Prediction integration

### Phase 5: Advanced Features (Future)
- [ ] Beam search optimization
- [ ] Hyperband support
- [ ] RL-based optimizer
- [ ] Walk-forward analysis

---

## 📝 Notes

- Located in the deprecated optimizer folder (see `components/optimizer/deprecated/`).
- All new development uses Optimizer

**Production Ready Modules** ✅:
- `metrics.py` - MetricsCalculator (30+ metrics)
- `stage_results.py` - Stage results storage & loading
- `data_exporter.py` - AI training data export
- `optimizer.py` - Main optimizer class (async, parallel execution)
- `cli.py` - CLI interface (single stage & auto mode)

**Coming Soon**:
- AI model training scripts
- Walk-forward analysis
- Results visualization dashboard

---

## 📞 Support

For issues or questions:
- GitHub: https://github.com/anthropics/SuperBot/issues
- Docs: `docs/plans/optimizer_v2_plan.md`

---

**Last Updated**: 2025-11-17
**Author**: SuperBot Team
