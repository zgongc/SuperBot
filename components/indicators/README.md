# 📊 Indicator User Guide - SuperBot

**Version**: 2.1.0
**Date**: 2025-11-20
**Author**: SuperBot Team

---

## 🎯 Overview

SuperBot features a **76+ indicator** (in 9 categories) and an **automatic registry system**. This guide explains in detail how to use indicators in strategy templates.

### 📦 Indicator Kategorileri (9)

1. **Trend Indicators** (15) - SMA, EMA, MACD, SuperTrend, ADX
2. **Momentum Indicators** (11) - RSI, Stochastic, CCI, Williams %R
3. **Volatility Indicators** (8) - Bollinger Bands, ATR, Keltner Channels
4. **Volume Indicators** (9) - OBV, MFI, VWAP, CMF, Volume Profile
5. **Support/Resistance** (8) - Pivot Points, Fibonacci, Supply/Demand
6. **Combo Indicators** (5) - Ichimoku, Elder Ray, Awesome Oscillator
7. **Breakout Indicators** (5) - Donchian, Price Channel, Range Breakout
8. **Statistical Indicators** (5) - Z-Score, Correlation, Linear Regression
9. **Structure (SMC)** (6) - FVG, iFVG, BoS, CHoCH, Order Blocks, Liquidity Zones

### 📚 Contents

1. [Quick Start](#-quick-start)
2. [Strategy Template Structure](#strategy-template-yapısı)
3. [Indicator Definition](#-indicator-definition)
4. [Entry/Exit Conditions](#-entryexit-conditions)
5. [Indicator Kategorileri](#-indicator-kategorileri)
6. [Pattern Detection](#-pattern-detection)
7. [Example Strategies](#-example-strategies)
8. [Best Practices](#-best-practices)

---

## 🚀 Quick Start

### Registry Usage

```python
from components.indicators import INDICATOR_REGISTRY, get_indicator_class

# List all indicators
for name, info in INDICATOR_REGISTRY.items():
    print(f"{name}: {info['description']}")
    print(f"  - Default params: {info['default_params']}")
    print(f"  - Output keys: {info['output_keys']}")

# Get and use the Indicator class.
RSI = get_indicator_class('rsi')
rsi = RSI(period=14)
result = rsi.calculate(data)
print(result.value)  # {'rsi': 45.67} or single value
```

---

## 🏗️ Strategy Template Structure

All strategies inherit from the `BaseStrategy` class and use the following structure:

```python
from components.strategies.base_strategy import (
    BaseStrategy,
    TechnicalParameters,
    ExitStrategy,
    RiskManagement
)

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        # Strategy metadata
        self.strategy_name = "my_strategy"
        self.strategy_version = "1.0.0"

        # 1. INDICATOR TANIMLAMA
        self.technical_parameters = TechnicalParameters(
            indicators={
                "rsi": {"period": 14},
                "ema_20": {"period": 20},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            }
        )

        # 2. ENTRY CONDITIONS
        self.entry_conditions = {
            'long': [
                ['rsi', '<', 30],                    # RSI oversold
                ['ema_20', 'crossover', 'ema_50'],   # Golden cross
                ['macd_macd', '>', 'macd_signal']    # MACD bullish
            ],
            'short': [
                ['rsi', '>', 70],
                ['ema_20', 'crossunder', 'ema_50'],
                ['macd_macd', '<', 'macd_signal']
            ]
        }

        # 3. EXIT CONDITIONS
        self.exit_conditions = {
            'long': [
                ['macd_macd', 'crossunder', 'macd_signal']  # MACD turns bearish, exit
            ],
            'short': [
                ['macd_macd', 'crossover', 'macd_signal']
            ]
        }

        # 4. EXIT STRATEGY
        self.exit_strategy = ExitStrategy(
            stop_loss_percent=1.0,        # %1 stop loss
            take_profit_percent=2.0,      # %2 take profit
            trailing_stop_enabled=True,   # Trailing stop active
            trailing_callback_percent=0.5 # %0.5 trailing distance
        )
```

---

## 📊 Indicator Definition

### Temel Syntax

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        # Basit indicator (default params)
        "rsi": {"period": 14},

        # Custom name (multiple from the same indicator)
        "rsi_fast": {"period": 7},
        "rsi_slow": {"period": 21},

        # Multi-parameter indicator
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },

        # Volume indicator
        "obv": {},  # Parametresiz

        # Bollinger Bands
        "bollinger": {
            "period": 20,
            "std_dev": 2.0
        }
    }
)
```

### Indicator Naming

Indicator names are automatically formatted:

```python
# Definition
"rsi": {"period": 14}

# Created output keys
# - rsi_14 (or just 'rsi' if the default parameter is set in the registry)

# Custom isim
"rsi_fast": {"period": 7}
# Output: rsi_fast_7 (or 'rsi_fast')

# Multi-value indicator (MACD example)
"macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
# Outputs:
# - macd_macd  (main line)
# - macd_signal (signal line)
# - macd_histogram (histogram)
```

---

## 🎯 Entry/Exit Conditions

### Condition Format

```python
self.entry_conditions = {
    'long': [
        # [left_operand, operator, right_operand]
        ['rsi', '<', 30],                    # Value comparison
        ['ema_20', '>', 'ema_50'],           # Two indicator comparison
        ['ema_20', 'crossover', 'ema_50'],   # Crossover tespiti
        ['close', '>', 'bollinger_upper'],   # Price vs indicator
    ],
    'short': [...]
}
```

### Supported Operators

#### 1. Comparison Operators

| Operator | Description | Example |
|----------|----------|-------|
| `'>'` | Greater than | `['rsi', '>', 70]` |
| `'<'` | Less than | `['rsi', '<', 30]` |
| `'>='` | Greater than or equal to | `['close', '>=', 'ema_20']` |
| `'<='` | Less than or equal to | `['atr', '<=', 0.5]` |
| `'=='` | Equals | `['squeeze', '==', True]` |
| `'!='` | Not equal | `['squeeze', '!=', False]` |

#### 2. Trend and Movement Operators

| Operator | Description | Example |
|----------|----------|-------|
| `'crossover'` | Upper intersection | `['ema_20', 'crossover', 'ema_50']` |
| `'crossunder'` | Below intersection | `['ema_20', 'crossunder', 'ema_50']` |
| `'rising'` | Rising (N bar) | `['close', 'rising', 3]` |
| `'falling'` | Falling (N bar) | `['close', 'falling', 3]` |
| `'between'` | Between | `['rsi', 'between', [40, 60]]` |
| `'outside'` | Outside the range | `['rsi', 'outside', [30, 70]]` |

### Indicator Output Keys

Multi-value indicators return multiple outputs:

```python
# MACD outputs
'macd_macd'       # Main line
'macd_signal'     # Signal line
'macd_histogram'  # Histogram

# Bollinger Bands outputs
'bollinger_upper'   # Upper band
'bollinger_middle'  # Orta bant (SMA)
'bollinger_lower'   # Alt bant
'bollinger_width'   # Band width
'bollinger_percent_b'  # %B indicator

# SuperTrend outputs
'supertrend_supertrend'  # SuperTrend line
'supertrend_upper'       # Upper band
'supertrend_lower'       # Alt bant
'supertrend_trend'       # Trend direction (1=UP, -1=DOWN, 0=NEUTRAL)

# ADX outputs
'adx_adx'        # ADX value
'adx_plus_di'    # +DI
'adx_minus_di'   # -DI

# Stochastic outputs
'stochastic_k'   # %K line
'stochastic_d'   # %D line

# Pivot Points outputs
'pivot_points_P'   # Pivot
'pivot_points_R1'  # Resistance 1
'pivot_points_R2'  # Resistance 2
'pivot_points_R3'  # Resistance 3
'pivot_points_S1'  # Support 1
'pivot_points_S2'  # Support 2
'pivot_points_S3'  # Support 3
```

---

## 📁 Indicator Kategorileri

### 1. Trend Indicators (15 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        # Moving Averages
        "sma": {"period": 50},           # Simple MA
        "ema": {"period": 20},           # Exponential MA
        "wma": {"period": 30},           # Weighted MA
        "hma": {"period": 20},           # Hull MA
        "vwma": {"period": 20},          # Volume Weighted MA
        "dema": {"period": 21},          # Double EMA
        "tema": {"period": 21},          # Triple EMA

        # Trend Indicators
        "macd": {                        # MACD
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },
        "supertrend": {                  # SuperTrend
            "period": 10,
            "multiplier": 3.0
        },
        "adx": {"period": 14},           # Average Directional Index
        "aroon": {"period": 25},         # Aroon
        "donchian": {"period": 20},      # Donchian Channel
    }
)
```

**Usage Examples:**

```python
# EMA Crossover
self.entry_conditions = {
    'long': [
        ['ema_20', 'crossover', 'ema_50'],   # Golden cross
    ],
    'short': [
        ['ema_20', 'crossunder', 'ema_50'],  # Death cross
    ]
}

# SuperTrend
self.entry_conditions = {
    'long': [
        ['supertrend_trend', '==', 1],           # Bullish trend
        ['close', '>', 'supertrend_supertrend']  # Price is above SuperTrend
    ]
}

# ADX Trend Strength
self.entry_conditions = {
    'long': [
        ['adx_adx', '>', 25],              # Strong trend
        ['adx_plus_di', '>', 'adx_minus_di']  # Bullish direction
    ]
}
```

---

### 2. Momentum Indicators (11 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "rsi": {"period": 14},           # RSI
        "stochastic": {                  # Stochastic
            "k_period": 14,
            "k_smooth": 3,
            "d_smooth": 3
        },
        "stochastic_rsi": {              # Stochastic RSI
            "rsi_period": 14,
            "stoch_period": 14,
            "k_smooth": 3,
            "d_smooth": 3
        },
        "cci": {"period": 20},           # CCI
        "williams_r": {"period": 14},    # Williams %R
        "roc": {"period": 12},           # Rate of Change
        "tsi": {                         # True Strength Index
            "long_period": 25,
            "short_period": 13,
            "signal_period": 13
        },
        "mfi": {"period": 14},           # Money Flow Index
        "ultimate": {                    # Ultimate Oscillator
            "period1": 7,
            "period2": 14,
            "period3": 28
        },
        "awesome": {},                   # Awesome Oscillator (fixed params)
    }
)
```

**Usage Examples:**

```python
# RSI Oversold/Overbought
self.entry_conditions = {
    'long': [
        ['rsi', '<', 30],    # Oversold
    ],
    'short': [
        ['rsi', '>', 70],    # Overbought
    ]
}

# Stochastic Crossover
self.entry_conditions = {
    'long': [
        ['stochastic_k', 'crossover', 'stochastic_d'],  # Bullish cross
        ['stochastic_k', '<', 20],                       # Oversold zone
    ]
}

# CCI Extreme Levels
self.entry_conditions = {
    'long': [
        ['cci', '>', 100],   # Strong uptrend
    ],
    'short': [
        ['cci', '<', -100],  # Strong downtrend
    ]
}
```

---

### 3. Volatility Indicators (8 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "bollinger": {                   # Bollinger Bands
            "period": 20,
            "std_dev": 2.0
        },
        "atr": {"period": 14},           # Average True Range
        "natr": {"period": 14},          # Normalized ATR
        "keltner": {                     # Keltner Channel
            "period": 20,
            "multiplier": 2.0
        },
        "donchian": {"period": 20},      # Donchian Channel
        "chandelier": {                  # Chandelier Exit
            "period": 22,
            "multiplier": 3.0
        },
        "standard_dev": {"period": 20},  # Standard Deviation
        "squeeze": {                     # TTM Squeeze
            "bb_period": 20,
            "kc_period": 20
        },
    }
)
```

**Usage Examples:**

```python
# Bollinger Bands Breakout
self.entry_conditions = {
    'long': [
        ['close', '>', 'bollinger_upper'],     # Breakout from the upper band
        ['bollinger_width', '<', 0.02],        # Squeeze durumu
    ]
}

# ATR for Stop Loss (dynamic)
# Used in the exit strategy (automatic)
self.exit_strategy = ExitStrategy(
    stop_loss_method=StopLossMethod.ATR_BASED,
    stop_loss_atr_multiplier=2.0,  # 2x ATR stop
)

# Squeeze Momentum
self.entry_conditions = {
    'long': [
        ['squeeze_momentum_squeeze', '==', 0],      # Squeeze fired (breakout)
        ['squeeze_momentum_momentum', '>', 0],      # Bullish momentum
    ]
}
```

---

### 4. Volume Indicators (9 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "vwap": {},                      # VWAP
        "vwap_bands": {                  # VWAP Bands
            "std_multiplier": 2.0
        },
        "obv": {},                       # On-Balance Volume
        "ad": {},                        # Accumulation/Distribution
        "cmf": {"period": 20},           # Chaikin Money Flow
        "mfi": {"period": 14},           # Money Flow Index
        "force_index": {"period": 13},   # Force Index
        "eom": {"period": 14},           # Ease of Movement
        "volume_oscillator": {           # Volume Oscillator
            "fast_period": 5,
            "slow_period": 10
        },
    }
)
```

**Usage Examples:**

```python
# VWAP
self.entry_conditions = {
    'long': [
        ['close', '>', 'vwap'],   # Price is above VWAP
    ]
}

# MFI (Volume-based RSI)
self.entry_conditions = {
    'long': [
        ['mfi', '<', 20],   # Oversold with volume
    ]
}

# Volume Confirmation
self.entry_conditions = {
    'long': [
        ['ema_20', 'crossover', 'ema_50'],  # Signal
        ['volume', '>', 'volume_sma_20'],   # Volume confirmation (custom SMA eklenmeli)
    ]
}
```

---

### 5. Support/Resistance Levels (8 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "pivot_points": {},              # Standard Pivots
        "fibonacci_pivot": {},           # Fibonacci Pivots
        "camarilla": {},                 # Camarilla Pivots
        "woodie": {},                    # Woodie Pivots
        "fib_retracement": {             # Fibonacci Retracement
            "lookback": 50
        },
        "swing_points": {                # Swing High/Low
            "left_bars": 5,
            "right_bars": 5,
            "lookback": 50
        },
    }
)
```

**Output Keys:**

```python
# Pivot Points: P, R1, R2, R3, S1, S2, S3
# Fibonacci Pivot: P, R1, R2, R3, S1, S2, S3
# Camarilla: R1, R2, R3, R4, S1, S2, S3, S4
# Woodie: P, R1, R2, S1, S2
# Fib Retracement: Fib_0.0, Fib_23.6, Fib_38.2, Fib_50.0, Fib_61.8, Fib_78.6, Fib_100.0
# Swing Points: swing_high, swing_low
```

**Usage Examples:**

```python
# Pivot Point Breakout
self.entry_conditions = {
    'long': [
        ['close', '>', 'pivot_points_R1'],  # Broke through R1
        ['rsi', '>', 50],                    # Momentum indicator
    ]
}

# Fibonacci Golden Zone
self.entry_conditions = {
    'long': [
        ['close', '>', 'fib_retracement_Fib_61.8'],   # Above 61.8%
        ['close', '<', 'fib_retracement_Fib_50.0'],   # below %50
        # Golden zone: between 50-61.8%
    ]
}

# Camarilla Extreme Levels
self.entry_conditions = {
    'long': [
        ['close', '<', 'camarilla_S3'],  # S3 support (extreme oversold)
        ['rsi', '<', 30],
    ]
}
```

---

### 6. Combo Indicators (5 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "rsi_bollinger": {               # RSI + Bollinger
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std": 2.0
        },
        "stochastic_rsi": {              # Stochastic RSI
            "rsi_period": 14,
            "stoch_period": 14,
            "k_smooth": 3,
            "d_smooth": 3
        },
        "macd_rsi": {                    # MACD + RSI
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14
        },
        "ema_ribbon": {                  # EMA Ribbon
            "periods": [5, 8, 13, 21, 34, 55, 89]
        },
    }
)
```

**Usage Examples:**

```python
# RSI Bollinger
self.entry_conditions = {
    'long': [
        ['rsi_bollinger_rsi', '<', 'rsi_bollinger_bb_lower'],  # RSI oversold (Bollinger based)
    ]
}

# Stochastic RSI
self.entry_conditions = {
    'long': [
        ['stochastic_rsi_k', '<', 20],                         # Oversold
        ['stochastic_rsi_k', 'crossover', 'stochastic_rsi_d'], # Bullish cross
    ]
}

# MACD RSI Combo
self.entry_conditions = {
    'long': [
        ['macd_rsi_macd', '>', 0],      # MACD bullish
        ['macd_rsi_rsi', '>', 50],      # RSI bullish
    ]
}
```

---

### 7. Breakout Indicators (5 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "range_breakout": {              # Range Breakout
            "period": 20
        },
        "volatility_breakout": {         # Volatility Breakout
            "period": 20,
            "k": 0.5
        },
        "squeeze_momentum": {            # Squeeze Momentum
            "bb_period": 20,
            "kc_period": 20,
            "kc_mult": 1.5
        },
    }
)
```

---

### 8. Statistical Indicators (5 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "z_score": {"period": 20},       # Z-Score
        "correlation": {                 # Correlation
            "period": 20,
            "high_correlation": 0.7,
            "low_correlation": -0.7
        },
        "linear_regression": {           # Linear Regression
            "period": 14
        },
        "percentile_rank": {             # Percentile Rank
            "period": 100
        },
    }
)
```

---

### 9. Structure Indicators - SMC (Smart Money Concepts) (6 indicator)

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        # Fair Value Gap (FVG)
        "fvg": {
            "min_gap_percent": 0.05,    # Minimum gap size (%0.05)
            "max_zones": 15,            # Max active zones to track
        },

        # Inverse Fair Value Gap (iFVG)
        "ifvg": {
            "min_gap_percent": 0.05,    # Minimum gap size
            "lookback_bars": 10,        # Lookback period for reversal
        },

        # Break of Structure (BoS)
        "bos": {
            "left_bars": 5,             # Left swing bars
            "right_bars": 5,            # Right swing bars
        },

        # Change of Character (CHoCH)
        "choch": {
            "left_bars": 5,             # Left swing bars
            "right_bars": 5,            # Right swing bars
            "trend_strength": 3,        # Minimum swing count for trend
        },

        # Order Blocks
        "order_blocks": {
            "lookback_bars": 20,        # Lookback period
            "min_size_percent": 0.5,    # Minimum block size
        },

        # Liquidity Zones
        "liquidity_zones": {
            "lookback_bars": 50,        # Lookback period
            "min_touches": 2,           # Minimum touches
        },
    }
)
```

**FVG (Fair Value Gap) - Usage Examples:**

FVG detects price gaps that form between 3 candles. The `calculate_batch()` method returns the **net FVG value**:

```python
# Output Format:
# Positive value = Bullish FVG dominance (bullish_zones - bearish_zones)
# Negative value = Bearish FVG dominance
# Zero = No FVG or balanced

# ✅ CORRECT USAGE:

# Is there a bullish FVG?
self.entry_conditions = {
    'long': [
        ["fvg", ">", 0],                    # Any bullish FVG present
        ["close", ">", "ema_55", "1h"],     # Trend filter
    ]
}

# Is there a bearish FVG?
self.entry_conditions = {
    'short': [
        ["fvg", "<", 0],                    # Any bearish FVG present
        ["close", "<", "ema_55", "1h"],     # Trend filter
    ]
}

# Strong Bullish FVG (2+ net zones)
self.entry_conditions = {
    'long': [
        ["fvg", ">=", 2],                   # Strong bullish FVG
        ["rsi_14", ">", 50],
    ]
}

# Strong Bearish FVG (2+ net zones)
self.entry_conditions = {
    'short': [
        ["fvg", "<=", -2],                  # Strong bearish FVG
        ["rsi_14", "<", 50],
    ]
}

# FVG is not present or is balanced
self.entry_conditions = {
    'long': [
        ["fvg", "==", 0],                   # No FVG or equal bull/bear
        # ... other conditions
    ]
}

# ❌ INCORRECT USAGE (old format - no longer works):
["fvg", "==", 100]   # WRONG! FVG is no longer 100/-100, it returns the number of net zones.
["fvg", "==", -100]  # WRONG!
```

**BoS (Break of Structure) - Usage Examples:**

BoS detects breakouts of swing high/low levels. It uses `SwingPoints` for pivot detection (algorithm compatible with TradingView).

**Pivot Algorithm (compatible with TradingView):**
- Sol taraf: Strictly greater/less (current > left bars)
- Right side: Greater/less or equal (current >= right bars) - the first pivot to form wins.

```python
# BoS outputs: 1 (bullish BoS), -1 (bearish BoS), 0 (none)

# Basic usage
self.entry_conditions = {
    'long': [
        ["bos", "==", 1],                   # Bullish BoS on primary timeframe
        ["bos", "==", 1, "15m"],            # Bullish BoS on 15m (MTF)
    ],
    'short': [
        ["bos", "==", -1],                  # Bearish BoS
        ["bos", "==", -1, "15m"],           # Bearish BoS on 15m
    ]
}

# Empty parameters
"bos": {
    "left_bars": 5,    # Number of bars on the left side for the pivot (default: 5)
    "right_bars": 5,   # Number of right-side bars for the pivot (default: 5)
    "max_levels": 3,   # Maximum swing level to track (default: 3)
}
```

**CHoCH (Change of Character) - Usage Examples:**

```python
# CHoCH outputs: 1 (bullish CHoCH), -1 (bearish CHoCH), 0 (none)

self.entry_conditions = {
    'long': [
        ["choch", "==", 1],                 # Bullish trend change
        ["fvg", ">", 0],                    # + FVG confirmation
    ],
    'short': [
        ["choch", "==", -1],                # Bearish trend change
        ["fvg", "<", 0],                    # + FVG confirmation
    ]
}

# Exit on reversal
self.exit_conditions = {
    'long': [
        ["choch", "==", -1],                # CHoCH reversal (bullish → bearish)
    ],
    'short': [
        ["choch", "==", 1],                 # CHoCH reversal (bearish → bullish)
    ]
}
```

**iFVG (Inverse Fair Value Gap) - Usage Examples:**

```python
# iFVG outputs: 1 (bullish reversal), -1 (bearish reversal), 0 (none)

self.entry_conditions = {
    'long': [
        ["ifvg", "==", 1],                  # Bullish iFVG (reversal signal)
        ["rsi_14", "<", 30],                # Oversold confirmation
    ],
    'short': [
        ["ifvg", "==", -1],                 # Bearish iFVG
        ["rsi_14", ">", 70],                # Overbought confirmation
    ]
}
```

**SMC Kombinasyon Stratejisi:**

```python
self.entry_conditions = {
    'long': [
        # SMC core signals
        ["fvg", ">", 0],                    # Bullish FVG
        ["choch", "==", 1],                 # Trend change confirmation
        ["bos", "==", 1, "15m"],            # BoS on higher timeframe

        # Traditional filters
        ["close", ">", "ema_55", "1h"],     # Trend filter
        ["rsi_14", "between", [45, 70]],    # Momentum
        ["cmf_20", ">", 0],                 # Volume confirmation
    ],
    'short': [
        ["fvg", "<", 0],
        ["choch", "==", -1],
        ["bos", "==", -1, "15m"],
        ["close", "<", "ema_55", "1h"],
        ["rsi_14", "between", [30, 55]],
        ["cmf_20", "<", 0],
    ]
}
```

---

## 🎭 Pattern Detection

### Candlestick Patterns

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "candlestick_patterns": {}  # All patterns are automatically detected
    }
)

# Entry conditions
self.entry_conditions = {
    'long': [
        # Bullish patterns
        ['hammer', '==', 1],              # Hammer pattern
        ['engulfing_bullish', '==', 1],   # Bullish Engulfing
        ['morning_star', '==', 1],        # Morning Star

        # Trend filter (IMPORTANT!)
        ['close', '>', 'ema_50'],
        ['rsi', '<', 50],
    ],
    'short': [
        # Bearish patterns
        ['shooting_star', '==', 1],       # Shooting Star
        ['engulfing_bearish', '==', 1],   # Bearish Engulfing
        ['evening_star', '==', 1],        # Evening Star

        # Trend filter
        ['close', '<', 'ema_50'],
        ['rsi', '>', 50],
    ]
}
```

**Desteklenen Candlestick Pattern'ler (15+):**

- `hammer` - Hammer (bullish reversal)
- `shooting_star` - Shooting Star (bearish reversal)
- `engulfing_bullish` - Bullish Engulfing
- `engulfing_bearish` - Bearish Engulfing
- `harami_bullish` - Bullish Harami
- `harami_bearish` - Bearish Harami
- `morning_star` - Morning Star (bullish reversal)
- `evening_star` - Evening Star (bearish reversal)
- `three_white_soldiers` - Three White Soldiers (strong bullish)
- `three_black_crows` - Three Black Crows (strong bearish)
- `piercing_line` - Piercing Line (bullish)
- `dark_cloud_cover` - Dark Cloud Cover (bearish)
- `doji` - Doji (indecision)
- `marubozu_bullish` - Bullish Marubozu
- `marubozu_bearish` - Bearish Marubozu

### TALib Patterns

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "talib_patterns": {}  # TALib pattern library
    }
)

# Entry conditions
self.entry_conditions = {
    'long': [
        # TALib patterns return 100 (bullish), -100 (bearish), 0 (none)
        ['CDLHAMMER', '==', 100],          # Hammer
        ['CDLMORNINGSTAR', '==', 100],     # Morning Star
        ['CDL3WHITESOLDIERS', '==', 100],  # Three White Soldiers
    ],
    'short': [
        ['CDLSHOOTINGSTAR', '==', -100],   # Shooting Star
        ['CDLEVENINGSTAR', '==', -100],    # Evening Star
        ['CDL3BLACKCROWS', '==', -100],    # Three Black Crows
    ]
}
```

**Desteklenen TALib Pattern'ler (12+):**

- `CDLDOJI` - Doji
- `CDLHAMMER` - Hammer
- `CDLSHOOTINGSTAR` - Shooting Star
- `CDLENGULFING` - Engulfing Pattern
- `CDLMORNINGSTAR` - Morning Star
- `CDLEVENINGSTAR` - Evening Star
- `CDL3WHITESOLDIERS` - Three White Soldiers
- `CDL3BLACKCROWS` - Three Black Crows
- `CDLHARAMI` - Harami Pattern
- `CDLPIERCING` - Piercing Line
- `CDLABANDONEDBABY` - Abandoned Baby
- `CDLADVANCEBLOCK` - Advance Block

**Pattern Combination Strategy:**

```python
self.technical_parameters = TechnicalParameters(
    indicators={
        "candlestick_patterns": {},
        "talib_patterns": {},
        "rsi": {"period": 14},
        "ema_20": {"period": 20},
        "ema_50": {"period": 50},
    }
)

self.entry_conditions = {
    'long': [
        # Pattern confirmation (any pattern)
        ['hammer', '==', 1],  # OR CDLHAMMER == 100

        # Technical confirmation (IMPORTANT!)
        ['rsi', '<', 40],              # Oversold
        ['close', '>', 'ema_20'],      # Short-term uptrend
        ['ema_20', '>', 'ema_50'],     # Medium-term uptrend
    ]
}
```

---

## 🔄 Multi-Timeframe (MTF) Usage

Multi-timeframe analysis uses indicators from different timeframes to generate more reliable signals.

### MTF Configuration

```python
class MTFStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.strategy_name = "MTF_Strategy"

        # 1. DEFINE THE TIMEFRAMES
        self.primary_timeframe = "5m"         # Ana entry timeframe
        self.mtf_timeframes = ["15m", "1h"]   # Ek timeframe'ler

        # 2. DEFINE INDICATORS (to be used in all timeframes)
        self.technical_parameters = TechnicalParameters(
            indicators={
                "ema_20": {"period": 20},
                "ema_50": {"period": 50},
                "ema_200": {"period": 200},
                "rsi": {"period": 14},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "supertrend": {"period": 10, "multiplier": 3.0}
            }
        )

        # 3. ENTRY CONDITIONS (MTF usage)
        self.entry_conditions = {
            'long': [
                # 1h - Major trend (higher timeframe confirmation)
                ['close', '>', 'ema_200', '1h'],           # 1h'de uptrend
                ['supertrend_trend', '==', 1, '1h'],       # 1h SuperTrend bullish

                # 15m - Orta trend (intermediate confirmation)
                ['ema_20', '>', 'ema_50', '15m'],          # 15m'de EMA aligned
                ['macd_macd', '>', 'macd_signal', '15m'],  # 15m MACD bullish

                # 5m - Entry trigger (entry timeframe)
                ['rsi', '<', 50, '5m'],                    # 5m RSI pullback
                ['close', '>', 'ema_20', '5m'],            # 5m price above EMA
                ['ema_20', 'crossover', 'ema_50', '5m'],   # 5m fresh crossover
            ],
            'short': [
                # 1h trend bearish
                ['close', '<', 'ema_200', '1h'],
                ['supertrend_trend', '==', -1, '1h'],

                # 15m confirmation
                ['ema_20', '<', 'ema_50', '15m'],
                ['macd_macd', '<', 'macd_signal', '15m'],

                # 5m entry
                ['rsi', '>', 50, '5m'],
                ['close', '<', 'ema_20', '5m'],
                ['ema_20', 'crossunder', 'ema_50', '5m'],
            ]
        }

        self.exit_strategy = ExitStrategy(
            stop_loss_percent=1.0,
            take_profit_percent=2.0,
            trailing_stop_enabled=True
        )
```

### MTF Entry Logic Explanation

**Timeframe Hierarchy:**
1. **1h (Highest)**: Major trend direction (filter)
2. **15m (Middle)**: Medium-term momentum (confirmation)
3. **5m (Entry)**: Entry timing (trigger)

**Logic:**
- There should be a 1-hour uptrend (major filter)
- At 15m, the momentum should be bullish (confirmation).
- At 5 minutes, enter after the pullback (timing)

### MTF Example 1: Trend Alignment Strategy

```python
class TrendAlignmentMTF(BaseStrategy):
    """
    The trend should be aligned in all timeframes.
    """
    def __init__(self):
        super().__init__()

        self.strategy_name = "Trend_Alignment_MTF"
        self.primary_timeframe = "15m"
        self.mtf_timeframes = ["1h", "4h"]

        self.technical_parameters = TechnicalParameters(
            indicators={
                "ema_20": {"period": 20},
                "ema_50": {"period": 50},
                "adx": {"period": 14},
                "rsi": {"period": 14}
            }
        )

        self.entry_conditions = {
            'long': [
                # 4h - Major trend MUST be bullish
                ['ema_20', '>', 'ema_50', '4h'],

                # 1h - Intermediate trend bullish
                ['ema_20', '>', 'ema_50', '1h'],
                ['adx_adx', '>', 25, '1h'],  # Strong trend

                # 15m - Entry timing
                ['ema_20', 'crossover', 'ema_50', '15m'],  # Fresh crossover
                ['rsi', '>', 50, '15m'],  # Momentum confirmation
            ]
        }
```

### MTF Example 2: Higher TF Filter + Lower TF Entry

```python
class HTFFilterLTFEntry(BaseStrategy):
    """
    Higher timeframe filter, lower timeframe entry
    Classic MTF approach
    """
    def __init__(self):
        super().__init__()

        self.strategy_name = "HTF_Filter_LTF_Entry"
        self.primary_timeframe = "5m"
        self.mtf_timeframes = ["15m", "1h"]

        self.technical_parameters = TechnicalParameters(
            indicators={
                # HTF - Filter indicators
                "ema_200": {"period": 200},
                "supertrend": {"period": 10, "multiplier": 3.0},

                # LTF - Entry indicators
                "rsi": {"period": 14},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "bollinger": {"period": 20, "std_dev": 2.0}
            }
        )

        self.entry_conditions = {
            'long': [
                # === HIGHER TIMEFRAME FILTER (1h) ===
                ['close', '>', 'ema_200', '1h'],         # 1h major uptrend
                ['supertrend_trend', '==', 1, '1h'],     # 1h SuperTrend bullish

                # === INTERMEDIATE FILTER (15m) ===
                ['macd_macd', '>', 0, '15m'],            # 15m MACD above zero
                ['rsi', '>', 40, '15m'],                 # 15m not oversold

                # === LOWER TIMEFRAME ENTRY (5m) ===
                ['rsi', '<', 40, '5m'],                  # 5m oversold (pullback)
                ['close', '<', 'bollinger_lower', '5m'], # 5m at lower band
                ['macd_macd', 'crossover', 'macd_signal', '5m'],  # 5m MACD turn
            ],
            'short': [
                # HTF filter
                ['close', '<', 'ema_200', '1h'],
                ['supertrend_trend', '==', -1, '1h'],

                # Intermediate
                ['macd_macd', '<', 0, '15m'],
                ['rsi', '<', 60, '15m'],

                # LTF entry
                ['rsi', '>', 60, '5m'],
                ['close', '>', 'bollinger_upper', '5m'],
                ['macd_macd', 'crossunder', 'macd_signal', '5m'],
            ]
        }

        self.exit_strategy = ExitStrategy(
            stop_loss_percent=1.0,
            take_profit_percent=2.5,
            trailing_stop_enabled=True,
            trailing_callback_percent=0.5
        )
```

### MTF Example 3: Confluence Strategy

```python
class ConfluenceMTF(BaseStrategy):
    """
    Multiple timeframe confluence (signal at the same level in different timeframes)
    """
    def __init__(self):
        super().__init__()

        self.strategy_name = "Confluence_MTF"
        self.primary_timeframe = "15m"
        self.mtf_timeframes = ["5m", "1h"]

        self.technical_parameters = TechnicalParameters(
            indicators={
                "pivot_points": {},
                "fibonacci_retracement": {"lookback": 50},
                "ema_50": {"period": 50},
                "rsi": {"period": 14}
            }
        )

        self.entry_conditions = {
            'long': [
                # Confluence: Price is at an important level (multiple TF)
                # 1h pivot support
                ['close', '>', 'pivot_points_S1', '1h'],
                ['close', '<', 'pivot_points_P', '1h'],

                # 15m Fib golden zone
                ['close', '>', 'fib_retracement_Fib_61.8', '15m'],
                ['close', '<', 'fib_retracement_Fib_50.0', '15m'],

                # 5m momentum turn
                ['rsi', '<', 40, '5m'],  # Was oversold
                ['rsi', 'rising', 2, '5m'],  # Now rising
            ]
        }
```

### MTF Best Practices

✅ **DO:**
- Use a trend filter for higher TF (1h, 4h).
- Use entry timing for lower TF (5m, 15m).
- Timeframe ratio 3:1 or 4:1 (e.g., 5m + 15m + 1h or 1m + 5m + 15m)
- Use at least 2, at most 3 timeframes.

❌ **DON'T:**
- Do not use too many timeframes (4+), it can cause confusion.
- Do not use short timeframes (5m + 10m, too similar).
- In lower TF, filter; in higher TF, entry (reversed logic)
- All timeframes' in the same indicator (redundant)

### MTF Timeframe Combinations

**Scalping (fast):**
- 1m + 5m + 15m
- Primary: 1m (entry)
- Filter: 5m, 15m

**Intraday:**
- 5m + 15m + 1h
- Primary: 5m (entry)
- Filter: 15m, 1h

**Swing:**
- 1h + 4h + 1d
- Primary: 1h (entry)
- Filter: 4h, 1d

**Position:**
- 4h + 1d + 1w
- Primary: 4h (entry)
- Filter: 1d, 1w

### MTF Exit Strategy

```python
self.exit_conditions = {
    'long': [
        # Exit on higher TF reversal
        ['supertrend_trend', '==', -1, '1h'],  # 1h trend reversal

        # OR exit on entry TF signal
        ['macd_macd', 'crossunder', 'macd_signal', '5m'],  # 5m reversal
    ],
    'short': [
        ['supertrend_trend', '==', 1, '1h'],
        ['macd_macd', 'crossover', 'macd_signal', '5m'],
    ]
}
```

**Exit Logic:**
- Higher TF reversal → exit immediately (major trend change)
- Entry TF signal → let some room (noise filtration)
- Use trailing stop to lock profits

---

## 📚 Example Strategies

### 1. Simple RSI Strategy

```python
class SimpleRSIStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.strategy_name = "Simple_RSI"

        self.technical_parameters = TechnicalParameters(
            indicators={
                "rsi": {"period": 14},
                "ema_20": {"period": 20}
            }
        )

        self.entry_conditions = {
            'long': [
                ['rsi', '<', 30],           # Oversold
                ['close', '>', 'ema_20']    # Price above EMA
            ],
            'short': [
                ['rsi', '>', 70],
                ['close', '<', 'ema_20']
            ]
        }

        self.exit_strategy = ExitStrategy(
            stop_loss_percent=1.0,
            take_profit_percent=2.0
        )
```

### 2. MACD + SuperTrend Strategy

```python
class MACDSupertrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.strategy_name = "MACD_Supertrend"

        self.technical_parameters = TechnicalParameters(
            indicators={
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "supertrend": {"period": 10, "multiplier": 3.0},
                "atr": {"period": 14}
            }
        )

        self.entry_conditions = {
            'long': [
                ['supertrend_trend', '==', 1],              # Bullish SuperTrend
                ['macd_macd', 'crossover', 'macd_signal'],  # MACD crossover
            ],
            'short': [
                ['supertrend_trend', '==', -1],
                ['macd_macd', 'crossunder', 'macd_signal'],
            ]
        }

        # ATR-based dynamic stop loss
        self.exit_strategy = ExitStrategy(
            stop_loss_method=StopLossMethod.ATR_BASED,
            stop_loss_atr_multiplier=2.0,
            take_profit_method=ExitMethod.RISK_REWARD,
            take_profit_risk_reward_ratio=3.0
        )
```

### 3. Multi-Indicator Trend Following

```python
class MultiIndicatorStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.strategy_name = "Multi_Indicator_Trend"

        self.technical_parameters = TechnicalParameters(
            indicators={
                # Trend
                "ema_20": {"period": 20},
                "ema_50": {"period": 50},
                "ema_200": {"period": 200},

                # Momentum
                "rsi": {"period": 14},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},

                # Volatility
                "bollinger": {"period": 20, "std_dev": 2.0},
                "atr": {"period": 14}
            }
        )

        self.entry_conditions = {
            'long': [
                # Trend alignment
                ['ema_20', '>', 'ema_50'],
                ['ema_50', '>', 'ema_200'],
                ['close', '>', 'ema_20'],

                # Momentum
                ['rsi', 'between', [40, 70]],
                ['macd_macd', '>', 'macd_signal'],

                # Volatility
                ['close', '>', 'bollinger_middle'],
            ],
            'short': [
                ['ema_20', '<', 'ema_50'],
                ['ema_50', '<', 'ema_200'],
                ['close', '<', 'ema_20'],
                ['rsi', 'between', [30, 60]],
                ['macd_macd', '<', 'macd_signal'],
                ['close', '<', 'bollinger_middle'],
            ]
        }

        self.exit_strategy = ExitStrategy(
            stop_loss_method=StopLossMethod.ATR_BASED,
            stop_loss_atr_multiplier=2.0,
            take_profit_method=ExitMethod.RISK_REWARD,
            take_profit_risk_reward_ratio=2.0,
            trailing_stop_enabled=True,
            trailing_callback_percent=1.0
        )
```

### 4. Pattern + Momentum Strategy

```python
class HammerMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.strategy_name = "Hammer_Momentum"

        self.technical_parameters = TechnicalParameters(
            indicators={
                "candlestick_patterns": {},
                "rsi": {"period": 14},
                "ema_20": {"period": 20},
                "ema_50": {"period": 50}
            }
        )

        self.entry_conditions = {
            'long': [
                # Pattern
                ['hammer', '==', 1],  # Hammer pattern

                # Trend context (CRITICAL!)
                ['close', '>', 'ema_20'],
                ['ema_20', '>', 'ema_50'],

                # Momentum
                ['rsi', 'between', [30, 50]],  # Oversold recovery
            ],
            'short': [
                ['shooting_star', '==', 1],
                ['close', '<', 'ema_20'],
                ['ema_20', '<', 'ema_50'],
                ['rsi', 'between', [50, 70]],
            ]
        }

        # Tight stops for pattern trades
        self.exit_strategy = ExitStrategy(
            stop_loss_percent=0.8,
            take_profit_percent=1.5,
            trailing_stop_enabled=True
        )
```

---

## 💡 Best Practices

### 1. Indicator Selection

✅ **DO:**
- 3-5 indicators are sufficient (from different categories)
- Trend + Momentum + Volume/Volatility kombinasyonu
- Know the purpose of each indicator.

❌ **DON'T:**
- Use fewer indicators (over-fitting)
- Multiple indicators from the same category (5 different MAs).
- Redundant indicator'lar (RSI + CCI + Williams %R hepsi momentum)

### 2. Entry Conditions

✅ **DO:**
- 2-4 core condition + 1-2 filter optimal
- Mutlaka trend filter ekle
- Use volume/momentum confirmation.

❌ **DON'T:**
- 10+ condition (no process is opened)
- Pattern-only entry (low win rate)
- Trade without a trend filter.

### 3. Pattern Trading

✅ **DO:**
- Pattern + Trend filter + Momentum confirmation
- Volume confirmation ekle
- Use a tight stop loss (patterns quick)
- Trend context is important (hammer in an uptrend, shooting star in a downtrend)

❌ **DON'T:**
- Pattern alone (win rate %30)
- Counter-trend pattern trades (hammer in downtrend)
- Ignore volume

### 4. Exit Strategy

✅ **DO:**
- Always use stop loss.
- Risk/Reward minimum 1:1.5
- Trailing stop protects profit.
- ATR-based dynamic stop loss

❌ **DON'T:**
- No stop loss
- Take profit too tight
- Fixed stop in volatile markets

### 5. Common Mistakes

**❌ Mistake 1: Parameter Mismatch**
```python
# Wrong
"squeeze": {"bb_mult": 2.0}  # Parameter name is incorrect

# Correct
"squeeze": {"bb_std": 2.0}   # The parameter name in the registry
```

**❌ Mistake 2: Wrong Condition Format**
```python
# Wrong
self.entry_conditions = {
    "buy": [...],   # "buy" instead of "buy"
    "sell": [...],  # "sell" is not
}

# Correct
self.entry_conditions = {
    'long': [...],   # Use 'long'
    'short': [...],  # Use 'short'
}
```

**❌ Mistake 3: Over-fitting**
```python
# Bad: 11 filters! No trades will open.
self.entry_conditions = {
    'long': [
        ['rsi', '>', 50], ['rsi', '<', 70],
        ['macd_macd', '>', 'macd_signal'],
        ['ema_20', '>', 'ema_50'], ['ema_50', '>', 'ema_200'],
        ['adx_adx', '>', 25],
        ['stochastic_k', '>', 20], ['stochastic_k', '<', 80],
        ['volume', '>', 'volume_sma_20'],
        ['atr', '>', 0.5'],
        ['close', '>', 'bollinger_middle']
    ]
}

# Good: 3-4 filters
self.entry_conditions = {
    'long': [
        ['ema_20', 'crossover', 'ema_50'],  # Entry signal
        ['rsi', '>', 50],                    # Momentum
        ['adx_adx', '>', 25],                # Trend strength
    ]
}
```

---

## 🔗 Kaynaklar

### Documentation
- [Indicator Registry](./__init__.py) - 76 indicator list
- [Base Indicator](./base_indicator.py) - Indicator base class
- [Discovery Script](./discovery_indicators.py) - Auto-discovery

### Stratejiler
- [Strategy Templates](../../strategies/templates/) - Example strategies
- [Base Strategy](../../strategies/base_strategy.py) - Strategy base class

### Testler
- [Consistency Test](../../../test_indicator_consistency.py) - Indicator tests

---

**Last Updated**: 2025-12-24
**Version**: 2.2.0
**Author**: SuperBot Team

**Changelog v2.2.0**:
- ✅ BOS now uses SwingPoints (code duplication removed)
- ✅ TradingView compatible pivot algorithm documented
- ✅ BoS parameters and usage examples have been expanded.

**Changelog v2.1.0**:
- ✅ SMC (Structure) indicators section added (FVG, iFVG, BoS, CHoCH)
- ✅ FVG usage examples with correct output format
- ✅ FVG bug fix documented (net zone count instead of 100/-100)

🚀 **Happy Trading!** 🚀
