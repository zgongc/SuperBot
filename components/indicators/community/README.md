# Community Indicators

TradingView'dan ported community indicators.

## MavilimW

**Kaynak:** Kivanc Ozbilgic (@mavilim0732)
**Type:** Trend Following

### Description

It performs 6-stage chained weighted moving average (WMA) calculations to filter noise and produce clean trend signals. It uses a Fibonacci-like period increase (3, 5, 8, 13, 21, 34).

### Formula

```
fmal=3, smal=5 için:
M1 = WMA(close, 3)
M2 = WMA(M1, 5)
M3 = WMA(M2, 8)      # 3+5
M4 = WMA(M3, 13)     # 5+8
M5 = WMA(M4, 21)     # 8+13
MAVW = WMA(M5, 34)   # 13+21
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|----------|
| `fmal` | 3 | First MA length |
| `smal` | 5 | Second MA length |

### Output

| Column | Type | Description |
|-------|-----|----------|
| `mavw` | float | Blue value (price overlay) |
| `trend_direction` | int | 1=up (blue), -1=down (red), 0=neutral |

### Used Internal Indicators

- `indicators.trend.wma.WMA` - For a 6-stage WMA chain.

### Strategy Usage

```python
# indicators
"mavilimw": {"fmal": 3, "smal": 5}

# entry_conditions
"long": [
    ['close', '>', 'mavilimw_mavw'],           # Price is above MAVW
    ['mavilimw_trend_direction', '==', 1],     # Blue (uptrend)
],
"short": [
    ['close', '<', 'mavilimw_mavw'],           # Price is below MAVW
    ['mavilimw_trend_direction', '==', -1],   # Red (decline)
],
```

### Signals

- **Long:** When the price goes above MAVW and MAVW is rising (blue)
- **Short:** When the price falls below MAVW and MAVW is decreasing (red).
- **Exit Long:** When `trend_direction` is -1.
- **Exit Short:** When `trend_direction` is 1.

---

## PMax

**Source:** Kivanc Ozbilgic (@KivancOzbilgic)
**Type:** Trend Following / Dynamic Support-Resistance

### Description

PMax is an ATR-based trend indicator that creates dynamic support and resistance levels. It uses EMA (or other moving average types) and ATR to determine trend direction and generate buy/sell signals. PMax filters market noise and provides clear trend-following signals.

**Key Features:**
- Dynamic support/resistance levels based on volatility (ATR)
- Clear trend direction (UP/DOWN)
- Reduced noise with EMA smoothing
- ATR-based adaptive stop levels
- Realtime calculation with incremental updates

### Formula

```
Source = (High + Low) / 2  (hl2)
MAvg = EMA(Source, ma_period)
ATR = ATR(atr_period)

longStop = MAvg - (atr_multiplier × ATR)
shortStop = MAvg + (atr_multiplier × ATR)

Trend Direction Logic:
- UP trend: when MAvg > previous shortStop
- DOWN trend: when MAvg < previous longStop
- Maintain previous trend otherwise

PMax = Trend == 1 ? longStop : shortStop
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 10 | ATR calculation period |
| `atr_multiplier` | 3.0 | ATR multiplier for stop levels (sensitivity) |
| `ma_period` | 10 | Moving average period (EMA by default) |

**Parameter Tuning:**
- **Lower multiplier (1.5-2.5):** More sensitive, more signals, more noise
- **Higher multiplier (3.0-4.0):** Less sensitive, fewer signals, cleaner trends (default: 3.0)
- **Shorter periods (7-10):** Faster response, more whipsaws
- **Longer periods (14-20):** Slower response, more stable

### Output

| Column | Type | Description |
|--------|------|-------------|
| `pmax` | float | PMax value (longStop or shortStop based on trend) |
| `long_stop` | float | Dynamic support level |
| `short_stop` | float | Dynamic resistance level |
| `mavg` | float | Moving average (EMA of hl2) |
| `trend_direction` | int | 1=UP (bullish), -1=DOWN (bearish), 0=neutral |

### Used Internal Indicators

- `indicators.trend.ema.EMA` - For moving average calculation
- `indicators.volatility.atr.ATR` - For volatility-based stop levels

### Strategy Usage

#### Strategy 1: Crossover Entry (Trend Start)

**Best for:** Catching trend reversals, fewer trades, clean entries

```python
# indicators
"pmax": {
    "atr_period": 10,
    "atr_multiplier": 3.0,  # TradingView default
    "ma_period": 10
}

# entry_conditions
"long": [
    ['pmax_mavg', 'crossover', 'pmax_pmax'],    # MAvg crosses above PMax (support)
],
"short": [
    ['pmax_mavg', 'crossunder', 'pmax_pmax'],   # MAvg crosses below PMax (resistance)
],
```

**Characteristics:**
- ✅ **Pros:** Clean entry timing, catches trend start, fewer false signals
- ❌ **Cons:** Only one entry per crossover, cannot re-enter if trend continues after exit
- **Trade frequency:** Low (only at crossover moments)
- **Best timeframe:** 15m, 1h, 4h

#### Strategy 2: Trend Direction Entry (Re-entry Allowed)

**Best for:** Riding strong trends, multiple entries, scalping

```python
# indicators
"pmax": {
    "atr_period": 10,
    "atr_multiplier": 3.0,  # TradingView default
    "ma_period": 10
},
"ema_200": {"period": 200}  # Major trend filter (REQUIRED!)

# entry_conditions
"long": [
    ['close', '>', 'ema_200'],                  # Must use trend filter!
    ['pmax_trend_direction', '==', 1],          # Uptrend active
],
"short": [
    ['close', '<', 'ema_200'],                  # Must use trend filter!
    ['pmax_trend_direction', '==', -1],         # Downtrend active
],
```

**Characteristics:**
- ✅ **Pros:** Can re-enter after exit if trend continues, rides strong trends
- ❌ **Cons:** More trades = more commission/slippage, needs strong trend filter
- **Trade frequency:** High (every bar during trend)
- **Best timeframe:** 5m, 15m (with position management)
- **⚠️ WARNING:** Must use EMA200 or similar filter to avoid over-trading!

**Example Scenario:**
```
Bar 1: Crossover → Entry at 90,000
Bar 2-10: Uptrend continues (trend=1)
Bar 11: Exit at 92,000 (break-even/trailing stop)
Bar 12-20: STILL uptrend (trend=1, close > EMA200)

Crossover strategy: Cannot re-enter (no new crossover)
Trend direction: CAN re-enter at Bar 12 ✅ (trend still active)
```

#### Strategy 3: Crossover + EMA200 Filter (Recommended)

**Best for:** Balanced approach, high-quality signals, medium frequency

```python
# indicators
"pmax": {"atr_period": 10, "atr_multiplier": 3.0, "ma_period": 10},
"ema_200": {"period": 200}  # Major trend filter

# entry_conditions
"long": [
    ['close', '>', 'ema_200'],                  # Major uptrend confirmation
    ['pmax_mavg', 'crossover', 'pmax_pmax'],    # PMax crossover signal
],
"short": [
    ['close', '<', 'ema_200'],                  # Major downtrend confirmation
    ['pmax_mavg', 'crossunder', 'pmax_pmax'],   # PMax crossunder signal
],
```

**Characteristics:**
- ✅ **Pros:** Best win rate, filters false signals, clean entries
- ❌ **Cons:** Fewer trades, might miss some opportunities
- **Trade frequency:** Low-Medium
- **Best timeframe:** 15m, 1h, 4h
- **Recommended:** ⭐ Best for beginners and consistent profits

#### Strategy 4: Price Action + Trend Direction

**Best for:** Using PMax as dynamic support/resistance levels

```python
# indicators
"pmax": {"atr_period": 10, "atr_multiplier": 3.0, "ma_period": 10},
"ema_200": {"period": 200}  # Trend filter

# entry_conditions
"long": [
    ['close', '>', 'ema_200'],                  # Major trend filter
    ['pmax_trend_direction', '==', 1],          # Uptrend confirmed
    ['close', '>', 'pmax_long_stop'],           # Price above dynamic support
],
"short": [
    ['close', '<', 'ema_200'],                  # Major trend filter
    ['pmax_trend_direction', '==', -1],         # Downtrend confirmed
    ['close', '<', 'pmax_short_stop'],          # Price below dynamic resistance
],
```

**Characteristics:**
- ✅ **Pros:** Uses PMax as S/R levels, multiple entry confirmation
- ❌ **Cons:** More complex logic, needs good trending market
- **Trade frequency:** Medium
- **Best timeframe:** 15m, 1h

#### Exit Conditions

```python
# exit_conditions
"long": [
    ['pmax_mavg', 'crossunder', 'pmax_pmax'],   # Exit when trend reverses
],
"short": [
    ['pmax_mavg', 'crossover', 'pmax_pmax'],    # Exit when trend reverses
],
```

### Signals

**Entry Signals:**
- **Long (Crossover):** When `pmax_mavg` crosses above `pmax_pmax`
  - Indicates uptrend confirmation
  - Price breaking above dynamic support

- **Short (Crossunder):** When `pmax_mavg` crosses below `pmax_pmax`
  - Indicates downtrend confirmation
  - Price breaking below dynamic resistance

**Trend Signals:**
- **Uptrend:** `trend_direction == 1`
  - PMax acts as dynamic support (longStop)
  - Price tends to stay above PMax

- **Downtrend:** `trend_direction == -1`
  - PMax acts as dynamic resistance (shortStop)
  - Price tends to stay below PMax

**Exit Signals:**
- **Exit Long:** When trend changes to -1 or price crosses below PMax
- **Exit Short:** When trend changes to 1 or price crosses above PMax

### Best Practices

1. **Choose the right entry method:**
   - **Crossover:** For trend reversals, cleaner entries, fewer trades
   - **Trend Direction:** For riding trends, re-entry capability, more trades (needs EMA filter!)
   - **Decision factor:** If your exit strategy uses break-even/trailing stops and you want to re-enter during same trend, use trend_direction with EMA200 filter

2. **Combine with trend filter:** Use EMA200 or higher timeframe trend for better accuracy
   - **CRITICAL for trend_direction entries:** Without EMA200, you'll get too many signals
   - Optional for crossover entries (but still recommended)

3. **Use appropriate multiplier:** Default 3.0 is balanced; lower for scalping, higher for swing trading
   - Multiplier 2.0: More sensitive, more crossovers, more noise
   - Multiplier 3.0: Balanced (default, TradingView default)
   - Multiplier 4.0: Less sensitive, cleaner trends, fewer trades

4. **Avoid ranging markets:** PMax works best in trending markets
   - Use ADX filter (ADX > 25) to avoid ranging markets
   - Or combine with higher timeframe trend confirmation

5. **Consider timeframe:** Lower timeframes (5m, 15m) more signals; higher timeframes (1h, 4h) more reliable
   - 5m-15m: Use trend_direction with tight filters for scalping
   - 1h-4h: Use crossover for swing trading

6. **Backtest first:** Test different parameter combinations for your trading style
   - Test both entry methods (crossover vs trend_direction)
   - Compare trade frequency, win rate, and profit factor

### Strategy Comparison

| Strategy | Entry Signal | Re-entry | Trade Freq | Win Rate | Best For |
|----------|-------------|----------|------------|----------|----------|
| **1. Crossover** | `mavg crossover pmax` | ❌ No | Low | High | Trend reversals |
| **2. Trend Direction** | `trend_direction == 1` | ✅ Yes | High | Medium | Riding trends |
| **3. Crossover + EMA** | Crossover + EMA200 | ❌ No | Low-Med | Very High | **Recommended** |
| **4. Price Action** | Price + Trend + S/R | ✅ Yes | Medium | High | Advanced traders |

**Key Differences:**

**Crossover vs Trend Direction:**
```
Example: Uptrend from 90,000 to 95,000

Crossover approach:
- Bar 1: Crossover at 90,000 → Entry ✅
- Bar 5: Exit at 92,000 (break-even/trailing)
- Bar 6-20: Still uptrend, but NO crossover
- Result: Cannot re-enter, missed 92k → 95k move ❌

Trend Direction approach:
- Bar 1: Trend=1, close>EMA200 → Entry ✅
- Bar 5: Exit at 92,000 (break-even/trailing)
- Bar 6: Trend=1, close>EMA200 → Re-entry ✅
- Result: Captures 92k → 95k move too! ✅
```

**When to use which:**
- **Single entry per trend:** Use crossover (Strategy 1 or 3)
- **Multiple entries per trend:** Use trend_direction (Strategy 2 or 4)
- **Beginner:** Use Strategy 3 (Crossover + EMA200)
- **Active trader/Scalper:** Use Strategy 2 (Trend direction + EMA200)

### Comparison with Similar Indicators

| Feature | PMax | SuperTrend | Parabolic SAR |
|---------|------|-----------|---------------|
| Base | ATR + EMA | ATR + Close | High/Low tracking |
| Smoothing | Yes (EMA) | No | No |
| Noise filtering | Better | Good | Moderate |
| Trending markets | Excellent | Excellent | Good |
| Ranging markets | Poor | Poor | Poor |
| Lag | Low-Medium | Low | Low |

### Example: Simple PMax Strategy

See [simple_pmax.py](../../strategies/templates/simple_pmax.py) for a complete working example with:
- PMax + EMA200 trend filter
- Break-even and trailing stop
- Partial exits
- Risk management
