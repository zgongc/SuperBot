# modules/analysis - Market Structure Analysis Module

## Purpose
This is a source-agnostic analysis module that detects SMC (Smart Money Concepts) formations from the given candle data.

## Usage Scenarios
1. **Replay Mode** - Shows formations from historical data.
2. **Live Mode** - Detection from real-time WebSocket data.
3. **Backtest** - Analysis for strategy development.
4. **Standalone** - Direct analysis via CLI or API.

## Mimari

```
modules/analysis/
├── __init__.py              # Module exports
├── analysis_engine.py       # Ana engine - orchestrator
├── detectors/
│   ├── __init__.py
│   ├── base_detector.py     # Abstract base class
│   ├── bos_detector.py      # Break of Structure
│   ├── choch_detector.py    # Change of Character
│   ├── fvg_detector.py      # Fair Value Gap
│   ├── ob_detector.py       # Order Blocks
│   ├── swing_detector.py    # Swing High/Low
│   └── liquidity_detector.py # Liquidity Zones
├── models/
│   ├── __init__.py
│   ├── formations.py        # Formation dataclasses (BOS, CHoCH, FVG, etc.)
│   └── analysis_result.py   # Unified result container
└── utils/
    ├── __init__.py
    └── candle_utils.py      # Candle processing helpers
```

## Core Classes

### 1. AnalysisEngine (Orchestrator)
```python
class AnalysisEngine:
    """
    Main analysis engine - coordinates all detectors.

    Usage:
        engine = AnalysisEngine()

        # DataFrame'den analiz
        result = engine.analyze(df)

        # Single candle update (streaming)
        result = engine.update(candle)

        # Query specific formations
        bos_list = engine.get_formations('bos')
        fvg_list = engine.get_formations('fvg', active_only=True)
    """
```

### 2. BaseDetector (Abstract)
```python
class BaseDetector(ABC):
    """
    This is the base class for all detectors.

    Methods:
        detect(df: DataFrame) -> List[Formation]
        update(candle: dict) -> Optional[Formation]
        get_active() -> List[Formation]
        invalidate(formation_id) -> None
    """
```

### 3. Formation Models
```python
@dataclass
class BOSFormation:
    """Break of Structure"""
    id: str
    type: Literal['bullish', 'bearish']
    broken_level: float      # Broken swing level
    break_price: float       # Break price
    break_time: int          # Break timestamp
    swing_index: int         # The bar index of the broken swing
    strength: float          # 0-100 strength score

@dataclass
class CHoCHFormation:
    """Change of Character"""
    id: str
    type: Literal['bullish', 'bearish']
    previous_trend: str
    broken_level: float
    break_time: int
    significance: float      # Importance of the trend change

@dataclass
class FVGFormation:
    """Fair Value Gap"""
    id: str
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    filled: bool
    filled_percent: float    # How much is filled (0-100)
    age: int                 # How many bars ago it was created

@dataclass
class SwingPoint:
    """Swing High/Low"""
    id: str
    type: Literal['high', 'low']
    price: float
    time: int
    index: int
    broken: bool

@dataclass
class OrderBlockFormation:
    """Order Block"""
    id: str
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    mitigated: bool          # Visited?
    strength: float
```

### 4. AnalysisResult
```python
@dataclass
class AnalysisResult:
    """Unified analysis result"""
    timestamp: int

    # Current bar formations
    new_bos: Optional[BOSFormation]
    new_choch: Optional[CHoCHFormation]
    new_fvg: Optional[FVGFormation]
    new_swing: Optional[SwingPoint]
    new_ob: Optional[OrderBlockFormation]

    # Active formations (unfilled/unbroken)
    active_fvgs: List[FVGFormation]
    active_obs: List[OrderBlockFormation]

    # Current levels
    swing_high: Optional[float]
    swing_low: Optional[float]

    # Bias
    market_bias: Literal['bullish', 'bearish', 'neutral']
    trend: str

    def to_dict(self) -> dict:
        """JSON serializable dict"""

    def get_chart_annotations(self) -> List[dict]:
        """Annotation list for LightweightCharts"""
```

## API Usage

### Batch Analysis (DataFrame)
```python
from modules.analysis import AnalysisEngine

engine = AnalysisEngine(config={
    'swing_left_bars': 5,
    'swing_right_bars': 5,
    'fvg_min_size_pct': 0.1,
    'fvg_max_age': 50,
})

# DataFrame loaded from Parquet/CSV
df = pd.read_parquet('data.parquet')

# Analyze all data
results = engine.analyze_batch(df)

# Get the occurrences for each bar
for i, result in enumerate(results):
    if result.new_bos:
        print(f"Bar {i}: {result.new_bos.type} BOS at {result.new_bos.break_price}")
    if result.new_fvg:
        print(f"Bar {i}: {result.new_fvg.type} FVG ({result.new_fvg.bottom}-{result.new_fvg.top})")
```

### Streaming Analysis (Real-time)
```python
engine = AnalysisEngine()
engine.warmup(historical_df)  # Warm up with the first N bars

# For each candle received from WebSocket
async def on_candle(candle: dict):
    result = engine.update(candle)

    if result.new_bos:
        await notify_bos(result.new_bos)

    # Check for active FVG (fair value gaps) (has the price approached?)
    for fvg in result.active_fvgs:
        if is_price_near(candle['close'], fvg):
            await notify_fvg_approach(fvg)
```

### ReplayMode Entegrasyonu
```python
# Usage in ReplayService
class ReplayService:
    def __init__(self):
        self.analysis_engine = AnalysisEngine()

    async def create_session(self, strategy_id, analyze_smc=False):
        # ...
        if analyze_smc:
            # Analyze all data
            self.smc_results = self.analysis_engine.analyze_batch(df)

    async def get_candles(self, session_id, start, limit):
        candles = ...

        # SMC annotations ekle
        annotations = []
        for i in range(start, start + limit):
            if i < len(self.smc_results):
                annotations.extend(
                    self.smc_results[i].get_chart_annotations()
                )

        return {
            'candles': candles,
            'annotations': annotations
        }
```

## Relationship with the Existing SMC Indicator

`components/indicators/structure/smc.py` already calculates BOS/CHoCH/FVG. However:

| Feature | SMC Indicator | Analysis Module |
|---------|---------------|-----------------|
| Purpose | Vectorized for backtesting | Analysis + visualization |
| Output | Numpy arrays (batch) | Formation objects (rich) |
| Streaming | Yes, but basic | Full featured |
| History | Only the last value | All active formations |
| Visualization | None | Chart annotations |

**Decision**: The analysis module can either **use the SMC Indicator internally** or implement its own logic. Let's wrap the SMC in the initial version, and we can separate it later if necessary.

## Phase 1 - Basic Structure ✅ DONE
1. [x] PLAN.md (this file)
2. [x] `models/formations.py` - Dataclass
3. [x] `models/analysis_result.py` - Result container
4. [x] `analysis_engine.py` - Temel engine
5. [x] `detectors/base_detector.py` - Abstract base

## Faz 2 - Detectors ✅ DONE
1. [x] `detectors/swing_detector.py` - Swing High/Low
2. [x] `detectors/structure_detector.py` - BOS + CHoCH (combined)
3. [x] `detectors/fvg_detector.py` - FVG
4. [x] `detectors/pattern_detector.py` - Candlestick patterns wrapper
5. [ ] `detectors/ob_detector.py` - Order Blocks (TODO)

## Faz 3 - Entegrasyon
1. [ ] ReplayMode entegrasyonu
2. [ ] WebUI API endpoint'i
3. [ ] Frontend chart annotations
4. [ ] CLI interface

## Phase 4 - Advanced
1. [ ] MTF (Multi-Timeframe) analiz
2. [ ] Confluence detection (multiple occurrences)
3. [ ] Alert system
4. [ ] AI-powered pattern recognition

## Configuration
```yaml
# config/analysis.yaml
analysis:
  swing:
    left_bars: 5
    right_bars: 5

  bos:
    max_levels: 5
    trend_strength: 2

  fvg:
    min_size_pct: 0.1
    max_age: 50

  order_blocks:
    strength_threshold: 1.0
    max_blocks: 3

  output:
    include_filled_fvg: false
    include_broken_swings: false
```
