# Chart Pattern Detection - Implementation Plan

> **Version:** 1.0.0
> **Date:** 2025-01-23
> **Status:** Draft
> **Author:** SuperBot Team

---

## 1. Overview

### 1.1 Objective
Implement geometric chart pattern detection system (Head & Shoulders, Triangles, Wedges, etc.) with WebUI visualization similar to `/analysis/smc`.

### 1.2 Naming Convention
- **Detector:** `chart_pattern_detector.py` (not `formation_detector.py` - formations.py already exists for SMC)
- **Service:** `ChartPatternService`
- **Model:** `ChartPattern` dataclass
- **WebUI Route:** `/analysis/chart-patterns`

### 1.3 Existing Resources to Use
| Resource | Location | Usage |
|----------|----------|-------|
| ZigZag Indicator | `components/indicators/support_resistance/zigzag.py` | Swing point detection base |
| SwingPoint Model | `modules/analysis/models/formations.py` | Swing high/low data structure |
| BaseDetector | `modules/analysis/detectors/base_detector.py` | Detector base class |
| SMC Service | `modules/webui/services/smc_service.py` | Service pattern reference |
| Patterns Service | `modules/webui/services/patterns_service.py` | Candlestick pattern reference |

---

## 2. Chart Patterns to Detect

### 2.1 Phase 1 - Core Patterns (Priority)
| Pattern | Type | Min Swings | Description |
|---------|------|------------|-------------|
| **Double Top** | Bearish | 3 (H-L-H) | Two peaks at similar level |
| **Double Bottom** | Bullish | 3 (L-H-L) | Two troughs at similar level |
| **Head & Shoulders** | Bearish | 5 | Left shoulder, head, right shoulder |
| **Inverse H&S** | Bullish | 5 | Inverted head & shoulders |
| **Ascending Triangle** | Bullish | 4+ | Flat top, rising bottoms |
| **Descending Triangle** | Bearish | 4+ | Flat bottom, falling tops |

### 2.2 Phase 2 - Extended Patterns
| Pattern | Type | Description |
|---------|------|-------------|
| Symmetric Triangle | Neutral | Converging highs and lows |
| Rising Wedge | Bearish | Both highs and lows rising, converging |
| Falling Wedge | Bullish | Both highs and lows falling, converging |
| Channel Up | Bullish | Parallel ascending lines |
| Channel Down | Bearish | Parallel descending lines |
| Rectangle | Neutral | Horizontal consolidation |

### 2.3 Phase 3 - Advanced Patterns
| Pattern | Type | Description |
|---------|------|-------------|
| Cup & Handle | Bullish | U-shape with small pullback |
| Triple Top | Bearish | Three peaks at similar level |
| Triple Bottom | Bullish | Three troughs at similar level |
| Broadening Formation | Neutral | Expanding highs and lows |

---

## 3. Architecture

### 3.1 File Structure
```
modules/analysis/
├── detectors/
│   ├── chart_pattern_detector.py   # NEW - Main detector
│   └── ...
├── models/
│   ├── formations.py               # EXISTING - Add ChartPattern
│   └── ...

modules/webui/
├── services/
│   ├── chart_pattern_service.py    # NEW - WebUI service
│   └── ...
├── api/
│   ├── chart_patterns.py           # NEW - API endpoints
│   └── ...
├── views/
│   ├── chart_patterns.py           # NEW - View route
│   └── ...
├── templates/
│   ├── chart_patterns.html         # NEW - Template
│   └── ...
├── static/js/
│   ├── chart_patterns.js           # NEW - Frontend JS
│   └── ...

config/
├── analysis.yaml                   # ADD - chart_patterns section
```

### 3.2 Dependency Flow
```
ZigZag (indicator)
    ↓
SwingDetector (existing)
    ↓
ChartPatternDetector (NEW)
    ↓
ChartPatternService (NEW)
    ↓
WebUI API/Views
```

---

## 4. Data Models

### 4.1 ChartPattern Dataclass
```python
# Add to modules/analysis/models/formations.py

@dataclass
class ChartPattern:
    """
    Geometric Chart Pattern

    Attributes:
        id: Unique identifier
        name: Pattern name (e.g., 'double_top', 'head_shoulders')
        display_name: Human readable name
        type: 'bullish', 'bearish', 'neutral'
        status: 'forming', 'completed', 'confirmed', 'failed'
        swings: List of swing points forming the pattern
        neckline: Neckline price (for H&S patterns)
        target: Target price projection
        start_time: Pattern start timestamp
        end_time: Pattern completion timestamp
        start_index: Start bar index
        end_index: End bar index
        confidence: Pattern confidence score (0-100)
        breakout_price: Breakout level
        breakout_confirmed: Was breakout confirmed?
    """
    name: str
    display_name: str
    type: Literal['bullish', 'bearish', 'neutral']
    status: Literal['forming', 'completed', 'confirmed', 'failed']
    swings: List[SwingPoint]
    start_time: int
    end_time: int
    start_index: int
    end_index: int
    neckline: Optional[float] = None
    target: Optional[float] = None
    confidence: float = 50.0
    breakout_price: Optional[float] = None
    breakout_confirmed: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
```

### 4.2 Pattern Detection Config
```yaml
# config/analysis.yaml - ADD section

  # Chart Pattern Detection (Geometric Patterns)
  chart_patterns:
    enabled: true
    show: true

    # Swing detection (uses SMC SwingDetector)
    swing_left_bars: 5
    swing_right_bars: 5

    # Pattern tolerance (how close prices must be to count as "equal")
    price_tolerance_pct: 0.5  # 0.5% tolerance for double top/bottom
    neckline_tolerance_pct: 1.0  # 1% tolerance for neckline flatness

    # Pattern size constraints
    min_pattern_bars: 10
    max_pattern_bars: 200

    # Update behavior
    update_on_swing_only: true  # Only update patterns when new swing confirmed

    # Pattern overlap handling
    allow_overlap: true         # Show multiple patterns in same region
    max_patterns_display: 20    # Max patterns to show on chart

    # Breakout confirmation
    breakout_candles: 2         # Number of candles to confirm breakout
    breakout_pct: 0.5           # % move beyond breakout level

    # Individual patterns (Phase 1)
    patterns:
      double_top: true
      double_bottom: true
      head_shoulders: true
      inverse_head_shoulders: true
      ascending_triangle: true
      descending_triangle: true
      # Phase 2 (disabled by default)
      symmetric_triangle: false
      rising_wedge: false
      falling_wedge: false
      channel_up: false
      channel_down: false
```

---

## 5. Detection Algorithm

### 5.1 Core Logic Flow
```
1. Get Swing Points (from SwingDetector or ZigZag)
   - Filter by left_bars/right_bars confirmation
   - Get alternating High-Low sequence

2. Scan for Patterns
   - Sliding window over swing sequence
   - Match against pattern templates
   - Calculate confidence score

3. Validate Pattern
   - Check price tolerances
   - Check time constraints
   - Check volume (optional)

4. Track Pattern Status
   - forming: Still building
   - completed: All swings in place
   - confirmed: Breakout occurred
   - failed: Pattern invalidated
```

### 5.2 Pattern Detection Examples

#### Double Top Detection
```python
def detect_double_top(swings: List[SwingPoint]) -> Optional[ChartPattern]:
    """
    Double Top: H1 - L - H2 where H1 ≈ H2

    Conditions:
    1. H1 is swing high
    2. L is swing low (neckline)
    3. H2 is swing high
    4. |H1 - H2| / H1 < tolerance (e.g., 0.5%)
    5. H2 < H1 (slightly lower second peak = stronger)
    """
    # Need at least 3 swings: H, L, H
    if len(swings) < 3:
        return None

    # Find H-L-H sequence
    for i in range(len(swings) - 2):
        s1, s2, s3 = swings[i], swings[i+1], swings[i+2]

        if s1.type == 'high' and s2.type == 'low' and s3.type == 'high':
            # Check if peaks are at similar level
            diff_pct = abs(s1.price - s3.price) / s1.price * 100

            if diff_pct < self.config['price_tolerance_pct']:
                return ChartPattern(
                    name='double_top',
                    display_name='Double Top',
                    type='bearish',
                    status='completed',
                    swings=[s1, s2, s3],
                    neckline=s2.price,
                    target=s2.price - (s1.price - s2.price),  # Measured move
                    ...
                )
    return None
```

#### Head & Shoulders Detection
```python
def detect_head_shoulders(swings: List[SwingPoint]) -> Optional[ChartPattern]:
    """
    Head & Shoulders: H1 - L1 - H2 - L2 - H3

    Conditions:
    1. H2 > H1 and H2 > H3 (head is highest)
    2. H1 ≈ H3 (shoulders at similar level)
    3. L1 ≈ L2 (neckline)
    """
    if len(swings) < 5:
        return None

    for i in range(len(swings) - 4):
        s = swings[i:i+5]  # H1, L1, H2, L2, H3

        if (s[0].type == 'high' and s[1].type == 'low' and
            s[2].type == 'high' and s[3].type == 'low' and
            s[4].type == 'high'):

            # Head is highest
            if s[2].price > s[0].price and s[2].price > s[4].price:
                # Shoulders at similar level
                shoulder_diff = abs(s[0].price - s[4].price) / s[0].price * 100
                # Neckline relatively flat
                neckline_diff = abs(s[1].price - s[3].price) / s[1].price * 100

                if shoulder_diff < 3.0 and neckline_diff < 2.0:
                    neckline = (s[1].price + s[3].price) / 2
                    head_height = s[2].price - neckline

                    return ChartPattern(
                        name='head_shoulders',
                        display_name='Head & Shoulders',
                        type='bearish',
                        status='completed',
                        swings=s,
                        neckline=neckline,
                        target=neckline - head_height,
                        ...
                    )
    return None
```

---

## 6. WebUI Implementation

### 6.1 API Endpoints
```python
# modules/webui/api/chart_patterns.py

POST /api/chart-patterns/analyze
    Body: { symbol, timeframe, limit, start_date, end_date }
    Returns: { patterns: [...], candles: [...], annotations: {...} }

GET /api/chart-patterns/config
    Returns: { pattern_settings... }

GET /api/chart-patterns/info
    Returns: { pattern_definitions... }
```

### 6.2 View Route
```python
# modules/webui/views/chart_patterns.py

GET /analysis/chart-patterns
    Template: chart_patterns.html
```

### 6.3 Chart Visualization
```javascript
// Pattern visualization on LightweightCharts

// 1. Swing points as markers
candleSeries.setMarkers(swingMarkers);

// 2. Pattern lines (connecting swings)
// Use LineSeries for neckline, trendlines

// 3. Pattern zones (shaded area)
// Use BaselineSeries for pattern body

// 4. Target projection
// Use PriceLine for target level
```

---

## 7. Implementation Steps

### Phase 1: Core Infrastructure (1-2 days)
- [ ] Add `ChartPattern` dataclass to `formations.py`
- [ ] Create `chart_pattern_detector.py` with base structure
- [ ] Implement `detect_double_top()` and `detect_double_bottom()`
- [ ] Add config section to `analysis.yaml`

### Phase 2: WebUI Service (1 day)
- [ ] Create `chart_pattern_service.py`
- [ ] Create API endpoints in `chart_patterns.py`
- [ ] Create view route

### Phase 3: Frontend (1-2 days)
- [ ] Create `chart_patterns.html` template
- [ ] Create `chart_patterns.js` with chart logic
- [ ] Implement pattern visualization (lines, zones)
- [ ] Add filter checkboxes

### Phase 4: Extended Patterns (2-3 days)
- [ ] Implement Head & Shoulders detection
- [ ] Implement Triangle detection (ascending, descending, symmetric)
- [ ] Implement Wedge detection
- [ ] Implement Channel detection

### Phase 5: Testing & Refinement (1 day)
- [ ] Test with real market data
- [ ] Tune detection parameters
- [ ] Add pattern statistics

---

## 8. Technical Notes

### 8.1 SwingDetector Integration (CONFIRMED)
```python
# Use existing SwingDetector from modules/analysis/detectors/swing_detector.py
# This is the same detector used in SMC Analysis - well tested!
from modules.analysis.detectors.swing_detector import SwingDetector

class ChartPatternDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        # Reuse SMC's SwingDetector
        self.swing_detector = SwingDetector({
            'left_bars': config.get('swing_left_bars', 5),
            'right_bars': config.get('swing_right_bars', 5)
        })

    def detect(self, data: pd.DataFrame) -> List[ChartPattern]:
        # First get swing points
        swings = self.swing_detector.detect(data)

        # Then scan for chart patterns
        patterns = []
        patterns.extend(self._detect_double_tops(swings))
        patterns.extend(self._detect_double_bottoms(swings))
        patterns.extend(self._detect_head_shoulders(swings))
        # ... more pattern detectors

        return patterns
```

### 8.3 Pattern Confidence Scoring
```python
def calculate_confidence(pattern: ChartPattern) -> float:
    """
    Calculate pattern confidence score (0-100)

    Factors:
    - Symmetry (shoulders equal, neckline flat)
    - Volume pattern (decreasing volume = stronger)
    - Price tolerance (tighter = higher confidence)
    - Time ratio (balanced left/right sides)
    """
    score = 50.0  # Base score

    # Symmetry bonus
    if pattern.name == 'head_shoulders':
        shoulder_diff = abs(swings[0].price - swings[4].price) / swings[0].price
        score += (1 - shoulder_diff) * 20  # Up to +20

    # Price tolerance bonus
    # Tighter tolerance = higher confidence

    return min(score, 100.0)
```

---

## 9. Design Decisions (Resolved)

| Question | Decision | Config Key |
|----------|----------|------------|
| Swing Detection | **SwingDetector** (SMC tested) | `swing_left_bars`, `swing_right_bars` |
| Real-time Updates | Only on confirmed swings | `update_on_swing_only: true` |
| Pattern Overlap | Show all, newest on top | `allow_overlap: true` |
| Breakout Confirmation | Configurable candles + % | `breakout_candles: 2`, `breakout_pct: 0.5` |

---

## 10. References

- TradingView Chart Patterns: https://www.tradingview.com/chart-patterns/
- Bulkowski's Pattern Site: http://thepatternsite.com/
- Technical Analysis Patterns: https://school.stockcharts.com/doku.php?id=chart_analysis:chart_patterns

---

**Approval Required Before Implementation**

- [ ] Pattern list approved
- [ ] Architecture approved
- [ ] Config structure approved
- [ ] UI design approved
