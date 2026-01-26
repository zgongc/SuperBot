"""
modules/analysis/models/formations.py

SMC Formation Dataclasses
- BOS, CHoCH, FVG, Gap, SwingPoint, OrderBlock, ChartPattern
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List
from datetime import datetime
import uuid


@dataclass
class SwingPoint:
    """
    Swing High/Low point.

    Attributes:
        id: Unique identifier
        type: 'high' or 'low'
        price: Swing price
        time: Timestamp (ms)
        index: Bar index
        broken: Is it broken?
        broken_index: Break bar index (if any)
        broken_time: Break time (if any)
        label: HH, HL, LH, LL etiketi (TradingView ZigZag uyumlu)
    """
    type: Literal['high', 'low']
    price: float
    time: int
    index: int
    broken: bool = False
    broken_index: Optional[int] = None
    broken_time: Optional[int] = None
    label: Optional[str] = None  # HH, HL, LH, LL
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'price': self.price,
            'time': self.time,
            'index': self.index,
            'broken': self.broken,
            'broken_index': self.broken_index,
            'broken_time': self.broken_time,
            'label': self.label
        }

    def to_chart_annotation(self) -> dict:
        """LightweightCharts marker format"""
        return {
            'time': self.time // 1000,  # Unix seconds
            'position': 'aboveBar' if self.type == 'high' else 'belowBar',
            'color': '#26a69a' if self.type == 'high' else '#ef5350',
            'shape': 'arrowDown' if self.type == 'high' else 'arrowUp',
            'text': f"SH {self.price:.2f}" if self.type == 'high' else f"SL {self.price:.2f}",
            'size': 1
        }


@dataclass
class BOSFormation:
    """
    Break of Structure

    Attributes:
        id: Unique identifier
        type: 'bullish' (swing high broken) or 'bearish' (swing low broken)
        broken_level: The broken swing level
        break_price: The price at the time of the breakout (close)
        break_time: The breakout timestamp (ms)
        swing_index: The bar index of the broken swing
        break_index: The bar index of the breakout
        strength: Strength score (0-100)
    """
    type: Literal['bullish', 'bearish']
    broken_level: float
    break_price: float
    break_time: int
    swing_index: int
    break_index: int
    strength: float = 50.0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'broken_level': self.broken_level,
            'break_price': self.break_price,
            'break_time': self.break_time,
            'swing_index': self.swing_index,
            'break_index': self.break_index,
            'strength': self.strength
        }

    def to_chart_annotation(self) -> dict:
        """LightweightCharts marker format"""
        return {
            'time': self.break_time // 1000,
            'position': 'aboveBar' if self.type == 'bullish' else 'belowBar',
            'color': '#26a69a' if self.type == 'bullish' else '#ef5350',
            'shape': 'circle',
            'text': f"BOS {'↑' if self.type == 'bullish' else '↓'}",
            'size': 2
        }


@dataclass
class CHoCHFormation:
    """
    Change of Character (Trend change)

    Attributes:
        id: Unique identifier
        type: 'bullish' (downtrend broken) or 'bearish' (uptrend broken)
        previous_trend: Previous trend ('uptrend', 'downtrend')
        broken_level: Broken level
        break_price: Breakout price
        break_time: Breakout timestamp (ms)
        break_index: Breakout bar index
        significance: Significance score (0-100)
    """
    type: Literal['bullish', 'bearish']
    previous_trend: str
    broken_level: float
    break_price: float
    break_time: int
    break_index: int
    significance: float = 50.0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'previous_trend': self.previous_trend,
            'broken_level': self.broken_level,
            'break_price': self.break_price,
            'break_time': self.break_time,
            'break_index': self.break_index,
            'significance': self.significance
        }

    def to_chart_annotation(self) -> dict:
        """LightweightCharts marker format"""
        return {
            'time': self.break_time // 1000,
            'position': 'aboveBar' if self.type == 'bullish' else 'belowBar',
            'color': '#ffeb3b',  # Yellow - important signal
            'shape': 'square',
            'text': f"CHoCH {'↑' if self.type == 'bullish' else '↓'}",
            'size': 2
        }


@dataclass
class FVGFormation:
    """
    Fair Value Gap (Imbalance)

    Attributes:
        id: Unique identifier
        type: 'bullish' (gap up) or 'bearish' (gap down)
        top: Upper limit
        bottom: Lower limit
        created_time: Creation timestamp (ms)
        created_index: Creation bar index
        filled: Tamamen dolduruldu mu
        filled_percent: Fill percentage (0-100)
        filled_time: Fill time (if available)
        filled_index: Dolum bar index'i (varsa)
        age: How many bars ago it occurred
        fill_history: Fill history - [(time, level), ...] for stepped visualization
    """
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    created_index: int
    filled: bool = False
    filled_percent: float = 0.0
    filled_time: Optional[int] = None
    filled_index: Optional[int] = None
    age: int = 0
    fill_history: list = field(default_factory=list)  # [(time, level), ...]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def size(self) -> float:
        """FVG size (price difference)"""
        return self.top - self.bottom

    @property
    def size_pct(self) -> float:
        """FVG size in percentage"""
        mid = (self.top + self.bottom) / 2
        return (self.size / mid) * 100 if mid > 0 else 0

    @property
    def midpoint(self) -> float:
        """FVG midpoint"""
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'size': self.size,
            'size_pct': self.size_pct,
            'midpoint': self.midpoint,
            'created_time': self.created_time,
            'created_index': self.created_index,
            'filled': self.filled,
            'filled_percent': self.filled_percent,
            'filled_time': self.filled_time,
            'filled_index': self.filled_index,
            'age': self.age,
            'fill_history': self.fill_history  # [(time, level), ...]
        }

    def to_chart_box(self) -> dict:
        """
        Box/rectangle data for LightweightCharts.

        Not: LightweightCharts native box desteklemiyor,
        A BaselineSeries or a custom overlay must be used.
        """
        return {
            'type': 'fvg_box',
            'id': self.id,
            'fvg_type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'start_time': self.created_time // 1000,
            'color': 'rgba(38, 166, 154, 0.2)' if self.type == 'bullish' else 'rgba(239, 83, 80, 0.2)',
            'border_color': '#26a69a' if self.type == 'bullish' else '#ef5350'
        }


@dataclass
class GapFormation:
    """
    Price Gap (space between 2 candles)

    Unlike FVG, it does not require 3 candle patterns.
    It only detects the price gap between two consecutive candles.

    Attributes:
        id: Unique identifier
        type: 'bullish' (gap up) or 'bearish' (gap down)
        top: Upper limit
        bottom: Lower limit
        created_time: Creation timestamp (ms)
        created_index: Creation bar index
        filled: Dolduruldu mu
        filled_time: Filling time (if available)
        filled_index: Dolum bar index'i (varsa)
    """
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    created_index: int
    filled: bool = False
    filled_time: Optional[int] = None
    filled_index: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def size(self) -> float:
        """Gap size (price difference)"""
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """Gap midpoint"""
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'size': self.size,
            'midpoint': self.midpoint,
            'created_time': self.created_time,
            'created_index': self.created_index,
            'filled': self.filled,
            'filled_time': self.filled_time,
            'filled_index': self.filled_index
        }

    def to_chart_box(self) -> dict:
        """
        Box data for LightweightCharts.

        - Full box: The gap has not yet been filled
        - Just the frame: The gap has been filled
        """
        # If not filled, it's a filled box; if filled, only the border.
        if self.filled:
            # Only frame (transparent fill)
            fill_color = 'rgba(0, 0, 0, 0)'
            border_width = 2
        else:
            # Dolu kutu
            fill_color = 'rgba(38, 166, 154, 0.25)' if self.type == 'bullish' else 'rgba(239, 83, 80, 0.25)'
            border_width = 1

        border_color = '#26a69a' if self.type == 'bullish' else '#ef5350'

        return {
            'type': 'gap_box',
            'id': self.id,
            'gap_type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'start_time': self.created_time // 1000,
            'filled': self.filled,
            'filled_time': self.filled_time // 1000 if self.filled_time else None,
            'color': fill_color,
            'border_color': border_color,
            'border_width': border_width
        }


@dataclass
class OrderBlockFormation:
    """
    Order Block (Corporate order region)

    Attributes:
        id: Unique identifier
        type: 'bullish' or 'bearish'
        top: Upper limit (high of OB candle)
        bottom: Lower limit (low of OB candle)
        created_time: Creation timestamp
        created_index: Creation bar index
        mitigated: Was it visited (price entered the zone)
        mitigated_time: Mitigation time
        strength: Strength score (size of impulse move)
        impulse_size: Size of the impulsive move
    """
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    created_index: int
    mitigated: bool = False
    mitigated_time: Optional[int] = None
    strength: float = 50.0
    impulse_size: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def size(self) -> float:
        """OB size"""
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """OB midpoint (50% level)"""
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'size': self.size,
            'midpoint': self.midpoint,
            'created_time': self.created_time,
            'created_index': self.created_index,
            'mitigated': self.mitigated,
            'mitigated_time': self.mitigated_time,
            'strength': self.strength,
            'impulse_size': self.impulse_size
        }

    def to_chart_box(self) -> dict:
        """Box data for LightweightCharts"""
        return {
            'type': 'ob_box',
            'id': self.id,
            'ob_type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'start_time': self.created_time // 1000,
            'color': 'rgba(33, 150, 243, 0.2)' if self.type == 'bullish' else 'rgba(156, 39, 176, 0.2)',
            'border_color': '#2196f3' if self.type == 'bullish' else '#9c27b0'
        }


@dataclass
class LiquidityZone:
    """
    Liquidity Zone (Equal highs/lows, swing clusters)

    Attributes:
        id: Unique identifier
        type: 'buy_side' (above EQH) or 'sell_side' (below EQL)
        price: Zone price
        touch_count: How many times it was tested
        swept: Was liquidity taken?
        swept_time: Sweep time
    """
    type: Literal['buy_side', 'sell_side']
    price: float
    created_time: int
    touch_count: int = 1
    swept: bool = False
    swept_time: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'price': self.price,
            'created_time': self.created_time,
            'touch_count': self.touch_count,
            'swept': self.swept,
            'swept_time': self.swept_time
        }

    def to_chart_line(self) -> dict:
        """Price line data for LightweightCharts"""
        return {
            'type': 'liquidity_line',
            'id': self.id,
            'price': self.price,
            'color': '#ff9800',  # Turuncu
            'lineWidth': 1,
            'lineStyle': 2,  # Dashed
            'title': f"LIQ {'BSL' if self.type == 'buy_side' else 'SSL'}"
        }


@dataclass
class FTRZone:
    """
    FTR (Failed to Return) / FTB (First Time Back) Zone

    SMC Concepts:
    - FTR: The price continued without returning after leaving the price zone
           (Zone is strong, there is momentum)
    - FTB: The first return of the price to the zone (The strongest entry point)

    Zone Resources:
    - Order Block origin
    - CHoCH/BOS origin
    - Supply/Demand zone

    Attributes:
        id: Unique identifier
        type: 'bullish' (demand zone) or 'bearish' (supply zone)
        top: Upper limit
        bottom: Lower limit
        created_time: Zone creation time (ms)
        created_index: Zone creation bar index
        source: Zone source ('ob', 'choch', 'bos', 'swing')
        source_id: Kaynak formation ID'si
        status: 'fresh' (not yet tested), 'ftb' (initial test), 'tested' (tested multiple times)
        test_count: How many times it was tested
        ftb_time: Initial test time (FTB)
        ftb_index: Initial test bar index
        invalidated: Is the zone invalid (broken in the opposite direction)?
        invalidated_time: Invalidity time
        strength: Zone strength (0-100)
    """
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    created_time: int
    created_index: int
    source: Literal['ob', 'choch', 'bos', 'swing', 'impulse', 'imbalance'] = 'impulse'
    source_id: Optional[str] = None
    status: Literal['fresh', 'ftb', 'tested'] = 'fresh'
    test_count: int = 0
    ftb_time: Optional[int] = None
    ftb_index: Optional[int] = None
    invalidated: bool = False
    invalidated_time: Optional[int] = None
    strength: float = 50.0
    ftr_candle_index: Optional[int] = None  # Index of the FTR candle (for the orange marker)
    ftr_candle_time: Optional[int] = None   # FTR mumunun timestamp'i
    pulled_away: bool = False  # Did the price move away from the zone? (Required for FTB)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def size(self) -> float:
        """Zone size"""
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """Zone midpoint (optimal entry)"""
        return (self.top + self.bottom) / 2

    @property
    def is_ftb(self) -> bool:
        """First Time Back mi?"""
        return self.status == 'ftb' and self.test_count == 1

    @property
    def is_fresh(self) -> bool:
        """Has it not been tested yet?"""
        return self.status == 'fresh' and self.test_count == 0

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'top': float(self.top),
            'bottom': float(self.bottom),
            'size': float(self.size),
            'midpoint': float(self.midpoint),
            'created_time': int(self.created_time),
            'created_index': int(self.created_index),
            'source': self.source,
            'source_id': self.source_id,
            'status': self.status,
            'test_count': int(self.test_count),
            'ftb_time': int(self.ftb_time) if self.ftb_time is not None else None,
            'ftb_index': int(self.ftb_index) if self.ftb_index is not None else None,
            'is_ftb': self.is_ftb,
            'is_fresh': self.is_fresh,
            'invalidated': self.invalidated,
            'invalidated_time': int(self.invalidated_time) if self.invalidated_time is not None else None,
            'strength': float(self.strength),
            'ftr_candle_index': int(self.ftr_candle_index) if self.ftr_candle_index is not None else None,
            'ftr_candle_time': int(self.ftr_candle_time) if self.ftr_candle_time is not None else None
        }

    def to_chart_box(self) -> dict:
        """Box data for LightweightCharts"""
        # Renk: Fresh=parlak, FTB=normal, Tested=soluk
        if self.type == 'bullish':
            if self.is_fresh:
                color = 'rgba(76, 175, 80, 0.3)'  # Green - fresh
                border = '#4caf50'
            elif self.is_ftb:
                color = 'rgba(76, 175, 80, 0.2)'  # Green - ftb
                border = '#81c784'
            else:
                color = 'rgba(76, 175, 80, 0.1)'  # Green - tested
                border = '#a5d6a7'
        else:
            if self.is_fresh:
                color = 'rgba(244, 67, 54, 0.3)'  # Red - fresh
                border = '#f44336'
            elif self.is_ftb:
                color = 'rgba(244, 67, 54, 0.2)'  # Red - ftb
                border = '#e57373'
            else:
                color = 'rgba(244, 67, 54, 0.1)'  # Red - tested
                border = '#ef9a9a'

        return {
            'type': 'ftr_box',
            'id': self.id,
            'ftr_type': self.type,
            'status': self.status,
            'top': float(self.top),
            'bottom': float(self.bottom),
            'start_time': int(self.created_time) // 1000,
            'end_time': int(self.ftb_time) // 1000 if self.ftb_time else None,
            'color': color,
            'border_color': border,
            'label': 'FTR' if self.is_fresh else ('FTB' if self.is_ftb else 'Zone')
        }


@dataclass
class ChartPattern:
    """
    Geometric Chart Pattern (Double Top, Head & Shoulders, Triangles, etc.)

    Attributes:
        id: Unique identifier
        name: Pattern code (e.g., 'double_top', 'head_shoulders')
        display_name: Human readable name
        type: 'bullish', 'bearish', 'neutral'
        status: 'forming', 'completed', 'confirmed', 'failed'
        swings: List of swing points forming the pattern
        start_time: Pattern start timestamp (ms)
        end_time: Pattern completion timestamp (ms)
        start_index: Start bar index
        end_index: End bar index
        neckline: Neckline price (for H&S patterns)
        target: Target price projection
        confidence: Pattern confidence score (0-100)
        breakout_price: Breakout level
        breakout_confirmed: Was breakout confirmed?
    """
    name: str
    display_name: str
    type: Literal['bullish', 'bearish', 'neutral']
    status: Literal['forming', 'completed', 'confirmed', 'failed']
    swings: List['SwingPoint']
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

    @property
    def duration_bars(self) -> int:
        """Pattern duration in bars"""
        return self.end_index - self.start_index

    @property
    def price_range(self) -> float:
        """Pattern price range (high - low of all swings)"""
        if not self.swings:
            return 0.0
        prices = [s.price for s in self.swings]
        return max(prices) - min(prices)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'type': self.type,
            'status': self.status,
            'swings': [s.to_dict() for s in self.swings],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'duration_bars': self.duration_bars,
            'neckline': self.neckline,
            'target': self.target,
            'confidence': self.confidence,
            'breakout_price': self.breakout_price,
            'breakout_confirmed': self.breakout_confirmed
        }

    def to_chart_annotation(self) -> dict:
        """
        Chart annotation data for LightweightCharts

        Returns lines connecting swing points and pattern label
        """
        colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#9e9e9e'
        }
        color = colors.get(self.type, '#9e9e9e')

        # Build line segments connecting swings
        lines = []
        for i in range(len(self.swings) - 1):
            s1, s2 = self.swings[i], self.swings[i + 1]
            lines.append({
                'type': 'pattern_line',
                'start_time': s1.time // 1000,
                'start_price': s1.price,
                'end_time': s2.time // 1000,
                'end_price': s2.price,
                'color': color,
                'lineWidth': 2,
                'lineStyle': 0  # Solid
            })

        # Add neckline if exists
        if self.neckline and len(self.swings) >= 2:
            lines.append({
                'type': 'neckline',
                'start_time': self.swings[0].time // 1000,
                'start_price': self.neckline,
                'end_time': self.swings[-1].time // 1000,
                'end_price': self.neckline,
                'color': color,
                'lineWidth': 1,
                'lineStyle': 2  # Dashed
            })

        # Add target line if exists
        if self.target:
            lines.append({
                'type': 'target_line',
                'price': self.target,
                'start_time': self.end_time // 1000,
                'color': color,
                'lineWidth': 1,
                'lineStyle': 1  # Dotted
            })

        return {
            'pattern_id': self.id,
            'pattern_name': self.display_name,
            'pattern_type': self.type,
            'lines': lines,
            'label': {
                'time': self.end_time // 1000,
                'price': self.swings[-1].price if self.swings else 0,
                'text': self.display_name,
                'color': color
            }
        }
