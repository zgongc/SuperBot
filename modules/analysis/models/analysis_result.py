"""
modules/analysis/models/analysis_result.py

Unified analysis result container
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from .formations import (
    BOSFormation,
    CHoCHFormation,
    FVGFormation,
    SwingPoint,
    OrderBlockFormation,
    LiquidityZone,
    FTRZone
)


@dataclass
class AnalysisResult:
    """
    Analysis result for a single bar.

    Includes both new formations and active (unfilled/unbroken) formations.

    Attributes:
        timestamp: Bar timestamp (ms)
        bar_index: Bar index

        # New formations (those that occur in this container)
        new_bos: Yeni BOS (varsa)
        new_choch: Yeni CHoCH (varsa)
        new_fvg: Yeni FVG (varsa)
        new_swing: Yeni swing point (varsa)
        new_ob: Yeni order block (varsa)

        # Active formations (not yet invalidated)
        active_fvgs: List of active (not filled) FVG
        active_obs: List of active (not mitigated) OB
        active_liquidity: Active liquidity zones
        active_ftr_zones: Active FTR/FTB zones

        # FTR/FTB events
        new_ftb: Did FTB occur in this cell (first conversion to zone)?

        # Current levels
        swing_high: The most recent swing high
        swing_low: The most recent swing low

        # Piyasa durumu
        market_bias: 'bullish', 'bearish', 'neutral'
        trend: Current trend
        structure: 'hhhl', 'lllh', 'ranging'
    """
    timestamp: int
    bar_index: int

    # New formations (this bar)
    new_bos: Optional[BOSFormation] = None
    new_choch: Optional[CHoCHFormation] = None
    new_fvg: Optional[FVGFormation] = None
    new_swing: Optional[SwingPoint] = None
    new_ob: Optional[OrderBlockFormation] = None

    # Active formations
    active_fvgs: List[FVGFormation] = field(default_factory=list)
    active_obs: List[OrderBlockFormation] = field(default_factory=list)
    active_liquidity: List[LiquidityZone] = field(default_factory=list)
    active_ftr_zones: List[FTRZone] = field(default_factory=list)

    # FTR/FTB events
    new_ftb: Optional[FTRZone] = None

    # Current levels
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

    # Market state
    market_bias: Literal['bullish', 'bearish', 'neutral'] = 'neutral'
    trend: str = 'ranging'
    structure: str = 'ranging'

    def has_new_formation(self) -> bool:
        """Is there anything new in this container?"""
        return any([
            self.new_bos,
            self.new_choch,
            self.new_fvg,
            self.new_swing,
            self.new_ob,
            self.new_ftb
        ])

    def get_new_formations(self) -> List[Any]:
        """List all new creations"""
        formations = []
        if self.new_bos:
            formations.append(('bos', self.new_bos))
        if self.new_choch:
            formations.append(('choch', self.new_choch))
        if self.new_fvg:
            formations.append(('fvg', self.new_fvg))
        if self.new_swing:
            formations.append(('swing', self.new_swing))
        if self.new_ob:
            formations.append(('ob', self.new_ob))
        if self.new_ftb:
            formations.append(('ftb', self.new_ftb))
        return formations

    def to_dict(self) -> Dict[str, Any]:
        """JSON serializable dict"""
        return {
            'timestamp': self.timestamp,
            'bar_index': self.bar_index,

            # New formations
            'new_bos': self.new_bos.to_dict() if self.new_bos else None,
            'new_choch': self.new_choch.to_dict() if self.new_choch else None,
            'new_fvg': self.new_fvg.to_dict() if self.new_fvg else None,
            'new_swing': self.new_swing.to_dict() if self.new_swing else None,
            'new_ob': self.new_ob.to_dict() if self.new_ob else None,

            # Active formations
            'active_fvgs': [f.to_dict() for f in self.active_fvgs],
            'active_obs': [o.to_dict() for o in self.active_obs],
            'active_liquidity': [l.to_dict() for l in self.active_liquidity],
            'active_ftr_zones': [z.to_dict() for z in self.active_ftr_zones],

            # FTR/FTB
            'new_ftb': self.new_ftb.to_dict() if self.new_ftb else None,

            # Levels
            'swing_high': self.swing_high,
            'swing_low': self.swing_low,

            # Market state
            'market_bias': self.market_bias,
            'trend': self.trend,
            'structure': self.structure
        }

    def get_chart_annotations(self) -> List[Dict[str, Any]]:
        """
        Annotation list for LightweightCharts.

        Returns:
            [
                {'type': 'marker', ...},  # BOS, CHoCH, Swing markers
                {'type': 'fvg_box', ...},  # FVG rectangles
                {'type': 'ob_box', ...},   # OB rectangles
                {'type': 'line', ...},     # Swing levels
            ]
        """
        annotations = []

        # Markers (BOS, CHoCH, Swing)
        if self.new_bos:
            annotations.append({
                'type': 'marker',
                **self.new_bos.to_chart_annotation()
            })

        if self.new_choch:
            annotations.append({
                'type': 'marker',
                **self.new_choch.to_chart_annotation()
            })

        if self.new_swing:
            annotations.append({
                'type': 'marker',
                **self.new_swing.to_chart_annotation()
            })

        # FVG boxes
        if self.new_fvg:
            annotations.append(self.new_fvg.to_chart_box())

        # OB boxes
        if self.new_ob:
            annotations.append(self.new_ob.to_chart_box())

        # Swing levels (horizontal lines)
        if self.swing_high:
            annotations.append({
                'type': 'price_line',
                'price': self.swing_high,
                'color': '#26a69a',
                'lineWidth': 1,
                'lineStyle': 2,
                'title': 'SH'
            })

        if self.swing_low:
            annotations.append({
                'type': 'price_line',
                'price': self.swing_low,
                'color': '#ef5350',
                'lineWidth': 1,
                'lineStyle': 2,
                'title': 'SL'
            })

        return annotations

    def get_active_zones(self) -> List[Dict[str, Any]]:
        """
        Get the active zones (FVG + OB)

        Returns:
            List of zones (to be displayed on the chart)
        """
        zones = []

        for fvg in self.active_fvgs:
            zones.append({
                **fvg.to_chart_box(),
                'active': True
            })

        for ob in self.active_obs:
            zones.append({
                **ob.to_chart_box(),
                'active': True
            })

        return zones

    def summary(self) -> str:
        """Short summary string"""
        parts = []

        if self.new_bos:
            parts.append(f"BOS({self.new_bos.type})")
        if self.new_choch:
            parts.append(f"CHoCH({self.new_choch.type})")
        if self.new_fvg:
            parts.append(f"FVG({self.new_fvg.type})")
        if self.new_swing:
            parts.append(f"Swing({self.new_swing.type})")
        if self.new_ob:
            parts.append(f"OB({self.new_ob.type})")
        if self.new_ftb:
            parts.append(f"FTB({self.new_ftb.type})")

        if not parts:
            parts.append("No new formations")

        return f"[{self.bar_index}] {' | '.join(parts)} | Bias: {self.market_bias}"


@dataclass
class BatchAnalysisResult:
    """
    Batch analysis result (for all data)

    Attributes:
        results: A list of AnalysisResult objects for each bar.
        summary: Overall summary statistics.
    """
    results: List[AnalysisResult] = field(default_factory=list)

    # Summary stats (lazy calculated)
    _summary: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> AnalysisResult:
        return self.results[index]

    def __iter__(self):
        return iter(self.results)

    @property
    def summary(self) -> Dict[str, Any]:
        """Genel istatistikler"""
        if self._summary is None:
            self._summary = self._calculate_summary()
        return self._summary

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate statistics"""
        bos_count = sum(1 for r in self.results if r.new_bos)
        choch_count = sum(1 for r in self.results if r.new_choch)
        fvg_count = sum(1 for r in self.results if r.new_fvg)
        swing_count = sum(1 for r in self.results if r.new_swing)
        ob_count = sum(1 for r in self.results if r.new_ob)

        bullish_bos = sum(1 for r in self.results if r.new_bos and r.new_bos.type == 'bullish')
        bearish_bos = sum(1 for r in self.results if r.new_bos and r.new_bos.type == 'bearish')

        bullish_choch = sum(1 for r in self.results if r.new_choch and r.new_choch.type == 'bullish')
        bearish_choch = sum(1 for r in self.results if r.new_choch and r.new_choch.type == 'bearish')

        return {
            'total_bars': len(self.results),
            'bos_count': bos_count,
            'bullish_bos': bullish_bos,
            'bearish_bos': bearish_bos,
            'choch_count': choch_count,
            'bullish_choch': bullish_choch,
            'bearish_choch': bearish_choch,
            'fvg_count': fvg_count,
            'swing_count': swing_count,
            'ob_count': ob_count,
            'formation_rate': (bos_count + choch_count + fvg_count) / len(self.results) if self.results else 0
        }

    def get_all_formations(self, formation_type: str = None) -> List[Any]:
        """
        Get all formations.

        Args:
            formation_type: 'bos', 'choch', 'fvg', 'swing', 'ob' or None (all)

        Returns:
            Formation list.
        """
        formations = []

        for result in self.results:
            if formation_type is None or formation_type == 'bos':
                if result.new_bos:
                    formations.append(result.new_bos)

            if formation_type is None or formation_type == 'choch':
                if result.new_choch:
                    formations.append(result.new_choch)

            if formation_type is None or formation_type == 'fvg':
                if result.new_fvg:
                    formations.append(result.new_fvg)

            if formation_type is None or formation_type == 'swing':
                if result.new_swing:
                    formations.append(result.new_swing)

            if formation_type is None or formation_type == 'ob':
                if result.new_ob:
                    formations.append(result.new_ob)

        return formations

    def get_active_at(self, bar_index: int) -> AnalysisResult:
        """
        Get the active status for a specific bar.

        Args:
            bar_index: Bar index

        Returns:
            O anki AnalysisResult
        """
        if 0 <= bar_index < len(self.results):
            return self.results[bar_index]
        return None

    def to_dataframe(self):
        """Convert to Pandas DataFrame"""
        import pandas as pd

        data = []
        for r in self.results:
            row = {
                'timestamp': r.timestamp,
                'bar_index': r.bar_index,
                'has_bos': r.new_bos is not None,
                'bos_type': r.new_bos.type if r.new_bos else None,
                'has_choch': r.new_choch is not None,
                'choch_type': r.new_choch.type if r.new_choch else None,
                'has_fvg': r.new_fvg is not None,
                'fvg_type': r.new_fvg.type if r.new_fvg else None,
                'has_swing': r.new_swing is not None,
                'swing_type': r.new_swing.type if r.new_swing else None,
                'has_ob': r.new_ob is not None,
                'ob_type': r.new_ob.type if r.new_ob else None,
                'active_fvg_count': len(r.active_fvgs),
                'active_ob_count': len(r.active_obs),
                'swing_high': r.swing_high,
                'swing_low': r.swing_low,
                'market_bias': r.market_bias,
                'trend': r.trend
            }
            data.append(row)

        return pd.DataFrame(data)
