"""
modules/analysis/detectors/liquidity_detector.py

Liquidity Zone Detector - components/indicators/structure/liquidityzones.py wrapper

Detects liquidity pools (stop-loss density).
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

from .base_detector import BaseDetector


@dataclass
class LiquidityZone:
    """Likidite zone'u"""
    type: str  # 'buy_side', 'sell_side', 'buy_side_equal', 'sell_side_equal'
    level: float
    index: int  # The bar it was created in
    strength: int  # Number of levels (1 = single level)
    swept: bool  # Was it swept?
    swept_index: Optional[int] = None  # Bar that was swept
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'level': self.level,
            'index': self.index,
            'strength': self.strength,
            'swept': self.swept,
            'swept_index': self.swept_index,
            'timestamp': self.timestamp,
        }

    def to_chart_annotation(self) -> Dict[str, Any]:
        """Annotation format for the chart"""
        is_buy_side = 'buy_side' in self.type
        return {
            'type': 'horizontal_line',
            'subtype': 'liquidity',
            'direction': 'buy_side' if is_buy_side else 'sell_side',
            'price': self.level,
            'start_index': self.index,
            'color': '#f59e0b' if is_buy_side else '#8b5cf6',  # Amber / Purple
            'style': 'dashed',
            'width': self.strength,
            'opacity': 0.2 if self.swept else 0.6,
        }


class LiquidityDetector(BaseDetector):
    """
    Liquidity Zone Detector

    components/indicators/structure/liquidityzones.py'yi kullanarak
    detects liquidity pools.

    Args:
        config: Configuration
            - left_bars: Number of bars on the left side (default: 5)
            - right_bars: Number of bars on the right side (default: 5)
            - equal_tolerance: Equal level tolerance % (default: 0.1)
            - max_zones: Maximum number of zones (default: 5)
            - sweep_lookback: Lookback period for sweep control (default: 3)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.config = config or {}

        self.left_bars = self.config.get('left_bars', 5)
        self.right_bars = self.config.get('right_bars', 5)
        self.equal_tolerance = self.config.get('equal_tolerance', 0.1)
        self.max_zones = self.config.get('max_zones', 5)
        self.sweep_lookback = self.config.get('sweep_lookback', 3)

        # Lazy import
        self._indicator = None
        self._zones: List[LiquidityZone] = []
        self._active: List[LiquidityZone] = []
        self._swept: List[LiquidityZone] = []

    def _ensure_indicator(self):
        """Lazy load indicator"""
        if self._indicator is None:
            from components.indicators.structure.liquidityzones import LiquidityZones
            self._indicator = LiquidityZones(
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                equal_tolerance=self.equal_tolerance,
                max_zones=self.max_zones,
                sweep_lookback=self.sweep_lookback
            )

    def detect(self, data: pd.DataFrame) -> List[LiquidityZone]:
        """
        Batch detection

        Args:
            data: OHLCV DataFrame

        Returns:
            List[LiquidityZone]: Detected liquidity zones.
        """
        self._ensure_indicator()
        self.reset()

        timestamps = data['timestamp'].values if 'timestamp' in data.columns else [0] * len(data)

        # Run the indicator
        result = self._indicator.calculate(data)

        # Active zones
        if result.value:
            for zone in result.value:
                liq_zone = LiquidityZone(
                    type=zone['type'],
                    level=zone['level'],
                    index=len(data) - 1,  # Approx
                    strength=zone['strength'],
                    swept=zone['swept'],
                    timestamp=int(timestamps[-1]) if len(timestamps) > 0 else None
                )
                self._zones.append(liq_zone)
                if not liq_zone.swept:
                    self._active.append(liq_zone)

        # Swept zones
        if result.metadata.get('swept_zones'):
            for swept in result.metadata['swept_zones']:
                swept_zone = LiquidityZone(
                    type=swept['type'],
                    level=swept['level'],
                    index=swept.get('swept_at', len(data) - 1),
                    strength=1,
                    swept=True,
                    swept_index=swept.get('swept_at')
                )
                self._swept.append(swept_zone)

        return self._zones

    def update(self, candle: dict, index: int) -> Optional[LiquidityZone]:
        """
        Incremental update

        Args:
            candle: Yeni mum verisi
            index: Bar index

        Returns:
            Returns the new sweep if it exists.
        """
        self._ensure_indicator()

        result = self._indicator.update(candle)

        # Is there a new sweep?
        new_sweep = None
        if result.metadata.get('swept_zones'):
            for swept in result.metadata['swept_zones']:
                # If it doesn't exist in the current sweeps, create a new one.
                existing = any(
                    s.level == swept['level'] and s.type == swept['type']
                    for s in self._swept
                )
                if not existing:
                    new_sweep = LiquidityZone(
                        type=swept['type'],
                        level=swept['level'],
                        index=index,
                        strength=1,
                        swept=True,
                        swept_index=index,
                        timestamp=candle.get('timestamp')
                    )
                    self._swept.append(new_sweep)

        # Update active zones
        self._active = []
        if result.value:
            for zone in result.value:
                if not zone['swept']:
                    liq_zone = LiquidityZone(
                        type=zone['type'],
                        level=zone['level'],
                        index=index,
                        strength=zone['strength'],
                        swept=False
                    )
                    self._active.append(liq_zone)

        return new_sweep

    def get_active(self) -> List[LiquidityZone]:
        """Returns the active (un-swept) zones."""
        return self._active

    def get_swept(self) -> List[LiquidityZone]:
        """Returns the swept zones"""
        return self._swept

    def get_history(self) -> List[LiquidityZone]:
        """Returns all zones"""
        return self._zones + self._swept

    def reset(self) -> None:
        """Reset the state"""
        self._zones = []
        self._active = []
        self._swept = []
        if self._indicator:
            self._indicator.liquidityzones = []


__all__ = ['LiquidityDetector', 'LiquidityZone']
