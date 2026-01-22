"""
modules/analysis/detectors/order_block_detector.py

Order Block Detector - components/indicators/structure/orderblocks.py wrapper

Detects order blocks and returns a format compatible with the AnalysisEngine.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

from .base_detector import BaseDetector


@dataclass
class OrderBlockFormation:
    """Order Block formation"""
    type: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    index: int  # The bar it was created in
    move_index: int  # Index of the relevant strong move
    strength: float  # Power percentage
    status: str  # 'active', 'tested', 'broken'
    test_count: int  # How many times was it tested
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'index': self.index,
            'move_index': self.move_index,
            'strength': self.strength,
            'status': self.status,
            'test_count': self.test_count,
            'timestamp': self.timestamp,
        }

    def to_chart_annotation(self) -> Dict[str, Any]:
        """Annotation format for the chart"""
        return {
            'type': 'box',
            'subtype': 'order_block',
            'direction': self.type,
            'top': self.top,
            'bottom': self.bottom,
            'start_index': self.index,
            'color': '#22c55e' if self.type == 'bullish' else '#ef4444',
            'opacity': 0.3 if self.status == 'active' else 0.1,
        }


class OrderBlockDetector(BaseDetector):
    """
    Order Block Detector

    components/indicators/structure/orderblocks.py'yi kullanarak
    Detects Order Blocks.

    Args:
        config: Configuration
            - strength_threshold: Strength movement threshold (%) (default: 1.0)
            - max_blocks: Maximum number of active blocks (default: 5)
            - lookback: Lookback period (default: 20)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.config = config or {}

        self.strength_threshold = self.config.get('strength_threshold', 1.0)
        self.max_blocks = self.config.get('max_blocks', 5)
        self.lookback = self.config.get('lookback', 20)

        # Lazy import - components/indicators'dan
        self._indicator = None
        self._formations: List[OrderBlockFormation] = []
        self._active: List[OrderBlockFormation] = []

    def _ensure_indicator(self):
        """Lazy load indicator"""
        if self._indicator is None:
            from components.indicators.structure.orderblocks import OrderBlocks
            self._indicator = OrderBlocks(
                strength_threshold=self.strength_threshold,
                max_blocks=self.max_blocks,
                lookback=self.lookback
            )

    def detect(self, data: pd.DataFrame) -> List[OrderBlockFormation]:
        """
        Batch detection - analyze all data

        Args:
            data: OHLCV DataFrame

        Returns:
            List[OrderBlockFormation]: Detected Order Blocks
        """
        self._ensure_indicator()
        self.reset()

        timestamps = data['timestamp'].values if 'timestamp' in data.columns else [0] * len(data)

        # Run the indicator
        result = self._indicator.calculate(data)

        # Convert the results
        if result.value:
            for block in result.value:
                formation = OrderBlockFormation(
                    type=block['type'],
                    top=block['top'],
                    bottom=block['bottom'],
                    index=self._indicator.active_blocks[result.value.index(block)]['index'] if self._indicator.active_blocks else len(data) - 1,
                    move_index=self._indicator.active_blocks[result.value.index(block)]['move_index'] if self._indicator.active_blocks else len(data) - 1,
                    strength=block['strength'],
                    status=block['status'],
                    test_count=block['test_count'],
                    timestamp=int(timestamps[-1]) if len(timestamps) > 0 else None
                )
                self._formations.append(formation)
                if formation.status == 'active':
                    self._active.append(formation)

        return self._formations

    def update(self, candle: dict, index: int) -> Optional[OrderBlockFormation]:
        """
        Incremental update

        Args:
            candle: Yeni mum verisi
            index: Bar index

        Returns:
            Returns the new Order Block if it exists.
        """
        self._ensure_indicator()

        result = self._indicator.update(candle)

        # Check if there is a new block
        new_block = None
        if result.value:
            for block in result.value:
                # If it doesn't exist in the current formations, create a new one.
                existing = any(
                    f.top == block['top'] and f.bottom == block['bottom']
                    for f in self._formations
                )
                if not existing:
                    new_block = OrderBlockFormation(
                        type=block['type'],
                        top=block['top'],
                        bottom=block['bottom'],
                        index=index,
                        move_index=index,
                        strength=block['strength'],
                        status=block['status'],
                        test_count=block['test_count'],
                        timestamp=candle.get('timestamp')
                    )
                    self._formations.append(new_block)
                    if new_block.status == 'active':
                        self._active.append(new_block)

        # Remove broken ones from active
        self._active = [f for f in self._active if f.status == 'active']

        return new_block

    def get_active(self) -> List[OrderBlockFormation]:
        """Returns active order blocks"""
        return self._active

    def get_history(self) -> List[OrderBlockFormation]:
        """Returns all Order Blocks"""
        return self._formations

    def reset(self) -> None:
        """Reset the state"""
        self._formations = []
        self._active = []
        if self._indicator:
            self._indicator.active_blocks = []


__all__ = ['OrderBlockDetector', 'OrderBlockFormation']
