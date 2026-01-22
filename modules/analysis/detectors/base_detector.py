"""
modules/analysis/detectors/base_detector.py

Abstract base class for all formation detectors
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd


class BaseDetector(ABC):
    """
    This is the base class for all detectors.

    Her detector:
    1. detect(df) - Batch analiz
    2. update(candle) - Streaming/incremental
    3. get_active() - Active (unfilled/unbroken) formations
    4. reset() - Clears the state
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Detector-specific configuration
        """
        self.config = config or {}
        self._active_formations: List[Any] = []
        self._history: List[Any] = []

    @property
    def name(self) -> str:
        """Detector name"""
        return self.__class__.__name__

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> List[Any]:
        """
        Batch detection - analysis on the entire dataset.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of detected formations
        """
        pass

    @abstractmethod
    def update(self, candle: dict, current_index: int) -> Optional[Any]:
        """
        Incremental update - update with a single candle.

        Args:
            candle: New candle data
            current_index: Current bar index

        Returns:
            New formation if detected, else None
        """
        pass

    def get_active(self) -> List[Any]:
        """
        Active (unfilled/unbroken) formations

        Returns:
            List of active formations
        """
        return self._active_formations.copy()

    def get_history(self) -> List[Any]:
        """
        All past formations (including filled/broken)

        Returns:
            List of all formations
        """
        return self._history.copy()

    def reset(self) -> None:
        """Clear state"""
        self._active_formations = []
        self._history = []

    def _add_formation(self, formation: Any, active: bool = True) -> None:
        """
        Formation ekle

        Args:
            formation: New formation
            active: True ise active listesine de ekle
        """
        self._history.append(formation)
        if active:
            self._active_formations.append(formation)

    def _deactivate_formation(self, formation_id: str) -> None:
        """
        Deactivate the formation (remove it from the active list).

        Args:
            formation_id: Formation ID
        """
        self._active_formations = [
            f for f in self._active_formations
            if f.id != formation_id
        ]
