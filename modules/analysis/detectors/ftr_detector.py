"""
modules/analysis/detectors/ftr_detector.py

FTR (Failed to Return) / FTB (First Time Back) Detector

TradingView FTR Style:
- Bullish FTR: 3 or more consecutive bullish candles + 1 small bearish candle (FTR candle)
- Bearish FTR: 3 or more consecutive bearish candles + 1 small bullish candle (FTR candle)

FTR candle = a small pullback, indicating that the trend will continue
Zone = the body of the FTR candle (between open and close)

Zone states:
- fresh: Not yet tested
- ftb: First test (First Time Back) - strongest signal
- tested: Tested multiple times
- invalidated: Broken in the opposite direction
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .base_detector import BaseDetector
from ..models.formations import FTRZone


class FTRDetector(BaseDetector):
    """
    FTR/FTB Zone Detector - TradingView Style

    Algoritma:
    1. Find consecutive momentum candles (minimum 3 candles with the same direction)
    2. The next reverse-direction small candle = FTR candle
    3. The FTR candle must be smaller than the momentum candles
    4. Zone = FTR mumunun body'si

    Bullish FTR:
    - 3+ consecutive bullish candles -> 1 small bearish candle = FTR zone

    Bearish FTR:
    - 3+ consecutive bearish candles -> 1 small bullish candle = FTR zone

    Args:
        config: Configuration
            - min_momentum_candles: Minimum consecutive momentum candle count (default: 3)
            - max_ftr_ratio: Maximum FTR candle / momentum candle ratio (default: 0.8)
            - require_confirmation: Does the next candle require confirmation (default: True)
            - max_zones: Maximum number of active zones (default: 20)
            - invalidation_threshold: Zone invalidation threshold percentage (default: 0.5)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.config = config or {}

        self.min_momentum_candles = self.config.get('min_momentum_candles', 3)
        self.min_confirmation_candles = self.config.get('min_confirmation_candles', 1)
        self.max_ftr_ratio = self.config.get('max_ftr_ratio', 0.8)
        self.require_confirmation = self.config.get('require_confirmation', True)
        self.max_zones = self.config.get('max_zones', 20)
        self.invalidation_threshold = self.config.get('invalidation_threshold', 0.5)

        self._zones: List[FTRZone] = []

    def detect(self, data: pd.DataFrame) -> List[FTRZone]:
        """
        FTR zone tespiti

        Algoritma:
        1. Count the previous consecutive momentum candles for each bar.
        2. Yeterli momentum varsa (min 3), sonraki ters mum FTR olabilir
        3. If the FTR candle's momentum is smaller than the momentum candles = valid FTR
        4. Optional: If the next candle confirms, create a zone

        Args:
            data: OHLCV DataFrame

        Returns:
            List[FTRZone]: Detected FTR zones.
        """
        self.reset()

        if len(data) < self.min_momentum_candles + 2:
            return []

        # OHLCV verilerini al
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        timestamps = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))

        n = len(data)

        # Calculate the direction and body size of each candle.
        directions = []  # 1: bullish, -1: bearish, 0: doji
        body_sizes = []

        for i in range(n):
            body = closes[i] - opens[i]
            body_size = abs(body)
            body_sizes.append(body_size)

            if body > 0:
                directions.append(1)  # Bullish
            elif body < 0:
                directions.append(-1)  # Bearish
            else:
                directions.append(0)  # Doji

        # Detect FTR zones
        i = self.min_momentum_candles

        while i < n - 1:  # -1 because a confirmation candle is needed
            # Check for consecutive momentum of previous candles
            momentum_dir = directions[i - 1]

            if momentum_dir == 0:
                i += 1
                continue

            # How many consecutive bullish candles are there? (excluding dojis, only candles with the same direction)
            momentum_count = 0
            avg_momentum_body = 0

            for j in range(i - 1, -1, -1):
                if directions[j] == momentum_dir:
                    momentum_count += 1
                    avg_momentum_body += body_sizes[j]
                else:
                    # Different direction or doji - consecutive momentum ended
                    break

            if momentum_count < self.min_momentum_candles:
                i += 1
                continue

            avg_momentum_body /= momentum_count if momentum_count > 0 else 1

            # Is the current candle reversed? (FTR candidate)
            current_dir = directions[i]

            if current_dir == 0 or current_dir == momentum_dir:
                i += 1
                continue

            # The FTR candle body must be smaller than both the previous and the next candle (body comparison)
            ftr_body = body_sizes[i]
            prev_body = body_sizes[i - 1]

            # Check if there is a next candle
            if i + 1 >= n:
                i += 1
                continue

            next_body = body_sizes[i + 1]

            # RULE: The FTR body must be smaller than both the previous and the next candle.
            # "Confirmation candle MUST be larger than FTR candle"
            if ftr_body >= prev_body or ftr_body >= next_body:
                i += 1
                continue

            # Additional check: comparison of average momentum body with max_ftr_ratio
            # "If FTR candle is too big, invalid as Risk to Reward will be too high"
            # Note: FTR is already smaller than prev and next, this is an additional filter
            body_ratio = ftr_body / avg_momentum_body if avg_momentum_body > 0 else 1
            # If max_ftr_ratio is 0, skip this check.
            if self.max_ftr_ratio > 0 and body_ratio > self.max_ftr_ratio:
                i += 1
                continue

            # Approval check - There must be at least N confirmation blocks after the FTR block.
            # RULE: The confirmation candle's momentum must be in the same direction AND strong.
            # "Confirmation candle MUST be larger than FTR candle"

            # Check if there are enough candles.
            if i + self.min_confirmation_candles >= n:
                i += 1
                continue

            # The momentum must be in the same direction for min_confirmation_candles candles (MANDATORY)
            # This is a fundamental rule - independent of require_confirmation
            all_confirmations_valid = True
            has_strong_confirmation = True
            min_confirmation_body = avg_momentum_body * 0.5

            for conf_idx in range(1, self.min_confirmation_candles + 1):
                conf_i = i + conf_idx
                conf_dir = directions[conf_i]
                conf_body = body_sizes[conf_i]

                # Is the confirmation candle's momentum in the same direction? (MANDATORY)
                if conf_dir != momentum_dir:
                    all_confirmations_valid = False
                    has_strong_confirmation = False
                    break

                # The initial confirmation threshold should be greater than the FTR value and more robust (optional - for require_confirmation)
                if conf_idx == 1:
                    if conf_body <= ftr_body or conf_body < min_confirmation_body:
                        has_strong_confirmation = False

            # If not all confirmation candles are in the momentum direction, FTR is invalid.
            if not all_confirmations_valid:
                i += 1
                continue

            # If require_confirmation=True, strong confirmation is also required (body size check)
            if self.require_confirmation and not has_strong_confirmation:
                i += 1
                continue

            # Create FTR Zone
            # Zone = The high/low of the FTR candle (not the body)
            zone_top = highs[i]
            zone_bottom = lows[i]

            # Zone type according to momentum direction
            # Bullish momentum + bearish FTR mumu = Bullish FTR zone (destek)
            # Bearish momentum + bullish FTR candle = Bearish FTR zone (resistance)
            zone_type = 'bullish' if momentum_dir == 1 else 'bearish'

            # Strength calculation: momentum score + small FTR bonus + approval bonus
            base_strength = momentum_count * 15
            ftr_size_bonus = (1 - body_ratio) * 25
            confirmation_bonus = 20 if has_strong_confirmation else 0
            total_strength = min(100, base_strength + ftr_size_bonus + confirmation_bonus)

            zone = FTRZone(
                type=zone_type,
                top=float(zone_top),
                bottom=float(zone_bottom),
                created_time=int(timestamps[i]),
                created_index=int(i),
                source='impulse',
                source_id=f"ftr_{zone_type}_{i}",
                strength=float(total_strength),
                ftr_candle_index=int(i),
                ftr_candle_time=int(timestamps[i])
            )
            self._zones.append(zone)

            # Skip to the next potential FTR.
            i += 2

        # Update zone statuses (FTB, tested, invalidated)
        self._update_zone_status(highs, lows, closes, timestamps, n)

        # Filter active zones
        self._active_formations = [z for z in self._zones if not z.invalidated]

        # Max zone limitini uygula (en yenileri tut)
        if len(self._active_formations) > self.max_zones:
            self._active_formations = sorted(
                self._active_formations,
                key=lambda z: z.created_index,
                reverse=True
            )[:self.max_zones]

        return self._zones

    def _update_zone_status(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        n: int
    ) -> None:
        """
        Update zone statuses: FTB, tested, invalidated

        FTB rule: The price must first move away from the zone (pull away), then return.
        - Bullish zone: The price must rise above zone.top (upward movement).
        - Bearish zone: The price must fall below zone.bottom (downward movement).
        Confirmation candles are already in the momentum direction, so generally
        the pull away occurs after min_confirmation_candles candles.
        """
        for zone in self._zones:
            if zone.invalidated:
                continue

            # Check the bars after zone formation
            for i in range(zone.created_index + 1, n):
                if zone.invalidated:
                    break

                current_high = highs[i]
                current_low = lows[i]
                current_close = closes[i]
                current_time = int(timestamps[i])

                # Pull away check - Has enough time passed since the FTR?
                # After passing min_confirmation_candles candles, the FTB can be counted.
                bars_since_ftr = i - zone.created_index
                if not zone.pulled_away and bars_since_ftr > self.min_confirmation_candles:
                    zone.pulled_away = True

                # Invalidation check (always check)
                if zone.type == 'bullish':
                    # Invalidation: Did the price close below the zone?
                    invalidation_level = zone.bottom * (1 - self.invalidation_threshold / 100)
                    if current_close < invalidation_level:
                        zone.invalidated = True
                        zone.invalidated_time = current_time
                        break
                else:
                    # Invalidation: Did the price close above the zone?
                    invalidation_level = zone.top * (1 + self.invalidation_threshold / 100)
                    if current_close > invalidation_level:
                        zone.invalidated = True
                        zone.invalidated_time = current_time
                        break

                # Zone test control - only after being pulled_away
                if not zone.pulled_away:
                    continue

                zone_tested = False

                if zone.type == 'bullish':
                    # Bullish zone (support): Did it enter the close zone?
                    if current_close >= zone.bottom and current_close <= zone.top:
                        zone_tested = True
                else:
                    # Bearish zone (resistance): Did it enter the close zone?
                    if current_close >= zone.bottom and current_close <= zone.top:
                        zone_tested = True

                if zone_tested:
                    zone.test_count += 1

                    if zone.test_count == 1:
                        # First Time Back!
                        zone.status = 'ftb'
                        zone.ftb_time = current_time
                        zone.ftb_index = i
                    else:
                        # Multiple tests
                        zone.status = 'tested'

    def update(self, candle: dict, current_index: int) -> Optional[FTRZone]:
        """
        Incremental update - updates zones with a single candle.

        Args:
            candle: Yeni candle verisi
            current_index: The current bar index.

        Returns:
            If a new FTB is detected, return the zone; otherwise, return None.
        """
        current_high = candle.get('high', 0)
        current_low = candle.get('low', 0)
        current_close = candle.get('close', 0)
        current_time = candle.get('timestamp', 0)

        new_ftb = None

        for zone in self._zones:
            if zone.invalidated:
                continue

            # Pull away check - Has enough time passed since the FTR?
            # After passing min_confirmation_candles candles, the FTB can be counted.
            bars_since_ftr = current_index - zone.created_index
            if not zone.pulled_away and bars_since_ftr > self.min_confirmation_candles:
                zone.pulled_away = True

            # Invalidation check (always check)
            if zone.type == 'bullish':
                invalidation_level = zone.bottom * (1 - self.invalidation_threshold / 100)
                if current_close < invalidation_level:
                    zone.invalidated = True
                    zone.invalidated_time = current_time
                    continue
            else:
                invalidation_level = zone.top * (1 + self.invalidation_threshold / 100)
                if current_close > invalidation_level:
                    zone.invalidated = True
                    zone.invalidated_time = current_time
                    continue

            # Zone test control - only after being pulled_away
            if not zone.pulled_away:
                continue

            zone_tested = False

            # Did the object enter the zone?
            if current_close >= zone.bottom and current_close <= zone.top:
                zone_tested = True

            if zone_tested:
                zone.test_count += 1

                if zone.test_count == 1:
                    # First Time Back!
                    zone.status = 'ftb'
                    zone.ftb_time = current_time
                    zone.ftb_index = current_index
                    new_ftb = zone
                else:
                    zone.status = 'tested'

        # Update active formations
        self._active_formations = [z for z in self._zones if not z.invalidated]

        return new_ftb

    def get_fresh_zones(self) -> List[FTRZone]:
        """Retrieves zones that have not yet been tested"""
        return [z for z in self._zones if z.is_fresh and not z.invalidated]

    def get_ftb_zones(self) -> List[FTRZone]:
        """Get zones in FTB status (most powerful entry)"""
        return [z for z in self._zones if z.is_ftb and not z.invalidated]

    def get_bullish_zones(self) -> List[FTRZone]:
        """Retrieves bullish (support) zones"""
        return [z for z in self._zones if z.type == 'bullish' and not z.invalidated]

    def get_bearish_zones(self) -> List[FTRZone]:
        """Returns bearish (resistance) zones"""
        return [z for z in self._zones if z.type == 'bearish' and not z.invalidated]

    def reset(self) -> None:
        """Clear state"""
        super().reset()
        self._zones = []


__all__ = ['FTRDetector']
