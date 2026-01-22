"""
modules/analysis/analysis_engine.py

Market Structure Analysis Engine - Main Orchestrator

It coordinates all detectors and produces unified results.
"""

from typing import List, Optional, Dict, Any, Literal
import pandas as pd
from dataclasses import dataclass

from .detectors.swing_detector import SwingDetector
from .detectors.structure_detector import StructureDetector
from .detectors.fvg_detector import FVGDetector
from .detectors.gap_detector import GapDetector
from .detectors.pattern_detector import PatternDetector, CandlePattern
from .detectors.order_block_detector import OrderBlockDetector, OrderBlockFormation
from .detectors.liquidity_detector import LiquidityDetector, LiquidityZone
from .detectors.qml_detector import QMLDetector, QMLFormation
from .detectors.ftr_detector import FTRDetector
from .models.formations import (
    BOSFormation,
    CHoCHFormation,
    FVGFormation,
    GapFormation,
    SwingPoint,
    FTRZone,
)
from .models.analysis_result import AnalysisResult, BatchAnalysisResult


class AnalysisEngine:
    """
    Market Structure Analysis Engine

    Detects SMC formations from the given candle data:
    - Swing High/Low
    - BOS (Break of Structure)
    - CHoCH (Change of Character)
    - FVG (Fair Value Gap)
    - Candlestick Patterns

    Usage:
        engine = AnalysisEngine()

        # Batch analiz
        result = engine.analyze(df)

        # Streaming analiz
        engine.warmup(historical_df)
        for candle in new_candles:
            result = engine.update(candle)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize AnalysisEngine

        Args:
            config: Configuration dict
                - swing.left_bars: Swing detection left bars (default: 5)
                - swing.right_bars: Swing detection right bars (default: 5)
                - structure.max_levels: Max swing levels to track (default: 5)
                - structure.trend_strength: Trend detection strength (default: 2)
                - fvg.min_size_pct: Min FVG size % (default: 0.1)
                - fvg.max_age: Max FVG age in bars (default: 50)
                - patterns.enabled: Enable pattern detection (default: True)
                - patterns.use_talib: Use TALib patterns (default: False)
                - orderblocks.enabled: Enable Order Block detection (default: True)
                - orderblocks.strength_threshold: OB strength threshold (default: 1.0)
                - liquidity.enabled: Enable Liquidity Zone detection (default: True)
                - qml.enabled: Enable QML detection (default: True)
        """
        self.config = config or {}

        # Extract configs
        swing_config = {
            'left_bars': self.config.get('swing', {}).get('left_bars', 5),
            'right_bars': self.config.get('swing', {}).get('right_bars', 5),
            'max_levels': self.config.get('swing', {}).get('max_levels', 10),
        }

        structure_config = {
            **swing_config,
            'max_levels': self.config.get('structure', {}).get('max_levels', 5),
            'trend_strength': self.config.get('structure', {}).get('trend_strength', 2),
        }

        fvg_config = {
            'min_size_pct': self.config.get('fvg', {}).get('min_size_pct', 0.1),
            'max_age': self.config.get('fvg', {}).get('max_age', 50),
        }

        pattern_config = {
            'use_talib': self.config.get('patterns', {}).get('use_talib', False),
        }

        order_block_config = {
            'strength_threshold': self.config.get('order_blocks', {}).get('strength_threshold', 1.0),
            'max_blocks': self.config.get('order_blocks', {}).get('max_blocks', 5),
            'lookback': self.config.get('order_blocks', {}).get('lookback', 20),
        }

        liquidity_config = {
            **swing_config,
            'equal_tolerance': self.config.get('liquidity', {}).get('equal_tolerance', 0.1),
            'max_zones': self.config.get('liquidity', {}).get('max_zones', 5),
            'sweep_lookback': self.config.get('liquidity', {}).get('sweep_lookback', 3),
        }

        qml_config = {
            # QML uses its own swing parameters (larger swings for better patterns)
            'left_bars': self.config.get('qml', {}).get('left_bars', swing_config['left_bars']),
            'right_bars': self.config.get('qml', {}).get('right_bars', swing_config['right_bars']),
            'lookback_bars': self.config.get('qml', {}).get('lookback_bars', 30),
            'break_threshold': self.config.get('qml', {}).get('break_threshold', 0.1),
        }

        # Initialize core detectors
        self._swing_detector = SwingDetector(swing_config)
        self._structure_detector = StructureDetector(structure_config)
        self._fvg_detector = FVGDetector(fvg_config)

        # Pattern detector (optional)
        self._pattern_detector = None
        if self.config.get('patterns', {}).get('enabled', True):
            self._pattern_detector = PatternDetector(pattern_config)

        # Order Block detector (optional)
        self._order_block_detector = None
        if self.config.get('order_blocks', {}).get('enabled', True):
            self._order_block_detector = OrderBlockDetector(order_block_config)

        # Liquidity detector (optional)
        self._liquidity_detector = None
        if self.config.get('liquidity', {}).get('enabled', True):
            self._liquidity_detector = LiquidityDetector(liquidity_config)

        # QML detector (optional)
        self._qml_detector = None
        if self.config.get('qml', {}).get('enabled', True):
            self._qml_detector = QMLDetector(qml_config)

        # FTR/FTB detector (optional) - Momentum candles + opposite candle
        ftr_config = {
            'min_momentum_candles': self.config.get('ftr', {}).get('min_momentum_candles', 3),
            'min_confirmation_candles': self.config.get('ftr', {}).get('min_confirmation_candles', 1),
            'max_ftr_ratio': self.config.get('ftr', {}).get('max_ftr_ratio', 0.3),
            'require_confirmation': self.config.get('ftr', {}).get('require_confirmation', True),
            'max_zones': self.config.get('ftr', {}).get('max_zones', 20),
            'invalidation_threshold': self.config.get('ftr', {}).get('invalidation_threshold', 0.5),
        }
        self._ftr_detector = None
        if self.config.get('ftr', {}).get('enabled', True):
            self._ftr_detector = FTRDetector(ftr_config)

        # Gap detector (optional) - space between 2 candles
        gap_config = {
            'min_size_pct': self.config.get('gap', {}).get('min_size_pct', 0.05),
            'max_age': self.config.get('gap', {}).get('max_age', 500),
            'max_zones': self.config.get('gap', {}).get('max_zones', 100),
        }
        self._gap_detector = None
        if self.config.get('gap', {}).get('enabled', True):
            self._gap_detector = GapDetector(gap_config)

        # State
        self._initialized = False
        self._current_index = 0
        self._results: List[AnalysisResult] = []

    def analyze(self, data: pd.DataFrame) -> BatchAnalysisResult:
        """
        Batch analysis - analyze all data

        Args:
            data: OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)

        Returns:
            BatchAnalysisResult with all formations
        """
        self.reset()

        n = len(data)
        times = data['timestamp'].values if 'timestamp' in data.columns else list(range(n))
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # 1. Detect all formations
        swings = self._swing_detector.detect(data)
        structures = self._structure_detector.detect(data)
        fvgs = self._fvg_detector.detect(data)

        patterns = []
        if self._pattern_detector:
            patterns = self._pattern_detector.detect(data)

        # Detect optional formations
        # Order Blocks: Use CHoCH-based detection (SMC correct approach)
        # OB is the last opposite candle before the impulsive move that caused CHoCH/BOS
        orderblocks = []
        if self._order_block_detector:
            # Create OBs from CHoCH/BOS structure breaks
            orderblocks = self._create_orderblocks_from_structure(
                structures, data, times
            )

        liquidityzones = []
        if self._liquidity_detector:
            liquidityzones = self._liquidity_detector.detect(data)

        qml_patterns = []
        if self._qml_detector:
            qml_patterns = self._qml_detector.detect(data)

        # FTR/FTB zones: ATR-based impulse detection (TradingView FTR style)
        # Detect impulse candles and create the FTR zone
        ftr_zones = []
        if self._ftr_detector:
            ftr_zones = self._ftr_detector.detect(data)

        # Gap detection - space between 2 candles
        gaps = []
        if self._gap_detector:
            gaps = self._gap_detector.detect(data)

        # 2. Build index maps
        swing_map: Dict[int, SwingPoint] = {s.index: s for s in swings}
        bos_map: Dict[int, BOSFormation] = {}
        choch_map: Dict[int, CHoCHFormation] = {}
        fvg_map: Dict[int, FVGFormation] = {}
        pattern_map: Dict[int, List[CandlePattern]] = {}

        for f in structures:
            if isinstance(f, BOSFormation):
                bos_map[f.break_index] = f
            elif isinstance(f, CHoCHFormation):
                choch_map[f.break_index] = f

        for f in fvgs:
            fvg_map[f.created_index] = f

        for p in patterns:
            if p.index not in pattern_map:
                pattern_map[p.index] = []
            pattern_map[p.index].append(p)

        # Build order block index map
        ob_map: Dict[int, OrderBlockFormation] = {}
        for ob in orderblocks:
            ob_map[ob.index] = ob

        # Build liquidity zone index map
        liq_map: Dict[int, LiquidityZone] = {}
        for lz in liquidityzones:
            liq_map[lz.index] = lz

        # Build QML index map
        qml_map: Dict[int, QMLFormation] = {}
        for qml in qml_patterns:
            qml_map[qml.index] = qml

        # Build FTR zone maps (by creation and FTB index)
        ftr_created_map: Dict[int, FTRZone] = {}
        ftr_ftb_map: Dict[int, FTRZone] = {}
        for ftr in ftr_zones:
            ftr_created_map[ftr.created_index] = ftr
            if ftr.ftb_index is not None:
                ftr_ftb_map[ftr.ftb_index] = ftr

        # Build Gap index map
        gap_map: Dict[int, GapFormation] = {}
        for gap in gaps:
            gap_map[gap.created_index] = gap

        # 3. Build per-bar results
        results = []
        current_swing_high = None
        current_swing_low = None

        for i in range(n):
            # New formations at this bar
            new_swing = swing_map.get(i)
            new_bos = bos_map.get(i)
            new_choch = choch_map.get(i)
            new_fvg = fvg_map.get(i)
            new_ob = ob_map.get(i)
            new_liq = liq_map.get(i)
            new_qml = qml_map.get(i)

            # Update current swing levels
            if new_swing:
                if new_swing.type == 'high':
                    current_swing_high = new_swing.price
                else:
                    current_swing_low = new_swing.price

            # Active formations at this bar
            active_fvgs = [f for f in self._fvg_detector.get_active()
                          if f.created_index <= i and not f.filled]

            # Update OB status and get active ones
            active_obs = []
            if self._order_block_detector:
                current_close = closes[i]
                current_high = highs[i]
                current_low = lows[i]

                for ob in self._order_block_detector._formations:
                    if ob.index > i:
                        continue  # OB not yet created

                    # Check if OB is broken
                    if ob.status == 'active':
                        if ob.type == 'bullish' and current_close < ob.bottom:
                            ob.status = 'broken'
                        elif ob.type == 'bearish' and current_close > ob.top:
                            ob.status = 'broken'
                        # Check if price tested the OB
                        elif current_low <= ob.top and current_high >= ob.bottom:
                            ob.test_count += 1

                    if ob.status == 'active' and ob.index <= i:
                        active_obs.append(ob)

                # Update detector's active list
                self._order_block_detector._active = [
                    ob for ob in self._order_block_detector._formations
                    if ob.status == 'active'
                ]

            # Active FTR zones and FTB events
            active_ftr_zones = []
            new_ftb = None
            if self._ftr_detector:
                # Get active (non-invalidated) zones that exist at this bar
                active_ftr_zones = [
                    z for z in self._ftr_detector._zones
                    if z.created_index <= i and not z.invalidated
                ]
                # Check if FTB happened at this bar
                new_ftb = ftr_ftb_map.get(i)

            # Market bias
            market_bias = self._determine_bias(new_bos, new_choch, new_fvg)
            trend = self._structure_detector.get_current_trend()

            result = AnalysisResult(
                timestamp=int(times[i]),
                bar_index=i,
                new_bos=new_bos,
                new_choch=new_choch,
                new_fvg=new_fvg,
                new_swing=new_swing,
                new_ob=new_ob,
                active_fvgs=active_fvgs,
                active_obs=active_obs,
                active_ftr_zones=active_ftr_zones,
                new_ftb=new_ftb,
                swing_high=current_swing_high,
                swing_low=current_swing_low,
                market_bias=market_bias,
                trend=trend,
                structure=self._determine_structure(swings, i)
            )

            results.append(result)

        self._results = results
        self._initialized = True

        return BatchAnalysisResult(results=results)

    def update(self, candle: dict) -> AnalysisResult:
        """
        Incremental update - update with a single candle.

        Args:
            candle: New candle dict (timestamp, open, high, low, close, volume)

        Returns:
            AnalysisResult for this bar
        """
        self._current_index += 1
        time = candle.get('timestamp', candle.get('t', self._current_index))
        close = candle.get('close', candle.get('c', 0))

        # Update detectors
        new_swing = self._swing_detector.update(candle, self._current_index)
        new_structure = self._structure_detector.update(candle, self._current_index)
        new_fvg = self._fvg_detector.update(candle, self._current_index)

        new_patterns = []
        if self._pattern_detector:
            new_patterns = self._pattern_detector.update(candle, self._current_index)

        # Determine new BOS/CHoCH
        new_bos = None
        new_choch = None
        if new_structure:
            if isinstance(new_structure, BOSFormation):
                new_bos = new_structure
            elif isinstance(new_structure, CHoCHFormation):
                new_choch = new_structure

        # Current levels
        swing_high = self._swing_detector.get_current_swing_high()
        swing_low = self._swing_detector.get_current_swing_low()

        # Active formations
        active_fvgs = self._fvg_detector.get_active()
        active_obs = []

        # Market bias
        market_bias = self._determine_bias(new_bos, new_choch, new_fvg)
        trend = self._structure_detector.get_current_trend()

        result = AnalysisResult(
            timestamp=int(time),
            bar_index=self._current_index,
            new_bos=new_bos,
            new_choch=new_choch,
            new_fvg=new_fvg,
            new_swing=new_swing,
            new_ob=None,
            active_fvgs=active_fvgs,
            active_obs=active_obs,
            swing_high=swing_high,
            swing_low=swing_low,
            market_bias=market_bias,
            trend=trend,
            structure='ranging'
        )

        self._results.append(result)
        return result

    def warmup(self, data: pd.DataFrame) -> None:
        """
        Warmup with historical data (for streaming mode)

        Args:
            data: Historical OHLCV DataFrame
        """
        self.analyze(data)
        self._current_index = len(data) - 1

    def _create_orderblocks_from_structure(
        self,
        structures: List[Any],
        data: pd.DataFrame,
        times: Any
    ) -> List[OrderBlockFormation]:
        """
        Create Order Blocks from CHoCH (MSB) structure breaks.

        TradingView MSB-OB Logic (EmreKb indicator):
        - MSB = Market Structure Break (l0 < l1 or h0 > h1)
        - Bullish OB: When MSB is upwards, between the previous high (h1) and the last low (l0),
                      the last RED (bearish) candle.
        - Bearish OB: When MSB is downwards, between the previous low (l1) and the last high (h0),
                      the last GREEN (bullish) candle.

        Logic:
        - When CHoCH is detected, an OB is searched within the range between swing points.
        """
        orderblocks = []
        max_lookback = self.config.get('order_blocks', {}).get('lookback', 30)
        max_blocks = self.config.get('order_blocks', {}).get('max_blocks', 10)

        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        times = data['timestamp'].values if 'timestamp' in data.columns else [0] * len(data)

        # Get the swings - to determine the OB search range.
        swings = self._swing_detector.get_history()

        # Sort swings based on index
        sorted_swings = sorted(swings, key=lambda s: s.index)

        for structure in structures:
            # Only process CHoCH - OB is formed at trend reversal points (MSB)
            if not isinstance(structure, CHoCHFormation):
                continue

            break_index = structure.break_index
            is_bullish = structure.type == 'bullish'

            ob_candle_index = None

            # Find the swing points before CHoCH (TradingView logic)
            # Bullish MSB: intermediate between h1 -> l0 (from previous high to current low)
            # Bearish MSB: search for a valley (from previous low to last high) between l1 and h0.

            # Find the last 2 opposite swings before CHoCH.
            prev_swings = [s for s in sorted_swings if s.index < break_index]

            if is_bullish:
                # Bullish CHoCH: between the previous high (h1) and the previous low (l0)
                # Find the last low
                recent_lows = [s for s in prev_swings if s.type == 'low']
                recent_highs = [s for s in prev_swings if s.type == 'high']

                if recent_lows and recent_highs:
                    l0 = recent_lows[-1]  # Last low
                    # Find the high value before index l0.
                    h1_candidates = [s for s in recent_highs if s.index < l0.index]
                    if h1_candidates:
                        h1 = h1_candidates[-1]  # last high before l0
                        search_start = h1.index
                        search_end = l0.index
                    else:
                        search_start = max(0, break_index - max_lookback)
                        search_end = break_index - 1
                else:
                    search_start = max(0, break_index - max_lookback)
                    search_end = break_index - 1
            else:
                # Bearish CHoCH: between the previous low (l1) and the previous high (h0)
                recent_highs = [s for s in prev_swings if s.type == 'high']
                recent_lows = [s for s in prev_swings if s.type == 'low']

                if recent_highs and recent_lows:
                    h0 = recent_highs[-1]  # Last high
                    # Find the low value before h0
                    l1_candidates = [s for s in recent_lows if s.index < h0.index]
                    if l1_candidates:
                        l1 = l1_candidates[-1]  # The last low before h0
                        search_start = l1.index
                        search_end = h0.index
                    else:
                        search_start = max(0, break_index - max_lookback)
                        search_end = break_index - 1
                else:
                    search_start = max(0, break_index - max_lookback)
                    search_end = break_index - 1

            # Search for a candlestick with an inverse color within the specified range.
            for i in range(search_end, search_start - 1, -1):
                candle_is_bullish = closes[i] > opens[i]
                candle_is_bearish = closes[i] < opens[i]

                if is_bullish and candle_is_bearish:
                    # Bullish OB = last red candle
                    ob_candle_index = i
                    break
                elif not is_bullish and candle_is_bullish:
                    # Bearish OB = last green candle
                    ob_candle_index = i
                    break

            # Fallback: If a reversed color candle is not found, take the candle before CHoCH.
            if ob_candle_index is None:
                ob_candle_index = break_index - 1 if break_index > 0 else 0

            if ob_candle_index is not None:
                # Calculate strength (how far price moved after OB)
                if is_bullish:
                    move = (closes[break_index] - lows[ob_candle_index]) / lows[ob_candle_index] * 100
                else:
                    move = (highs[ob_candle_index] - closes[break_index]) / highs[ob_candle_index] * 100

                ob_top = float(highs[ob_candle_index])
                ob_bottom = float(lows[ob_candle_index])

                # Determine the OB status - did the price break the OB?
                ob_status = 'active'
                test_count = 0
                for j in range(break_index, len(closes)):
                    if is_bullish:
                        # Bullish Order Block: if the price falls below the bottom, it's broken.
                        if closes[j] < ob_bottom:
                            ob_status = 'broken'
                            break
                        # Test: price entered the zone
                        if lows[j] <= ob_top and lows[j] >= ob_bottom:
                            test_count += 1
                    else:
                        # Bearish OB: if the price goes above the top, it's broken.
                        if closes[j] > ob_top:
                            ob_status = 'broken'
                            break
                        # Test: price entered the zone
                        if highs[j] >= ob_bottom and highs[j] <= ob_top:
                            test_count += 1

                ob = OrderBlockFormation(
                    type='bullish' if is_bullish else 'bearish',
                    top=ob_top,
                    bottom=ob_bottom,
                    index=ob_candle_index,
                    move_index=break_index,
                    strength=float(abs(move)),
                    status=ob_status,
                    test_count=test_count,
                    timestamp=int(times[ob_candle_index]) if ob_candle_index < len(times) else None
                )

                # Avoid duplicates (same index AND same type)
                if not any(existing.index == ob.index and existing.type == ob.type for existing in orderblocks):
                    orderblocks.append(ob)

        # Store in detector for get_active/get_history queries
        if self._order_block_detector:
            self._order_block_detector._formations = orderblocks
            self._order_block_detector._active = [ob for ob in orderblocks if ob.status == 'active']

        # Limit to max_blocks (keep most recent)
        if len(orderblocks) > max_blocks:
            orderblocks = orderblocks[-max_blocks:]

        return orderblocks

    def _determine_bias(
        self,
        bos: Optional[BOSFormation],
        choch: Optional[CHoCHFormation],
        fvg: Optional[FVGFormation]
    ) -> Literal['bullish', 'bearish', 'neutral']:
        """Determine market bias"""
        # CHoCH is the strongest signal
        if choch:
            return choch.type

        # BOS ikinci
        if bos:
            return bos.type

        # FVG is the weakest
        if fvg:
            return fvg.type

        return 'neutral'

    def _determine_structure(self, swings: List[SwingPoint], current_index: int) -> str:
        """
        Determine market structure

        Returns:
            'hhhl' (Higher Highs Higher Lows - uptrend)
            'lllh' (Lower Lows Lower Highs - downtrend)
            'ranging'
        """
        # Get the last swings
        recent_highs = [s for s in swings if s.type == 'high' and s.index <= current_index][-3:]
        recent_lows = [s for s in swings if s.type == 'low' and s.index <= current_index][-3:]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return 'ranging'

        # Higher Highs check
        hh = all(recent_highs[i].price > recent_highs[i-1].price for i in range(1, len(recent_highs)))
        # Higher Lows check
        hl = all(recent_lows[i].price > recent_lows[i-1].price for i in range(1, len(recent_lows)))

        # Lower Lows check
        ll = all(recent_lows[i].price < recent_lows[i-1].price for i in range(1, len(recent_lows)))
        # Lower Highs check
        lh = all(recent_highs[i].price < recent_highs[i-1].price for i in range(1, len(recent_highs)))

        if hh and hl:
            return 'hhhl'
        elif ll and lh:
            return 'lllh'
        else:
            return 'ranging'

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_formations(
        self,
        formation_type: str = None,
        active_only: bool = False
    ) -> List[Any]:
        """
        Get formations by type

        Args:
            formation_type: 'bos', 'choch', 'fvg', 'swing', 'ob', 'liquidity', 'qml' or None (all)
            active_only: Only active (unfilled/unbroken) formations

        Returns:
            List of formations
        """
        if formation_type == 'bos':
            return self._structure_detector.get_bos_formations()
        elif formation_type == 'choch':
            return self._structure_detector.get_choch_formations()
        elif formation_type == 'fvg':
            if active_only:
                return self._fvg_detector.get_active()
            return self._fvg_detector.get_history()
        elif formation_type == 'swing':
            return self._swing_detector.get_history()
        elif formation_type == 'ob':
            if self._order_block_detector:
                if active_only:
                    return self._order_block_detector.get_active()
                return self._order_block_detector.get_history()
            return []
        elif formation_type == 'liquidity':
            if self._liquidity_detector:
                if active_only:
                    return self._liquidity_detector.get_active()
                return self._liquidity_detector.get_history()
            return []
        elif formation_type == 'qml':
            if self._qml_detector:
                return self._qml_detector.get_history()
            return []
        elif formation_type == 'ftr':
            if self._ftr_detector:
                if active_only:
                    return self._ftr_detector.get_active()
                return self._ftr_detector._zones
            return []
        elif formation_type == 'gap':
            if self._gap_detector:
                if active_only:
                    return self._gap_detector.get_active()
                return self._gap_detector.get_history()
            return []
        else:
            # All formations
            formations = []
            formations.extend(self._structure_detector.get_history())
            formations.extend(self._fvg_detector.get_history())
            formations.extend(self._swing_detector.get_history())
            if self._order_block_detector:
                formations.extend(self._order_block_detector.get_history())
            if self._liquidity_detector:
                formations.extend(self._liquidity_detector.get_history())
            if self._qml_detector:
                formations.extend(self._qml_detector.get_history())
            if self._ftr_detector:
                formations.extend(self._ftr_detector._zones)
            if self._gap_detector:
                formations.extend(self._gap_detector.get_history())
            return formations

    def get_active_zones(self) -> Dict[str, List[Any]]:
        """
        Get all active zones (FVG, OB, Liquidity, Gap)

        Returns:
            {'fvg': [...], 'ob': [...], 'liquidity': [...], 'gap': [...]}
        """
        result = {
            'fvg': self._fvg_detector.get_active(),
            'ob': [],
            'liquidity': [],
            'gap': []
        }

        if self._order_block_detector:
            result['ob'] = self._order_block_detector.get_active()

        if self._liquidity_detector:
            result['liquidity'] = self._liquidity_detector.get_active()

        if self._gap_detector:
            result['gap'] = self._gap_detector.get_active()

        return result

    def get_current_levels(self) -> Dict[str, Optional[float]]:
        """
        Get current swing levels

        Returns:
            {'swing_high': ..., 'swing_low': ...}
        """
        return {
            'swing_high': self._swing_detector.get_current_swing_high(),
            'swing_low': self._swing_detector.get_current_swing_low()
        }

    def get_unbroken_levels(self) -> Dict[str, list]:
        """
        Get all unbroken swing levels for horizontal lines

        Returns:
            {
                'highs': [SwingPoint, ...],  # Unbroken swing highs
                'lows': [SwingPoint, ...]    # Unbroken swing lows
            }
        """
        return {
            'highs': self._swing_detector.get_unbroken_highs(),
            'lows': self._swing_detector.get_unbroken_lows()
        }

    def get_result_at(self, index: int) -> Optional[AnalysisResult]:
        """Get result at specific bar index"""
        if 0 <= index < len(self._results):
            return self._results[index]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary"""
        summary = {
            'total_bars': len(self._results),
            'bos_count': len(self._structure_detector.get_bos_formations()),
            'choch_count': len(self._structure_detector.get_choch_formations()),
            'fvg_count': len(self._fvg_detector.get_history()),
            'swing_count': len(self._swing_detector.get_history()),
            'active_fvg_count': len(self._fvg_detector.get_active()),
            'current_trend': self._structure_detector.get_current_trend(),
        }

        # Optional detector counts
        if self._order_block_detector:
            summary['ob_count'] = len(self._order_block_detector.get_history())
            summary['active_ob_count'] = len(self._order_block_detector.get_active())

        if self._liquidity_detector:
            summary['liquidity_count'] = len(self._liquidity_detector.get_history())
            summary['active_liquidity_count'] = len(self._liquidity_detector.get_active())
            summary['swept_count'] = len(self._liquidity_detector.get_swept())

        if self._qml_detector:
            summary['qml_count'] = len(self._qml_detector.get_history())
            summary['bullish_qml_count'] = len(self._qml_detector.get_bullish())
            summary['bearish_qml_count'] = len(self._qml_detector.get_bearish())

        if self._ftr_detector:
            ftr_zones = self._ftr_detector._zones
            summary['ftr_count'] = len(ftr_zones)
            summary['ftr_fresh_count'] = len([z for z in ftr_zones if z.status == 'fresh'])

        if self._gap_detector:
            summary['gap_count'] = len(self._gap_detector.get_history())
            summary['active_gap_count'] = len(self._gap_detector.get_active())

        return summary

    def reset(self) -> None:
        """Reset all detectors"""
        self._swing_detector.reset()
        self._structure_detector.reset()
        self._fvg_detector.reset()
        if self._pattern_detector:
            self._pattern_detector.reset()
        if self._order_block_detector:
            self._order_block_detector.reset()
        if self._liquidity_detector:
            self._liquidity_detector.reset()
        if self._qml_detector:
            self._qml_detector.reset()
        if self._ftr_detector:
            self._ftr_detector.reset()
        if self._gap_detector:
            self._gap_detector.reset()
        self._results = []
        self._current_index = 0
        self._initialized = False


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_candles(
    data: pd.DataFrame,
    config: Dict[str, Any] = None
) -> BatchAnalysisResult:
    """
    Quick analysis function

    Args:
        data: OHLCV DataFrame
        config: Optional config

    Returns:
        BatchAnalysisResult
    """
    engine = AnalysisEngine(config)
    return engine.analyze(data)


def get_formations_at(
    data: pd.DataFrame,
    bar_index: int,
    config: Dict[str, Any] = None
) -> AnalysisResult:
    """
    Get formations at specific bar

    Args:
        data: OHLCV DataFrame
        bar_index: Bar index to check
        config: Optional config

    Returns:
        AnalysisResult at that bar
    """
    engine = AnalysisEngine(config)
    result = engine.analyze(data)
    return result.get_active_at(bar_index)
