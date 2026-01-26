"""
SMC (Smart Money Concepts) Analysis Service

AnalysisEngine wrapper for WebUI
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import yaml


def get_utc_offset() -> int:
    """Get UTC offset from system config (default: 3 for UTC+3)"""
    try:
        from core.config_engine import get_config
        config = get_config()
        return config.get('system', {}).get('utc_offset', 3)
    except Exception:
        return 3  # Default UTC+3


def load_analysis_config() -> Dict[str, Any]:
    """Load analysis config from config/analysis.yaml via ConfigEngine or direct file read"""
    # Try ConfigEngine first (if WebUI is running)
    try:
        from core.config_engine import ConfigEngine
        config_engine = ConfigEngine()
        # ConfigEngine might already have loaded analysis.yaml
        analysis_config = config_engine.get("analysis", None)
        if analysis_config:
            return analysis_config
    except Exception:
        pass

    # Fallback: Direct file read
    config_path = Path("config/analysis.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('analysis', {})
        except Exception:
            pass
    return {}


class SMCService:
    """
    SMC Analysis Service

    Wrapper for the AnalysisEngine for the WebUI.
    Loads Parquet data and performs analysis.
    """

    def __init__(self, parquets_engine=None, logger=None):
        """
        Args:
            parquets_engine: ParquetsEngine instance (optional)
            logger: Logger instance
        """
        self.parquets_engine = parquets_engine
        self.logger = logger

        # Analysis engine (lazy load)
        self._engine = None
        self._cache: Dict[str, Any] = {}  # symbol_tf -> analysis result

    def _get_engine(self, force_reload: bool = False):
        """Lazy load AnalysisEngine with config from analysis.yaml"""
        if self._engine is None or force_reload:
            from modules.analysis import AnalysisEngine

            # Load config from file
            config = load_analysis_config()

            # Build engine config with defaults
            engine_config = {
                'swing': {
                    'left_bars': config.get('swing', {}).get('left_bars', 5),
                    'right_bars': config.get('swing', {}).get('right_bars', 5),
                    'max_levels': config.get('levels', {}).get('max_levels', 10),
                },
                'structure': {
                    'max_levels': config.get('bos', {}).get('max_levels', 5),
                    'trend_strength': config.get('bos', {}).get('trend_strength', 2),
                },
                'fvg': {
                    'min_size_pct': config.get('fvg', {}).get('min_size_pct', 0.1),
                    'max_age': config.get('fvg', {}).get('max_age', 50),
                    'max_box': config.get('fvg', {}).get('max_box', 10),
                },
                'patterns': {'enabled': False},
                'orderblocks': {
                    'enabled': config.get('orderblocks', {}).get('enabled', False),
                    'strength_threshold': config.get('orderblocks', {}).get('strength_threshold', 1.0),
                    'max_blocks': config.get('orderblocks', {}).get('max_blocks', 3),
                    'lookback': config.get('orderblocks', {}).get('lookback', 20),
                },
                'liquidity': {
                    'enabled': config.get('liquidity', {}).get('enabled', False),
                    'equal_tolerance': config.get('liquidity', {}).get('equal_tolerance', 0.1),
                    'max_zones': config.get('liquidity', {}).get('max_zones', 5),
                    'sweep_lookback': config.get('liquidity', {}).get('sweep_lookback', 3),
                },
                'qml': {
                    'enabled': config.get('qml', {}).get('enabled', False),
                    'lookback_bars': config.get('qml', {}).get('lookback_bars', 30),
                    'break_threshold': config.get('qml', {}).get('break_threshold', 0.1),
                },
                'ftr': {
                    'enabled': config.get('ftr', {}).get('enabled', True),
                    'min_momentum_candles': config.get('ftr', {}).get('min_momentum_candles', 3),
                    'min_confirmation_candles': config.get('ftr', {}).get('min_confirmation_candles', 1),
                    'max_ftr_ratio': config.get('ftr', {}).get('max_ftr_ratio', 0.3),
                    'require_confirmation': config.get('ftr', {}).get('require_confirmation', True),
                    'max_zones': config.get('ftr', {}).get('max_zones', 20),
                    'invalidation_threshold': config.get('ftr', {}).get('invalidation_threshold', 0.5),
                },
                'gap': {
                    'enabled': config.get('gap', {}).get('enabled', True),
                    'min_size_pct': config.get('gap', {}).get('min_size_pct', 0.05),
                    'max_age': config.get('gap', {}).get('max_age', 500),
                    'max_zones': config.get('gap', {}).get('max_zones', 100),
                },
            }

            if self.logger:
                self.logger.info(f"SMC Engine config: {engine_config}")

            self._engine = AnalysisEngine(engine_config)
        return self._engine

    def reload_config(self):
        """Reload config and recreate engine"""
        self._engine = None
        self._cache.clear()
        return self._get_engine(force_reload=True)

    async def _load_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data (from parquets_engine or from a file).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Max bars (ignored if start_date is given)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) - defaults to now if start_date given

        Returns:
            DataFrame or None
        """
        # Date mode: ParquetsEngine is used if both dates are provided.
        use_date_range = start_date is not None and end_date is not None

        # ParquetsEngine is only used in the date range mode.
        if self.parquets_engine and use_date_range:
            try:
                df = await self.parquets_engine.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and len(df) > 0:
                    return df.reset_index(drop=True)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"ParquetsEngine error: {e}")

        # Fallback: Load directly from the parquet file - new format: data/parquets/{symbol}/
        parquet_dir = Path("data/parquets")
        symbol_dir = parquet_dir / symbol
        if not symbol_dir.exists():
            return None

        pattern = f"{symbol}_{timeframe}_*.parquet"
        files = list(symbol_dir.glob(pattern))

        if not files:
            return None

        parquet_file = sorted(files)[-1]

        try:
            df = pd.read_parquet(parquet_file)

            # Column mapping
            if 'open_time' in df.columns and 'timestamp' not in df.columns:
                open_time = df['open_time']
                # Timestamp/datetime tipini int64 millisecond'a cevir
                if pd.api.types.is_datetime64_any_dtype(open_time):
                    df['timestamp'] = open_time.astype('int64') // 10**6
                elif open_time.dtype == 'object':
                    df['timestamp'] = pd.to_datetime(open_time).astype('int64') // 10**6
                else:
                    # Numeric - can be millisecond or second.
                    first_val = float(open_time.iloc[0])
                    if first_val > 1e12:
                        df['timestamp'] = open_time.astype('int64')
                    else:
                        df['timestamp'] = (open_time * 1000).astype('int64')

            # Date filter
            if use_date_range:
                start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
                df = df[df['timestamp'] >= start_ts]

                if end_date:
                    end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
                    df = df[df['timestamp'] <= end_ts]
                else:
                    # If end_date is not provided, use now()
                    now_ts = int(datetime.now().timestamp() * 1000)
                    df = df[df['timestamp'] <= now_ts]
            else:
                # Bars mode: apply limit
                if limit and len(df) > limit:
                    df = df.tail(limit)

            return df.reset_index(drop=True)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Parquet loading error: {e}")
            return None

    async def analyze(
        self,
        symbol: str,
        timeframe: str = '5m',
        limit: int = 500,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        Tam analiz yap

        Returns:
            {
                "summary": {...},
                "formations": {...},
                "levels": {...},
                "last_bar": {...},
                "annotations": {...}
            }
        """
        # Load data
        df = await self._load_data(symbol, timeframe, limit, start_date, end_date)

        if df is None or len(df) == 0:
            return {
                'error': 'Data not found',
                'symbol': symbol,
                'timeframe': timeframe
            }

        # Analiz - force_reload=True to pick up config changes
        engine = self._get_engine(force_reload=True)
        engine.reset()
        result = engine.analyze(df)

        # Cache
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = {
            'engine': engine,
            'result': result,
            'data': df,
            'timestamp': datetime.now()
        }

        # Summary
        summary = engine.get_summary()
        levels = engine.get_current_levels()

        # Formations - swing'leri indexle
        swings = engine.get_formations('swing')
        swing_dict = {s.index: s for s in swings}

        # TODO - add swing_time
        bos_list = []
        for f in engine.get_formations('bos'):
            d = f.to_dict()
            # Find the swing time from the swing index
            if f.swing_index in swing_dict:
                d['swing_time'] = swing_dict[f.swing_index].time
            bos_list.append(d)

        # CHoCH - add swing_time (since swing_index doesn't exist in CHoCH, let's use break_index - offset)
        choch_list = []
        for f in engine.get_formations('choch'):
            d = f.to_dict()
            # Find the nearest swing (based on broken_level)
            closest_swing = None
            for s in swings:
                if abs(s.price - f.broken_level) < 0.01:  # Same price
                    if closest_swing is None or s.index > closest_swing.index:
                        if s.index < f.break_index:
                            closest_swing = s
            if closest_swing:
                d['swing_time'] = closest_swing.time
                d['swing_index'] = closest_swing.index
            choch_list.append(d)

        fvg_list = [f.to_dict() for f in engine.get_formations('fvg')]
        swing_list = [f.to_dict() for f in swings]

        # Order Blocks
        ob_list = [f.to_dict() for f in engine.get_formations('ob')]

        # Liquidity Zones
        liq_list = [f.to_dict() for f in engine.get_formations('liquidity')]

        # QML Patterns
        qml_list = [f.to_dict() for f in engine.get_formations('qml')]

        # FTR/FTB Zones
        ftr_list = [f.to_dict() for f in engine.get_formations('ftr')]

        # Gap (space between 2 candles)
        gap_list = [f.to_dict() for f in engine.get_formations('gap')]

        if self.logger:
            self.logger.info(f"Gap formations detected: {len(gap_list)}")
            if gap_list:
                self.logger.info(f"First gap: {gap_list[0]}")

        # Last bar
        last = result[-1]
        last_bar = last.to_dict()

        # Annotations for chart
        annotations = await self.get_annotations(symbol, timeframe, 0, len(df))

        # Get max_box from config
        analysis_config = load_analysis_config()
        fvg_max_box = analysis_config.get('fvg', {}).get('max_box', 10)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': len(df),
            'utc_offset': get_utc_offset(),  # UTC offset for timezone display
            'fvg_max_box': fvg_max_box,  # Max FVG boxes to display per type
            'summary': summary,
            'levels': levels,
            'formations': {
                'bos': bos_list[-20:],  # Last 20
                'choch': choch_list[-10:],
                'fvg': fvg_list[-20:],
                'swing': swing_list[-20:],
                'ob': ob_list[-20:],
                'liquidity': liq_list[-20:],
                'qml': qml_list[-10:],
                'ftr_zones': ftr_list[-30:],  # FTR/FTB zones
                'gap': gap_list[-50:]  # Gap zones (space between 2 candles)
            },
            'active': {
                'fvg': [f.to_dict() for f in engine.get_formations('fvg', active_only=True)],
                'ob': [f.to_dict() for f in engine.get_formations('ob', active_only=True)],
                'liquidity': [f.to_dict() for f in engine.get_formations('liquidity', active_only=True)],
                'ftr_zones': [f.to_dict() for f in engine.get_formations('ftr', active_only=True)],
                'gap': [f.to_dict() for f in engine.get_formations('gap', active_only=True)]
            },
            'last_bar': last_bar,
            'annotations': annotations
        }

    async def get_formations(
        self,
        symbol: str,
        timeframe: str = '5m',
        formation_type: str = None,
        active_only: bool = False,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Formation list"""
        cache_key = f"{symbol}_{timeframe}"

        # Cache control
        if cache_key not in self._cache:
            await self.analyze(symbol, timeframe)

        if cache_key not in self._cache:
            return {'error': 'Data not found'}

        engine = self._cache[cache_key]['engine']

        formations = engine.get_formations(formation_type, active_only)

        return {
            'formations': [f.to_dict() for f in formations[-limit:]],
            'total': len(formations),
            'type': formation_type or 'all',
            'active_only': active_only
        }

    async def get_annotations(
        self,
        symbol: str,
        timeframe: str = '5m',
        start: int = 0,
        end: int = None
    ) -> Dict[str, Any]:
        """
        Chart annotations (LightweightCharts format)

        Returns:
            {
                "markers": [...],
                "zones": [...],
                "lines": [...]
            }
        """
        cache_key = f"{symbol}_{timeframe}"

        if cache_key not in self._cache:
            await self.analyze(symbol, timeframe)

        if cache_key not in self._cache:
            return {'markers': [], 'zones': [], 'lines': []}

        result = self._cache[cache_key]['result']
        engine = self._cache[cache_key]['engine']

        markers = []
        zones = []
        lines = []

        # Bar range
        end = end or len(result)

        for i in range(start, min(end, len(result))):
            r = result[i]

            # Markers (BOS, CHoCH, Swing)
            if r.new_bos:
                markers.append({
                    'time': r.timestamp // 1000,
                    'position': 'aboveBar' if r.new_bos.type == 'bullish' else 'belowBar',
                    'color': '#26a69a' if r.new_bos.type == 'bullish' else '#ef5350',
                    'shape': 'circle',
                    'text': f"BOS",
                    'size': 2
                })

            if r.new_choch:
                markers.append({
                    'time': r.timestamp // 1000,
                    'position': 'aboveBar' if r.new_choch.type == 'bullish' else 'belowBar',
                    'color': '#ffeb3b',
                    'shape': 'square',
                    'text': f"CHoCH",
                    'size': 2
                })

            if r.new_swing:
                # Use label (HH, HL, LH, LL) if available, otherwise SH/SL
                swing_label = r.new_swing.label if hasattr(r.new_swing, 'label') and r.new_swing.label else ('SH' if r.new_swing.type == 'high' else 'SL')
                markers.append({
                    'time': r.timestamp // 1000,
                    'position': 'aboveBar' if r.new_swing.type == 'high' else 'belowBar',
                    'color': '#26a69a' if r.new_swing.type == 'high' else '#ef5350',
                    'shape': 'arrowDown' if r.new_swing.type == 'high' else 'arrowUp',
                    'text': swing_label,
                    'size': 1
                })

            # FVG zones
            if r.new_fvg:
                zones.append({
                    'type': 'fvg',
                    'fvg_type': r.new_fvg.type,
                    'top': r.new_fvg.top,
                    'bottom': r.new_fvg.bottom,
                    'start_time': r.timestamp // 1000,
                    'color': 'rgba(38, 166, 154, 0.15)' if r.new_fvg.type == 'bullish' else 'rgba(239, 83, 80, 0.15)',
                    'border_color': '#26a69a' if r.new_fvg.type == 'bullish' else '#ef5350'
                })

            # Order Block zones
            if r.new_ob:
                zones.append({
                    'type': 'ob',
                    'ob_type': r.new_ob.type,
                    'top': r.new_ob.top,
                    'bottom': r.new_ob.bottom,
                    'start_time': r.timestamp // 1000,
                    'color': 'rgba(156, 39, 176, 0.2)' if r.new_ob.type == 'bullish' else 'rgba(156, 39, 176, 0.2)',
                    'border_color': '#9c27b0'
                })

        # Active Order Blocks as zones
        for ob in engine.get_formations('ob', active_only=True):
            zones.append({
                'type': 'ob',
                'ob_type': ob.type,
                'top': ob.top,
                'bottom': ob.bottom,
                'start_time': ob.timestamp // 1000 if ob.timestamp else 0,
                'color': 'rgba(156, 39, 176, 0.15)',
                'border_color': '#9c27b0'
            })

        # Liquidity zones are drawn separately via formations.liquidity in JS
        # (drawLiquidityLine function) - no need to add to annotations.lines

        # QML markers
        for qml in engine.get_formations('qml'):
            is_bullish = 'bullish' in qml.type
            markers.append({
                'time': qml.timestamp // 1000 if qml.timestamp else 0,
                'position': 'belowBar' if is_bullish else 'aboveBar',
                'color': '#2196f3',
                'shape': 'diamond',
                'text': 'QML',
                'size': 2
            })

        # All unbroken swing levels as horizontal lines
        unbroken = engine.get_unbroken_levels()

        # Unbroken swing highs (resistance levels)
        for i, swing in enumerate(unbroken['highs']):
            label = swing.label if swing.label else 'SH'
            lines.append({
                'price': swing.price,
                'color': '#26a69a',  # Green for highs
                'lineWidth': 1,
                'lineStyle': 2,  # Dashed
                'title': f'{label} {swing.price:.2f}',
                'start_time': swing.time // 1000,  # For reference
                'type': 'swing_high'
            })

        # Unbroken swing lows (support levels)
        for i, swing in enumerate(unbroken['lows']):
            label = swing.label if swing.label else 'SL'
            lines.append({
                'price': swing.price,
                'color': '#ef5350',  # Red for lows
                'lineWidth': 1,
                'lineStyle': 2,  # Dashed
                'title': f'{label} {swing.price:.2f}',
                'start_time': swing.time // 1000,  # For reference
                'type': 'swing_low'
            })

        return {
            'markers': markers,
            'zones': zones,
            'lines': lines
        }

    async def get_levels(
        self,
        symbol: str,
        timeframe: str = '5m'
    ) -> Dict[str, Any]:
        """Current swing levels"""
        cache_key = f"{symbol}_{timeframe}"

        if cache_key not in self._cache:
            await self.analyze(symbol, timeframe)

        if cache_key not in self._cache:
            return {'swing_high': None, 'swing_low': None}

        engine = self._cache[cache_key]['engine']
        levels = engine.get_current_levels()

        # Trend info
        result = self._cache[cache_key]['result']
        last = result[-1] if result else None

        return {
            **levels,
            'trend': last.trend if last else 'unknown',
            'bias': last.market_bias if last else 'neutral',
            'structure': last.structure if last else 'unknown'
        }

    async def get_bar_analysis(
        self,
        symbol: str,
        timeframe: str,
        bar_index: int
    ) -> Optional[Dict[str, Any]]:
        """Tek bar analizi"""
        cache_key = f"{symbol}_{timeframe}"

        if cache_key not in self._cache:
            await self.analyze(symbol, timeframe)

        if cache_key not in self._cache:
            return None

        result = self._cache[cache_key]['result']

        if 0 <= bar_index < len(result):
            return result[bar_index].to_dict()

        return None
