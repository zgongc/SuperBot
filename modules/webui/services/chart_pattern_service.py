"""
Chart Pattern Analysis Service

Geometric pattern detection (Head & Shoulders, Double Top/Bottom, Triangles, etc.)
WebUI wrapper for ChartPatternDetector
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml


def get_utc_offset() -> int:
    """Get UTC offset from system config (default: 3 for UTC+3)"""
    try:
        from core.config_engine import get_config
        config = get_config()
        return config.get('system', {}).get('utc_offset', 3)
    except Exception:
        return 3


def load_analysis_config() -> Dict[str, Any]:
    """Load analysis config from config/analysis.yaml"""
    try:
        from core.config_engine import ConfigEngine
        config_engine = ConfigEngine()
        analysis_config = config_engine.get("analysis", None)
        if analysis_config:
            return analysis_config
    except Exception:
        pass

    config_path = Path("config/analysis.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('analysis', {})
        except Exception:
            pass
    return {}


class ChartPatternService:
    """
    Chart Pattern Analysis Service

    Detects geometric chart patterns using ChartPatternDetector.
    Loads Parquet data and performs pattern detection.
    """

    def __init__(self, parquets_engine=None, logger=None):
        """
        Args:
            parquets_engine: ParquetsEngine instance (optional)
            logger: Logger instance
        """
        self.parquets_engine = parquets_engine
        self.logger = logger
        self._detector = None
        self._cache: Dict[str, Any] = {}

    def _get_detector(self, force_reload: bool = False):
        """Lazy load ChartPatternDetector with config"""
        if self._detector is None or force_reload:
            from modules.analysis.detectors.chart_pattern_detector import ChartPatternDetector

            config = load_analysis_config()
            chart_config = config.get('chart_patterns', {})

            # TSR (Trend Lines, Supports and Resistances) parameters
            detector_config = {
                'pivot_length': chart_config.get('pivot_length', 20),
                'points_to_check': chart_config.get('points_to_check', 3),
                'max_violation': chart_config.get('max_violation', 0),
                'except_bars': chart_config.get('except_bars', 3),
                'extend_lines': chart_config.get('extend_lines', True),
                'patterns': chart_config.get('patterns', {}),
            }

            if self.logger:
                self.logger.info(f"ChartPatternDetector config: {detector_config}")

            self._detector = ChartPatternDetector(detector_config)

        return self._detector

    def reload_config(self):
        """Reload config and recreate detector"""
        self._detector = None
        self._cache.clear()
        return self._get_detector(force_reload=True)

    async def _load_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """Load data from parquets_engine or file"""
        use_date_range = start_date is not None and end_date is not None

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

            if 'open_time' in df.columns and 'timestamp' not in df.columns:
                open_time = df['open_time']
                if pd.api.types.is_datetime64_any_dtype(open_time):
                    df['timestamp'] = open_time.astype('int64') // 10**6
                elif open_time.dtype == 'object':
                    df['timestamp'] = pd.to_datetime(open_time).astype('int64') // 10**6
                else:
                    first_val = float(open_time.iloc[0])
                    if first_val > 1e12:
                        df['timestamp'] = open_time.astype('int64')
                    else:
                        df['timestamp'] = (open_time * 1000).astype('int64')

            if use_date_range:
                start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
                df = df[df['timestamp'] >= start_ts]

                if end_date:
                    end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
                    df = df[df['timestamp'] <= end_ts]
                else:
                    now_ts = int(datetime.now().timestamp() * 1000)
                    df = df[df['timestamp'] <= now_ts]
            else:
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
        Analyze chart patterns

        Returns:
            {
                "symbol": str,
                "timeframe": str,
                "bars": int,
                "patterns": [...],
                "swings": [...],
                "annotations": {...},
                "candles": [...]
            }
        """
        df = await self._load_data(symbol, timeframe, limit, start_date, end_date)

        if df is None or len(df) == 0:
            return {
                'error': 'Data not found',
                'symbol': symbol,
                'timeframe': timeframe
            }

        detector = self._get_detector(force_reload=True)
        detector.reset()

        patterns = detector.detect(df)
        swings = detector.get_swings()
        trendlines = detector.get_trendlines()

        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = {
            'detector': detector,
            'patterns': patterns,
            'swings': swings,
            'trendlines': trendlines,
            'data': df,
            'timestamp': datetime.now()
        }

        pattern_list = [self._pattern_to_dict(p) for p in patterns]
        swing_list = [self._swing_to_dict(s) for s in swings]
        trendline_list = [self._trendline_to_dict(t, df) for t in trendlines]

        candles = self._prepare_candles(df)

        annotations = self._get_annotations(patterns, swings, trendlines, df)

        config = load_analysis_config()
        chart_config = config.get('chart_patterns', {})
        enabled_patterns = chart_config.get('patterns', {})

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': len(df),
            'utc_offset': get_utc_offset(),
            'patterns': pattern_list,
            'swings': swing_list,
            'trendlines': trendline_list,
            'candles': candles,
            'annotations': annotations,
            'config': {
                'enabled_patterns': enabled_patterns,
                'pivot_length': chart_config.get('pivot_length', 20),
                'points_to_check': chart_config.get('points_to_check', 3),
                'max_violation': chart_config.get('max_violation', 0),
                'except_bars': chart_config.get('except_bars', 3),
                'extend_lines': chart_config.get('extend_lines', True),
            },
            'stats': {
                'total_patterns': len(patterns),
                'bullish': len([p for p in patterns if p.type == 'bullish']),
                'bearish': len([p for p in patterns if p.type == 'bearish']),
                'trendlines': len(trendlines),
                'by_type': self._count_by_type(patterns),
            }
        }

    def _to_python_type(self, value):
        """Convert numpy types to native Python types for JSON serialization"""
        if value is None:
            return None
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        return value

    def _pattern_to_dict(self, pattern) -> Dict[str, Any]:
        """Convert ChartPattern to dict"""
        return {
            'id': pattern.id,
            'name': pattern.name,
            'display_name': pattern.display_name,
            'type': pattern.type,
            'status': pattern.status,
            'swings': [self._swing_to_dict(s) for s in pattern.swings],
            'start_time': int(pattern.start_time) if pattern.start_time else 0,
            'end_time': int(pattern.end_time) if pattern.end_time else 0,
            'start_index': int(pattern.start_index) if pattern.start_index is not None else 0,
            'end_index': int(pattern.end_index) if pattern.end_index is not None else 0,
            'neckline': float(pattern.neckline) if pattern.neckline is not None else None,
            'target': float(pattern.target) if pattern.target is not None else None,
            'confidence': float(pattern.confidence) if pattern.confidence is not None else 0,
            'breakout_price': float(pattern.breakout_price) if pattern.breakout_price is not None else None,
            'breakout_confirmed': pattern.breakout_confirmed,
        }

    def _swing_to_dict(self, swing) -> Dict[str, Any]:
        """Convert SwingPoint to dict"""
        return {
            'index': int(swing.index),
            'time': int(swing.time),
            'price': float(swing.price),
            'type': swing.type,
            'label': getattr(swing, 'label', None),
        }

    def _trendline_to_dict(self, trendline: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert trendline dict for frontend"""
        start_time = trendline.get('start_time', 0)
        end_time = trendline.get('end_time', 0)

        # Map TSR type to frontend type
        # uptrend (rising lows) = support line
        # downtrend (falling highs) = resistance line
        tl_type = trendline.get('type', '')
        if tl_type == 'uptrend':
            frontend_type = 'support'
        elif tl_type == 'downtrend':
            frontend_type = 'resistance'
        else:
            frontend_type = tl_type

        return {
            'type': frontend_type,
            'direction': trendline.get('direction', ''),
            'slope': float(trendline['slope']),
            'intercept': float(trendline['intercept']),
            'r_squared': float(trendline.get('r_squared', 1.0)),
            'touches': int(trendline['touches']),
            'score': float(trendline.get('score', 0)),
            'violations': int(trendline.get('violations', 0)),
            'is_violated': trendline.get('is_violated', False),
            'start_index': int(trendline['start_index']),
            'end_index': int(trendline['end_index']),
            'start_value': float(trendline['start_value']),
            'end_value': float(trendline['end_value']),
            'current_value': float(trendline.get('current_value', trendline['end_value'])),
            'start_time': int(start_time // 1000) if start_time else 0,
            'end_time': int(end_time // 1000) if end_time else 0,
        }

    def _prepare_candles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare candle data for chart"""
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': int(row['timestamp'] // 1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
            })
        return candles

    def _get_annotations(
        self,
        patterns: List,
        swings: List,
        trendlines: List,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate chart annotations"""
        markers = []
        lines = []
        zones = []

        # Swing markers
        for swing in swings:
            markers.append({
                'time': int(swing.time // 1000),
                'position': 'aboveBar' if swing.type == 'high' else 'belowBar',
                'color': '#26a69a' if swing.type == 'high' else '#ef5350',
                'shape': 'arrowDown' if swing.type == 'high' else 'arrowUp',
                'text': getattr(swing, 'label', 'SH' if swing.type == 'high' else 'SL'),
                'size': 1
            })

        # Trendlines
        for tl in trendlines:
            # Map TSR type: uptrend = support (green), downtrend = resistance (red)
            tl_type = tl.get('type', '')
            if tl_type == 'uptrend':
                color = '#26a69a'  # Green for support
                frontend_type = 'support'
            elif tl_type == 'downtrend':
                color = '#ef5350'  # Red for resistance
                frontend_type = 'resistance'
            else:
                color = '#26a69a' if tl_type == 'support' else '#ef5350'
                frontend_type = tl_type

            start_time = tl.get('start_time', 0)
            end_time = tl.get('end_time', 0)
            lines.append({
                'type': 'trendline',
                'trendline_type': frontend_type,
                'start_time': int(start_time // 1000) if start_time else 0,
                'end_time': int(end_time // 1000) if end_time else 0,
                'start_value': float(tl['start_value']),
                'end_value': float(tl['end_value']),
                'color': color,
                'lineWidth': 2,
                'lineStyle': 0,
                'r_squared': float(tl.get('r_squared', 1.0)),
                'touches': int(tl['touches']),
            })

        for pattern in patterns:
            color = self._get_pattern_color(pattern.type)

            if pattern.neckline:
                lines.append({
                    'type': 'neckline',
                    'pattern_id': pattern.id,
                    'pattern_name': pattern.display_name,
                    'price': float(pattern.neckline),
                    'color': color,
                    'lineWidth': 2,
                    'lineStyle': 0,
                    'start_time': int(pattern.start_time // 1000) if pattern.start_time else 0,
                    'end_time': int(pattern.end_time // 1000) if pattern.end_time else 0,
                })

            if pattern.target:
                lines.append({
                    'type': 'target',
                    'pattern_id': pattern.id,
                    'pattern_name': pattern.display_name,
                    'price': float(pattern.target),
                    'color': color,
                    'lineWidth': 1,
                    'lineStyle': 2,
                    'start_time': int(pattern.end_time // 1000) if pattern.end_time else 0,
                })

            if len(pattern.swings) >= 2:
                swing_points = []
                for s in pattern.swings:
                    swing_points.append({
                        'time': int(s.time // 1000),
                        'price': float(s.price)
                    })
                zones.append({
                    'type': 'pattern',
                    'pattern_id': pattern.id,
                    'pattern_name': pattern.display_name,
                    'pattern_type': pattern.type,
                    'status': pattern.status,
                    'points': swing_points,
                    'color': self._get_pattern_fill_color(pattern.type),
                    'border_color': color,
                })

        return {
            'markers': markers,
            'lines': lines,
            'zones': zones,
        }

    def _get_pattern_color(self, pattern_type: str) -> str:
        """Get color for pattern type"""
        colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#ffeb3b',
        }
        return colors.get(pattern_type, '#9e9e9e')

    def _get_pattern_fill_color(self, pattern_type: str) -> str:
        """Get fill color for pattern type"""
        colors = {
            'bullish': 'rgba(38, 166, 154, 0.1)',
            'bearish': 'rgba(239, 83, 80, 0.1)',
            'neutral': 'rgba(255, 235, 59, 0.1)',
        }
        return colors.get(pattern_type, 'rgba(158, 158, 158, 0.1)')

    def _count_by_type(self, patterns: List) -> Dict[str, int]:
        """Count patterns by name"""
        counts = {}
        for p in patterns:
            counts[p.name] = counts.get(p.name, 0) + 1
        return counts

    async def get_pattern_info(self) -> Dict[str, Any]:
        """Get pattern definitions and info"""
        from modules.analysis.detectors.chart_pattern_detector import ChartPatternDetector

        return {
            'patterns': ChartPatternDetector.PATTERN_INFO,
            'description': {
                'resistance_breakout': 'Price breaks above falling resistance trendline (bullish)',
                'support_breakout': 'Price breaks below rising support trendline (bearish)',
            },
            'trendlines': {
                'uptrend': 'Rising low pivots forming support line',
                'downtrend': 'Falling high pivots forming resistance line',
            }
        }

    async def get_config(self) -> Dict[str, Any]:
        """Get current chart pattern config"""
        config = load_analysis_config()
        return config.get('chart_patterns', {})
