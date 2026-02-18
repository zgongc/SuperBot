"""
Candlestick Pattern Analysis Service

PatternDetector wrapper for WebUI
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


def load_candlestick_config() -> Dict[str, bool]:
    """Load candlestick pattern config from config/analysis.yaml"""
    config_path = Path("config/analysis.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('analysis', {}).get('candlestick', {})
        except Exception:
            pass
    # Default: all patterns enabled
    return {}


class PatternsService:
    """
    Candlestick Pattern Analysis Service

    Wrapper for PatternDetector for WebUI.
    Loads Parquet data and performs pattern analysis.
    """

    # Pattern info with descriptions and signal type
    PATTERN_INFO = {
        # Single candle - Bullish
        'hammer': {'name': 'Hammer', 'type': 'bullish', 'description': 'Bullish reversal at bottom', 'bars': 1},
        'inverted_hammer': {'name': 'Inverted Hammer', 'type': 'bullish', 'description': 'Bullish reversal at bottom', 'bars': 1},
        'dragonfly_doji': {'name': 'Dragonfly Doji', 'type': 'bullish', 'description': 'Potential bullish reversal', 'bars': 1},
        'marubozu_bullish': {'name': 'Bullish Marubozu', 'type': 'bullish', 'description': 'Strong bullish momentum', 'bars': 1},

        # Single candle - Bearish
        'hanging_man': {'name': 'Hanging Man', 'type': 'bearish', 'description': 'Bearish reversal at top', 'bars': 1},
        'shooting_star': {'name': 'Shooting Star', 'type': 'bearish', 'description': 'Bearish reversal at top', 'bars': 1},
        'gravestone_doji': {'name': 'Gravestone Doji', 'type': 'bearish', 'description': 'Potential bearish reversal', 'bars': 1},
        'marubozu_bearish': {'name': 'Bearish Marubozu', 'type': 'bearish', 'description': 'Strong bearish momentum', 'bars': 1},

        # Single candle - Neutral
        'doji': {'name': 'Doji', 'type': 'neutral', 'description': 'Market indecision', 'bars': 1},
        'longlegged_doji': {'name': 'Long-legged Doji', 'type': 'neutral', 'description': 'High volatility indecision', 'bars': 1},
        'spinning_top': {'name': 'Spinning Top', 'type': 'neutral', 'description': 'Market indecision', 'bars': 1},

        # Multi candle - Bullish
        'engulfing_bullish': {'name': 'Bullish Engulfing', 'type': 'bullish', 'description': 'Strong bullish reversal', 'bars': 2},
        'harami_bullish': {'name': 'Bullish Harami', 'type': 'bullish', 'description': 'Potential bullish reversal', 'bars': 2},
        'piercing_line': {'name': 'Piercing Line', 'type': 'bullish', 'description': 'Bullish reversal pattern', 'bars': 2},
        'morning_star': {'name': 'Morning Star', 'type': 'bullish', 'description': 'Strong bullish reversal', 'bars': 3},
        'three_white_soldiers': {'name': 'Three White Soldiers', 'type': 'bullish', 'description': 'Strong bullish continuation', 'bars': 3},

        # Multi candle - Bearish
        'engulfing_bearish': {'name': 'Bearish Engulfing', 'type': 'bearish', 'description': 'Strong bearish reversal', 'bars': 2},
        'harami_bearish': {'name': 'Bearish Harami', 'type': 'bearish', 'description': 'Potential bearish reversal', 'bars': 2},
        'dark_cloud_cover': {'name': 'Dark Cloud Cover', 'type': 'bearish', 'description': 'Bearish reversal pattern', 'bars': 2},
        'evening_star': {'name': 'Evening Star', 'type': 'bearish', 'description': 'Strong bearish reversal', 'bars': 3},
        'three_black_crows': {'name': 'Three Black Crows', 'type': 'bearish', 'description': 'Strong bearish continuation', 'bars': 3},
    }

    def __init__(self, parquets_engine=None, logger=None):
        """
        Args:
            parquets_engine: ParquetsEngine instance (optional)
            logger: Logger instance
        """
        self.parquets_engine = parquets_engine
        self.logger = logger

        # Pattern detector (lazy load)
        self._detector = None
        self._cache: Dict[str, Any] = {}

    def _get_detector(self):
        """Lazy load PatternDetector"""
        if self._detector is None:
            from modules.analysis.detectors.pattern_detector import PatternDetector
            self._detector = PatternDetector({
                'doji_threshold': 0.1,
                'shadow_ratio': 2.0,
                'min_body_size': 0.0001,
                'use_talib': False
            })
        return self._detector

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

        # Fallback: Direct parquet file read
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
                if pd.api.types.is_datetime64_any_dtype(open_time):
                    # Resolution-aware: datetime64[ns] vs datetime64[ms]
                    int_values = open_time.astype('int64')
                    first_val = int_values.iloc[0]
                    if first_val > 1e15:
                        df['timestamp'] = int_values // 10**6  # nanoseconds → ms
                    elif first_val > 1e12:
                        df['timestamp'] = int_values  # already milliseconds
                    else:
                        df['timestamp'] = int_values * 1000  # seconds → ms
                elif open_time.dtype == 'object':
                    df['timestamp'] = pd.to_datetime(open_time).astype('int64') // 10**6
                else:
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
        Full pattern analysis

        Returns:
            {
                "summary": {...},
                "patterns": [...],
                "candles": [...],
                "statistics": {...}
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

        # Detect all patterns (config controls checkbox defaults, not detection)
        detector = self._get_detector()
        detector.reset()
        patterns = detector.detect(df)

        # Cache
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = {
            'patterns': patterns,
            'data': df,
            'timestamp': datetime.now()
        }

        # Build candles for chart
        candles = []
        for i, row in df.iterrows():
            candles.append({
                'time': int(row['timestamp']) // 1000,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            })

        # Pattern list with enriched info
        pattern_list = []
        for p in patterns:
            info = self.PATTERN_INFO.get(p.name, {})
            pattern_list.append({
                'id': p.id,
                'name': info.get('name', p.name),
                'code': p.name,
                'type': p.type,
                'time': p.time,
                'index': p.index,
                'strength': p.strength,
                'description': info.get('description', ''),
                'bars': info.get('bars', 1)
            })

        # Statistics
        bullish_count = len([p for p in patterns if p.type == 'bullish'])
        bearish_count = len([p for p in patterns if p.type == 'bearish'])
        neutral_count = len([p for p in patterns if p.type == 'neutral'])

        # Pattern frequency
        pattern_counts = {}
        for p in patterns:
            if p.name not in pattern_counts:
                pattern_counts[p.name] = 0
            pattern_counts[p.name] += 1

        # Recent patterns (last 20)
        recent_patterns = pattern_list[-20:] if len(pattern_list) > 20 else pattern_list

        # Summary
        summary = {
            'total_patterns': len(patterns),
            'bullish': bullish_count,
            'bearish': bearish_count,
            'neutral': neutral_count,
            'bias': 'bullish' if bullish_count > bearish_count else ('bearish' if bearish_count > bullish_count else 'neutral'),
            'pattern_ratio': round(bullish_count / max(bearish_count, 1), 2)
        }

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': len(df),
            'utc_offset': get_utc_offset(),
            'summary': summary,
            'patterns': pattern_list,
            'recent_patterns': recent_patterns,
            'pattern_counts': pattern_counts,
            'candles': candles,
            'annotations': self._build_annotations(patterns)
        }

    def _build_annotations(self, patterns) -> Dict[str, List]:
        """Build chart annotations for patterns"""
        markers = []

        for p in patterns:
            colors = {
                'bullish': '#26a69a',
                'bearish': '#ef5350',
                'neutral': '#9e9e9e'
            }

            info = self.PATTERN_INFO.get(p.name, {})
            display_name = info.get('name', p.name)

            markers.append({
                'time': p.time // 1000,
                'position': 'belowBar' if p.type == 'bullish' else 'aboveBar',
                'color': colors.get(p.type, '#9e9e9e'),
                'shape': 'arrowUp' if p.type == 'bullish' else ('arrowDown' if p.type == 'bearish' else 'circle'),
                'text': display_name,
                'size': 1
            })

        return {
            'markers': markers,
            'zones': [],
            'lines': []
        }

    async def get_pattern_info(self) -> Dict[str, Any]:
        """Get all pattern definitions"""
        return self.PATTERN_INFO

    async def get_config(self) -> Dict[str, Any]:
        """Get candlestick pattern config with pattern info"""
        config = load_candlestick_config()

        # Build config with pattern info
        result = {}
        for pattern_code, info in self.PATTERN_INFO.items():
            result[pattern_code] = {
                'name': info['name'],
                'type': info['type'],
                'enabled': config.get(pattern_code, True),
                'description': info['description'],
                'bars': info['bars']
            }

        return result
