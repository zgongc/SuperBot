"""
Replay Trading Service
WebUI for ReplayMode wrapper - a thin layer

ReplayMode (Trading Engine) does all the work:
- Parquet loading
- Playback control
- Strategy execution
- Order/Position/Balance management
- Calculate indicator.

ReplayService only:
- Session management (dict: session_id -> ReplayMode)
- Chart data formatting (WebUI'ye uygun JSON)
- List of strategies.

Session Persistence:
- To prevent sessions from being lost when Flask reloads
- Session ID = strategy_id (same strategy = same session)
- Session info is saved as JSON
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import uuid
import importlib.util
import asyncio
import numpy as np
import json

from .base_service import BaseService


# Session cache file
SESSION_CACHE_FILE = Path("data/replay_sessions.json")


@dataclass
class ReplaySessionInfo:
    """Session metadata (for WebUI)"""
    id: str
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    current_position: int = 0
    total_bars: int = 0
    status: str = 'paused'
    speed: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    indicator_names: List[str] = field(default_factory=list)
    indicator_config: Dict = field(default_factory=dict)  # Strategy indicator config

    def to_dict(self) -> Dict:
        d = asdict(self)
        # indicator_config'i JSON serializable yap
        d.pop('indicator_config', None)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ReplaySessionInfo':
        """Create a ReplaySessionInfo object from a dictionary."""
        return cls(
            id=d['id'],
            strategy_id=d['strategy_id'],
            strategy_name=d['strategy_name'],
            symbol=d['symbol'],
            timeframe=d['timeframe'],
            start_date=d['start_date'],
            end_date=d['end_date'],
            current_position=d.get('current_position', 0),
            total_bars=d.get('total_bars', 0),
            status=d.get('status', 'paused'),
            speed=d.get('speed', 1.0),
            created_at=d.get('created_at', datetime.now().isoformat()),
            indicator_names=d.get('indicator_names', []),
            indicator_config=d.get('indicator_config', {})
        )


class ReplayService(BaseService):
    """
    Replay Trading Service - ReplayMode Wrapper

    WebUI ↔ ReplayMode bridge

    Sessions are strategy-based:
    - session_id = strategy_id (e.g., "bluish")
    - ReplayMode is recreated in Flask reload
    - Position information is loaded from JSON
    """

    def __init__(self, data_manager=None, logger=None, parquets_engine=None, strategy_manager=None):
        super().__init__(data_manager, logger)
        self.parquets_engine = parquets_engine
        self.strategy_manager = strategy_manager
        self.template_path = Path("components/strategies/templates/")

        # Active sessions: session_id → (ReplayMode, ReplaySessionInfo)
        self._sessions: Dict[str, tuple] = {}

        # Load the session cache (only metadata, ReplayMode lazy load)
        self._session_cache: Dict[str, Dict] = {}
        self._load_session_cache()

    def _load_session_cache(self):
        """Load saved session information"""
        try:
            if SESSION_CACHE_FILE.exists():
                with open(SESSION_CACHE_FILE, 'r') as f:
                    self._session_cache = json.load(f)
                if self.logger:
                    self.logger.info(f"📂 Session cache loaded: {len(self._session_cache)} sessions")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Session cache could not be loaded: {e}")
            self._session_cache = {}

    def _save_session_cache(self):
        """Save session information to a file"""
        try:
            SESSION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SESSION_CACHE_FILE, 'w') as f:
                json.dump(self._session_cache, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Session cache kaydedilemedi: {e}")

    def _load_strategy_class(self, strategy_id: str):
        """Loads the strategy file and returns the Strategy class"""
        template_path = self.template_path / f"{strategy_id}.py"
        if not template_path.exists():
            raise FileNotFoundError(f"Strategy not found: {strategy_id}")

        spec = importlib.util.spec_from_file_location("strategy_module", template_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        strategy = module.Strategy()
        return strategy

    async def get_strategies(self) -> List[Dict[str, Any]]:
        """Gets the list of existing strategies"""
        strategies = []

        if self.template_path.exists():
            for file_path in self.template_path.glob("*.py"):
                if file_path.name.startswith('__'):
                    continue

                try:
                    strategy = self._load_strategy_class(file_path.stem)

                    # Symbol: SymbolConfig'den al
                    symbol = "BTCUSDT"
                    if hasattr(strategy, 'symbols') and strategy.symbols:
                        sym_config = strategy.symbols[0]
                        if hasattr(sym_config, 'symbol') and hasattr(sym_config, 'quote'):
                            first_symbol = sym_config.symbol[0] if isinstance(sym_config.symbol, list) else sym_config.symbol
                            symbol = f"{first_symbol}{sym_config.quote}"

                    strategy_info = {
                        'id': file_path.stem,
                        'name': strategy.strategy_name if hasattr(strategy, 'strategy_name') else file_path.stem,
                        'description': strategy.description if hasattr(strategy, 'description') else '',
                        'symbol': symbol,
                        'timeframe': strategy.primary_timeframe if hasattr(strategy, 'primary_timeframe') else '15m',
                        'start_date': strategy.backtest_start_date if hasattr(strategy, 'backtest_start_date') else '',
                        'end_date': strategy.backtest_end_date if hasattr(strategy, 'backtest_end_date') else '',
                    }
                    strategies.append(strategy_info)

                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"⚠️ Strategy loading error {file_path}: {e}")
                    strategies.append({
                        'id': file_path.stem,
                        'name': file_path.stem.replace('_', ' ').title(),
                        'description': f"Error: {str(e)[:50]}",
                        'symbol': '',
                        'timeframe': '',
                        'start_date': '',
                        'end_date': '',
                    })

        strategies.sort(key=lambda x: x['name'])
        return strategies

    async def create_session(
        self,
        strategy_id: str,
        override_timeframe: str = None,
        override_start_date: str = None,
        override_end_date: str = None,
        calculate_indicators: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new replay session.

        Creates a ReplayMode instance and loads data.

        Args:
            strategy_id: Strategy ID
            override_timeframe: Timeframe override
            override_start_date: Override the start date.
            override_end_date: Override the end date.
            calculate_indicators: If True, calculate indicators (default: False - for performance).
        """
        from modules.trading.modes.replay_mode import ReplayMode

        if self.logger:
            self.logger.info(f"🎮 Creating replay session: {strategy_id}")

        # 1. Load strategy
        strategy = self._load_strategy_class(strategy_id)

        # 2. Symbol, timeframe, dates
        symbol = "BTCUSDT"
        if hasattr(strategy, 'symbols') and strategy.symbols:
            sym_config = strategy.symbols[0]
            if hasattr(sym_config, 'symbol') and hasattr(sym_config, 'quote'):
                first_symbol = sym_config.symbol[0] if isinstance(sym_config.symbol, list) else sym_config.symbol
                symbol = f"{first_symbol}{sym_config.quote}"

        timeframe = override_timeframe or getattr(strategy, 'primary_timeframe', '15m')
        start_date = override_start_date or getattr(strategy, 'backtest_start_date', '2024-01-01T00:00')
        end_date = override_end_date or getattr(strategy, 'backtest_end_date', '2024-03-01T00:00')
        initial_balance = getattr(strategy, 'initial_balance', 10000.0)
        warmup_period = getattr(strategy, 'warmup_period', 200)

        if self.logger:
            self.logger.info(f"   Symbol: {symbol}, TF: {timeframe}")
            self.logger.info(f"   Start: {start_date}, End: {end_date}")

        # 3. Create ReplayMode
        mode_config = {
            "strategy": strategy,
            "initial_balance": initial_balance,
            "calculate_indicators": calculate_indicators,  # Performance: False ise indicator hesaplamaz
        }
        replay_mode = ReplayMode(mode_config, self.logger)

        if self.logger:
            self.logger.info(f"   Indicators: {'enabled' if calculate_indicators else 'disabled (performance mode)'}")

        # 4. Load data from Parquet
        if self.parquets_engine:
            try:
                df = await self.parquets_engine.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    warmup_candles=warmup_period,
                    utc_offset=3
                )

                # Load data into ReplayMode (DataFrame -> Candle list)
                await self._load_dataframe_to_mode(replay_mode, df, symbol, timeframe)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Data loading error: {e}")
                raise RuntimeError(f"Data could not be loaded: {e}")
        else:
            raise RuntimeError("ParquetsEngine is not defined!")

        # 5. ReplayMode initialize
        await replay_mode.initialize()

        # 6. Get the indicator configuration and names (from the strategy)
        indicator_names = []
        indicator_config = {}
        if hasattr(strategy, 'technical_parameters') and hasattr(strategy.technical_parameters, 'indicators'):
            indicator_config = strategy.technical_parameters.indicators
            indicator_names = list(indicator_config.keys())

        # 7. Create session information
        # Session ID = strategy_id (so that it can be continued after reload)
        session_id = strategy_id

        # Get the previous position from the cache
        cached_position = 0
        if session_id in self._session_cache:
            cached_position = self._session_cache[session_id].get('current_position', 0)
            if self.logger:
                self.logger.info(f"📂 Previous position loaded: {cached_position}")

        session_info = ReplaySessionInfo(
            id=session_id,
            strategy_id=strategy_id,
            strategy_name=getattr(strategy, 'strategy_name', strategy_id),
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            current_position=cached_position,
            total_bars=replay_mode._stats.get("total_candles", 0),
            indicator_names=indicator_names,
            indicator_config=indicator_config
        )

        # Update the ReplayMode's position as well.
        key = f"{symbol}_{timeframe}"
        if cached_position > 0 and key in replay_mode._data:
            replay_mode._current_index[key] = min(cached_position, len(replay_mode._data[key]) - 1)

        # 8. Save (memory + file)
        self._sessions[session_id] = (replay_mode, session_info)
        self._session_cache[session_id] = session_info.to_dict()
        self._save_session_cache()

        if self.logger:
            self.logger.info(f"✅ Replay session created: {session_id}")
            self.logger.info(f"   Bars: {session_info.total_bars}, Position: {cached_position}")

        return session_info.to_dict()

    async def _load_dataframe_to_mode(self, mode, df, symbol: str, timeframe: str):
        """Load DataFrame into ReplayMode"""
        from modules.trading.modes.base_mode import Candle

        candles = []
        for _, row in df.iterrows():
            ts = row.get('timestamp', row.get('open_time', 0))
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=int(ts),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                is_closed=True
            )
            candles.append(candle)

        key = f"{symbol}_{timeframe}"
        mode._data[key] = candles
        mode._df_data[key] = df.reset_index(drop=True)  # DataFrame for indicators - KRITIK!
        mode._current_index[key] = 0
        mode._stats["total_candles"] = len(candles)

    async def _restore_session(self, session_id: str) -> bool:
        """Restore session from cache"""
        if session_id not in self._session_cache:
            return False

        cached = self._session_cache[session_id]
        strategy_id = cached.get('strategy_id', session_id)

        if self.logger:
            self.logger.info(f"🔄 Session restore ediliyor: {session_id}")

        try:
            # Recreate the session (using the position from the cache)
            await self.create_session(strategy_id)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Session restore error: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Session bilgilerini getir"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                if await self._restore_session(session_id):
                    pass  # Restore successful, continue
                else:
                    return None
            else:
                return None

        mode, info = self._sessions[session_id]

        # Get the current status
        info.current_position = sum(mode._current_index.values())
        info.status = 'playing' if mode.is_playing else ('paused' if mode.is_paused else 'stopped')
        info.speed = mode.speed

        return info.to_dict()

    async def get_candles(self, session_id: str, start: int = 0, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Get candle data for the chart"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                return None

        mode, info = self._sessions[session_id]

        # Get candles from ReplayMode
        symbol = info.symbol
        timeframe = info.timeframe
        key = f"{symbol}_{timeframe}"

        all_candles = mode._data.get(key, [])
        current_pos = mode._current_index.get(key, 0)

        # Show only up to the current_position
        visible_end = min(current_pos + 1, len(all_candles))
        visible_start = max(0, visible_end - limit)

        visible_candles = all_candles[visible_start:visible_end]

        # Candle format (for Lightweight Charts)
        candles = []
        volumes = []
        timestamps = []

        for c in visible_candles:
            ts = int(c.timestamp / 1000)
            candles.append({
                'time': ts,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
            })
            volumes.append({
                'time': ts,
                'value': c.volume,
                'color': 'rgba(38, 166, 154, 0.5)' if c.close >= c.open else 'rgba(239, 83, 80, 0.5)'
            })
            timestamps.append(ts)

        # Get indicators from ReplayMode (to avoid duplicate calculations)
        indicators = self._get_indicators_from_mode(
            mode, info, timestamps, visible_start, visible_end
        )

        # Signal markers (from ReplayMode - real-time signals)
        signal_markers = self._get_signals_from_mode(
            mode, info, timestamps, visible_start, visible_end
        )

        # Trade markers (ReplayMode trades + backtest results)
        markers, trades = self._get_trade_markers(
            mode, info.symbol, info.timeframe, timestamps,
            visible_start, visible_end, all_candles
        )

        # Merge signal and trade markers (signal_markers take precedence)
        all_markers = signal_markers + markers

        return {
            'candles': candles,
            'volumes': volumes,
            'indicators': indicators,
            'markers': all_markers,
            'trades': trades,
            'indicator_names': info.indicator_names,
            'visible_start': visible_start,
            'visible_end': visible_end,
            'current_position': current_pos,
            'total_bars': info.total_bars
        }

    async def play(self, session_id: str) -> Dict[str, Any]:
        """Start replay"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                raise ValueError("Session not found")

        mode, info = self._sessions[session_id]

        # Start playback in the background (non-blocking)
        # Will monitor the status using WebUI polling.
        asyncio.create_task(self._play_loop(session_id))

        return {'status': 'playing', 'current_position': sum(mode._current_index.values())}

    async def _play_loop(self, session_id: str):
        """Background play loop"""
        if session_id not in self._sessions:
            return

        mode, info = self._sessions[session_id]
        mode._playing = True
        mode._paused = False

        key = f"{info.symbol}_{info.timeframe}"

        while mode._playing and session_id in self._sessions:
            if mode._paused:
                await asyncio.sleep(0.1)
                continue

            # Step forward
            current_idx = mode._current_index.get(key, 0)
            total = len(mode._data.get(key, []))

            if current_idx >= total - 1:
                mode._playing = False
                break

            mode._current_index[key] = current_idx + 1
            mode._stats["processed_candles"] = sum(mode._current_index.values())

            # Wait according to speed
            await asyncio.sleep(1.0 / mode.speed)

        mode._playing = False

    async def pause(self, session_id: str) -> Dict[str, Any]:
        """Replay duraklat"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                raise ValueError("Session not found")

        mode, info = self._sessions[session_id]
        mode.pause()

        return {'status': 'paused', 'current_position': sum(mode._current_index.values())}

    async def step(self, session_id: str, direction: int = 1) -> Dict[str, Any]:
        """Go one bar forward/backward"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                raise ValueError("Session not found")

        mode, info = self._sessions[session_id]
        key = f"{info.symbol}_{info.timeframe}"

        current_idx = mode._current_index.get(key, 0)
        total = len(mode._data.get(key, []))

        new_idx = max(0, min(current_idx + direction, total - 1))
        mode._current_index[key] = new_idx

        # Update the cache (save every 10 steps - for performance)
        if new_idx % 10 == 0:
            self._session_cache[session_id]['current_position'] = new_idx
            self._save_session_cache()

        # Update state
        state = await self._get_state_dict(mode, info)

        return {
            'current_position': new_idx,
            'total_bars': total,
            'state': state
        }

    async def seek(self, session_id: str, position: int) -> Dict[str, Any]:
        """Belirli pozisyona git"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                raise ValueError("Session not found")

        mode, info = self._sessions[session_id]
        key = f"{info.symbol}_{info.timeframe}"

        total = len(mode._data.get(key, []))
        new_idx = max(0, min(position, total - 1))
        mode._current_index[key] = new_idx

        # Update the cache
        self._session_cache[session_id]['current_position'] = new_idx
        self._save_session_cache()

        state = await self._get_state_dict(mode, info)

        return {
            'current_position': new_idx,
            'total_bars': total,
            'state': state
        }

    async def set_speed(self, session_id: str, speed: float) -> Dict[str, Any]:
        """Set speed"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                raise ValueError("Session not found")

        mode, info = self._sessions[session_id]
        mode.set_speed(speed)

        return {'speed': mode.speed}

    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Gets the current state"""
        # Try to restore if not in memory
        if session_id not in self._sessions:
            if session_id in self._session_cache:
                await self._restore_session(session_id)
            if session_id not in self._sessions:
                return None

        mode, info = self._sessions[session_id]
        return await self._get_state_dict(mode, info)

    async def _get_state_dict(self, mode, info) -> Dict[str, Any]:
        """Create a state dictionary from ReplayMode"""
        balance = await mode.get_balance()
        positions = await mode.get_positions()

        key = f"{info.symbol}_{info.timeframe}"
        current_idx = mode._current_index.get(key, 0)

        # Price from the current candle
        candles = mode._data.get(key, [])
        current_price = candles[current_idx].close if current_idx < len(candles) else 0

        # Get indicator values for the current bar from ReplayMode
        current_indicators = {}
        if hasattr(mode, 'get_indicators_at'):
            current_indicators = mode.get_indicators_at(current_idx, info.symbol, info.timeframe)

        # Signal bilgisi
        signal_info = None
        if hasattr(mode, 'get_signal_at'):
            signal_info = mode.get_signal_at(current_idx, info.symbol, info.timeframe)

        return {
            'current_position': current_idx,
            'total_bars': len(candles),
            'status': 'playing' if mode.is_playing else ('paused' if mode.is_paused else 'stopped'),
            'speed': mode.speed,
            'balance': balance.total if balance else 10000,
            'equity': balance.total if balance else 10000,
            'positions': [vars(p) if hasattr(p, '__dict__') else p for p in positions],
            'current_price': current_price,
            'indicators': current_indicators,
            'signal': signal_info,
            'drawdown': 0.0,
        }

    async def get_trades(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the trade list"""
        if session_id not in self._sessions:
            return []

        mode, info = self._sessions[session_id]

        # Get trade history from ReplayMode
        if hasattr(mode, 'get_trades'):
            return mode.get_trades()
        return []

    async def delete_session(self, session_id: str) -> None:
        """Delete session"""
        if session_id in self._sessions:
            mode, info = self._sessions[session_id]

            # Disable ReplayMode
            if mode.is_playing:
                mode.stop()
            await mode.shutdown()

            del self._sessions[session_id]

            if self.logger:
                self.logger.info(f"🗑️  Replay session silindi: {session_id}")

    # ========== Indicator Data Access (ReplayMode'dan) ==========

    def _get_indicators_from_mode(
        self,
        mode,
        info,
        timestamps: List[int],
        visible_start: int,
        visible_end: int
    ) -> Dict[str, List[Dict]]:
        """
        Get pre-calculated indicator values from ReplayMode.

        NO DUPLICATE CALCULATION - ReplayMode performs all calculations.
        This method only converts to chart format.

        Args:
            timestamps: List of timestamps for the displayed candles (len = visible_end - visible_start)
            visible_start: Starting index in the data array
            visible_end: Ending index in the data array
        """
        indicators = {}

        if not hasattr(mode, '_indicators'):
            return indicators

        key = f"{info.symbol}_{info.timeframe}"
        mode_indicators = mode._indicators.get(key, {})

        if not mode_indicators:
            return indicators

        # Convert each indicator to chart format
        for ind_name, values in mode_indicators.items():
            try:
                if not isinstance(values, np.ndarray):
                    continue

                # Get the values within the visible range
                # timestamps[ts_idx] ↔ values[data_idx] mapping
                chart_data = []
                for ts_idx, data_idx in enumerate(range(visible_start, min(visible_end, len(values)))):
                    if ts_idx < len(timestamps):
                        val = values[data_idx]
                        if not np.isnan(val):
                            chart_data.append({
                                'time': timestamps[ts_idx],
                                'value': float(val)
                            })

                if chart_data:
                    indicators[ind_name] = chart_data

            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Indicator format error ({ind_name}): {e}")

        return indicators

    def _get_signals_from_mode(
        self,
        mode,
        info,
        timestamps: List[int],
        visible_start: int,
        visible_end: int
    ) -> List[Dict]:
        """
        Get signal markers from ReplayMode.

        Signals are calculated in ReplayMode.initialize().
        This method only converts to chart marker format.
        """
        markers = []

        if not hasattr(mode, '_signal_dict'):
            return markers

        key = f"{info.symbol}_{info.timeframe}"
        signal_dict = mode._signal_dict.get(key, {})

        if not signal_dict:
            return markers

        # Convert signals to chart marker format
        for idx, signal in signal_dict.items():
            if visible_start <= idx < visible_end:
                ts_idx = idx - visible_start
                if ts_idx < len(timestamps):
                    is_long = signal > 0
                    markers.append({
                        'time': timestamps[ts_idx],
                        'position': 'belowBar' if is_long else 'aboveBar',
                        'color': '#26a69a' if is_long else '#ef5350',
                        'shape': 'arrowUp' if is_long else 'arrowDown',
                        'text': 'LONG' if is_long else 'SHORT'
                    })

        # Sort by time
        markers.sort(key=lambda x: x['time'])

        return markers

    # ========== Replay Trade Formatting ==========

    def _format_replay_trades(
        self,
        replay_trades: List,
        visible_timestamps: List[int],
        visible_start: int,
        visible_end: int
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Converts the ReplayMode._trades list to a chart marker format.

        Args:
            replay_trades: Trade list from ReplayMode
            visible_timestamps: Visible candle timestamps
            visible_start: Visible starting index
            visible_end: Visible ending index

        Returns:
            (trades, markers): Trade list and chart markers.
        """
        trades = []
        markers = []

        if not visible_timestamps:
            return trades, markers

        min_ts = min(visible_timestamps)
        max_ts = max(visible_timestamps)
        ts_set = set(visible_timestamps)

        def snap_to_candle(ts: int) -> int:
            """Snap the timestamp to the nearest candle."""
            if ts in ts_set:
                return ts
            closest = min(visible_timestamps, key=lambda x: abs(x - ts))
            return closest

        for trade in replay_trades:
            # Get values from the Trade object
            entry_ts = trade.entry_time
            exit_ts = trade.exit_time

            # Convert timestamps to seconds
            if hasattr(entry_ts, 'timestamp'):
                entry_ts_sec = int(entry_ts.timestamp())
            elif isinstance(entry_ts, (int, float)):
                entry_ts_sec = int(entry_ts / 1000) if entry_ts > 1e12 else int(entry_ts)
            else:
                continue

            exit_ts_sec = None
            if exit_ts:
                if hasattr(exit_ts, 'timestamp'):
                    exit_ts_sec = int(exit_ts.timestamp())
                elif isinstance(exit_ts, (int, float)):
                    exit_ts_sec = int(exit_ts / 1000) if exit_ts > 1e12 else int(exit_ts)

            # Is this trade within the visible range?
            if not (min_ts <= entry_ts_sec <= max_ts or (exit_ts_sec and min_ts <= exit_ts_sec <= max_ts)):
                continue

            # Side
            is_long = str(trade.side).upper() in ['LONG', 'POSITIONSIDE.LONG', '1']
            pnl = trade.pnl_usd if hasattr(trade, 'pnl_usd') else 0
            pnl_pct = trade.pnl_pct if hasattr(trade, 'pnl_pct') else 0

            # Trade dictionary
            trades.append({
                'id': trade.id if hasattr(trade, 'id') else len(trades) + 1,
                'side': 'LONG' if is_long else 'SHORT',
                'entry_time': entry_ts_sec,
                'exit_time': exit_ts_sec,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price if hasattr(trade, 'exit_price') else None,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'exit_reason': trade.exit_reason.value if hasattr(trade, 'exit_reason') and trade.exit_reason else '',
                'win': pnl > 0
            })

            # Entry marker
            if min_ts <= entry_ts_sec <= max_ts:
                snapped_entry = snap_to_candle(entry_ts_sec)
                markers.append({
                    'time': snapped_entry,
                    'position': 'belowBar' if is_long else 'aboveBar',
                    'color': '#26a69a' if is_long else '#ef5350',
                    'shape': 'arrowUp' if is_long else 'arrowDown',
                    'text': ''
                })

            # Exit marker
            if exit_ts_sec and min_ts <= exit_ts_sec <= max_ts:
                snapped_exit = snap_to_candle(exit_ts_sec)
                is_profit = pnl >= 0
                pnl_text = f'+{pnl:.0f}' if is_profit else f'{pnl:.0f}'

                markers.append({
                    'time': snapped_exit,
                    'position': 'aboveBar' if is_long else 'belowBar',
                    'color': '#26a69a' if is_profit else '#ef5350',
                    'shape': 'circle',
                    'text': pnl_text
                })

        # Sort by time
        markers.sort(key=lambda x: x['time'])

        return trades, markers

    # ========== Trade Markers (ReplayMode + Backtest) ==========

    def _get_trade_markers(
        self,
        mode,
        symbol: str,
        timeframe: str,
        visible_timestamps: List[int],
        visible_start: int,
        visible_end: int,
        all_candles: List
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Get trade markers - first ReplayMode, then backtest

        Returns:
            (markers, trades): Chart markers and trade list.

        Lightweight Charts marker format:
        {
            time: unix_timestamp,
            position: 'belowBar' | 'aboveBar',
            color: '#color',
            shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square',
            text: 'label'
        }
        """
        markers = []
        trades = []

        # ═══════════════════════════════════════════════════════════════════
        # 1. REPLAY MODE TRADES (live trades)
        # ═══════════════════════════════════════════════════════════════════
        if mode and hasattr(mode, '_trades') and mode._trades:
            trades, markers = self._format_replay_trades(
                mode._trades, visible_timestamps, visible_start, visible_end
            )
            if trades:
                # If there are replay trades, use them, don't look at the backtest.
                return markers, trades

        # ═══════════════════════════════════════════════════════════════════
        # 2. BACKTEST PARQUET (old backtest results)
        # ═══════════════════════════════════════════════════════════════════
        try:
            import pandas as pd
            from pathlib import Path

            # Find the most recent backtest file
            backtest_dir = Path("data/ai/features/backtest")
            if not backtest_dir.exists():
                return markers, trades

            # Find the backtest files for this symbol and timeframe
            pattern = f"backtest_{symbol}_{timeframe}_*.parquet"
            files = sorted(backtest_dir.glob(pattern), reverse=True)

            if not files:
                return markers, trades

            # Load the latest file
            latest_file = files[0]
            df = pd.read_parquet(latest_file)

            if df.empty:
                return markers, trades

            # Visible timestamp range
            if not visible_timestamps:
                return markers, trades

            min_ts = min(visible_timestamps)
            max_ts = max(visible_timestamps)
            ts_set = set(visible_timestamps)  # For fast lookup

            def snap_to_candle(ts: int) -> int:
                """Snap the timestamp to the nearest candle."""
                if ts in ts_set:
                    return ts
                # Find the nearest candle
                closest = min(visible_timestamps, key=lambda x: abs(x - ts))
                return closest

            # Filter trades
            for _, trade in df.iterrows():
                entry_ts = trade.get('entry_time')
                exit_ts = trade.get('exit_time')

                if not pd.notna(entry_ts):
                    continue

                # Millisecond to second
                entry_ts_sec = int(entry_ts / 1000) if entry_ts > 1e12 else int(entry_ts)
                exit_ts_sec = int(exit_ts / 1000) if pd.notna(exit_ts) and exit_ts > 1e12 else (int(exit_ts) if pd.notna(exit_ts) else None)

                # Is this trade within the visible range?
                if not (min_ts <= entry_ts_sec <= max_ts or (exit_ts_sec and min_ts <= exit_ts_sec <= max_ts)):
                    continue

                side = trade.get('side', 'LONG')
                is_long = str(side).upper() in ['LONG', 'BUY', '1']
                pnl = trade.get('pnl', 0)
                pnl = float(pnl) if pd.notna(pnl) else 0

                # Get SL/TP values
                stop_loss = float(trade.get('stop_loss', 0)) if pd.notna(trade.get('stop_loss')) else None
                take_profit = float(trade.get('take_profit', 0)) if pd.notna(trade.get('take_profit')) else None
                exit_reason = str(trade.get('exit_reason', '')) if pd.notna(trade.get('exit_reason')) else ''
                entry_price = float(trade.get('entry_price', 0))
                exit_price = float(trade.get('exit_price', 0)) if pd.notna(trade.get('exit_price')) else None
                pnl_percent = float(trade.get('pnl_percent', 0)) if pd.notna(trade.get('pnl_percent')) else 0
                break_even_activated = bool(trade.get('break_even_activated', False))
                is_partial_exit = bool(trade.get('is_partial_exit', False))
                partial_exit_level = int(trade.get('partial_exit_level', 0)) if pd.notna(trade.get('partial_exit_level')) else 0

                # Edit the exit reason
                display_exit_reason = exit_reason
                if is_partial_exit:
                    display_exit_reason = f'PE{partial_exit_level}'  # Partial Exit (PE1, PE2, ...)
                elif break_even_activated and exit_reason.upper() in ['SL', 'STOP_LOSS']:
                    display_exit_reason = 'BE'  # Break-even exit

                # Add to the trade list (with detailed information)
                trades.append({
                    'id': trade.get('trade_id', len(trades) + 1),
                    'side': 'LONG' if is_long else 'SHORT',
                    'entry_time': entry_ts_sec,
                    'exit_time': exit_ts_sec,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'exit_reason': display_exit_reason,
                    'break_even_activated': break_even_activated,
                    'is_partial_exit': is_partial_exit,
                    'partial_exit_level': partial_exit_level,
                    'win': bool(trade.get('win', False)),
                    'duration_minutes': int(trade.get('duration_minutes', 0)) if pd.notna(trade.get('duration_minutes')) else 0
                })

                # Entry marker - simple arrow symbol
                if min_ts <= entry_ts_sec <= max_ts:
                    snapped_entry = snap_to_candle(entry_ts_sec)
                    markers.append({
                        'time': snapped_entry,
                        'position': 'belowBar' if is_long else 'aboveBar',
                        'color': '#26a69a' if is_long else '#ef5350',
                        'shape': 'arrowUp' if is_long else 'arrowDown',
                        'text': ''  # Empty - will show the box
                    })

                # Exit marker - Show P&L
                if exit_ts_sec and min_ts <= exit_ts_sec <= max_ts:
                    snapped_exit = snap_to_candle(exit_ts_sec)
                    is_profit = pnl >= 0
                    pnl_text = f'+{pnl:.0f}' if is_profit else f'{pnl:.0f}'

                    markers.append({
                        'time': snapped_exit,
                        'position': 'aboveBar' if is_long else 'belowBar',
                        'color': '#26a69a' if is_profit else '#ef5350',
                        'shape': 'circle',
                        'text': pnl_text
                    })

            # Sort by time (Lightweight Charts requirement)
            markers.sort(key=lambda x: x['time'])

        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Trade markers could not be loaded: {e}")

        return markers, trades
