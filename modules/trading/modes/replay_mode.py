#!/usr/bin/env python3
"""
modules/trading/modes/replay_mode.py
SuperBot - Replay Trading Mode (TradingView-style)
Author: SuperBot Team
Date: 2025-11-28
Versiyon: 3.0.0

Replay Trading Mode - Independent calculation in the style of BacktestEngine.

REPLAY MODE:
- Historical data (Parquet files)
- Candle-by-candle playback
- Custom indicator calculation (IndicatorManager)
- Custom signal generation (vectorized_conditions)
- Entry/exit loglama (WHY ENTRY/WHY EXIT)
- Speed control (0.5x, 1x, 2x, 4x)
- Pause/Resume

ARCHITECTURE:
- TradingEngine/WebUI → ReplayMode (TEK KAYNAK)
- ReplayMode performs its own calculations.
- CLI/WebUI only provides visualization.

PURPOSE:
- Trading education
- Strategy analysis
- Answering the question "Why was this trade opened?"

NOTE: Use BacktestEngine for performance testing!

Usage:
    mode = ReplayMode(config, logger)
    await mode.load_data("data/parquets/BTCUSDT/BTCUSDT_5m_2024.parquet")
    await mode.initialize()  # Calculate indicators

    await mode.play()  # Starts playback (signals + positions)

    mode.pause()       # Duraklat
    mode.resume()      # Continue
    mode.set_speed(2)  # Set speed to 2 x

Dependencies:
    - python>=3.12
    - pandas
    - numpy
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# Add project root to path for direct execution
import sys
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from modules.trading.modes.base_mode import (
    BaseMode,
    ModeType,
    OrderSide,
    OrderType,
    OrderStatus,
    Candle,
    Order,
    OrderResult,
    Position,
    Balance
)

# Backtest types for trade tracking
from modules.backtest.backtest_types import Trade, PositionSide, ExitReason


class ReplayMode(BaseMode):
    """
    Replay Trading Mode - Independent calculation in the style of BacktestEngine.

    Loads data from Parquet files, calculates indicators, generates signals,
    and replays candles sequentially. For each candle:
    - Indicator values
    - Signal status (LONG/SHORT/NONE)
    - WHY ENTRY/WHY EXIT explanation

    ARCHITECTURE:
    - ReplayMode TEK KAYNAK (single source of truth)
    - The TradingEngine/WebUI only provides visualization.
    - No duplicate calculations are performed.

    KONTROLLER:
    - play() : Start replay
    - pause()   : Duraklat
    - resume()  : Continue
    - stop()    : Stop
    - set_speed(n) : Set speed (0.5 x, 1 x, 2 x, 4 x)
    - seek(index)  : Belirli index'e git
    """

    # Speed multipliers
    AVAILABLE_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        """
        Args:
            config: Mode config
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Strategy reference
        self.strategy = config.get("strategy")
        self.debug = config.get("debug", False)

        # Performance mode: If False, indicator/signal calculation is skipped.
        self._calculate_indicators = config.get("calculate_indicators", True)

        # Replay state - Raw candle data
        self._data: Dict[str, List[Candle]] = {}  # symbol_tf → candles
        self._current_index: Dict[str, int] = defaultdict(int)
        self._current_candle: Dict[str, Candle] = {}

        # Playback control
        self._speed = 1.0
        self._paused = False
        self._playing = False

        # Callbacks
        self._candle_callbacks: List[Callable] = []

        # ═══════════════════════════════════════════════════════════════════
        # INDICATOR & SIGNAL STATE (BacktestEngine-style)
        # ═══════════════════════════════════════════════════════════════════
        self._df_data: Dict[str, pd.DataFrame] = {}  # symbol_tf → DataFrame (with indicators)
        self._indicators: Dict[str, Dict[str, np.ndarray]] = {}  # symbol_tf → {ind_name: values}
        self._long_signals: Dict[str, List[int]] = {}  # symbol_tf → signal indices
        self._short_signals: Dict[str, List[int]] = {}  # symbol_tf → signal indices
        self._signal_dict: Dict[str, Dict[int, int]] = {}  # symbol_tf → {idx: side} (1=long, -1=short)

        # ═══════════════════════════════════════════════════════════════════
        # VIRTUAL TRADING (like paper mode)
        # ═══════════════════════════════════════════════════════════════════
        self._initial_balance = config.get("initial_balance", 10000.0)
        self._balance = self._initial_balance
        self._available_balance = self._initial_balance
        self._positions: Dict[str, Position] = {}  # symbol → Position
        self._active_positions: List[Dict] = []  # Active position dicts for simulation
        self._trades: List[Trade] = []  # Completed trades
        self._trade_counter = 0

        # Slippage/fee (from strategy or config)
        if self.strategy:
            backtest_params = getattr(self.strategy, 'backtest_parameters', {})
            self.fee_rate = backtest_params.get("commission", 0.0004)
            self.slippage_rate = backtest_params.get("max_slippage", 0.0005)
        else:
            self.fee_rate = config.get("fee_rate", 0.0004)
            self.slippage_rate = config.get("slippage_rate", 0.0005)

        # Stats
        self._stats = {
            "total_candles": 0,
            "processed_candles": 0,
            "trades": 0,
            "long_signals": 0,
            "short_signals": 0
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.REPLAY
    
    @property
    def is_live_data(self) -> bool:
        return False  # Previous data
    
    @property
    def is_real_execution(self) -> bool:
        return False  # Simulate
    
    @property
    def speed(self) -> float:
        return self._speed
    
    @property
    def is_playing(self) -> bool:
        return self._playing
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def progress(self) -> float:
        """Progress percentage (0-100)"""
        if not self._data:
            return 0.0
        
        total = sum(len(candles) for candles in self._data.values())
        processed = sum(self._current_index.values())
        
        return (processed / total * 100) if total > 0 else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════
    
    async def load_parquet(
        self,
        filepath: str,
        symbol: str = None,
        timeframe: str = "5m"
    ):
        """
        Load data from a Parquet file.

        Args:
            filepath: Path to the Parquet file
            symbol: Symbol name (if None, it is extracted from the file name)
            timeframe: Timeframe
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            # Extract the symbol name
            if not symbol:
                symbol = path.stem.split("_")[0].upper()

            self.log(f"📂 Loading Parquet: {filepath}")

            # Load the data
            df = pd.read_parquet(filepath)

            # Column mapping (support different formats)
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                    col_map[col] = col_lower
                elif col_lower == 'open_time':
                    col_map[col] = 'open_time'
                elif col_lower == 'timestamp':
                    col_map[col] = 'timestamp'

            df = df.rename(columns=col_map)

            # Timestamp handling
            if 'open_time' in df.columns:
                df['timestamp'] = df['open_time']
            elif 'timestamp' not in df.columns:
                # Index'ten al
                if hasattr(df.index, 'to_series'):
                    df['timestamp'] = df.index.to_series().apply(
                        lambda x: int(x.timestamp() * 1000) if hasattr(x, 'timestamp') else 0
                    )
                else:
                    df['timestamp'] = 0

            # Convert to candles (legacy support)
            candles = []
            for idx, row in df.iterrows():
                ts = row.get('timestamp', 0)
                if hasattr(ts, 'timestamp'):
                    ts_ms = int(ts.timestamp() * 1000)
                elif isinstance(ts, (int, float)):
                    ts_ms = int(ts) if ts > 1e12 else int(ts * 1000)
                else:
                    ts_ms = 0

                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts_ms,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                    is_closed=True
                )
                candles.append(candle)

            # Save - both the Candle list and the DataFrame
            key = f"{symbol}_{timeframe}"
            self._data[key] = candles
            self._df_data[key] = df.reset_index(drop=True)  # DataFrame for indicators
            self._current_index[key] = 0
            self._stats["total_candles"] = sum(len(c) for c in self._data.values())

            self.log(f"   ✅ {len(candles)} candle loaded ({symbol} {timeframe})")

        except ImportError:
            self.log("❌ pandas is not installed: pip install pandas", "error")
        except Exception as e:
            self.log(f"❌ Parquet loading error: {e}", "error")
            import traceback
            traceback.print_exc()
    
    # ═══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════
    
    async def initialize(self) -> None:
        """
        Initialize replay mode.

        At this stage:
        1. All indicators are calculated (like in BacktestEngine)
        2. Signals are generated (vectorized_conditions)
        3. It becomes ready for replay
        """
        self.log("🎬 ReplayMode is starting...")
        self.log(f"   Balance: ${self._balance:,.2f}")
        self.log(f"   Speed: {self._speed}x")
        self.log(f"   Total candles: {self._stats['total_candles']}")

        # If there is no strategy, use only playback mode
        if not self.strategy:
            self.log("⚠️  No strategy - only playback mode (no indicator/signal)")
            self._initialized = True
            self._running = True
            return

        # Performance mode: indicator calculation disabled
        if not self._calculate_indicators:
            self.log("⚡ Performance mode - indicator/signal calculation skipped")
            self._initialized = True
            self._running = True
            return

        # ═══════════════════════════════════════════════════════════════════
        # 1. INDICATOR CALCULATION (BacktestEngine-style)
        # ═══════════════════════════════════════════════════════════════════
        self.log("📊 Indicators are being calculated...")
        await self._calculate_all_indicators()

        # ═══════════════════════════════════════════════════════════════════
        # 2. SIGNAL GENERATION (vectorized_conditions)
        # ═══════════════════════════════════════════════════════════════════
        self.log("🎯 Signals are being generated...")
        await self._generate_all_signals()

        self._initialized = True
        self._running = True
        self.log("✅ ReplayMode ready")
        self.log(f"   📊 LONG signals: {self._stats['long_signals']}")
        self.log(f"   📊 SHORT signals: {self._stats['short_signals']}")
    
    async def shutdown(self) -> None:
        """Disable replay mode"""
        self.log("🛑 ReplayMode is being disabled...")
        self._playing = False
        self._running = False
        
        pnl = self._balance - self._initial_balance
        self.log(f"   Processed: {self._stats['processed_candles']} candles")
        self.log(f"   Trades: {self._stats['trades']}")
        self.log(f"   Final PnL: ${pnl:+,.2f}")
        self.log("✅ ReplayMode disabled")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PLAYBACK CONTROL
    # ═══════════════════════════════════════════════════════════════════════
    
    async def play(self):
        """
        Start replay - With signal check and position simulation.

        Her candle'da:
        1. Signal check (LONG/SHORT)
        2. If signal exists -> log WHY ENTRY + open position
        3. If open position exists -> check SL/TP
        4. Exit olursa → WHY EXIT logu
        """
        if not self._data:
            self.log("⚠️ Data has not been loaded! Call load_parquet() first.", "warning")
            return

        self._playing = True
        self._paused = False
        self.log("▶️ Replay started")

        # If a strategy exists, create a StrategyExecutor (for exit logic)
        strategy_executor = None
        if self.strategy:
            try:
                from components.strategies.strategy_executor import StrategyExecutor
                strategy_executor = StrategyExecutor(self.strategy, logger=self.logger)
            except ImportError:
                pass

        warmup = getattr(self.strategy, 'warmup_period', 200) if self.strategy else 0

        # Main replay loop
        while self._playing:
            # Pause control
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            # Process for each symbol
            any_remaining = False

            for key, candles in self._data.items():
                idx = self._current_index[key]

                if idx < len(candles):
                    any_remaining = True
                    candle = candles[idx]
                    self._current_candle[candle.symbol] = candle

                    # Get DataFrame row (for indicator values)
                    df = self._df_data.get(key)
                    row = df.iloc[idx] if df is not None and idx < len(df) else None

                    # ═══════════════════════════════════════════════════════════
                    # 1. CHECK OPEN POSITIONS (SL/TP)
                    # ═══════════════════════════════════════════════════════════
                    if self.strategy and strategy_executor and idx >= warmup:
                        await self._check_exits(
                            candle, row, idx, key, strategy_executor
                        )

                    # ═══════════════════════════════════════════════════════════
                    # 2. SIGNAL CHECK AND ENTRY
                    # ═══════════════════════════════════════════════════════════
                    if self.strategy and idx >= warmup:
                        signal = self.get_signal_at(idx, candle.symbol, candle.timeframe)

                        if signal != 0 and row is not None:
                            # WHY ENTRY logu
                            self._log_entry_signal(row, signal, candle.symbol)

                            # Open position
                            await self._open_position(candle, row, signal, idx)

                    # ═══════════════════════════════════════════════════════════
                    # 3. DISPLAY (her N candle'da bir)
                    # ═══════════════════════════════════════════════════════════
                    self._display_candle(candle, idx, row)

                    # Call callbacks
                    for callback in self._candle_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(candle)
                            else:
                                callback(candle)
                        except Exception as e:
                            self.log(f"⚠️ Callback error: {e}", "warning")

                    # Index ilerlet
                    self._current_index[key] = idx + 1
                    self._stats["processed_candles"] += 1

            # No candles left
            if not any_remaining:
                break

            # Wait according to speed
            await asyncio.sleep(1.0 / self._speed)

        self._playing = False
        self.log("⏹️ Replay completed")
        self._print_summary()
    
    def pause(self):
        """Replay duraklat"""
        self._paused = True
        self.log("⏸️ Replay paused")
    
    def resume(self):
        """Continue replay"""
        self._paused = False
        self.log("▶️ Replay is continuing")
    
    def stop(self):
        """Stop replay"""
        self._playing = False
        self.log("⏹️ Replay durduruldu")
    
    def set_speed(self, speed: float):
        """
        Set playback speed.
        
        Args:
            speed: Speed multiplier (0.25, 0.5, 1, 2, 4, 10)
        """
        speed = max(0.25, min(10.0, speed))
        self._speed = speed
        self.log(f"⏩ Speed: {speed}x")
    
    def seek(self, index: int, symbol: str = None, timeframe: str = "5m"):
        """
        Belirli index'e git
        
        Args:
            index: Candle index
            symbol: Symbol (if None, uses the first symbol)
            timeframe: Timeframe
        """
        if symbol:
            key = f"{symbol}_{timeframe}"
        else:
            key = list(self._data.keys())[0] if self._data else None
        
        if key and key in self._data:
            max_idx = len(self._data[key]) - 1
            self._current_index[key] = max(0, min(index, max_idx))
            self.log(f"⏭️ Seek: index {self._current_index[key]}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ═══════════════════════════════════════════════════════════════════════

    def _display_candle(
        self,
        candle: Candle,
        idx: int = 0,
        row: Optional[pd.Series] = None,
        verbose: bool = False
    ):
        """
        Displays candle information in a compact format.

        Default: 1 line per 10 candles.
        verbose=True: Her candle'da tam detay
        """
        # Progress hesapla
        total = self._stats.get("total_candles", 1)
        processed = self._stats.get("processed_candles", 0)

        # Compact output every 10 candles (or always in verbose mode)
        if not verbose and processed % 10 != 0:
            return

        timestamp = datetime.fromtimestamp(candle.timestamp / 1000).strftime("%Y-%m-%d %H:%M")

        # Price change
        change_pct = ((candle.close - candle.open) / candle.open) * 100 if candle.open > 0 else 0
        change_emoji = "🟢" if change_pct >= 0 else "🔴"

        # Signal info
        signal = self.get_signal_at(idx, candle.symbol, candle.timeframe)
        signal_str = ""
        if signal > 0:
            signal_str = " 🎯LONG"
        elif signal < 0:
            signal_str = " 🎯SHORT"

        # Active positions
        pos_str = ""
        if self._active_positions:
            pos_str = f" │ Pos: {len(self._active_positions)}"

        # Speed/pause
        speed_str = f"{self._speed}x" if self._speed != 1.0 else "1x"
        pause_str = " ⏸️" if self._paused else ""

        print(f"🎬 {timestamp} │ {candle.symbol} ${candle.close:,.2f} {change_emoji}{change_pct:+.2f}%{signal_str} │ Bar {processed}/{total} ({self.progress:.1f}%){pos_str} │ {speed_str}{pause_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    async def _open_position(
        self,
        candle: Candle,
        row: pd.Series,
        signal: int,
        idx: int
    ) -> None:
        """
        Create a new position.

        Args:
            candle: Current candle
            row: DataFrame row with indicators
            signal: 1 = LONG, -1 = SHORT
            idx: Current index
        """
        # Is there already a position for the same symbol?
        for pos in self._active_positions:
            if pos['symbol'] == candle.symbol:
                return  # There is already a position

        # Entry price (slippage dahil)
        if signal > 0:
            entry_price = candle.close * (1 + self.slippage_rate)
        else:
            entry_price = candle.close * (1 - self.slippage_rate)

        # Position sizing
        position_size = self._calculate_position_size(entry_price)

        # SL/TP hesapla
        sl_price, tp_price = self._calculate_sl_tp(entry_price, signal, row)

        # Create position
        self._trade_counter += 1
        position = {
            'id': self._trade_counter,
            'symbol': candle.symbol,
            'side': 'LONG' if signal > 0 else 'SHORT',
            'entry_price': entry_price,
            'quantity': position_size / entry_price,
            'entry_time': candle.timestamp,
            'entry_idx': idx,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'highest_price': entry_price,  # For trailing stop
            'lowest_price': entry_price,
        }

        self._active_positions.append(position)
        self._stats['trades'] += 1

    async def _check_exits(
        self,
        candle: Candle,
        row: Optional[pd.Series],
        idx: int,
        key: str,
        strategy_executor
    ) -> None:
        """
        Check open positions - SL/TP/Trailing

        Args:
            candle: Current candle
            row: DataFrame row
            idx: Current index
            key: symbol_timeframe key
            strategy_executor: StrategyExecutor instance
        """
        for position in self._active_positions[:]:  # Copy to allow removal
            if position['symbol'] != candle.symbol:
                continue

            current_price = candle.close
            entry_price = position['entry_price']
            side = position['side']

            # Update highest/lowest for trailing
            if side == 'LONG':
                position['highest_price'] = max(position['highest_price'], candle.high)
            else:
                position['lowest_price'] = min(position['lowest_price'], candle.low)

            # Exit control with StrategyExecutor
            df = self._df_data.get(key)
            if df is not None and strategy_executor:
                data_to_pass = df.iloc[:idx + 1]

                exit_result = strategy_executor.evaluate_exit(
                    symbol=candle.symbol,
                    position=position,
                    data=data_to_pass,
                    current_price=current_price
                )

                # SL update (trailing/break-even)
                if exit_result.get('updated_sl'):
                    position['sl_price'] = exit_result['updated_sl']

            # Exit check
            exit_reason = None
            exit_price = current_price

            sl_price = position.get('sl_price', 0)
            tp_price = position.get('tp_price', 0)

            if side == 'LONG':
                if sl_price > 0 and candle.low <= sl_price:
                    exit_reason = ExitReason.STOP_LOSS
                    exit_price = sl_price
                elif tp_price > 0 and candle.high >= tp_price:
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_price = tp_price
            else:  # SHORT
                if sl_price > 0 and candle.high >= sl_price:
                    exit_reason = ExitReason.STOP_LOSS
                    exit_price = sl_price
                elif tp_price > 0 and candle.low <= tp_price:
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_price = tp_price

            # If there is an exit, close the position.
            if exit_reason:
                await self._close_position(position, exit_price, exit_reason, candle.timestamp)

    async def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_reason: ExitReason,
        exit_time: int
    ) -> None:
        """Close the position and create a Trade"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']

        # PnL hesapla
        if side == 'LONG':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = (pnl_pct / 100) * (quantity * entry_price)

        # Fee
        fee = (quantity * exit_price) * self.fee_rate
        net_pnl = pnl_usd - fee

        # WHY EXIT logu
        self._log_exit_signal(position, exit_price, exit_reason, net_pnl, pnl_pct)

        # Create trade
        trade = Trade(
            id=position['id'],
            symbol=position['symbol'],
            side=PositionSide.LONG if side == 'LONG' else PositionSide.SHORT,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_time=datetime.fromtimestamp(position['entry_time'] / 1000),
            exit_time=datetime.fromtimestamp(exit_time / 1000),
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            fee_usd=fee,
            net_pnl_usd=net_pnl,
            exit_reason=exit_reason
        )
        self._trades.append(trade)

        # Update balance
        self._balance += net_pnl

        # Remove the position
        self._active_positions.remove(position)

    def _calculate_position_size(self, price: float) -> float:
        """Position size hesapla (USD)"""
        if not self.strategy:
            return min(self._balance * 0.1, 1000)  # Default %10

        risk_mgmt = getattr(self.strategy, 'risk_management', None)
        if risk_mgmt:
            pct = getattr(risk_mgmt, 'position_percent_size', 10)
            return self._balance * (pct / 100)

        return min(self._balance * 0.1, 1000)

    def _calculate_sl_tp(
        self,
        entry_price: float,
        signal: int,
        row: pd.Series
    ) -> tuple[float, float]:
        """Calculate SL (Stop Loss) and TP (Take Profit) prices"""
        if not self.strategy:
            return 0, 0

        exit_strat = getattr(self.strategy, 'exit_strategy', None)
        if not exit_strat:
            return 0, 0

        sl_pct = getattr(exit_strat, 'stop_loss_percent', 1) / 100
        tp_pct = getattr(exit_strat, 'take_profit_percent', 2) / 100

        if signal > 0:  # LONG
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # SHORT
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        return sl_price, tp_price

    def _print_summary(self) -> None:
        """Print replay summary"""
        self.log("\n" + "=" * 60)
        self.log("📊 REPLAY SUMMARY")
        self.log("=" * 60)

        total_trades = len(self._trades)
        if total_trades == 0:
            self.log("   No trades executed")
            return

        wins = sum(1 for t in self._trades if t.pnl_usd > 0)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = sum(t.net_pnl_usd for t in self._trades)
        gross_profit = sum(t.pnl_usd for t in self._trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self._trades if t.pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        self.log(f"   Total Trades: {total_trades}")
        self.log(f"   Wins: {wins} | Losses: {losses}")
        self.log(f"   Win Rate: {win_rate:.1f}%")
        self.log(f"   Profit Factor: {profit_factor:.2f}")
        self.log(f"   Total PnL: ${total_pnl:+,.2f}")
        self.log(f"   Final Balance: ${self._balance:,.2f}")
        self.log("=" * 60 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════
    
    def on_candle(self, callback: Callable) -> None:
        """Register candle callback"""
        self._candle_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORDER EXECUTION (Simulated)
    # ═══════════════════════════════════════════════════════════════════════
    
    async def execute_order(self, order: Order) -> OrderResult:
        """Simulated order execution"""
        candle = self._current_candle.get(order.symbol)
        if not candle:
            return self._failed_order(order, "Candle not found")
        
        current_price = candle.close
        
        # Slippage
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_rate)
        else:
            fill_price = current_price * (1 - self.slippage_rate)
        
        # Fee
        notional = order.quantity * fill_price
        fee = notional * self.fee_rate
        
        # Order ID
        order_id = f"REPLAY-{uuid.uuid4().hex[:8].upper()}"
        
        self._stats["trades"] += 1
        
        return OrderResult(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(),
            raw=None
        )
    
    def _failed_order(self, order: Order, reason: str) -> OrderResult:
        """Failed order"""
        return OrderResult(
            order_id="FAILED",
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.REJECTED,
            quantity=order.quantity,
            filled_quantity=0.0,
            price=0.0,
            fee=0.0,
            timestamp=datetime.now(),
            raw={"error": reason}
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACCOUNT INFO
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_balance(self) -> Balance:
        return Balance(
            total=self._balance,
            available=self._available_balance,
            in_position=self._balance - self._available_balance,
            unrealized_pnl=0.0,
            currency="USDT"
        )
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)
    
    async def get_positions(self) -> List[Position]:
        return list(self._positions.values())
    
    async def get_current_price(self, symbol: str) -> float:
        candle = self._current_candle.get(symbol)
        return candle.close if candle else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA ACCESS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def subscribe(self, symbols: List[str], timeframe: str = "5m") -> None:
        pass
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        pass
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> List[Candle]:
        """Candles up to the current index"""
        key = f"{symbol}_{timeframe}"
        if key not in self._data:
            return []
        
        idx = self._current_index.get(key, 0)
        candles = self._data[key][:idx + 1]
        
        return candles[-limit:] if len(candles) > limit else candles
    
    def get_statistics(self) -> Dict[str, Any]:
        """Replay istatistikleri"""
        pnl = self._balance - self._initial_balance

        return {
            "mode": "replay",
            "speed": self._speed,
            "playing": self._playing,
            "paused": self._paused,
            "progress": self.progress,
            **self._stats,
            "pnl": pnl,
            "trades": len(self._trades)
        }

    # ═══════════════════════════════════════════════════════════════════════
    # INDICATOR CALCULATION (BacktestEngine-style)
    # ═══════════════════════════════════════════════════════════════════════

    async def _calculate_all_indicators(self) -> None:
        """
        Calculate indicators for all datasets.

        Same as the _calculate_indicators() method in BacktestEngine.
        Retrieves indicators from the technical_parameters in Strategy and calculates them.
        """
        if not self.strategy:
            return

        try:
            from components.strategies.helpers.strategy_indicator_bridge import (
                create_indicator_manager_from_strategy,
                format_indicator_results_for_strategy
            )
        except ImportError as e:
            self.log(f"❌ Indicator modules could not be loaded: {e}", "error")
            return

        for key, df in self._df_data.items():
            symbol, timeframe = key.rsplit("_", 1)

            # Create IndicatorManager
            indicator_logger = self.logger if self.debug else None
            indicator_manager = create_indicator_manager_from_strategy(
                strategy=self.strategy,
                logger=indicator_logger
            )

            # Calculate all indicators
            raw_results = {}
            for ind_name, ind_instance in indicator_manager.indicators.items():
                try:
                    if hasattr(ind_instance, 'calculate_batch'):
                        result = ind_instance.calculate_batch(df)
                    elif hasattr(ind_instance, 'calculate'):
                        result = ind_instance.calculate(df)
                    else:
                        continue

                    # Save the result
                    if isinstance(result, pd.Series):
                        raw_results[ind_name] = result
                    elif isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            raw_results[f"{ind_name}_{col}" if col != ind_name else col] = result[col]
                    elif isinstance(result, dict):
                        for k, v in result.items():
                            if hasattr(v, 'values'):
                                raw_results[k] = v

                except Exception as e:
                    if self.debug:
                        self.log(f"   ⚠️  {ind_name} could not be calculated: {e}", "warning")

            # Bridge formatting (smart aliasing)
            formatted = format_indicator_results_for_strategy(raw_results, timeframe, ohlcv_data=df)

            # Add to DataFrame and save as a numpy array
            indicators_dict = {}
            for ind_key, value in formatted.items():
                if hasattr(value, 'values'):
                    df[ind_key] = value.values
                    indicators_dict[ind_key] = value.values
                else:
                    df[ind_key] = value
                    indicators_dict[ind_key] = value

            self._indicators[key] = indicators_dict
            self._df_data[key] = df

            if self.debug:
                self.log(f"   ✅ {key}: {len(indicators_dict)} indicator calculated")

    # ═══════════════════════════════════════════════════════════════════════
    # SIGNAL GENERATION (vectorized_conditions)
    # ═══════════════════════════════════════════════════════════════════════

    async def _generate_all_signals(self) -> None:
        """
        Generate signals for all datasets.

        Same as the _generate_signals() method in BacktestEngine.
        Generates signals from the entry_conditions in Strategy.
        """
        if not self.strategy:
            return

        try:
            from modules.backtest.vectorized_conditions import build_conditions_mask
        except ImportError as e:
            self.log(f"❌ vectorized_conditions could not be loaded: {e}", "error")
            return

        warmup = getattr(self.strategy, 'warmup_period', 200)

        for key, df in self._df_data.items():
            # LONG signals
            long_conditions = self.strategy.entry_conditions.get('long', [])
            if long_conditions:
                long_mask = build_conditions_mask(
                    long_conditions,
                    df,
                    warmup,
                    logic='AND',
                    debug=self.debug
                )
                long_indices = long_mask[long_mask].index.tolist()
            else:
                long_indices = []

            # SHORT signals
            short_conditions = self.strategy.entry_conditions.get('short', [])
            if short_conditions:
                short_mask = build_conditions_mask(
                    short_conditions,
                    df,
                    warmup,
                    logic='AND',
                    debug=self.debug
                )
                short_indices = short_mask[short_mask].index.tolist()
            else:
                short_indices = []

            # Save
            self._long_signals[key] = long_indices
            self._short_signals[key] = short_indices

            # Signal dictionary (for fast lookup)
            signal_dict = {}
            for idx in long_indices:
                signal_dict[idx] = 1  # LONG
            for idx in short_indices:
                signal_dict[idx] = -1  # SHORT
            self._signal_dict[key] = signal_dict

            # Update statistics
            self._stats["long_signals"] += len(long_indices)
            self._stats["short_signals"] += len(short_indices)

            if self.debug:
                self.log(f"   ✅ {key}: LONG={len(long_indices)}, SHORT={len(short_indices)}")

    # ═══════════════════════════════════════════════════════════════════════
    # WHY ENTRY/EXIT LOGGING
    # ═══════════════════════════════════════════════════════════════════════

    def _log_entry_signal(self, row: pd.Series, signal: int, symbol: str) -> None:
        """
        Entry signal for WHY ENTRY log

        A copy of the same method in BacktestEngine.
        Displays the value of each condition.
        """
        side = "LONG" if signal > 0 else "SHORT"

        # Timestamp handling - support different formats
        ts = row.get('timestamp', row.get('open_time', 0))
        if hasattr(ts, 'timestamp'):  # pandas Timestamp
            timestamp = ts.to_pydatetime()
        elif isinstance(ts, (int, float)):
            timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)
        else:
            timestamp = datetime.now()

        self.log(f"\n{'='*60}")
        self.log(f"🎯 {side} SIGNAL @ {timestamp.strftime('%Y-%m-%d %H:%M')}")
        self.log(f"   Symbol: {symbol}")
        self.log(f"   Price: ${row['close']:,.2f}")

        # Entry conditions
        self.log(f"\n   📋 WHY ENTRY:")
        entry_conditions = self.strategy.entry_conditions.get('long' if signal > 0 else 'short', [])

        for condition in entry_conditions:
            left = condition[0]
            operator = condition[1]
            right = condition[2]

            # Get values
            left_val = row.get(left, left) if isinstance(left, str) else left
            right_val = row.get(right, right) if isinstance(right, str) else right

            # Format
            left_str = f"{left}={left_val:.4f}" if isinstance(left_val, float) else str(left)
            right_str = f"{right}={right_val:.4f}" if isinstance(right_val, float) else str(right)

            self.log(f"      ✅ {left_str} {operator} {right_str}")

        self.log(f"{'='*60}\n")

    def _log_exit_signal(
        self,
        position: Dict,
        exit_price: float,
        exit_reason: ExitReason,
        pnl: float,
        pnl_pct: float
    ) -> None:
        """
        Exit for WHY EXIT log.

        A copy of the same method in BacktestEngine.
        """
        side = position.get('side', 'UNKNOWN')
        symbol = position.get('symbol', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)

        emoji = "🟢" if pnl >= 0 else "🔴"

        self.log(f"\n{'='*60}")
        self.log(f"{emoji} {side} CLOSED - {exit_reason.value}")
        self.log(f"   Symbol: {symbol}")
        self.log(f"   Entry: ${entry_price:,.2f}")
        self.log(f"   Exit: ${exit_price:,.2f}")
        self.log(f"   PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

        # WHY EXIT?
        self.log(f"\n   📋 WHY EXIT:")

        if exit_reason == ExitReason.STOP_LOSS:
            self.log(f"      🛑 Stop Loss hit")
            self.log(f"         - Entry: ${entry_price:,.2f}")
            self.log(f"         - Stop Loss: ${position.get('sl_price', 0):,.2f}")
            self.log(f"         - Current: ${exit_price:,.2f}")

        elif exit_reason == ExitReason.TAKE_PROFIT:
            self.log(f"      🎯 Take Profit hit")
            self.log(f"         - Entry: ${entry_price:,.2f}")
            self.log(f"         - Take Profit: ${position.get('tp_price', 0):,.2f}")
            self.log(f"         - Current: ${exit_price:,.2f}")

        elif exit_reason == ExitReason.TRAILING_STOP:
            self.log(f"      📈 Trailing Stop hit")
            self.log(f"         - Entry: ${entry_price:,.2f}")
            self.log(f"         - Trailing Stop: ${position.get('sl_price', 0):,.2f}")

        elif exit_reason == ExitReason.BREAK_EVEN:
            self.log(f"      ⚖️ Break-even Stop hit")
            self.log(f"         - Entry: ${entry_price:,.2f}")

        elif exit_reason == ExitReason.MANUAL:
            self.log(f"      ⏱️ Position Timeout")

        self.log(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # INDICATOR/SIGNAL DATA ACCESS (for WebUI/CLI)
    # ═══════════════════════════════════════════════════════════════════════

    def get_indicators_at(self, index: int, symbol: str, timeframe: str) -> Dict[str, float]:
        """
        Returns the indicator values at the specified index.

        The WebUI/CLI uses this method to retrieve indicator values.
        It does NOT perform duplicate calculations - it returns pre-calculated values.
        """
        key = f"{symbol}_{timeframe}"
        if key not in self._indicators:
            return {}

        indicators = self._indicators[key]
        result = {}

        for ind_name, values in indicators.items():
            if isinstance(values, np.ndarray) and index < len(values):
                val = values[index]
                if not np.isnan(val):
                    result[ind_name] = float(val)

        return result

    def get_signal_at(self, index: int, symbol: str, timeframe: str) -> int:
        """
        Returns the signal at the specified index.

        Returns:
            1 = LONG signal
            -1 = SHORT signal
            0 = No signal
        """
        key = f"{symbol}_{timeframe}"
        if key not in self._signal_dict:
            return 0

        return self._signal_dict[key].get(index, 0)

    def get_data_with_indicators(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with an indicator.

        WebUI can retrieve all data using this method.
        """
        key = f"{symbol}_{timeframe}"
        return self._df_data.get(key)

    def get_trades(self) -> List[Trade]:
        """Returns completed trades"""
        return self._trades


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ReplayMode Test")
    print("=" * 60)
    
    async def test():
        config = {"initial_balance": 10000}
        
        print("\n1️⃣ Creating ReplayMode:")
        mode = ReplayMode(config)
        await mode.initialize()
        
        print(f"   Mode type: {mode.mode_type}")
        print(f"   Speed: {mode.speed}x")
        
        print("\n2️⃣ Mock candle data:")
        # Manual candle injection (parquet olmadan test)
        import random
        
        candles = []
        base_price = 95000.0
        base_time = 1700000000000
        
        for i in range(10):
            price = base_price + random.uniform(-500, 500)
            candle = Candle(
                symbol="BTCUSDT",
                timeframe="5m",
                timestamp=base_time + (i * 300000),
                open=price,
                high=price + random.uniform(0, 100),
                low=price - random.uniform(0, 100),
                close=price + random.uniform(-50, 50),
                volume=random.uniform(100, 1000),
                is_closed=True
            )
            candles.append(candle)
        
        mode._data["BTCUSDT_5m"] = candles
        mode._stats["total_candles"] = len(candles)
        
        print(f"   Loaded: {len(candles)} candles")
        
        print("\n3️⃣ Playback control:")
        mode.set_speed(4)  # 4 x speed
        print(f"   Speed: {mode.speed}x")
        
        print("\n4️⃣ Playing (first 3 candles)...")
        mode._playing = True
        for i in range(3):
            candle = candles[i]
            mode._display_candle(candle)
            mode._current_index["BTCUSDT_5m"] = i + 1
            await asyncio.sleep(0.2)
        
        print("\n5️⃣ Statistics:")
        stats = mode.get_statistics()
        print(f"   Progress: {stats['progress']:.1f}%")
        print(f"   Speed: {stats['speed']}x")
        
        await mode.shutdown()
        print("\n✅ Test completed!")
    
    import asyncio
    asyncio.run(test())
    
    print("=" * 60)