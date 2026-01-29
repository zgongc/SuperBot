#!/usr/bin/env python3
"""
modules/simple_train/backtest/trade_simulator.py
SuperBot - Simple Train Trade Simulator
Author: SuperBot Team
Date: 2026-01-15
Versiyon: 1.0.0

Simple_train for independent trade simulation.
Independent of BacktestEngine, it generates its own WIN/LOSE label.

Usage:
    from modules.simple_train.backtest.trade_simulator import TradeSimulator

    simulator = TradeSimulator(strategy_name="simple_rsi")
    results = simulator.simulate(df, signals)
    # results: DataFrame with 'label' column (1=WIN, 0=LOSE)

Features:
    - Takes TP/SL/BE/PE parameters from Strategy.
    - Takes indicators from IndicatorManager.
    - Calculates WIN/LOSE with forward simulation.
    - MTF support.

Dependencies:
    - python>=3.10
    - pandas>=2.0.0
    - numpy>=1.24.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# =============================================================================
# PATH SETUP
# =============================================================================

SIMPLE_TRAIN_ROOT = Path(__file__).parent.parent
SUPERBOT_ROOT = SIMPLE_TRAIN_ROOT.parent.parent

if str(SUPERBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERBOT_ROOT))

# =============================================================================
# LOGGER SETUP
# =============================================================================

try:
    from core.logger_engine import get_logger
    logger = get_logger("modules.simple_train.backtest.trade_simulator")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("trade_simulator")


# =============================================================================
# TYPES
# =============================================================================

class TradeResult(Enum):
    """Trade result."""
    WIN = 1
    LOSE = 0
    TIMEOUT = -1  # Reached the maximum number of bars


class ExitReason(Enum):
    """Exit reason."""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    BREAK_EVEN = "BE"
    PARTIAL_EXIT = "PE"
    TRAILING_STOP = "TS"
    TIMEOUT = "TIMEOUT"


@dataclass
class SimulatedTrade:
    """Simulated trade."""
    entry_idx: int
    entry_price: float
    side: str  # "LONG" or "SHORT"
    exit_idx: int = -1
    exit_price: float = 0.0
    exit_reason: ExitReason = ExitReason.TIMEOUT
    pnl_pct: float = 0.0
    result: TradeResult = TradeResult.TIMEOUT
    bars_held: int = 0

    # Exit tracking
    partial_exits: List[Dict] = field(default_factory=list)
    break_even_triggered: bool = False
    trailing_activated: bool = False


@dataclass
class ExitConfig:
    """Exit configuration (obtained from the strategy)."""
    # TP/SL
    tp_pct: float = 2.0
    sl_pct: float = 1.0

    # Break-even
    break_even_enabled: bool = False
    break_even_trigger_pct: float = 1.0
    break_even_offset: float = 0.1

    # Trailing stop
    trailing_enabled: bool = False
    trailing_activation_pct: float = 1.5
    trailing_callback_pct: float = 0.5

    # Partial exit
    partial_exit_enabled: bool = False
    partial_exit_levels: List[float] = field(default_factory=list)
    partial_exit_sizes: List[float] = field(default_factory=list)

    # Timeout
    max_bars: int = 200


# =============================================================================
# TRADE SIMULATOR
# =============================================================================

class TradeSimulator:
    """
    Independent trade simulation for Simple_train.

    It receives parameters from Strategy and generates its own WIN/LOSE label.
    """

    def __init__(
        self,
        strategy_name: str = "simple_rsi",
        exit_config: Optional[ExitConfig] = None
    ):
        """
        Initialize TradeSimulator.

        Args:
            strategy_name: Strategy name
            exit_config: Exit configuration (None=from strategy)
        """
        self.strategy_name = strategy_name
        self.strategy = None
        self.exit_config = exit_config
        self.indicator_manager = None

        # Load strategy and config
        self._load_strategy()

        logger.info(f"🎯 TradeSimulator started: {strategy_name}")

    def _load_strategy(self) -> None:
        """Load the strategy and get the exit config."""
        try:
            from components.strategies.strategy_manager import StrategyManager

            manager = StrategyManager()
            self.strategy, _ = manager.load_strategy(f"{self.strategy_name}_ai")

            # Exit config'i strategy'den al
            if self.exit_config is None:
                self.exit_config = self._extract_exit_config()

            logger.info(f"   📋 Strategy loaded: {self.strategy.strategy_name}")
            logger.info(f"   📊 TP: {self.exit_config.tp_pct}%, SL: {self.exit_config.sl_pct}%")

        except Exception as e:
            logger.warning(f"⚠️ Strategy could not be loaded: {e}, default config will be used")
            if self.exit_config is None:
                self.exit_config = ExitConfig()

    def _extract_exit_config(self) -> ExitConfig:
        """Extract the exit config from the Strategy."""
        config = ExitConfig()

        if not self.strategy:
            return config

        exit_strat = getattr(self.strategy, 'exit_strategy', None)
        if not exit_strat:
            return config

        # TP/SL
        config.tp_pct = getattr(exit_strat, 'take_profit_percent', 2.0)
        config.sl_pct = getattr(exit_strat, 'stop_loss_percent', 1.0)

        # Break-even
        config.break_even_enabled = getattr(exit_strat, 'break_even_enabled', False)
        config.break_even_trigger_pct = getattr(exit_strat, 'break_even_trigger_profit_percent', 1.0)
        config.break_even_offset = getattr(exit_strat, 'break_even_offset', 0.1)

        # Trailing stop
        config.trailing_enabled = getattr(exit_strat, 'trailing_stop_enabled', False)
        config.trailing_activation_pct = getattr(exit_strat, 'trailing_activation_profit_percent', 1.5)
        config.trailing_callback_pct = getattr(exit_strat, 'trailing_callback_percent', 0.5)

        # Partial exit
        config.partial_exit_enabled = getattr(exit_strat, 'partial_exit_enabled', False)
        config.partial_exit_levels = list(getattr(exit_strat, 'partial_exit_levels', []))
        config.partial_exit_sizes = list(getattr(exit_strat, 'partial_exit_sizes', []))

        return config

    def simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        max_bars: int = 200
    ) -> pd.DataFrame:
        """
        Perform trade simulations for all signals.

        Args:
            df: OHLCV DataFrame
            signals: Signal series (1=LONG, -1=SHORT, 0=NONE)
            max_bars: Maximum number of bars (timeout)

        Returns:
            DataFrame: Signal indices and labels
                - signal_idx: Signal bar index
                - side: LONG/SHORT
                - label: 1=WIN, 0=LOSE
                - pnl_pct: Profit and Loss percentage
                - exit_reason: Exit reason
                - bars_held: Number of bars held
        """
        self.exit_config.max_bars = max_bars

        results = []

        # Find signal indices
        signal_indices = signals[signals != 0].index.tolist()

        logger.info(f"📊 Simulating {len(signal_indices)} signals...")

        for idx in signal_indices:
            # DataFrame index to iloc
            try:
                iloc_idx = df.index.get_loc(idx)
            except KeyError:
                continue

            side = "LONG" if signals.loc[idx] == 1 else "SHORT"

            # Forward simulation
            trade = self._simulate_single_trade(df, iloc_idx, side)

            if trade:
                results.append({
                    'signal_idx': idx,
                    'entry_idx': trade.entry_idx,
                    'side': trade.side,
                    'label': trade.result.value if trade.result != TradeResult.TIMEOUT else 0,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason.value,
                    'bars_held': trade.bars_held,
                    'break_even_triggered': trade.break_even_triggered,
                    'trailing_activated': trade.trailing_activated
                })

        result_df = pd.DataFrame(results)

        if len(result_df) > 0:
            wins = (result_df['label'] == 1).sum()
            losses = (result_df['label'] == 0).sum()
            logger.info(f"   ✅ Result: {wins} WIN, {losses} LOSE ({wins/(wins+losses)*100:.1f}% WR)")

        return result_df

    def _simulate_single_trade(
        self,
        df: pd.DataFrame,
        entry_iloc: int,
        side: str
    ) -> Optional[SimulatedTrade]:
        """
        Forward simulation for a single trade.

        Args:
            df: OHLCV DataFrame
            entry_iloc: Entry bar iloc indexi
            side: "LONG" or "SHORT"

        Returns:
            SimulatedTrade: Simulated trade
        """
        if entry_iloc >= len(df) - 1:
            return None

        entry_price = df.iloc[entry_iloc]['close']

        trade = SimulatedTrade(
            entry_idx=entry_iloc,
            entry_price=entry_price,
            side=side
        )

        # TP/SL seviyeleri
        if side == "LONG":
            tp_price = entry_price * (1 + self.exit_config.tp_pct / 100)
            sl_price = entry_price * (1 - self.exit_config.sl_pct / 100)
        else:
            tp_price = entry_price * (1 - self.exit_config.tp_pct / 100)
            sl_price = entry_price * (1 + self.exit_config.sl_pct / 100)

        current_sl = sl_price
        trailing_high = entry_price if side == "LONG" else entry_price
        trailing_low = entry_price if side == "SHORT" else entry_price
        remaining_size = 1.0  # For partial exit
        partial_exit_idx = 0

        # Forward simulation
        max_idx = min(entry_iloc + self.exit_config.max_bars, len(df))

        for i in range(entry_iloc + 1, max_idx):
            row = df.iloc[i]
            high = row['high']
            low = row['low']
            close = row['close']

            trade.bars_held = i - entry_iloc

            # 1. Stop Loss control
            if side == "LONG":
                if low <= current_sl:
                    trade.exit_idx = i
                    trade.exit_price = current_sl
                    trade.exit_reason = ExitReason.STOP_LOSS if not trade.break_even_triggered else ExitReason.BREAK_EVEN
                    trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
                    trade.result = TradeResult.LOSE if trade.pnl_pct < 0 else TradeResult.WIN
                    return trade
            else:
                if high >= current_sl:
                    trade.exit_idx = i
                    trade.exit_price = current_sl
                    trade.exit_reason = ExitReason.STOP_LOSS if not trade.break_even_triggered else ExitReason.BREAK_EVEN
                    trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100
                    trade.result = TradeResult.LOSE if trade.pnl_pct < 0 else TradeResult.WIN
                    return trade

            # 2. Take Profit check
            if side == "LONG":
                if high >= tp_price:
                    trade.exit_idx = i
                    trade.exit_price = tp_price
                    trade.exit_reason = ExitReason.TAKE_PROFIT
                    trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
                    trade.result = TradeResult.WIN
                    return trade
            else:
                if low <= tp_price:
                    trade.exit_idx = i
                    trade.exit_price = tp_price
                    trade.exit_reason = ExitReason.TAKE_PROFIT
                    trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100
                    trade.result = TradeResult.WIN
                    return trade

            # 3. Break-even check
            if self.exit_config.break_even_enabled and not trade.break_even_triggered:
                if side == "LONG":
                    profit_pct = (high - entry_price) / entry_price * 100
                    if profit_pct >= self.exit_config.break_even_trigger_pct:
                        current_sl = entry_price * (1 + self.exit_config.break_even_offset / 100)
                        trade.break_even_triggered = True
                else:
                    profit_pct = (entry_price - low) / entry_price * 100
                    if profit_pct >= self.exit_config.break_even_trigger_pct:
                        current_sl = entry_price * (1 - self.exit_config.break_even_offset / 100)
                        trade.break_even_triggered = True

            # 4. Trailing stop control
            if self.exit_config.trailing_enabled:
                if side == "LONG":
                    if high > trailing_high:
                        trailing_high = high
                        profit_pct = (trailing_high - entry_price) / entry_price * 100
                        if profit_pct >= self.exit_config.trailing_activation_pct:
                            trade.trailing_activated = True
                            new_sl = trailing_high * (1 - self.exit_config.trailing_callback_pct / 100)
                            if new_sl > current_sl:
                                current_sl = new_sl
                else:
                    if low < trailing_low:
                        trailing_low = low
                        profit_pct = (entry_price - trailing_low) / entry_price * 100
                        if profit_pct >= self.exit_config.trailing_activation_pct:
                            trade.trailing_activated = True
                            new_sl = trailing_low * (1 + self.exit_config.trailing_callback_pct / 100)
                            if new_sl < current_sl:
                                current_sl = new_sl

            # 5. Partial exit check (only tracking, we are not doing a full exit)
            if self.exit_config.partial_exit_enabled and partial_exit_idx < len(self.exit_config.partial_exit_levels):
                target_pct = self.exit_config.partial_exit_levels[partial_exit_idx]

                if side == "LONG":
                    profit_pct = (high - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - low) / entry_price * 100

                if profit_pct >= target_pct:
                    size = self.exit_config.partial_exit_sizes[partial_exit_idx]
                    trade.partial_exits.append({
                        'bar': i,
                        'pct': target_pct,
                        'size': size
                    })
                    remaining_size -= size
                    partial_exit_idx += 1

        # Timeout - close at last bar
        trade.exit_idx = max_idx - 1
        trade.exit_price = df.iloc[trade.exit_idx]['close']
        trade.exit_reason = ExitReason.TIMEOUT

        if side == "LONG":
            trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
        else:
            trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100

        trade.result = TradeResult.WIN if trade.pnl_pct > 0 else TradeResult.LOSE

        return trade

    def get_exit_config(self) -> ExitConfig:
        """Returns the exit configuration."""
        return self.exit_config


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TradeSimulator Test")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    n = 500

    # Random walk price
    returns = np.random.randn(n) * 0.002
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 - np.random.rand(n) * 0.001),
        'high': prices * (1 + np.random.rand(n) * 0.005),
        'low': prices * (1 - np.random.rand(n) * 0.005),
        'close': prices,
        'volume': np.random.rand(n) * 1000000
    })

    # Random signals
    signals = pd.Series(0, index=df.index)
    signal_indices = np.random.choice(range(50, n-100), size=20, replace=False)
    for idx in signal_indices:
        signals.iloc[idx] = np.random.choice([1, -1])

    print(f"\n📊 Test data: {len(df)} bars, {(signals != 0).sum()} signals")

    # Test with default config
    print("\n1️⃣ Test with default config:")
    simulator = TradeSimulator(strategy_name="simple_rsi")
    results = simulator.simulate(df, signals)

    if len(results) > 0:
        print(f"   Total trade: {len(results)}")
        print(f"   Win rate: {(results['label'] == 1).mean() * 100:.1f}%")
        print(f"   Ortalama PnL: {results['pnl_pct'].mean():.2f}%")
        print(f"   Exit reasons: {results['exit_reason'].value_counts().to_dict()}")

    # Test with custom config
    print("\n2️⃣ Test with custom config:")
    custom_config = ExitConfig(
        tp_pct=3.0,
        sl_pct=1.5,
        break_even_enabled=True,
        break_even_trigger_pct=1.0,
        break_even_offset=0.1,
        trailing_enabled=True,
        trailing_activation_pct=2.0,
        trailing_callback_pct=0.5
    )

    simulator2 = TradeSimulator(strategy_name="simple_rsi", exit_config=custom_config)
    results2 = simulator2.simulate(df, signals)

    if len(results2) > 0:
        print(f"   Total trade: {len(results2)}")
        print(f"   Win rate: {(results2['label'] == 1).mean() * 100:.1f}%")
        print(f"   BE triggered: {results2['break_even_triggered'].sum()}")
        print(f"   Trailing activated: {results2['trailing_activated'].sum()}")

    print("\n✅ All tests completed!")
    print("=" * 60)
