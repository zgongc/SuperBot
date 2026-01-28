#!/usr/bin/env python3
"""
modules/backtest/backtest_types.py
SuperBot - Backtest Type Definitions
Author: SuperBot Team
Date: 2025-11-16
Version: 3.0.0

All data models and type definitions for backtest engine.

Features:
- Typed dataclasses (type safety)
- Comprehensive trade tracking
- Multi-timeframe & multi-symbol support
- Optimizer-friendly metrics

Usage:
    from modules.backtest.backtest_types import BacktestConfig, Trade, BacktestMetrics

    config = BacktestConfig(
        symbols=['BTCUSDT'],
        primary_timeframe='15m',
        ...
    )

Dependencies:
    - python>=3.10
    - dataclasses (stdlib)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class PositionSide(str, Enum):
    """Position direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    """Position exit reason"""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TRAILING_STOP = "TRAILING"
    BREAK_EVEN = "BE"
    SIGNAL = "SIGNAL"          # Opposite signal
    TIMEOUT = "TIMEOUT"        # Position timeout
    MANUAL = "MANUAL"          # Manual close (for testing)
    END_OF_DATA = "END"        # End of backtest


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Backtest configuration

    Comprehensive backtest settings with multi-TF and multi-symbol support.
    """
    # Symbol and timeframe
    symbols: List[str]                              # ['BTCUSDT', 'ETHUSDT']
    primary_timeframe: str                          # '15m'
    mtf_timeframes: List[str] = field(default_factory=list)  # ['15m', '1h', '4h']

    # Time range
    start_date: datetime = None
    end_date: datetime = None

    # Portfolio
    initial_balance: float = 10000.0

    # Data loading
    warmup_period: int = 200                        # Warmup candle count

    # Cost parameters
    commission_pct: float = 0.04                    # %0.04 (Binance maker)
    slippage_pct: float = 0.01                      # %0.01 average slippage
    spread_pct: float = 0.01                        # %0.01 bid-ask spread

    # Optional metadata
    strategy_name: Optional[str] = None
    strategy_version: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validation and default values"""
        # Primary TF should be in mtf_timeframes
        if self.primary_timeframe not in self.mtf_timeframes:
            self.mtf_timeframes.insert(0, self.primary_timeframe)

        # At least 1 symbol required
        if not self.symbols:
            raise ValueError("At least 1 symbol must be specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict"""
        return {
            'symbols': self.symbols,
            'primary_timeframe': self.primary_timeframe,
            'mtf_timeframes': self.mtf_timeframes,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_balance': self.initial_balance,
            'warmup_period': self.warmup_period,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct,
        }


# ============================================================================
# TRADE DATA
# ============================================================================

@dataclass
class Trade:
    """
    Completed trade record

    All information for detailed PnL tracking and analytics.
    """
    # Trade ID
    trade_id: int

    # Symbol and direction
    symbol: str
    side: PositionSide

    # Login Information
    entry_time: datetime
    entry_price: float

    #Exit information
    exit_time: datetime
    exit_price: float
    exit_reason: ExitReason

    # Position size
    quantity: float                     # Trade offered amount (e.g. 0.1 BTC)

    # Profit and Loss (gross - commission/premium before slippage)
    gross_pnl_usd: float               # Gross profit/loss (USD)
    gross_pnl_pct: float               # Gross profit/loss (%)

    # Profit and Loss (net - commission/slip loss after)
    net_pnl_usd: float                 # Net profit/loss ($)
    net_pnl_pct: float                 # Net profit/loss (%)

    # Maliyetler
    commission: float                   # Total commission ($)
    slippage: float                    # Total slippage ($)
    spread: float                      # Total spread cost ($)

    # Analytics (optional - advanced analysis)
    max_profit_usd: float = 0.0        # Trade during max profit
    max_profit_pct: float = 0.0
    max_loss_usd: float = 0.0          # Trade during max loss
    max_loss_pct: float = 0.0
    # Trade Duration (minutes)

    # Metadata (optional - for debugging purposes)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    entry_signal: Optional[str] = None  # Entry signal description
    break_even_activated: bool = False  # Is break-even activated?
    is_partial_exit: bool = False       # Partial exit mi?
    partial_exit_level: int = 0         # PE level (1, 2, 3...)

    def __post_init__(self):
        """Duration hesapla"""
        if self.entry_time and self.exit_time:
            # Handle both datetime and int/numeric types
            try:
                duration = self.exit_time - self.entry_time
                if hasattr(duration, 'total_seconds'):
                    self.duration_minutes = int(duration.total_seconds() / 60)
                else:
                    # Numeric difference (e.g., candle indices)
                    self.duration_minutes = int(duration)
            except:
                self.duration_minutes = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dict"""
        # Handle both datetime and numeric timestamps - convert to local timezone
        from core.config_engine import get_config
        from datetime import timedelta, timezone

        config = get_config()
        utc_offset = config.get('system', {}).get('utc_offset', 0)

        if hasattr(self.entry_time, 'isoformat'):
            # Convert to local timezone using config offset
            local_tz = timezone(timedelta(hours=utc_offset))
            entry_local = self.entry_time.astimezone(local_tz)
            entry_time_str = entry_local.isoformat()
        else:
            entry_time_str = str(self.entry_time)

        if hasattr(self.exit_time, 'isoformat'):
            local_tz = timezone(timedelta(hours=utc_offset))
            exit_local = self.exit_time.astimezone(local_tz)
            exit_time_str = exit_local.isoformat()
        else:
            exit_time_str = str(self.exit_time)

        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_time': entry_time_str,
            'entry_price': self.entry_price,
            'exit_time': exit_time_str,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason.value,
            'quantity': self.quantity,
            'gross_pnl_usd': self.gross_pnl_usd,
            'gross_pnl_pct': self.gross_pnl_pct,
            'net_pnl_usd': self.net_pnl_usd,
            'net_pnl_pct': self.net_pnl_pct,
            'commission': self.commission,
            'slippage': self.slippage,
            'duration_minutes': self.duration_minutes,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'break_even_activated': self.break_even_activated,
            'is_partial_exit': self.is_partial_exit,
            'partial_exit_level': self.partial_exit_level,
        }


# ============================================================================
# POSITION (Open Position Tracking for)
# ============================================================================

@dataclass
class Position:
    """
    Open position tracking

    For tracking open positions during trade simulation.
    """
    position_id: int
    symbol: str
    side: PositionSide

    # Entry
    entry_time: datetime
    entry_price: float
    quantity: float

    # Exit parameters
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None  # as a percentage
    break_even_activated: bool = False

    # Tracking
    # highest_price: float = 0.0          # For long positions, maximum price
    lowest_price: float = 999999.0      # For SHORT, minimum price

    # Maliyetler
    entry_commission: float = 0.0
    entry_slippage: float = 0.0

    def __post_init__(self):
        """Initial tracking values."""
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    def update_extremes(self, current_price: float):
        "Update highest/lowest price"
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

    def current_pnl_pct(self, current_price: float) -> float:
        """Current PnL percentage"""
        if self.side == PositionSide.LONG:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


# ============================================================================
#SIGNAL (Entry/exit signals)
# ============================================================================

@dataclass
class Signal:
    """
    Entry/Exit signal

    Signal information returned from vectorized signal generation.
    """
    timestamp: datetime
    signal_type: int            # 1=LONG, -1=SHORT, 0=NONE/EXIT
    symbol: str
    price: float

    # Optional - signal details
    confidence: Optional[float] = None      # Confidence score between 0 and 1
    reason: Optional[str] = None            # Signal reason explanation
    indicators: Optional[Dict] = None       # Indicators


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class BacktestMetrics:
    """
    Comprehensive backtest metrics

    All performance metrics for optimizer and analytics.
    """
    # Returns
    total_return_usd: float                 # Total profit/loss ($)
    total_return_pct: float                 # Total profit/loss (%)

    # Trade statistics
    total_trades: int                       # Total number of trades
    winners: int  # Number of winning trades
    losers: int                             # Number of trades lost
    win_rate: float                         # Win rate (%)

    # Win/Loss Details
    avg_win_usd: float                      # Average gain in USD
    avg_win_pct: float                      # Average win percentage (%)
    avg_loss_usd: float                     # Average loss in USD
    avg_loss_pct: float                     # Average loss percentage (%)
    largest_win_usd: float                  # Largest win in USD
    largest_loss_usd: float                 # Largest loss in USD

    # Ratio'lar
    profit_factor: float                    # Profit factor (gross profit / gross loss)
    sharpe_ratio: float                     # Sharpe ratio (risk-adjusted return)
    sortino_ratio: float                    # Sortino ratio (downside risk)
    calmar_ratio: float                     # Calmar ratio (return / max DD)

    # Drawdown
    max_drawdown_usd: float                 # Maximum drawdown ($)
    max_drawdown_pct: float                 # Maximum drawdown (%)
    avg_drawdown_pct: float                 # Ortalama drawdown (%)
    recovery_factor: float                  # Net profit / max DD

    # Maliyetler
    total_commission: float                 # Total commission
    total_slippage: float                   # Total slippage
    total_spread: float                     # Total spread cost
    total_costs: float                      # Total cost (commission + slippage + spread)

    # Other
    avg_trade_duration_minutes: float       # Average trade duration
    max_consecutive_wins: int               # Maximum consecutive wins
    max_consecutive_losses: int             # Maximum consecutive losses

    # Custom metric (for optimization purposes)
    custom_score: Optional[float] = None  # Custom scoring (e.g. PF × Return)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dict"""
        return {
            'total_return_usd': round(self.total_return_usd, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'total_trades': self.total_trades,
            'winners': self.winners,
            'losers': self.losers,
            'win_rate': round(self.win_rate, 2),
            'avg_win_usd': round(self.avg_win_usd, 2),
            'avg_loss_usd': round(self.avg_loss_usd, 2),
            'profit_factor': round(self.profit_factor, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'total_commission': round(self.total_commission, 2),
            'total_slippage': round(self.total_slippage, 2),
        }


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class BacktestResult:
    """
    Complete backtest result

    All backtest outputs in one object.
    """
    # Config
    config: BacktestConfig

    # Trade data
    trades: List[Trade]

    # Metrics
    metrics: BacktestMetrics

    # Equity curve
    equity_curve: List[Dict[str, Any]]      # [{time, balance, drawdown, pnl}, ...]

    # Performance
    execution_time_seconds: float

    # Optional - debugging
    signals: Optional[List[Signal]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dict"""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'total_trades': len(self.trades),
            'execution_time': round(self.execution_time_seconds, 2),
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': self.equity_curve,
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Backtest Types Test")
    print("=" * 60)

    # Test 1: BacktestConfig
    print("\n📋 Test 1: BacktestConfig")
    config = BacktestConfig(
        symbols=['BTCUSDT'],
        primary_timeframe='15m',
        mtf_timeframes=['1h', '4h'],
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 1),
        initial_balance=10000,
    )
    print(f"   ✔️ Configuration created: {config.symbols[0]}, {config.primary_timeframe}")
    print(f"   ✅ MTF timeframes: {config.mtf_timeframes}")

    # Test 2: Trade
    print("\n📊 Test 2: Trade")
    trade = Trade(
        trade_id=1,
        symbol='BTCUSDT',
        side=PositionSide.LONG,
        entry_time=datetime(2025, 1, 1, 10, 0),
        entry_price=100000,
        exit_time=datetime(2025, 1, 1, 12, 0),
        exit_price=105000,
        exit_reason=ExitReason.TAKE_PROFIT,
        quantity=0.1,
        gross_pnl_usd=500,
        gross_pnl_pct=5.0,
        net_pnl_usd=480,
        net_pnl_pct=4.8,
        commission=15,
        slippage=5,
    )
    print(f"   ✔️ Trade created: #{trade.trade_id}")
    print(f"   ✅ Side: {trade.side.value}, PnL: ${trade.net_pnl_usd}")
    print(f"   ✅ Duration: {trade.duration_minutes} minutes")

    # Test 3: Position
    print("\n📍 Test 3: Position")
    position = Position(
        position_id=1,
        symbol='BTCUSDT',
        side=PositionSide.LONG,
        entry_time=datetime(2025, 1, 1, 10, 0),
        entry_price=100000,
        quantity=0.1,
        stop_loss_price=98000,
        take_profit_price=105000,
    )
    position.update_extremes(102000)
    print(f"   ✔️ Position created: #{position.position_id}")
    print(f"   ✅ Entry: ${position.entry_price}, SL: ${position.stop_loss_price}")
    print(f"   ✅ Highest: ${position.highest_price}")
    print(f"   ✅ Current PnL: {position.current_pnl_pct(102000):.2f}%")

    # Test 4: BacktestMetrics
    print("\n📈 Test 4: BacktestMetrics")
    metrics = BacktestMetrics(
        total_return_usd=1000,
        total_return_pct=10.0,
        total_trades=10,
        winners=6,
        losers=4,
        win_rate=60.0,
        avg_win_usd=250,
        avg_win_pct=2.5,
        avg_loss_usd=-100,
        avg_loss_pct=-1.0,
        largest_win_usd=500,
        largest_loss_usd=-200,
        profit_factor=2.5,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=3.0,
        max_drawdown_usd=200,
        max_drawdown_pct=2.0,
        avg_drawdown_pct=0.5,
        recovery_factor=5.0,
        total_commission=50,
        total_slippage=20,
        total_spread=10,
        total_costs=80,
        avg_trade_duration_minutes=120,
        max_consecutive_wins=3,
        max_consecutive_losses=2,
    )
    print("Metrics created")
    print(f"   ✅ Total Return: {metrics.total_return_pct}%")
    print(f"   ✅ Win Rate: {metrics.win_rate}%")
    print(f"   ✅ Profit Factor: {metrics.profit_factor}")
    print(f"   ✅ Sharpe Ratio: {metrics.sharpe_ratio}")

    # Test 5: BacktestResult
    print("\n🎯 Test 5: BacktestResult")
    result = BacktestResult(
        config=config,
        trades=[trade],
        metrics=metrics,
        equity_curve=[
            {'time': datetime(2025, 1, 1), 'balance': 10000, 'drawdown': 0},
            {'time': datetime(2025, 1, 2), 'balance': 10500, 'drawdown': 0},
        ],
        execution_time_seconds=1.5,
    )
    print(f"   ✔️ Result created")
    print(f"   ✅ Trades: {len(result.trades)}")
    print(f"   ✅ Execution time: {result.execution_time_seconds}s")

    # Test 6: Serialization
    print("\n💾 Test 6: Serialization")
    result_dict = result.to_dict()
    print(f"   ✔️ Result converted to dictionary")
    print(f"   ✅ Keys: {list(result_dict.keys())}")

    print("\n👍 All tests were successful!")
    print("=" * 60)
