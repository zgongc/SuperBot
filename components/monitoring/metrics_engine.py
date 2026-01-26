#!/usr/bin/env python3
"""
monitoring/metrics_engine.py
SuperBot - Performance Metrics Engine
Author: SuperBot Team
Date: 2025-10-17
Versiyon: 1.0.0

Features:
- Trading performance metrics (PnL, Win Rate, Sharpe, Sortino, Calmar)
- System performance metrics (Latency, Uptime, Error Rate)
- Strategy performance comparison
- Real-time metric aggregation
- Historical metric tracking
- Alerting for metric thresholds

Usage:
    from monitoring import MetricsEngine, MetricType

    metrics = MetricsEngine(config, logger, event_bus)
    await metrics.start()

    # Record a trade
    metrics.record_trade(symbol="BTCUSDT", pnl=150.0, win=True)

    # Get metrics
    stats = metrics.get_trading_metrics()
    print(f"Win Rate: {stats.win_rate:.1%}")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")

Dependencies:
    - pandas>=1.3.0
    - numpy>=1.19.0
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Deque
import numpy as np


class MetricType(Enum):
    """Metric type enumeration"""
    TRADING = "trading"
    SYSTEM = "system"
    STRATEGY = "strategy"
    LATENCY = "latency"
    ERROR = "error"


@dataclass
class TradeMetric:
    """Individual trade metric"""
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    duration_seconds: float
    strategy: Optional[str] = None
    win: bool = False


@dataclass
class TradingMetrics:
    """Aggregated trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    profit_factor: float = 0.0  # Gross profit / Gross loss
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0  # in hours
    avg_trade_duration: float = 0.0  # in hours

    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Daily metrics
    trades_today: int = 0
    pnl_today: float = 0.0

    # Strategy breakdown
    strategy_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    uptime_seconds: float = 0.0
    uptime_percent: float = 100.0

    total_events: int = 0
    events_per_second: float = 0.0

    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    total_errors: int = 0
    error_rate: float = 0.0  # errors per minute
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    api_calls: int = 0
    api_weight: int = 0
    api_weight_limit: int = 1200
    api_utilization: float = 0.0  # percent

    websocket_connections: int = 0
    websocket_messages: int = 0

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class LatencyMetric:
    """Latency measurement"""
    timestamp: datetime
    operation: str
    latency_ms: float


class MetricsEngine:
    """
    Performance metrics tracking and analysis engine

    Tracks:
    - Trading performance (PnL, win rate, risk metrics)
    - System performance (latency, uptime, errors)
    - Strategy comparison
    """

    def __init__(self, config: Optional[Dict] = None, logger=None, event_bus=None):
        self.config = config or {}
        self.logger = logger
        self.event_bus = event_bus

        # Trading metrics
        self.trades: List[TradeMetric] = []
        self.trading_metrics = TradingMetrics()

        # System metrics
        self.system_metrics = SystemMetrics()
        self.start_time = time.time()

        # Latency tracking (last 1000 measurements per operation)
        self.latencies: Dict[str, Deque[LatencyMetric]] = defaultdict(lambda: deque(maxlen=1000))

        # Error tracking
        self.errors: List[Dict] = []
        self.error_window: Deque[datetime] = deque(maxlen=1000)

        # Event tracking
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.event_window: Deque[datetime] = deque(maxlen=10000)

        # API tracking (for rate limiting awareness)
        self.api_weight_window: Deque[tuple] = deque(maxlen=100)  # (timestamp, weight)

        # Running state
        self.running = False
        self.update_task = None
        self.update_interval = self.config.get("update_interval", 5)  # seconds

        self._log("info", "[MetricsEngine] Initialized")

    def _log(self, level: str, message: str):
        """Internal logging helper"""
        if self.logger:
            getattr(self.logger, level)(message)

    async def start(self):
        """Start metrics engine"""
        if self.running:
            self._log("warning", "[MetricsEngine] Already running")
            return

        self.running = True
        self.start_time = time.time()

        # Subscribe to events if event bus available
        if self.event_bus:
            await self._subscribe_to_events()

        # Start periodic update task
        self.update_task = asyncio.create_task(self._update_loop())

        self._log("info", "[MetricsEngine] Started")

    async def stop(self):
        """Stop metrics engine"""
        if not self.running:
            return

        self.running = False

        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        self._log("info", "[MetricsEngine] Stopped")

    async def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        events = [
            "trade.completed",
            "order.filled",
            "position.closed",
            "system.error",
            "api.request",
            "websocket.message"
        ]

        for event in events:
            await self.event_bus.subscribe(event, self._handle_event)

    async def _handle_event(self, event_type: str, data: Dict):
        """Handle incoming events"""
        self.event_counts[event_type] += 1
        self.event_window.append(datetime.now())
        self.system_metrics.total_events += 1

        # Handle specific events
        if event_type == "trade.completed":
            self._handle_trade_event(data)
        elif event_type == "system.error":
            self._handle_error_event(data)
        elif event_type == "api.request":
            self._handle_api_event(data)
        elif event_type == "websocket.message":
            self.system_metrics.websocket_messages += 1

    def _handle_trade_event(self, data: Dict):
        """Handle trade completion event"""
        try:
            trade = TradeMetric(
                timestamp=data.get("timestamp", datetime.now()),
                symbol=data["symbol"],
                side=data["side"],
                entry_price=data["entry_price"],
                exit_price=data["exit_price"],
                quantity=data["quantity"],
                pnl=data["pnl"],
                pnl_percent=data.get("pnl_percent", 0.0),
                duration_seconds=data.get("duration_seconds", 0.0),
                strategy=data.get("strategy"),
                win=data["pnl"] > 0
            )
            self.record_trade_metric(trade)
        except Exception as e:
            self._log("error", f"[MetricsEngine] Error handling trade event: {e}")

    def _handle_error_event(self, data: Dict):
        """Handle error event"""
        self.errors.append({
            "timestamp": datetime.now(),
            "type": data.get("type", "unknown"),
            "message": data.get("message", ""),
            "module": data.get("module", "unknown")
        })
        self.error_window.append(datetime.now())
        self.system_metrics.total_errors += 1

        error_type = data.get("type", "unknown")
        self.system_metrics.errors_by_type[error_type] = \
            self.system_metrics.errors_by_type.get(error_type, 0) + 1

    def _handle_api_event(self, data: Dict):
        """Handle API request event"""
        self.system_metrics.api_calls += 1
        weight = data.get("weight", 1)
        self.api_weight_window.append((time.time(), weight))
        self.system_metrics.api_weight += weight

    async def _update_loop(self):
        """Periodic metrics update loop"""
        while self.running:
            try:
                await asyncio.sleep(self.update_interval)
                await self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log("error", f"[MetricsEngine] Update loop error: {e}")

    async def _update_metrics(self):
        """Update aggregated metrics"""
        self._update_trading_metrics()
        self._update_system_metrics()

    def _update_trading_metrics(self):
        """Update trading metrics from trades"""
        if not self.trades:
            return

        metrics = self.trading_metrics

        # Basic counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.win)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0

        # PnL metrics
        metrics.total_pnl = sum(t.pnl for t in self.trades)
        metrics.avg_trade = metrics.total_pnl / metrics.total_trades if metrics.total_trades > 0 else 0.0

        wins = [t.pnl for t in self.trades if t.win]
        losses = [t.pnl for t in self.trades if not t.win]

        metrics.avg_win = sum(wins) / len(wins) if wins else 0.0
        metrics.avg_loss = sum(losses) / len(losses) if losses else 0.0
        metrics.largest_win = max(wins) if wins else 0.0
        metrics.largest_loss = min(losses) if losses else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Risk metrics
        returns = [t.pnl_percent for t in self.trades]
        if returns:
            metrics.sharpe_ratio = self._calculate_sharpe(returns)
            metrics.sortino_ratio = self._calculate_sortino(returns)
            metrics.max_drawdown = self._calculate_max_drawdown([t.pnl for t in self.trades])

        # Duration metrics
        durations = [t.duration_seconds / 3600 for t in self.trades]  # Convert to hours
        metrics.avg_trade_duration = sum(durations) / len(durations) if durations else 0.0

        # Consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for trade in self.trades:
            if trade.win:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        metrics.consecutive_wins = max(0, current_streak)
        metrics.consecutive_losses = abs(min(0, current_streak))
        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak

        # Today's metrics
        today = datetime.now().date()
        today_trades = [t for t in self.trades if t.timestamp.date() == today]
        metrics.trades_today = len(today_trades)
        metrics.pnl_today = sum(t.pnl for t in today_trades)

        # Strategy breakdown
        strategy_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
        for trade in self.trades:
            if trade.strategy:
                strategy_stats[trade.strategy]["trades"] += 1
                strategy_stats[trade.strategy]["pnl"] += trade.pnl
                if trade.win:
                    strategy_stats[trade.strategy]["wins"] += 1

        for strategy, stats in strategy_stats.items():
            stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0
            stats["avg_pnl"] = stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0.0

        metrics.strategy_metrics = dict(strategy_stats)
        metrics.last_updated = datetime.now()

    def _update_system_metrics(self):
        """Update system metrics"""
        metrics = self.system_metrics

        # Uptime
        metrics.uptime_seconds = time.time() - self.start_time

        # Events per second (last minute)
        now = datetime.now()
        recent_events = sum(1 for ts in self.event_window if (now - ts).total_seconds() < 60)
        metrics.events_per_second = recent_events / 60.0

        # Error rate (errors per minute)
        recent_errors = sum(1 for ts in self.error_window if (now - ts).total_seconds() < 60)
        metrics.error_rate = recent_errors

        # API weight utilization (last minute)
        cutoff_time = time.time() - 60
        recent_weights = sum(w for ts, w in self.api_weight_window if ts > cutoff_time)
        metrics.api_utilization = (recent_weights / metrics.api_weight_limit) * 100

        # Latency metrics (aggregate all operations)
        all_latencies = []
        for operation, latencies in self.latencies.items():
            all_latencies.extend([l.latency_ms for l in latencies])

        if all_latencies:
            metrics.avg_latency_ms = np.mean(all_latencies)
            metrics.p95_latency_ms = np.percentile(all_latencies, 95)
            metrics.p99_latency_ms = np.percentile(all_latencies, 99)
            metrics.max_latency_ms = max(all_latencies)

        metrics.last_updated = datetime.now()

    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        if np.std(returns_array) == 0:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252)
        return float(sharpe)

    def _calculate_sortino(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (only considers downside volatility)"""
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        # Calculate downside deviation
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)

    def _calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not pnl_series:
            return 0.0

        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

        return float(max_dd)

    def record_trade(self, symbol: str, pnl: float, win: bool, **kwargs):
        """
        Record a trade for metrics tracking

        Args:
            symbol: Trading pair
            pnl: Profit/loss amount
            win: Whether the trade was profitable
            **kwargs: Additional trade details (entry_price, exit_price, etc.)
        """
        trade = TradeMetric(
            timestamp=kwargs.get("timestamp", datetime.now()),
            symbol=symbol,
            side=kwargs.get("side", "UNKNOWN"),
            entry_price=kwargs.get("entry_price", 0.0),
            exit_price=kwargs.get("exit_price", 0.0),
            quantity=kwargs.get("quantity", 0.0),
            pnl=pnl,
            pnl_percent=kwargs.get("pnl_percent", 0.0),
            duration_seconds=kwargs.get("duration_seconds", 0.0),
            strategy=kwargs.get("strategy"),
            win=win
        )
        self.record_trade_metric(trade)

    def record_trade_metric(self, trade: TradeMetric):
        """Record a trade metric"""
        self.trades.append(trade)

        # Keep only last 10000 trades in memory
        if len(self.trades) > 10000:
            self.trades = self.trades[-10000:]

        # Immediate metrics update for key values
        self.trading_metrics.total_trades = len(self.trades)
        self.trading_metrics.total_pnl = sum(t.pnl for t in self.trades)

    def record_latency(self, operation: str, latency_ms: float):
        """
        Record operation latency

        Args:
            operation: Operation name (e.g., "order_placement", "websocket_ping")
            latency_ms: Latency in milliseconds
        """
        metric = LatencyMetric(
            timestamp=datetime.now(),
            operation=operation,
            latency_ms=latency_ms
        )
        self.latencies[operation].append(metric)

    def record_error(self, error_type: str, message: str, module: str = "unknown"):
        """
        Record an error

        Args:
            error_type: Error category
            message: Error message
            module: Module where error occurred
        """
        self._handle_error_event({
            "type": error_type,
            "message": message,
            "module": module
        })

    def record_api_call(self, weight: int = 1):
        """
        Record an API call for rate limiting tracking

        Args:
            weight: API weight of the call
        """
        self._handle_api_event({"weight": weight})

    def get_trading_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        return self.trading_metrics

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self.system_metrics

    def get_strategy_comparison(self) -> Dict[str, Dict]:
        """Get performance comparison by strategy"""
        return self.trading_metrics.strategy_metrics

    def get_latency_stats(self, operation: Optional[str] = None) -> Dict:
        """
        Get latency statistics

        Args:
            operation: Specific operation, or None for all operations

        Returns:
            Dictionary with latency statistics
        """
        if operation:
            latencies = [l.latency_ms for l in self.latencies.get(operation, [])]
            if not latencies:
                return {}

            return {
                "operation": operation,
                "count": len(latencies),
                "avg_ms": np.mean(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99)
            }
        else:
            # Aggregate all operations
            stats = {}
            for op in self.latencies.keys():
                stats[op] = self.get_latency_stats(op)
            return stats

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        trading = self.get_trading_metrics()
        system = self.get_system_metrics()

        return {
            "trading": {
                "total_trades": trading.total_trades,
                "win_rate": f"{trading.win_rate:.1%}",
                "total_pnl": f"${trading.total_pnl:,.2f}",
                "avg_trade": f"${trading.avg_trade:.2f}",
                "profit_factor": f"{trading.profit_factor:.2f}",
                "sharpe_ratio": f"{trading.sharpe_ratio:.2f}",
                "max_drawdown": f"${trading.max_drawdown:,.2f}",
                "trades_today": trading.trades_today,
                "pnl_today": f"${trading.pnl_today:,.2f}"
            },
            "system": {
                "uptime_hours": f"{system.uptime_seconds / 3600:.1f}",
                "events_per_second": f"{system.events_per_second:.1f}",
                "avg_latency_ms": f"{system.avg_latency_ms:.1f}",
                "p99_latency_ms": f"{system.p99_latency_ms:.1f}",
                "total_errors": system.total_errors,
                "error_rate": f"{system.error_rate:.1f}/min",
                "api_utilization": f"{system.api_utilization:.1f}%"
            },
            "strategies": trading.strategy_metrics
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.trades.clear()
        self.trading_metrics = TradingMetrics()
        self.system_metrics = SystemMetrics()
        self.latencies.clear()
        self.errors.clear()
        self.event_counts.clear()
        self.start_time = time.time()

        self._log("info", "[MetricsEngine] Metrics reset")


if __name__ == "__main__":
    # Example usage
    import json

    async def test_metrics_engine():
        # Initialize
        metrics = MetricsEngine(config={"update_interval": 2})
        await metrics.start()

        # Simulate some trades
        print("\n=== Simulating trades ===")
        trades_data = [
            {"symbol": "BTCUSDT", "pnl": 150.0, "win": True, "strategy": "AI_Strategy", "pnl_percent": 1.5},
            {"symbol": "ETHUSDT", "pnl": -50.0, "win": False, "strategy": "AI_Strategy", "pnl_percent": -0.8},
            {"symbol": "BTCUSDT", "pnl": 200.0, "win": True, "strategy": "Trend_Follow", "pnl_percent": 2.0},
            {"symbol": "BNBUSDT", "pnl": 75.0, "win": True, "strategy": "AI_Strategy", "pnl_percent": 1.2},
            {"symbol": "ETHUSDT", "pnl": -30.0, "win": False, "strategy": "Trend_Follow", "pnl_percent": -0.5},
        ]

        for trade_data in trades_data:
            metrics.record_trade(**trade_data)
            await asyncio.sleep(0.1)

        # Record some latencies
        metrics.record_latency("order_placement", 45.2)
        metrics.record_latency("order_placement", 52.1)
        metrics.record_latency("websocket_ping", 12.5)

        # Record API calls
        metrics.record_api_call(weight=1)
        metrics.record_api_call(weight=5)

        # Wait for metrics update
        await asyncio.sleep(3)

        # Get metrics
        print("\n=== Trading Metrics ===")
        trading = metrics.get_trading_metrics()
        print(f"Total Trades: {trading.total_trades}")
        print(f"Win Rate: {trading.win_rate:.1%}")
        print(f"Total PnL: ${trading.total_pnl:,.2f}")
        print(f"Avg Trade: ${trading.avg_trade:.2f}")
        print(f"Profit Factor: {trading.profit_factor:.2f}")
        print(f"Sharpe Ratio: {trading.sharpe_ratio:.2f}")
        print(f"Max Consecutive Wins: {trading.max_consecutive_wins}")

        print("\n=== System Metrics ===")
        system = metrics.get_system_metrics()
        print(f"Uptime: {system.uptime_seconds:.1f}s")
        print(f"Total Events: {system.total_events}")
        print(f"API Utilization: {system.api_utilization:.1f}%")

        print("\n=== Strategy Comparison ===")
        strategies = metrics.get_strategy_comparison()
        for strategy, stats in strategies.items():
            print(f"\n{strategy}:")
            print(f"  Trades: {stats['trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Avg PnL: ${stats['avg_pnl']:.2f}")

        print("\n=== Latency Stats ===")
        latency_stats = metrics.get_latency_stats()
        print(json.dumps(latency_stats, indent=2))

        print("\n=== Summary ===")
        summary = metrics.get_summary()
        print(json.dumps(summary, indent=2))

        await metrics.stop()

    # Run test
    asyncio.run(test_metrics_engine())
