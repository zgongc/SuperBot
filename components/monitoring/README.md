# Monitoring Module

**Author:** SuperBot Team
**Version:** 1.0.0
**Date:** 2025-11-22

---

## 📊 Overview

The Monitoring module provides comprehensive **real-time tracking** and **analysis** of trading performance, system resources, and operational metrics. It enables data-driven decisions and alerts for system health.

### Key Features

- ✅ **Trading Performance Metrics** - PnL, Win Rate, Sharpe, Sortino, Calmar
- ✅ **System Resource Monitoring** - CPU, RAM, Disk, Network
- ✅ **Latency Tracking** - Real-time latency metrics (avg, P95, P99, max)
- ✅ **API Usage Monitoring** - Binance API weight and rate limit tracking
- ✅ **WebSocket Monitoring** - Connection health and message stats
- ✅ **Error Tracking** - Error rate and breakdown by type
- ✅ **EventBus Integration** - Event-driven alerts and notifications
- ✅ **Strategy Comparison** - Side-by-side strategy performance analysis

---

## 📁 Module Structure

```
components/monitoring/
├── __init__.py              # Module exports
├── metrics_collector.py     # Trading & system metrics engine
├── resource_monitor.py      # System resource monitoring
└── README.md               # This file
```

---

## 🎯 Components

### 1. MetricsEngine (`metrics_collector.py`)

**Purpose:** Tracks and analyzes trading performance metrics

#### Trading Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Win Rate** | Percentage of winning trades | `winning_trades / total_trades * 100` |
| **Total PnL** | Cumulative profit/loss | Sum of all trade PnL |
| **Avg Win/Loss** | Average profit/loss per trade | `total_pnl / total_trades` |
| **Profit Factor** | Ratio of gross profit to loss | `gross_profit / gross_loss` |
| **Sharpe Ratio** | Risk-adjusted return | `(avg_return - risk_free) / std_dev` |
| **Sortino Ratio** | Downside risk-adjusted return | `avg_return / downside_deviation` |
| **Calmar Ratio** | Return vs max drawdown | `annual_return / max_drawdown` |
| **Max Drawdown** | Largest peak-to-trough decline | Maximum cumulative loss |
| **Consecutive Wins/Losses** | Longest win/loss streak | Track sequential outcomes |

#### Usage Example

```python
from components.monitoring import MetricsEngine

# Initialize
metrics = MetricsEngine(config, logger, event_bus)
await metrics.start()

# Record a trade
metrics.record_trade(
    symbol="BTCUSDT",
    side="BUY",
    entry_price=50000.0,
    exit_price=51000.0,
    quantity=0.1,
    pnl=100.0,
    pnl_percent=2.0,
    duration_seconds=3600,
    strategy="EMA_Cross",
    win=True
)

# Get metrics
stats = metrics.get_trading_metrics()
print(f"Win Rate: {stats.win_rate:.1%}")
print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
print(f"Total PnL: ${stats.total_pnl:.2f}")
print(f"Max Drawdown: {stats.max_drawdown:.1%}")

# Strategy comparison
for strategy, metrics in stats.strategy_metrics.items():
    print(f"{strategy}: Win Rate={metrics['win_rate']:.1%}, PnL=${metrics['pnl']:.2f}")
```

#### EventBus Integration

```python
# MetricsEngine publishes events:

# Trade recorded
event_bus.publish('monitoring.trade', {
    'symbol': 'BTCUSDT',
    'pnl': 150.0,
    'win': True,
    'strategy': 'EMA_Cross'
})

# Metric threshold alert
event_bus.publish('monitoring.alert.drawdown', {
    'level': 'warning',
    'value': 15.2,
    'threshold': 10.0
})
```

---

### 2. ResourceMonitor (`resource_monitor.py`)

**Purpose:** Monitors system resource usage and performance

#### System Metrics

| Resource | Metric | Threshold | Alert |
|----------|--------|-----------|-------|
| **CPU** | Usage percentage | 80% | Warning |
| **Memory** | RAM usage (MB + %) | 80% | Warning |
| **Disk** | Disk usage (GB + %) | 90% | Critical |
| **Network** | Sent/Received MB | N/A | Info |

#### Process Metrics

- Process CPU usage
- Process RAM usage
- Thread count
- Open file descriptors

#### Usage Example

```python
from components.monitoring import ResourceMonitor

# Initialize
monitor = ResourceMonitor(
    config={
        "cpu_threshold": 80,
        "memory_threshold": 80,
        "disk_threshold": 90,
        "monitor_interval": 5  # Check every 5 seconds
    },
    event_bus=event_bus
)

await monitor.start()

# Get current stats
stats = monitor.get_stats()
print(f"CPU: {stats.cpu_percent}%")
print(f"RAM: {stats.memory_mb:.1f} MB ({stats.memory_percent:.1f}%)")
print(f"Disk: {stats.disk_gb:.1f} GB ({stats.disk_percent:.1f}%)")
print(f"Network: ↑{stats.network_sent_mb:.1f} MB ↓{stats.network_recv_mb:.1f} MB")

# Get historical data
history = monitor.get_history(minutes=30)  # Last 30 minutes
```

#### EventBus Integration

```python
# ResourceMonitor publishes events:

# CPU threshold exceeded
event_bus.publish('monitoring.alert.cpu', {
    'level': 'warning',
    'value': 85.2,
    'threshold': 80
})

# Memory threshold exceeded
event_bus.publish('monitoring.alert.memory', {
    'level': 'warning',
    'value': 82.5,
    'threshold': 80
})

# Disk threshold exceeded
event_bus.publish('monitoring.alert.disk', {
    'level': 'critical',
    'value': 92.1,
    'threshold': 90
})
```

---

### 3. SystemMetrics (Data Classes)

**Purpose:** Structured data for system performance

#### Latency Metrics

```python
@dataclass
class LatencyMetric:
    timestamp: datetime
    operation: str
    latency_ms: float

# Track operation latency
metrics.record_latency("indicator_calculation", 25.5)  # 25.5ms
metrics.record_latency("api_call", 150.2)              # 150.2ms
metrics.record_latency("strategy_evaluation", 5.8)     # 5.8ms

# Get latency stats
latency_stats = metrics.get_latency_stats()
print(f"Avg: {latency_stats.avg_latency_ms:.1f}ms")
print(f"P95: {latency_stats.p95_latency_ms:.1f}ms")
print(f"P99: {latency_stats.p99_latency_ms:.1f}ms")
print(f"Max: {latency_stats.max_latency_ms:.1f}ms")
```

#### API Metrics

```python
@dataclass
class SystemMetrics:
    api_calls: int
    api_weight: int
    api_weight_limit: int = 1200
    api_utilization: float  # Percentage

# Track API usage
metrics.record_api_call(weight=5)
metrics.record_api_call(weight=10)

# Get API stats
api_stats = metrics.get_api_stats()
print(f"Total calls: {api_stats.api_calls}")
print(f"Weight used: {api_stats.api_weight}/{api_stats.api_weight_limit}")
print(f"Utilization: {api_stats.api_utilization:.1%}")
```

#### WebSocket Metrics

```python
# Track WebSocket activity
metrics.record_websocket_message()
metrics.record_websocket_connection()

# Get WebSocket stats
ws_stats = metrics.get_websocket_stats()
print(f"Active connections: {ws_stats.websocket_connections}")
print(f"Messages received: {ws_stats.websocket_messages}")
```

#### Error Metrics

```python
# Record errors
metrics.record_error("ConnectionError", "WebSocket disconnected")
metrics.record_error("TimeoutError", "API request timeout")

# Get error stats
error_stats = metrics.get_error_stats()
print(f"Total errors: {error_stats.total_errors}")
print(f"Error rate: {error_stats.error_rate:.2f} errors/min")
print(f"Errors by type: {error_stats.errors_by_type}")
# {'ConnectionError': 5, 'TimeoutError': 2}
```

---

## 🔗 EventBus Integration

The Monitoring module is fully integrated with EventBus for **real-time alerts** and **event-driven monitoring**.

### Published Events

| Event Topic | Trigger | Payload |
|-------------|---------|---------|
| `monitoring.trade` | Trade recorded | `{symbol, pnl, win, strategy}` |
| `monitoring.alert.cpu` | CPU > threshold | `{level, value, threshold}` |
| `monitoring.alert.memory` | Memory > threshold | `{level, value, threshold}` |
| `monitoring.alert.disk` | Disk > threshold | `{level, value, threshold}` |
| `monitoring.alert.drawdown` | Drawdown > threshold | `{level, value, threshold}` |
| `monitoring.latency.spike` | Latency spike detected | `{operation, latency_ms}` |
| `monitoring.api.limit` | API weight near limit | `{weight, limit, utilization}` |
| `monitoring.error` | Error occurred | `{type, message, count}` |

### Subscribing to Events

```python
from core.event_bus import EventBus

event_bus = EventBus()

# Subscribe to alerts
async def handle_cpu_alert(event_data):
    print(f"⚠️ CPU Alert: {event_data['value']:.1f}% (Threshold: {event_data['threshold']}%)")

async def handle_trade(event_data):
    print(f"📊 Trade: {event_data['symbol']} PnL=${event_data['pnl']:.2f} Win={event_data['win']}")

event_bus.subscribe('monitoring.alert.cpu', handle_cpu_alert)
event_bus.subscribe('monitoring.trade', handle_trade)
```

---

## 🚀 Usage Scenarios

### 1. Live Trading (Production)

```python
from components.monitoring import MetricsEngine, ResourceMonitor

# Initialize monitoring
metrics_engine = MetricsEngine(config, logger, event_bus)
resource_monitor = ResourceMonitor(config, event_bus)

# Start monitoring
await metrics_engine.start()
await resource_monitor.start()

# In trading loop...
while trading:
    # Record trades
    metrics_engine.record_trade(...)

    # Check system health
    if resource_monitor.get_stats().cpu_percent > 90:
        logger.warning("High CPU usage!")
```

### 2. Paper Trading (Testing)

```python
# Compare strategies
metrics_engine = MetricsEngine(config, logger, event_bus)

# Run strategies
for strategy in strategies:
    run_paper_trading(strategy)

# Compare results
stats = metrics_engine.get_trading_metrics()
for strategy_name, strategy_stats in stats.strategy_metrics.items():
    print(f"{strategy_name}:")
    print(f"  Win Rate: {strategy_stats['win_rate']:.1%}")
    print(f"  PnL: ${strategy_stats['pnl']:.2f}")
    print(f"  Sharpe: {strategy_stats['sharpe']:.2f}")
```

### 3. Backtest Analysis

```python
# After backtest completion
metrics_engine = MetricsEngine(config, logger, event_bus)

# Analyze results
for trade in backtest_trades:
    metrics_engine.record_trade(...)

# Get comprehensive metrics
stats = metrics_engine.get_trading_metrics()
print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {stats.sortino_ratio:.2f}")
print(f"Calmar Ratio: {stats.calmar_ratio:.2f}")
print(f"Max Drawdown: {stats.max_drawdown:.1%}")
```

---

## 📈 Dashboard Integration (Future)

Monitoring data can be exported for visualization:

```python
# Prometheus export
metrics_engine.export_prometheus()

# Grafana dashboard
dashboard_server.update_metrics(
    trading_metrics=metrics_engine.get_trading_metrics(),
    system_metrics=resource_monitor.get_stats()
)

# Web API endpoint
@app.get("/api/metrics")
async def get_metrics():
    return {
        "trading": metrics_engine.get_trading_metrics(),
        "system": resource_monitor.get_stats()
    }
```

---

## ⚙️ Configuration

### metrics_collector.yaml

```yaml
metrics:
  # Metric collection intervals
  collection_interval: 60  # seconds

  # Historical data retention
  history_days: 30

  # Alert thresholds
  thresholds:
    max_drawdown: 10.0  # percent
    min_sharpe: 1.0
    error_rate: 5  # errors per minute

  # Strategy comparison
  enable_strategy_breakdown: true
```

### resource_monitor.yaml

```yaml
resource_monitor:
  # Monitoring interval
  monitor_interval: 5  # seconds

  # History retention
  history_size: 100  # snapshots

  # Alert thresholds
  thresholds:
    cpu_percent: 80
    memory_percent: 80
    disk_percent: 90

  # Process monitoring
  track_process: true
```

---

## 📊 Performance Impact

- **CPU Overhead:** < 1% (background monitoring)
- **Memory Overhead:** ~10-20 MB (metric history)
- **Latency Impact:** < 1ms (metric recording)

**Recommendation:** Always enable monitoring in production for system health visibility.

---

## 🔧 Dependencies

```python
# Required packages
psutil>=5.8.0      # System resource monitoring
numpy>=1.19.0      # Statistical calculations
pandas>=1.3.0      # Data aggregation
```

---

## 📝 Example: Complete Setup

```python
import asyncio
from core.event_bus import EventBus
from core.logger_engine import LoggerEngine
from components.monitoring import MetricsEngine, ResourceMonitor

async def main():
    # Setup
    logger = LoggerEngine().get_logger(__name__)
    event_bus = EventBus()

    config = {
        "metrics": {
            "collection_interval": 60,
            "thresholds": {"max_drawdown": 10.0}
        },
        "resource_monitor": {
            "cpu_threshold": 80,
            "memory_threshold": 80,
            "monitor_interval": 5
        }
    }

    # Initialize monitoring
    metrics = MetricsEngine(config["metrics"], logger, event_bus)
    monitor = ResourceMonitor(config["resource_monitor"], event_bus)

    # Start monitoring
    await metrics.start()
    await monitor.start()

    # Subscribe to alerts
    async def handle_alert(event_data):
        logger.warning(f"Alert: {event_data}")

    event_bus.subscribe('monitoring.alert.*', handle_alert)

    # Your trading logic here...
    while True:
        # Record trade
        metrics.record_trade(
            symbol="BTCUSDT",
            pnl=100.0,
            win=True
        )

        # Check system health
        stats = monitor.get_stats()
        if stats.cpu_percent > 90:
            logger.error("Critical CPU usage!")

        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Best Practices

1. ✅ **Always enable monitoring in production**
2. ✅ **Set appropriate thresholds** for your hardware
3. ✅ **Review metrics daily** for performance trends
4. ✅ **Use strategy comparison** to optimize performance
5. ✅ **Monitor API usage** to avoid rate limits
6. ✅ **Track latency** for real-time trading
7. ✅ **Subscribe to alerts** for immediate notification

---

## 📚 Related Documentation

- [EventBus Integration](../core/event_bus/README.md)
- [TradingEngine](../engines/trading_engine.py)
- [Phase 3: Real-Time Testing](../../docs/plans/eventbus_integration_phase2.md)

---

**Version:** 1.0.0
**Author:** SuperBot Team
**Last Updated:** 2025-11-22
