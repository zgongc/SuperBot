# 🧠 SuperBot - Context Management Guide

> **Purpose:** For Claude to quickly catch project context in every session

---

## 🚀 Session Start (Every New Conversation)

### 1. Read This File
```
docs/claude/context_guide.md  (this file - quick reference)
```

### 2. 🔥 Daemon Architecture (NEW - READ FIRST!)
```
docs/claude/session-start-guide.md        # ⚡ QUICK START (5 minutes)
docs/claude/daemon-architecture-guide.md  # 📚 DETAILED GUIDE (full architecture)
```

**CRITICAL:** Don't write code without knowing daemon architecture!

### 3. Understand Project Vision
```
docs/claude/PROJECT_VISION.md      # WHY are we doing this? Success criteria
docs/plans/implementation_plan.md  # WHAT are we building? Technology stack
docs/plans/rules.md                # HOW are we doing it? Development principles
```

### 4. Learn Detailed Rules
```
docs/claude/claude_rules.md        # Detailed rules for Claude (329 lines)
docs/master/system_architecture.md # Architecture details (if exists)
```

---

## 🎯 Project Vision

> **📖 For full vision:** Read `docs/claude/PROJECT_VISION.md`

### What Are We Building?
**SuperBot**: AI-powered, multi-exchange crypto trading platform

### Why?
- Professional bot for crypto future trading
- Solo developer + 1-2 friends usage
- **Success criteria:** Profit in live trading

### Priority: Backtest Module (CRITICAL)
> "If the first backtest is complete, most of the project will be finished"
- Same strategy code: backtest + trading + optimization + AI

### Special Features:
- **Replay Mode**: TradingView-like live viewing
- **Multi-Timeframe (MTF)**: Cross-timeframe signals
- **Hybrid Strategy**: AI + Classical TA
- **Config-driven**: Memory/SQLite (dev) → Redis/PostgreSQL (prod)

### Core Principles:
1. **Plan-First**: Update plan before new development
2. **Backtest-First**: Strategies must pass backtest first
3. **Modularity**: Core/components shared, modules loosely coupled
4. **Observability**: Logging and metrics from day one

---

## 📋 Project Quick Reference

### Architecture Layers:
```
CORE (infrastructure)
  ↑
COMPONENTS (business logic)
  ↑
MODULES (applications)
```

**Rule:** Import only from top to bottom!

### 🔥 Critical Reminders:

#### 1. Logger & Config
```python
# ✅ ALWAYS
from core.logger_engine import get_logger
from core.config_engine import get_config

logger = get_logger("components.managers.risk_manager")
config = get_config()

# ❌ NEVER
import logging
logger = logging.getLogger(__name__)
```

#### 2. Emoji Preservation
```python
# ✅ NEVER delete emojis
logger.info("🚀 Engine starting...")

# ❌ Don't delete even if it looks garbled in console!
```

#### 4. File Structure
```python
#!/usr/bin/env python3
"""
path/to/file.py
SuperBot - Module Name
...docstring...
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent  # Adjust depth
    sys.path.insert(0, str(project_root))

# ... code ...

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Test...")
```

---

## 🗺️ Component Responsibility Map

**Check BEFORE writing new code:**

| Component | What Does It Do? |
|-----------|------------------|
| **BinanceClient** | API connection, order sending |
| **WebSocketEngine** | WebSocket management, auto-reconnect |
| **MultiTimeframeEngine** | 1m → 5m, 15m, 1h aggregation |
| **DataDownloader** | Real-time data orchestration |
| **HistoricalDataManager** | Parquet data loading |
| **AccountManager** | Balance, leverage, margin |
| **RiskManager** | Risk control + position sizing |
| **OrderManager** | Order validation + sending |
| **PositionManager** | Position lifecycle |
| **PortfolioManager** | Performance metrics (PnL, Sharpe) |
| **StrategyExecutor** | Entry/exit signal generation |
| **IndicatorEngine** | Technical indicator calculation |

### ⚠️ Common Mistakes:

```
❌ Before writing RSI → Check if indicators/momentum/rsi.py exists
❌ Position sizing logic → Already exists in RiskManager
❌ Order execution → Use OrderManager, don't rewrite
```

---

## 📂 Project Structure Summary

```
SuperBot/
├── core/                    # Logger, Config, EventBus, Cache, Rate Limiter
├── components/
│   ├── connectors/         # Binance, CCXT
│   ├── data/               # WebSocket, MultiTimeframe, DataDownloader
│   ├── managers/           # Account, Risk, Order, Position, Portfolio
│   ├── indicators/         # trend/, momentum/, volatility/
│   └── strategies/         # BaseStrategyTemplate, user strategies
├── modules/
│   ├── trading/           # Live/Paper/Demo/Replay
│   ├── backtest/          # Backtesting engine
│   ├── ai/                # ML models
│   └── webui/             # Flask dashboard
└── config/                # YAML configs + .env
```

---

## 🎯 New Task Checklist

- [ ] Read `context_guide.md` (this file)
- [ ] Read `claude_rules.md`
- [ ] Check if related component already exists
- [ ] Check layer dependency rules
- [ ] Use `get_logger()` and `get_config()`
- [ ] Preserve emojis, write Turkish output

---

## 📖 For More Information

| Category | File | What It Contains |
|----------|------|------------------|
| **⚡ Quick Start** | `docs/claude/session-start-guide.md` | 🔥 **READ FIRST!** Daemon architecture, async executor, event bus (5 min) |
| **📚 Daemon Architecture** | `docs/claude/daemon-architecture-guide.md` | 🔥 **DETAILED GUIDE!** Master daemon, shared resources, IPC/RPC |
| **🌟 Vision & Goals** | `docs/claude/PROJECT_VISION.md` | Why are we building this? Success criteria |
| **🎯 Master Plan** | `docs/plans/implementation_plan.md` | Technology stack, modules, roadmap |
| **📏 Principles** | `docs/plans/rules.md` | General development principles, processes |
| **🤖 Claude Rules** | `docs/claude/claude_rules.md` | Detailed development rules (329 lines) |
| **🏗️ Architecture** | `docs/master/system_architecture.md` | Full architecture documentation |
| **🇹🇷 Localization** | `docs/master/localization_guide.md` | Turkish translation dictionary |
| **📚 Overview** | `README.md` | Project summary, installation, quick start |

---

## 💡 If Context Is Lost

If session gets long and context is lost:

```bash
# Tell user:
"For context refresh, please read these files in order:
 1. docs/claude/context_guide.md
 2. docs/claude/session-start-guide.md
 3. docs/claude/daemon-architecture-guide.md (optional but recommended)"
```

## 🧠 Captain's Memory - Session Memory

SQLite-based memory system to remember information across sessions.

### Get Context at Session Start
```bash
python memory/captain_memory.py summary
```

### Usage (From Terminal)
```bash
# Add log
python memory/captain_memory.py log "Did X today"

# Save decision
python memory/captain_memory.py decision "topic" "decision"

# Save knowledge
python memory/captain_memory.py learn "topic" "learned info"

# See recent logs
python memory/captain_memory.py show

# Search
python memory/captain_memory.py search "QML"
```

### Usage From Python
```python
from memory.captain_memory import get_memory
m = get_memory()

# Get session summary (for Claude)
print(m.get_session_summary())

# Add log
m.log("QML pattern drawing completed", category="implementation")

# Save decision
m.decision("Starts from Zone Head", topic="QML", context="SMC logic")

# Learn knowledge
m.learn("BaselineSeries is used for drawing boxes", topic="charts")
```

---

## 🆕 Recent Additions

### 2025-12-22: Captain's Memory
- ✅ **memory/captain_memory.py** → Session memory system
- SQLite-based persistent memory
- Log, Decision, Knowledge tables
- CLI and Python API

### 2025-11-26: Daemon Architecture Documents
- ✅ **session-start-guide.md** → 5-minute quick start
- ✅ **daemon-architecture-guide.md** → Full daemon architecture guide

**Why added:**
- Daemon architecture not understood in old sessions
- Async executor pattern forgotten
- Exchange files deleted (connector_engine, connection_engine)
- "Reinvent the wheel" problem repeated

**What to do now:**
- Every new session: READ `session-start-guide.md`!
- For daemon questions: READ `daemon-architecture-guide.md`!
- When writing Exchange API: ALWAYS use async executor pattern!
- At session start: Run `python memory/captain_memory.py summary`!

---

**Version:** 1.2.0
**Last Updated:** 2025-12-22
**Maintainer:** SuperBot Team
