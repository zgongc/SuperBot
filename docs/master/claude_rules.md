# 🤖 SuperBot - Claude Development Rules

> **Last Updated:** 2025-01-17
> **Version:** 3.0.0
> **For:** VS Code Claude Assistant

---

## 🎯 PROJECT OVERVIEW

**SuperBot** is a crypto trading bot with modular architecture:

```
LAYERS:
  CORE        → Infrastructure (logger, config, events, cache, etc.)
  COMPONENTS  → Business logic (indicators, strategies, data, managers)
  MODULES     → Applications (backtest, trading, ai, webui)

RULE: Upper layers use lower layers. Lower layers are independent.
```

**Current Structure:**
```
trading-bot/
├── core/                    # Infrastructure layer
├── components/              # Reusable business logic
│   ├── engines/            # Active engines (start/stop)
│   ├── managers/           # Passive managers (CRUD)
│   ├── analysis/           # Analysis tools
│   ├── connectors/         # Exchange connections
│   ├── data/               # Data management
│   ├── monitoring/         # Monitoring & metrics
│   ├── notifiers/          # Notification system
│   ├── patterns/           # Pattern detection
│   └── strategies/         # Strategy templates
├── modules/                 # Application layer
│   ├── backtest/           # Backtesting module
│   ├── trading/            # Live trading module
│   ├── ai/                 # AI/ML module
│   └── webui/              # Web dashboard
└── config/                  # Configuration files
```

---

## 🚨 CRITICAL RULES - NEVER BREAK THESE

### 1. EMOJI PRESERVATION 🎨

**NEVER remove or replace emojis from any file!**

```python
# ❌ WRONG - Do not remove emojis
print("Loading data...")
logger.info("Engine started")

# ✅ CORRECT - Keep emojis as they are
print("📂 Loading data...")
logger.info("🚀 Engine started")
```

**Why:**
- Emojis are intentional and improve readability
- Windows console display issues are cosmetic only
- Code works perfectly with emojis internally
- `UnicodeEncodeError` in console is NOT a code error

**Action:** Ignore emoji display errors, do NOT modify the code

---

### 2. ENGLISH LANGUAGE STANDARD 🌐

**ALL code must be in English - logs, comments, exceptions, docstrings, prints!**

#### ✅ What Must Be English:
- All comments, log messages, exception messages, print statements, docstrings, test output, documentation

#### Quick Examples:

```python
# ✅ CORRECT
logger.info("🚀 Engine starting...")
logger.error(f"❌ Connection error: {e}")
raise ValueError("Invalid parameter")

def calculate_risk(self, position):
    """
    Calculate position risk.

    Args:
        position: Position information
    Returns:
        float: Risk percentage
    """
    if not position:
        raise ValueError("Position data is empty")
    return position['size'] * position['leverage']
```

---

### 3. FILE STRUCTURE STANDARD 📄

**Every Python module must have header documentation and test section!**

#### File Header (Required):

```python
#!/usr/bin/env python3
"""
path/to/file.py
SuperBot - Module Name
Author: SuperBot Team
Date: YYYY-MM-DD
Version: X.Y.Z

Module description (brief and concise)

Features:
- Feature 1
- Feature 2

Usage:
    from module import Class
    instance = Class()

Dependencies:
    - python>=3.10
    - package1>=1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
```

#### File Footer (Required for libraries):

```python
# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ModuleName Test")
    print("=" * 60)

    print("Test 1:")
    # Test code here
    print("   ✅ Test successful")

    print("\n✅ All tests completed!")
    print("=" * 60)
```

**How to Run Tests:**

```bash
# ✅ BOTH METHODS WORK (thanks to sys.path handling in header)
python -m core.logger_engine              # Module syntax
python .\core\logger_engine.py            # Direct file run

python -m components.managers.risk_manager
python .\components\managers\risk_manager.py
```

**Why both work?**
- Header template adds `sys.path.insert(0, project_root)` when `__name__ == "__main__"`
- Module syntax (`-m`) automatically adds project root to PYTHONPATH
- Direct file run uses sys.path from header template

**Reference:** See `core/event_bus.py` for perfect example

---

## 📋 CODING STANDARDS

### Python Best Practices:

```python
# ✅ Add to the beginning of every file (Python 3.7+)
from __future__ import annotations

# This provides:
# - Forward references (reference to classes not yet defined)
# - Type hints are not evaluated at runtime (performance)
# - Prevents circular import issues
```

### Logging Standards:

```python
# ✅ CORRECT - English + Emoji + Context
logger.debug(f"🔍 Debug: {variable}")
logger.info(f"📊 Statistics updated: {count} records")
logger.warning(f"⚠️  Warning: {message}")
logger.error(f"❌ Error: {error_message}")
logger.critical(f"🚨 Critical: {critical_issue}")
```

### Common Emojis:
- ✅ Success | ❌ Failed | ⚠️ Warning | 🔍 Debug
- 📊 Statistics | 🚀 Startup | 🛑 Stop | 🔄 Restart
- 💾 Data save | 🌐 Network | 🔐 Security | 💰 Money

---

## 🏗️ ARCHITECTURE GUIDELINES

### System Architecture Reference:

**CRITICAL:** Before writing ANY code, read `system_architecture.md` to understand:
- Project structure (core/components/modules)
- Component responsibilities
- Dependency relationships

### Layer Dependency Rules:

```
✅ ALLOWED:
  MODULES     → COMPONENTS → CORE
  COMPONENTS  → CORE
  MODULES     → CORE

❌ NOT ALLOWED:
  CORE        → COMPONENTS
  CORE        → MODULES
  COMPONENTS  → MODULES
```

### 🔥 CRITICAL: Always Use Core Engines

**NEVER create custom logger or config instances!**

```python
# ✅ CORRECT - Use core engine functions (singleton pattern)
from core.logger_engine import get_logger
from core.config_engine import get_config

logger = get_logger("components.managers.risk_manager")  # Named logger
config = get_config()  # Singleton config instance

# ❌ WRONG - Don't create custom loggers
import logging
logger = logging.getLogger(__name__)

# ❌ WRONG - Don't create custom config readers
with open('config.yaml') as f:
    config = yaml.load(f)

# ❌ WRONG - Don't instantiate directly
from core.logger_engine import LoggerEngine
logger = LoggerEngine()  # Creates new instance every time
```

**Why:**
- Singleton pattern - Same instance is used (memory efficient)
- Named loggers - Clear which module it came from
- Prevents context fragmentation across sessions
- Maintains centralized configuration
- Ensures consistent logging format

**Rule:** If you need logger or config anywhere, ALWAYS use `get_logger()` and `get_config()` from `core/`

### Component Organization:

```
components/
├── connectors/       # Exchange API connections
├── data/            # Data management
│   ├── websocket_engine.py
│   ├── multi_timeframe_engine.py
│   ├── data_downloader.py
│   └── historical_data_manager.py
├── managers/        # Business logic managers
│   ├── account_manager.py
│   ├── risk_manager.py
│   ├── order_manager.py
│   ├── position_manager.py
│   ├── portfolio_manager.py
│   └── strategy_executor.py
├── indicators/      # Technical indicators
└── strategies/      # Strategy templates
```

### CRITICAL: Component Responsibilities

**BEFORE writing code, check which component does what:**

| Component | Responsibility |
|-----------|---------------|
| **BinanceClient** | API connection, order sending, balance query |
| **WebSocketEngine** | WebSocket connection management, auto-reconnect |
| **MultiTimeframeEngine** | 1m → 5m, 15m, 1h aggregation |
| **DataDownloader** | Real-time data orchestration |
| **HistoricalDataManager** | Parquet data loading |
| **AccountManager** | Balance, leverage, margin management |
| **RiskManager** | Risk checks + position sizing calculation |
| **OrderManager** | Order validation + sending |
| **PositionManager** | Position lifecycle management |
| **PortfolioManager** | Performance metrics, win rate, PnL, Sharpe |
| **StrategyExecutor** | Entry/exit signal generation |
| **IndicatorEngine** | Technical indicator calculations |

### ⚠️ COMMON MISTAKES TO AVOID:

1. **DON'T create new components without checking existing ones**
   - ❌ Writing RSI function when it exists in indicators/momentum/
   - ❌ Creating OrderExecutor when OrderManager exists
   - ❌ Writing position sizing logic when RiskManager has it

2. **DON'T duplicate functionality**
   - Check `components/` before writing anything

3. **DON'T break dependency rules**
   - Core components NEVER import from components/
   - Components NEVER import from modules/

### Naming Conventions:

```python
# ✅ CORRECT
multi_timeframe_engine.py     # Active component (start/stop)
order_manager.py              # Passive component (CRUD)
correlation_analyzer.py       # Analysis tool
binance_client.py            # Connector

# ❌ WRONG
multi_timeframe_manager.py    # Manager but behaves like engine
order_engine.py               # Engine but behaves like manager
```

### Before Writing Code Checklist:

- [ ] Read system_architecture.md
- [ ] Check if component already exists
- [ ] Verify correct component location
- [ ] Confirm dependency rules
- [ ] Check component responsibility table
- [ ] Ensure no duplication

---

## 📝 FINAL NOTES

### Important Reminders:

1. **Emojis are never deleted** - Display errors are ignored
2. **All outputs in English** - Code and outputs in English
3. **Standard file structure** - Header + body + test section
4. **Layer dependencies** - Only top to bottom
5. **Naming conventions** - Engine, Manager, Analyzer difference matters

### Code Review Rejection Criteria:

❌ PR rejected if:
- Emoji deleted
- Header/footer missing
- Layer dependency violation

✅ PR approved if:
- All rules applied
- Test section exists
- Component responsibilities correct

---

**Last Updated:** 2025-01-17
**Version:** 3.0.0
**Maintainer:** SuperBot Team

**This guide must be followed by all developers and AI assistants.**
