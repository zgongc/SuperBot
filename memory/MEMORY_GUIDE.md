# Captain's Memory - AI Agent Guide

> **Purpose:** How AI agents should use the memory system to persist knowledge across sessions.

---

## Quick Start (For AI Agents)

### When User Says "Remember this" or "Save this"
```python
from memory.captain_memory import get_memory
m = get_memory()

# Save knowledge
m.learn("BaselineSeries is used to draw boxes", topic="charts")
```

### When User Says "What did we decide about X?"
```python
decisions = m.get_decisions(topic="X")
for d in decisions:
    print(f"{d['topic']}: {d['decision']}")
```

### When User Says "What did we do today?"
```python
logs = m.get_today_logs()
for log in logs:
    print(f"[{log['category']}] {log['summary']}")
```

---

## The 4 Memory Types

### 1. **Logs** (Captain's Log)
For recording what was done in a session.

```python
m.log("Implemented RSI indicator", category="implementation")
m.log("Fixed timezone bug", category="fix")
m.log("Researched WebSocket reconnection", category="research")
```

**Categories:**
- `implementation` - New feature/code written
- `fix` - Bug fix
- `research` - Investigation/learning
- `decision` - Decision made (auto-added by decision())
- `general` - Default

### 2. **Decisions**
For recording architectural/design decisions with context.

```python
m.decision(
    decision="Use SQLite for local, PostgreSQL for production",
    topic="database",
    context="Need simple local dev, scalable production",
    alternatives=["SQLite only", "PostgreSQL only", "MongoDB"]
)
```

**When to use:**
- Architecture choices
- Library/framework selections
- Design pattern decisions
- Trade-offs made

### 3. **Knowledge** (Learn)
For storing facts that should be remembered permanently.

```python
m.learn(
    fact="Binance WebSocket reconnects automatically after 24h",
    topic="binance",
    source="binance_docs",
    confidence=1.0
)
```

**When to use:**
- API behaviors discovered
- Code patterns that work
- User preferences
- Project-specific knowledge

### 4. **Context**
For storing project-wide settings/state.

```python
m.set_context("current_phase", "backtest_development")
m.set_context("priority_module", "indicators")

# Retrieve
phase = m.get_context("current_phase")  # "backtest_development"
all_context = m.get_context()  # Returns all as dict
```

---

## Session Start Protocol

At the beginning of every session, AI agents should:

```python
from memory.captain_memory import get_memory
m = get_memory()

# Get session summary
summary = m.get_session_summary()
print(summary)
```

Or from terminal:
```bash
python memory/captain_memory.py summary
```

This returns a markdown summary of:
- Recent decisions
- Recent work logs
- Learned knowledge
- Project context

---

## Common Patterns

### Pattern 1: After Completing a Task
```python
m.log("Completed RSI indicator with divergence detection", category="implementation")
```

### Pattern 2: After Making a Design Decision
```python
m.decision(
    decision="RSI divergence uses 14-period lookback",
    topic="indicators",
    context="Standard RSI period, matches TradingView default"
)
```

### Pattern 3: After Discovering Something
```python
m.learn(
    fact="pandas_ta RSI returns NaN for first 14 rows",
    topic="pandas_ta",
    source="testing"
)
```

### Pattern 4: User Teaches Something
```python
# User: "Remember that I prefer type hints on all functions"
m.learn(
    fact="User prefers type hints on all functions",
    topic="code_style",
    source="user_preference"
)
```

### Pattern 5: Searching Past Knowledge
```python
# User: "What do we know about WebSocket?"
results = m.recall(search="WebSocket")
for r in results:
    print(f"[{r['topic']}] {r['fact']}")
```

---

## CLI Commands Reference

```bash
# Add a log entry
python memory/captain_memory.py log "Did something important"

# Record a decision
python memory/captain_memory.py decision "topic" "the decision made"

# Learn/remember something
python memory/captain_memory.py learn "topic" "the fact to remember"

# Show recent logs
python memory/captain_memory.py show

# Get session summary
python memory/captain_memory.py summary

# Search knowledge
python memory/captain_memory.py search "keyword"
```

---

## Database Schema

The memory is stored in SQLite at `memory/superbot_memory.db`.

### Tables:
1. **captains_log** - Session logs with timestamp, category, summary, details
2. **decisions** - Decisions with topic, context, alternatives (JSON)
3. **knowledge** - Facts with topic, source, confidence score (unique per topic+fact)
4. **project_context** - Key-value store for project state

---

## Best Practices for AI Agents

### DO:
- Log significant work at the end of implementation
- Record decisions with context and alternatives considered
- Learn facts that will be useful in future sessions
- Search memory before asking user for information
- Get session summary at start of conversation

### DON'T:
- Log every small change (be selective)
- Store temporary/session-specific data as knowledge
- Duplicate existing knowledge (it will be rejected)
- Forget to check memory before making repeated decisions

---

## Example: Full Session Flow

```python
from memory.captain_memory import get_memory

# 1. Session start - get context
m = get_memory()
print(m.get_session_summary())

# 2. User asks about past decision
decisions = m.get_decisions(topic="database")
# Returns: "Use SQLite for local, PostgreSQL for production"

# 3. Work on task...
# ... coding ...

# 4. Task complete - log it
m.log("Added PostgreSQL support to config_engine", category="implementation")

# 5. Made a decision during work
m.decision(
    decision="Use asyncpg for PostgreSQL connections",
    topic="database",
    context="Better async support than psycopg2"
)

# 6. Discovered something useful
m.learn(
    fact="asyncpg requires Python 3.8+",
    topic="asyncpg",
    source="documentation"
)
```

---

**Version:** 1.0.0
**Last Updated:** 2025-01-18
**For:** AI Agents working on SuperBot
