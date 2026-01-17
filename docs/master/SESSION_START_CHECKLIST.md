# 🚀 Session Start Checklist

> **For AI Assistants**: Follow this list at the start of every new session

---

## ✅ Checklist

### 1️⃣ Quick Context Loading (5 minutes)
- [ ] Read `docs/claude/context_guide.md`
  - Learn critical rules
  - See component map
  - Get quick reference

### 2️⃣ Project Vision Understanding (10 minutes)
- [ ] Read `docs/claude/PROJECT_VISION.md` ⭐ **IMPORTANT**
  - Why are we doing this?
  - What are the success criteria?
  - Solo developer, backtest priority
  - Replay mode, MTF, hybrid strategy

- [ ] Read `docs/plans/implementation_plan.md`
  - What are we building?
  - What is the technology stack?
  - How do modules work?

- [ ] Read `docs/plans/rules.md`
  - Plan-First principle
  - Backtest-First principle
  - Modularity and observability

### 3️⃣ Detailed Rules (If Needed)
- [ ] Read `docs/claude/claude_rules.md`
  - Emoji preservation
  - Turkish localization
  - File structure standard
  - Core engine usage
  - Component organization

---

## 🎯 Session Start Commands

### Minimum (Quick start):
```
"Read docs/claude/context_guide.md and give a summary"
```

### Full (Comprehensive context):
```
"I'm starting a session. Read these files in order:
1. docs/claude/context_guide.md
2. docs/claude/PROJECT_VISION.md
3. docs/plans/implementation_plan.md
4. docs/plans/rules.md

Then give a brief summary about the project."
```

### Context Refresh (Mid-session):
```
"Refresh context - read docs/claude/context_guide.md"
```

---

## 📊 Context Loading Levels

| Level | Files | Duration | When? |
|-------|-------|----------|-------|
| **Quick** | context_guide.md | 2 min | For small changes |
| **Standard** | context_guide + implementation_plan | 5 min | Normal development |
| **Full** | All docs | 15 min | Major feature development |

---

## 🧠 Context Priority Order

1. **context_guide.md** - Quick reference (PRIORITY 1)
2. **PROJECT_VISION.md** - Why are we doing this? Success criteria (PRIORITY 2)
3. **implementation_plan.md** - What are we building? Technology stack
4. **rules.md** - Development principles
5. **claude_rules.md** - Detailed rules
6. **system_architecture.md** - Architecture details (if needed)

---

## 💡 Mid-Session Reminder

If Claude does any of the following, refresh context:

- ❌ Creating custom logger (`logging.getLogger`)
- ❌ Removing emojis
- ❌ Writing English log/exception
- ❌ Rewriting existing component
- ❌ Layer dependency violation

**Command:**
```
"We lost context. Read docs/master/context_guide.md and remember the rules"
```

---

## 🎓 Learning Notes

### Critical Rules (Never Forget):
1. ✅ Always use `get_logger()` and `get_config()`
2. ✅ Preserve emojis
3. ✅ All output in Turkish
4. ✅ Add `from __future__ import annotations`
5. ✅ Check component map before writing new code

### Common Components:
- **RiskManager**: Position sizing + risk control
- **OrderManager**: Order validation + sending
- **PositionManager**: Position lifecycle
- **WebSocketEngine**: WebSocket management
- **MultiTimeframeEngine**: Timeframe aggregation

---

**Version:** 1.0.0
**Last Updated:** 2025-11-14
**Maintainer:** SuperBot Team
