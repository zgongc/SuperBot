"""
Captain's Memory - SuperBot Session Memory System

Star Trek inspired captain's log system.
Persistent memory for agent and user.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class CaptainMemory:
    """
    Captain's Log - Session-based memory system.

    Usage:
        memory = CaptainMemory()

        # Add log
        memory.log("QML pattern drawing completed", category="implementation")

        # Save decision
        memory.decision("Start from Zone Head", context="QML zone drawing")

        # Save knowledge
        memory.learn("BaselineSeries is used to draw boxes", topic="charts")

        # Query
        recent = memory.get_recent_logs(5)
        decisions = memory.get_decisions(topic="QML")
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "superbot_memory.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Captain's Log (Session Logs)
                CREATE TABLE IF NOT EXISTS captains_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stardate TEXT NOT NULL,           -- YYYY-MM-DD HH:MM:SS
                    category TEXT DEFAULT 'general',  -- implementation, fix, research, decision
                    summary TEXT NOT NULL,            -- Short summary
                    details TEXT,                     -- Detailed description (optional)
                    author TEXT DEFAULT 'claude',     -- claude or user
                    session_id TEXT,                  -- Session identifier
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Decisions
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,              -- Topic (QML, OB, charts, etc.)
                    decision TEXT NOT NULL,           -- Decision made
                    context TEXT,                     -- Context/reason
                    alternatives TEXT,                -- Evaluated alternatives (JSON)
                    author TEXT DEFAULT 'claude',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Learned Knowledge (Knowledge Base)
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,              -- Topic category
                    fact TEXT NOT NULL,               -- Learned fact
                    source TEXT,                      -- Source (file, docs, experiment)
                    confidence REAL DEFAULT 1.0,      -- Confidence score (0-1)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(topic, fact)               -- Same fact not added twice
                );

                -- Project Context
                CREATE TABLE IF NOT EXISTS project_context (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_log_stardate ON captains_log(stardate);
                CREATE INDEX IF NOT EXISTS idx_log_category ON captains_log(category);
                CREATE INDEX IF NOT EXISTS idx_decisions_topic ON decisions(topic);
                CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
            """)

    # ==================== LOGGING ====================

    def log(self, summary: str, category: str = "general",
            details: str = None, author: str = "claude") -> int:
        """
        Add log to captain's log.

        Args:
            summary: Short summary (1-2 sentences)
            category: implementation, fix, research, decision, general
            details: Detailed description
            author: claude or user

        Returns:
            Added log ID
        """
        stardate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = datetime.now().strftime("%Y%m%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO captains_log (stardate, category, summary, details, author, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (stardate, category, summary, details, author, session_id))
            return cursor.lastrowid

    def get_recent_logs(self, limit: int = 10, category: str = None) -> List[Dict]:
        """Get last N log records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if category:
                rows = conn.execute("""
                    SELECT * FROM captains_log
                    WHERE category = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (category, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM captains_log
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(row) for row in rows]

    def get_today_logs(self) -> List[Dict]:
        """Get today's logs."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM captains_log
                WHERE stardate LIKE ?
                ORDER BY created_at ASC
            """, (f"{today}%",)).fetchall()
            return [dict(row) for row in rows]

    # ==================== DECISIONS ====================

    def decision(self, decision: str, topic: str,
                 context: str = None, alternatives: List[str] = None,
                 author: str = "claude") -> int:
        """
        Save decision.

        Args:
            decision: Decision made
            topic: Topic (QML, OB, architecture, etc.)
            context: Why this decision was made
            alternatives: Evaluated alternatives

        Returns:
            Decision ID
        """
        alt_json = json.dumps(alternatives) if alternatives else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO decisions (topic, decision, context, alternatives, author)
                VALUES (?, ?, ?, ?, ?)
            """, (topic, decision, context, alt_json, author))
            decision_id = cursor.lastrowid

        # Also add to log (outside connection)
        self.log(f"Decision: {decision}", category="decision",
                details=f"Topic: {topic}, Context: {context}", author=author)

        return decision_id

    def get_decisions(self, topic: str = None, limit: int = 20) -> List[Dict]:
        """Get decisions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if topic:
                rows = conn.execute("""
                    SELECT * FROM decisions
                    WHERE topic LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                """, (f"%{topic}%", limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM decisions
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()

            results = []
            for row in rows:
                d = dict(row)
                if d.get('alternatives'):
                    d['alternatives'] = json.loads(d['alternatives'])
                results.append(d)
            return results

    # ==================== KNOWLEDGE ====================

    def learn(self, fact: str, topic: str,
              source: str = None, confidence: float = 1.0) -> bool:
        """
        Save knowledge (learn).

        Args:
            fact: Learned fact
            topic: Topic (charts, smc, python, etc.)
            source: Where it was learned from
            confidence: Confidence score (0-1)

        Returns:
            True if new, False if already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO knowledge (topic, fact, source, confidence)
                    VALUES (?, ?, ?, ?)
                """, (topic, fact, source, confidence))
                return True
        except sqlite3.IntegrityError:
            # Already exists
            return False

    def recall(self, topic: str = None, search: str = None) -> List[Dict]:
        """Recall/search knowledge."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if search:
                rows = conn.execute("""
                    SELECT * FROM knowledge
                    WHERE fact LIKE ? OR topic LIKE ?
                    ORDER BY confidence DESC, created_at DESC
                """, (f"%{search}%", f"%{search}%")).fetchall()
            elif topic:
                rows = conn.execute("""
                    SELECT * FROM knowledge
                    WHERE topic = ?
                    ORDER BY confidence DESC, created_at DESC
                """, (topic,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM knowledge
                    ORDER BY created_at DESC LIMIT 50
                """).fetchall()

            return [dict(row) for row in rows]

    # ==================== PROJECT CONTEXT ====================

    def set_context(self, key: str, value: Any):
        """Set project context."""
        value_str = json.dumps(value) if not isinstance(value, str) else value

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO project_context (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value_str))

    def get_context(self, key: str = None) -> Any:
        """Get project context."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if key:
                row = conn.execute("""
                    SELECT value FROM project_context WHERE key = ?
                """, (key,)).fetchone()
                if row:
                    try:
                        return json.loads(row['value'])
                    except json.JSONDecodeError:
                        return row['value']
                return None
            else:
                rows = conn.execute("SELECT * FROM project_context").fetchall()
                result = {}
                for row in rows:
                    try:
                        result[row['key']] = json.loads(row['value'])
                    except json.JSONDecodeError:
                        result[row['key']] = row['value']
                return result

    # ==================== SESSION SUMMARY ====================

    def get_session_summary(self, max_tokens: int = 3000) -> str:
        """
        Create session summary - format for Claude's context.

        Returns:
            Summary in markdown format
        """
        summary_parts = []

        # Recent decisions
        decisions = self.get_decisions(limit=10)
        if decisions:
            summary_parts.append("## Recent Decisions")
            for d in decisions[:5]:
                summary_parts.append(f"- **{d['topic']}**: {d['decision']}")

        # Recent logs
        logs = self.get_recent_logs(limit=10)
        if logs:
            summary_parts.append("\n## Recent Work")
            for log in logs[:5]:
                summary_parts.append(f"- [{log['category']}] {log['summary']}")

        # Important knowledge
        knowledge = self.recall()
        if knowledge:
            summary_parts.append("\n## Learned Knowledge")
            for k in knowledge[:10]:
                summary_parts.append(f"- **{k['topic']}**: {k['fact']}")

        # Project context
        context = self.get_context()
        if context:
            summary_parts.append("\n## Project Context")
            for key, value in list(context.items())[:5]:
                if isinstance(value, str) and len(value) < 100:
                    summary_parts.append(f"- **{key}**: {value}")

        return "\n".join(summary_parts)

    # ==================== CLI HELPERS ====================

    def quick_log(self, message: str):
        """Quick log - for user."""
        return self.log(message, category="user_note", author="user")

    def quick_decision(self, topic: str, decision: str):
        """Quick decision - for user."""
        return self.decision(decision, topic, author="user")

    def quick_learn(self, topic: str, fact: str):
        """Quick learn - for user."""
        return self.learn(fact, topic, source="user_input")


# Singleton instance
_memory_instance = None

def get_memory() -> CaptainMemory:
    """Global memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = CaptainMemory()
    return _memory_instance


# CLI interface
if __name__ == "__main__":
    import sys

    memory = get_memory()

    if len(sys.argv) < 2:
        print("""
Captain's Log - SuperBot Memory System

Usage:
    python captain_memory.py log "message"              # Add log
    python captain_memory.py decision "topic" "decision" # Save decision
    python captain_memory.py learn "topic" "fact"       # Save knowledge
    python captain_memory.py show                       # Show recent logs
    python captain_memory.py summary                    # Show summary
    python captain_memory.py search "query"             # Search
        """)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "log" and len(sys.argv) >= 3:
        msg = " ".join(sys.argv[2:])
        log_id = memory.quick_log(msg)
        print(f"Log added (ID: {log_id})")

    elif cmd == "decision" and len(sys.argv) >= 4:
        topic = sys.argv[2]
        decision = " ".join(sys.argv[3:])
        dec_id = memory.quick_decision(topic, decision)
        print(f"Decision saved (ID: {dec_id})")

    elif cmd == "learn" and len(sys.argv) >= 4:
        topic = sys.argv[2]
        fact = " ".join(sys.argv[3:])
        is_new = memory.quick_learn(topic, fact)
        print(f"Knowledge {'saved' if is_new else 'already exists'}")

    elif cmd == "show":
        logs = memory.get_recent_logs(10)
        print("\n=== Recent Logs ===")
        for log in logs:
            print(f"[{log['stardate']}] ({log['category']}) {log['summary']}")

    elif cmd == "summary":
        print(memory.get_session_summary())

    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        results = memory.recall(search=query)
        print(f"\n=== Search: {query} ===")
        for r in results:
            print(f"[{r['topic']}] {r['fact']}")

    else:
        print("Invalid command. Help: python captain_memory.py")
