"""
Captain's Memory - SuperBot Session Memory System

Star Trek'ten ilham alan seyir defteri sistemi.
Agent ve kullanıcı için kalıcı hafıza.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class CaptainMemory:
    """
    Kaptan'ın Seyir Defteri - Session bazlı hafıza sistemi

    Kullanım:
        memory = CaptainMemory()

        # Log ekle
        memory.log("QML pattern çizimi tamamlandı", category="implementation")

        # Karar kaydet
        memory.decision("Zone Head'den başlar", context="QML zone drawing")

        # Bilgi kaydet
        memory.learn("BaselineSeries box çizmek için kullanılır", topic="charts")

        # Sorgula
        recent = memory.get_recent_logs(5)
        decisions = memory.get_decisions(topic="QML")
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "superbot_memory.db"
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Veritabanı tablolarını oluştur"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Seyir Defteri (Session Logs)
                CREATE TABLE IF NOT EXISTS captains_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stardate TEXT NOT NULL,           -- YYYY-MM-DD HH:MM:SS
                    category TEXT DEFAULT 'general',  -- implementation, fix, research, decision
                    summary TEXT NOT NULL,            -- Kısa özet
                    details TEXT,                     -- Detaylı açıklama (opsiyonel)
                    author TEXT DEFAULT 'claude',     -- claude veya user
                    session_id TEXT,                  -- Session tanımlayıcı
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Kararlar (Decisions)
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,              -- Konu (QML, OB, charts, vb.)
                    decision TEXT NOT NULL,           -- Alınan karar
                    context TEXT,                     -- Bağlam/neden
                    alternatives TEXT,                -- Değerlendirilen alternatifler (JSON)
                    author TEXT DEFAULT 'claude',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Öğrenilen Bilgiler (Knowledge Base)
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,              -- Konu kategorisi
                    fact TEXT NOT NULL,               -- Öğrenilen bilgi
                    source TEXT,                      -- Kaynak (file, docs, experiment)
                    confidence REAL DEFAULT 1.0,      -- Güven skoru (0-1)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(topic, fact)               -- Aynı bilgi tekrar eklenmez
                );

                -- Proje Bağlamı (Project Context)
                CREATE TABLE IF NOT EXISTS project_context (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- İndeksler
                CREATE INDEX IF NOT EXISTS idx_log_stardate ON captains_log(stardate);
                CREATE INDEX IF NOT EXISTS idx_log_category ON captains_log(category);
                CREATE INDEX IF NOT EXISTS idx_decisions_topic ON decisions(topic);
                CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
            """)

    # ==================== LOGGING ====================

    def log(self, summary: str, category: str = "general",
            details: str = None, author: str = "claude") -> int:
        """
        Seyir defterine log ekle

        Args:
            summary: Kısa özet (1-2 cümle)
            category: implementation, fix, research, decision, general
            details: Detaylı açıklama
            author: claude veya user

        Returns:
            Eklenen log ID'si
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
        """Son N log kaydını getir"""
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
        """Bugünkü logları getir"""
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
        Karar kaydet

        Args:
            decision: Alınan karar
            topic: Konu (QML, OB, architecture, vb.)
            context: Neden bu karar alındı
            alternatives: Değerlendirilen alternatifler

        Returns:
            Karar ID'si
        """
        alt_json = json.dumps(alternatives) if alternatives else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO decisions (topic, decision, context, alternatives, author)
                VALUES (?, ?, ?, ?, ?)
            """, (topic, decision, context, alt_json, author))
            decision_id = cursor.lastrowid

        # Aynı zamanda log'a da ekle (connection dışında)
        self.log(f"Karar: {decision}", category="decision",
                details=f"Topic: {topic}, Context: {context}", author=author)

        return decision_id

    def get_decisions(self, topic: str = None, limit: int = 20) -> List[Dict]:
        """Kararları getir"""
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
        Bilgi kaydet (öğren)

        Args:
            fact: Öğrenilen bilgi
            topic: Konu (charts, smc, python, vb.)
            source: Nereden öğrenildi
            confidence: Güven skoru (0-1)

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
            # Zaten var
            return False

    def recall(self, topic: str = None, search: str = None) -> List[Dict]:
        """Bilgi hatırla/ara"""
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
        """Proje bağlamı ayarla"""
        value_str = json.dumps(value) if not isinstance(value, str) else value

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO project_context (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value_str))

    def get_context(self, key: str = None) -> Any:
        """Proje bağlamı getir"""
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
        Session özeti oluştur - Claude'un context'e ekleyeceği format

        Returns:
            Markdown formatında özet
        """
        summary_parts = []

        # Son kararlar
        decisions = self.get_decisions(limit=10)
        if decisions:
            summary_parts.append("## Son Kararlar")
            for d in decisions[:5]:
                summary_parts.append(f"- **{d['topic']}**: {d['decision']}")

        # Son loglar
        logs = self.get_recent_logs(limit=10)
        if logs:
            summary_parts.append("\n## Son Çalışmalar")
            for log in logs[:5]:
                summary_parts.append(f"- [{log['category']}] {log['summary']}")

        # Önemli bilgiler
        knowledge = self.recall()
        if knowledge:
            summary_parts.append("\n## Öğrenilen Bilgiler")
            for k in knowledge[:10]:
                summary_parts.append(f"- **{k['topic']}**: {k['fact']}")

        # Proje bağlamı
        context = self.get_context()
        if context:
            summary_parts.append("\n## Proje Bağlamı")
            for key, value in list(context.items())[:5]:
                if isinstance(value, str) and len(value) < 100:
                    summary_parts.append(f"- **{key}**: {value}")

        return "\n".join(summary_parts)

    # ==================== CLI HELPERS ====================

    def quick_log(self, message: str):
        """Hızlı log - kullanıcı için"""
        return self.log(message, category="user_note", author="user")

    def quick_decision(self, topic: str, decision: str):
        """Hızlı karar - kullanıcı için"""
        return self.decision(decision, topic, author="user")

    def quick_learn(self, topic: str, fact: str):
        """Hızlı öğrenme - kullanıcı için"""
        return self.learn(fact, topic, source="user_input")


# Singleton instance
_memory_instance = None

def get_memory() -> CaptainMemory:
    """Global memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = CaptainMemory()
    return _memory_instance


# CLI arayüzü
if __name__ == "__main__":
    import sys

    memory = get_memory()

    if len(sys.argv) < 2:
        print("""
Kaptan'ın Seyir Defteri - SuperBot Memory System

Kullanım:
    python captain_memory.py log "mesaj"              # Log ekle
    python captain_memory.py decision "topic" "karar" # Karar kaydet
    python captain_memory.py learn "topic" "bilgi"    # Bilgi kaydet
    python captain_memory.py show                     # Son logları göster
    python captain_memory.py summary                  # Özet göster
    python captain_memory.py search "arama"           # Ara
        """)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "log" and len(sys.argv) >= 3:
        msg = " ".join(sys.argv[2:])
        log_id = memory.quick_log(msg)
        print(f"Log eklendi (ID: {log_id})")

    elif cmd == "decision" and len(sys.argv) >= 4:
        topic = sys.argv[2]
        decision = " ".join(sys.argv[3:])
        dec_id = memory.quick_decision(topic, decision)
        print(f"Karar kaydedildi (ID: {dec_id})")

    elif cmd == "learn" and len(sys.argv) >= 4:
        topic = sys.argv[2]
        fact = " ".join(sys.argv[3:])
        is_new = memory.quick_learn(topic, fact)
        print(f"Bilgi {'kaydedildi' if is_new else 'zaten mevcut'}")

    elif cmd == "show":
        logs = memory.get_recent_logs(10)
        print("\n=== Son Loglar ===")
        for log in logs:
            print(f"[{log['stardate']}] ({log['category']}) {log['summary']}")

    elif cmd == "summary":
        print(memory.get_session_summary())

    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        results = memory.recall(search=query)
        print(f"\n=== Arama: {query} ===")
        for r in results:
            print(f"[{r['topic']}] {r['fact']}")

    else:
        print("Geçersiz komut. Yardım için: python captain_memory.py")
