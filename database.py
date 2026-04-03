import sqlite3
import uuid
from typing import List, Dict

class ChatDatabase:
    def __init__(self, db_path="research_chat.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        """Creates the necessary tables if they don't exist."""
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            );
        """)
        self.conn.commit()

    def create_session(self) -> str:
        """Generates a new UUID for a chat session."""
        session_id = str(uuid.uuid4())
        self.cursor.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
        self.conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Saves a message to the database."""
        self.cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.conn.commit()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieves the last N messages for the LLM context window."""
        self.cursor.execute("""
            SELECT role, content FROM messages 
            WHERE session_id = ? 
            ORDER BY created_at ASC 
            LIMIT ?
        """, (session_id, limit))
        
        # Format it exactly how llama-cpp-python expects it!
        return [{"role": row[0], "content": row[1]} for row in self.cursor.fetchall()]

# Quick Test
if __name__ == "__main__":
    db = ChatDatabase()
    sid = db.create_session()
    db.add_message(sid, "user", "What is QLoRA?")
    db.add_message(sid, "assistant", "QLoRA is a quantization method...")
    print(db.get_history(sid))