import os
import json
import time
import sqlite3
import faiss
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

MEMORY_DIR = Path("chat_memory")
MEMORY_DIR.mkdir(exist_ok=True)
SQLITE_LIMIT = 50
FAISS_TOP_K = 3
MIGRATION_BATCH = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def _to_text(msg: BaseMessage) -> str:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return f"[{msg.type}] {content}" if len(content.strip()) > 10 else ""

class Memory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.db_path = MEMORY_DIR / f"{session_id}.db"
        self.faiss_idx_path = MEMORY_DIR / f"{session_id}.faiss"
        self.faiss_meta_path = MEMORY_DIR / f"{session_id}_meta.jsonl"
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.IndexFlatL2(384)
        self.metadata: List[dict] = []
        self._init_db()
        self._load_faiss()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, timestamp REAL
        )""")
        conn.commit(); conn.close()

    def _load_faiss(self):
        if self.faiss_idx_path.exists() and self.faiss_meta_path.exists():
            self.index = faiss.read_index(str(self.faiss_idx_path))
            with open(self.faiss_meta_path, "r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]

    def _save_faiss(self):
        faiss.write_index(self.index, str(self.faiss_idx_path))
        with open(self.faiss_meta_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def add_messages(self, messages: List[BaseMessage]):
        conn = sqlite3.connect(self.db_path)
        for msg in messages:
            if isinstance(msg, SystemMessage): continue
            conn.execute("""INSERT INTO messages (type, content, tool_call_id, tool_calls, timestamp)
                            VALUES (?, ?, ?, ?, ?)""",
                         (msg.type, msg.content, 
                          msg.tool_call_id if hasattr(msg, 'tool_call_id') else None,
                          json.dumps(msg.tool_calls) if hasattr(msg, 'tool_calls') and msg.tool_calls else None,
                          time.time()))
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if count > SQLITE_LIMIT:
            self._migrate_oldest(conn)
        conn.close()

    def _migrate_oldest(self, conn):
        rows = conn.execute("""SELECT id, type, content, timestamp FROM messages
                               ORDER BY timestamp ASC LIMIT ?""", (MIGRATION_BATCH,)).fetchall()
        for msg_id, mtype, content, ts in rows:
            if content and mtype in ("human", "ai"):
                vec = self.embedder.encode([f"[{mtype}] {content}"], show_progress_bar=False)
                self.index.add(vec)
                self.metadata.append({"id": msg_id, "type": mtype, "content": content, "timestamp": ts})
            conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
        conn.commit()
        self._save_faiss()

    def load_short_term(self) -> List[BaseMessage]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""SELECT type, content, tool_call_id, tool_calls FROM messages
                               ORDER BY timestamp ASC""").fetchall()
        msgs = []
        for t, c, tc_id, tc_json in rows:
            if t == "human": msgs.append(HumanMessage(content=c))
            elif t == "ai": 
                tc = json.loads(tc_json) if tc_json else None
                msgs.append(AIMessage(content=c, tool_calls=tc))
            elif t == "tool": msgs.append(ToolMessage(content=c, tool_call_id=tc_id))
        conn.close()
        return msgs

    def search_long_term(self, query: str) -> str:
        if self.index.ntotal == 0: return ""
        vec = self.embedder.encode([query], show_progress_bar=False)
        _, indices = self.index.search(vec, min(FAISS_TOP_K, self.index.ntotal))
        res = [f"- [{self.metadata[i]['type']}] {self.metadata[i]['content']}" for i in indices[0] if i < len(self.metadata)]
        return "\n🧠 ДОЛГОСРОЧНАЯ ПАМЯТЬ:\n" + "\n".join(res) if res else ""

    def clear(self):
        for p in [self.db_path, self.faiss_idx_path, self.faiss_meta_path]:
            if p.exists(): p.unlink()
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []