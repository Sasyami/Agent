import os
import time
import uuid
import json
import shutil
import sqlite3
import faiss
from pathlib import Path
from typing import List, Dict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import BaseDocStore

MEMORY_DIR = Path("chat_memory")
MEMORY_DIR.mkdir(exist_ok=True)
RECENT_LIMIT = 50
LONG_TERM_TOP_K = 3

# 🔹 Встроенный докстор на SQLite (хранит тексты на диске, 0 RAM)
class SQLiteDocstore(BaseDocStore):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, content TEXT, metadata TEXT)")
            conn.commit()

    def add(self, texts: Dict[str, Document]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
                [(id_, doc.page_content, json.dumps(doc.metadata, ensure_ascii=False, default=str)) for id_, doc in texts.items()]
            )
            conn.commit()

    def search(self, search_id: str) -> Union[Document, str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT content, metadata FROM docs WHERE id = ?", (search_id,)).fetchone()
        if row is None:
            return f"ID {search_id} not found"
        return Document(page_content=row[0], metadata=json.loads(row[1]))

    def search_batch(self, search_ids: List[str]) -> Dict[str, Union[Document, str]]:
        return {sid: self.search(sid) for sid in search_ids}


class LangChainMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.storage_path = MEMORY_DIR / session_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "index.faiss"
        self.id_map_path = self.storage_path / "index_to_docstore_id.json"
        self.docstore_path = self.storage_path / "docstore.db"

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.dimension = self.embeddings.client.get_sentence_embedding_dimension() or 384

        # Инициализируем дисковый докстор
        self.docstore = SQLiteDocstore(self.docstore_path)
        self._id_timeline: List[str] = []

        self._init_store()

    def _init_store(self):
        timeline_path = self.storage_path / "timeline.json"
        if timeline_path.exists():
            self._id_timeline = json.loads(timeline_path.read_text())

        # Загружаем или создаём FAISS-индекс и маппинг ID вручную (чтобы обойти InMemoryDocstore)
        if self.index_path.exists() and self.id_map_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.id_map_path, "r", encoding="utf-8") as f:
                self.index_to_docstore_id = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_to_docstore_id = {}

        # Собираем векторстор с нашим SQLite-докстором
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id
        )

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.vectorstore.index_to_docstore_id, f)
        timeline_path = self.storage_path / "timeline.json"
        timeline_path.write_text(json.dumps(self._id_timeline))
        # SQLiteDocstore сохраняет данные сам внутри .add(), тут сохранять не нужно

    def add_messages(self, messages: List[BaseMessage]):
        docs = []
        new_ids = []
        for msg in messages:
            if isinstance(msg, SystemMessage): continue
            doc_id = str(uuid.uuid4())
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content.strip()) < 5: continue

            docs.append(Document(
                page_content=content,
                metadata={
                    "role": msg.type,
                    "timestamp": time.time(),
                    "id": doc_id,
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                    "tool_calls": getattr(msg, "tool_calls", None)
                }
            ))
            new_ids.append(doc_id)
            self._id_timeline.append(doc_id)

        if docs:
            self.vectorstore.add_documents(docs, ids=new_ids)
            if len(self._id_timeline) > 1000:
                self._id_timeline = self._id_timeline[-1000:]
            self._save()

    def get_recent(self, n: int = RECENT_LIMIT) -> List[BaseMessage]:
        """Возвращает последние N сообщений напрямую из FAISS docstore."""
        recent_ids = self._id_timeline[-n:]
        msgs = []
        for doc_id in recent_ids:
            doc = self.vectorstore.docstore.search(doc_id)
            if isinstance(doc, Document):
                msgs.append(self._doc_to_msg(doc))
        return msgs

    def search_long_term(self, query: str, k: int = LONG_TERM_TOP_K) -> str:
        """Семантический поиск по всей истории."""
        if self.vectorstore.index.ntotal == 0: return ""
        docs = self.vectorstore.similarity_search(query, k=k)
        if not docs: return ""
        res = [f"- [{d.metadata['role']}] {d.page_content}" for d in docs]
        return "\n🧠 ДОЛГОСРОЧНАЯ ПАМЯТЬ:\n" + "\n".join(res)

    def _doc_to_msg(self, doc: Document) -> BaseMessage:
        role = doc.metadata.get("role")
        content = doc.page_content
        if role == "human": return HumanMessage(content=content)
        if role == "ai": return AIMessage(content=content, tool_calls=doc.metadata.get("tool_calls"))
        if role == "tool": return ToolMessage(content=content, tool_call_id=doc.metadata.get("tool_call_id"))
        return HumanMessage(content=content)

    def clear(self):
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
        self._init_store()
        self._id_timeline = []