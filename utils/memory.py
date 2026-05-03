# utils/memory.py
import os
import json
import time
import shutil
import sqlite3
import faiss
from pathlib import Path
from typing import List, Dict, Union, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.base import AddableMixin, Docstore
MEMORY_DIR = Path("chat_memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class SQLiteDocstore(Docstore, AddableMixin):
    """Дисковый докстор, совместимый с FAISS в LangChain."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL
            )""")
            conn.commit()

    def add(self, texts: Dict[str, Any]) -> None:
        """LangChain передаёт Dict[str, Document] или Dict[str, str]."""
        with sqlite3.connect(self.db_path) as conn:
            for doc_id, content in texts.items():
                if isinstance(content, Document):
                    text = content.page_content
                    meta = content.metadata
                else:
                    text = str(content)
                    meta = {}
                conn.execute(
                    "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
                    (doc_id, text, json.dumps(meta, ensure_ascii=False, default=str))
                )
            conn.commit()

    def search(self, search_id: str) -> Document:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content, metadata FROM docs WHERE id = ?", 
                (search_id,)
            ).fetchone()

        if row is None:
            raise KeyError(f"ID {search_id} not found")

        return Document(
            page_content=row[0],
            metadata=json.loads(row[1])
        )

    def search_batch(self, search_ids: List[str]) -> Dict[str, Union[Document, str]]:
        return {sid: self.search(sid) for sid in search_ids}


class LangChainMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.storage_path = MEMORY_DIR / session_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "index.faiss"
        self.id_map_path = self.storage_path / "id_map.json"
        self.docstore_path = self.storage_path / "docstore.db"

        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        vec = self.embeddings.embed_query("test")
        self.dimension = len(vec)

        self.docstore = SQLiteDocstore(self.docstore_path)
        self._init_store()

    def _init_store(self):
        if self.index_path.exists() and self.id_map_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.id_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
                self.index_to_docstore_id = {int(k): v for k, v in raw_map.items()}
        else:
            self.index = faiss.IndexFlatL2(int(self.dimension))
            self.index_to_docstore_id = {}

        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
        )

    def add_messages(self, messages: List[BaseMessage]):
        docs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content.strip()) < 5:
                continue
            docs.append(Document(
                page_content=content,
                metadata={"role": msg.type, "timestamp": time.time()}
            ))
        
        if docs:
            # LangChain сам сгенерирует ID, обновит index_to_docstore_id и вызовет docstore.add()
            self.vectorstore.add_documents(docs)
            self._save()

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.vectorstore.index_to_docstore_id, f)

    def search_long_term(self, query: str, k: int = 3, threshold: float = 0.6) -> str:
        """Семантический поиск с фильтрацией релевантности."""
        
        
        if self.vectorstore.index.ntotal == 0:
            return ""
        
        # 🔹 Шаг 2: поиск + фильтрация по порогу
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        if not docs_and_scores:
            return ""
        
        relevant = []
        for doc, l2_dist in docs_and_scores:
            cosine_sim = 1 - (l2_dist ** 2) / 2
            if cosine_sim >= threshold:
                relevant.append(f"- [{doc.metadata.get('role', '?')}] {doc.page_content}")
        
        if not relevant:
            return ""
        
        return "\n🧠 ДОЛГОСРОЧНАЯ ПАМЯТЬ:\n" + "\n".join(relevant)

    def clear(self):
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
        self._init_store()