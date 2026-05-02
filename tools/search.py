from ddgs import DDGS
from pydantic import BaseModel, Field
import json
from typing import Optional
from langchain_core.tools import tool  # ← добавить в импорты

class SearchInput(BaseModel):
    query: str = Field(description="Поисковый запрос на естественном языке")
    max_results: int = Field(default=3, ge=1, le=5, description="Макс. количество результатов (1-5)")
@tool
def search_tool(query: str, max_results: int = 3) -> str:
    """Ищет информацию в DuckDuckGo и возвращает структурированный JSON."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(max_results, 5)))
        
        if not results:
            return json.dumps({"error": "Ничего не найдено по запросу"}, ensure_ascii=False)

        formatted = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка поиска: {str(e)}"}, ensure_ascii=False)