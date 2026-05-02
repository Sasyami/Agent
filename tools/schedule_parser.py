import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
import json
from typing import Optional
from langchain_core.tools import tool  # ← добавить в импорты

class ParseScheduleInput(BaseModel):
    url: str = Field(description="Полный URL страницы с расписанием")
    target_date: Optional[str] = Field(default=None, description="Фильтр по дате YYYY-MM-DD (опционально)")
@tool
def parse_schedule_tool(url: str, target_date: Optional[str] = None) -> str:
    """Парсит HTML и извлекает элементы расписания. Возвращает JSON."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        schedule = []

        # 🔍 УНИВЕРСАЛЬНЫЙ ПОИСК: адаптируй селекторы под конкретный сайт
        items = soup.select(".schedule-item, .event, .match, .fixture, table tr, li")
        for item in items:
            text = item.get_text(separator=" ", strip=True)
            if not text or len(text) < 8:
                continue
            if target_date and target_date not in text:
                continue
            schedule.append({"raw": text})

        if not schedule:
            # Fallback: отдаём LLM контекст страницы, чтобы она сама извлекла данные
            title = soup.title.string if soup.title else "Unknown"
            preview = soup.get_text(separator=" ", strip=True)[:600]
            return json.dumps({
                "warning": "Стандартные селекторы не нашли расписание. Используй этот контекст.",
                "title": title,
                "preview": preview
            }, ensure_ascii=False)

        return json.dumps(schedule[:15], ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка парсинга: {str(e)}"}, ensure_ascii=False)