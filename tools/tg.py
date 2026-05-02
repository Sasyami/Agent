import os
import httpx
from pydantic import BaseModel, Field
from typing import Optional
import json
from langchain_core.tools import tool  # ← добавить в импорты

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_API_URL = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

class SendMessageInput(BaseModel):
    chat_id: int | str = Field(description="ID чата, пользователя или канала (число или @username канала/группы)")
    text: str = Field(description="Текст сообщения. Макс. 4096 символов.")
    parse_mode: Optional[str] = Field(default=None, description="Форматирование: 'HTML' или 'MarkdownV2'. Оставь None для простого текста.")
@tool
def send_telegram_tool(chat_id: int | str, text: str, parse_mode: Optional[str] = None) -> str:
    """Отправляет сообщение в Telegram через Bot API. Возвращает статус и ID сообщения."""
    if not TG_BOT_TOKEN:
        return json.dumps({"error": "Переменная TG_BOT_TOKEN не найдена в .env"}, ensure_ascii=False)

    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True  # Экономим токены, отключаем превью ссылок
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode.upper()

    try:
        with httpx.Client() as client:
            response = client.post(f"{TG_API_URL}/sendMessage", json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            if data.get("ok"):
                return json.dumps({
                    "status": "ok",
                    "message_id": data["result"]["message_id"],
                    "chat_id": data["result"]["chat"]["id"]
                }, ensure_ascii=False)
            else:
                return json.dumps({"error": f"Telegram API error: {data.get('description', 'Unknown')}"}, ensure_ascii=False)
    except httpx.RequestError as e:
        return json.dumps({"error": f"Ошибка сети/таймаут: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Непредвиденная ошибка: {str(e)}"}, ensure_ascii=False)