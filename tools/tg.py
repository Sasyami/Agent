# tools/telegram.py
import os
import httpx
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.tools import tool
from dotenv import load_dotenv  # ← добавили

# 🔹 Явно загружаем .env (если ещё не загружен в main)
load_dotenv()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

TG_API_URL = f"https://api.telegram.org/bot{TG_BOT_TOKEN}" if TG_BOT_TOKEN else ""

class SendMessageInput(BaseModel):
    chat_id: int | str = Field(description="ID чата. Для личных сообщений: число (напр. 123456789). Для канала/группы: @username или -100...")
    text: str = Field(description="Текст сообщения. Макс. 4096 символов.")
    parse_mode: Optional[str] = Field(default=None, description="Форматирование: 'HTML' или 'MarkdownV2'. Оставь None для простого текста.")

@tool
def send_telegram_tool(chat_id: int | str, text: str, parse_mode: Optional[str] = None) -> str:
    """Отправляет сообщение в Telegram. Перед первым использованием"""
    
    # 🔹 Явная проверка токена
    if not TG_BOT_TOKEN or not TG_API_URL:
        return json.dumps({
            "error": "TG_BOT_TOKEN не настроен. Добавь в .env: TG_BOT_TOKEN=123456:ABC-DEF...",
            "hint": "Создай бота в @BotFather, нажми /start, узнай chat_id в @userinfobot"
        }, ensure_ascii=False)

    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode.upper()

    try:
        # 🔹 Telegram API лучше работает с form-data, а не JSON
        with httpx.Client() as client:
            response = client.post(
                f"{TG_API_URL}/sendMessage",
                data=payload,  # ← form-urlencoded (надёжнее, чем json=)
                timeout=10.0
            )
            
            # 🔹 Логирование для отладки (раскомментируй при проблемах)
            print(f"[TG DEBUG] Status: {response.status_code}, Body: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("ok"):
                return json.dumps({
                    "status": "ok",
                    "message_id": data["result"]["message_id"],
                    "chat_id": data["result"]["chat"]["id"],
                    "text": data["result"]["text"][:50] + "..." if len(data["result"]["text"]) > 50 else data["result"]["text"]
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "error": f"Telegram API error: {data.get('description', 'Unknown')}",
                    "error_code": data.get("error_code"),
                    "parameters": data.get("parameters")
                }, ensure_ascii=False)
                
    except httpx.HTTPStatusError as e:
        return json.dumps({
            "error": f"HTTP ошибка {e.response.status_code}: {e.response.text[:200]}"
        }, ensure_ascii=False)
    except httpx.RequestError as e:
        return json.dumps({"error": f"Сетевая ошибка/таймаут: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Непредвиденная ошибка: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)