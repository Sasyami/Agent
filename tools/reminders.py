# tools/reminders.py
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from langchain_core.tools import tool
from utils.reminders import add_reminder

@tool
def create_reminder_tool(chat_id: int, text: str, due_time: str) -> str:
    """Создает напоминание, которое бот отправит в Telegram.
    due_time: YYYY-MM-DD HH:MM (часовой пояс Europe/Moscow).
    text: текст напоминания.
    chat_id: ID чата пользователя (обязательно)."""
    try:
        msk_tz = ZoneInfo("Europe/Moscow")
        dt_msk = datetime.strptime(due_time, "%Y-%m-%d %H:%M").replace(tzinfo=msk_tz)
        due_utc = dt_msk.timestamp()
        
        if due_utc < time.time() + 30:
            return json.dumps({"error": "Время напоминания слишком близко или уже прошло (минимум через 30 сек)"}, ensure_ascii=False)
            
        rem_id = add_reminder(chat_id, text, due_utc)
        return json.dumps({
            "status": "ok", 
            "id": rem_id, 
            "message": f"Напоминание создано. Бот отправит его {due_time} МСК."
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)