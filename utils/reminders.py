# utils/reminders.py
import sqlite3
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
DB_PATH = Path("data/reminders.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            due_utc REAL NOT NULL,
            status TEXT DEFAULT 'pending'
        )""")
        conn.commit()

def add_reminder(chat_id: int, text: str, due_utc: float) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("INSERT INTO reminders (chat_id, text, due_utc) VALUES (?, ?, ?)", (chat_id, text, due_utc))
        conn.commit()
        return cur.lastrowid

def get_due_reminders():
    now = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT id, chat_id, text FROM reminders WHERE due_utc <= ? AND status = 'pending'", 
            (now,)
        ).fetchall()

def mark_sent(rem_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE reminders SET status = 'sent' WHERE id = ?", (rem_id,))
        conn.commit()

async def start_reminder_checker(bot):
    """Фоновая задача: проверяет БД каждые 10 сек и отправляет уведомления."""
    init_db()
    logger.info("⏰ Запущен фоновый проверщик напоминаний")
    while True:
        try:
            due = get_due_reminders()
            for rem_id, chat_id, text in due:
                try:
                    await bot.send_message(
                        chat_id=chat_id, 
                        text=f"⏰ *Напоминание:*\n{text}", 
                        parse_mode="Markdown"
                    )
                    mark_sent(rem_id)
                    logger.info(f"✅ Напоминание #{rem_id} отправлено в {chat_id}")
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки #{rem_id}: {e}")
        except Exception as e:
            logger.error(f"Ошибка в проверщике: {e}")
        await asyncio.sleep(10)