import sqlite3
import time
import asyncio
import logging
from pathlib import Path

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
            status TEXT DEFAULT 'pending',
            created_at REAL DEFAULT (strftime('%s', 'now'))
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

def delete_reminder(rem_id: int):
    """Удаляет напоминание из БД."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM reminders WHERE id = ?", (rem_id,))
        conn.commit()

def cleanup_expired():
    """Удаляет просроченные 'pending' и старые 'sent' записи (хранит историю 24ч)."""
    now = time.time()
    cutoff = now - 86400  
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM reminders WHERE due_utc < ? AND status = 'pending'", (now,))
        conn.execute("DELETE FROM reminders WHERE status = 'sent' AND due_utc < ?", (cutoff,))
        conn.commit()

async def start_reminder_checker(bot):
    """Фоновая задача: проверяет БД каждые 10 сек, отправляет и чистит."""
    init_db()
    logger.info("Запущен фоновый проверщик напоминаний")
    
    cleanup_counter = 0
    while True:
        try:
            due = get_due_reminders()
            for rem_id, chat_id, text in due:
                try:
                    await bot.send_message(
                        chat_id=chat_id, 
                        text=f"*Напоминание:*\n{text}", 
                        parse_mode="Markdown"
                    )
                    delete_reminder(rem_id)
                    logger.info(f"Напоминание #{rem_id} отправлено и удалено из БД")
                except Exception as e:
                    logger.error(f"Ошибка отправки #{rem_id}: {e}")
            
            cleanup_counter += 1
            if cleanup_counter >= 6:
                cleanup_expired()
                cleanup_counter = 0
                
        except Exception as e:
            logger.error(f"Ошибка в проверщике: {e}")
        await asyncio.sleep(10)

def cancel_reminder(rem_id: int, chat_id: int) -> bool:
    """Отменяет напоминание. Возвращает True, если запись найдена и удалена."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM reminders WHERE id = ? AND chat_id = ? AND status = 'pending'",
            (rem_id, chat_id)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM reminders WHERE id = ?", (rem_id,))
            conn.commit()
            return True
        return False