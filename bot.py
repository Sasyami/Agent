# bot.py
import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# Импортируем твоего агента
from utils.agent import run_agent 
from dotenv import load_dotenv

load_dotenv()

# Настройки логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота из .env
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
if not TG_BOT_TOKEN:
    raise ValueError("❌ Не найден TG_BOT_TOKEN в .env")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие сообщения от пользователя."""
    user = update.effective_user
    chat_id = user.id
    text = update.message.text
    
    logger.info(f"👤 Пользователь {user.username} ({chat_id}): {text}")
    
    # Отправляем статус "печатает...", чтобы пользователь видел активность
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        # 🔹 Вызываем твоего агента
        # session_id = chat_id, чтобы у каждого пользователя была своя история
        response = run_agent(text, session_id=str(chat_id))
        
        # 🔹 Отправляем ответ (разбиваем на части, если длинный)
        # Лимит Телеграма ~4096 символов
        for chunk in [response[i:i+4000] for i in range(0, len(response), 4000)]:
            await update.message.reply_text(chunk)
            
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")

def main():
    """Запуск бота."""
    logger.info("🚀 Запуск Телеграм-бота...")
    
    # Создаем приложение
    application = Application.builder().token(TG_BOT_TOKEN).build()
    
    # Регистрируем обработчик текстовых сообщений
    # Игнорируем команды (начинающиеся с /), чтобы не ломать стандартные функции бота
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Запускаем поллинг (опрос сервера)
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()