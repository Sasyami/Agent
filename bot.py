# bot.py (замени main() и handle_message)
import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from utils.agent import run_agent
from utils.reminders import start_reminder_checker
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_user.id
    text = update.message.text
    logger.info(f"Пользователь {chat_id}: {text}")
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        response = run_agent(text, session_id=str(chat_id), chat_id=chat_id)
        for chunk in [response[i:i+4000] for i in range(0, len(response), 4000)]:
            await update.message.reply_text(chunk)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        await update.message.reply_text("Ошибка обработки. Попробуйте позже.")

async def main():
    application = Application.builder().token(TG_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    asyncio.create_task(start_reminder_checker(application.bot))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    logger.info("Бот запущен. Жду сообщений...")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Остановка бота...")
        await application.stop()

if __name__ == "__main__":
    asyncio.run(main())