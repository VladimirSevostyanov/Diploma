import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from dotenv import load_dotenv

load_dotenv()

from config import BOT_TOKEN
from handlers import router
import warnings
from multiprocessing import resource_tracker
warnings.filterwarnings("ignore", category=UserWarning, module=resource_tracker.__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


dp.include_router(router)

commands = [
    types.BotCommand(command="start", description="Начать работу с ботом"),
    types.BotCommand(command="help", description="Помощь по использованию"),
    types.BotCommand(command="about", description="О боте")
]
async def set_commands():
    await bot.set_my_commands(commands=commands, scope=types.BotCommandScopeAllPrivateChats())

async def main():
    logging.basicConfig(level=logging.INFO)
    await bot.delete_webhook(drop_pending_updates=True)
    await set_commands()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())