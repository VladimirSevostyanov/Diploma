from aiogram import Router, types, F
from aiogram.filters import CommandStart, Command

from config import BOT_TOKEN
from keyboard import reply_keyboard, ask_button_text
from rag import generate_answer_with_images

router = Router()

@router.message(CommandStart())
async def start_cmd(message: types.Message):
    welcome_text = (
        "Привет! Добро пожаловать в новостного бота.\n"
        "Этот бот может отвечать на ваши вопросы на основе свежих новостей.\n"
        "Нажмите \"Задать вопрос\" ниже или просто введите свой вопрос."
    )
    await message.answer(welcome_text, reply_markup=reply_keyboard)

@router.message(Command("help"))
async def help_cmd(message: types.Message):
    help_text = (
        "Использование:\n"
        "/start - начать работу с ботом и показать меню\n"
        "/help - показать эту справку\n"
        "/about - информация о боте\n\n"
        "Просто введите любой вопрос, и бот попытается найти ответ в новостях."
    )
    await message.answer(help_text, reply_markup=reply_keyboard)

@router.message(Command("about"))
async def about_cmd(message: types.Message):
    about_text = (
        "Этот бот использует подход Retrieval-Augmented Generation (RAG).\n"
        "Он ищет релевантные новости и отвечает с помощью модели LLaMA 3 70B."
    )
    await message.answer(about_text, reply_markup=reply_keyboard)

@router.message(F.text.casefold() == ask_button_text.casefold())
async def ask_button_handler(message: types.Message):
    prompt_text = "Пожалуйста, введите ваш вопрос."
    await message.answer(prompt_text)

@router.message(F.text)
async def handle_query(message: types.Message):
    user_query = message.text.strip()
    if not user_query:
        return 


    answer = await generate_answer_with_images(user_query)
    if answer:
        await message.answer(answer, reply_markup=reply_keyboard)
    else:
        await message.answer("Извините, я не смог найти ответ на ваш вопрос.", reply_markup=reply_keyboard)