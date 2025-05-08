from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

ask_button_text = "Задать вопрос"
reply_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=ask_button_text)]], 
    resize_keyboard=True,
    input_field_placeholder="Введите ваш вопрос..."
)