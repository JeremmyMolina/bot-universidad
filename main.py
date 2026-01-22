import telebot
from telebot import apihelper
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- КОНФИГУРАЦИЯ ---
TOKEN = '8567781272:AAFwSG8dtzUwoKEyBlAvNPQ0yzL-rRGeU94'
DATA_FILE = 'datos.xlsx'

# --- НАСТРОЙКА ПРОКСИ ---
apihelper.proxy = {'https': 'http://proxy.server:3128'}

# --- РАБОТА С ДАННЫМИ И ML ---

def load_data(filename):
    """Чтение и подготовка датасета из Excel."""
    print(f" Чтение файла: {filename}...")
    
    if not os.path.exists(filename):
        print(f" ОШИБКА: Файл '{filename}' не найден в директории проекта.")
        return None

    try:
        # Читаем все листы сразу
        all_sheets = pd.read_excel(filename, sheet_name=None)
        dfs = []
        
        for name, df in all_sheets.items():
            # Убираем пробелы в названиях колонок
            df.columns = df.columns.str.strip()
            
            # Проверка структуры листа
            if 'Вопросы' in df.columns and 'Ответы' in df.columns:
                dfs.append(df)
        
        if not dfs:
            return None
            
        # Объединяем данные и удаляем пустые строки
        full_data = pd.concat(dfs, ignore_index=True).dropna(subset=['Вопросы', 'Ответы'])
        print(f" База знаний загружена. Всего записей: {len(full_data)}.")
        return full_data

    except Exception as e:
        print(f" Критическая ошибка при чтении: {e}")
        return None

class FAQBotModel:
    """Класс для обработки запросов на основе TF-IDF."""
    def __init__(self, data):
        self.questions = data['Вопросы'].tolist()
        self.answers = data['Ответы'].tolist()
        
        print("⚙️ Векторизация данных и обучение модели...")
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def get_response(self, text):
        # Преобразуем запрос пользователя в вектор
        vec = self.vectorizer.transform([text])
        
        # Считаем косинусное сходство
        similarities = cosine_similarity(vec, self.matrix)
        idx = np.argmax(similarities)
        score = similarities[0][idx]
        
        # Фильтр по порогу уверенности (защита от ложных срабатываний)
        if score < 0.2:
            return "Извините, я не нашел информации по этому вопросу в базе знаний.", score
            
        return self.answers[idx], score

# --- ЗАПУСК БОТА ---

dataset = load_data(DATA_FILE)

if dataset is not None:
    # Инициализация модели
    model = FAQBotModel(dataset)
    # Инициализация API Telegram
    bot = telebot.TeleBot(TOKEN)
    
    print(" Сервер запущен. Ожидание сообщений...")

    # Обработчик команды /start
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        welcome_text = (
            "Здравствуйте! Я виртуальный помощник учебной программы. \n"
            "Вы можете задать мне вопросы о расписании, оплате или кураторах."
        )
        try:
            bot.reply_to(message, welcome_text)
        except Exception as e:
            print(f" Ошибка отправки приветствия: {e}")

    # Обработчик текстовых сообщений
    @bot.message_handler(func=lambda m: True)
    def handle_message(message):
        try:
            # Получение ответа от ML-модели
            response_text, confidence = model.get_response(message.text)
            
            # Логирование в консоль (для отладки)
            print(f" User: {message.text} |  Bot: {response_text[:30]}... (Score: {confidence:.2f})")
            
            bot.reply_to(message, response_text)
            
        except Exception as e:
            print(f" Ошибка обработки сообщения: {e}")

    # Бесконечный цикл polling с обработкой ошибок сети
    while True:
        try:
            bot.polling(none_stop=True, interval=2) # interval=2 снижает нагрузку на прокси
        except Exception as e:
            print(f" Потеря соединения с Telegram API: {e}")
            print(" Переподключение через 10 секунд...")
            time.sleep(10)
else:
    input("❌ Ошибка запуска: база данных пуста или не найдена. Нажмите Enter для выхода.")


