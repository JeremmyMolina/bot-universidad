import telebot
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN ---
# TU TOKEN REAL
TOKEN = '8567781272:AAFwSG8dtzUwoKEyBlAvNPQ0yzL-rRGeU94'
ИМЯ_ФАЙЛА = 'datos.xlsx'

# --- LÓGICA DEL BOT ---
def cargar_datos_excel(файл):
    print(f"🔄 Buscando archivo: {файл}...")
    if not os.path.exists(файл):
        print(f"❌ ERROR: No encuentro '{файл}' en esta carpeta.")
        return None
    try:
        все_листы = pd.read_excel(файл, sheet_name=None)
        dfs = []
        for nombre, df in все_листы.items():
            df.columns = df.columns.str.strip()
            if 'Вопросы' in df.columns and 'Ответы' in df.columns:
                dfs.append(df)
        if not dfs: return None
        full_data = pd.concat(dfs, ignore_index=True).dropna(subset=['Вопросы', 'Ответы'])
        print(f"✅ Datos cargados: {len(full_data)} preguntas.")
        return full_data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

class ChatbotAI:
    def __init__(self, datos):
        self.preguntas = datos['Вопросы'].tolist()
        self.respuestas = datos['Ответы'].tolist()
        print("🧠 Entrenando cerebro...")
        self.vectorizer = TfidfVectorizer()
        self.matriz = self.vectorizer.fit_transform(self.preguntas)

    def responder(self, texto):
        vec = self.vectorizer.transform([texto])
        similitud = cosine_similarity(vec, self.matriz)
        idx = np.argmax(similitud)
        score = similitud[0][idx]
        if score < 0.2:
            return "Извините, я не нашел информации по этому вопросу.", score
        return self.respuestas[idx], score

# --- ARRANQUE ---
datos = cargar_datos_excel(ИМЯ_ФАЙЛА)
if datos is not None:
    bot_cerebro = ChatbotAI(datos)
    bot = telebot.TeleBot(TOKEN)
    print("🚀 EL BOT ESTÁ VIVO EN TU PC. (No cierres esta ventana)")

    @bot.message_handler(commands=['start'])
    def welcome(message):
        bot.reply_to(message, "¡Hola! Soy el asistente virtual. Pregúntame algo.")

    @bot.message_handler(func=lambda m: True)
    def chat(message):
        try:
            resp, conf = bot_cerebro.responder(message.text)
            print(f"📩 Usuario: {message.text} | 🤖 Bot: {resp} ({conf:.2f})")
            bot.reply_to(message, resp)
        except Exception as e:
            print(f"Error: {e}")

    # Esto mantiene al bot despierto siempre, incluso si hay fallos de red
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            print(f"⚠️ Error de conexión: {e}")
            time.sleep(5)
else:
    input("❌ Error al cargar datos. Presiona Enter para salir.")