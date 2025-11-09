# ==============================
# 🧩 Telegram AI Price Bot
# ==============================

import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import numpy as np

# 🔑 Вставь свои ключи сюда
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ==============================
# 🛒 Эмуляция парсинга магазинов
# ==============================
def fake_parser(query: str):
    """
    Эмуляция результатов парсинга с answear.sk, prm.com и adidas.com.
    В реальном проекте сюда можно подставить парсеры BeautifulSoup.
    """
    sample_data = [
        {"name": f"{query} Adidas Originals Black 42", "price": 99.9, "source": "adidas.com"},
        {"name": f"{query} Adidas Originals 42 čierne", "price": 97.5, "source": "answear.sk"},
        {"name": f"{query} by Adidas, black shoes size 42", "price": 103.0, "source": "prm.com"},
        {"name": f"{query} limited edition white 42", "price": 110.0, "source": "adidas.com"},
    ]
    random.shuffle(sample_data)
    return sample_data


# ==============================
# 🧠 AI-модуль для сравнения
# ==============================

# Загружаем бесплатную модель эмбеддингов
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    # Получаем векторное представление текста
    emb = model.encode(text)
    return emb.tolist()



def group_similar_products(products):
    """Группирует одинаковые товары по косинусному сходству."""
    embeddings = [get_embedding(p["name"]) for p in products]
    embeddings = np.array(embeddings)

    used = set()
    groups = []

    for i, emb1 in enumerate(embeddings):
        if i in used:
            continue
        group = [products[i]]
        used.add(i)
        for j, emb2 in enumerate(embeddings):
            if j in used:
                continue
            sim = cosine_similarity([emb1], [emb2])[0][0]
            if sim > 0.9:  # порог схожести
                group.append(products[j])
                used.add(j)
        groups.append(group)

    return groups


# ==============================
# 🤖 Telegram Bot
# ==============================
async def start(update: Update, context):
    await update.message.reply_text(
        "👋 Привет! Я бот, который сравнивает цены на одинаковые товары.\n\n"
        "Просто напиши название товара, например:\n"
        "`Adidas Stan Smith 42`",
        parse_mode="Markdown",
    )


async def search(update: Update, context):
    query = update.message.text.strip()
    await update.message.reply_text("🔎 Ищу товары, подожди немного...")

    # 1️⃣ Получаем товары с разных "магазинов"
    products = fake_parser(query)

    # 2️⃣ Группируем похожие товары
    groups = group_similar_products(products)

    # 3️⃣ Формируем сообщение
    reply = ""
    for g in groups:
        reply += f"\n🛍 *{g[0]['name']}*\n"
        sorted_group = sorted(g, key=lambda x: x["price"])
        for item in sorted_group:
            reply += f" - {item['source']}: *{item['price']} €*\n"
    if not reply:
        reply = "😔 Ничего не найдено."
    await update.message.reply_text(reply, parse_mode="Markdown")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    print("🚀 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
