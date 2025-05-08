"""
Retrieval-Augmented Generation (RAG) c поддержкой изображений.
Файл предоставляет две публичные корутины:

    generate_answer(query)               – только текст
    generate_answer_with_images(query)   – текст + подписи к релевантным изображениям
"""
import json
from typing import List, Tuple
import os
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from groq import Groq

from config import (
    # текстовые константы
    EMBEDDING_MODEL, FAISS_INDEX_PATH, NEWS_DATA_PATH, TOP_K,
    # для изображений
    CLIP_MODEL_NAME, BLIP_MODEL_NAME, CLIP_MAX_TOKENS, BLIP_INDEX_PATH,
    CLIP_INDEX_PATH, CLIP_VALID_PATHS_PKL, BLIP_CAPTIONS_PKL, TOP_IMAGE_K,
    # LLM
    GROQ_API_KEY, LLAMA_MODEL,
)
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
# ---------------------------------------------------------------------------
# 1. Инициализация ресурсов (загружаются однократно при импорте модуля)
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

with open(NEWS_DATA_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)
df = pd.DataFrame(_data)

# --- CLIP & BLIP -------------------------------------------------------------
print("⏳ Загружаю CLIP и BLIP – это займёт 1–2 мин на CPU …")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model     = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

clip_tokenizer = clip_processor.tokenizer          # нужен для обрезки запросов
print("⏳ Загрузил CLIP и BLIP ")
import pickle
with open(CLIP_VALID_PATHS_PKL, "rb") as f:
    clip_valid_paths: List[str] = pickle.load(f)
with open(BLIP_CAPTIONS_PKL, "rb") as f:
    blip_captions: List[str] = pickle.load(f)

# --- Groq LLM ---------------------------------------------------------------
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# 2. Вспомогательные функции
# ---------------------------------------------------------------------------
def retrieve_similar_documents(search_text: str, text_index, k: int = 5):
    """Возвращает список из k наиболее релевантных текстовых фрагментов для заданного запроса."""
    search_vector = embedding_model.encode(search_text)
    new_vector = np.array([search_vector])
    faiss.normalize_L2(new_vector)
    distances, ann = text_index.search(new_vector, k=5)
    indices = ann[0]
    top_texts = df['clean_text'].iloc[indices].reset_index(drop=True)
    results = pd.DataFrame({
        'distances': distances[0],
        'clean_text': top_texts
    })
    return results


def truncate_query(query: str, max_length: int = 77):
    """
    Возвращает готовый токенизированный запрос (input_ids) для CLIP-модели,
    обрезанный до допустимой длины (77 токенов включая спецсимволы).
    """
    encoding = clip_tokenizer(query, truncation=True, max_length=max_length, return_tensors="pt")
    token_count = encoding['input_ids'].shape[1]
    print(f"Final tokenized query length: {token_count} (should be <= {max_length})")
    return encoding


def retrieve_similar_images(query: str, clip_index, blip_index, k: int = 5):
    """
    По заданному текстовому запросу с помощью CLIP модели
    ищет наиболее релевантные изображения из clip_index,
    после чего для найденных изображений возвращает соответствующие подписи (BLIP).
    Возвращается список кортежей (путь_к_изображению, подпись).
    """
    # Токенизируем сразу до допустимой длины
    text_input = truncate_query(query)
    
    # Проверяем итоговую длину для дебага
    print(f"Input_ids shape: {text_input['input_ids'].shape}")

    # Получаем текстовое представление запроса
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_input)
    text_features = text_features.cpu().numpy().astype("float32")

    query_emb = embedding_model.encode([query]).astype("float32")
    distances, indicesb = blip_index.search(query_emb, k) 

    # Поиск в CLIP индексе
    distances, indicesc = clip_index.search(text_features, k)
    retrieved = []
    for idxc, idxb in zip(indicesc[0], indicesb[0]):
        image_path = clip_valid_paths[idxc]
        caption = blip_captions[idxb]
        retrieved.append((image_path, caption))
    return retrieved


async def generate_answer_with_images(
    query: str,
    top_text_k: int = TOP_K,
    top_image_k: int = TOP_IMAGE_K,
) -> str:
    text_index = faiss.read_index(FAISS_INDEX_PATH)
    blip_index = faiss.read_index(BLIP_INDEX_PATH)
    clip_index = faiss.read_index(CLIP_INDEX_PATH)
    
    print(query)
    # Получаем релевантные текстовые фрагменты по запросу
    context_chunks = retrieve_similar_documents(query, text_index, k=top_text_k)
    context_text = "\n\n".join(context_chunks["clean_text"])
    print(context_text)

    # Получаем релевантные изображения и их подписи
    similar_images = retrieve_similar_images(query, clip_index, blip_index, k=top_image_k)
    image_captions = [f" {caption}" for _, caption in similar_images]
    images_context = "\n\n".join(image_captions)
    print(image_captions)

    # Объединяем оба контекста
    final_context = context_text + "\n\n" + images_context
    print(final_context)
    
    prompt = (
        "Используя информацию из приведенного контекста, ответь на следующий вопрос.\n\n"
        f"Контекст:\n{final_context}\n\n"
        f"Вопрос: {query}\n\nОтвет:"
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content
    except Exception:
        return "Извините, произошла ошибка при генерации ответа."


