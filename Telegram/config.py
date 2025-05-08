"""
Все пути и ключевые параметры сосредоточены здесь.
При смене ресурсов достаточно поправить только этот файл.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- credentials -------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")          # ключ Groq
API_ID       = int(os.getenv("API_ID"))
API_HASH     = os.getenv("API_HASH")
BOT_TOKEN    = os.getenv("TOKEN")

# --- текстовый RAG -----------------------------------------------------------
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "/Applications/Study/Diploma/news_date.faiss"
NEWS_DATA_PATH   = "/Applications/Study/Diploma/gpt_news.json"
TOP_K            = 5                               # кол-во текстовых чанков

# --- мультимодальный (изображения) -------------------------------------------
CLIP_MODEL_NAME      = "openai/clip-vit-large-patch14"
BLIP_MODEL_NAME      = "Salesforce/blip-image-captioning-large"
CLIP_MAX_TOKENS      = 77

CLIP_INDEX_PATH      = "/Applications/Study/Diploma/clip_index_last.faiss"
BLIP_INDEX_PATH      = "/Applications/Study/Diploma/blip_index_last.faiss"
CLIP_VALID_PATHS_PKL = "/Applications/Study/Diploma/clip_valid_paths.pkl"
BLIP_CAPTIONS_PKL    = "/Applications/Study/Diploma/blip_captions.pkl"
TOP_IMAGE_K          = 5                            # кол-во изображений

# --- LLM ---------------------------------------------------------------------
LLAMA_MODEL = "llama-3.3-70b-versatile"
