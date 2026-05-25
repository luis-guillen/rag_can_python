"""Configuración central del módulo RAG.

Todas las opciones pueden sobrescribirse vía variables de entorno o un fichero .env
en la raíz del proyecto. Las rutas se resuelven respecto a la raíz del repo.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return int(value)


# --- Rutas -----------------------------------------------------------------
DEFAULT_CORPUS = r"C:\Users\jaime\source\repos\luis-guillen\rag_can_webform\App_Data\p2"
CORPUS_DIR = Path(_env_str("RAG_CORPUS_DIR", DEFAULT_CORPUS))
DATA_DIR = Path(_env_str("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))
CHUNKS_FILE = Path(_env_str("RAG_CHUNKS_FILE", str(DATA_DIR / "chunks.jsonl")))

# --- Qdrant ----------------------------------------------------------------
QDRANT_URL = _env_str("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = _env_str("RAG_COLLECTION", "rag_canarias")

# --- Embeddings ------------------------------------------------------------
# Modelo multilingüe ligero, ideal para RTX 3050 Laptop o CPU.
EMBEDDING_MODEL = _env_str("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
# multilingual-e5-small produce vectores de 384 dimensiones.
EMBEDDING_DIM = _env_int("RAG_EMBED_DIM", 384)
EMBED_BATCH_SIZE = _env_int("RAG_EMBED_BATCH", 32)
INDEX_UPSERT_BATCH = _env_int("RAG_UPSERT_BATCH", 128)

# Los modelos E5 requieren prefijos específicos en query/passage.
E5_QUERY_PREFIX = _env_str("RAG_E5_QUERY_PREFIX", "query: ")
E5_PASSAGE_PREFIX = _env_str("RAG_E5_PASSAGE_PREFIX", "passage: ")

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 1200)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 180)
MIN_CHUNK_CHARS = _env_int("RAG_MIN_CHUNK_CHARS", 80)

# --- Retrieval / API -------------------------------------------------------
TOP_K_DEFAULT = _env_int("RAG_TOP_K", 5)
TOP_K_MAX = _env_int("RAG_TOP_K_MAX", 20)
ANSWER_PREVIEW_CHARS = _env_int("RAG_ANSWER_PREVIEW_CHARS", 350)

# --- CORS para Chat.aspx ---------------------------------------------------
# IIS Express / ASP.NET típicamente sirven en localhost con un puerto dinámico.
# Mantener una lista permisiva en desarrollo; ajustar en producción.
ALLOWED_ORIGINS = [
    o.strip()
    for o in _env_str(
        "RAG_ALLOWED_ORIGINS",
        "http://localhost,http://127.0.0.1",
    ).split(",")
    if o.strip()
]
