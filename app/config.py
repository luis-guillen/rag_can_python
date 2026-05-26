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
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 2200)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 250)
MIN_CHUNK_CHARS = _env_int("RAG_MIN_CHUNK_CHARS", 150)
# Skip documentos sospechosamente grandes (típicamente PDFs/JPGs leídos como
# texto por el crawler). 500K caracteres ≈ ~150 páginas reales: cualquier
# página HTML legítima cabe holgado, los binarios mal interpretados no.
MAX_DOC_CHARS = _env_int("RAG_MAX_DOC_CHARS", 500_000)

# --- Retrieval / API -------------------------------------------------------
TOP_K_DEFAULT = _env_int("RAG_TOP_K", 5)
TOP_K_MAX = _env_int("RAG_TOP_K_MAX", 20)
ANSWER_PREVIEW_CHARS = _env_int("RAG_ANSWER_PREVIEW_CHARS", 350)

# --- Alias de fuentes ------------------------------------------------------
# Mapeo "menciona-esto-en-la-pregunta" -> domain real. Si el usuario escribe
# "¿Qué dice Memoria de Lanzarote sobre...?" priorizamos resultados de ese
# dominio en la búsqueda. Configurable vía RAG_SOURCE_ALIASES con formato:
#   "alias1=domain1;alias2=domain2;..."
DEFAULT_SOURCE_ALIASES_RAW = (
    "canarias azul=canarias-azul.iatext.ulpgc.es;"
    "iatext=canarias-azul.iatext.ulpgc.es;"
    "memoria de lanzarote=memoriadelanzarote.com;"
    "museo canario=elmuseocanario.com;"
    "el museo canario=elmuseocanario.com;"
    "academia canaria de la lengua=www.academiacanarialengua.org;"
    "academia canaria=www.academiacanarialengua.org;"
    "cultura gran canaria=cultura.grancanaria.com;"
    "cultura de gran canaria=cultura.grancanaria.com;"
    "izuran=izuran.blogspot.com"
)


def _parse_aliases(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        alias, domain = entry.split("=", 1)
        alias = alias.strip().lower()
        domain = domain.strip()
        if alias and domain:
            out[alias] = domain
    return out


SOURCE_ALIASES = _parse_aliases(_env_str("RAG_SOURCE_ALIASES", DEFAULT_SOURCE_ALIASES_RAW))


# --- Filtros de ruido ------------------------------------------------------
# Páginas administrativas/legales/búsqueda que normalmente no aportan
# contenido informativo y conviene NO indexar. Se comparan contra `url` y
# `title` con plegado de acentos y límites de palabra.
DEFAULT_NOISE_PATTERNS = (
    "aviso-legal,aviso legal,politica,privacidad,cookies,"
    "login,buscar,search,tags,feed,rss"
)
NOISE_PATTERNS = [
    p.strip()
    for p in _env_str("RAG_NOISE_PATTERNS", DEFAULT_NOISE_PATTERNS).split(",")
    if p.strip()
]


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
