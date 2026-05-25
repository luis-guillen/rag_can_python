"""Lógica de retrieval compartida entre el CLI (query.py) y la API (api.py).

Mantiene un único `SentenceTransformer` cargado en memoria.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from qdrant_client import QdrantClient

from . import config
from .models import Source


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


@lru_cache(maxsize=1)
def get_model():
    # Import perezoso: torch/sentence_transformers son pesados.
    from sentence_transformers import SentenceTransformer

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return SentenceTransformer(config.EMBEDDING_MODEL, device=device)


def search(question: str, top_k: int) -> List[Source]:
    model = get_model()
    client = get_client()

    query_text = f"{config.E5_QUERY_PREFIX}{question}"
    vector = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    hits = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    results: List[Source] = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        preview = text[: config.ANSWER_PREVIEW_CHARS]
        if len(text) > config.ANSWER_PREVIEW_CHARS:
            preview += "..."
        results.append(
            Source(
                score=float(hit.score),
                title=payload.get("title"),
                url=payload.get("url"),
                domain=payload.get("domain"),
                source_name=payload.get("source_name"),
                text_preview=preview,
            )
        )
    return results


def build_extractive_answer(question: str, sources: List[Source]) -> str:
    """Respuesta extractiva sencilla, sin LLM.

    Concatena los fragmentos más relevantes con su fuente para que el usuario
    pueda evaluar la información antes de conectar un modelo generativo.
    """
    if not sources:
        return (
            "No he encontrado contenido relevante en el corpus indexado para esta pregunta. "
            "Comprueba que la indexación ha terminado y que la colección Qdrant no está vacía."
        )

    lines = [f"Sobre «{question}», esto es lo más relevante del corpus indexado:\n"]
    for i, src in enumerate(sources, start=1):
        title = src.title or src.source_name or src.domain or "Fuente"
        lines.append(f"[{i}] {title} ({src.domain or '-'}) — score={src.score:.3f}")
        lines.append(src.text_preview)
        if src.url:
            lines.append(f"Fuente: {src.url}")
        lines.append("")
    lines.append("Nota: respuesta extractiva sin LLM. Conecta un modelo generativo para síntesis.")
    return "\n".join(lines)
