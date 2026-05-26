"""Lógica de retrieval compartida entre el CLI (query.py) y la API (api.py).

Mantiene un único `SentenceTransformer` cargado en memoria. Detecta menciones
explícitas de fuentes conocidas en la pregunta (vía `config.SOURCE_ALIASES`)
y prioriza resultados de ese dominio.

Reglas:
  - Si el llamante pasa `prefer_domain` con `hard_filter=True` (CLI `--domain`),
    se devuelven SOLO chunks de ese dominio. Sin fallback. Lista vacía si no
    hay matches.
  - Si `prefer_domain` viene de auto-detección (`hard_filter=False`), primero
    se intentan todos los hits del dominio; si faltan para llenar `top_k`,
    se rellena con hits globales marcados con `is_fallback=True`.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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


# --- Detección de fuente --------------------------------------------------


def _normalize(s: Optional[str]) -> str:
    """Minúsculas + plegado de acentos (NFKD)."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def detect_source(question: str) -> Optional[Tuple[str, str]]:
    """Devuelve `(alias_detectado, domain)` o None.

    Ordena alias por longitud descendente para que
    "academia canaria de la lengua" gane sobre "academia canaria".
    """
    norm_q = _normalize(question)
    aliases = sorted(config.SOURCE_ALIASES.keys(), key=len, reverse=True)
    for alias in aliases:
        norm_alias = _normalize(alias)
        if not norm_alias:
            continue
        if re.search(rf"\b{re.escape(norm_alias)}\b", norm_q):
            return alias, config.SOURCE_ALIASES[alias]
    return None


# --- Búsqueda --------------------------------------------------------------


def _to_source(hit, *, is_fallback: bool = False) -> Source:
    payload = hit.payload or {}
    text = payload.get("text", "")
    preview = text[: config.ANSWER_PREVIEW_CHARS]
    if len(text) > config.ANSWER_PREVIEW_CHARS:
        preview += "..."
    return Source(
        score=float(hit.score),
        title=payload.get("title"),
        url=payload.get("url"),
        domain=payload.get("domain"),
        source_name=payload.get("source_name"),
        text_preview=preview,
        is_fallback=is_fallback,
    )


def _embed_query(question: str) -> List[float]:
    model = get_model()
    return (
        model.encode(
            [f"{config.E5_QUERY_PREFIX}{question}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        .tolist()
    )


def _domain_filter(domain: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="domain",
                match=qmodels.MatchValue(value=domain),
            )
        ]
    )


def search(
    question: str,
    top_k: int,
    *,
    prefer_domain: Optional[str] = None,
    hard_filter: bool = False,
    auto_detect: bool = True,
) -> List[Source]:
    """Busca en Qdrant.

    Parámetros:
        prefer_domain: dominio a priorizar (None → sin preferencia).
        hard_filter:   si True, prefer_domain es OBLIGATORIO; sin fallback.
        auto_detect:   si prefer_domain es None y True, detecta vía alias.
                       La auto-detección SIEMPRE produce hard_filter=False.
    """
    client = get_client()

    if prefer_domain is None and auto_detect:
        match = detect_source(question)
        if match is not None:
            prefer_domain = match[1]
            # auto-detección no fuerza filtro duro
            hard_filter = False

    vector = _embed_query(question)

    # --- Sin preferencia de dominio: query global plana --------------------
    if not prefer_domain:
        hits = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [_to_source(h) for h in hits]

    # --- Con preferencia: query filtrada -----------------------------------
    filtered_hits = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=_domain_filter(prefer_domain),
        with_payload=True,
    ).points

    # FILTRO DURO: solo lo que sale del dominio, punto.
    if hard_filter:
        return [_to_source(h, is_fallback=False) for h in filtered_hits]

    # FILTRO SUAVE: dominio primero, fallback global marcado como is_fallback.
    if len(filtered_hits) >= top_k:
        return [_to_source(h, is_fallback=False) for h in filtered_hits[:top_k]]

    seen_ids = {hit.id for hit in filtered_hits}
    extras = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k * 2,
        with_payload=True,
    ).points

    results: List[Source] = [_to_source(h, is_fallback=False) for h in filtered_hits]
    for hit in extras:
        if hit.id in seen_ids:
            continue
        results.append(_to_source(hit, is_fallback=True))
        seen_ids.add(hit.id)
        if len(results) >= top_k:
            break
    return results[:top_k]


# --- Helpers de debug ------------------------------------------------------


def inspect_collection() -> Dict[str, Any]:
    """Devuelve metadatos de la colección y un payload de ejemplo.

    Útil para verificar que `payload["domain"]` existe y tiene los valores
    esperados antes de fiarse del filtro Qdrant.
    """
    client = get_client()
    info: Dict[str, Any] = {
        "collection": config.COLLECTION_NAME,
        "qdrant_url": config.QDRANT_URL,
    }
    if not client.collection_exists(config.COLLECTION_NAME):
        info["exists"] = False
        return info

    info["exists"] = True
    coll = client.get_collection(config.COLLECTION_NAME)
    info["points_count"] = coll.points_count
    info["vector_size"] = coll.config.params.vectors.size
    info["distance"] = str(coll.config.params.vectors.distance)

    # Sacar un punto de muestra
    sample, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if sample:
        payload = sample[0].payload or {}
        info["sample_payload_keys"] = sorted(payload.keys())
        info["sample_domain_field_present"] = "domain" in payload
        info["sample_domain_value"] = payload.get("domain")
        info["sample_url"] = payload.get("url")
        info["sample_title"] = payload.get("title")

    # Contar puntos por dominio (top 20) usando scroll + agregación local
    domain_counts: Dict[str, int] = {}
    offset = None
    scanned = 0
    while scanned < 5000:
        batch, offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=500,
            with_payload=["domain"],
            with_vectors=False,
            offset=offset,
        )
        if not batch:
            break
        for p in batch:
            d = (p.payload or {}).get("domain") or "<missing>"
            domain_counts[d] = domain_counts.get(d, 0) + 1
        scanned += len(batch)
        if offset is None:
            break
    info["domain_counts_sampled"] = dict(
        sorted(domain_counts.items(), key=lambda kv: -kv[1])
    )
    info["domain_counts_sample_size"] = scanned
    return info


# --- Respuesta extractiva --------------------------------------------------


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
        tag = " [fallback]" if src.is_fallback else ""
        lines.append(f"[{i}]{tag} {title} ({src.domain or '-'}) — score={src.score:.3f}")
        lines.append(src.text_preview)
        if src.url:
            lines.append(f"Fuente: {src.url}")
        lines.append("")
    lines.append("Nota: respuesta extractiva sin LLM. Conecta un modelo generativo para síntesis.")
    return "\n".join(lines)
