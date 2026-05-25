"""FastAPI: endpoint /query consumido por Chat.aspx.

Levantar con:
    uvicorn app.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .models import QueryRequest, QueryResponse
from .retrieval import build_extractive_answer, get_client, get_model, search

app = FastAPI(
    title="RAG Canarias",
    description="Servicio de retrieval extractivo sobre el corpus generado por ASP.NET Web Forms.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup() -> None:
    # Precargar modelo y verificar conexión a Qdrant.
    try:
        get_model()
    except Exception as exc:  # pragma: no cover - log de arranque
        print(f"[api] WARN: no se pudo precargar el modelo: {exc}")


@app.get("/health")
def health() -> dict:
    info: dict = {
        "status": "ok",
        "collection": config.COLLECTION_NAME,
        "model": config.EMBEDDING_MODEL,
        "qdrant_url": config.QDRANT_URL,
    }
    try:
        client = get_client()
        info["qdrant_collection_exists"] = client.collection_exists(config.COLLECTION_NAME)
        if info["qdrant_collection_exists"]:
            count = client.count(collection_name=config.COLLECTION_NAME, exact=False)
            info["qdrant_points"] = count.count
    except Exception as exc:
        info["status"] = "degraded"
        info["qdrant_error"] = str(exc)
    return info


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    top_k = min(req.top_k, config.TOP_K_MAX)
    try:
        sources = search(req.question, top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error consultando Qdrant: {exc}") from exc

    answer = build_extractive_answer(req.question, sources)
    return QueryResponse(answer=answer, sources=sources)
