"""Genera embeddings y los sube a Qdrant.

Crea (o recrea) la colección `rag_canarias` y carga los chunks de
data/chunks.jsonl, anteponiendo el prefijo `passage:` requerido por los
modelos E5 antes de codificarlos.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Iterator, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from . import config


def _select_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(device: str):
    from sentence_transformers import SentenceTransformer

    print(f"[embed_index] Cargando modelo: {config.EMBEDDING_MODEL} en {device}")
    return SentenceTransformer(config.EMBEDDING_MODEL, device=device)


def _iter_chunks(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and not recreate:
        print(f"[embed_index] Colección '{name}' ya existe (usar --recreate para borrar).")
        return
    if exists and recreate:
        print(f"[embed_index] Borrando colección existente '{name}'...")
        client.delete_collection(name)

    print(f"[embed_index] Creando colección '{name}' (dim={dim}, COSINE)")
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )


def _point_id_for(chunk_id: str) -> str:
    """Qdrant requiere UUID o entero como ID. Derivamos un UUID estable del chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _batched(it: Iterator[dict], size: int) -> Iterator[List[dict]]:
    batch: List[dict] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_chunks(
    chunks_path: Path,
    *,
    recreate: bool,
    batch_size: int,
    upsert_batch: int,
) -> dict:
    if not chunks_path.exists():
        print(f"ERROR: no existe {chunks_path}. Ejecuta primero `python -m app.chunk`.", file=sys.stderr)
        sys.exit(2)

    device = _select_device()
    model = _load_model(device)

    # Verificar dimensión real del modelo
    real_dim = model.get_sentence_embedding_dimension()
    if real_dim != config.EMBEDDING_DIM:
        print(
            f"[embed_index] AVISO: dim del modelo = {real_dim} pero config.EMBEDDING_DIM = {config.EMBEDDING_DIM}.",
            f"Usando {real_dim}.",
        )

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    _ensure_collection(client, config.COLLECTION_NAME, real_dim, recreate)

    total = 0
    # Contamos para barra de progreso (lectura barata)
    with chunks_path.open("r", encoding="utf-8") as f:
        n_chunks = sum(1 for line in f if line.strip())

    progress = tqdm(total=n_chunks, desc=f"Indexing -> {config.COLLECTION_NAME}", unit="chunk")
    for batch in _batched(_iter_chunks(chunks_path), batch_size):
        texts = [f"{config.E5_PASSAGE_PREFIX}{c['text']}" for c in batch]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        points = [
            qmodels.PointStruct(
                id=_point_id_for(c["chunk_id"]),
                vector=vec.tolist(),
                payload={
                    **c["metadata"],
                    "text": c["text"],
                    "chunk_id": c["chunk_id"],
                    "source_id": c["source_id"],
                },
            )
            for c, vec in zip(batch, vectors)
        ]

        # Hacer upsert en sub-batches para no saturar el cliente HTTP
        for i in range(0, len(points), upsert_batch):
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points[i : i + upsert_batch],
                wait=False,
            )
        total += len(points)
        progress.update(len(points))
    progress.close()

    return {
        "collection": config.COLLECTION_NAME,
        "model": config.EMBEDDING_MODEL,
        "device": device,
        "dim": real_dim,
        "points_upserted": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding e indexación en Qdrant.")
    parser.add_argument("--chunks", default=str(config.CHUNKS_FILE))
    parser.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE)
    parser.add_argument("--upsert-batch", type=int, default=config.INDEX_UPSERT_BATCH)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Borrar y recrear la colección antes de indexar.",
    )
    args = parser.parse_args()

    stats = index_chunks(
        Path(args.chunks),
        recreate=args.recreate,
        batch_size=args.batch_size,
        upsert_batch=args.upsert_batch,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
