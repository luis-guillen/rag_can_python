"""Genera chunks a partir del corpus y los persiste en data/chunks.jsonl.

Usa RecursiveCharacterTextSplitter de langchain-text-splitters cuando esté
disponible (chunking respetuoso con párrafos), con un fallback propio si no.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from . import config
from .corpus_utils import iter_corpus_entries, load_document
from .models import Chunk, Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover - fallback
    _HAS_LANGCHAIN = False


def _build_splitter(chunk_size: int, chunk_overlap: int):
    if _HAS_LANGCHAIN:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )
    return None


def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback: corte por párrafos con ventana deslizante por caracteres."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= chunk_size:
                buf = p
            else:
                # Párrafo enorme: ventana deslizante
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    chunks.append(p[start:end])
                    if end == len(p):
                        break
                    start = end - chunk_overlap
                buf = ""
    if buf:
        chunks.append(buf)

    # Aplicar overlap entre chunks adyacentes
    if chunk_overlap > 0 and len(chunks) > 1:
        with_overlap: List[str] = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            with_overlap.append(f"{tail}\n\n{curr}")
        return with_overlap
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = _build_splitter(chunk_size, chunk_overlap)
    if splitter is not None:
        return [c for c in splitter.split_text(text) if c.strip()]
    return [c for c in _simple_split(text, chunk_size, chunk_overlap) if c.strip()]


def _chunk_id(source_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}:{index}:{text[:64]}".encode("utf-8")).hexdigest()[:16]
    return f"{source_id[:24]}-{index:04d}-{digest}"


def _payload_for(doc: Document) -> dict:
    """Aplana metadata para que sea trivial buscar/filtrar en Qdrant."""
    dom = doc.domain_metadata
    page = doc.page_metadata
    return {
        "url": page.url,
        "title": page.title,
        "domain": dom.domain,
        "domain_slug": dom.domain_slug,
        "source_name": dom.source_name,
        "source_type": dom.source_type,
        "category": dom.category,
        "region": dom.region,
        "island": dom.island,
        "language": dom.language,
        "license": dom.license,
        "topics": dom.topics,
        "file": page.file,
        "depth": page.depth,
        "source_id": doc.source_id,
    }


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> Iterable[Chunk]:
    pieces = split_text(doc.text, chunk_size, chunk_overlap)
    base_payload = _payload_for(doc)
    for i, piece in enumerate(pieces):
        if len(piece) < config.MIN_CHUNK_CHARS:
            continue
        cid = _chunk_id(doc.source_id, i, piece)
        payload = dict(base_payload)
        payload["chunk_id"] = cid
        yield Chunk(
            chunk_id=cid,
            source_id=doc.source_id,
            text=piece,
            metadata=payload,
        )


def build_chunks(
    corpus_dir: Path,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs_ok = docs_skipped = chunks_total = 0

    entries = list(iter_corpus_entries(corpus_dir))
    with output_path.open("w", encoding="utf-8") as out:
        for entry in tqdm(entries, desc="Chunking", unit="doc"):
            doc = load_document(entry)
            if doc is None:
                docs_skipped += 1
                continue
            docs_ok += 1
            for chunk in chunk_document(doc, chunk_size, chunk_overlap):
                out.write(
                    json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
                )
                chunks_total += 1

    return {
        "docs_processed": docs_ok,
        "docs_skipped": docs_skipped,
        "chunks_written": chunks_total,
        "output": str(output_path),
        "splitter": "langchain.RecursiveCharacterTextSplitter" if _HAS_LANGCHAIN else "builtin",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera chunks RAG desde el corpus ASP.NET.")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--output", default=str(config.CHUNKS_FILE))
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"ERROR: corpus not found: {corpus}", file=sys.stderr)
        sys.exit(2)

    stats = build_chunks(corpus, Path(args.output), args.chunk_size, args.chunk_overlap)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
