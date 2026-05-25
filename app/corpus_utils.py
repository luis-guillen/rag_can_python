"""Utilidades compartidas para recorrer y cargar el corpus generado por ASP.NET."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .models import Document, DomainMetadata, PageMetadata


@dataclass
class CorpusEntry:
    """Par .txt + .metadata.json encontrado en el corpus."""

    txt_path: Path
    meta_path: Path


def iter_corpus_entries(corpus_dir: Path) -> Iterator[CorpusEntry]:
    """Recorre el corpus y devuelve cada par .txt + .metadata.json.

    Solo emite entradas con AMBOS ficheros presentes.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for txt_path in sorted(corpus_dir.rglob("*.txt")):
        meta_path = txt_path.parent / (txt_path.stem + ".metadata.json")
        if meta_path.exists():
            yield CorpusEntry(txt_path=txt_path, meta_path=meta_path)


def load_metadata(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_document(entry: CorpusEntry) -> Optional[Document]:
    """Carga un documento completo a partir de un CorpusEntry.

    Devuelve None si la metadata es inválida o el texto está vacío.
    """
    try:
        raw_meta = load_metadata(entry.meta_path)
    except (json.JSONDecodeError, OSError):
        return None

    dom = raw_meta.get("domain_metadata")
    page = raw_meta.get("page_metadata")
    if not isinstance(dom, dict) or not isinstance(page, dict):
        return None

    try:
        domain_md = DomainMetadata(**dom)
        page_md = PageMetadata(**page)
    except Exception:
        return None

    text = entry.txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    source_id = page_md.sha256 or f"{page_md.domain_slug or page_md.domain}:{entry.txt_path.stem}"

    return Document(
        source_id=source_id,
        text=text,
        txt_path=str(entry.txt_path),
        domain_metadata=domain_md,
        page_metadata=page_md,
    )
