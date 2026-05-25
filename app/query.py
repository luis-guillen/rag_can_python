"""Consulta RAG por línea de comandos.

Ejemplo:
    python -m app.query "¿Qué sabes de Memoria de Lanzarote?"
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .retrieval import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el índice RAG.")
    parser.add_argument("question", nargs="+", help="Pregunta a responder.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        print("ERROR: pregunta vacía", file=sys.stderr)
        sys.exit(2)

    results = search(question, args.top_k)

    if args.json:
        print(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False))
        return

    print(f"\nPregunta: {question}")
    print(f"Top-{args.top_k} resultados:\n")
    for i, src in enumerate(results, start=1):
        print(f"[{i}] score={src.score:.3f}  domain={src.domain}")
        print(f"    title: {src.title}")
        print(f"    url:   {src.url}")
        print(f"    text:  {src.text_preview}")
        print()


if __name__ == "__main__":
    main()
