"""Consulta RAG por línea de comandos.

Ejemplos:
    python -m app.query "¿Qué sabes de Memoria de Lanzarote?"
    python -m app.query "Museo Canario" --domain elmuseocanario.com     # filtro duro
    python -m app.query "Lanzarote" --no-detect                          # sin auto
    python -m app.query --inspect                                        # debug payload
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .retrieval import detect_source, inspect_collection, search


def _print_inspect() -> int:
    info = inspect_collection()
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    if not info.get("exists"):
        print("\nERROR: la colección no existe. Ejecuta `python -m app.embed_index --recreate`.", file=sys.stderr)
        return 2
    if not info.get("sample_domain_field_present"):
        print("\nERROR: el payload NO contiene la clave 'domain'. El filtro Qdrant no puede funcionar.", file=sys.stderr)
        return 3
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el índice RAG.")
    parser.add_argument("question", nargs="*", help="Pregunta a responder.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument(
        "--domain",
        default=None,
        help="FILTRO DURO: devuelve SOLO resultados de este dominio (sin fallback).",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Desactivar la detección automática de fuentes en la pregunta.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Imprimir info de la colección y un payload de ejemplo, y salir.",
    )
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args()

    if args.inspect:
        sys.exit(_print_inspect())

    if not args.question:
        parser.error("question es obligatoria salvo con --inspect")

    question = " ".join(args.question).strip()
    if not question:
        print("ERROR: pregunta vacía", file=sys.stderr)
        sys.exit(2)

    # Resolver dominio y modo (duro vs suave) ANTES de buscar para informar.
    forced_domain = args.domain
    detected_alias = None
    detected_domain = None
    if forced_domain is None and not args.no_detect:
        match = detect_source(question)
        if match is not None:
            detected_alias, detected_domain = match

    prefer_domain = forced_domain or detected_domain
    hard_filter = forced_domain is not None

    results = search(
        question,
        args.top_k,
        prefer_domain=prefer_domain,
        hard_filter=hard_filter,
        auto_detect=False,  # ya resuelto arriba
    )

    domain_count = sum(
        1 for r in results if not r.is_fallback and r.domain == prefer_domain
    )
    fallback_count = sum(1 for r in results if r.is_fallback)

    if args.json:
        out = {
            "question": question,
            "mode": "hard_filter" if hard_filter else ("auto_detect" if detected_domain else "global"),
            "detected_alias": detected_alias,
            "prefer_domain": prefer_domain,
            "results_total": len(results),
            "results_from_domain": domain_count,
            "results_fallback": fallback_count,
            "results": [r.model_dump() for r in results],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"\nPregunta: {question}")
    if hard_filter:
        print(f"Filtro duro aplicado → domain == {forced_domain}")
    elif detected_domain:
        print(f"Fuente detectada: «{detected_alias}» → {detected_domain}")
        print("(filtro suave con fallback global)")
    else:
        print("(sin filtro de dominio)")

    print(f"Resultados: {len(results)} total"
          f" — {domain_count} del dominio preferido"
          f" — {fallback_count} fallback")

    if not results:
        if hard_filter:
            print(f"\nNo hay chunks indexados con domain == {forced_domain}.")
            print("Verifica con: python -m app.query --inspect")
        else:
            print("\nLa colección no devolvió resultados. ¿Está indexada?")
        return

    print()
    for i, src in enumerate(results, start=1):
        tag = " [fallback]" if src.is_fallback else ""
        print(f"[{i}]{tag} score={src.score:.3f}  domain={src.domain}")
        print(f"    title: {src.title}")
        print(f"    url:   {src.url}")
        print(f"    text:  {src.text_preview}")
        print()


if __name__ == "__main__":
    main()
