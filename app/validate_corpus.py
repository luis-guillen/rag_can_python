"""Valida el corpus generado por ASP.NET sin modificar archivos.

Comprueba que cada .txt tiene .metadata.json, que la metadata contiene
domain_metadata + page_metadata válidos, y reporta:
  - válidos
  - inválidos (metadata corrupta o incompleta)
  - vacíos (.txt sin contenido)
  - duplicados (por sha256 o por url)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import config
from .corpus_utils import iter_corpus_entries, load_metadata


def validate(corpus_dir: Path) -> dict:
    report = {
        "corpus_dir": str(corpus_dir),
        "valid": 0,
        "invalid": [],
        "empty": [],
        "orphan_txt": [],
        "orphan_meta": [],
        "duplicates_by_sha256": [],
        "duplicates_by_url": [],
        "by_domain": defaultdict(int),
    }

    if not corpus_dir.exists():
        print(f"ERROR: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(2)

    # Detectar .txt sin .metadata.json y .metadata.json sin .txt
    all_txt = {p for p in corpus_dir.rglob("*.txt")}
    all_meta = {p for p in corpus_dir.rglob("*.metadata.json")}

    txt_to_meta = {t: t.parent / (t.stem + ".metadata.json") for t in all_txt}
    meta_to_txt = {m: m.parent / (m.name.replace(".metadata.json", ".txt")) for m in all_meta}

    for txt, meta in txt_to_meta.items():
        if meta not in all_meta:
            report["orphan_txt"].append(str(txt))

    for meta, txt in meta_to_txt.items():
        if txt not in all_txt:
            report["orphan_meta"].append(str(meta))

    seen_sha = {}
    seen_url = {}

    for entry in iter_corpus_entries(corpus_dir):
        try:
            raw = load_metadata(entry.meta_path)
        except (json.JSONDecodeError, OSError) as exc:
            report["invalid"].append({"file": str(entry.meta_path), "reason": str(exc)})
            continue

        dom = raw.get("domain_metadata")
        page = raw.get("page_metadata")
        if not isinstance(dom, dict) or not isinstance(page, dict):
            report["invalid"].append(
                {"file": str(entry.meta_path), "reason": "missing domain_metadata or page_metadata"}
            )
            continue

        if not dom.get("domain") or not page.get("url"):
            report["invalid"].append(
                {"file": str(entry.meta_path), "reason": "missing required fields (domain/url)"}
            )
            continue

        text = entry.txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            report["empty"].append(str(entry.txt_path))
            continue

        report["valid"] += 1
        report["by_domain"][dom["domain"]] += 1

        sha = page.get("sha256")
        if sha:
            if sha in seen_sha:
                report["duplicates_by_sha256"].append(
                    {"sha256": sha, "files": [seen_sha[sha], str(entry.txt_path)]}
                )
            else:
                seen_sha[sha] = str(entry.txt_path)

        url = page.get("url")
        if url:
            if url in seen_url:
                report["duplicates_by_url"].append(
                    {"url": url, "files": [seen_url[url], str(entry.txt_path)]}
                )
            else:
                seen_url[url] = str(entry.txt_path)

    report["by_domain"] = dict(report["by_domain"])
    return report


def _print_report(report: dict) -> None:
    print("=" * 60)
    print(f"Corpus: {report['corpus_dir']}")
    print("=" * 60)
    print(f"Documentos válidos:        {report['valid']}")
    print(f"Documentos inválidos:      {len(report['invalid'])}")
    print(f"Documentos vacíos:         {len(report['empty'])}")
    print(f".txt sin metadata:         {len(report['orphan_txt'])}")
    print(f".metadata sin .txt:        {len(report['orphan_meta'])}")
    print(f"Duplicados por sha256:     {len(report['duplicates_by_sha256'])}")
    print(f"Duplicados por url:        {len(report['duplicates_by_url'])}")
    print()
    print("Por dominio:")
    for domain, count in sorted(report["by_domain"].items()):
        print(f"  {domain:40s} {count}")

    if report["invalid"]:
        print("\nPrimeros inválidos:")
        for item in report["invalid"][:5]:
            print(f"  - {item['file']}: {item['reason']}")
    if report["empty"]:
        print("\nPrimeros vacíos:")
        for f in report["empty"][:5]:
            print(f"  - {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el corpus RAG generado por ASP.NET.")
    parser.add_argument(
        "--corpus",
        default=str(config.CORPUS_DIR),
        help=f"Ruta al directorio del corpus (default: {config.CORPUS_DIR}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprimir el reporte completo como JSON.",
    )
    args = parser.parse_args()

    report = validate(Path(args.corpus))

    if args.json:
        # default=str para Path en duplicados
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_report(report)

    if report["invalid"] or report["orphan_txt"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
