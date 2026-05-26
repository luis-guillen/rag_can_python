"""Smoke tests de retrieval contra el Qdrant local.

Ejecuta:
    python scripts/smoke_test_retrieval.py

Requiere que la colección `rag_canarias` esté poblada
(`python -m app.embed_index --recreate` previamente).

Cada test imprime OK/FAIL con detalle. Sale con código 0 si todo pasa, 1 si
algún test falla. Tests previstos:

  T1. --domain elmuseocanario.com NUNCA devuelve cultura.grancanaria.com
      (filtro duro, sin fallback).
  T2. --domain canarias-azul.iatext.ulpgc.es NUNCA devuelve otros dominios.
  T3. Auto-detección de "Museo Canario": si hay chunks de elmuseocanario.com,
      los primeros resultados (no fallback) tienen que ser de ese dominio.
  T4. inspect_collection() confirma que el payload tiene la clave 'domain'.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que `app/` es importable cuando se ejecuta desde la raíz.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retrieval import inspect_collection, search  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}OK  {RESET}{msg}")


def _fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")


def _dim(msg: str) -> None:
    print(f"{DIM}    {msg}{RESET}")


failures = 0


def expect(cond: bool, label: str, detail: str = "") -> None:
    global failures
    if cond:
        _ok(label)
        if detail:
            _dim(detail)
    else:
        failures += 1
        _fail(label)
        if detail:
            _dim(detail)


# --- T4 primero: comprobación previa del payload --------------------------

print("\n=== T4. inspect_collection ===")
info = inspect_collection()

if not info.get("exists"):
    print(f"{RED}Colección no existe. Aborto: ejecuta `python -m app.embed_index --recreate` primero.{RESET}")
    sys.exit(2)

expect(
    info.get("sample_domain_field_present") is True,
    "payload tiene la clave 'domain'",
    f"sample_payload_keys={info.get('sample_payload_keys')}",
)
expect(
    bool(info.get("sample_domain_value")),
    "domain de muestra tiene valor no vacío",
    f"sample_domain_value={info.get('sample_domain_value')!r}",
)
print()
print(f"Dominios encontrados ({info.get('domain_counts_sample_size')} puntos muestreados):")
for d, c in info.get("domain_counts_sampled", {}).items():
    print(f"  {c:6d}  {d}")
print()

domain_counts = info.get("domain_counts_sampled", {})

# --- T1. Filtro duro en elmuseocanario.com ---------------------------------

print("=== T1. --domain elmuseocanario.com (filtro duro) ===")
HARD_DOMAIN = "elmuseocanario.com"
if HARD_DOMAIN not in domain_counts:
    print(f"{DIM}    Saltado: no hay chunks de {HARD_DOMAIN} en la colección.{RESET}")
else:
    results_hard = search(
        "Museo Canario",
        top_k=10,
        prefer_domain=HARD_DOMAIN,
        hard_filter=True,
        auto_detect=False,
    )
    bad = [r for r in results_hard if r.domain != HARD_DOMAIN]
    expect(
        len(bad) == 0,
        f"todos los resultados son de {HARD_DOMAIN}",
        f"resultados={len(results_hard)}, dominios={sorted({r.domain for r in results_hard})}",
    )
    expect(
        not any(r.is_fallback for r in results_hard),
        "ningún resultado es fallback en filtro duro",
    )

# --- T2. Filtro duro en canarias-azul.iatext.ulpgc.es ----------------------

print("\n=== T2. --domain canarias-azul.iatext.ulpgc.es (filtro duro) ===")
HARD_DOMAIN2 = "canarias-azul.iatext.ulpgc.es"
if HARD_DOMAIN2 not in domain_counts:
    print(f"{DIM}    Saltado: no hay chunks de {HARD_DOMAIN2} en la colección.{RESET}")
else:
    results_hard2 = search(
        "recursos para procesamiento de lenguaje",
        top_k=10,
        prefer_domain=HARD_DOMAIN2,
        hard_filter=True,
        auto_detect=False,
    )
    bad2 = [r for r in results_hard2 if r.domain != HARD_DOMAIN2]
    expect(
        len(bad2) == 0,
        f"todos los resultados son de {HARD_DOMAIN2}",
        f"resultados={len(results_hard2)}, dominios={sorted({r.domain for r in results_hard2})}",
    )

# --- T3. Auto-detección con fallback marcado ------------------------------

print("\n=== T3. auto-detección «Museo Canario» (filtro suave + fallback marcado) ===")
AUTO_DOMAIN = "elmuseocanario.com"
if AUTO_DOMAIN not in domain_counts:
    print(f"{DIM}    Saltado: no hay chunks de {AUTO_DOMAIN}.{RESET}")
else:
    results_auto = search(
        "Háblame del Museo Canario",
        top_k=10,
        auto_detect=True,
    )
    # Los primeros resultados (los no-fallback) deben ser todos del dominio.
    domain_first = [r for r in results_auto if not r.is_fallback]
    fallback_after = [r for r in results_auto if r.is_fallback]
    bad_in_first = [r for r in domain_first if r.domain != AUTO_DOMAIN]

    expect(
        len(domain_first) >= 1,
        "hay al menos 1 resultado no-fallback",
        f"non_fallback={len(domain_first)}, fallback={len(fallback_after)}",
    )
    expect(
        len(bad_in_first) == 0,
        f"los resultados no-fallback son todos de {AUTO_DOMAIN}",
        f"dominios non-fallback={sorted({r.domain for r in domain_first})}",
    )
    # El orden: todos los no-fallback deben venir antes que cualquier fallback.
    seen_fallback = False
    order_ok = True
    for r in results_auto:
        if r.is_fallback:
            seen_fallback = True
        elif seen_fallback:
            order_ok = False
            break
    expect(order_ok, "los fallback aparecen al final (no entremezclados)")

# --- Resumen ---------------------------------------------------------------

print()
if failures == 0:
    print(f"{GREEN}TODOS LOS TESTS PASAN.{RESET}")
    sys.exit(0)
else:
    print(f"{RED}{failures} test(s) fallaron.{RESET}")
    sys.exit(1)
