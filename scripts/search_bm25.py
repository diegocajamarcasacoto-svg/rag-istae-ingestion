#!/usr/bin/env python
"""Search the local BM25 index created by scripts/index_bm25.py."""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from typing import Any
from index_bm25 import tokenize

def main() -> int:
    parser = argparse.ArgumentParser(description="Busca fragmentos con BM25 en el índice local.")
    parser.add_argument("query", nargs="?", help="Consulta entre comillas; si se omite, entra en modo interactivo")
    parser.add_argument("--index", default="outputs/bm25_index.pkl", help="Ruta al índice BM25")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados (default: 5)")
    args = parser.parse_args()
    if args.top_k < 1:
        print("--top-k debe ser >= 1", file=sys.stderr); return 2
    index_path = Path(args.index).resolve()
    if not index_path.is_file():
        print(f"No existe el índice: {index_path}. Ejecuta primero scripts/index_bm25.py.", file=sys.stderr); return 2
    # Only load pickle files created locally by the trusted indexer. Never load an untrusted pickle.
    with index_path.open("rb") as f: payload: dict[str, Any] = pickle.load(f)
    bm25, records = payload["bm25"], payload["records"]
    print(f"Índice BM25 cargado: {len(records):,} chunks", file=sys.stderr)
    def search(query: str) -> None:
        terms = tokenize(query)
        if not terms:
            print("La consulta no contiene términos buscables."); return
        scores = bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        ranked = [i for i in ranked if float(scores[i]) > 0][:args.top_k]
        if not ranked:
            print("No se encontraron coincidencias léxicas con puntuación > 0."); return
        for rank, i in enumerate(ranked, start=1):
            r = records[i]
            print(f"\n{'=' * 88}")
            print(f"Resultado #{rank} | BM25 score={float(scores[i]):.4f}")
            print(f"chunk_id: {r.get('chunk_id')}")
            print(f"document_id: {r.get('document_id')}")
            print(f"source_file: {r.get('source_file')}")
            print(f"page_numbers: {r.get('page_numbers')}")
            if r.get("headings"): print(f"headings: {r['headings']}")
            print("\n" + r.get("text", ""))
    if args.query is not None:
        search(args.query); return 0
    print("Modo interactivo BM25. Escribe 'salir' para terminar.")
    while True:
        try: query = input("\nConsulta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if query.lower() in {"salir", "exit", "quit"}: break
        if query: search(query)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
