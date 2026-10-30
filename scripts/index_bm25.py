#!/usr/bin/env python
"""Build a local BM25 index from outputs/chunks.jsonl without modifying source chunks."""
from __future__ import annotations
import argparse, pickle, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .utils import read_chunks_generic
else:
    from utils import read_chunks_generic

# Expresión regular (regex) para identificar las palabras de un texto antes de pasarlas a BM25.
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)

def tokenize(text: str) -> list[str]:
    # Convertir a minúsculas y tokenizar; conservar acentos, números, referencias legales y negaciones. 
    # De forma deliberada, NO eliminar palabras vacías (stopwords), ni aplicar derivación (stemming), lematización o eliminación de signos de puntuación dentro de los términos.
    return TOKEN_RE.findall(text.lower())

def main() -> int:
    parser = argparse.ArgumentParser(description="Construye un índice léxico BM25 desde chunks.jsonl.")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl", help="Ruta a chunks.jsonl")
    parser.add_argument("--index", default="outputs/bm25_index.pkl", help="Archivo de índice BM25 de salida")
    parser.add_argument("--k1", type=float, default=1.5, help="Parámetro BM25 k1 (default: 1.5)")
    parser.add_argument("--b", type=float, default=0.75, help="Parámetro BM25 b (default: 0.75)")
    args = parser.parse_args()
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Falta rank-bm25. Instálalo con: python -m pip install rank-bm25", file=sys.stderr)
        return 2
    chunks_path, index_path = Path(args.chunks).resolve(), Path(args.index).resolve()
    if not chunks_path.is_file():
        print(f"No existe el archivo de chunks: {chunks_path}", file=sys.stderr); return 2
    if args.k1 <= 0 or not 0 <= args.b <= 1:
        print("Parámetros inválidos: k1 debe ser > 0 y b debe estar entre 0 y 1.", file=sys.stderr); return 2
    # Leer chunks y construir el índice BM25
    print(f"Leyendo chunks: {chunks_path}")
    chunks = read_chunks_generic(chunks_path)
    records = [
        {
            "chunk_id": chunk["chunk_id"], "document_id": chunk.get("document_id"),
            "source_file": chunk.get("source_file"), "chunk_index": chunk.get("chunk_index"),
            "headings": chunk.get("headings", []), "page_numbers": chunk.get("page_numbers", []),
            "item_types": chunk.get("item_types", []), "text": chunk["text"],
            "char_count": chunk.get("char_count", len(chunk["text"])),
            "chunking_method": chunk.get("chunking_method"), "source_sha256": chunk.get("source_sha256"),
        }
        for chunk in chunks
    ]
    tokenized = [tokenize(r["text"]) for r in records]
    print(f"Chunks leídos: {len(records):,}")
    print(f"Tokens totales: {sum(map(len, tokenized)):,}")
    print(f"Chunks sin tokens: {sum(not t for t in tokenized)}")
    print("Construyendo índice BM25...")
    bm25 = BM25Okapi(tokenized, k1=args.k1, b=args.b)
    payload = {
        "format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_chunks_path": str(chunks_path), "source_chunks_count": len(records),
        "tokenizer": "unicode_words_lowercase_preserve_accents_no_stopword_removal",
        "k1": args.k1, "b": args.b, "records": records,
        "tokenized_corpus": tokenized, "bm25": bm25,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Índice guardado: {index_path}")
    print(f"Chunks indexados: {len(records):,}")
    print("Texto original conservado; no se eliminaron stopwords ni se aplicó stemming/lematización.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
