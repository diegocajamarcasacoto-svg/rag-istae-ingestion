#!/usr/bin/env python
"""Consulta semántica con medición de tiempos por etapa."""

import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    for key in ("chunks_path", "model_local_path", "qdrant_path"):
        p = Path(cfg[key])
        cfg[key] = (root / p).resolve() if not p.is_absolute() else p

    return cfg


def main():
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Pregunta en español")
    parser.add_argument("--config", default="config/rag_config.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())

    # 1. Carga del modelo
    t0 = time.perf_counter()
    local_path = cfg["model_local_path"]

    if local_path.exists() and any(local_path.iterdir()):
        model_source = str(local_path)
    elif args.offline:
        raise FileNotFoundError(
            f"No existe modelo local en {local_path}"
        )
    else:
        model_source = cfg["model_name"]

    model = SentenceTransformer(model_source, device="cpu")
    model.max_seq_length = int(cfg.get("max_seq_length", 512))
    t_model = time.perf_counter() - t0

    # 2. Embedding de la pregunta
    t0 = time.perf_counter()
    query_vector = model.encode(
        "query: " + args.query.strip(),
        convert_to_numpy=True,
        normalize_embeddings=bool(
            cfg.get("normalize_embeddings", True)
        ),
    ).tolist()
    t_embedding = time.perf_counter() - t0

    # 3. Apertura de Qdrant local
    t0 = time.perf_counter()
    client = QdrantClient(path=str(cfg["qdrant_path"]))
    t_qdrant_open = time.perf_counter() - t0

    try:
        if not client.collection_exists(cfg["collection_name"]):
            raise RuntimeError("No existe la colección Qdrant.")

        # 4. Búsqueda vectorial
        t0 = time.perf_counter()
        results = client.query_points(
            collection_name=cfg["collection_name"],
            query=query_vector,
            limit=max(1, args.top_k),
            with_payload=True,
        ).points
        t_search = time.perf_counter() - t0

        for i, point in enumerate(results, start=1):
            p = point.payload or {}
            print(f"\n{'=' * 70}")
            print(f"Resultado #{i} | score={point.score:.4f}")
            print(f"chunk_id: {p.get('chunk_id')}")
            print(f"document_id: {p.get('document_id')}")
            print(f"source_file: {p.get('source_file')}")
            print(f"page_numbers: {p.get('page_numbers')}")
            print((p.get("text") or "")[:1000])

    finally:
        client.close()

    t_total = time.perf_counter() - total_start

    print("\n--- MÉTRICAS DE RENDIMIENTO ---")
    print(f"Carga modelo:       {t_model:.3f} s")
    print(f"Embedding consulta: {t_embedding:.3f} s")
    print(f"Apertura Qdrant:    {t_qdrant_open:.3f} s")
    print(f"Búsqueda Qdrant:    {t_search:.3f} s")
    print(f"Total:              {t_total:.3f} s")
    print(f"Resultados:         {len(results)}")


if __name__ == "__main__":
    main()
