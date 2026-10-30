#!/usr/bin/env python
"""Consulta híbrida: E5 + BM25 nativo de Qdrant + RRF."""

import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models
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

    parser = argparse.ArgumentParser(
        description="Búsqueda híbrida E5 + BM25 + RRF en Qdrant."
    )
    parser.add_argument("query", help="Pregunta en español")
    parser.add_argument("--config", default="config/rag_config.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--prefetch-k",
        type=int,
        default=20,
        help="Candidatos recuperados por cada buscador antes de RRF.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Constante k de RRF.",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1 or args.prefetch_k < 1:
        parser.error("--top-k y --prefetch-k deben ser mayores que cero.")
    if args.rrf_k < 1:
        parser.error("--rrf-k debe ser mayor que cero.")

    cfg = load_config(Path(args.config).resolve())

    # 1. Cargar modelo E5
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

    # 2. Generar embedding de consulta
    t0 = time.perf_counter()
    query_text = args.query.strip()

    query_vector = model.encode(
        "query: " + query_text,
        convert_to_numpy=True,
        normalize_embeddings=bool(
            cfg.get("normalize_embeddings", True)
        ),
    ).tolist()

    t_embedding = time.perf_counter() - t0

    # 3. Abrir Qdrant local
    t0 = time.perf_counter()
    client = QdrantClient(path=str(cfg["qdrant_path"]))
    t_qdrant_open = time.perf_counter() - t0

    try:
        collection = cfg["collection_name"]

        if not client.collection_exists(collection):
            raise RuntimeError(
                f"No existe la colección Qdrant: {collection}"
            )

        # 4. Recuperación híbrida y fusión RRF nativa
        t0 = time.perf_counter()

        results = client.query_points(
            collection_name=collection,
            prefetch=[
                # Recuperación semántica: vector E5.
                models.Prefetch(
                    query=query_vector,
                    limit=args.prefetch_k,
                ),
                # Recuperación léxica: BM25 sobre payload.text.
                models.Prefetch(
                    query=models.TextQuery(text=query_text),
                    using="text",
                    limit=args.prefetch_k,
                ),
            ],
            query=models.RrfQuery(
                rrf=models.Rrf(k=args.rrf_k)
            ),
            limit=args.top_k,
            with_payload=True,
        ).points

        t_hybrid_search = time.perf_counter() - t0

        # 5. Mostrar resultados fusionados
        print("\n" + "=" * 75)
        print("BÚSQUEDA HÍBRIDA: E5 + BM25 NATIVO + RRF")
        print("=" * 75)
        print(f"Consulta:       {query_text}")
        print(f"Colección:      {collection}")
        print(f"Prefetch por recuperador: {args.prefetch_k}")
        print(f"RRF k:          {args.rrf_k}")
        print(f"Resultados:     {len(results)}")

        for i, point in enumerate(results, start=1):
            p = point.payload or {}

            print(f"\n{'=' * 75}")
            print(f"Resultado #{i} | RRF score={point.score:.6f}")
            print(f"chunk_id:     {p.get('chunk_id')}")
            print(f"document_id:  {p.get('document_id')}")
            print(f"source_file:  {p.get('source_file')}")
            print(f"page_numbers: {p.get('page_numbers')}")
            print("-" * 75)
            print((p.get("text") or "")[:1000])

    finally:
        client.close()

    t_total = time.perf_counter() - total_start

    print("\n--- MÉTRICAS DE RENDIMIENTO ---")
    print(f"Carga modelo:       {t_model:.3f} s")
    print(f"Embedding consulta: {t_embedding:.3f} s")
    print(f"Apertura Qdrant:    {t_qdrant_open:.3f} s")
    print(f"Búsqueda híbrida:   {t_hybrid_search:.3f} s")
    print(f"Total:              {t_total:.3f} s")
    print(f"Resultados:         {len(results)}")


if __name__ == "__main__":
    main()
