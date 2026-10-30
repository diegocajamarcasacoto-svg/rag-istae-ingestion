#!/usr/bin/env python
"""Consulta híbrida E5 + BM25 + RRF con reranking BGE."""

import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer


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
        description="Búsqueda híbrida E5 + BM25 + RRF + BGE Reranker."
    )
    parser.add_argument("query", help="Pregunta en español")
    parser.add_argument("--config", default="config/rag_config.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Número de candidatos RRF que serán evaluados por el reranker.",
    )
    parser.add_argument(
        "--prefetch-k",
        type=int,
        default=20,
        help="Candidatos recuperados por cada buscador antes de RRF.",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--reranker-model",
        default="models/bge-reranker-v2-m3",
        help="Ruta local del modelo BGE Reranker.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Pares consulta-fragmento procesados por lote.",
    )
    args = parser.parse_args()

    if min(args.top_k, args.candidate_k, args.prefetch_k, args.batch_size) < 1:
        parser.error("top-k, candidate-k, prefetch-k y batch-size deben ser positivos.")
    if args.top_k > args.candidate_k:
        parser.error("--top-k no puede ser mayor que --candidate-k.")
    if args.rrf_k < 1:
        parser.error("--rrf-k debe ser mayor que cero.")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    project_root = config_path.parent.parent

    # 1. Resolver rutas locales de los modelos.
    reranker_path = Path(args.reranker_model)
    if not reranker_path.is_absolute():
        reranker_path = (project_root / reranker_path).resolve()

    if not reranker_path.exists() or not any(reranker_path.iterdir()):
        raise FileNotFoundError(
            f"No se encuentra el modelo reranker en: {reranker_path}\n"
            "Descárgalo con:\n"
            "hf download BAAI/bge-reranker-v2-m3 "
            "--local-dir .\\models\\bge-reranker-v2-m3"
        )

    # 2. Cargar modelo E5 en CPU.
    t0 = time.perf_counter()
    local_path = cfg["model_local_path"]

    if local_path.exists() and any(local_path.iterdir()):
        model_source = str(local_path)
    else:
        raise FileNotFoundError(
            f"No existe el modelo E5 local en: {local_path}"
        )

    model = SentenceTransformer(model_source, device="cpu")
    model.max_seq_length = int(cfg.get("max_seq_length", 512))
    t_e5_load = time.perf_counter() - t0

    # 3. Cargar BGE Reranker en CPU.
    t0 = time.perf_counter()
    reranker = CrossEncoder(
        str(reranker_path),
        device="cpu",
        max_length=512,
    )
    t_reranker_load = time.perf_counter() - t0

    # 4. Generar embedding de consulta.
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

    # 5. Abrir Qdrant local.
    t0 = time.perf_counter()
    client = QdrantClient(path=str(cfg["qdrant_path"]))
    t_qdrant_open = time.perf_counter() - t0

    results = []

    try:
        collection = cfg["collection_name"]

        if not client.collection_exists(collection):
            raise RuntimeError(
                f"No existe la colección Qdrant: {collection}"
            )

        # 6. Recuperación híbrida y fusión RRF.
        t0 = time.perf_counter()

        candidates = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    limit=args.prefetch_k,
                ),
                models.Prefetch(
                    query=models.TextQuery(text=query_text),
                    using="text",
                    limit=args.prefetch_k,
                ),
            ],
            query=models.RrfQuery(
                rrf=models.Rrf(k=args.rrf_k)
            ),
            limit=args.candidate_k,
            with_payload=True,
        ).points

        t_hybrid_search = time.perf_counter() - t0

        if not candidates:
            print("La búsqueda híbrida no devolvió candidatos.")
            return
        
        # Diagnóstico: mostrar los 20 candidatos originales de RRF.
        print("\n--- DIAGNÓSTICO: CANDIDATOS RRF ---")

        for i, point in enumerate(candidates, start=1):
            payload = point.payload or {}

            print(
                f"RRF #{i} | "
                f"chunk_id={payload.get('chunk_id')} | "
                f"score={point.score:.6f}"
            )

        # 7. Preparar pares consulta-fragmento.
        pairs = []
        valid_candidates = []

        for point in candidates:
            payload = point.payload or {}
            text = payload.get("text", "")

            if text and text.strip():
                pairs.append([query_text, text])
                valid_candidates.append(point)

        if not pairs:
            print("Los candidatos recuperados no contienen texto.")
            return

        # 8. Reranking con BGE en CPU.
        t0 = time.perf_counter()

        reranker_scores = reranker.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        t_reranking = time.perf_counter() - t0

        # 9. Asociar puntuaciones y ordenar.
        ranked_results = []

        for point, score in zip(valid_candidates, reranker_scores):
            ranked_results.append({
                "point": point,
                "reranker_score": float(score),
            })

        ranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        results = ranked_results[:args.top_k]

        # 10. Mostrar resultados.
        print("\n" + "=" * 80)
        print("BÚSQUEDA HÍBRIDA + RERANKING BGE")
        print("=" * 80)
        print(f"Consulta:             {query_text}")
        print(f"Colección:            {collection}")
        print(f"Modelo reranker:      {reranker_path}")
        print("Dispositivo:          CPU")
        print(f"Prefetch por motor:   {args.prefetch_k}")
        print(f"Candidatos RRF:       {len(candidates)}")
        print(f"Pares evaluados:      {len(pairs)}")
        print(f"Resultados finales:   {len(results)}")
        print(f"RRF k:                {args.rrf_k}")

        for i, item in enumerate(results, start=1):
            point = item["point"]
            p = point.payload or {}

            print(f"\n{'=' * 80}")
            print(f"Resultado #{i}")
            print(f"Reranker score: {item['reranker_score']:.6f}")
            print(f"RRF score:      {point.score:.6f}")
            print(f"chunk_id:       {p.get('chunk_id')}")
            print(f"document_id:    {p.get('document_id')}")
            print(f"source_file:    {p.get('source_file')}")
            print(f"page_numbers:   {p.get('page_numbers')}")
            print("-" * 80)
            print((p.get("text") or "")[:1000])

    finally:
        client.close()

    t_total = time.perf_counter() - total_start

    print("\n--- MÉTRICAS DE RENDIMIENTO ---")
    print(f"Carga modelo E5:      {t_e5_load:.3f} s")
    print(f"Carga reranker:       {t_reranker_load:.3f} s")
    print(f"Embedding consulta:   {t_embedding:.3f} s")
    print(f"Apertura Qdrant:      {t_qdrant_open:.3f} s")
    print(f"Búsqueda híbrida:     {t_hybrid_search:.3f} s")
    print(f"Reranking:            {t_reranking:.3f} s")
    print(f"Total:                {t_total:.3f} s")
    print(f"Candidatos evaluados: {len(pairs)}")
    print(f"Resultados finales:   {len(results)}")


if __name__ == "__main__":
    main()
