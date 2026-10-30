import argparse
import json
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    root = path.resolve().parent.parent

    for key in ("chunks_path", "model_local_path", "qdrant_path"):
        p = Path(cfg[key])
        cfg[key] = (root / p).resolve() if not p.is_absolute() else p

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Búsqueda léxica BM25 nativa de Qdrant."
    )
    parser.add_argument(
        "query",
        help="Consulta textual en español",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rag_config.json"),
        help="Ruta al archivo rag_config.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k debe ser mayor o igual a 1.")

    cfg = load_config(args.config.resolve())
    collection = cfg["collection_name"]

    print("Colección:", collection)
    print("Consulta:", args.query)

    t0 = time.perf_counter()
    client = QdrantClient(path=str(cfg["qdrant_path"]))
    t_open = time.perf_counter() - t0

    try:
        if not client.collection_exists(collection):
            raise RuntimeError(
                f"No existe la colección Qdrant: {collection}"
            )

        print("\nEjecutando búsqueda textual nativa...")

        t0 = time.perf_counter()
        result = client.query_points(
            collection_name=collection,
            query=models.TextQuery(text=args.query),
            using="text",
            limit=args.top_k,
            with_payload=True,
        )
        t_search = time.perf_counter() - t0

        if not result.points:
            print("No se encontraron resultados.")
            return

        for rank, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            text = str(payload.get("text") or "")

            print(f"\n{'=' * 70}")
            print(f"Resultado #{rank} | score={point.score:.4f}")
            print("chunk_id:", payload.get("chunk_id"))
            print("document_id:", payload.get("document_id"))
            print("source_file:", payload.get("source_file"))
            print("page_numbers:", payload.get("page_numbers"))
            print("Texto:")
            print(text[:1000])

        print("\n--- MÉTRICAS DE RENDIMIENTO ---")
        print(f"Apertura Qdrant: {t_open:.3f} s")
        print(f"Búsqueda léxica: {t_search:.3f} s")
        print(f"Resultados:      {len(result.points)}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
