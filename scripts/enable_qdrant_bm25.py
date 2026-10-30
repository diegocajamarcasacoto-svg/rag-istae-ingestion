
import argparse
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Ruta al archivo rag_config.json",
    )
    parser.add_argument(
        "--query",
        default="Art. 357 Constitución de la República",
        help="Consulta de prueba",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config.resolve())
    collection = cfg["collection_name"]
    qdrant_path = str(cfg["qdrant_path"])

    print("Colección:", collection)
    print("Consulta:", args.query)

    client = QdrantClient(path=qdrant_path)

    try:
        # Verificar que el campo text existe en un punto.
        points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=["text"],
            with_vectors=False,
        )

        if not points or not (points[0].payload or {}).get("text"):
            raise RuntimeError(
                "No se encontró un campo text con contenido en el payload."
            )

        # Crear índice textual con BM25.
        # Si ya existe, no hace falta volver a crearlo.
        schema = client.get_collection(collection).payload_schema or {}

        if "text" not in schema:
            print("\nCreando índice de texto BM25...")
            client.create_payload_index(
                collection_name=collection,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                    scoring=True,
                ),
                wait=True,
            )
            print("Índice creado.")
        else:
            print("\nYa existe un índice para text:")
            print(schema["text"])
            print(
                "Si no es un índice textual con scoring BM25, "
                "detente y revisa el esquema antes de continuar."
            )
            if not isinstance(
                schema["text"], models.TextIndexParams
            ):
                print(
                    "El esquema existente no se reconoce como "
                    "TextIndexParams; no se ejecutará la búsqueda."
                )
                return

        # Probar la búsqueda léxica de Qdrant.
        print("\nBuscando con BM25 nativo de Qdrant...")
        result = client.query_points(
            collection_name=collection,
            query=models.TextQuery(text=args.query),
            using="text",
            limit=args.top_k,
            with_payload=True,
        )

        if not result.points:
            print("No se encontraron resultados.")
            return

        for rank, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            print(f"\n#{rank} | score={point.score:.4f}")
            print("chunk_id:", payload.get("chunk_id"))
            print("source_file:", payload.get("source_file"))
            print("page_numbers:", payload.get("page_numbers"))
            print("text:", str(payload.get("text", ""))[:500])

        print("\nPrueba de búsqueda textual finalizada.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
