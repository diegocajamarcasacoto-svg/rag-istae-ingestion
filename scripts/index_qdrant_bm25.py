
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
    parser = argparse.ArgumentParser(
        description="Prepara el índice textual BM25 de Qdrant."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rag_config.json"),
        help="Ruta al archivo rag_config.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config.resolve())
    collection = cfg["collection_name"]

    print("Colección:", collection)
    print("Qdrant:", cfg["qdrant_path"])

    client = QdrantClient(path=str(cfg["qdrant_path"]))

    try:
        if not client.collection_exists(collection):
            raise RuntimeError(
                f"No existe la colección Qdrant: {collection}"
            )

        # Verificar que el payload contiene texto.
        points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=["text"],
            with_vectors=False,
        )

        if not points or not (points[0].payload or {}).get("text"):
            raise RuntimeError(
                "No se encontró un campo text con contenido "
                "en el payload."
            )

        # Comprobar el índice mediante una operación real.
        # No dependemos de get_collection().payload_schema,
        # porque puede no reflejar los índices como esperamos.
        try:
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
            print("\nÍndice textual creado correctamente.")

        except Exception as exc:
            # No asumimos que cualquier error significa
            # que el índice ya existe.
            print("\nNo se pudo crear el índice textual.")
            print("Error:", exc)
            print(
                "Verifica si ya existe un índice compatible "
                "antes de volver a intentarlo."
            )
            raise

        print("\nPreparación del índice BM25 finalizada.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
