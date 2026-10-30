
#!/usr/bin/env python
"""
Inspecciona una colección Qdrant local sin modificarla.

Funciones:
- Muestra información general de la colección.
- Inspecciona los campos del payload.
- Busca un fragmento por chunk_id.
- Recupera fragmentos vecinos del mismo documento,
  utilizando document_id y chunk_index.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


def load_config(path: Path) -> dict[str, Any]:
    """Carga la configuración y resuelve las rutas relativas."""

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    root = path.resolve().parent.parent

    for key in ("chunks_path", "model_local_path", "qdrant_path"):
        if key in cfg:
            p = Path(cfg[key])
            cfg[key] = (
                (root / p).resolve()
                if not p.is_absolute()
                else p
            )

    return cfg


def inspect_neighbors(
    client: QdrantClient,
    collection: str,
    chunk_id: str,
    neighbors: int = 1,
) -> None:
    """
    Busca un fragmento y recupera sus vecinos anteriores
    y posteriores dentro del mismo documento.
    """

    print("\n" + "=" * 60)
    print("INSPECCIÓN DE FRAGMENTO Y VECINOS")
    print("=" * 60)

    # 1. Buscar el fragmento seleccionado.
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchValue(value=chunk_id),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print(f"No se encontró el fragmento: {chunk_id}")
        return

    selected = points[0]
    payload = selected.payload or {}

    document_id = payload.get("document_id")
    chunk_index = payload.get("chunk_index")

    print(f"Chunk solicitado: {chunk_id}")
    print(f"Documento:        {document_id}")
    print(f"Índice:           {chunk_index}")

    # 2. Validar los metadatos necesarios.
    if document_id is None:
        print("ERROR: el payload no contiene document_id.")
        return

    if chunk_index is None:
        print("ERROR: el payload no contiene chunk_index.")
        return

    if isinstance(chunk_index, bool) or not isinstance(
        chunk_index, int
    ):
        print(
            "ERROR: chunk_index debe ser un entero. "
            f"Valor encontrado: {chunk_index!r}"
        )
        return

    # 3. Calcular el rango de vecinos.
    lower = max(1, chunk_index - neighbors)
    upper = chunk_index + neighbors

    print(f"Rango solicitado: {lower} - {upper}")

    # 4. Recuperar fragmentos del mismo documento.
    related, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="chunk_index",
                    range=models.Range(
                        gte=lower,
                        lte=upper,
                    ),
                ),
            ]
        ),
        limit=(neighbors * 2) + 1,
        with_payload=True,
        with_vectors=False,
    )

    # 5. Ordenar los fragmentos por posición documental.
    related.sort(
        key=lambda point: (
            (point.payload or {}).get("chunk_index", 0)
        )
    )

    print(f"\nFragmentos recuperados: {len(related)}")

    for point in related:
        p = point.payload or {}

        current_chunk_id = p.get("chunk_id")
        current_index = p.get("chunk_index")

        marker = (
            " [SELECCIONADO]"
            if current_chunk_id == chunk_id
            else " [VECINO]"
        )

        print("\n" + "-" * 75)
        print(
            f"{current_chunk_id} | "
            f"índice={current_index}{marker}"
        )
        print(f"Documento: {p.get('document_id')}")
        print(f"Archivo:   {p.get('source_file')}")
        print(f"Sección:   {p.get('headings')}")
        print("-" * 75)
        print(p.get("text") or "")

    print("\nInspección de vecinos terminada.")


def inspect_collection(
    client: QdrantClient,
    collection: str,
) -> None:
    """Muestra información general de la colección."""

    info = client.get_collection(collection)

    print("\n--- Información de la colección ---")
    print(f"Puntos registrados: {info.points_count}")
    print(f"Estado:             {info.status}")
    print(f"Vectores:           {info.config.params.vectors}")

    # Índices de payload.
    print("\n--- Índices de payload ---")

    try:
        payload_schema = info.payload_schema or {}

        if payload_schema:
            for field, schema in payload_schema.items():
                print(f"- {field}: {schema}")
        else:
            print("No se reportan índices de payload.")

    except Exception as exc:
        print(f"No se pudo consultar el esquema: {exc}")

    # Inspeccionar un punto de ejemplo.
    print("\n--- Inspección de un fragmento de ejemplo ---")

    points, _ = client.scroll(
        collection_name=collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print("La colección no contiene puntos.")
        return

    point = points[0]
    payload = point.payload or {}

    print(f"ID interno: {point.id}")

    print("\nCampos del payload:")

    for key, value in payload.items():
        if isinstance(value, str):
            preview = value[:180].replace("\n", " ")
            print(f"- {key}: {preview!r}")
        else:
            print(f"- {key}: {type(value).__name__} = {value}")

    # Verificar campos necesarios.
    print("\n--- Campos relevantes para recuperación ---")

    required_fields = (
        "chunk_id",
        "document_id",
        "chunk_index",
        "text",
        "source_file",
        "headings",
    )

    for field in required_fields:
        exists = field in payload
        value = payload.get(field)

        print(
            f"{field}: "
            f"{'SÍ' if exists else 'NO'}"
            f" | valor={value if exists else 'N/A'}"
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona una colección Qdrant local "
            "y permite recuperar fragmentos vecinos."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Ruta al archivo JSON de configuración.",
    )

    parser.add_argument(
        "--chunk-id",
        type=str,
        help="Identificador del fragmento que se desea inspeccionar.",
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=1,
        help="Cantidad de vecinos anteriores y posteriores.",
    )

    args = parser.parse_args()

    if args.neighbors < 0:
        parser.error("--neighbors no puede ser negativo.")

    cfg_path = args.config.resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {cfg_path}"
        )

    cfg = load_config(cfg_path)

    required = ("qdrant_path", "collection_name")
    missing = [key for key in required if key not in cfg]

    if missing:
        raise KeyError(
            f"Faltan estas claves en la configuración: {missing}"
        )

    qdrant_path = Path(cfg["qdrant_path"])
    collection = cfg["collection_name"]

    print("=" * 60)
    print("INSPECCIÓN DE QDRANT")
    print("=" * 60)
    print(f"Configuración: {cfg_path}")
    print(f"Ruta Qdrant:   {qdrant_path}")
    print(f"Colección:     {collection}")

    if not qdrant_path.exists():
        raise FileNotFoundError(
            f"No existe la ruta local de Qdrant: {qdrant_path}"
        )

    client = QdrantClient(path=str(qdrant_path))

    try:
        if not client.collection_exists(collection):
            raise RuntimeError(
                f"No existe la colección '{collection}' "
                "en esa ruta."
            )

        # Inspección general.
        inspect_collection(client, collection)

        # Inspección opcional de vecinos.
        if args.chunk_id:
            inspect_neighbors(
                client=client,
                collection=collection,
                chunk_id=args.chunk_id,
                neighbors=args.neighbors,
            )

        print("\nInspección terminada.")
        print("No se modificaron los puntos.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
