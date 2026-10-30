#!/usr/bin/env python
"""Genera embeddings locales E5 Base e indexa chunks.jsonl en Qdrant local."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

if __package__:
    from .utils import read_chunks_generic
else:
    from utils import read_chunks_generic


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    root = path.resolve().parent.parent
    for key in ("chunks_path", "model_local_path", "qdrant_path"):
        p = Path(cfg[key])
        cfg[key] = (root / p).resolve() if not p.is_absolute() else p
    return cfg


def stable_point_id(chunk_id: str) -> str:
    # Qdrant requiere IDs UUID o enteros; el UUID5 permite indexaciones reproducibles.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "rag-istae:" + chunk_id))


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rag_config.json", help="Ruta a la configuración JSON")
    parser.add_argument("--recreate", action="store_true", help="Borra y recrea la colección; usar con cuidado")
    parser.add_argument("--offline", action="store_true", help="Exige que los pesos existan localmente; no intenta descargar")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    chunks_path: Path = cfg["chunks_path"]
    model_path: Path = cfg["model_local_path"]
    qdrant_path: Path = cfg["qdrant_path"]
    collection = cfg["collection_name"]
    batch_size = int(cfg.get("batch_size", 8))

    print(f"[1/5] Leyendo fragmentos: {chunks_path}")
    # Leyendo fragmentos 
    chunks = read_chunks_generic(chunks_path)
    print(f"      Fragmentos válidos: {len(chunks):,}")

    # Identificador del corpus para dejar trazabilidad básica.
    sha = hashlib.sha256()
    with chunks_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    corpus_sha256 = sha.hexdigest()
    print(f"      SHA-256 chunks.jsonl: {corpus_sha256}")

    print("[2/5] Cargando modelo de embeddings...")
    if model_path.exists() and any(model_path.iterdir()):
        model_source = str(model_path)
        print(f"      Usando modelo local: {model_source}")
    elif args.offline:
        raise FileNotFoundError(
            f"Modo offline solicitado, pero no se encontró el modelo en {model_path}. "
            "Descárgalo primero o configura model_local_path."
        )
    else:
        model_source = cfg["model_name"]
        print(f"      No se encontró copia local; se usará {model_source}.")
        print("      Si es la primera ejecución, SentenceTransformers descargará los pesos.")

    model = SentenceTransformer(model_source, device="cpu")
    model.max_seq_length = int(cfg.get("max_seq_length", 512))
    dim = model.get_sentence_embedding_dimension()
    if not dim:
        raise RuntimeError("No se pudo determinar la dimensión del modelo.")
    print(f"      Dimensiones: {dim}; dispositivo: CPU; longitud máxima: {model.max_seq_length}")

    client = QdrantClient(path=str(qdrant_path))
    try:
        exists = client.collection_exists(collection)
        if exists and (args.recreate or cfg.get("recreate_collection", False)):
            print(f"[3/5] Eliminando colección existente: {collection}")
            client.delete_collection(collection)
            exists = False
        elif exists:
            info = client.get_collection(collection)
            existing_count = info.points_count or 0
            print(f"[3/5] Colección existente: {collection} ({existing_count:,} puntos)")
            if existing_count:
                print("      Para reconstruir desde cero, ejecuta con --recreate.")
                print("      No se sobrescribe automáticamente una colección existente.")
                return 2

        if not exists:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )

        print("[4/5] Generando embeddings e indexando por lotes...")
        started = time.perf_counter()
        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            texts = ["passage: " + item["text"].strip() for item in batch]
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(cfg.get("normalize_embeddings", True))
            )
            points = []
            for item, vector in zip(batch, vectors):
                payload = {k: json_safe(v) for k, v in item.items()}
                payload["_index_metadata"] = {
                    "embedding_model": cfg["model_name"],
                    "embedding_dimensions": int(dim),
                    "corpus_sha256": corpus_sha256
                }
                points.append(models.PointStruct(
                    id=stable_point_id(str(item["chunk_id"])),
                    vector=vector.tolist(),
                    payload=payload
                ))
            client.upsert(collection_name=collection, points=points, wait=True)
            done = min(start + len(batch), total)
            if done == total or done % 200 < batch_size:
                print(f"      Indexados: {done:,}/{total:,}")
        elapsed = time.perf_counter() - started
        final_info = client.get_collection(collection)
        final_count = final_info.points_count or 0
        print("[5/5] Validación")
        print(f"      Esperados: {total:,}")
        print(f"      En Qdrant: {final_count:,}")
        print(f"      Tiempo indexación (sin cargar modelo): {elapsed:.1f} s ({elapsed/60:.1f} min)")
        print(f"      Colección local: {qdrant_path / collection}")
        if final_count != total:
            print("ERROR: el número de puntos no coincide con los fragmentos.")
            return 3
        print("OK: indexación y conteo validados.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
