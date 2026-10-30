from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_chunks_generic(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {path}")

    chunks: list[dict[str, Any]] = []
    seen: set[Any] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en línea {line_no}: {exc}") from exc

            chunk_id = item.get("chunk_id")
            text = item.get("text")
            if not chunk_id:
                raise ValueError(f"Falta chunk_id en línea {line_no}")
            if chunk_id in seen:
                raise ValueError(f"chunk_id duplicado: {chunk_id!r} (línea {line_no})")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Texto vacío o inválido para chunk_id={chunk_id!r}")

            seen.add(chunk_id)
            chunks.append(item)

    if not chunks:
        raise ValueError(f"No se encontraron chunks válidos en {path}")
    return chunks
