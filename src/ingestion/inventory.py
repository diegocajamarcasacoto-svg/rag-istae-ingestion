from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import safe_relative, sha256_file

SUPPORTED_INVENTORY = {".pdf", ".docx", ".xlsx", ".xls", ".doc"}


def build_inventory(
    corpus_root: Path,
    excluded_dirs: set[str] | None = None,
    extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not corpus_root.exists() or not corpus_root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del corpus: {corpus_root}")

    excluded_dirs = excluded_dirs or set()
    extensions = extensions or SUPPORTED_INVENTORY
    rows: list[dict[str, Any]] = []

    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(corpus_root).parts
        if any(part.lower() in {d.lower() for d in excluded_dirs} for part in relative_parts[:-1]):
            continue
        if path.suffix.lower() not in extensions:
            continue

        stat = path.stat()
        rows.append({
            "relative_path": safe_relative(path, corpus_root),
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_time_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "sha256": sha256_file(path),
        })
    return rows
