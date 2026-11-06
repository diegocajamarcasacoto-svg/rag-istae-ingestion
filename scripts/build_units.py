"""
Genera units.jsonl (índice chunk -> artículo) a partir del chunks.jsonl existente.

Aplica la misma etapa que run_pipeline.py sin volver a extraer ni fragmentar:
sirve para obtener las unidades de una ingesta ya validada. No modifica
chunks.jsonl, run_manifest.json ni Qdrant.

Uso:
    python .\\scripts\\build_units.py --config .\\config\\config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.pipeline import load_config, resolve_paths, write_units  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; ejecute antes run_pipeline.py")
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera units.jsonl desde chunks.jsonl")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.json"))
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    _, output_dir = resolve_paths(load_config(config_path), config_path)
    chunks = read_jsonl(output_dir / "chunks.jsonl")
    report = write_units(output_dir, chunks)
    validation = report["validation"]

    print(f"Unidades: {report['units_count']} {report['units_by_kind']}")
    print(f"Documentos por modo: {report['documents_by_mode']}")
    print(f"Cobertura: {validation['chunks_covered']}/{validation['chunks_count']} fragmentos")
    print(f"Salidas: {output_dir / 'units.jsonl'} | {output_dir / 'units_report.json'}")
    if not validation["passed"]:
        print("ERRORES: " + "; ".join(validation["errors"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
