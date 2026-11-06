"""
Compara los unit_id de units.jsonl con los que produce rag-istae-api.

Herramienta de validación de solo lectura: ejecuta el parser de la API
(app/normativa/cleaning.py + structure.py) sobre su propia carpeta data/ en un
proceso aparte, sin escribir bytecode ni ningún archivo en ese repositorio.
Los repositorios no se importan entre sí: la ruta de la API se pasa como
argumento y solo se usa aquí.

Uso:
    python .\\scripts\\compare_units_with_api.py --config .\\config\\config.json --api-root E:\\...\\rag-istae-api
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.pipeline import load_config, resolve_paths  # noqa: E402

_API_PARSER = r"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from app.normativa.cleaning import clean_markdown
from app.normativa.structure import parse_document
data = root / "data"
result = {}
for line in (data / "document_mapping.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    text = (data / "extracted_text" / f"{row['source_sha256']}.txt").read_text(encoding="utf-8")
    parsed = parse_document(row["document_id"], clean_markdown(text))
    result[row["document_id"]] = {"mode": parsed.mode, "units": [u.unit_id for u in parsed.units]}
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
"""


def api_units(api_root: Path) -> dict[str, dict]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _API_PARSER, str(api_root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
    )
    return json.loads(completed.stdout)


def compare(ours: dict[str, set[str]], api: dict[str, dict]) -> dict:
    documents = {}
    total_api = total_ours = total_common = 0
    for document_id in sorted(set(ours) | set(api)):
        mine = ours.get(document_id, set())
        theirs = set(api.get(document_id, {}).get("units", []))
        common = mine & theirs
        total_api += len(theirs)
        total_ours += len(mine)
        total_common += len(common)
        documents[document_id] = {
            "api_mode": api.get(document_id, {}).get("mode"),
            "api_units": len(theirs),
            "our_units": len(mine),
            "common": len(common),
            "api_coverage": round(len(common) / len(theirs), 3) if theirs else None,
            "only_api_sample": sorted(theirs - mine)[:15],
            "only_ours_sample": sorted(mine - theirs)[:15],
        }
    return {
        "api_units": total_api,
        "our_units": total_ours,
        "common_units": total_common,
        "api_coverage": round(total_common / total_api, 3) if total_api else None,
        "our_precision": round(total_common / total_ours, 3) if total_ours else None,
        "by_document": documents,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara units.jsonl con las unidades de rag-istae-api")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.json"))
    parser.add_argument("--api-root", required=True, help="Ruta absoluta al repositorio rag-istae-api")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    _, output_dir = resolve_paths(load_config(config_path), config_path)
    ours: dict[str, set[str]] = {}
    with (output_dir / "units.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                unit = json.loads(line)
                ours.setdefault(unit["document_id"], set()).add(unit["unit_id"])

    result = compare(ours, api_units(Path(args.api_root).resolve()))
    out = output_dir / "units_api_comparison.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Unidades API: {result['api_units']} | propias: {result['our_units']} | "
        f"coincidentes: {result['common_units']} | cobertura de la API: {result['api_coverage']} | "
        f"precisión: {result['our_precision']}"
    )
    worst = sorted(
        (d for d in result["by_document"].items() if d[1]["api_coverage"] is not None),
        key=lambda item: item[1]["api_coverage"],
    )[:8]
    for document_id, info in worst:
        print(f"  {document_id}: {info['common']}/{info['api_units']} ({info['api_mode']})")
    print(f"Detalle: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
