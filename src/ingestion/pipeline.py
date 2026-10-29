
"""
Pipeline reproducible de ingestión para el corpus normativo ISTAE.

Flujo:
1. Leer la configuración.
2. Construir el inventario y calcular hashes.
3. Extraer PDF y DOCX mediante Docling.
4. Fragmentar con HierarchicalChunker.
5. Guardar texto extraído, fragmentos y resultados por documento.
6. Validar la ejecución y generar el manifiesto.

Los documentos originales no se modifican.
El pipeline no genera embeddings ni se conecta a una base vectorial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk import chunk_document
from .extract import extract_document
from .inventory import build_inventory
from .utils import sha256_file, write_jsonl
from .validate import validate_run


def load_config(path: Path) -> dict[str, Any]:
    """Lee y devuelve la configuración JSON del proyecto."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_document_id(relative_path: str) -> str:
    """
    Extrae el código documental inicial del nombre del archivo.

    Ejemplos:
        EXT07 - REGLAMENTO.pdf  -> EXT07
        INT03 - Reglamento.docx -> INT03

    Si el archivo no tiene un prefijo reconocible, utiliza el nombre
    del archivo sin extensión como alternativa.
    """
    filename = Path(relative_path).stem

    match = re.match(
        r"^((?:EXT|INT)\d+)\b",
        filename,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return filename.strip()


def _format_duration(seconds: float) -> str:
    """Convierte segundos en una duración legible."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def run_pipeline(config_path: Path) -> dict[str, Any]:
    """Ejecuta el proceso completo de ingestión y mide sus duraciones."""

    pipeline_start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)

    # Resolver las rutas relativas respecto del proyecto.
    corpus_root = Path(config["corpus_root"]).expanduser()

    if not corpus_root.is_absolute():
        corpus_root = (config_path.parent / corpus_root).resolve()

    output_dir = Path(config.get("output_dir", "outputs"))

    if not output_dir.is_absolute():
        output_dir = (
            config_path.parent.parent / output_dir
        ).resolve()

    # Inventario: incluye archivos auxiliares que no necesariamente
    # serán extraídos en esta versión.
    excluded_dirs = set(
        config.get("exclude_directory_names", [])
    )

    inventory_extensions = {
        extension.lower()
        for extension in config.get(
            "inventory_extensions",
            [".pdf", ".docx", ".xlsx", ".xls", ".doc"],
        )
    }

    inventory = build_inventory(
        corpus_root,
        excluded_dirs=excluded_dirs,
        extensions=inventory_extensions,
    )

    write_jsonl(
        output_dir / "inventory.jsonl",
        inventory,
    )

    # Solo se procesan los formatos declarados en include_extensions.
    # XLS/XLSX/DOC se inventarían, pero no se extraen en esta versión.
    extract_extensions = {
        extension.lower()
        for extension in config.get(
            "include_extensions",
            [".pdf", ".docx"],
        )
    }

    supported_extensions = {".pdf", ".docx"}
    unsupported = extract_extensions - supported_extensions

    if unsupported:
        raise ValueError(
            "Hay formatos declarados para extracción que todavía "
            f"no están soportados: {sorted(unsupported)}"
        )

    document_results: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    document_mapping: list[dict[str, str]] = []

    # Umbral utilizado para decidir si un PDF necesita una segunda
    # pasada con OCR. No interviene en la fragmentación.
    min_text_chars_before_ocr = int(
        config.get("min_text_chars_before_ocr", 100)
    )

    # Número de documentos que cumplen los filtros de extracción.
    eligible_documents = 0

    # Procesar cada documento de forma independiente para que un error
    # no detenga automáticamente la ingestión de los demás.
    for item in inventory:
        if item["extension"] not in extract_extensions:
            continue

        eligible_documents += 1
        document_start = time.perf_counter()
        extraction_seconds = None
        chunking_seconds = None
        source_path = corpus_root / Path(item["relative_path"])
        document_id = stable_document_id(item["relative_path"])

        base = {
            "document_id": document_id,
            "source_file": item["file_name"],
            "relative_path": item["relative_path"],
            "source_sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }

        try:
            # Docling decide si debe realizar una segunda pasada con OCR.
            extraction_start = time.perf_counter()
            extracted = extract_document(
                source_path,
                min_text_chars_before_ocr=int(
                    config.get("min_text_chars_before_ocr", 100)
                ),
                min_chars_per_page_before_ocr=int(
                    config.get("min_chars_per_page_before_ocr", 100)
                ),
            )
            extraction_seconds = time.perf_counter() - extraction_start

            text = extracted["text"]

            # HierarchicalChunker recibe el documento estructurado,
            # no el Markdown exportado ni un texto dividido manualmente.
            chunking_start = time.perf_counter()
            chunks = chunk_document(
                document=extracted["document"],
                document_id=document_id,
                source_file=item["relative_path"],
            )
            chunking_seconds = time.perf_counter() - chunking_start

            # Enriquecer cada fragmento con información del documento de origen.
            for chunk in chunks:
                chunk["source_sha256"] = item["sha256"]
                chunk["extraction_method"] = extracted["extraction_method"]
                chunk["ocr_attempted"] = extracted["ocr_attempted"]
                chunk["ocr_enabled"] = extracted["ocr_enabled"]
                chunk["ocr_reason"] = extracted.get("ocr_reason")

            result = {
                **base,
                "status": "ok",
                "extracted_chars": extracted["extracted_chars"],
                "chunk_count": len(chunks),
                "pages": extracted["pages"],
                "extraction_method": extracted["extraction_method"],
                "ocr_attempted": extracted["ocr_attempted"],
                "ocr_enabled": extracted["ocr_enabled"],
                "warnings": extracted["warnings"],
                "chunking_method": "docling_hierarchical",
                "chars_per_page": extracted["chars_per_page"],
                "ocr_reason": extracted["ocr_reason"],
                "first_pass_duration_seconds": extracted.get("first_pass_duration_seconds"),
                "ocr_pass_duration_seconds": extracted.get("ocr_pass_duration_seconds"),
                "extraction_duration_seconds": round(extraction_seconds, 3),
                "chunking_duration_seconds": round(chunking_seconds, 3),
                "processing_duration_seconds": round(
                    time.perf_counter() - document_start, 3
                ),
            }

            all_chunks.extend(chunks)
            document_results.append(result)

            document_mapping.append({
                "document_id": document_id,
                "relative_path": item["relative_path"],
                "source_sha256": item["sha256"],
            })

            # Guardar el texto extraído por hash para conservar
            # trazabilidad y facilitar inspecciones posteriores.
            text_path = (
                output_dir
                / "extracted_text"
                / f"{item['sha256']}.txt"
            )
            text_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            text_path.write_text(
                text,
                encoding="utf-8",
            )

            # El tiempo total del documento incluye extracción, chunking y escritura.
            result["processing_duration_seconds"] = round(
                time.perf_counter() - document_start, 3
            )

            print(
                f"[OK] {item['relative_path']} | "
                f"extracción={extraction_seconds:.2f}s | "
                f"chunking={chunking_seconds:.2f}s | "
                f"total_documento={result['processing_duration_seconds']:.2f}s | "
                f"chunks={len(chunks)}"
            )

        except Exception as exc:
            # Registrar el error y continuar con el siguiente documento.
            error_duration = time.perf_counter() - document_start
            document_results.append({
                **base,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "extraction_duration_seconds": (
                    round(extraction_seconds, 3)
                    if extraction_seconds is not None else None
                ),
                "chunking_duration_seconds": (
                    round(chunking_seconds, 3)
                    if chunking_seconds is not None else None
                ),
                "processing_duration_seconds": round(error_duration, 3),
            })
            print(
                f"[ERROR] {item['relative_path']} | "
                f"total_documento={error_duration:.2f}s | "
                f"{type(exc).__name__}: {exc}"
            )

    # Persistir los resultados. Los objetos Docling no se serializan:
    # solo guardamos texto y metadatos simples en los JSONL.
    write_jsonl(
        output_dir / "document_results.jsonl",
        document_results,
    )
    write_jsonl(
        output_dir / "chunks.jsonl",
        all_chunks,
    )
    write_jsonl(
        output_dir / "document_mapping.jsonl",
        document_mapping,
    )

    # Validación posterior a la extracción y fragmentación.
    report = validate_run(
        inventory,
        document_results,
        all_chunks,
    )

    report.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(config_path),
        "pipeline_version": "0.3.0",
        "source_extensions": sorted({
            item["extension"] for item in inventory
        }),
        "extraction_extensions": sorted(extract_extensions),
        "extraction_engine": "docling",
        "chunking_engine": "HierarchicalChunker",
        "table_structure_enabled": True,
        "min_text_chars_before_ocr": min_text_chars_before_ocr,
        "timing_method": "time.perf_counter",
        "timed_documents_count": eligible_documents,
        "documents_with_timing_count": sum(
            "processing_duration_seconds" in item_result
            for item_result in document_results
        ),
    })

    # El manifiesto documenta los parámetros y el resultado de la corrida.
    report_path = output_dir / "run_manifest.json"
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Primera escritura para incluir el coste de serializar el manifiesto.
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["pipeline_duration_seconds"] = round(
        time.perf_counter() - pipeline_start, 3
    )
    report["pipeline_duration_minutes"] = round(
        report["pipeline_duration_seconds"] / 60, 2
    )
    report["pipeline_duration_human"] = _format_duration(
        report["pipeline_duration_seconds"]
    )
    # La segunda escritura persiste la duración medida hasta finalizar la primera.
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n[FIN] Duración total del pipeline: "
        f"{report['pipeline_duration_human']} "
        f"({report['pipeline_duration_seconds']:.2f} s)"
    )
    return report


def main() -> None:
    """Punto de entrada para ejecución desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Pipeline de ingestión RAG ISTAE con Docling"
    )

    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "config"
            / "config.json"
        ),
        help="Ruta a config/config.json",
    )

    args = parser.parse_args()
    report = run_pipeline(Path(args.config))

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
