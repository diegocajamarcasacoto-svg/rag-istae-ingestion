
import json
import re
import time
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from unstructured.partition.pdf import partition_pdf


# ==================================================
# CONFIGURACIÓN
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unstructured_hi_res_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENT_KEY = "EXT10"

# Procesar el PDF completo.
# No se limita el procesamiento a páginas específicas.
STRATEGY = "hi_res"
INFER_TABLE_STRUCTURE = True
LANGUAGES = ["spa"]


# ==================================================
# UTILIDADES
# ==================================================

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_corpus_root(config):
    configured = Path(config["corpus_root"])

    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured

    if not configured.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus configurado: {configured}"
        )

    return configured.resolve()


def find_ext10(corpus_root):
    matches = [
        path
        for path in corpus_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and re.search(r"EXT10", path.stem, re.IGNORECASE)
    ]

    if not matches:
        raise FileNotFoundError(
            "No se encontró el PDF EXT10 en el corpus."
        )

    if len(matches) > 1:
        print("Advertencia: se encontraron varios archivos EXT10:")
        for path in matches:
            print(f"  - {path}")

    return matches[0]


def json_safe(value):
    """Convierte metadatos en valores serializables a JSON."""
    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    return str(value)


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_metadata(element):
    metadata = getattr(element, "metadata", None)

    if metadata is None:
        return {}

    try:
        return json_safe(metadata.to_dict())
    except Exception:
        return {}


def get_table_html(element):
    metadata = getattr(element, "metadata", None)

    if metadata is None:
        return None

    return getattr(metadata, "text_as_html", None)


# ==================================================
# EJECUCIÓN
# ==================================================

def main():
    total_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    config = load_config()
    corpus_root = find_corpus_root(config)
    pdf_path = find_ext10(corpus_root)

    relative_path = str(pdf_path.relative_to(corpus_root))

    print("=" * 65)
    print("PRUEBA: UNSTRUCTURED HI_RES")
    print("=" * 65)
    print(f"Documento: {pdf_path.name}")
    print(f"Ruta relativa: {relative_path}")
    print(f"Estrategia: {STRATEGY}")
    print(f"Inferencia de tablas: {INFER_TABLE_STRUCTURE}")
    print(f"Idiomas OCR configurados: {LANGUAGES}")
    print("Páginas: documento completo")
    print("Embeddings: desactivados")
    print("Qdrant: no utilizado")
    print(f"Salida: {OUTPUT_DIR}")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Extracción
    # --------------------------------------------------

    extraction_start = time.perf_counter()

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=STRATEGY,
            infer_table_structure=INFER_TABLE_STRUCTURE,
            languages=LANGUAGES,
        )
    except Exception as exc:
        extraction_seconds = round(
            time.perf_counter() - extraction_start, 3
        )
        total_seconds = round(
            time.perf_counter() - total_start, 3
        )

        error_summary = {
            "experiment": "Unstructured hi_res",
            "document_key": DOCUMENT_KEY,
            "source_file": relative_path,
            "generated_at_utc": started_at,
            "strategy": STRATEGY,
            "infer_table_structure": INFER_TABLE_STRUCTURE,
            "languages": LANGUAGES,
            "status": "failed",
            "timings_seconds": {
                "extraction": extraction_seconds,
                "total_until_failure": total_seconds,
            },
            "error": repr(exc),
        }

        write_json(
            OUTPUT_DIR / "summary_error.json",
            error_summary,
        )

        print("\nERROR durante la extracción:")
        print(repr(exc))
        print(f"Tiempo hasta el error: {total_seconds:.3f} s")
        print(
            "Detalle guardado en: "
            f"{OUTPUT_DIR / 'summary_error.json'}"
        )

        raise

    extraction_seconds = round(
        time.perf_counter() - extraction_start, 3
    )

    print(f"\nExtracción terminada en {extraction_seconds:.3f} s")
    print(f"Elementos encontrados: {len(elements)}")

    # --------------------------------------------------
    # 2. Registrar elementos y tablas
    # --------------------------------------------------

    elements_records = []
    table_records = []
    elements_by_type = Counter()
    elements_by_page = Counter()

    for index, element in enumerate(elements):
        element_type = str(
            getattr(element, "category", "Unknown")
        )

        text = str(element)
        metadata = get_metadata(element)

        elements_by_type[element_type] += 1

        page_number = metadata.get("page_number")

        if page_number is not None:
            elements_by_page[str(page_number)] += 1

        record = {
            "document_key": DOCUMENT_KEY,
            "source_file": relative_path,
            "element_index": index,
            "element_type": element_type,
            "text": text,
            "metadata": metadata,
        }

        elements_records.append(record)

        if element_type.lower() == "table":
            table_records.append({
                **record,
                "text_as_html": get_table_html(element),
            })

    write_json(
        OUTPUT_DIR / "EXT10_hi_res_elements.json",
        elements_records,
    )

    write_json(
        OUTPUT_DIR / "EXT10_hi_res_tables.json",
        table_records,
    )

    # --------------------------------------------------
    # 3. Exportar una representación legible
    # --------------------------------------------------

    extracted_md_path = (
        OUTPUT_DIR / "EXT10_hi_res_extracted.md"
    )

    with extracted_md_path.open(
        "w", encoding="utf-8"
    ) as file:
        file.write(
            f"# Extracción Unstructured hi_res: {pdf_path.name}\n\n"
        )

        for record in elements_records:
            page = record["metadata"].get("page_number", "N/D")

            file.write(
                f"<!-- elemento {record['element_index']} "
                f"| tipo: {record['element_type']} "
                f"| página: {page} -->\n\n"
            )

            file.write(record["text"])
            file.write("\n\n---\n\n")

    # --------------------------------------------------
    # 4. Resumen
    # --------------------------------------------------

    total_seconds = round(
        time.perf_counter() - total_start, 3
    )

    summary = {
        "experiment": "Unstructured hi_res",
        "document_key": DOCUMENT_KEY,
        "source_file": relative_path,
        "generated_at_utc": started_at,
        "status": "completed",
        "extraction_strategy": STRATEGY,
        "infer_table_structure": INFER_TABLE_STRUCTURE,
        "languages": LANGUAGES,
        "embeddings_generated": False,
        "qdrant_used": False,
        "counts": {
            "elements": len(elements_records),
            "tables_detected": len(table_records),
            "elements_by_type": dict(elements_by_type),
            "elements_by_page": dict(elements_by_page),
        },
        "timings_seconds": {
            "extraction": extraction_seconds,
            "total": total_seconds,
        },
        "outputs": [
            "EXT10_hi_res_elements.json",
            "EXT10_hi_res_tables.json",
            "EXT10_hi_res_extracted.md",
            "summary.json",
        ],
    }

    write_json(OUTPUT_DIR / "summary.json", summary)

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print(f"Estado: {summary['status']}")
    print(f"Elementos extraídos: {len(elements_records)}")
    print(f"Tablas detectadas: {len(table_records)}")
    print(f"Tiempo de extracción: {extraction_seconds:.3f} s")
    print(f"Tiempo total: {total_seconds:.3f} s")
    print(f"Tipos de elementos: {dict(elements_by_type)}")
    print(f"Resultados: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
