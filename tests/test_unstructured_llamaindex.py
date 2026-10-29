import json
import re
import time
import statistics
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


# --------------------------------------------------
# Configuración
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unstructured_llamaindex_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_corpus_root(config):
    configured = Path(config["corpus_root"])

    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured

    if not configured.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus: {configured}"
        )

    return configured.resolve()


def find_ext10(corpus_root):
    matches = [
        p for p in corpus_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
        and re.search(r"EXT10", p.stem, re.IGNORECASE)
    ]

    if not matches:
        raise FileNotFoundError(
            "No se encontró el PDF EXT10 en el corpus."
        )

    if len(matches) > 1:
        print("Advertencia: se encontraron varios EXT10:")
        for path in matches:
            print(f"  - {path}")

    return matches[0]


def json_safe(value):
    """Convierte metadatos a tipos serializables en JSON."""
    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    return str(value)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str
                ) + "\n"
            )


# --------------------------------------------------
# Ejecución
# --------------------------------------------------

def main():
    total_start = time.perf_counter()

    config = load_config()
    corpus_root = find_corpus_root(config)
    pdf_path = find_ext10(corpus_root)

    relative_path = str(
        pdf_path.relative_to(corpus_root)
    )

    print(f"Documento: {pdf_path.name}")
    print("Extractor: Unstructured")
    print("Fragmentador: LlamaIndex SentenceSplitter")
    print("Embeddings: desactivados")
    print("Qdrant: no utilizado")

    # ----------------------------------------------
    # 1. Extracción con Unstructured
    # ----------------------------------------------

    extraction_start = time.perf_counter()
    extraction_error = None
    elements = []

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            infer_table_structure=False,
        )
    except Exception as exc:
        extraction_error = repr(exc)
        raise
    finally:
        extraction_seconds = round(
            time.perf_counter() - extraction_start, 3
        )

    print(f"Elementos extraídos: {len(elements)}")
    print(f"Tiempo de extracción: {extraction_seconds:.3f} s")

    # ----------------------------------------------
    # 2. Guardar elementos extraídos
    # ----------------------------------------------

    element_records = []
    element_texts = []
    element_types = Counter()
    table_count = 0

    for index, element in enumerate(elements):
        category = str(
            getattr(element, "category", "Unknown")
        )

        metadata = getattr(element, "metadata", None)
        metadata_dict = {}

        if metadata is not None:
            try:
                metadata_dict = metadata.to_dict()
            except Exception:
                metadata_dict = {}

        metadata_dict = json_safe(metadata_dict)
        text = str(element)

        element_types[category] += 1

        if category.lower() == "table":
            table_count += 1

        element_records.append({
            "document_key": "EXT10",
            "source_file": relative_path,
            "element_index": index,
            "element_type": category,
            "text": text,
            "metadata": metadata_dict,
        })

        element_texts.append({
            "element_index": index,
            "element_type": category,
            "text": text,
            "metadata": metadata_dict,
        })

    write_jsonl(
        OUTPUT_DIR / "EXT10_elements.jsonl",
        element_records,
    )

    # Exportación legible de elementos.
    extracted_md_path = OUTPUT_DIR / "EXT10_extracted.md"

    with extracted_md_path.open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"# Extracción Unstructured: {pdf_path.name}\n\n")

        for item in element_texts:
            f.write(
                f"<!-- elemento {item['element_index']} "
                f"| tipo: {item['element_type']} -->\n\n"
            )
            f.write(item["text"])
            f.write("\n\n---\n\n")

    # ----------------------------------------------
    # 3. Preparar documentos para LlamaIndex
    # ----------------------------------------------

    documents = []

    for item in element_texts:
        if not item["text"].strip():
            continue

        metadata = {
            "document_key": "EXT10",
            "source_file": relative_path,
            "element_index": item["element_index"],
            "element_type": item["element_type"],
        }

        source_metadata = item["metadata"]

        for key in ("page_number", "page_name", "filename"):
            value = source_metadata.get(key)
            if value is not None:
                metadata[key] = json_safe(value)

        documents.append(
            Document(
                text=item["text"],
                metadata=metadata,
                id_=f"EXT10-element-{item['element_index']}",
            )
        )

    # ----------------------------------------------
    # 4. Fragmentar con LlamaIndex
    # ----------------------------------------------

    chunk_start = time.perf_counter()

    splitter = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    chunk_seconds = round(
        time.perf_counter() - chunk_start, 3
    )

    chunk_records = []

    for index, node in enumerate(nodes):
        metadata = json_safe(node.metadata or {})

        chunk_records.append({
            "document_key": "EXT10",
            "source_file": relative_path,
            "chunk_index": index,
            "text": node.get_content(),
            "metadata": metadata,
        })

    write_jsonl(
        OUTPUT_DIR / "EXT10_chunks.jsonl",
        chunk_records,
    )

    # ----------------------------------------------
    # 5. Métricas y resumen
    # ----------------------------------------------

    total_seconds = round(
        time.perf_counter() - total_start, 3
    )

    chunk_lengths = [
        len(record["text"])
        for record in chunk_records
    ]

    summary = {
        "experiment": "Unstructured + LlamaIndex",
        "document_key": "EXT10",
        "source_file": relative_path,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "extraction_strategy": "auto",
        "infer_table_structure": True,
        "chunker": "SentenceSplitter",
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "embeddings_generated": False,
        "qdrant_used": False,
        "counts": {
            "elements": len(elements),
            "documents_for_chunking": len(documents),
            "chunks": len(chunk_records),
            "tables_detected": table_count,
            "elements_by_type": dict(element_types),
        },
        "timings_seconds": {
            "extraction": extraction_seconds,
            "chunking": chunk_seconds,
            "total": total_seconds,
        },
        "chunk_character_lengths": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "mean": round(
                statistics.mean(chunk_lengths), 2
            ) if chunk_lengths else 0,
        },
        "errors": {
            "extraction": extraction_error,
        },
        "outputs": [
            "EXT10_elements.jsonl",
            "EXT10_chunks.jsonl",
            "EXT10_extracted.md",
            "summary.json",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- Resumen ---")
    print(f"Elementos: {len(elements)}")
    print(f"Tablas detectadas: {table_count}")
    print(f"Documentos LlamaIndex: {len(documents)}")
    print(f"Fragmentos: {len(nodes)}")
    print(f"Tiempo extracción: {extraction_seconds:.3f} s")
    print(f"Tiempo fragmentación: {chunk_seconds:.3f} s")
    print(f"Tiempo total: {total_seconds:.3f} s")
    print(f"Resultados: {OUTPUT_DIR}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
