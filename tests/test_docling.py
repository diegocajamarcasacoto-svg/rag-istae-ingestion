import time
import json
import re
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.chunking import HierarchicalChunker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_corpus_root(config):
    configured = Path(config["corpus_root"])

    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured

    if configured.exists():
        return configured

    raise FileNotFoundError(
        f"No se encontró el corpus configurado: {configured}"
    )


def choose_documents(corpus_root):
    files = [
        p for p in corpus_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}
    ]

    selected = {}

    # Documentos de prueba difíciles: busca EXT07, EXT07 y EXT07
    for code in ("EXT07", "EXT07", "EXT07"):
        matches = [
            p for p in files
            if re.search(code, p.stem, re.IGNORECASE)
        ]
        if matches:
            selected[code] = matches[0]

    # Un documento interno DOCX como referencia
    internal_docs = [
        p for p in files
        if p.suffix.lower() == ".docx"
        and "INTERNO" in str(p.parent).upper()
        and not any(
            re.search(code, p.stem, re.IGNORECASE)
            for code in ("EXT07", "EXT07", "EXT07")
        )
    ]

    if internal_docs:
        selected["INTERNO_MUESTRA"] = internal_docs[0]

    if not selected:
        raise RuntimeError(
            "No se encontraron documentos de prueba. "
            "Revisa config.json y la carpeta del corpus."
        )

    return selected


def main():
    config = load_config()
    corpus_root = find_corpus_root(config)

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "docling_test"
        / "hierarchical_no_ocr"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prueba exclusiva del documento con tablas complejas
    matches = [
        p for p in corpus_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
        and re.search("EXT07", p.stem, re.IGNORECASE)
    ]

    if not matches:
        raise FileNotFoundError("No se encontró el PDF EXT07.")

    documents = {"EXT07": matches[0]}

    # Prueba B: desactivar OCR y conservar la estructura de tablas
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    chunker = HierarchicalChunker()

    summary = []

    for label, path in documents.items():
        print(f"\nProcesando {label}: {path.name}")
        ## INICIO
        start_time = time.perf_counter()
        try:
            result = converter.convert(str(path))
            doc = result.document
            chunks = list(chunker.chunk(doc))

            processing_time_seconds = round(
                time.perf_counter() - start_time, 3
            )
        ##  FIN
            records = []

            for index, chunk in enumerate(chunks):
                headings = list(chunk.meta.headings or [])
                page_numbers = set()
                item_types = set()

                for item in (chunk.meta.doc_items or []):
                    if getattr(item, "label", None):
                        item_types.add(str(item.label))

                    for prov in (getattr(item, "prov", None) or []):
                        page_no = getattr(prov, "page_no", None)
                        if page_no is not None:
                            page_numbers.add(page_no)

                records.append({
                    "document_key": label,
                    "source_file": str(
                        path.relative_to(corpus_root)
                    ),
                    "chunk_index": index,
                    "text": chunk.text,
                    "headings": headings,
                    "page_numbers": sorted(page_numbers),
                    "item_types": sorted(item_types),
                })

            jsonl_path = output_dir / f"{label}_chunks.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

            markdown_path = output_dir / f"{label}_converted.md"
            markdown_path.write_text(
                doc.export_to_markdown(),
                encoding="utf-8"
            )

            summary.append({
                "document_key": label,
                "source_file": path.name,
                "chunks": len(records),
                "markdown_chars": len(
                    doc.export_to_markdown()
                ),
                "chunks_with_headings": sum(
                    bool(r["headings"]) for r in records
                ),
                "chunks_with_pages": sum(
                    bool(r["page_numbers"]) for r in records
                ),
                "jsonl": jsonl_path.name,
                "markdown": markdown_path.name,
                "error": None,
                "processing_time_seconds": processing_time_seconds,
            })

            print(
                f"  Fragmentos: {len(records)} | "
                f"Con encabezados: "
                f"{summary[-1]['chunks_with_headings']}"
                f"Tiempo: {processing_time_seconds:.3f} s"
            )

        except Exception as exc:
            summary.append({
                "document_key": label,
                "source_file": path.name,
                "error": repr(exc),
                "processing_time_seconds": processing_time_seconds,
            })
            print(f"  ERROR: {exc}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nPrueba finalizada. Resultados: {output_dir}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
