
"""
Fragmentación jerárquica del corpus normativo ISTAE mediante Docling.

Cada fragmento conserva:
- Identificadores del documento y del fragmento.
- Nombre del archivo de origen.
- Encabezados y números de página disponibles.
- Tipos de elementos detectados por Docling.
- Texto y longitud del fragmento.

Los metadatos adicionales de extracción y el hash del archivo
se incorporan en pipeline.py.
"""

from __future__ import annotations

from typing import Any

from docling.chunking import HierarchicalChunker


def _get_chunk_metadata(chunk: Any) -> dict[str, Any]:
    """
    Extrae metadatos estructurales del fragmento Docling.

    Se usan valores predeterminados cuando Docling no proporciona
    encabezados, procedencia o tipos de elementos.
    """

    metadata = chunk.meta

    headings = list(getattr(metadata, "headings", None) or [])
    doc_items = list(getattr(metadata, "doc_items", None) or [])

    page_numbers: set[int] = set()
    item_types: set[str] = set()

    for item in doc_items:
        # Conservar el tipo de elemento: texto, tabla, título, etc.
        label = getattr(item, "label", None)

        if label is not None:
            label_value = getattr(label, "value", label)
            item_types.add(str(label_value))

        # La procedencia puede contener una o varias páginas.
        for provenance in getattr(item, "prov", None) or []:
            page_number = getattr(provenance, "page_no", None)

            if page_number is not None:
                try:
                    page_numbers.add(int(page_number))
                except (TypeError, ValueError):
                    # Un dato de procedencia inesperado no debe
                    # interrumpir toda la fragmentación.
                    continue

    return {
        "headings": [str(heading) for heading in headings],
        "page_numbers": sorted(page_numbers),
        "item_types": sorted(item_types),
    }


def chunk_document(
    document: Any,
    document_id: str,
    source_file: str,
) -> list[dict[str, Any]]:
    """
    Genera fragmentos jerárquicos a partir del documento estructurado.

    El chunk_id es determinista dentro de una versión de la
    fragmentación: combina el document_id estable con la posición
    del fragmento en el resultado.
    """

    chunker = HierarchicalChunker()
    chunks: list[dict[str, Any]] = []

    for chunk in chunker.chunk(document):
        text = (chunk.text or "").strip()

        # No guardar fragmentos sin contenido.
        if not text:
            continue

        metadata = _get_chunk_metadata(chunk)

        # Asignar el índice después de descartar los fragmentos vacíos,
        # para evitar saltos en la numeración.
        chunk_index = len(chunks) + 1

        chunks.append({
            "chunk_id": f"{document_id}:{chunk_index:05d}",
            "document_id": document_id,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "headings": metadata["headings"],
            "page_numbers": metadata["page_numbers"],
            "item_types": metadata["item_types"],
            "text": text,
            "char_count": len(text),
            "chunking_method": "docling_hierarchical",
        })

    return chunks
