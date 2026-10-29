
"""
Validación de resultados de la ingesta documental RAG ISTAE.

Comprueba:
- Errores de extracción y documentos sin contenido.
- Identificadores de fragmentos ausentes o duplicados.
- Presencia de metadatos obligatorios.
- Tipos de datos y coherencia de los campos.
- Fragmentos vacíos y longitudes inconsistentes.
- Coherencia entre el resultado documental y sus fragmentos.

Esta validación es técnica. No demuestra por sí sola que el contenido
extraído sea jurídicamente fiel al documento original.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# Contrato mínimo que deben cumplir los fragmentos generados por chunk.py
# y enriquecidos por pipeline.py.
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "source_file",
    "chunk_index",
    "headings",
    "page_numbers",
    "item_types",
    "text",
    "char_count",
    "chunking_method",
    "source_sha256",
    "extraction_method",
    "ocr_attempted",
    "ocr_enabled",
    "ocr_reason",
}


def validate_run(
    inventory: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Valida los resultados de una ejecución del pipeline.

    Devuelve un diccionario serializable en JSON con contadores,
    errores, advertencias y un indicador global de aprobación.
    """

    warnings: list[str] = []
    errors: list[str] = []

    # ---------------------------------------------------------
    # 1. Validar los resultados de extracción documental
    # ---------------------------------------------------------

    failed = [
        doc for doc in documents
        if doc.get("status") == "error"
    ]

    empty_documents = [
        doc for doc in documents
        if doc.get("status") == "ok"
        and not doc.get("extracted_chars")
    ]

    no_chunks_documents = [
        doc for doc in documents
        if doc.get("status") == "ok"
        and doc.get("extracted_chars", 0) > 0
        and doc.get("chunk_count", 0) == 0
    ]

    if failed:
        warnings.append(
            f"{len(failed)} documentos no se pudieron extraer."
        )

    if empty_documents:
        warnings.append(
            f"{len(empty_documents)} documentos produjeron texto vacío."
        )

    if no_chunks_documents:
        warnings.append(
            f"{len(no_chunks_documents)} documentos tienen texto, "
            "pero no produjeron fragmentos."
        )

    # ---------------------------------------------------------
    # 2. Validar identificadores de fragmentos
    # ---------------------------------------------------------

    missing_ids = [
        index
        for index, chunk in enumerate(chunks, start=1)
        if not isinstance(chunk.get("chunk_id"), str)
        or not chunk.get("chunk_id", "").strip()
    ]

    if missing_ids:
        errors.append(
            f"Fragmentos sin chunk_id válido: {len(missing_ids)}"
        )

    valid_ids = [
        chunk["chunk_id"]
        for chunk in chunks
        if isinstance(chunk.get("chunk_id"), str)
        and chunk["chunk_id"].strip()
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(valid_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(
            f"IDs de fragmento duplicados: {len(duplicate_ids)}"
        )

    # ---------------------------------------------------------
    # 3. Validar presencia de metadatos obligatorios
    # ---------------------------------------------------------

    missing_fields_by_name: dict[str, int] = {}

    for field in REQUIRED_CHUNK_FIELDS:
        missing_count = sum(
            field not in chunk
            for chunk in chunks
        )

        if missing_count:
            missing_fields_by_name[field] = missing_count

    if missing_fields_by_name:
        details = ", ".join(
            f"{field}: {count}"
            for field, count in sorted(
                missing_fields_by_name.items()
            )
        )

        errors.append(
            f"Metadatos obligatorios ausentes en fragmentos ({details})."
        )

    # ---------------------------------------------------------
    # 4. Validar texto y coherencia de char_count
    # ---------------------------------------------------------

    empty_chunks = [
        chunk for chunk in chunks
        if not isinstance(chunk.get("text"), str)
        or not chunk.get("text", "").strip()
    ]

    if empty_chunks:
        errors.append(
            f"Fragmentos sin contenido textual válido: "
            f"{len(empty_chunks)}"
        )

    invalid_char_count = []

    for chunk in chunks:
        text = chunk.get("text")
        char_count = chunk.get("char_count")

        # bool es una subclase de int en Python; no debe aceptarse
        # como una longitud válida.
        if (
            not isinstance(text, str)
            or not isinstance(char_count, int)
            or isinstance(char_count, bool)
            or char_count != len(text)
        ):
            invalid_char_count.append(chunk)

    if invalid_char_count:
        errors.append(
            "Longitud char_count inconsistente con text: "
            f"{len(invalid_char_count)} fragmentos."
        )

    # ---------------------------------------------------------
    # 5. Validar los tipos de los metadatos estructurales
    # ---------------------------------------------------------

    invalid_headings = [
        chunk for chunk in chunks
        if not isinstance(chunk.get("headings"), list)
        or any(
            not isinstance(value, str)
            for value in chunk.get("headings", [])
        )
    ]

    invalid_pages = [
        chunk for chunk in chunks
        if not isinstance(chunk.get("page_numbers"), list)
        or any(
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
            for page in chunk.get("page_numbers", [])
        )
    ]

    invalid_item_types = [
        chunk for chunk in chunks
        if not isinstance(chunk.get("item_types"), list)
        or any(
            not isinstance(value, str)
            for value in chunk.get("item_types", [])
        )
    ]

    if invalid_headings:
        errors.append(
            f"Metadatos headings inválidos: {len(invalid_headings)}"
        )

    if invalid_pages:
        errors.append(
            f"Metadatos page_numbers inválidos: {len(invalid_pages)}"
        )

    if invalid_item_types:
        errors.append(
            f"Metadatos item_types inválidos: {len(invalid_item_types)}"
        )

    # ---------------------------------------------------------
    # 6. Validar índices de fragmentos
    # ---------------------------------------------------------

    invalid_chunk_index = [
        chunk for chunk in chunks
        if not isinstance(chunk.get("chunk_index"), int)
        or isinstance(chunk.get("chunk_index"), bool)
        or chunk.get("chunk_index", 0) < 1
    ]

    if invalid_chunk_index:
        errors.append(
            f"Índices de fragmento inválidos: "
            f"{len(invalid_chunk_index)}"
        )

    # ---------------------------------------------------------
    # 7. Validar referencias entre documentos y fragmentos
    # ---------------------------------------------------------

    document_ids = {
        doc.get("document_id")
        for doc in documents
        if doc.get("document_id")
    }

    chunks_without_document = [
        chunk for chunk in chunks
        if not chunk.get("document_id")
        or chunk.get("document_id") not in document_ids
    ]

    if chunks_without_document:
        errors.append(
            "Fragmentos que no corresponden a un documento procesado: "
            f"{len(chunks_without_document)}"
        )

    # Comparar el número de fragmentos registrado por documento
    # con el número real de fragmentos agrupados por document_id.
    chunk_counts = Counter(
        chunk.get("document_id")
        for chunk in chunks
        if chunk.get("document_id")
    )

    inconsistent_chunk_counts = []

    for doc in documents:
        if doc.get("status") != "ok":
            continue

        document_id = doc.get("document_id")
        declared_count = doc.get("chunk_count")

        if (
            isinstance(declared_count, int)
            and not isinstance(declared_count, bool)
            and declared_count != chunk_counts.get(document_id, 0)
        ):
            inconsistent_chunk_counts.append(doc)

    if inconsistent_chunk_counts:
        errors.append(
            "El chunk_count declarado no coincide con los fragmentos "
            f"reales en {len(inconsistent_chunk_counts)} documentos."
        )

    # ---------------------------------------------------------
    # 8. Validar el estado del inventario
    # ---------------------------------------------------------

    inventory_count = len(inventory)
    processed_count = len(documents)

    if processed_count > inventory_count:
        warnings.append(
            "Hay más resultados documentales que registros "
            "en el inventario."
        )

    # No se exige que todos los elementos del inventario se extraigan:
    # puede contener XLSX, XLS u otros formatos solo inventariados.
    inventory_paths = {
        item.get("relative_path")
        for item in inventory
        if item.get("relative_path")
    }

    unexpected_documents = [
        doc for doc in documents
        if doc.get("relative_path")
        and doc["relative_path"] not in inventory_paths
    ]

    if unexpected_documents:
        errors.append(
            "Resultados documentales que no aparecen en el inventario: "
            f"{len(unexpected_documents)}"
        )

    # ---------------------------------------------------------
    # 9. Construir el informe final
    # ---------------------------------------------------------

    passed = not errors and not failed

    return {
        "inventory_count": inventory_count,
        "processed_documents": processed_count,
        "chunks_count": len(chunks),
        "failed_count": len(failed),
        "empty_text_count": len(empty_documents),
        "documents_without_chunks_count": len(no_chunks_documents),
        "missing_chunk_ids_count": len(missing_ids),
        "duplicate_chunk_ids_count": len(duplicate_ids),
        "empty_chunks_count": len(empty_chunks),
        "missing_metadata_fields": missing_fields_by_name,
        "invalid_char_count_count": len(invalid_char_count),
        "invalid_headings_count": len(invalid_headings),
        "invalid_page_numbers_count": len(invalid_pages),
        "invalid_item_types_count": len(invalid_item_types),
        "invalid_chunk_index_count": len(invalid_chunk_index),
        "chunks_without_document_count": len(chunks_without_document),
        "inconsistent_document_chunk_count": len(
            inconsistent_chunk_counts
        ),
        "unexpected_documents_count": len(unexpected_documents),
        "errors": errors,
        "warnings": warnings,
        "passed": passed,
    }
