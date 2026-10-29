
"""
Extracción documental del corpus normativo ISTAE mediante Docling.

Responsabilidades:
- Extraer PDF y DOCX.
- Mantener activada la detección de estructura de tablas.
- Intentar primero la extracción sin OCR.
- Repetir con OCR si el contenido extraído es insuficiente.
- Devolver el documento estructurado y su representación textual.

El objeto estructurado de Docling se conserva en memoria para la
fragmentación posterior y no se serializa directamente en JSONL.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from .utils import normalize_text


def build_converter(enable_ocr: bool = False) -> DocumentConverter:
    """
    Construye un conversor Docling.

    La estructura de tablas permanece habilitada tanto en la primera
    conversión como en la conversión de respaldo con OCR.
    """

    pipeline_options = PdfPipelineOptions()

    # Primera pasada: extracción sin OCR.
    # Segunda pasada: OCR si la extracción inicial resulta insuficiente.
    pipeline_options.do_ocr = enable_ocr

    # Decisión del proyecto: mantener la detección de tablas activada.
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def _extract_with_docling(
    path: Path,
    enable_ocr: bool,
) -> dict[str, Any]:
    """
    Convierte el documento y devuelve el objeto estructurado,
    el texto Markdown normalizado y estadísticas básicas.
    """

    converter = build_converter(enable_ocr=enable_ocr)
    conversion_result = converter.convert(str(path))
    document = conversion_result.document

    # El Markdown permite guardar e inspeccionar el contenido extraído.
    # La fragmentación posterior utiliza el documento estructurado.
    markdown = document.export_to_markdown()
    text = normalize_text(markdown)

    is_pdf = path.suffix.lower() == ".pdf"
    pages = len(document.pages) if is_pdf else None

    # Es una medida aproximada: el Markdown puede incluir encabezados,
    # tablas y otros elementos estructurales, no solo texto del cuerpo.
    chars_per_page = (
        len(text) / pages
        if pages and pages > 0
        else None
    )

    return {
        "document": document,
        "text": text,
        "extracted_chars": len(text),
        "pages": pages,
        "chars_per_page": chars_per_page,
        "ocr_enabled": enable_ocr if is_pdf else None,
    }


def extract_document(
    path: Path,
    min_text_chars_before_ocr: int = 100,
    min_chars_per_page_before_ocr: int = 100,
) -> dict[str, Any]:
    """
    Extrae un documento PDF o DOCX.

    PDF:
      1. Ejecutar una primera conversión sin OCR.
      2. Activar OCR si el total de caracteres o el promedio por página
         está por debajo de los umbrales configurados.
      3. Registrar advertencias si el resultado continúa siendo escaso.

    DOCX:
      Se convierte una sola vez, sin OCR.

    Los umbrales son heurísticos: no garantizan que el contenido sea
    correcto, completo ni jurídicamente fiel al documento original.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in {".pdf", ".docx"}:
        raise ValueError(
            f"Formato no soportado para extracción: {suffix}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el documento de origen: {path}"
        )

    if min_text_chars_before_ocr < 0:
        raise ValueError(
            "min_text_chars_before_ocr no puede ser negativo."
        )

    if min_chars_per_page_before_ocr < 0:
        raise ValueError(
            "min_chars_per_page_before_ocr no puede ser negativo."
        )

    # Primera pasada: extracción digital sin OCR.
    first_pass_start = time.perf_counter()
    extracted = _extract_with_docling(
        path=path,
        enable_ocr=False,
    )
    first_pass_seconds = time.perf_counter() - first_pass_start
    ocr_pass_seconds = 0.0

    ocr_attempted = False
    ocr_reason = None

    if suffix == ".pdf":
        total_chars = extracted["extracted_chars"]
        chars_per_page = extracted["chars_per_page"]

        # No basta con medir el total: un PDF largo podría contener
        # texto en unas pocas páginas y carecer de texto en las demás.
        insufficient_total = (
            total_chars < min_text_chars_before_ocr
        )

        insufficient_per_page = (
            chars_per_page is not None
            and chars_per_page < min_chars_per_page_before_ocr
        )

        if insufficient_total or insufficient_per_page:
            ocr_attempted = True

            reasons = []
            if insufficient_total:
                reasons.append(
                    f"total de caracteres inferior a "
                    f"{min_text_chars_before_ocr}"
                )
            if insufficient_per_page:
                reasons.append(
                    f"promedio inferior a "
                    f"{min_chars_per_page_before_ocr} caracteres por página"
                )

            ocr_reason = "; ".join(reasons)

            # Segunda pasada: repetir la conversión completa con OCR.
            ocr_pass_start = time.perf_counter()
            extracted = _extract_with_docling(
                path=path,
                enable_ocr=True,
            )
            ocr_pass_seconds = time.perf_counter() - ocr_pass_start

    warnings: list[str] = []

    # Evaluar el resultado final, después del posible OCR.
    if extracted["extracted_chars"] < min_text_chars_before_ocr:
        warnings.append(
            "El texto extraído quedó por debajo del umbral de "
            f"{min_text_chars_before_ocr} caracteres."
        )

    chars_per_page = extracted["chars_per_page"]

    if (
        suffix == ".pdf"
        and chars_per_page is not None
        and chars_per_page < min_chars_per_page_before_ocr
    ):
        warnings.append(
            "El promedio de caracteres por página quedó por debajo "
            f"del umbral de {min_chars_per_page_before_ocr}."
        )

    if ocr_attempted:
        warnings.append(
            "Se ejecutó una segunda conversión con OCR. "
            f"Motivo: {ocr_reason}."
        )

    extracted.update({
        "extraction_method": "docling",
        "ocr_attempted": ocr_attempted,
        "ocr_reason": ocr_reason,
        "warnings": warnings,
        "first_pass_duration_seconds": round(first_pass_seconds, 3),
        "ocr_pass_duration_seconds": round(ocr_pass_seconds, 3),
        "extraction_duration_seconds": round(
            first_pass_seconds + ocr_pass_seconds, 3
        ),
    })

    return extracted
