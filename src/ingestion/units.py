"""
Unidades normativas (artículos) a partir de los fragmentos de Docling.

Docling sigue siendo el único fragmentador: esta etapa no corta, no
reescribe ni limpia texto. Solo agrupa fragmentos consecutivos de un mismo
documento en la unidad normativa a la que pertenecen, para que la consulta
pueda recuperar sobre los fragmentos finos y devolver el artículo completo
(patrón *small-to-big* o *parent-document retrieval*).

Cada fragmento pertenece exactamente a una unidad. La unidad conserva la
lista de sus ``chunk_id`` (rango contiguo), por lo que ``units.jsonl`` es el
índice chunk → artículo.

Reglas de agrupación (equivalentes a las de ``rag-istae-api``,
``app/normativa/structure.py``, para obtener los mismos ``unit_id``):

- Modo ``articulado``: la norma numera artículos desde 1. Un fragmento que
  empieza con «Artículo N» / «Art. N» (o un encabezado Docling con esa forma)
  abre la unidad ``<doc>-artN``. Solo se aceptan números que siguen la
  secuencia (el primero es 1, saltos de como máximo ``MAX_ARTICLE_GAP`` o
  reinicio en 1), para no confundir menciones a artículos de otras normas.
- Las disposiciones (generales, transitorias, derogatorias, finales,
  reformatorias) abren unidades ``<doc>-disp-<tipo>-<ordinal>``.
- Modo ``secciones`` (normas sin articulado propio): cada encabezado abre
  ``<doc>-secNN``.
- Lo anterior al primer artículo queda en ``<doc>-preambulo``.

Los encabezados de Docling (``headings``) no forman parte del texto del
fragmento; cuando cambian se incorporan como líneas de encabezado de la
unidad, porque a veces contienen el propio artículo (p. ej. «Art. 99.- …»).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

UNITS_VERSION = "1.0.0"

MAX_ARTICLE_GAP = 5
MIN_ARTICLES_FOR_ARTICULATED = 3
_MAX_PLAIN_HEADING = 120
_SAMPLE_LIMIT = 10

_ARTICLE = re.compile(
    r"^\s*(?:[-*>]\s+)?(?:#{1,6}\s*)?(?:\*\*)?\s*(?:art[íi]culo|art\.)\s*(?:\*\*)?\s*"
    r"(?P<num>\d+(?:\.\d+)?(?:[-–]\s*[a-z]\b|\s+(?:bis|ter|quater)\b)?|innumerado|[úu]nico)",
    re.IGNORECASE,
)
_ORDINAL_WORDS = (
    r"PRIMER[AO]|SEGUND[AO]|TERCER[AO]|CUART[AO]|QUINT[AO]|SEXT[AO]|S[ÉE]PTIM[AO]|"
    r"OCTAV[AO]|NOVEN[AO]"
)
_ORDINAL = re.compile(
    rf"^\s*(?:[-*]\s+)?(?:#{{1,6}}\s*)?(?:\*\*)?\s*(?P<ord>"
    rf"D[ÉE]CIM[AO](?:\s+(?:{_ORDINAL_WORDS}))?|UND[ÉE]CIM[AO]|DUOD[ÉE]CIM[AO]|"
    rf"VIG[ÉE]SIM[AO](?:\s+(?:{_ORDINAL_WORDS}))?|{_ORDINAL_WORDS}|[ÚU]NIC[AO])"
    rf"\b\s*(?:\*\*)?\s*[.\-–:,]",
    re.IGNORECASE,
)
_DISPOSITIONS = re.compile(
    r"DISPOSICI[ÓO]N(?:ES)?\s+(?P<tipo>GENERAL|TRANSITORIA|DEROGATORIA|FINAL|REFORMATORIA)",
    re.IGNORECASE,
)
_STRUCTURAL = re.compile(r"^(?P<kind>T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|PAR[ÁA]GRAFO)\b", re.IGNORECASE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<t>.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*\*\*(?P<t>[^*]{2,160})\*\*\s*[:.]?\s*$")
# Encabezado de artículo pegado a mitad de un fragmento («… vigentes. Artículo 82.- …»).
# No se separa (el fragmento es atómico): solo se informa.
_INLINE_ARTICLE = re.compile(
    r"(?<=[.;:»”)])[ \t]+(?:\*\*)?\s*(?:Art[íi]culo|Art\.)\s*\d+\s*(?:\*\*)?\s*\.?\s*[-–—]\s",
)


def _plain(line: str) -> str:
    text = re.sub(r"^\s*(?:[-*>]\s+)?#{0,6}\s*", "", line)
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def _ascii_slug(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9.]+", "-", ascii_text).strip("-")


def _article_number(raw: str) -> tuple[str, int | None]:
    slug = _ascii_slug(raw.replace("–", "-"))
    main = re.match(r"\d+", slug)
    return slug, int(main.group()) if main else None


def _article_title(line: str) -> str:
    plain = _plain(line)
    match = _ARTICLE.match(plain)
    rest = plain[match.end():] if match else plain
    rest = rest.lstrip(" .,:;-–—")
    title = re.split(r"\.\s*[-–—]|\.\s|:\s|\s[-–—]\s", rest, maxsplit=1)[0]
    return title.strip(" .-–—:")[:_MAX_PLAIN_HEADING]


def _heading_text(line: str) -> str | None:
    for pattern in (_MD_HEADING, _BOLD_LINE):
        match = pattern.match(line)
        if match:
            return _plain(match.group("t"))
    plain = line.strip()
    if plain and len(plain) <= _MAX_PLAIN_HEADING and _STRUCTURAL.match(plain):
        return _plain(plain)
    return None


def _first_line(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line
    return ""


@dataclass
class _Entry:
    """Elemento de la secuencia de un documento: un fragmento o un encabezado Docling."""

    line: str  # línea que se clasifica (primera línea del fragmento o "## encabezado")
    text: str  # texto que aporta a la unidad
    chunk_id: str | None = None
    chunk_index: int | None = None

    @property
    def is_chunk(self) -> bool:
        return self.chunk_id is not None


@dataclass
class _Unit:
    unit_id: str
    kind: str
    label: str
    title: str
    path: str
    entries: list[_Entry] = field(default_factory=list)


def _accepted_articles(lines: list[str]) -> tuple[dict[int, str], list[str], list[str]]:
    """Devuelve {posición: número aceptado}, candidatos rechazados y saltos de numeración.

    Un salto mayor que ``MAX_ARTICLE_GAP`` se acepta solo si el candidato
    siguiente continúa la nueva numeración (55 → 61 → 62); así se distingue
    un tramo ausente de una cita aislada de otra norma (55 → 95 → 56).
    """
    candidates = []
    for index, line in enumerate(lines):
        match = _ARTICLE.match(line)
        if match:
            slug, number = _article_number(match.group("num"))
            candidates.append((index, slug, number))
    numeric = [number for _, _, number in candidates]
    if len(candidates) < MIN_ARTICLES_FOR_ARTICULATED or 1 not in numeric:
        return {}, [lines[i].strip()[:100] for i, _, _ in candidates], []

    accepted: dict[int, str] = {}
    rejected: list[str] = []
    jumps: list[str] = []
    last: int | None = None
    for position, (index, slug, number) in enumerate(candidates):
        if number is None:  # artículo innumerado o único
            accepted[index] = slug
            continue
        if last is None:
            ok = number == 1
        elif number == 1 or last <= number <= last + MAX_ARTICLE_GAP:
            ok = True
        elif number > last + MAX_ARTICLE_GAP:
            following = next((n for _, _, n in candidates[position + 1:] if n is not None), None)
            ok = following is not None and number < following <= number + MAX_ARTICLE_GAP
            if ok:
                jumps.append(f"{last} → {number}")
        else:
            ok = False
        if ok:
            accepted[index] = slug
            last = number
        else:
            rejected.append(lines[index].strip()[:100])
    return accepted, rejected, jumps


class _Builder:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.units: list[_Unit] = []
        self.warnings: list[str] = []
        self.used_ids: Counter[str] = Counter()
        self.pending: list[_Entry] = []
        self.body_entries = 0

    def start(self, kind: str, local_id: str, label: str, title: str, path: str, entry: _Entry | None) -> None:
        unit_id = f"{self.document_id}-{local_id}"
        self.used_ids[unit_id] += 1
        count = self.used_ids[unit_id]
        if count > 1:
            self.warnings.append(f"ID repetido {unit_id}: se asignó {unit_id}-{count}")
            unit_id = f"{unit_id}-{count}"
        unit = _Unit(unit_id, kind, label, title, path, list(self.pending))
        if entry is not None:
            unit.entries.append(entry)
        self.pending.clear()
        self.units.append(unit)
        self.body_entries = 0

    def append(self, entry: _Entry) -> None:
        unit = self.units[-1]
        unit.entries.extend(self.pending)
        self.pending.clear()
        unit.entries.append(entry)
        self.body_entries += 1

    def finish(self) -> list[_Unit]:
        # A diferencia de la API, los encabezados pendientes al final no se
        # descartan: pueden ser fragmentos y todo fragmento debe tener unidad.
        if self.pending:
            self.units[-1].entries.extend(self.pending)
            self.pending.clear()
        return [unit for unit in self.units if any(e.text.strip() for e in unit.entries)]


def _document_entries(chunks: list[dict[str, Any]]) -> list[_Entry]:
    """Secuencia de encabezados Docling (cuando cambian) y fragmentos, en orden."""
    entries: list[_Entry] = []
    previous: list[str] = []
    for chunk in chunks:
        headings = [str(h).strip() for h in chunk.get("headings") or [] if str(h).strip()]
        common = 0
        while common < min(len(previous), len(headings)) and previous[common] == headings[common]:
            common += 1
        for heading in headings[common:]:
            entries.append(_Entry(line=f"## {heading}", text=heading))
        previous = headings
        text = chunk.get("text") or ""
        entries.append(_Entry(_first_line(text), text, chunk["chunk_id"], chunk.get("chunk_index")))
    return entries


def _parse_document(document_id: str, entries: list[_Entry]) -> tuple[str, list[_Unit], list[str], list[str], list[str]]:
    lines = [entry.line for entry in entries]
    article_lines, rejected, jumps = _accepted_articles(lines)
    mode = "articulado" if article_lines else "secciones"

    builder = _Builder(document_id)
    builder.start("preambulo", "preambulo", "Preámbulo", "", "", None)
    path_parts: dict[str, str] = {}
    disposition_type: str | None = None
    section_number = 0

    def current_path() -> str:
        order = ("titulo", "capitulo", "seccion", "disposiciones")
        return " › ".join(path_parts[key] for key in order if key in path_parts)

    for index, entry in enumerate(entries):
        line = entry.line
        if index in article_lines:
            disposition_type = None
            path_parts.pop("disposiciones", None)
            number = article_lines[index]
            last = builder.units[-1]
            if last.unit_id == f"{document_id}-art{number}" and builder.body_entries == 0:
                # El mismo artículo como encabezado Docling y como inicio del fragmento.
                builder.append(entry)
                continue
            builder.start("articulo", f"art{number}", f"Art. {number}", _article_title(line), current_path(), entry)
            continue

        heading = _heading_text(line)
        dispositions = _DISPOSITIONS.search(line) if len(line.strip()) <= _MAX_PLAIN_HEADING else None
        if dispositions and (heading or len(_plain(line)) <= 60):
            disposition_type = _ascii_slug(dispositions.group("tipo"))
            path_parts["disposiciones"] = _plain(line)
            builder.pending.append(entry)
            continue

        if disposition_type:
            ordinal = _ORDINAL.match(line)
            if ordinal:
                word = ordinal.group("ord")
                builder.start(
                    "disposicion",
                    f"disp-{disposition_type}-{_ascii_slug(word)}",
                    f"Disposición {disposition_type} {word.capitalize()}",
                    "",
                    current_path(),
                    entry,
                )
                continue

        if heading is not None:
            structural = _STRUCTURAL.match(heading)
            if structural:
                kind = _ascii_slug(structural.group("kind")).replace("parrafo", "seccion")
                kind = {"titulo": "titulo", "capitulo": "capitulo"}.get(kind, "seccion")
                path_parts[kind] = heading
                if kind == "titulo":
                    path_parts.pop("capitulo", None)
                    path_parts.pop("seccion", None)
                elif kind == "capitulo":
                    path_parts.pop("seccion", None)
            if mode == "secciones":
                last = builder.units[-1]
                if last.kind == "seccion" and builder.body_entries == 0:
                    last.title = f"{last.title} › {heading}" if last.title else heading
                    last.entries.extend(builder.pending)
                    builder.pending.clear()
                    last.entries.append(entry)
                else:
                    section_number += 1
                    builder.start("seccion", f"sec{section_number:02d}", f"Sección {section_number}", heading, current_path(), entry)
                continue
            if structural:
                builder.pending.append(entry)
                continue

        builder.append(entry)

    return mode, builder.finish(), builder.warnings, rejected, jumps


def _unit_record(document_id: str, mode: str, unit: _Unit) -> dict[str, Any]:
    chunk_entries = [entry for entry in unit.entries if entry.is_chunk]
    text = "\n\n".join(entry.text for entry in unit.entries if entry.text.strip())
    indexes = [entry.chunk_index for entry in chunk_entries if entry.chunk_index is not None]
    return {
        "unit_id": unit.unit_id,
        "document_id": document_id,
        "structure_mode": mode,
        "kind": unit.kind,
        "label": unit.label,
        "title": unit.title,
        "path": unit.path,
        "chunk_ids": [entry.chunk_id for entry in chunk_entries],
        "first_chunk_index": min(indexes) if indexes else None,
        "last_chunk_index": max(indexes) if indexes else None,
        "chunk_count": len(chunk_entries),
        "word_count": len(text.split()),
        "text": text,
    }


def build_units(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Agrupa los fragmentos en unidades normativas.

    Devuelve las unidades (en orden de documento y de fragmento) y un informe
    serializable por documento. No modifica los fragmentos recibidos.
    """
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)

    units: list[dict[str, Any]] = []
    documents_report: dict[str, Any] = {}
    for document_id in sorted(by_document):
        doc_chunks = sorted(by_document[document_id], key=lambda c: c.get("chunk_index") or 0)
        entries = _document_entries(doc_chunks)
        mode, doc_units, warnings, rejected, jumps = _parse_document(document_id, entries)
        records = [_unit_record(document_id, mode, unit) for unit in doc_units]
        units.extend(records)
        inline = [c["chunk_id"] for c in doc_chunks if _INLINE_ARTICLE.search(c.get("text") or "")]
        documents_report[document_id] = {
            "structure_mode": mode,
            "chunks": len(doc_chunks),
            "units": len(records),
            "units_by_kind": dict(Counter(record["kind"] for record in records)),
            "rejected_article_candidates": len(rejected),
            "rejected_samples": rejected[:_SAMPLE_LIMIT],
            "numbering_jumps": jumps,
            "chunks_with_inline_article": len(inline),
            "inline_article_samples": inline[:_SAMPLE_LIMIT],
            "warnings": warnings,
        }

    words = sorted((record["word_count"] for record in units), reverse=True)
    report = {
        "units_version": UNITS_VERSION,
        "documents": len(documents_report),
        "units_count": len(units),
        "units_by_kind": dict(Counter(record["kind"] for record in units)),
        "documents_by_mode": dict(Counter(doc["structure_mode"] for doc in documents_report.values())),
        "max_unit_words": words[0] if words else 0,
        "median_unit_words": words[len(words) // 2] if words else 0,
        "units_over_400_words": sum(w > 400 for w in words),
        "by_document": documents_report,
    }
    report["validation"] = validate_units(chunks, units)
    return units, report


def validate_units(chunks: list[dict[str, Any]], units: list[dict[str, Any]]) -> dict[str, Any]:
    """Comprueba que cada fragmento esté en exactamente una unidad, en rangos contiguos."""
    errors: list[str] = []
    membership = Counter(chunk_id for unit in units for chunk_id in unit["chunk_ids"])
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    known = set(chunk_ids)
    missing = [chunk_id for chunk_id in chunk_ids if membership[chunk_id] == 0]
    repeated = [chunk_id for chunk_id, count in membership.items() if count > 1]
    unknown = [chunk_id for chunk_id in membership if chunk_id not in known]
    duplicated_units = [unit_id for unit_id, count in Counter(u["unit_id"] for u in units).items() if count > 1]

    index_by_id = {chunk["chunk_id"]: chunk.get("chunk_index") for chunk in chunks}
    non_contiguous = []
    for unit in units:
        indexes = [index_by_id.get(chunk_id) for chunk_id in unit["chunk_ids"]]
        if indexes and indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            non_contiguous.append(unit["unit_id"])

    if missing:
        errors.append(f"Fragmentos sin unidad: {len(missing)}")
    if repeated:
        errors.append(f"Fragmentos en más de una unidad: {len(repeated)}")
    if unknown:
        errors.append(f"Unidades con fragmentos inexistentes: {len(unknown)}")
    if duplicated_units:
        errors.append(f"unit_id duplicados: {len(duplicated_units)}")
    if non_contiguous:
        errors.append(f"Unidades con fragmentos no contiguos: {len(non_contiguous)}")

    return {
        "chunks_count": len(chunk_ids),
        "chunks_covered": len(chunk_ids) - len(missing),
        "chunks_without_unit": missing[:_SAMPLE_LIMIT],
        "chunks_in_multiple_units": repeated[:_SAMPLE_LIMIT],
        "duplicate_unit_ids": duplicated_units[:_SAMPLE_LIMIT],
        "non_contiguous_units": non_contiguous[:_SAMPLE_LIMIT],
        "errors": errors,
        "passed": not errors,
    }
