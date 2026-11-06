"""Pruebas de la agrupación de fragmentos en unidades normativas (units.py).

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_units -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion.units import build_units, validate_units  # noqa: E402


def _chunks(document_id: str, texts: list[str | tuple[str, list[str]]]) -> list[dict]:
    """Fragmentos sintéticos; cada elemento es texto o (texto, headings)."""
    result = []
    for index, item in enumerate(texts, start=1):
        text, headings = (item, []) if isinstance(item, str) else item
        result.append({
            "chunk_id": f"{document_id}:{index:05d}",
            "document_id": document_id,
            "chunk_index": index,
            "headings": headings,
            "text": text,
        })
    return result


def _by_id(units: list[dict]) -> dict[str, dict]:
    return {unit["unit_id"]: unit for unit in units}


class ArticuladoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = _chunks("INT08", [
            "**REGLAMENTO DE TITULACIÓN**",
            "Considerando: que el Estatuto establece…",
            "Artículo 1.- Objeto. El presente reglamento regula…",
            "**Artículo 2. Ámbito.** Se aplica a…",
            "- a) Estudiantes.",
            "- b) Docentes.",
            "Artículo **3. Definiciones.** Para efectos…",
            "Conforme al Art. 357 de la Constitución…",
            "Art. 95.- Cita de otra norma que no sigue la numeración.",
            "Artículo 4.- Vigencia.",
        ])

    def test_groups_consecutive_chunks_by_article(self) -> None:
        units, report = build_units(self.chunks)
        units = _by_id(units)
        self.assertEqual(
            list(units),
            ["INT08-preambulo", "INT08-art1", "INT08-art2", "INT08-art3", "INT08-art4"],
        )
        self.assertEqual(units["INT08-art2"]["chunk_ids"], ["INT08:00004", "INT08:00005", "INT08:00006"])
        self.assertEqual(units["INT08-art2"]["title"], "Ámbito")
        self.assertEqual(units["INT08-preambulo"]["chunk_ids"], ["INT08:00001", "INT08:00002"])
        self.assertEqual(report["by_document"]["INT08"]["structure_mode"], "articulado")

    def test_out_of_sequence_article_stays_in_current_unit(self) -> None:
        units, report = build_units(self.chunks)
        art3 = _by_id(units)["INT08-art3"]
        self.assertIn("INT08:00009", art3["chunk_ids"])
        self.assertEqual(report["by_document"]["INT08"]["rejected_article_candidates"], 1)

    def test_text_is_concatenation_of_chunks_without_changes(self) -> None:
        units, _ = build_units(self.chunks)
        art2 = _by_id(units)["INT08-art2"]
        expected = "\n\n".join(c["text"] for c in self.chunks[3:6])
        self.assertEqual(art2["text"], expected)
        self.assertEqual(art2["first_chunk_index"], 4)
        self.assertEqual(art2["last_chunk_index"], 6)
        self.assertEqual(art2["chunk_count"], 3)

    def test_every_chunk_in_exactly_one_contiguous_unit(self) -> None:
        units, report = build_units(self.chunks)
        self.assertTrue(report["validation"]["passed"], report["validation"]["errors"])
        flattened = [chunk_id for unit in units for chunk_id in unit["chunk_ids"]]
        self.assertEqual(flattened, [c["chunk_id"] for c in self.chunks])

    def test_input_chunks_are_not_modified(self) -> None:
        snapshot = json.dumps(self.chunks, sort_keys=True)
        build_units(self.chunks)
        self.assertEqual(json.dumps(self.chunks, sort_keys=True), snapshot)


class DoclingHeadingsTests(unittest.TestCase):
    def test_article_in_docling_heading_opens_unit_and_keeps_heading_text(self) -> None:
        chunks = _chunks("EXT01", [
            "Art. 1.- Objeto.",
            ("Nota: Artículo derogado por Ley…", ["Art. 2.- La autoevaluación.-"]),
            ("Art. 3.- Reglamento.- El Consejo…", ["Art. 2.- La autoevaluación.-"]),
            ("En cada proceso…", ["Art. 2.- La autoevaluación.-"]),
        ])
        units = _by_id(build_units(chunks)[0])
        self.assertEqual(units["EXT01-art2"]["chunk_ids"], ["EXT01:00002"])
        self.assertTrue(units["EXT01-art2"]["text"].startswith("Art. 2.- La autoevaluación.-"))
        # El encabezado repetido (sin cambio) no abre otra unidad.
        self.assertEqual(units["EXT01-art3"]["chunk_ids"], ["EXT01:00003", "EXT01:00004"])

    def test_same_article_as_heading_and_chunk_start_is_one_unit(self) -> None:
        chunks = _chunks("INT02", [
            "Artículo 1.- Objeto.",
            ("Artículo 2.- Ámbito. Se aplica…", ["Artículo 2.- Ámbito"]),
            "Artículo 3.- Fin.",
        ])
        units, report = build_units(chunks)
        ids = [unit["unit_id"] for unit in units]
        self.assertEqual(ids, ["INT02-art1", "INT02-art2", "INT02-art3"])
        self.assertEqual(report["by_document"]["INT02"]["warnings"], [])


class DisposicionesTests(unittest.TestCase):
    def test_dispositions_open_their_own_units(self) -> None:
        chunks = _chunks("INT10", [
            "Artículo 1.- Objeto.",
            "Artículo 2.- Ámbito.",
            "Artículo 3.- Vigencia.",
            "**DISPOSICIONES GENERALES**",
            "PRIMERA.- Lo no previsto será resuelto por el OCS.",
            "SEGUNDA.- Se deroga el reglamento anterior.",
            "**DISPOSICIÓN TRANSITORIA**",
            "ÚNICA.- En el plazo de 30 días…",
        ])
        units = _by_id(build_units(chunks)[0])
        self.assertEqual(
            list(units),
            [
                "INT10-art1", "INT10-art2", "INT10-art3",
                "INT10-disp-general-primera", "INT10-disp-general-segunda",
                "INT10-disp-transitoria-unica",
            ],
        )
        # El encabezado de las disposiciones queda en la primera disposición.
        self.assertEqual(units["INT10-disp-general-primera"]["chunk_ids"], ["INT10:00004", "INT10:00005"])
        self.assertEqual(units["INT10-disp-transitoria-unica"]["chunk_ids"], ["INT10:00007", "INT10:00008"])


class SeccionesTests(unittest.TestCase):
    def test_document_without_articles_groups_by_headings(self) -> None:
        chunks = _chunks("INT09", [
            ("Introducción al modelo.", ["OBJETIVO"]),
            ("Texto del objetivo.", ["OBJETIVO"]),
            ("Paso 1…", ["PROCEDIMIENTO"]),
            ("Paso 2…", ["PROCEDIMIENTO"]),
        ])
        units, report = build_units(chunks)
        units = _by_id(units)
        self.assertEqual(report["by_document"]["INT09"]["structure_mode"], "secciones")
        self.assertEqual(units["INT09-sec01"]["chunk_ids"], ["INT09:00001", "INT09:00002"])
        self.assertEqual(units["INT09-sec02"]["title"], "PROCEDIMIENTO")
        self.assertTrue(units["INT09-sec02"]["text"].startswith("PROCEDIMIENTO\n\nPaso 1"))

    def test_document_without_headings_is_a_single_preamble_unit(self) -> None:
        chunks = _chunks("INT20", ["Texto uno.", "Texto dos."])
        units, report = build_units(chunks)
        self.assertEqual([u["unit_id"] for u in units], ["INT20-preambulo"])
        self.assertTrue(report["validation"]["passed"])


class ValidationTests(unittest.TestCase):
    def test_detects_missing_repeated_and_non_contiguous(self) -> None:
        chunks = _chunks("INT01", ["a", "b", "c"])
        units = [
            {"unit_id": "INT01-art1", "chunk_ids": ["INT01:00001", "INT01:00003"]},
            {"unit_id": "INT01-art1", "chunk_ids": ["INT01:00003"]},
        ]
        result = validate_units(chunks, units)
        self.assertFalse(result["passed"])
        self.assertEqual(result["chunks_without_unit"], ["INT01:00002"])
        self.assertEqual(result["chunks_in_multiple_units"], ["INT01:00003"])
        self.assertEqual(result["duplicate_unit_ids"], ["INT01-art1"])
        self.assertEqual(result["non_contiguous_units"], ["INT01-art1"])

    def test_multiple_documents_are_independent(self) -> None:
        chunks = _chunks("EXT01", ["Art. 1.- A", "Art. 2.- B", "Art. 3.- C"]) + _chunks(
            "INT01", ["Artículo 1.- X", "Artículo 2.- Y", "Artículo 3.- Z"]
        )
        units, report = build_units(chunks)
        self.assertEqual(len(units), 6)
        self.assertEqual(report["documents_by_mode"], {"articulado": 2})


class PipelineIntegrationTests(unittest.TestCase):
    """run_pipeline escribe units.jsonl y registra la etapa en el manifiesto."""

    def test_pipeline_writes_units_and_manifest(self) -> None:
        from ingestion import pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config" / "config.json"
            config_path.parent.mkdir()
            config_path.write_text(json.dumps({
                "corpus_root": str(root / "corpus"),
                "output_dir": str(root / "outputs"),
            }), encoding="utf-8")
            inventory = [{
                "relative_path": "INTERNOS/INT01 - Reglamento.docx",
                "file_name": "INT01 - Reglamento.docx",
                "extension": ".docx",
                "sha256": "abc123",
                "size_bytes": 10,
            }]
            extracted = {
                "text": "texto", "document": object(), "extracted_chars": 5, "pages": 1,
                "extraction_method": "docling", "ocr_attempted": False, "ocr_enabled": False,
                "warnings": [], "chars_per_page": 5, "ocr_reason": None,
            }

            def fake_chunks(document, document_id, source_file):
                chunks = _chunks(document_id, ["Artículo 1.- A", "Artículo 2.- B", "- a) inciso", "Artículo 3.- C"])
                for chunk in chunks:
                    chunk.update({
                        "source_file": source_file, "page_numbers": [], "item_types": ["text"],
                        "char_count": len(chunk["text"]), "chunking_method": "docling_hierarchical",
                    })
                return chunks

            with mock.patch.object(pipeline, "build_inventory", return_value=inventory), \
                 mock.patch.object(pipeline, "extract_document", return_value=extracted), \
                 mock.patch.object(pipeline, "chunk_document", side_effect=fake_chunks), \
                 mock.patch("builtins.print"):
                report = pipeline.run_pipeline(config_path)

            units = [
                json.loads(line)
                for line in (root / "outputs" / "units.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            units_report = json.loads((root / "outputs" / "units_report.json").read_text(encoding="utf-8"))

        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["units"]["units_count"], 3)
        self.assertTrue(report["units"]["passed"])
        self.assertEqual(units[1]["chunk_ids"], ["INT01:00002", "INT01:00003"])
        self.assertEqual(units_report["validation"]["chunks_covered"], 4)

    def test_build_units_false_skips_stage(self) -> None:
        from ingestion import pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config" / "config.json"
            config_path.parent.mkdir()
            config_path.write_text(json.dumps({
                "corpus_root": str(root / "corpus"),
                "output_dir": str(root / "outputs"),
                "build_units": False,
            }), encoding="utf-8")
            with mock.patch.object(pipeline, "build_inventory", return_value=[]), \
                 mock.patch("builtins.print"):
                report = pipeline.run_pipeline(config_path)
            self.assertFalse((root / "outputs" / "units.jsonl").exists())
        self.assertNotIn("units", report)


if __name__ == "__main__":
    unittest.main()
