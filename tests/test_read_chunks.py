import json
import tempfile
import unittest
from pathlib import Path

from scripts.utils import read_chunks_generic


class TestReadChunksGeneric(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "chunks.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preserves_records_and_skips_blank_lines(self):
        record = {"chunk_id": "doc:1", "text": "Contenido", "custom": {"keep": True}}
        self.path.write_text(
            "\ufeff" + json.dumps(record, ensure_ascii=False) + "\n\n",
            encoding="utf-8",
        )

        self.assertEqual(read_chunks_generic(self.path), [record])

    def test_rejects_invalid_json(self):
        self.path.write_text("{invalid}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "JSON inválido en línea 1"):
            read_chunks_generic(self.path)

    def test_rejects_invalid_records(self):
        cases = [
            ([{"text": "Contenido"}], "Falta chunk_id"),
            ([{"chunk_id": "doc:1", "text": " "}], "Texto vacío o inválido"),
            (
                [
                    {"chunk_id": "doc:1", "text": "Uno"},
                    {"chunk_id": "doc:1", "text": "Dos"},
                ],
                "chunk_id duplicado",
            ),
        ]
        for records, message in cases:
            with self.subTest(message=message):
                self.path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_chunks_generic(self.path)

    def test_rejects_empty_file(self):
        self.path.write_text("\n \n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "No se encontraron chunks"):
            read_chunks_generic(self.path)
