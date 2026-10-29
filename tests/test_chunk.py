import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.chunk import chunk_document


def test_article_chunking():
    text = "CAPÍTULO I\nDISPOSICIONES GENERALES\n\nArtículo 1.- Objeto\nTexto del artículo uno.\n\nArtículo 2.- Ámbito\nTexto del artículo dos."
    chunks = chunk_document(text, "TEST", "test.docx", max_chars=1800, overlap_chars=100)
    assert len(chunks) == 2
    assert "Artículo 1" in chunks[0]["article_heading"]
    assert "Artículo 2" in chunks[1]["article_heading"]


def test_fallback_is_marked():
    chunks = chunk_document("Primer párrafo.\n\nSegundo párrafo.", "TEST", "test.pdf")
    assert chunks
    assert chunks[0]["structure_detection"] == "fallback_paragraphs"
