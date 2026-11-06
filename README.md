# RAG ISTAE Ingestion

Ingestión e indexación **offline** del corpus normativo del ISTAE para el sistema RAG del Trabajo de Fin de Máster en Inteligencia Artificial Aplicada.

Este repositorio forma parte de una solución de tres repositorios independientes:

| Repositorio | Responsabilidad |
|---|---|
| `rag_istae_ingestion` (este) | Inventario, extracción, fragmentación, embeddings e indexación en Qdrant. |
| `rag-istae-query` | Consulta de solo lectura: recuperación híbrida, RRF, reranking, Context Builder y generación con Gemini. |
| `rag-istae-resources` | Almacenamiento compartido: modelos, corpus, salidas de ingestión y persistencia Qdrant. |

Este repositorio no contiene lógica de consulta. Los scripts de búsqueda e inspección que antes vivían aquí se trasladaron a `rag-istae-query`.

## Instalación

1. Cree un entorno virtual con Python 3.11 e instale `pip install -r requirements.txt` (`docling==2.144.0`, `sentence-transformers`, `qdrant-client`, `torch`).
2. Prepare y valide `rag-istae-resources`.
3. Copie `config/config.example.json` a `config/config.json` y `config/rag_config.example.json` a `config/rag_config.json`. Asigne **rutas absolutas** al almacenamiento compartido (`rag-istae-resources/storage`). Ambos archivos locales están excluidos de git.

Entorno de referencia: Windows, PowerShell, inferencia en CPU, Qdrant local con persistencia en disco.

## Flujo

```text
storage/corpus (PDF, DOCX)
      │  run_pipeline.py
      ▼
inventario + hash ─► extracción Docling (OCR condicional) ─► HierarchicalChunker ─► validación
      │
      ▼
storage/outputs/chunks.jsonl (+ inventory, document_results, document_mapping, run_manifest, extracted_text/)
      │  scripts/index_vectors.py
      ▼
Qdrant: colección istae_chunks_e5_base (E5 768 d, coseno)
      │  scripts/index_qdrant_bm25.py
      ▼
índice de texto nativo de Qdrant sobre payload.text (usado por BM25 en rag-istae-query)
```

### 1. Ingestión documental

```powershell
python .\run_pipeline.py --config .\config\config.json
```

El pipeline (`src/ingestion/`) no genera embeddings ni se conecta a Qdrant:

| Módulo | Función |
|---|---|
| `inventory.py` | Recorre `corpus_root`, excluye `exclude_directory_names` y calcula SHA-256 de los archivos con `inventory_extensions`. |
| `extract.py` | Convierte PDF y DOCX con Docling. Los PDF se convierten primero sin OCR; si el total de caracteres o el promedio por página queda bajo `min_text_chars_before_ocr` / `min_chars_per_page_before_ocr`, se repite la conversión con OCR. |
| `chunk.py` | Fragmenta el documento estructurado de Docling con `HierarchicalChunker` (no el Markdown exportado). Descarta fragmentos vacíos. |
| `validate.py` | Validación técnica: identificadores ausentes o duplicados, metadatos obligatorios, tipos, coherencia documento–fragmentos. No demuestra fidelidad jurídica del texto extraído. |
| `pipeline.py` | Orquesta las etapas, procesa cada documento de forma aislada (un error no detiene al resto) y escribe las salidas y el manifiesto. |

Solo se extraen `.pdf` y `.docx` (`include_extensions`); `.xlsx`, `.xls` y `.doc` se inventarían pero no se procesan.

**Identificadores.** `document_id` se toma del prefijo `EXT##` / `INT##` del nombre del archivo (si no existe, se usa el nombre sin extensión). `chunk_id` es `<document_id>:<chunk_index con 5 dígitos>`, por ejemplo `INT08:00055`; `chunk_index` empieza en 1 y es consecutivo dentro del documento.

**Campos de cada chunk** (`chunks.jsonl`): `chunk_id`, `document_id`, `source_file` (ruta relativa), `chunk_index`, `headings`, `page_numbers`, `item_types`, `text`, `char_count`, `chunking_method` (`docling_hierarchical`), `source_sha256`, `extraction_method`, `ocr_attempted`, `ocr_enabled`, `ocr_reason`. En DOCX `page_numbers` y `headings` suelen quedar vacíos.

**Salidas** en `output_dir`:

| Archivo | Contenido |
|---|---|
| `inventory.jsonl` | Archivos inventariados con hash y tamaño. |
| `document_results.jsonl` | Estado, métricas de extracción/OCR y tiempos por documento. |
| `chunks.jsonl` | Fragmentos con metadatos. |
| `document_mapping.jsonl` | `document_id` ↔ ruta relativa ↔ hash. |
| `extracted_text/<sha256>.txt` | Texto extraído por documento. |
| `run_manifest.json` | Resultado de la validación, versiones, configuración y tiempos de la ejecución. |

> **Nota.** `corpus_root` apunta a `rag-istae-resources/storage/corpus`, que también contiene la carpeta `GOLD STANDARD` (DOCX/XLSX de evaluación). Por eso `GOLD STANDARD` figura en `exclude_directory_names` (en `config.json` y en el ejemplo); la comparación no distingue mayúsculas. No retire esa exclusión: el acta de validación se procesaría como un documento del corpus.

### 2. Indexación vectorial

```powershell
python .\scripts\index_vectors.py --config .\config\rag_config.json --offline
```

- Codifica cada chunk como `passage: <texto>` con `intfloat/multilingual-e5-base` (CPU, `max_seq_length` 512, embeddings normalizados).
- El ID del punto es un UUID5 determinista derivado de `chunk_id`, por lo que la indexación es reproducible.
- El payload es el chunk completo más `_index_metadata` (`embedding_model`, `embedding_dimensions`, `corpus_sha256`).
- `--offline` exige que los pesos existan localmente.
- Si la colección ya existe, el script no la reconstruye. `--recreate` (o `recreate_collection: true`) **borra y recrea** la colección: no lo use sobre la colección validada.

### 3. Índice de texto para BM25

```powershell
python .\scripts\index_qdrant_bm25.py --config .\config\rag_config.json
```

Crea sobre la colección existente un índice de payload de texto en `text` (tokenizador `word`, minúsculas, `scoring` habilitado). Este índice es el que usa `models.TextQuery` en `rag-istae-query`. No confundir con BM25 standalone (`rank-bm25`), que ya no forma parte del proyecto.

No ejecute la indexación mientras `rag-istae-query` tenga abierta la persistencia Qdrant local.

## Estado actual

Última ejecución validada (`run_manifest.json`, `pipeline_version` 0.3.0, `passed: true`):

| Indicador | Resultado |
|---|---:|
| Archivos inventariados | 47 |
| Documentos procesados (13 EXT + 33 INT) | 46 |
| Archivos XLSX solo inventariados | 1 |
| Chunks generados | 9.623 |
| Fallos de procesamiento | 0 |

Colección Qdrant `istae_chunks_e5_base`: 9.623 puntos, 768 dimensiones, distancia coseno, índice de texto sobre `text`.

## Pruebas

Los archivos de `tests/` son en su mayoría scripts exploratorios de la etapa de selección del motor de extracción (Docling, Docling híbrido, Unstructured hi-res, Unstructured + LlamaIndex) y requieren el corpus y `config/config.json`. `test_chunk.py` corresponde a una versión anterior del chunker basada en texto y no es compatible con la firma actual de `chunk_document`. No existe todavía una suite de regresión automatizada para este repositorio.

## Decisiones que deben preservarse

- Modelo `intfloat/multilingual-e5-base`, prefijos `passage:` / `query:`, 768 dimensiones y normalización.
- Colección `istae_chunks_e5_base`, identificadores `chunk_id` y estructura del payload: `rag-istae-query` depende de `document_id` + `chunk_index` consecutivos para expandir vecinos.
- Índice de texto nativo de Qdrant para BM25.
- No reindexar el corpus ni recrear la colección sin una decisión explícita; cualquier cambio de modelo requiere evaluación comparativa.
- No mezclar ingestión offline con consulta online.
- Los scripts no descargan modelos; los modelos se gestionan en `rag-istae-resources`.
