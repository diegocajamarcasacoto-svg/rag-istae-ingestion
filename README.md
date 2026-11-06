# RAG ISTAE Ingestion

Ingestión e indexación **offline** del corpus normativo del ISTAE para el sistema RAG del Trabajo de Fin de Máster en Inteligencia Artificial Aplicada.

Este repositorio forma parte de una solución de repositorios independientes:

| Repositorio | Responsabilidad |
|---|---|
| `rag_istae_ingestion` (este) | Inventario, extracción, fragmentación, embeddings e indexación en Qdrant. |
| `rag-istae-query` | Consulta de solo lectura: recuperación híbrida, RRF, reranking, Context Builder y generación (Gemini u OpenRouter). |
| `rag-istae-resources` | Almacenamiento compartido: modelos, corpus, salidas de ingestión y persistencia Qdrant. |
| `rag-istae-api` | Backend FastAPI del chat (repositorio externo, de solo lectura para este proyecto). Arquitectura de contexto largo sobre el texto extraído por esta ingestión. |
| `rag-istae-web` | Interfaz Next.js 16 del chat (repositorio externo, de solo lectura). Consume `rag-istae-api` por HTTP (`NEXT_PUBLIC_API_URL`); no accede a ningún dato de esta ingestión. |

**Consumidor del texto extraído:** `rag-istae-api` no usa Qdrant ni `chunks.jsonl`. Trabaja sobre una copia de `outputs/document_mapping.jsonl` y `outputs/extracted_text/<sha256>.txt` en su carpeta `data/`; al 2026-11-06 sus 46 archivos de texto son idénticos a los de `storage/outputs`. Si se vuelve a ejecutar la ingestión y cambia el texto extraído, esa copia queda desactualizada y debe regenerarse en `rag-istae-api` (por sus responsables). No cambie el formato de esos dos archivos sin coordinarlo.

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
inventario + hash ─► extracción Docling (OCR condicional) ─► HierarchicalChunker ─► unidades (artículos) ─► validación
      │
      ▼
storage/outputs/chunks.jsonl + units.jsonl (+ inventory, document_results, document_mapping, units_report, run_manifest, extracted_text/)
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
| `units.py` | Agrupa los fragmentos consecutivos de cada documento en su unidad normativa (artículo, disposición, sección o preámbulo). No corta ni modifica texto. Ver «Unidades». |
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
| `units.jsonl` | Unidades normativas: `unit_id`, `kind`, `label`, `title`, `path`, `chunk_ids` (rango contiguo), índices, `word_count` y `text`. Es el índice chunk → artículo que usa `rag-istae-query`. |
| `units_report.json` | Unidades por documento y modo, candidatos de artículo rechazados, saltos de numeración, artículos pegados dentro de un fragmento y validación de cobertura. |
| `run_manifest.json` | Resultado de la validación, versiones, configuración y tiempos de la ejecución. |

> **Nota.** `corpus_root` apunta a `rag-istae-resources/storage/corpus`, que también contiene la carpeta `GOLD STANDARD` (DOCX/XLSX de evaluación). Por eso `GOLD STANDARD` figura en `exclude_directory_names` (en `config.json` y en el ejemplo); la comparación no distingue mayúsculas. No retire esa exclusión: el acta de validación se procesaría como un documento del corpus.

### Unidades (artículos)

La recuperación trabaja sobre los fragmentos finos de Docling (mediana de 31 palabras), y la consulta devuelve el **artículo completo** al que pertenecen (*small-to-big*). La etapa `units.py` arma ese índice **sin escribir un fragmentador propio**: Docling sigue decidiendo dónde se corta el texto. Medido sobre la ingesta validada, ningún fragmento cruza un límite de artículo y 1.946 fragmentos empiezan con «Art. N». Solo hace falta agrupar los consecutivos.

- **Modo `articulado`:** un fragmento que empieza con «Artículo N» / «Art. N», o un encabezado Docling con esa forma, abre `<doc>-artN`, y los siguientes se agregan hasta la próxima apertura. Solo se aceptan números en secuencia (el primero es 1, saltos de hasta 5 o reinicio en 1), para no confundir menciones a artículos de otras normas.
- **Disposiciones:** las generales, transitorias, derogatorias, finales y reformatorias abren `<doc>-disp-<tipo>-<ordinal>`.
- **Modo `secciones`:** en las normas sin articulado, cada encabezado abre `<doc>-secNN`. Lo anterior al primer artículo queda en `<doc>-preambulo`.
- **Encabezados de Docling:** cuando cambian, se incorporan al texto de la unidad, porque a veces contienen el propio artículo. Fuera de eso, el texto de la unidad es la concatenación de sus fragmentos.
- **Validación:** cada fragmento pertenece exactamente a una unidad, en un rango contiguo, y los `unit_id` son únicos. Un fallo marca la ejecución como no aprobada.
- **IDs:** usan las mismas reglas y el mismo esquema que `rag-istae-api` (`INT08-art12`), para que las arquitecturas RAG citen como las de contexto largo. Resultado sobre la ingesta actual:
  - 2.748 unidades: 1.982 artículos, 314 disposiciones, 413 secciones y 39 preámbulos;
  - **todas existen en la API** (precisión 1,0) y cubren el **97,9 %** de las 2.807 unidades de la API.
- **Lo que falta (59 unidades)** son artículos que Docling dejó dentro de otro fragmento: un «Art. 12.-» dentro de una lista, o pegado tras un punto. Su texto queda en la unidad anterior. Esos casos se informan en `units_report.json` (`chunks_with_inline_article`).

`run_pipeline.py` genera las unidades después del chunking; se desactiva con `"build_units": false`. Para una ingesta ya existente, sin volver a extraer ni fragmentar:

```powershell
python .\scripts\build_units.py --config .\config\config.json
python .\scripts\compare_units_with_api.py --config .\config\config.json --api-root <ruta absoluta a rag-istae-api>   # opcional, solo lectura
```

`compare_units_with_api.py` ejecuta el parser de la API sobre su `data/` en un proceso aparte, sin escribir bytecode ni archivos en ese repositorio, y deja el detalle en `units_api_comparison.json`.

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

Los archivos de `tests/` son en su mayoría scripts exploratorios de la etapa de selección del motor de extracción (Docling, Docling híbrido, Unstructured hi-res, Unstructured + LlamaIndex) y requieren el corpus y `config/config.json`. `test_chunk.py` corresponde a una versión anterior del chunker basada en texto y no es compatible con la firma actual de `chunk_document`. La primera suite de regresión es `tests/test_units.py` (unittest, 14 pruebas con fragmentos sintéticos y el pipeline simulado; no requiere corpus ni Docling real):

```powershell
python -m unittest tests.test_units -v
```

## Decisiones que deben preservarse

- Modelo `intfloat/multilingual-e5-base`, prefijos `passage:` / `query:`, 768 dimensiones y normalización.
- Colección `istae_chunks_e5_base`, identificadores `chunk_id` y estructura del payload: `rag-istae-query` depende de `document_id` + `chunk_index` consecutivos para expandir vecinos.
- Índice de texto nativo de Qdrant para BM25.
- No reindexar el corpus ni recrear la colección sin una decisión explícita; cualquier cambio de modelo requiere evaluación comparativa.
- No mezclar ingestión offline con consulta online.
- Los scripts no descargan modelos; los modelos se gestionan en `rag-istae-resources`.
- Las unidades se derivan de los fragmentos de Docling; no se escribe un fragmentador propio. `units.jsonl` no altera `chunks.jsonl`, los `chunk_id` ni Qdrant.
- **Sin limpieza de texto (decisión del 2026-11-06).** No se limpia el texto, ni en una nueva ingesta ni en `units.jsonl`.
  - **Ruido medido** en los 9.623 fragmentos:
    - 67 con rayas `_____`/`-----`/`.....` (≈13.000 caracteres, 0,4 % del corpus);
    - 14 sin letras ni números;
    - 158 con membretes repetidos (`UBICACIÓN:`, `VERSIÓN:`, `ÁREA:`…);
    - 10 con solo un número de página;
    - ningún marcador de imagen ni carácter de control.
  - **Motivos:**
    1. el impacto es bajo, y el problema de calidad relevante (la fragmentación) lo resuelven las unidades;
    2. una re-ingesta cambiaría los `chunk_id` e invalidaría el Gold Standard anotado por `chunk_id_esperado` y las mediciones hechas;
    3. obligaría a rehacer y revalidar la colección y la copia de datos de `rag-istae-api`;
    4. cambiar la base a mitad de la evaluación rompe la comparabilidad.
  - **Se reconsidera** si la evaluación muestra ruido en el top-k que perjudique la recuperación, o si un cambio del corpus obliga a reingestar. En ese caso, la limpieza se haría sobre el DoclingDocument antes del chunking.
