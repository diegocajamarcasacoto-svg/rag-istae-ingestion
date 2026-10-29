# Pipeline de ingesta reproducible — RAG normativo ISTAE

**Versión del pipeline:** `0.2.0`  
**Entorno:** Windows, Python 3.11.13  
**Motor de extracción:** Docling 2.144.0  
**Motor de fragmentación:** `HierarchicalChunker`  
**Coste de software:** cero, utilizando herramientas de código abierto.

El objetivo de esta etapa es preparar el corpus normativo del ISTAE para los experimentos posteriores de recuperación aumentada por generación (RAG).

La ingesta trabaja sobre los documentos originales sin modificarlos, renombrarlos ni moverlos. Los resultados se guardan en archivos derivados dentro de `outputs/`.

**Estado actual:** extracción y fragmentación implementadas; validación técnica ampliada; ejecución completa del corpus pendiente. Embeddings, índice vectorial, recuperación y generación todavía no están implementados.

## Qué hace

1. Inventaría los documentos y archivos de apoyo, registrando ruta relativa, extensión, tamaño, fecha de modificación y hash SHA-256.
2. Extrae PDF y DOCX mediante Docling.
3. Mantiene habilitada la detección de estructura de tablas durante la conversión de PDF.
4. Ejecuta primero la extracción sin OCR y repite la conversión con OCR cuando el texto extraído resulta insuficiente según los umbrales configurados.
5. Genera fragmentos mediante `HierarchicalChunker`, conservando la estructura disponible en el documento Docling.
6. Asigna identificadores de documento y fragmento, y conserva metadatos de procedencia, estructura y extracción.
7. Valida IDs, metadatos obligatorios, texto, longitudes, tipos de datos y coherencia entre documentos y fragmentos.
8. Genera archivos JSONL, un manifiesto de ejecución y archivos de texto extraído.
9. Permite repetir la ingesta regenerando las salidas derivadas, sin alterar los documentos originales.

La detección de tablas no garantiza que todas sus celdas se reconstruyan correctamente. Las advertencias de Docling deben revisarse cuando puedan afectar al contenido normativo.

## 1. Requisitos e instalación

Se utiliza Python 3.11 en un entorno virtual dedicado.

Desde PowerShell, en la carpeta raíz del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, se puede invocar directamente su intérprete:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La dependencia directa principal de la versión actual es:

```text
docling==2.144.0
```

El archivo `requirements.txt` debe reflejar las dependencias directas que utiliza el código vigente. No debe copiarse automáticamente todo el contenido de `pip freeze`, porque puede incluir herramientas instaladas para experimentos anteriores.

### Registro del entorno experimental

Se conserva `requirements-lock.txt` como registro de las versiones instaladas durante el entorno experimental:

```powershell
python -m pip freeze > requirements-lock.txt
```

Este archivo es una fotografía del entorno, no una lista depurada de dependencias directas.

Antes de recrear el entorno virtual, se recomienda conservar el registro y comprobar que los modelos necesarios de Docling estén disponibles localmente.

### Modelos de Docling

Los modelos se descargaron previamente en la carpeta local:

```text
models/docling/
```

La disponibilidad de los modelos debe comprobarse en el entorno limpio antes de procesar el corpus completo. No se debe asumir que una instalación nueva puede ejecutarse sin conexión hasta verificarlo.

## 2. Configurar el corpus

La configuración se encuentra en:

```text
config/config.json
```

La propiedad `corpus_root` debe apuntar a la carpeta que contiene las subcarpetas del corpus, por ejemplo:

```json
{
  "corpus_root": "E:\\UIDE\\Maestria\\TMF\\Entregable 2\\app\\CORPUS",
  "output_dir": "outputs",
  "include_extensions": [
    ".pdf",
    ".docx"
  ],
  "inventory_extensions": [
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".doc"
  ],
  "exclude_directory_names": [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "outputs",
    "build",
    "dist"
  ],
  "min_text_chars_before_ocr": 100,
  "min_chars_per_page_before_ocr": 100
}
```

Este bloque representa la configuración de referencia para la estrategia actual. Debe conservarse cualquier otro parámetro requerido por el código vigente.

### Inventario y extracción

- `inventory_extensions`: extensiones que se registran en el inventario.
- `include_extensions`: extensiones que se procesan mediante Docling.
- `exclude_directory_names`: directorios excluidos de la exploración.
- `min_text_chars_before_ocr`: umbral mínimo de caracteres totales para evaluar la extracción de un PDF.
- `min_chars_per_page_before_ocr`: umbral mínimo del promedio de caracteres por página.

El inventario puede contener archivos que no se extraen, como XLSX, XLS y DOC. Por tanto, el número de documentos inventariados no tiene que coincidir necesariamente con el número de documentos procesados.

**Importante:** la ruta del corpus debe ser lo suficientemente específica para evitar incluir documentos ajenos al experimento. No es necesario mover ni borrar los originales para resolver este problema.

## 3. Estrategia de extracción y OCR

La extracción documental utiliza Docling.

### PDF

1. Se ejecuta una primera conversión sin OCR.
2. Se calcula la cantidad total de caracteres y, cuando se dispone del número de páginas, el promedio de caracteres por página.
3. Si se incumple alguno de los umbrales configurados, se repite la conversión con OCR.
4. Se registran el resultado final, el uso de OCR y el motivo de activación.

El promedio de caracteres por página es una heurística. No demuestra que todas las páginas contengan texto ni que el texto sea correcto.

### DOCX

Los documentos DOCX se procesan mediante Docling sin OCR.

### Tablas

La detección de estructura de tablas permanece habilitada en las conversiones de PDF, incluidas las que usan OCR.

Docling puede advertir que algunas celdas no encajan en la cuadrícula reconstruida y que han sido descartadas. Estas advertencias requieren revisión cuando afectan a tablas con datos, relaciones entre columnas o información normativa.

No se considera validada una tabla únicamente porque aparezca en el Markdown exportado.

## 4. Fragmentación y metadatos

La fragmentación utiliza `HierarchicalChunker` sobre el documento estructurado de Docling.

No se aplica una fragmentación jurídica personalizada por artículos en esta etapa. La calidad de la estructura jerárquica debe evaluarse mediante inspección de muestras.

### Esquema de los fragmentos

Cada registro de `chunks.jsonl` contiene los siguientes campos:

| Campo | Descripción |
|---|---|
| `chunk_id` | Identificador del fragmento, compuesto por el código documental y su índice. |
| `document_id` | Identificador del documento, preferiblemente su código estable, por ejemplo `EXT07`. |
| `source_file` | Ruta relativa del documento de origen. |
| `chunk_index` | Índice del fragmento dentro del documento. |
| `headings` | Encabezados estructurales disponibles. |
| `page_numbers` | Números de página disponibles en la procedencia de Docling. |
| `item_types` | Tipos de elementos detectados, como texto o tabla. |
| `text` | Contenido textual del fragmento. |
| `char_count` | Número de caracteres de `text`. |
| `chunking_method` | Método de fragmentación utilizado. |
| `source_sha256` | Hash SHA-256 del documento de origen. |
| `extraction_method` | Motor de extracción utilizado. |
| `ocr_attempted` | Indica si se ejecutó la conversión de respaldo con OCR. |
| `ocr_enabled` | Indica si la conversión asociada utilizó OCR. |
| `ocr_reason` | Motivo de activación del OCR, cuando corresponde. |

### Identificación documental

Se utiliza el código documental como identificador cuando puede extraerse de forma inequívoca del nombre del archivo:

```text
EXT07:00001
EXT07:00002
EXT07:00003
```

El `document_id` identifica el documento dentro del corpus, mientras que `source_sha256` identifica la versión exacta del archivo.

El `chunk_id` no identifica por sí solo una versión del contenido. Si se modifica el documento o la fragmentación, un mismo ID podría corresponder a texto