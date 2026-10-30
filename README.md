# RAG ISTAE
## Estado del proyecto y contexto técnico

**Proyecto:** Sistema de recuperación aumentada por generación (RAG) sobre corpus normativo del ISTAE.  
**Contexto académico:** Trabajo de Fin de Máster en Inteligencia Artificial Aplicada.  
**Estado:** Prototipo funcional de ingestión, indexación y recuperación híbrida con reranking.  
**Etapa actual:** Evaluación del rendimiento y preparación de una refactorización arquitectónica.

---

# 1. Objetivo del proyecto

Desarrollar un sistema RAG capaz de consultar un corpus documental institucional y normativo del ISTAE, recuperando información relevante mediante técnicas de búsqueda léxica y semántica, y posteriormente utilizando un modelo de lenguaje para generar respuestas fundamentadas en los documentos recuperados.

El proyecto busca establecer una arquitectura modular que permita:

- Procesar documentos heterogéneos.
- Extraer y fragmentar información documental.
- Generar representaciones vectoriales.
- Indexar contenido para recuperación semántica y léxica.
- Ejecutar recuperación híbrida.
- Mejorar la relevancia mediante reranking.
- Construir contextos documentales para el modelo generativo.
- Generar respuestas fundamentadas en el corpus.
- Mantener trazabilidad hacia los documentos originales.

El objetivo de evolución es pasar del prototipo ejecutado mediante scripts independientes a una arquitectura de consulta persistente y preparada para producción.

# 2. Entorno de desarrollo

| Elemento | Configuración |
|---|---|
| Sistema operativo | Windows |
| IDE | Visual Studio Code |
| Lenguaje | Python 3.11 |
| Entorno virtual | `.venv` |
| Ejecución | PowerShell |
| Inferencia | CPU |
| Base vectorial | Qdrant local |
| Modelos | Almacenamiento local |

Ruta del proyecto:

```text
E:\UIDE\Maestria\TMF\Entregable 2\app\rag_istae_ingestion
```

Ruta del corpus:

```text
E:\UIDE\Maestria\TMF\Entregable 2\app\CORPUS
```

Configuración principal:

```text
config/rag_config.json
```

# 3. Estado de la ingestión documental

Se ejecutó el procesamiento del corpus y se generaron los fragmentos documentales utilizados en la indexación.

## 3.1 Inventario

| Indicador | Resultado |
|---|---:|
| Archivos inventariados | 47 |
| Documentos procesados | 46 |
| Archivos XLSX solo inventariados | 1 |
| Chunks generados | 9.623 |
| Fallos de procesamiento | 0 |

El corpus contiene documentos PDF y DOCX, entre otros formatos inventariados.

## 3.2 Archivos de salida de ingestión

Se generaron, entre otros, los siguientes artefactos:

```text
outputs/
    chunks.jsonl
    inventory.jsonl
    document_results.jsonl
    document_mapping.jsonl
    run_manifest.json
    extracted_text/
```

Estos archivos constituyen los resultados de la etapa de ingestión.

# 4. Modelos utilizados

## 4.1 Modelo de embeddings

**Modelo:** `intfloat/multilingual-e5-base`

Ubicación local:

```text
models/multilingual-e5-base
```

Características:

- Dimensión vectorial: 768.
- Prefijo documental: `passage:`.
- Prefijo de consulta: `query:`.
- Normalización de embeddings habilitada.
- Inferencia CPU.
- Longitud máxima configurada según la configuración del proyecto.

Los prefijos son parte fundamental del funcionamiento del modelo y deben preservarse durante cualquier refactorización.

## 4.2 Modelo de reranking

**Modelo:** `BAAI/bge-reranker-v2-m3`

Ubicación local:

```text
models/bge-reranker-v2-m3
```

Implementación mediante `CrossEncoder` de Sentence Transformers.

Responsabilidad:

Evaluar pares consulta-fragmento para volver a ordenar los candidatos recuperados por el sistema híbrido.

Configuración actual:

- Inferencia CPU.
- Carga local.
- Evaluación de candidatos recuperados mediante RRF.

## 4.3 Modelo de procesamiento documental

Ubicación:

```text
models/docling
```

Se mantiene como parte de los recursos locales utilizados por la etapa documental.

# 5. Base de datos vectorial

**Motor:** Qdrant.

**Modo:** Local con persistencia en disco.

Ubicación:

```text
vector_store/qdrant
```

Colección:

```text
istae_chunks_e5_base
```

Características conocidas:

| Propiedad | Valor |
|---|---|
| Puntos indexados | 9.623 |
| Dimensiones | 768 |
| Distancia | Cosine |
| Modelo de embeddings | multilingual-e5-base |

El payload contiene información documental y de fragmentación, incluyendo:

- `chunk_id`
- `document_id`
- `source_file`
- `page_numbers`
- `text`
- `headings`
- `chunk_index`

También se incluye metadata de indexación relacionada con el modelo, dimensiones y hash del corpus.

La colección existente es un recurso persistente validado. No debe eliminarse ni reconstruirse durante una refactorización ordinaria.

# 6. Arquitectura de recuperación implementada

Actualmente el proyecto cuenta con una arquitectura de recuperación híbrida compuesta por cuatro etapas.

## 6.1 Recuperación vectorial

Se utiliza E5 para generar el embedding de la consulta y recuperar fragmentos semánticamente similares desde Qdrant.

## 6.2 Recuperación léxica

Se utiliza BM25 nativo de Qdrant mediante `TextQuery`.

El índice textual se configura sobre el campo:

```text
text
```

La preparación del índice textual se encuentra separada de la ejecución de consultas.

## 6.3 Fusión de resultados

Se utiliza Reciprocal Rank Fusion (RRF) para combinar los resultados de la recuperación vectorial y léxica.

Configuración conocida:

```text
rrf-k = 60
```

La fusión se realiza utilizando las posiciones de los resultados de cada recuperador.

## 6.4 Reranking

Los candidatos resultantes de RRF son evaluados por BGE Reranker.

El modelo vuelve a ordenar los fragmentos en función de su relevancia para la consulta.

La salida actual permite visualizar:

- Posición original en RRF.
- Score RRF.
- Score del reranker.
- Identificador del fragmento.
- Documento de origen.
- Página.
- Texto recuperado.

# 7. Scripts existentes

Los siguientes scripts forman parte de la implementación conocida:

| Archivo | Responsabilidad |
|---|---|
| `scripts/index_vectors.py` | Indexación de embeddings en Qdrant |
| `scripts/index_qdrant_bm25.py` | Preparación del índice textual nativo de Qdrant |
| `scripts/search_qdrant_bm25.py` | Consulta léxica BM25 nativo |
| `scripts/search_hybrid.py` | Recuperación híbrida E5 + BM25 + RRF |
| `scripts/search_reranked.py` | Recuperación híbrida con reranking BGE |
| `scripts/inspect_qdrant.py` | Inspección de colección, fragmentos y vecinos |
| `scripts/search_vectors.py` | Consulta vectorial |
| `scripts/index_bm25.py` | Indexación BM25 standalone |
| `scripts/search_bm25.py` | Consulta BM25 standalone |
| `scripts/utils.py` | Funciones auxiliares |
| `scripts/enable_qdrant_bm25.py` | Script anterior de preparación y consulta BM25 |

La lista debe contrastarse con el inventario real del repositorio antes de reorganizar archivos.

Los scripts standalone BM25 (`rank-bm25`) no deben confundirse con la implementación BM25 nativa de Qdrant.

# 8. Recuperación de fragmentos vecinos

Se implementó y probó una funcionalidad de inspección que permite recuperar fragmentos anteriores y posteriores a uno determinado.

La búsqueda se realiza mediante:

- `document_id`
- `chunk_index`

Ejemplo:

```powershell
python -u .\scripts\inspect_qdrant.py --config .\config\rag_config.json --chunk-id INT08:00055 --neighbors 1
```

La funcionalidad permite recuperar contexto adicional del mismo documento.

**Decisión arquitectónica actual:**

La recuperación de vecinos no se ha integrado todavía en el pipeline de reranking.

Se considera una responsabilidad posterior, correspondiente a la construcción del contexto que será enviado al LLM.

# 9. Validaciones funcionales realizadas

Se han ejecutado pruebas con consultas normativas y académicas.

## 9.1 Consulta normativa

Pregunta:

```text
Art. 357 Constitución de la República
```

El recuperador BM25 identificó como resultado relevante:

```text
EXT01:00089
```

Correspondiente a la LOES, página 6, con contenido sustantivo sobre el financiamiento de las instituciones públicas de educación superior.

El reranker BGE ubicó este fragmento en primera posición.

## 9.2 Consulta sobre egreso y titulación

Pregunta:

```text
¿Qué requisitos necesito cumplir para que me declaren egresado y luego poder titularme?
```

Se identificaron los siguientes fragmentos relacionados:

| Chunk | Contenido |
|---|---|
| INT08:00039 | Artículo 12. Declaración de egresado |
| INT08:00040 | Lista de requisitos para declaración de egresado |
| INT08:00041 | Solicitud de declaración de egresado y condiciones |
| INT08:00042 | Artículo 13. Requisitos para titularse |

Se identificó una limitación importante:

El fragmento `INT08:00040`, que contiene la lista específica de requisitos de egreso, no apareció entre los 20 candidatos recuperados por RRF.

Esto demuestra que el reranker no puede recuperar información que no está presente en su conjunto inicial de candidatos.

La mejora de cobertura debe evaluarse desde la recuperación inicial, no solamente desde el reranking.

## 9.3 Consulta compuesta

Pregunta:

```text
¿Puedo hacer el trabajo de integración curricular con un compañero y cuál es la nota mínima para aprobarlo?
```

Los resultados incluyeron fragmentos relacionados con:

- Trabajo individual o en equipo.
- Condiciones de aprobación.
- Nota mínima.
- Ponderación de evaluaciones.

Se observó que una pregunta puede necesitar varios fragmentos relevantes y no solamente el resultado con mayor score.

# 10. Rendimiento actual

Se han realizado pruebas con 20 y 10 candidatos.

## 10.1 Prueba con 20 candidatos

| Etapa | Tiempo |
|---|---:|
| Carga modelo E5 | 4,047 s |
| Carga reranker | 4,739 s |
| Embedding consulta | 0,191 s |
| Apertura Qdrant | 4,719 s |
| Búsqueda híbrida | 0,337 s |
| Reranking | 5,742 s |
| Total | 19,782 s |

## 10.2 Prueba con 10 candidatos

| Etapa | Tiempo |
|---|---:|
| Carga modelo E5 | 3,584 s |
| Carga reranker | 3,758 s |
| Embedding consulta | 0,210 s |
| Apertura Qdrant | 5,413 s |
| Búsqueda híbrida | 0,355 s |
| Reranking | 3,115 s |
| Total | 16,442 s |

## 10.3 Interpretación

Los tiempos muestran que:

1. La carga de modelos representa un coste importante.
2. La apertura de Qdrant también representa un coste significativo.
3. La búsqueda híbrida tiene tiempos reducidos.
4. El reranking aumenta su coste al incrementar los candidatos.
5. El número de candidatos debe equilibrar cobertura, calidad y latencia.

La principal oportunidad de optimización identificada es mantener los modelos y el cliente Qdrant inicializados durante la vida de un servicio persistente.

Todavía no se ha medido la latencia de un servicio con recursos residentes en memoria.

# 11. Arquitectura lógica actual

```text
                    CONSULTA
                       |
                       v
              EMBEDDING E5
                       |
                       v
          +------------+------------+
          |                         |
          v                         v
    BÚSQUEDA VECTORIAL         BÚSQUEDA BM25
          |                         |
          +------------+------------+
                       |
                       v
                      RRF
                       |
                       v
                TOP K CANDIDATOS
                       |
                       v
                 BGE RERANKER
                       |
                       v
               TOP K RESULTADOS
                       |
                       v
                    FIN
```

La construcción del contexto y la generación mediante LLM todavía no forman parte del pipeline implementado.

# 12. Arquitectura objetivo

Se plantea evolucionar hacia una arquitectura modular que separe claramente la ingestión offline y la consulta online.

La siguiente estructura es una propuesta inicial, no una estructura ya implementada:

```text
rag_istae_ingestion/
│
├── app/
│   ├── core/
│   │   └── configuration
│   │
│   ├── ingestion/
│   │
│   ├── indexing/
│   │
│   ├── retrieval/
│   │
│   ├── reranking/
│   │
│   ├── context/
│   │
│   ├── services/
│   │
│   └── api/
│
├── scripts/
│
├── config/
│
├── models/
│
├── outputs/
│
├── vector_store/
│
├── evaluation/
│
└── tests/
```

## 12.1 Servicio persistente

Se propone utilizar FastAPI y su mecanismo de ciclo de vida (`lifespan`) para inicializar una sola vez:

- Modelo E5.
- Modelo BGE.
- Cliente Qdrant.

Las consultas posteriores reutilizarán estos recursos.

## 12.2 Separación de responsabilidades

La arquitectura deberá distinguir:

**Ingestión e indexación offline**
- Extracción documental.
- Fragmentación.
- Generación de embeddings.
- Indexación vectorial.
- Indexación léxica.

**Consulta online**
- Embedding de consulta.
- Recuperación híbrida.
- Fusión RRF.
- Reranking.
- Construcción del contexto.
- Generación mediante LLM.

La implementación de contexto y generación se realizará posteriormente.

# 13. Decisiones técnicas que deben preservarse

Durante la refactorización se deben mantener:

1. Modelo `intfloat/multilingual-e5-base`.
2. Modelo `BAAI/bge-reranker-v2-m3`.
3. Prefijos `passage:` y `query:`.
4. Dimensión vectorial de 768.
5. Normalización de embeddings.
6. Colección Qdrant `istae_chunks_e5_base`.
7. BM25 nativo de Qdrant.
8. Fusión RRF.
9. Parámetro RRF k = 60, salvo evaluación que justifique cambios.
10. Identificadores de chunks existentes.
11. Estructura del payload documental.
12. Compatibilidad con la configuración JSON.
13. Compatibilidad con los artefactos de ingestión existentes.

No se debe reindexar el corpus ni recrear la colección como parte de la refactorización.

# 14. Próximas etapas

## Etapa 1. Auditoría arquitectónica

- Inventariar archivos.
- Analizar dependencias.
- Identificar duplicaciones.
- Revisar responsabilidades.
- Diseñar arquitectura objetivo.

## Etapa 2. Refactorización estructural

- Separar configuración.
- Separar servicios de recuperación.
- Centralizar inicialización de modelos.
- Separar lógica de negocio y scripts.
- Preservar compatibilidad.

## Etapa 3. Servicio persistente

- Incorporar FastAPI.
- Inicializar E5, BGE y Qdrant una sola vez.
- Implementar endpoint de consulta.
- Medir latencia real.
- Evaluar concurrencia.

## Etapa 4. Evaluación de recuperación

Construir un conjunto de preguntas representativas y fragmentos relevantes verificados manualmente.

Métricas previstas:

- Recall@20.
- Recall@50.
- MRR@5.
- nDCG@5.
- Latencia por etapa.

## Etapa 5. Construcción del contexto

Implementar una capa independiente que gestione:

- Recuperación de vecinos.
- Agrupación documental.
- Deduplicación.
- Solapamientos.
- Límite de tokens.
- Trazabilidad de fuentes.

## Etapa 6. Integración del LLM

Incorporar generación de respuestas fundamentadas en el contexto recuperado.

Se deberá evaluar:

- Fidelidad documental.
- Precisión de las respuestas.
- Trazabilidad de citas.
- Respuestas sin evidencia suficiente.
- Comportamiento ante preguntas compuestas.

# 15. Reglas para futuras modificaciones

Este proyecto ya cuenta con componentes funcionales y resultados validados.

Por tanto, cualquier refactorización deberá respetar las siguientes reglas:

- No modificar componentes validados sin justificación.
- No eliminar scripts antes de comprobar sus dependencias.
- No alterar los datos persistentes.
- No descargar nuevamente los modelos si ya están disponibles localmente.
- No cambiar los modelos sin una evaluación comparativa.
- No sacrificar cobertura documental para mejorar latencia.
- No mezclar ingestión offline y consulta online.
- Mantener pruebas de regresión.
- Realizar modificaciones por fases.
- Documentar decisiones arquitectónicas.
- Mantener compatibilidad con Windows y el entorno Python existente.

**Principio rector:** cada fase debe dejar el proyecto en un estado funcional y verificable antes de comenzar la siguiente.

---

# 16. Estado resumido

| Componente | Estado |
|---|---|
| Inventario documental | Implementado |
| Procesamiento del corpus | Implementado |
| Fragmentación | Implementada |
| Embeddings E5 | Implementados |
| Indexación vectorial | Implementada |
| BM25 nativo Qdrant | Implementado |
| Recuperación híbrida | Implementada |
| Fusión RRF | Implementada |
| BGE Reranker | Implementado |
| Inspección de Qdrant | Implementada |
| Recuperación de vecinos | Probada |
| Evaluación sistemática | Pendiente |
| Context Builder | Pendiente |
| API persistente | Pendiente |
| Integración LLM | Pendiente |

**Última actualización:** 29 de octubre de 2026.