# RAG Canarias — módulo Python

Pipeline RAG local que **consume** el corpus generado por la app ASP.NET Web Forms
(`rag_can_webform`) y expone un endpoint FastAPI listo para que `Chat.aspx` lo consulte.

```
ASP.NET Web Forms (crawler/indexer/chat)         Python (este módulo)
─────────────────────────────────────         ──────────────────────────
 App_Data/p2/<dominio>/NNN_*.txt    ───┐
 App_Data/p2/<dominio>/NNN_*.metadata.json ──►  validate_corpus
                                       │       chunk → data/chunks.jsonl
                                       │       embed_index → Qdrant
                                       │       FastAPI :8000/query
 Chat.aspx ─────────── HTTP POST ──────────────► /query → {answer, sources}
```

## Arquitectura en una página

1. **Validación** (`app/validate_corpus.py`) — recorre el corpus en disco, comprueba que
   cada `.txt` tiene su `.metadata.json` con `domain_metadata` y `page_metadata`, y reporta
   válidos / inválidos / vacíos / duplicados. No modifica archivos.
2. **Chunking** (`app/chunk.py`) — divide cada documento en trozos de ~1200 caracteres con
   solapamiento de ~180, respetando párrafos cuando es posible
   (`langchain_text_splitters.RecursiveCharacterTextSplitter`, con fallback propio).
   Persiste `data/chunks.jsonl` con `chunk_id`, `source_id`, `text` y `metadata`.
3. **Indexación** (`app/embed_index.py`) — carga `intfloat/multilingual-e5-small` (384 dims),
   usa GPU si `torch.cuda.is_available()`, antepone el prefijo `passage:` requerido por E5
   y sube los puntos a la colección Qdrant `rag_canarias`.
4. **Consulta CLI** (`app/query.py`) — embedding de la pregunta (con prefijo `query:`),
   búsqueda top-K y pretty-print por terminal.
5. **API** (`app/api.py`) — FastAPI con `GET /health` y `POST /query`. Mientras no haya LLM
   configurado, la respuesta es **extractiva**: concatena los pasajes más relevantes con su
   fuente. `sources` siempre se devuelve estructurado (Chat.aspx puede renderizarlo aparte).

## Estructura

```
rag_can_python/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── chunk.py
│   ├── config.py
│   ├── corpus_utils.py
│   ├── embed_index.py
│   ├── models.py
│   ├── query.py
│   ├── retrieval.py
│   └── validate_corpus.py
├── data/                  # generado: chunks.jsonl, etc.
├── requirements.txt
├── README_RAG.md
└── .env                   # opcional, ver «Configuración»
```

## Configuración

Toda la configuración vive en `app/config.py` y puede sobrescribirse vía variables de
entorno o un fichero `.env` en la raíz del repo. Las claves más útiles:

| Variable                | Default                                                      | Descripción                                |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| `RAG_CORPUS_DIR`        | `C:\Users\jaime\source\repos\luis-guillen\rag_can_webform\App_Data\p2` | Raíz del corpus ASP.NET            |
| `RAG_DATA_DIR`          | `./data`                                                     | Salida de chunks                           |
| `QDRANT_URL`            | `http://localhost:6333`                                      | Endpoint Qdrant                            |
| `QDRANT_API_KEY`        | *(vacío)*                                                    | Solo en cloud / con auth                   |
| `RAG_COLLECTION`        | `rag_canarias`                                               | Colección Qdrant                           |
| `RAG_EMBED_MODEL`       | `intfloat/multilingual-e5-small`                             | Modelo de embeddings                       |
| `RAG_CHUNK_SIZE`        | `1200`                                                       | Tamaño en caracteres                       |
| `RAG_CHUNK_OVERLAP`     | `180`                                                        | Overlap entre chunks                       |
| `RAG_TOP_K`             | `5`                                                          | Top-K por defecto                          |
| `RAG_ALLOWED_ORIGINS`   | `http://localhost,http://127.0.0.1`                          | Orígenes CORS adicionales                  |

## Instalación (Windows + PowerShell)

```powershell
cd C:\Users\jaime\source\repos\luis-guillen\rag_can_python

# 1) venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Dependencias (incluye PyTorch con CUDA 12.8 si tienes RTX 3050)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si la instalación de torch/cu128 falla porque tu driver no soporta CUDA 12.8, edita
> `requirements.txt` y comenta el bloque `--index-url ... torch ... cu128`; pip instalará la
> versión CPU. El módulo detecta GPU/CPU automáticamente.

## Levantar Qdrant local en Docker

```powershell
docker run -d --name qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

UI web: <http://localhost:6333/dashboard>

## Flujo completo (paso a paso)

```powershell
# 1) Validar el corpus generado por ASP.NET
python -m app.validate_corpus --corpus "C:\Users\jaime\source\repos\luis-guillen\rag_can_webform\App_Data\p2"

# 2) Generar chunks → data/chunks.jsonl
python -m app.chunk --corpus "C:\Users\jaime\source\repos\luis-guillen\rag_can_webform\App_Data\p2"

# 3) Embeddings + indexación en Qdrant (la primera vez, --recreate)
python -m app.embed_index --recreate

# 4) Probar por terminal
python -m app.query "¿Qué sabes de Memoria de Lanzarote?"

# 5) Arrancar la API
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

> Los scripts también funcionan como ficheros directos (`python app\query.py "..."`),
> pero recomendamos `python -m app.<script>` para que los imports relativos funcionen
> sin sorpresas.

## Probar la API

- Healthcheck: <http://127.0.0.1:8000/health>
- Docs interactivas (Swagger): <http://127.0.0.1:8000/docs>

```powershell
# curl en PowerShell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d '{ "question": "¿Qué sabes de Memoria de Lanzarote?", "top_k": 5 }'
```

Respuesta esperada (resumen):

```json
{
  "answer": "Sobre «...», esto es lo más relevante del corpus indexado: ...",
  "sources": [
    {
      "score": 0.81,
      "title": "Memoria de Lanzarote",
      "url": "https://memoriadelanzarote.com/",
      "domain": "memoriadelanzarote.com",
      "source_name": "Memoria de Lanzarote",
      "text_preview": "..."
    }
  ]
}
```

## Conectar `Chat.aspx`

Desde el code-behind C# de `Chat.aspx.cs`, llama al endpoint con `HttpClient`:

```csharp
using System.Net.Http;
using System.Net.Http.Json;   // si .NET 4.8 no lo trae, usa JsonConvert
using Newtonsoft.Json;

private static readonly HttpClient _http = new HttpClient {
    BaseAddress = new Uri("http://localhost:8000/")
};

protected async void BtnPreguntar_Click(object sender, EventArgs e)
{
    var payload = new {
        question = TxtPregunta.Text,
        top_k = 5
    };

    var json = JsonConvert.SerializeObject(payload);
    var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

    var resp = await _http.PostAsync("query", content);
    resp.EnsureSuccessStatusCode();

    var body = await resp.Content.ReadAsStringAsync();
    dynamic data = JsonConvert.DeserializeObject(body);

    LblRespuesta.Text = data.answer;
    // data.sources → lista con score, title, url, domain, text_preview
}
```

Notas:

- El servidor permite CORS desde `localhost`/`127.0.0.1` por regex en `config.ALLOWED_ORIGINS`.
- En Web Forms, normalmente *no* necesitas CORS porque la llamada se hace desde el servidor
  ASP.NET, no desde el navegador. Si en algún momento llamas vía AJAX del cliente, ajusta
  `RAG_ALLOWED_ORIGINS` con el origen real (p.ej. `http://localhost:44380`).

## Modelo y dimensiones

- Default: `intfloat/multilingual-e5-small` → 384 dims, multilingüe (incluye español).
- Si cambias a otro modelo (p.ej. `e5-base`/`e5-large`, `bge-m3`, ...), recuerda:
  1. ajustar `RAG_EMBED_MODEL`,
  2. **recrear** la colección con `--recreate`,
  3. re-indexar todo el corpus.

## Lo que NO incluye esta primera versión (a propósito)

- LLM generativo (extracción → síntesis). La estructura ya soporta enchufarlo en
  `retrieval.build_extractive_answer`.
- Reranking (cross-encoder).
- Crawl4AI / RAGAS / evaluación automática.
- Auth en la API (se asume localhost-only).

## Roadmap inmediato

1. Conectar un LLM (OpenAI / Ollama / Mistral local) en `retrieval.build_extractive_answer`.
2. Añadir filtros por `domain`, `island`, `language` desde Qdrant payload.
3. Añadir un reranker ligero (`BAAI/bge-reranker-v2-m3`) sobre los top-50.
4. Endpoint `POST /reindex` para que la pestaña «Indexación» de Web Forms dispare
   chunking + embedding sin tocar consola.
