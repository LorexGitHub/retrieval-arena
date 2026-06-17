# RAG Embedding Experiment

Benchmark how different embedding models affect retrieval-augmented generation (RAG) output quality. Query via the Streamlit dashboard, the FastAPI, or directly from Claude Desktop through an MCP server.

### The Problem

Different embedding models capture different semantic nuances. A query like _"A programming language named after a snake"_ might retrieve different documents depending on whether you use MiniLM, BGE, GTE, Granite, Harrier, MPNet, Qwen3, or Jina. Which model works best for your domain and dataset?

### The Solution

A containerized RAG pipeline with three access paths:

- **Streamlit UI** — Interactive dashboard for single/comparison/batch runs
- **FastAPI** — REST + SSE streaming for programmatic access
- **MCP Server** — Connects to Claude Desktop so you can run RAG queries via natural language

All retrieval runs asynchronously via Celery workers, with results cached in Redis and persisted to disk.

### Supported Embedding Models

**Fast (< 100M params):**
- **MiniLM-L12** — `sentence-transformers/all-MiniLM-L12-v2` (33M)
- **BGE-Small** — `BAAI/bge-small-en-v1.5` (33M)
- **GTE-Small** — `thenlper/gte-small` (33M)
- **Granite** — `ibm-granite/granite-embedding-small-english-r2` (102M)
- **Harrier** — `microsoft/harrier-oss-v1-270m` (270M)

**Medium (100–150M params):**
- **BGE-Base** — `BAAI/bge-base-en-v1.5` (110M)
- **MPNet** — `sentence-transformers/all-mpnet-base-v2` (110M)

**High-quality (300M+ params):**
- **Qwen3** — `Qwen/Qwen3-Embedding-0.6B` (600M)
- **Jina** — `jinaai/jina-embeddings-v5-text-small` (580M)
- **BGE-Large** — `BAAI/bge-large-en-v1.5` (335M)

### Project Structure

```
├── src/
│   ├── rag/              # RAG pipeline package
│   │   ├── config.py     # Environment-based configuration + model registry
│   │   ├── schemas.py    # Pydantic request/response models
│   │   ├── retriever.py  # Subprocess-isolated embedding retrieval
│   │   ├── generator.py  # LLM/template answer generation
│   │   ├── evaluator.py  # Evaluation metrics (EM, ROUGE, similarity, LLM)
│   │   ├── pipeline.py   # Orchestrates retrieval + generation + evaluation
│   │   └── experiment.py # Batch experiment runner
│   ├── api/
│   │   └── rag_api.py    # FastAPI with SSE streaming
│   ├── mcp/
│   │   ├── mcp_server.py # MCP SSE server (exposes tools over SSE)
│   │   ├── stdio_server.py # Stdio entry point for Claude Desktop
│   │   └── tasks.py      # Celery async tasks
│   └── ui/
│       └── rag_ui.py     # Streamlit RAG experiment dashboard
├── data/
│   ├── datasets.json     # 10 category datasets (cars, cuisines, tech, etc.)
│   └── rag_queries.json  # 20 evaluation queries with ground truth
├── infra/
│   └── main.tf           # Terraform for Hetzner CX23
├── Dockerfile            # Python 3.12, CPU-based torch
├── docker-compose.yaml   # 5 services (redis, rag-api, celery-worker, mcp-sse, rag-ui)
└── requirements.txt
```

### Tech Stack

- **Language**: Python 3.12
- **API**: FastAPI + Uvicorn with SSE streaming
- **ML**: Sentence-Transformers, PyTorch (CPU)
- **Retrieval**: Subprocess isolation (`multiprocessing.spawn`) per model, with in-process fallback for Celery workers
- **Async tasks**: Celery + Redis broker/backend
- **Frontend**: Streamlit with custom dark theme
- **MCP**: Model Context Protocol server for Claude Desktop integration
- **Infrastructure**: Docker Compose (local), Terraform + Hetzner (cloud)
- **Generator backends**: Local HF, OpenAI-compatible, Ollama, template fallback

### Architecture

```
                     ┌─────────────────┐
                     │  Claude Desktop  │
                     │  (MCP stdio)     │
                     └────────┬────────┘
                              │ docker exec -i
                     ┌────────▼────────┐
                     │    mcp-sse      │
                     │  (SSE server)   │
                     └────────┬────────┘
                              │ Celery task
                     ┌────────▼────────┐
                     │  celery-worker  │
                     │  (async tasks)  │
                     │  ┌────────────┐ │
                     │  │ RAGPipeline│ │
                     │  └────────────┘ │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────┐
     │    Redis      │ │  HF Hub    │ │  /data/    │
     │  (broker +    │ │ (downloads)│ │ (results)  │
     │   cache)      │ │           │ │            │
     └───────────────┘ └────────────┘ └────────────┘

  Streamlit UI ──► rag-api ──► (direct RAG, no Celery)
```

### Getting Started (Local)

**Prerequisites**: Docker Desktop

1. Clone and start:
   ```bash
   docker compose up --build -d
   ```

2. Access the services:
   | Service        | URL                             |
   |----------------|---------------------------------|
   | Streamlit UI   | http://localhost:8766           |
   | FastAPI        | http://localhost:8765           |
   | MCP SSE        | http://localhost:5100/mcp/      |

3. Stop:
   ```bash
   docker compose down
   ```

### Connecting Claude Desktop

Add this to `claude_desktop_config.json` (located at `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "rag": {
      "command": "docker",
      "args": ["exec", "-i", "rag-embedding-experiment-mcp-sse-1", "python", "/app/src/mcp/stdio_server.py"]
    }
  }
}
```

Restart Claude Desktop — you'll see a hammer icon with 4 tools:

| Tool | Purpose |
|---|---|
| `submit_rag_job` | Submit async RAG query (returns job_id) |
| `check_job_status` | Poll job result by ID |
| `list_cached_results` | Browse recent results |
| `get_cached_result` | Fetch specific result by job_id |

Example prompt: *"Submit a RAG job to find which tech company created the iPhone using minilm-l12 on the tech_companies dataset"*

### Generator Configuration

On default the App only shows result. To have a proper RAG pipeline there is also the option to integrate a proper LLM to respond via natural language
Generator selection is explicit via environment variables (no auto-detection):

| Env var                          | Behavior                                                                |
| -------------------------------- | ----------------------------------------------------------------------- |
| (none set)                       | `_TemplateGenerator` — returns top retrieved document                   |
| `LLM_BASE_URL`                   | OpenAI-compatible API (e.g., vLLM, OpenAI); set `LLM_API_KEY` if needed |
| `LLM_MODEL` or `LOCAL_LLM_MODEL` | Local HuggingFace model                                                 |
| `LLM_USE_OLLAMA=1`               | Ollama (uses `OLLAMA_BASE_URL`, default `http://localhost:11434`)       |

Set these in `docker-compose.yaml` under the `rag-api` service environment.

### API Endpoints (FastAPI)

- `GET /models` — List available embedding models
- `GET /datasets` — List available datasets
- `GET /datasets/{name}` — Get categories for a dataset
- `POST /datasets/{name}` — Update categories
- `GET /queries` — List evaluation queries
- `POST /run` — Single-model RAG with SSE streaming
- `POST /compare` — All-model comparison with per-model SSE progress
- `POST /run-batch` — Batch experiment across 20 queries

### Cloud Deployment (Hetzner)

1. Set your API token:
   ```bash
   export TF_VAR_hcloud_token="your-hcloud-api-token"
   ```

2. Provision:
   ```bash
   cd infra
   terraform init
   terraform apply
   ```

   This creates a CX23 (2 vCPU, 4 GB RAM) with Docker + Docker Compose installed via cloud-init.

### Evaluation Metrics

- **Substring Match**: Binary — is the ground truth found within the generated answer?
- **ROUGE-L F1**: Measures longest common subsequence overlap
- **Semantic Similarity**: Cosine similarity between answer and ground truth embeddings
- **LLM Quality Score**: 1–5 rating from a judge LLM (requires generator LLM)
