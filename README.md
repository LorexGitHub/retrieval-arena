# Retrieval Arena

A framework for benchmarking and comparing embedding models in a retrieval-augmented generation (RAG) context. Access results through a React SPA, a REST API, or an MCP server for LLM integration.

### The Problem

Choosing an embedding model for a retrieval pipeline involves trade-offs between speed, memory, and retrieval quality. Models vary wildly in capability — from lightweight 33M-parameter models to high-dimensional 600M-parameter ones — and the best choice depends on your specific domain and data. Without a structured comparison, teams rely on intuition rather than empirical evidence.

### The Solution

A containerized evaluation platform that runs a query against multiple embedding models side-by-side and scores each result using 10 metrics (4 retrieval + 6 generation). Three access paths:

- **React SPA** — Dark-themed dashboard with SSE real-time progress, run comparisons, browse results, manage datasets
- **FastAPI** — REST + SSE streaming for programmatic access
- **MCP Server** — Exposes RAG tools over the Model Context Protocol, allowing any MCP-compatible LLM to query and evaluate models externally

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
│   │   ├── retriever.py  # In-process model caching with subprocess fallback
│   │   ├── generator.py  # LLM/template answer generation
│   │   ├── evaluator.py  # Evaluation metrics (EM, ROUGE, semantic sim, LLM judge)
│   │   ├── pipeline.py   # Orchestrates retrieval + generation + evaluation
│   │   ├── database.py   # PostgreSQL layer with JSON file fallback
│   │   └── experiment.py # Batch experiment runner
│   ├── api/
│   │   └── rag_api.py    # FastAPI with SSE streaming + static file serving
│   ├── mcp/
│   │   ├── mcp_server.py # MCP SSE server (exposes tools over HTTP/SSE)
│   │   ├── stdio_server.py # Stdio entry point for Claude Desktop
│   │   └── tasks.py      # Celery async tasks
│   └── combined_app.py   # Routes /mcp* → MCP, everything else → FastAPI + SPA
├── rag-ui/               # React frontend (Vite + Tailwind + shadcn/ui)
│   ├── src/
│   │   ├── components/   # 18 components (layout, benchmark, ui primitives)
│   │   ├── hooks/        # useDatasets, SSE integration
│   │   ├── lib/          # API client, cn() utility
│   │   └── types/        # TypeScript interfaces matching Pydantic schemas
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── datasets.json     # 11 category datasets (cars, cuisines, programming, etc.)
│   ├── rag_queries.json  # 20 evaluation queries with ground truth
│   └── vector_cache/     # Local file-based embedding cache
├── infra/
│   ├── main.tf           # Terraform for Hetzner CX23
│   └── nginx.conf        # Reverse proxy config
├── Dockerfile            # Multi-stage: Node builds frontend, Python serves it
├── docker-compose.yaml   # 6 services (postgres, redis, rag-api, combined, celery-worker, mcp-sse)
└── requirements.txt
```

### Tech Stack

- **Frontend**: React 19, Tailwind CSS v4, shadcn/ui, Vite
- **Backend**: Python 3.12, FastAPI, Celery + Redis, Sentence-Transformers
- **Database**: PostgreSQL 16 (psycopg2), local file-based vector cache
- **Infrastructure**: Docker Compose, Terraform + Hetzner (cloud), Nginx

### Architecture

```
                     ┌─────────────────┐
                     │  Claude Desktop  │
                     │  (MCP stdio)     │
                     └────────┬────────┘
                              │ docker exec -i
                     ┌────────▼────────┐
                     │    mcp-sse      │
                     │  (HTTP/SSE)     │
                     │  port 5100      │
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

  Browser ──► nginx:80 ──► combined:8000 (SPA + API + /mcp/)
                      or ──► rag-api:8002 (API only)
                      or ──► mcp-sse:5100 (MCP only)

  React SPA ──► /api/* ──► FastAPI (direct RAG, SSE streaming)
```

### Getting Started (Local)

**Prerequisites**: Docker Desktop

1. Clone and start:
   ```bash
   docker compose up --build -d
   ```

2. Access the services:
   | Service       | URL                             |
   |---------------|---------------------------------|
   | React SPA     | http://localhost:8000           |
   | FastAPI       | http://localhost:8765           |
   | MCP SSE       | http://localhost:5100/mcp/      |

3. Stop:
   ```bash
   docker compose down
   ```

### Frontend Development

Run the React dev server with hot reload (proxies `/api` to the FastAPI backend):

```bash
cd rag-ui
npm install
npm run dev
```

Opens at `http://localhost:5173` — points `/api/*` requests to `http://localhost:8002`.

### Connecting Claude Desktop

Add this to `claude_desktop_config.json` (located at `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "rag": {
      "command": "docker",
      "args": ["exec", "-i", "retrieval-arena-mcp-sse-1", "python", "/app/src/mcp/stdio_server.py"]
    }
  }
}
```

Restart Claude Desktop — you'll see a hammer icon with 9 tools:

| Tool | Purpose |
|---|---|
| `list_datasets` | List available datasets |
| `list_queries` | List evaluation queries |
| `submit_rag_job` | Submit async RAG query (returns job_id) |
| `check_job_status` | Poll job result by ID |
| `list_cached_results` | Browse recent results |
| `get_cached_result` | Fetch specific result by job_id |
| `get_dataset` | Get dataset documents by name |
| `create_dataset` | Create or overwrite a dataset |
| `delete_dataset` | Delete a dataset |

The `submit_rag_job` tool also accepts an optional `ground_truth` parameter for evaluation. If omitted, evaluation metrics will not be meaningful.

Example prompt: *"Submit a RAG job to find which tech company created the iPhone using minilm-l12 on the tech_companies dataset"*

### Generator Configuration

By default the app returns the top retrieved document as the "answer" (template mode). To use a real LLM, set environment variables on the `rag-api` service in `docker-compose.yaml`:

| Env var                          | Behavior                                                             |
| -------------------------------- | -------------------------------------------------------------------- |
| (none set)                       | Returns top retrieved document                                       |
| `LLM_BASE_URL`                   | OpenAI-compatible API (vLLM, OpenAI, etc.); set `LLM_API_KEY` if needed |
| `LLM_MODEL`                      | Local HuggingFace model (key or full model ID)                       |
| `LLM_USE_OLLAMA=1`               | Ollama (uses `OLLAMA_BASE_URL`, default `http://localhost:11434`)    |

### API Endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List available embedding models |
| GET | `/api/datasets` | List available datasets |
| GET | `/api/datasets/{name}/documents` | Get dataset documents |
| PUT | `/api/datasets/{name}` | Create or overwrite a dataset |
| DELETE | `/api/datasets/{name}` | Delete a dataset |
| GET | `/api/queries` | List evaluation queries |
| POST | `/api/run` | Single-model RAG with SSE streaming |
| POST | `/api/compare` | Multi-model comparison with per-model SSE progress |
| POST | `/api/run-batch` | Batch experiment across all queries |

### Evaluation Metrics

**Retrieval (4):**
- **Hit Rate@k** — Was the relevant document in the top-k results?
- **MRR@k** — Mean reciprocal rank of the first relevant result
- **Precision@k** — Fraction of relevant documents in top-k
- **NDCG@k** — Discounted cumulative gain (position-weighted)

**Generation (6):**
- **Exact Match** — Binary match between answer and ground truth
- **ROUGE-L F1** — Longest common subsequence overlap
- **Semantic Similarity** — Cosine similarity between answer and ground truth embeddings
- **Faithfulness** — Does the answer stay factual relative to retrieved documents? (requires LLM judge)
- **Answer Relevancy** — How well does the answer address the query? (requires LLM judge)
- **LLM Quality Score** — 1-5 rating from a judge LLM (requires LLM judge)

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

### Local File-Based Vector Cache

Vectors and results are cached locally in `data/vector_cache/{model}/{dataset}/{hash}.json` — no cloud dependency, no API key needed. Cache is keyed by SHA256 hash of the concatenated document texts.
