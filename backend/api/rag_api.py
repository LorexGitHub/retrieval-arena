from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import json
import threading
import queue
from pathlib import Path

from backend.rag.experiment import run_batch, load_queries, load_dataset, DATA_DIR
from backend.rag.config import EMBEDDING_MODELS, LLM_MODELS
from backend.rag.pipeline import RAGPipeline
from backend.rag.database import is_available as db_available

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

api = APIRouter(prefix="/api")


class SingleRunRequest(BaseModel):
    query: str
    ground_truth: str
    dataset_name: str
    embedding_model: str
    top_k: int = 5
    llm_model: str = ""


class CompareRequest(BaseModel):
    query: str
    ground_truth: str
    dataset_name: str
    embedding_models: Optional[list[str]] = None
    llm_model: str = ""
    top_k: int = 5


class QueryItem(BaseModel):
    query: str
    ground_truth: str = ""


class CompareMultiRequest(BaseModel):
    queries: list[QueryItem]
    dataset_name: str
    embedding_models: Optional[list[str]] = None
    llm_model: str = ""
    top_k: int = 5


class BatchRunRequest(BaseModel):
    embedding_models: Optional[list[str]] = None
    top_k: int = 5


class CreateDatasetRequest(BaseModel):
    documents: list[str] | list[dict]


class IngestURLRequest(BaseModel):
    url: str
    name: str
    chunk_size: int = 500
    chunk_overlap: int = 50


@api.get("/models")
def list_models():
    return {"available_models": list(EMBEDDING_MODELS.keys())}


@api.get("/llms")
def list_llms():
    models = [{"key": k, **v} for k, v in LLM_MODELS.items()]
    return {"available_llms": models}


@api.get("/datasets")
def list_datasets():
    if db_available():
        from backend.rag.database import get_datasets
        try:
            return {"available_datasets": get_datasets()}
        except Exception:
            pass
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    return {"available_datasets": list(datasets.keys())}


@api.get("/datasets/{dataset_name}/documents")
def get_dataset_documents(dataset_name: str):
    if db_available():
        from backend.rag.database import get_dataset_documents as db_docs
        try:
            docs = db_docs(dataset_name)
            return {"dataset_name": dataset_name, "documents": docs}
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
        except Exception:
            pass
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    docs = datasets.get(dataset_name)
    if docs is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    return {"dataset_name": dataset_name, "documents": docs}


@api.put("/datasets/{dataset_name}")
def create_dataset(dataset_name: str, req: CreateDatasetRequest):
    if db_available():
        from backend.rag.database import save_dataset
        try:
            save_dataset(dataset_name, req.documents)
            return {"message": f"Dataset '{dataset_name}' saved", "documents": req.documents}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    datasets_path = DATA_DIR / "datasets.json"
    data = json.loads(datasets_path.read_text()) if datasets_path.exists() else {}
    data[dataset_name] = req.documents
    datasets_path.write_text(json.dumps(data, indent=4))
    return {"message": f"Dataset '{dataset_name}' saved", "documents": req.documents}


@api.delete("/datasets/{dataset_name}")
def delete_dataset(dataset_name: str):
    if db_available():
        from backend.rag.database import remove_dataset
        try:
            remove_dataset(dataset_name)
            return {"message": f"Dataset '{dataset_name}' deleted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    datasets_path = DATA_DIR / "datasets.json"
    data = json.loads(datasets_path.read_text()) if datasets_path.exists() else {}
    if dataset_name in data:
        del data[dataset_name]
        datasets_path.write_text(json.dumps(data, indent=4))
    return {"message": f"Dataset '{dataset_name}' deleted"}


@api.post("/ingest-url")
def ingest_url(req: IngestURLRequest):
    from backend.rag.web_loader import ingest_url as do_ingest
    try:
        result = do_ingest(req.url, req.name, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.get("/queries")
def list_queries():
    return {"queries": load_queries()}


def _run_pipeline_thread(req, documents, on_stage, msg_queue):
    try:
        pipeline = RAGPipeline()
        result = pipeline.run(
            query=req.query,
            documents=documents,
            ground_truth=req.ground_truth,
            dataset_name=req.dataset_name,
            embedding_model=req.embedding_model,
            top_k=req.top_k,
            llm_model=getattr(req, "llm_model", ""),
            on_stage=on_stage,
        )
        msg_queue.put(("result", result))
    except Exception as e:
        msg_queue.put(("error", str(e)))


def _serialize(payload):
    return payload.model_dump() if hasattr(payload, "model_dump") else payload

def _event_stream(msg_queue):
    while True:
        msg_type, payload = msg_queue.get()
        if msg_type == "stage":
            yield f"data: {json.dumps({'type': 'stage', 'message': payload})}\n\n"
        elif msg_type == "result":
            yield f"data: {json.dumps({'type': 'result', 'result': _serialize(payload)})}\n\n"
            break
        elif msg_type == "error":
            yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
            break


@api.post("/run")
def run_single(req: SingleRunRequest):
    documents = load_dataset(req.dataset_name)
    msg_queue = queue.Queue()
    def on_stage(s):
        msg_queue.put(("stage", s))
    thread = threading.Thread(target=_run_pipeline_thread, args=(req, documents, on_stage, msg_queue), daemon=True)
    thread.start()
    return StreamingResponse(_event_stream(msg_queue), media_type="text/event-stream")


def _compare_thread(req, msg_queue):
    try:
        documents = load_dataset(req.dataset_name)
        models = req.embedding_models or list(EMBEDDING_MODELS.keys())
        results = {}
        for model_name in models:
            msg_queue.put(("stage", f"Processing: {model_name}"))
            pipeline = RAGPipeline()
            result = pipeline.run(
                query=req.query,
                documents=documents,
                ground_truth=req.ground_truth,
                dataset_name=req.dataset_name,
                embedding_model=model_name,
                top_k=req.top_k,
                llm_model=req.llm_model,
            )
            results[model_name] = result.model_dump()
        report = {
            "query": req.query,
            "ground_truth": req.ground_truth,
            "dataset": req.dataset_name,
            "llm_model": req.llm_model,
            "results": results,
        }
        _pick_best_report(report)
        msg_queue.put(("result", report))
    except Exception as e:
        import traceback
        msg_queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


@api.post("/compare")
def compare_models(req: CompareRequest):
    msg_queue = queue.Queue()
    thread = threading.Thread(target=_compare_thread, args=(req, msg_queue), daemon=True)
    thread.start()
    return StreamingResponse(_event_stream(msg_queue), media_type="text/event-stream")


def _compare_multi_thread(req: CompareMultiRequest, msg_queue):
    try:
        documents = load_dataset(req.dataset_name)
        models = req.embedding_models or list(EMBEDDING_MODELS.keys())
        reports = []
        for qi, qitem in enumerate(req.queries):
            results = {}
            for mi, model_name in enumerate(models):
                msg_queue.put(("stage", f"[{qi + 1}/{len(req.queries)}] {qitem.query[:40]} \u00b7 {model_name}"))
                pipeline = RAGPipeline()
                result = pipeline.run(
                    query=qitem.query,
                    documents=documents,
                    ground_truth=qitem.ground_truth,
                    dataset_name=req.dataset_name,
                    embedding_model=model_name,
                    top_k=req.top_k,
                    llm_model=req.llm_model,
                )
                results[model_name] = result.model_dump()
            report = {
                "query": qitem.query,
                "ground_truth": qitem.ground_truth,
                "dataset": req.dataset_name,
                "llm_model": req.llm_model,
                "results": results,
            }
            _pick_best_report(report)
            reports.append(report)
        msg_queue.put(("result", {"reports": reports, "total": len(reports)}))
    except Exception as e:
        import traceback
        msg_queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


@api.post("/compare-multi")
def compare_multi(req: CompareMultiRequest):
    msg_queue = queue.Queue()
    thread = threading.Thread(target=_compare_multi_thread, args=(req, msg_queue), daemon=True)
    thread.start()
    return StreamingResponse(_event_stream(msg_queue), media_type="text/event-stream")


def _pick_best_report(report: dict):
    scored = []
    for name, result in report["results"].items():
        if "error" in result:
            continue
        ev = result["evaluation"]
        exact_bonus = 1.0 if ev.get("exact_match") else 0.0
        composite = (
            exact_bonus * 50.0
            + ev.get("semantic_similarity", 0.0) * 25.0
            + ev.get("rouge_l_f1", 0.0) * 15.0
            + ((ev.get("llm_quality_score") or 0) / 5.0) * 10.0
        )
        scored.append((composite, name))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = scored[0][0]
        winners = [name for score, name in scored if score == best]
        report["best_model"] = ", ".join(sorted(winners)) if len(winners) > 1 else winners[0]


@api.post("/run-batch")
def run_batch_endpoint(req: BatchRunRequest):
    queries = load_queries()
    reports = run_batch(queries, req.embedding_models, req.top_k)
    return {"reports": reports, "total_queries": len(reports)}


# --- FastAPI app setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    if db_available():
        from backend.rag.database import init_db
        try:
            init_db()
        except Exception:
            pass
    yield


app = FastAPI(title="RAG Evaluation API", lifespan=lifespan)
app.include_router(api)


# Serve static frontend if built
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("mcp"):
            raise HTTPException(status_code=404, detail="Not found")
        fp = STATIC_DIR / full_path
        if fp.exists() and fp.is_file():
            return FileResponse(str(fp))
        return FileResponse(str(STATIC_DIR / "index.html"))
