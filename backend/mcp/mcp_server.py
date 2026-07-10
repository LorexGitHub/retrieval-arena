"""MCP Server for RAG Job Management"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import CallToolResult, TextContent, Tool
from starlette.responses import Response

logger = logging.getLogger(__name__)
server = Server("rag-mcp")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/data"))
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


class JobManager:
    @staticmethod
    def submit_job(query: str, model: str = "all", dataset: str = "tech", ground_truth: str = "") -> dict:
        try:
            from backend.mcp.tasks import run_rag_task

            job_id = str(uuid4())
            task = run_rag_task.delay(job_id, query, model, dataset)
            return {
                "job_id": job_id,
                "celery_task_id": task.id,
                "status": "queued",
                "query": query,
                "model": model,
                "dataset": dataset,
                "submitted_at": datetime.utcnow().isoformat(),
            }
        except Exception:
            logger.warning("Celery/Redis unavailable — running synchronously")
            return JobManager._run_sync(query, model, dataset, ground_truth)

    @staticmethod
    def _run_sync(query: str, model: str = "all", dataset: str = "tech", ground_truth: str = "") -> dict:
        job_id = str(uuid4())
        from backend.rag.pipeline import RAGPipeline
        from backend.rag.config import EMBEDDING_MODELS
        from backend.rag.experiment import load_dataset

        gt = ground_truth or "placeholder"
        pipeline = RAGPipeline()
        documents = load_dataset(dataset)
        models_to_run = (
            list(EMBEDDING_MODELS.keys()) if model == "all" else [model]
        )

        results = []
        for m in models_to_run:
            try:
                result = pipeline.run(
                    query=query, documents=documents,
                    ground_truth=gt, dataset_name=dataset,
                    embedding_model=m,
                )
                results.append(result.model_dump())
            except Exception as e:
                results.append({"error": str(e), "model": m, "query": query})

        output = {
            "job_id": job_id,
            "query": query,
            "dataset": dataset,
            "models_run": models_to_run,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }

        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(output) + "\n")

        from backend.rag.database import is_available as db_available
        if db_available():
            try:
                from backend.rag.database import save_result
                save_result(job_id, query, model, dataset, output)
            except Exception:
                pass

        return output

    @staticmethod
    def check_status(job_id: str) -> dict:
        from backend.mcp.tasks import celery_app

        try:
            result = celery_app.backend.get(f"rag:result:{job_id}")
            if result:
                return json.loads(result)
        except Exception:
            pass

        cached = JobManager.get_result(job_id)
        if cached:
            return cached
        return {"job_id": job_id, "status": "not_found"}

    @staticmethod
    def list_results(limit: int = 50, filter_model: str = None) -> list:
        from backend.rag.database import is_available as db_available
        if db_available():
            try:
                from backend.rag.database import list_results as db_list
                return db_list(limit, filter_model)
            except Exception:
                pass

        if not RESULTS_FILE.exists():
            return []
        results = []
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if filter_model and result.get("model_name") != filter_model:
                        continue
                    results.append(result)
                except json.JSONDecodeError:
                    pass
        return results[:limit]

    @staticmethod
    def get_result(job_id: str) -> dict:
        from backend.rag.database import is_available as db_available
        if db_available():
            try:
                from backend.rag.database import get_result as db_get
                return db_get(job_id)
            except Exception:
                pass

        if not RESULTS_FILE.exists():
            return None
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if result.get("job_id") == job_id:
                        return result
                except json.JSONDecodeError:
                    pass
        return None


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="list_datasets",
            description="List available datasets",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_queries",
            description="List evaluation queries with ground truth",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="submit_rag_job",
            description="Submit async RAG query (non-blocking, returns job_id)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query"},
                    "model": {"type": "string", "description": "Embedding model", "default": "all"},
                    "dataset": {"type": "string", "description": "Dataset name", "default": "programming_languages"},
                    "ground_truth": {"type": "string", "description": "Expected answer for evaluation", "default": ""},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="check_job_status",
            description="Check job progress and result",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        Tool(
            name="list_cached_results",
            description="Browse cached results (instant)",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 50},
                    "model": {"type": "string"},
                },
            },
        ),
        Tool(
            name="get_cached_result",
            description="Get result by job_id",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        Tool(
            name="get_dataset",
            description="Get a dataset's documents by name",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="create_dataset",
            description="Create or overwrite a dataset with documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "documents"],
            },
        ),
        Tool(
            name="delete_dataset",
            description="Delete a dataset",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
    ]


def _execute_tool_sync(name: str, arguments: dict) -> dict:
    if name == "list_datasets":
        from backend.rag.database import is_available as db_avail
        if db_avail():
            from backend.rag.database import get_datasets
            datasets = get_datasets()
        else:
            import json
            from pathlib import Path
            datasets_path = Path(__file__).resolve().parent.parent.parent / "data" / "datasets.json"
            datasets = list(json.loads(datasets_path.read_text()).keys())
        return {"datasets": datasets}
    elif name == "list_queries":
        from backend.rag.experiment import load_queries
        return {"queries": load_queries()}
    elif name == "submit_rag_job":
        return JobManager.submit_job(
            arguments["query"],
            arguments.get("model", "all"),
            arguments.get("dataset", "programming_languages"),
            arguments.get("ground_truth", ""),
        )
    elif name == "check_job_status":
        return JobManager.check_status(arguments["job_id"])
    elif name == "list_cached_results":
        results = JobManager.list_results(
            arguments.get("limit", 50),
            arguments.get("model"),
        )
        return {"count": len(results), "results": results}
    elif name == "get_cached_result":
        return JobManager.get_result(arguments["job_id"]) or {"error": "Not found"}
    elif name == "get_dataset":
        ds_name = arguments["name"]
        from backend.rag.database import is_available as db_avail
        if db_avail():
            from backend.rag.database import get_dataset_documents
            docs = get_dataset_documents(ds_name)
        else:
            import json
            from pathlib import Path
            ds_path = Path(__file__).resolve().parent.parent.parent / "data" / "datasets.json"
            data = json.loads(ds_path.read_text()) if ds_path.exists() else {}
            docs = data.get(ds_name, [])
        return {"name": ds_name, "documents": docs}
    elif name == "create_dataset":
        ds_name = arguments["name"]
        docs = arguments["documents"]
        ds_path = Path(__file__).resolve().parent.parent.parent / "data" / "datasets.json"
        data = json.loads(ds_path.read_text()) if ds_path.exists() else {}
        data[ds_name] = docs
        ds_path.write_text(json.dumps(data, indent=4))
        from backend.rag.database import is_available as db_avail
        if db_avail():
            try:
                from backend.rag.database import save_dataset
                save_dataset(ds_name, docs)
            except Exception:
                pass
        return {"message": f"Dataset '{ds_name}' created", "name": ds_name, "count": len(docs)}
    elif name == "delete_dataset":
        ds_name = arguments["name"]
        ds_path = Path(__file__).resolve().parent.parent.parent / "data" / "datasets.json"
        data = json.loads(ds_path.read_text()) if ds_path.exists() else {}
        if ds_name in data:
            del data[ds_name]
            ds_path.write_text(json.dumps(data, indent=4))
        else:
            return {"error": f"Dataset '{ds_name}' not found"}
        from backend.rag.database import is_available as db_avail
        if db_avail():
            try:
                from backend.rag.database import remove_dataset
                remove_dataset(ds_name)
            except Exception:
                pass
        return {"message": f"Dataset '{ds_name}' deleted", "name": ds_name}
    else:
        return {"error": f"Unknown: {name}"}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        result = _execute_tool_sync(name, arguments)
    except Exception as e:
        logger.exception("Tool call failed")
        result = {"error": str(e)}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])


sse = SseServerTransport("/mcp/messages/")


async def app(scope, receive, send):
    if scope["type"] != "http":
        response = Response(status_code=405)
        await response(scope, receive, send)
        return

    if scope.get("path") == "/mcp/" and scope.get("method") == "GET":
        from backend.rag.database import is_available as db_available
        if db_available():
            try:
                from backend.rag.database import init_db
                init_db()
            except Exception:
                pass

    path = scope["path"]
    method = scope["method"]

    if path == "/mcp/" and method == "GET":
        async with sse.connect_sse(scope, receive, send) as streams:
            init_opts = server.create_initialization_options()
            await server.run(streams[0], streams[1], init_opts)
    elif path.startswith("/mcp/messages/") and method == "POST":
        await sse.handle_post_message(scope, receive, send)
    elif path == "/call" and method == "POST":
        from starlette.requests import Request
        req = Request(scope, receive)
        body = await req.json()
        tool_name = body["tool"]
        arguments = body.get("arguments", {})
        try:
            result = _execute_tool_sync(tool_name, arguments)
        except Exception as e:
            logger.exception("Tool call failed")
            result = {"error": str(e)}
        response = Response(
            content=json.dumps(result),
            media_type="application/json",
            status_code=200,
        )
        await response(scope, receive, send)
    else:
        response = Response(status_code=404)
        await response(scope, receive, send)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5100)
