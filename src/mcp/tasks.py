"""Async Celery tasks for RAG pipeline"""
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from celery import Celery, Task
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
celery_app = Celery("rag", broker=redis_url, backend=redis_url)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=600,
    task_time_limit=660,
    result_expires=86400,
)

RESULTS_FILE = Path(os.getenv("RESULTS_DIR", "/data")) / "results.jsonl"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


class RAGTask(Task):
    autoretry_for = (Exception,)
    max_retries = 2
    default_retry_delay = 60


@celery_app.task(base=RAGTask, bind=True)
def run_rag_task(self, job_id: str, query: str, model: str, dataset: str):
    try:
        from rag.pipeline import RAGPipeline
        from rag.config import EMBEDDING_MODELS
        from rag.experiment import load_dataset
        from rag.database import is_available as db_available, save_result

        pipeline = RAGPipeline()
        documents = load_dataset(dataset)

        models_to_run = (
            list(EMBEDDING_MODELS.keys())
            if model == "all"
            else [model]
        )

        results = []
        for m in models_to_run:
            try:
                result = pipeline.run(
                    query=query,
                    documents=documents,
                    ground_truth="placeholder",
                    dataset_name=dataset,
                    embedding_model=m,
                )
                results.append(result.model_dump())
            except Exception as e:
                logger.exception("Model %s failed", m)
                results.append({"error": str(e), "model": m, "query": query})

        output = {
            "job_id": job_id,
            "query": query,
            "dataset": dataset,
            "models_run": models_to_run,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(output) + "\n")

        if db_available():
            try:
                save_result(job_id, query, model, dataset, output)
            except Exception as e:
                logger.warning("Failed to save result to DB: %s", e)

        celery_app.backend.set(
            f"rag:result:{job_id}",
            json.dumps(output),
            ex=86400,
        )

        return output

    except SoftTimeLimitExceeded:
        return {"error": "Task timeout (10 min)", "job_id": job_id}
    except Exception as e:
        self.retry(exc=e)
