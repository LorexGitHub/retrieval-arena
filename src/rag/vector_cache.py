"""Local file-based vector cache for RAG query results, stored in data/vector_cache/"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("VECTOR_CACHE_DIR", "")) or (
    Path(__file__).resolve().parent.parent.parent / "data" / "vector_cache"
)


def is_available() -> bool:
    return True


def _cache_path(query: str, model: str, dataset: str) -> Path:
    raw = f"{model}:{dataset}:{query.lower().strip()}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return CACHE_DIR / model / dataset / f"{h}.json"


def get_cached(query: str, model: str, dataset: str) -> Optional[dict]:
    path = _cache_path(query, model, dataset)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data["result"]
    except Exception as e:
        logger.debug("Cache read failed for %s: %s", path, e)
        return None


def set_cache(query: str, model: str, dataset: str, result: dict, vector: list[float]):
    path = _cache_path(query, model, dataset)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "query": query,
                "model": model,
                "dataset": dataset,
                "vector": vector,
                "result": result,
            }, f)
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", path, e)
