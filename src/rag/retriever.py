from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

from .config import EMBEDDING_MODELS, RETRIEVAL_TOP_K
from .schemas import RetrievalResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Config keys that should NOT be forwarded to SentenceTransformer()
_MODEL_KWARGS_SKIP = {"model_name", "default_task", "encode_kwargs", "size", "memory", "speed"}

# Normalized model name lookup (case-insensitive)
_MODEL_NAME_MAP = {k.lower(): k for k in EMBEDDING_MODELS}

# Estimated memory per model (GB) for cache budgeting
_MODEL_MEMORY_ESTIMATES = {
    "sentence-transformers/all-MiniLM-L12-v2": 0.09,
    "BAAI/bge-small-en-v1.5": 0.09,
    "thenlper/gte-small": 0.10,
    "ibm-granite/granite-embedding-small-english-r2": 0.20,
    "microsoft/harrier-oss-v1-270m": 0.55,
    "BAAI/bge-base-en-v1.5": 0.25,
    "sentence-transformers/all-mpnet-base-v2": 0.25,
    "Qwen/Qwen3-Embedding-0.6B": 1.20,
    "jinaai/jina-embeddings-v5-text-small": 1.20,
    "BAAI/bge-large-en-v1.5": 0.70,
}

# In-memory model cache: populated only if enough RAM is available
_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_ENABLED = False


def _check_cache_feasible() -> bool:
    total_est = sum(_MODEL_MEMORY_ESTIMATES.values())
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])
        avail_gb = mem.get("MemAvailable", 0) / 1024 / 1024
        return avail_gb > total_est * 1.5
    except Exception:
        return False


_MODEL_CACHE_ENABLED = _check_cache_feasible()


def _get_model(model_id: str, model_kwargs: dict) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    if _MODEL_CACHE_ENABLED:
        model = _MODEL_CACHE.get(model_id)
        if model is not None:
            return model
        model = SentenceTransformer(model_id, **model_kwargs)
        _MODEL_CACHE[model_id] = model
        return model
    return SentenceTransformer(model_id, **model_kwargs)


def _retrieve_worker(
    query: str,
    documents: list[str],
    model_id: str,
    model_kwargs: dict,
    encode_kwargs: dict,
    top_k: int,
    result_queue: mp.Queue,
):
    """Run inside a child process so all model memory is freed on exit."""
    try:
        from sentence_transformers import util

        model = _get_model(model_id, model_kwargs)
        doc_embs = model.encode(documents, convert_to_tensor=True, **encode_kwargs)
        query_emb = model.encode(query, convert_to_tensor=True, **encode_kwargs)
        scores = util.cos_sim(query_emb, doc_embs)[0]
        top_indices = scores.argsort(descending=True)[:top_k].tolist()

        result_queue.put({
            "documents": [documents[i] for i in top_indices],
            "scores": [float(scores[i]) for i in top_indices],
        })
    except Exception as e:
        result_queue.put({"error": str(e)})


def _retrieve_sync(
    query: str,
    documents: list[str],
    model_id: str,
    model_kwargs: dict,
    encode_kwargs: dict,
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Run in-process with model caching when feasible."""
    from sentence_transformers import util

    model = _get_model(model_id, model_kwargs)
    doc_embs = model.encode(documents, convert_to_tensor=True, **encode_kwargs)
    query_emb = model.encode(query, convert_to_tensor=True, **encode_kwargs)
    scores = util.cos_sim(query_emb, doc_embs)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [documents[i] for i in top_indices], [float(scores[i]) for i in top_indices]


class Retriever:
    @classmethod
    def retrieve(
        cls,
        query: str,
        documents: list[str],
        model_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> RetrievalResult:
        resolved = _MODEL_NAME_MAP.get(model_name.lower())
        if resolved is None:
            raise KeyError(f"Unknown embedding model: {model_name}")
        model_cfg = EMBEDDING_MODELS[resolved]
        model_id = model_cfg["model_name"]

        model_kwargs = {k: v for k, v in model_cfg.items() if k not in _MODEL_KWARGS_SKIP}

        encode_kwargs = {}
        task = model_cfg.get("default_task")
        if task:
            encode_kwargs["task"] = task
        extra_encode = model_cfg.get("encode_kwargs")
        if extra_encode:
            encode_kwargs.update(extra_encode)

        if _MODEL_CACHE_ENABLED or mp.current_process().daemon:
            docs, scores = _retrieve_sync(query, documents, model_id, model_kwargs, encode_kwargs, top_k)
        else:
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(
                target=_retrieve_worker,
                args=(query, documents, model_id, model_kwargs, encode_kwargs, top_k, q),
            )
            p.start()
            try:
                data = q.get(timeout=300)
            except Exception:
                p.terminate()
                p.join(timeout=10)
                raise RuntimeError("Retrieval subprocess timed out or failed")
            p.join(timeout=10)

            if "error" in data:
                raise RuntimeError(data["error"])
            docs, scores = data["documents"], data["scores"]

        return RetrievalResult(
            documents=docs,
            scores=scores,
            model_name=resolved,
            top_k=top_k,
        )
