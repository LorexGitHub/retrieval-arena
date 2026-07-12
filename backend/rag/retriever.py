from __future__ import annotations

import multiprocessing as mp
import os

from .config import EMBEDDING_MODELS, RETRIEVAL_TOP_K
from .schemas import RetrievalResult

# Normalized model name lookup (case-insensitive)
_MODEL_NAME_MAP = {k.lower(): k for k in EMBEDDING_MODELS}

# Module-level embedding model cache survives across _retrieve_sync calls
_EMBEDDING_CACHE: dict[str, any] = {}

# Always use in-process retrieval on Windows; subprocess path freezes memory on exit
_USE_SYNC = os.name == "nt" or mp.current_process().daemon


def _retrieve_worker(query, documents, model_id, top_k, result_queue):
    """Run inside a child process so all model memory is freed on exit."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        embeddings = HuggingFaceEmbeddings(model_name=model_id)
        vectorstore = InMemoryVectorStore.from_texts(documents, embeddings)
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        result_queue.put({
            "documents": [r[0].page_content for r in results],
            "scores": [r[1] for r in results],
        })
    except Exception as e:
        result_queue.put({"error": str(e)})


def _get_embeddings(model_name: str, model_config: dict | None = None):
    """Return cached HuggingFaceEmbeddings instance."""
    if model_name not in _EMBEDDING_CACHE:
        from langchain_huggingface import HuggingFaceEmbeddings
        kwargs = {"model_name": model_name}
        model_kwargs = {}
        encode_kwargs = {}
        if model_config:
            if model_config.get("trust_remote_code"):
                model_kwargs["trust_remote_code"] = True
            task = model_config.get("default_task")
            if task:
                encode_kwargs["task"] = task
            ek = model_config.get("encode_kwargs")
            if ek:
                encode_kwargs.update(ek)
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if encode_kwargs:
            kwargs["encode_kwargs"] = encode_kwargs
        _EMBEDDING_CACHE[model_name] = HuggingFaceEmbeddings(**kwargs)
    return _EMBEDDING_CACHE[model_name]


def _retrieve_sync(query, documents, model_id, top_k, model_config=None):
    """Run in-process with model caching when feasible."""
    from langchain_core.vectorstores import InMemoryVectorStore

    embeddings = _get_embeddings(model_id, model_config)
    vectorstore = InMemoryVectorStore.from_texts(documents, embeddings)
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    return [r[0].page_content for r in results], [r[1] for r in results]


class Retriever:
    @classmethod
    def retrieve(
        cls,
        query: str,
        documents: list[str] | list[dict],
        model_name: str,
        top_k: int = RETRIEVAL_TOP_K,
        dataset_name: str = "",
    ) -> RetrievalResult:
        resolved = _MODEL_NAME_MAP.get(model_name.lower())
        if resolved is None:
            raise KeyError(f"Unknown embedding model: {model_name}")

        if documents and isinstance(documents[0], dict):
            doc_ids = [d["id"] for d in documents]
            doc_texts = [d["text"] for d in documents]
        else:
            doc_ids = list(documents)
            doc_texts = list(documents)

        from .vector_cache import get_cached, set_cache

        if dataset_name:
            cached = get_cached(query, resolved, dataset_name)
            if cached is not None:
                return RetrievalResult(
                    documents=cached["documents"],
                    scores=cached["scores"],
                    model_name=resolved,
                    top_k=top_k,
                    document_ids=cached.get("document_ids", []),
                )

        model_cfg = EMBEDDING_MODELS[resolved]
        model_id = model_cfg["model_name"]

        if _USE_SYNC:
            docs, scores = _retrieve_sync(query, doc_texts, model_id, top_k, model_cfg)
        else:
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(
                target=_retrieve_worker,
                args=(query, doc_texts, model_id, top_k, q),
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

        result_ids = [doc_ids[doc_texts.index(d)] for d in docs]

        result = RetrievalResult(
            documents=docs,
            scores=scores,
            model_name=resolved,
            top_k=top_k,
            document_ids=result_ids,
        )

        if dataset_name:
            try:
                query_vec = _get_embeddings(model_id, model_cfg).embed_query(query)
                set_cache(query, resolved, dataset_name, {
                    "documents": docs,
                    "scores": scores,
                    "model_name": resolved,
                    "document_ids": result_ids,
                }, query_vec)
            except Exception:
                pass

        return result
