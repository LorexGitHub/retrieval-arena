import json
from pathlib import Path
from typing import Optional

from .config import EMBEDDING_MODELS
from .pipeline import RAGPipeline
from .schemas import ExperimentReport, ExperimentConfig, ErrorResult


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_queries(path: Optional[str] = None) -> list[dict]:
    path = path or str(DATA_DIR / "rag_queries.json")
    with open(path) as f:
        return json.load(f)


def load_dataset(name: str) -> list[str]:
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in datasets.json")
    return datasets[name]


def run_experiment(cfg: ExperimentConfig) -> ExperimentReport:
    pipeline = RAGPipeline()
    documents = load_dataset(cfg.dataset_name)

    results = {}
    for model_name in cfg.embedding_models:
        if model_name not in EMBEDDING_MODELS:
            results[model_name] = ErrorResult(
                error=f"Unknown embedding model: {model_name}"
            )
            continue
        try:
            result = pipeline.run(
                query=cfg.query,
                documents=documents,
                ground_truth=cfg.ground_truth,
                dataset_name=cfg.dataset_name,
                embedding_model=model_name,
                top_k=cfg.top_k,
            )
            results[model_name] = result
        except Exception as e:
            results[model_name] = ErrorResult(error=str(e))

    report = ExperimentReport(
        query=cfg.query,
        ground_truth=cfg.ground_truth,
        dataset=cfg.dataset_name,
        results=results,
    )
    _pick_best(report)
    return report


def _pick_best(report: ExperimentReport):
    scored = []
    for name, result in report.results.items():
        if isinstance(result, ErrorResult):
            continue
        metrics = result.evaluation
        exact_bonus = 1.0 if metrics.exact_match else 0.0
        composite = (
            exact_bonus * 50.0
            + metrics.semantic_similarity * 25.0
            + metrics.rouge_l_f1 * 15.0
            + (metrics.llm_quality_score / 5.0 if metrics.llm_quality_score else 0.0) * 10.0
        )
        scored.append((composite, name))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = scored[0][0]
        winners = [name for score, name in scored if score == best]
        report.best_model = ", ".join(sorted(winners)) if len(winners) > 1 else winners[0]


def run_batch(queries: list[dict], embedding_models: Optional[list[str]] = None, top_k: int = 5) -> list[ExperimentReport]:
    models = embedding_models or list(EMBEDDING_MODELS.keys())
    reports = []
    for q in queries:
        cfg = ExperimentConfig(
            query=q["query"],
            ground_truth=q["ground_truth"],
            dataset_name=q["relevant_dataset"],
            embedding_models=models,
            top_k=top_k,
        )
        report = run_experiment(cfg)
        reports.append(report)
    return reports



