import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import EMBEDDING_MODELS
from .pipeline import RAGPipeline
from .schemas import ExperimentReport, ExperimentConfig, ErrorResult


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "runs"


def load_queries(path: Optional[str] = None) -> list[dict]:
    from .database import is_available as db_available, get_queries

    if db_available():
        return get_queries()
    path = path or str(DATA_DIR / "rag_queries.json")
    with open(path) as f:
        return json.load(f)


def load_dataset(name: str) -> list[str] | list[dict]:
    from .database import is_available as db_available, get_dataset_documents

    if db_available():
        return get_dataset_documents(name)
    datasets_path = DATA_DIR / "datasets.json"
    with open(datasets_path) as f:
        datasets = json.load(f)
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in datasets.json")
    items = datasets[name]
    # Normalize flat arrays (old format) to {id, text} dicts
    if items and isinstance(items[0], str):
        return [{"id": item, "text": item} for item in items]
    return items


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


# ---- Export utilities ----

_METRIC_FIELDS = [
    "hit_rate", "mrr", "precision", "ndcg",
    "exact_match", "rouge_l_f1", "semantic_similarity",
    "faithfulness", "answer_relevancy", "llm_quality_score",
]


def _flatten_reports(reports: list[ExperimentReport]) -> list[dict]:
    """Convert a list of ExperimentReports into one dict per (query, model)."""
    rows = []
    for report in reports:
        for model_name, result in report.results.items():
            row = {
                "query": report.query,
                "ground_truth": report.ground_truth,
                "dataset": report.dataset,
                "model": model_name,
            }
            if isinstance(result, ErrorResult):
                row["error"] = result.error
                for f in _METRIC_FIELDS:
                    row[f] = ""
                row["generation_answer"] = ""
            else:
                row["error"] = ""
                ev = result.evaluation
                for f in _METRIC_FIELDS:
                    row[f] = getattr(ev, f, "")
                row["generation_answer"] = result.generation.answer
            rows.append(row)
    return rows


def export_csv(reports: list[ExperimentReport], path: str | Path) -> Path:
    """Write a CSV with one row per (query, model) and all metric columns."""
    path = Path(path)
    rows = _flatten_reports(reports)
    fieldnames = ["query", "ground_truth", "dataset", "model", "error",
                  "generation_answer"] + _METRIC_FIELDS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_json(reports: list[ExperimentReport], path: str | Path) -> Path:
    """Write newline-delimited JSON with one object per (query, model)."""
    path = Path(path)
    rows = _flatten_reports(reports)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _try_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_results(reports: list[ExperimentReport]) -> list[dict]:
    """Compute per-model mean, std, min, max for each metric across queries."""
    rows = _flatten_reports(reports)

    by_model: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model = row["model"]
        if row["error"]:
            continue
        if model not in by_model:
            by_model[model] = {f: [] for f in _METRIC_FIELDS}
        for f in _METRIC_FIELDS:
            v = _try_float(row[f])
            if v is not None:
                by_model[model][f].append(v)

    aggregates = []
    for model, metrics in sorted(by_model.items()):
        agg = {"model": model, "count": len(metrics[_METRIC_FIELDS[0]])}
        for f in _METRIC_FIELDS:
            vals = metrics[f]
            if vals:
                agg[f"{f}_mean"] = round(_mean(vals), 4)
                agg[f"{f}_std"] = round(_stdev(vals), 4)
                agg[f"{f}_min"] = round(min(vals), 4)
                agg[f"{f}_max"] = round(max(vals), 4)
            else:
                agg[f"{f}_mean"] = ""
                agg[f"{f}_std"] = ""
                agg[f"{f}_min"] = ""
                agg[f"{f}_max"] = ""
        aggregates.append(agg)
    return aggregates


def aggregate_to_csv(aggregates: list[dict], path: str | Path) -> Path:
    """Write aggregate statistics CSV."""
    path = Path(path)
    if not aggregates:
        path.write_text("")
        return path
    fieldnames = list(aggregates[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregates)
    return path


def save_experiment_run(
    reports: list[ExperimentReport],
    name: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """Save full results + aggregate stats to timestamped files under data/runs/.

    Returns dict with paths: {csv, json, aggregate_csv, aggregate_json}.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"{name}_{ts}" if name else ts

    meta = {
        "run_name": name,
        "timestamp": ts,
        "num_queries": len(reports),
        **(metadata or {}),
    }

    paths = {}

    # Full results (flattened)
    csv_path = RESULTS_DIR / f"{label}.csv"
    export_csv(reports, csv_path)
    paths["csv"] = str(csv_path)

    json_path = RESULTS_DIR / f"{label}.jsonl"
    export_json(reports, json_path)
    paths["jsonl"] = str(json_path)

    with open(RESULTS_DIR / f"{label}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    paths["meta"] = str(RESULTS_DIR / f"{label}_meta.json")

    # Aggregate stats
    aggregates = aggregate_results(reports)
    agg_csv_path = RESULTS_DIR / f"{label}_aggregate.csv"
    aggregate_to_csv(aggregates, agg_csv_path)
    paths["aggregate_csv"] = str(agg_csv_path)

    agg_json_path = RESULTS_DIR / f"{label}_aggregate.json"
    with open(agg_json_path, "w") as f:
        json.dump(aggregates, f, indent=2)
    paths["aggregate_json"] = str(agg_json_path)

    return paths



