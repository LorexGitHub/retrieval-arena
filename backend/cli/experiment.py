"""
CLI for batch RAG experiments across models, top-K values, and datasets.

Usage:
    python -m backend.cli.experiment --models minilm-l12,bge-small --top-k 3,5 --datasets cars,cuisines --name my_run
    python -m backend.cli.experiment --all --top-k 5 --name baseline
    python -m backend.cli.experiment --dry-run --all --top-k 3,5,10
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.rag.config import EMBEDDING_MODELS
from backend.rag.experiment import load_queries, run_batch, save_experiment_run


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(EMBEDDING_MODELS.keys())
    return [m.strip() for m in model_arg.split(",") if m.strip()]


def resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        queries = load_queries()
        return list({q.get("relevant_dataset", "") for q in queries if q.get("relevant_dataset")})
    return [d.strip() for d in dataset_arg.split(",") if d.strip()]


def build_plan(queries, datasets, models, top_k_values):
    """Build a list of (dataset, model, top_k) combinations, filtered to valid queries."""
    plan = []
    for ds in datasets:
        ds_queries = [q for q in queries if q.get("relevant_dataset") == ds]
        if not ds_queries:
            continue
        for k in top_k_values:
            for m in models:
                plan.append({
                    "dataset": ds,
                    "model": m,
                    "top_k": k,
                    "num_queries": len(ds_queries),
                })
    return plan


def run_experiments(queries, models, top_k_values, datasets, name, yes=False, dry_run=False):
    """Execute the full Cartesian product of parameters."""
    models_list = resolve_models(models)
    datasets_list = resolve_datasets(datasets)
    top_k_list = [int(k.strip()) for k in top_k_values.split(",") if k.strip()]

    if not models_list:
        print("No models specified or available.")
        sys.exit(1)
    if not datasets_list:
        print("No datasets found.")
        sys.exit(1)

    plan = build_plan(queries, datasets_list, models_list, top_k_list)
    if not plan:
        print("No valid dataset/model/top-k combinations to run.")
        sys.exit(1)

    total_evals = sum(p["num_queries"] for p in plan)
    unique_models = sorted(set(p["model"] for p in plan))
    unique_datasets = sorted(set(p["dataset"] for p in plan))
    unique_top_k = sorted(set(p["top_k"] for p in plan))

    print(f"Models:     {', '.join(unique_models)} ({len(unique_models)} total)")
    print(f"Datasets:   {', '.join(unique_datasets)} ({len(unique_datasets)} total)")
    print(f"Top-K:      {', '.join(str(k) for k in unique_top_k)}")
    print(f"Queries:    {total_evals} total evaluations")
    print()

    if dry_run:
        for p in plan:
            print(f"  dataset={p['dataset']:20s} model={p['model']:20s} top_k={p['top_k']}  ({p['num_queries']} queries)")
        print(f"\nWould run {len(plan)} experiment groups with {total_evals} total evaluations.")
        return

    total_groups = len(plan)
    if not yes:
        resp = input(f"Run {total_groups} experiment groups ({total_evals} evaluations)? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    # Group by top_k for separate result directories
    for k in sorted(set(p["top_k"] for p in plan)):
        k_models = sorted(set(p["model"] for p in plan if p["top_k"] == k))
        k_datasets = sorted(set(p["dataset"] for p in plan if p["top_k"] == k))
        label = f"{name}_k{k}" if name else f"k{k}"

        print(f"\n--- top_k={k} ({', '.join(k_models)} × {', '.join(k_datasets)}) ---")

        all_reports = []
        for ds in k_datasets:
            ds_queries = [q for q in queries if q.get("relevant_dataset") == ds]
            if not ds_queries:
                continue
            print(f"  Dataset '{ds}': {len(ds_queries)} queries × {len(k_models)} models...")
            t0 = time.time()
            reports = run_batch(ds_queries, embedding_models=k_models, top_k=k)
            elapsed = time.time() - t0
            all_reports.extend(reports)
            print(f"    done in {elapsed:.1f}s")

        meta = {
            "top_k": k,
            "models": k_models,
            "datasets": k_datasets,
            "total_evaluations": sum(p["num_queries"] for p in plan if p["top_k"] == k),
        }
        paths = save_experiment_run(all_reports, label, meta)
        print("  Results saved to:")
        for key, p in paths.items():
            print(f"    {key}: {p}")

    print("\nAll experiments complete!")


def main():
    parser = argparse.ArgumentParser(description="Run batch RAG experiments for thesis analysis.")
    parser.add_argument("--all", action="store_true",
                        help="Shorthand for --models all --datasets all")
    parser.add_argument("--models", "-m", default="all",
                        help="Comma-separated model keys, or 'all' (default: all)")
    parser.add_argument("--top-k", "-k", default="5",
                        help="Comma-separated top-K values, e.g. '3,5,10' (default: 5)")
    parser.add_argument("--datasets", "-d", default="all",
                        help="Comma-separated dataset names, or 'all' (default: all)")
    parser.add_argument("--name", "-n", default="",
                        help="Experiment name prefix for output files")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the experiment plan without running")

    args = parser.parse_args()
    if args.all:
        args.models = "all"
        args.datasets = "all"

    print("=== RAG Batch Experiment ===\n")

    t_start = time.time()
    print("Loading queries...")
    queries = load_queries()
    print(f"  {len(queries)} queries loaded\n")

    run_experiments(
        queries=queries,
        models=args.models,
        top_k_values=args.top_k,
        datasets=args.datasets,
        name=args.name,
        yes=args.yes,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
