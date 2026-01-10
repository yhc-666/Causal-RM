#!/usr/bin/env python3
"""
Grid Search Results Summarizer

Summarizes grid search results across different models and datasets.
Finds the best hyperparameters based on a specified metric.

Usage:
    python scripts/grid_search/summarize.py --model ips --metric "R2 on test" --top 5
    python scripts/grid_search/summarize.py --results_dir results/grid_search/naive --metric "AUROC on test"
"""

import argparse
import yaml
from pathlib import Path
from collections import defaultdict


# Metrics to display in output
DISPLAY_METRICS = ["R2 on test", "AUROC on test", "MAE on test", "RMSE on test"]

# Config keys to display (common hyperparameters)
DISPLAY_CONFIG_KEYS = [
    "lr",
    "batch_size",
    "l2_reg",
    "w_reg",
    "lamp",
    "calibration_sharpen_k",
    "alpha",
    "hidden_dim",
]


def load_results(results_dir: Path) -> list:
    """
    Load all performance.yaml and config.yaml from results directory.

    Returns:
        List of dicts with 'metrics', 'config', 'path' keys
    """
    results = []

    for perf_file in results_dir.glob("*/performance.yaml"):
        exp_dir = perf_file.parent
        config_file = exp_dir / "config.yaml"

        try:
            with open(perf_file) as f:
                metrics = yaml.safe_load(f)

            config = {}
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)

            results.append({
                "metrics": metrics,
                "config": config,
                "path": str(exp_dir)
            })
        except Exception as e:
            print(f"Warning: Failed to load {perf_file}: {e}")

    return results


def group_by_dataset(results: list) -> dict:
    """Group results by dataset name."""
    grouped = defaultdict(list)
    for r in results:
        dataset = r["config"].get("data_name", "unknown")
        grouped[dataset].append(r)
    return grouped


def sort_results(results: list, metric: str, ascending: bool = False) -> list:
    """Sort results by metric value."""
    def get_metric(r):
        val = r["metrics"].get(metric)
        if val is None:
            return float('-inf') if not ascending else float('inf')
        return val

    return sorted(results, key=get_metric, reverse=not ascending)


def format_value(val, width=10):
    """Format a value for display."""
    if val is None:
        return "-".center(width)
    if isinstance(val, float):
        if abs(val) < 0.0001 or abs(val) >= 1000:
            return f"{val:.2e}".rjust(width)
        return f"{val:.4f}".rjust(width)
    return str(val).rjust(width)


def print_results(grouped_results: dict, metric: str, top_n: int, ascending: bool = False):
    """Print formatted results table."""
    # Determine if metric should be minimized
    lower_is_better = ["MAE", "RMSE", "NLL", "loss"]
    ascending = any(m in metric for m in lower_is_better)

    print(f"\nRanking by: {metric} ({'lower is better' if ascending else 'higher is better'})")
    print("=" * 100)

    for dataset in sorted(grouped_results.keys()):
        results = grouped_results[dataset]
        sorted_results = sort_results(results, metric, ascending)[:top_n]

        if not sorted_results:
            continue

        print(f"\nDataset: {dataset}")
        print("-" * 100)

        # Header
        header = "Rank  "
        for m in DISPLAY_METRICS:
            short_name = m.replace(" on test", "")
            header += f"{short_name:>10}  "
        for key in DISPLAY_CONFIG_KEYS:
            if any(key in r["config"] for r in sorted_results):
                header += f"{key:>12}  "
        print(header)
        print("-" * 100)

        # Rows
        for rank, r in enumerate(sorted_results, 1):
            row = f"{rank:<4}  "
            for m in DISPLAY_METRICS:
                row += format_value(r["metrics"].get(m)) + "  "
            for key in DISPLAY_CONFIG_KEYS:
                if any(key in r["config"] for r in sorted_results):
                    row += format_value(r["config"].get(key), 12) + "  "
            print(row)


def save_best_config(grouped_results: dict, metric: str, output_path: Path, ascending: bool = False):
    """Save best config for each dataset to YAML file."""
    lower_is_better = ["MAE", "RMSE", "NLL", "loss"]
    ascending = any(m in metric for m in lower_is_better)

    best_configs = {}

    for dataset, results in grouped_results.items():
        sorted_results = sort_results(results, metric, ascending)
        if not sorted_results:
            continue

        best = sorted_results[0]

        # Extract relevant config keys
        config_subset = {k: best["config"].get(k) for k in DISPLAY_CONFIG_KEYS if k in best["config"]}

        # Extract display metrics
        metrics_subset = {m: best["metrics"].get(m) for m in DISPLAY_METRICS if m in best["metrics"]}

        best_configs[dataset] = {
            "config": config_subset,
            "metrics": metrics_subset,
            "path": best["path"]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_configs, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest configs saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize grid search results")
    parser.add_argument("--model", type=str, help="Model name (e.g., naive, ips, dr)")
    parser.add_argument("--results_dir", type=str, help="Custom results directory path")
    parser.add_argument("--metric", type=str, default="R2 on test",
                        help="Metric to rank by (default: 'R2 on test')")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--root", type=str, default="results/grid_search",
                        help="Root directory for grid search results")
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.model:
        results_dir = Path(args.root) / args.model
    else:
        parser.error("Either --model or --results_dir must be specified")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    model_name = args.model or results_dir.name
    print(f"Model: {model_name}")
    print(f"Results directory: {results_dir}")

    # Load and process results
    results = load_results(results_dir)
    if not results:
        print("No results found!")
        return 1

    print(f"Found {len(results)} experiment results")

    # Group by dataset
    grouped = group_by_dataset(results)

    # Print results
    print_results(grouped, args.metric, args.top)

    # Save best config
    output_path = results_dir / "best_config.yaml"
    save_best_config(grouped, args.metric, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
