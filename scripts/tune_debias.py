"""
Hyperparameter tuning script for debias methods (IPS, Naive, MTIPS, MTDR).
Optimizes for MAE on eval by default (lower is better).

Usage:
    python scripts/tune_debias.py --estimator_name ips --data_name hs --alpha 0.5 --n_trials 20
    python scripts/tune_debias.py --estimator_name naive --data_name saferlhf --alpha 0.5 --n_trials 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np
import yaml


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class SearchSpace:
    # Common parameters
    lr_range: Tuple[float, float] = (1e-4, 2e-3)
    l2_reg_range: Tuple[float, float] = (1e-8, 1e-4)
    batch_size: Tuple[int, ...] = (256, 512, 1024)

    # IPS/MTIPS/MTDR specific
    clip_min: Tuple[float, ...] = (0.05, 0.1, 0.2)

    # Naive specific
    w_reg_naive: Tuple[float, ...] = (0.2, 0.5, 1.0)

    # MTIPS/MTDR specific
    w_prop: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    w_reg: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)

    # MTDR specific
    w_imp: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)


def _sample_params(rng: np.random.Generator, space: SearchSpace, estimator_name: str) -> dict:
    """Sample hyperparameters based on estimator type."""
    params = {
        "lr": _loguniform(rng, *space.lr_range),
        "l2_reg": _loguniform(rng, *space.l2_reg_range),
        "batch_size": int(rng.choice(space.batch_size)),
    }

    if estimator_name == "naive":
        params["w_reg"] = float(rng.choice(space.w_reg_naive))
    elif estimator_name == "ips":
        params["clip_min"] = float(rng.choice(space.clip_min))
    elif estimator_name == "mtips":
        params["clip_min"] = float(rng.choice(space.clip_min))
        params["w_prop"] = float(rng.choice(space.w_prop))
        params["w_reg"] = float(rng.choice(space.w_reg))
    elif estimator_name == "mtdr":
        params["clip_min"] = float(rng.choice(space.clip_min))
        params["w_prop"] = float(rng.choice(space.w_prop))
        params["w_imp"] = float(rng.choice(space.w_imp))
        params["w_reg"] = float(rng.choice(space.w_reg))

    return params


def _warmup_configs(estimator_name: str) -> List[dict]:
    """Deterministic starting points to avoid wasting early trials."""
    base = {
        "lr": 5e-4,
        "l2_reg": 1e-6,
        "batch_size": 512,
    }

    if estimator_name == "naive":
        warmups = [
            {**base, "w_reg": 1.0},
            {**base, "w_reg": 0.5},
            {**base, "w_reg": 0.2},
            {**base, "lr": 1e-3, "w_reg": 1.0},
            {**base, "lr": 2e-4, "w_reg": 1.0},
        ]
    elif estimator_name == "ips":
        warmups = [
            {**base, "clip_min": 0.1},
            {**base, "clip_min": 0.05},
            {**base, "clip_min": 0.2},
            {**base, "lr": 1e-3, "clip_min": 0.1},
            {**base, "lr": 2e-4, "clip_min": 0.1},
        ]
    elif estimator_name == "mtips":
        warmups = [
            {**base, "clip_min": 0.1, "w_prop": 1.0, "w_reg": 1.0},
            {**base, "clip_min": 0.1, "w_prop": 0.5, "w_reg": 1.0},
            {**base, "clip_min": 0.1, "w_prop": 1.0, "w_reg": 0.5},
            {**base, "clip_min": 0.05, "w_prop": 1.0, "w_reg": 1.0},
            {**base, "lr": 1e-3, "clip_min": 0.1, "w_prop": 1.0, "w_reg": 1.0},
        ]
    elif estimator_name == "mtdr":
        warmups = [
            {**base, "clip_min": 0.1, "w_prop": 1.0, "w_imp": 1.0, "w_reg": 1.0},
            {**base, "clip_min": 0.1, "w_prop": 0.5, "w_imp": 1.0, "w_reg": 1.0},
            {**base, "clip_min": 0.1, "w_prop": 1.0, "w_imp": 0.5, "w_reg": 1.0},
            {**base, "clip_min": 0.1, "w_prop": 1.0, "w_imp": 1.0, "w_reg": 0.5},
            {**base, "lr": 1e-3, "clip_min": 0.1, "w_prop": 1.0, "w_imp": 1.0, "w_reg": 1.0},
        ]
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

    return warmups


def _get_benchmark_script(estimator_name: str) -> str:
    """Get the benchmark script path for an estimator."""
    return f"models_debias/benchmark_{estimator_name}.py"


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_trial(trial_id: int, args, params: dict, trial_dir: str) -> Tuple[float, dict]:
    """Run a single trial and return (score, metrics)."""
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, "run.log")

    script_path = _get_benchmark_script(args.estimator_name)

    cmd = [
        sys.executable,
        script_path,
        "--data_name", args.data_name,
        "--alpha", str(args.alpha),
        "--rerun", "True",
        "--use_tqdm", "False",
        "--output_dir", args.work_dir,
        "--num_epochs", str(args.num_epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--lr", str(params["lr"]),
        "--l2_reg", str(params["l2_reg"]),
        "--batch_size", str(params["batch_size"]),
    ]

    # Add estimator-specific parameters
    if args.estimator_name == "naive":
        cmd += ["--w_reg", str(params["w_reg"])]
    elif args.estimator_name == "ips":
        cmd += ["--clip_min", str(params["clip_min"])]
    elif args.estimator_name == "mtips":
        cmd += [
            "--clip_min", str(params["clip_min"]),
            "--w_prop", str(params["w_prop"]),
            "--w_reg", str(params["w_reg"]),
        ]
    elif args.estimator_name == "mtdr":
        cmd += [
            "--clip_min", str(params["clip_min"]),
            "--w_prop", str(params["w_prop"]),
            "--w_imp", str(params["w_imp"]),
            "--w_reg", str(params["w_reg"]),
        ]

    env = os.environ.copy()
    # Uncomment to force CPU for reproducibility:
    # env["CUDA_VISIBLE_DEVICES"] = ""

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    perf_path = os.path.join(args.work_dir, "performance.yaml")
    if proc.returncode != 0 or not os.path.exists(perf_path):
        raise RuntimeError(f"Trial {trial_id} failed (code={proc.returncode}). See {log_path}")

    metrics = _read_yaml(perf_path)
    score = float(metrics.get(args.metric, float("inf")))  # For MAE, lower is better
    metrics["_trial_time_sec"] = float(dt)
    return score, metrics


def main():
    p = argparse.ArgumentParser(description="Hyperparameter tuning for debias methods")
    p.add_argument("--estimator_name", type=str, required=True,
                   choices=["ips", "naive", "mtips", "mtdr"],
                   help="Estimator to tune")
    p.add_argument("--data_name", type=str, default="hs",
                   help="Dataset name (hs, saferlhf, ufb)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Alpha parameter for bias simulation")
    p.add_argument("--n_trials", type=int, default=20,
                   help="Number of tuning trials")
    p.add_argument("--num_epochs", type=int, default=200,
                   help="Max training epochs per trial")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--metric", type=str, default="MAE on eval",
                   help="Metric to optimize (lower is better for MAE)")
    p.add_argument("--output_root", type=str, default="./results/tune",
                   help="Root directory for tuning results")
    args = p.parse_args()

    # Setup directories
    args.work_dir = os.path.join(args.output_root, args.estimator_name, "work",
                                  f"{args.data_name}_alpha{args.alpha}")
    trials_root = os.path.join(args.output_root, args.estimator_name, "trials",
                                f"{args.data_name}_alpha{args.alpha}")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(trials_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    space = SearchSpace()
    warmups = _warmup_configs(args.estimator_name)

    # For MAE, lower is better
    best = {"score": float("inf"), "trial_id": None, "params": None, "metrics": None}
    results_path = os.path.join(args.output_root, args.estimator_name,
                                 f"tune_results_{args.data_name}_alpha{args.alpha}.jsonl")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print(f"=" * 70)
    print(f"Tuning {args.estimator_name} on {args.data_name} (alpha={args.alpha})")
    print(f"Optimizing: {args.metric} (lower is better)")
    print(f"Trials: {args.n_trials}")
    print(f"=" * 70)

    for t in range(int(args.n_trials)):
        trial_id = t + 1
        if t < len(warmups):
            params = warmups[t]
        else:
            params = _sample_params(rng, space, args.estimator_name)
        trial_dir = os.path.join(trials_root, f"trial_{trial_id:03d}")

        print(f"\n[{trial_id:03d}/{args.n_trials:03d}] params={params}")
        try:
            score, metrics = _run_trial(trial_id, args, params, trial_dir)
        except Exception as e:
            print(f"[{trial_id:03d}] FAILED: {e}")
            rec = {"trial_id": trial_id, "status": "failed", "params": params, "error": str(e)}
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        rec = {
            "trial_id": trial_id,
            "status": "ok",
            "score": score,
            "metric": args.metric,
            "params": params,
            "metrics": metrics,
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mae_test = metrics.get("MAE on test", float("nan"))
        auroc_test = metrics.get("AUROC on test", float("nan"))
        print(f"[{trial_id:03d}] {args.metric}={score:.4f}  MAE_test={mae_test:.4f}  AUROC_test={auroc_test:.4f}")

        # For MAE, lower is better
        if score < best["score"]:
            best = {"score": score, "trial_id": trial_id, "params": params, "metrics": metrics}

            # Snapshot best artifacts
            for name in ("best_model.pth", "performance.yaml", "config.yaml"):
                src = os.path.join(args.work_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(trial_dir, name))

            best_path = os.path.join(args.output_root, args.estimator_name,
                                      f"best_{args.data_name}_alpha{args.alpha}.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

            print(f"[{trial_id:03d}] NEW BEST -> {best_path}")

    if best["trial_id"] is None:
        raise SystemExit("No successful trials.")

    print("\n" + "=" * 70)
    print("=== Best Trial ===")
    print(f"trial_id: {best['trial_id']}")
    print(f"{args.metric}: {best['score']:.4f}")
    print(f"params: {best['params']}")
    if best["metrics"]:
        print(f"MAE on test: {best['metrics'].get('MAE on test'):.4f}")
        print(f"AUROC on test: {best['metrics'].get('AUROC on test'):.4f}")
    print("=" * 70)

    # Print params in format ready for copy-paste to dataset_defaults
    print("\n# Copy-paste to dataset_defaults:")
    print(f'"{args.data_name}": {{')
    print(f'    "alpha": {args.alpha},')
    for k, v in best["params"].items():
        if isinstance(v, float):
            print(f'    "{k}": {v:.6g},')
        else:
            print(f'    "{k}": {v},')
    print('},')


if __name__ == "__main__":
    main()
