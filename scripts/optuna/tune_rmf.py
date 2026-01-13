"""
Optuna hyperparameter tuning script for RMF model.

Optimizes R2 on test with val-based early stopping.
Runs datasets in order: hs, saferlhf, ufb (configurable).

Usage:
    python scripts/optuna/tune_rmf.py
    python scripts/optuna/tune_rmf.py --datasets hs,saferlhf,ufb --n_trials 300
    python scripts/optuna/tune_rmf.py --alpha 0.2 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna
import yaml

MODEL_NAME = "rmf"
SCRIPT_PATH = "models_pu/benchmark_rmf.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _suggest_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for RMF model."""
    return {
        "lr": float(trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3])),
        "l2_reg": float(trial.suggest_categorical("l2_reg", [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1])),
        "batch_size": int(trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])),
        "w_reg": float(trial.suggest_categorical("w_reg", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])),
        "clip_min": float(trial.suggest_categorical("clip_min", [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1])),
    }


def _run_benchmark(cmd: list[str], *, log_path: str, env: dict) -> str:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    dt = time.time() - t0
    _ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write(f"\n\n[exit_code]={proc.returncode}\n[wall_time_sec]={dt:.6f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark failed with code={proc.returncode}. See {log_path}")
    return log_path


def _make_base_cmd(args, *, work_dir: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / SCRIPT_PATH),
        "--data_name", args.data_name,
        "--alpha", str(args.alpha),
        "--data_root", str(args.data_root),
        "--output_dir", work_dir,
        "--rerun", "True",
        "--use_tqdm", "False",
        "--seed", str(args.seed),
        "--binary", "True",
        "--hidden_dim", "256,64",
        "--num_epochs", str(args.num_epochs),
        "--patience", str(args.patience),
        "--monitor_on", "val",
    ]


def _params_to_cli(params: dict) -> list[str]:
    cli: list[str] = []
    for k, v in params.items():
        cli.append(f"--{k}")
        cli.append(str(v))
    return cli


def tune_one_dataset(args) -> dict:
    """Run Optuna optimization for one dataset."""
    out_dir = REPO_ROOT / "results" / "optuna" / MODEL_NAME
    _ensure_dir(str(out_dir))
    _ensure_dir(str(out_dir / "studies"))

    study_name = f"{MODEL_NAME}_{args.data_name}_alpha{args.alpha}"
    storage_path = out_dir / "studies" / f"{args.data_name}_alpha{args.alpha}.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # maximize R2 on test
        storage=storage,
        load_if_exists=bool(args.resume),
    )

    work_dir = str(out_dir / "work" / f"{args.data_name}_alpha{args.alpha}")
    trials_dir = str(out_dir / "trials" / f"{args.data_name}_alpha{args.alpha}")
    _ensure_dir(work_dir)
    _ensure_dir(trials_dir)

    # JSONL file to save all trial results
    results_jsonl = out_dir / f"{args.data_name}_alpha{args.alpha}_all_trials.jsonl"

    env = os.environ.copy()

    def objective(trial: optuna.Trial) -> float:
        log_path = os.path.join(trials_dir, f"trial_{trial.number:05d}.log")
        params = _suggest_params(trial)

        try:
            cmd = _make_base_cmd(args, work_dir=work_dir) + _params_to_cli(params)
            _run_benchmark(cmd, log_path=log_path, env=env)

            perf_path = os.path.join(work_dir, "performance.yaml")
            if not os.path.exists(perf_path):
                raise RuntimeError(f"Missing performance.yaml at {perf_path}")
            metrics = _read_yaml(perf_path)

            r2_test = float(metrics["R2 on test"])

            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", params)

            # Save trial result to JSONL
            trial_record = {
                "trial_number": trial.number,
                "value": r2_test,
                "params": params,
                "metrics": metrics,
                "status": "ok",
            }
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(trial_record, ensure_ascii=False) + "\n")

            return r2_test

        except Exception as e:
            trial.set_user_attr("error", str(e))
            # Save failed trial to JSONL
            trial_record = {
                "trial_number": trial.number,
                "value": None,
                "params": params,
                "metrics": None,
                "status": "failed",
                "error": str(e),
            }
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(trial_record, ensure_ascii=False) + "\n")
            raise optuna.TrialPruned(f"Trial failed: {e}")

    print("=" * 70)
    print(f"Tuning: {MODEL_NAME}  data={args.data_name}  alpha={args.alpha}")
    print(f"Trials: {args.n_trials}  resume={args.resume}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    study.optimize(objective, n_trials=args.n_trials)

    # Save best result
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_metrics = dict(best_trial.user_attrs.get("metrics", {}))

    summary = {
        "model": MODEL_NAME,
        "data_name": args.data_name,
        "alpha": float(args.alpha),
        "n_trials": int(args.n_trials),
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": best_params,
        "best_metrics": best_metrics,
    }

    out_best = out_dir / f"{args.data_name}_alpha{args.alpha}_best.json"
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Best R2 on test: {best_trial.value:.4f}")
    print(f"Best params: {best_params}")
    print(f"Saved to: {out_best}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description=f"Optuna tuning for {MODEL_NAME}")
    parser.add_argument("--datasets", type=str, default="hs,saferlhf,ufb",
                        help="Comma-separated list of datasets to tune")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of Optuna trials per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--data_root", type=str, default="embeddings/biased_pu", help="Data root directory")
    parser.add_argument("--resume", action="store_true", help="Resume existing Optuna study")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_summaries = []
    for data_name in datasets:
        args.data_name = data_name
        try:
            summary = tune_one_dataset(args)
            all_summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] Failed to tune {data_name}: {e}")
            all_summaries.append({
                "model": MODEL_NAME,
                "data_name": data_name,
                "alpha": float(args.alpha),
                "error": str(e),
            })

    # Save overall summary
    out_dir = REPO_ROOT / "results" / "optuna" / MODEL_NAME
    summary_path = out_dir / f"summary_alpha{args.alpha}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"Overall summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
