import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import yaml


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    low = float(low)
    high = float(high)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class SearchSpace:
    lambda_point: tuple[float, ...] = (0.1, 1.0, 10.0)
    lambda_pair: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 5.0)
    lambda_ipm: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 5.0)
    ipm_lam: tuple[float, ...] = (1.0, 10.0, 50.0)
    ipm_its: tuple[int, ...] = (5, 10, 20)
    ipm_p: tuple[float, ...] = (0.2, 0.5, 0.8)
    target_percentile: tuple[float, ...] = (80.0, 85.0, 90.0, 95.0, 99.0)
    hn_percentile: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)


def _sample_params(rng: np.random.Generator, space: SearchSpace, tune_groups: bool) -> dict:
    params = {
        "lr": _loguniform(rng, 1e-4, 2e-3),
        "l2_reg": _loguniform(rng, 1e-8, 1e-4),
        "lambda_point": float(rng.choice(space.lambda_point)),
        "lambda_pair": float(rng.choice(space.lambda_pair)),
        "lambda_ipm": float(rng.choice(space.lambda_ipm)),
        "ipm_lam": float(rng.choice(space.ipm_lam)),
        "ipm_its": int(rng.choice(space.ipm_its)),
        "ipm_p": float(rng.choice(space.ipm_p)),
    }
    if tune_groups:
        params["target_percentile"] = float(rng.choice(space.target_percentile))
        params["hn_percentile"] = float(rng.choice(space.hn_percentile))
    return params


def _warmup_configs(tune_groups: bool) -> list[dict]:
    """
    Deterministic starting points to avoid wasting early trials.
    These are especially important here because the best region often has λ_pair/λ_ipm near 0.
    """
    base = {
        "lr": 5e-4,
        "l2_reg": 1e-6,
        "lambda_point": 1.0,
        "ipm_lam": 10.0,
        "ipm_its": 10,
        "ipm_p": 0.5,
    }
    if not tune_groups:
        warmups = [
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0},
            {**base, "lambda_pair": 0.01, "lambda_ipm": 0.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.01},
            {**base, "lambda_pair": 0.1, "lambda_ipm": 0.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.1},
        ]
    else:
        warmups = [
            # In this repo's PU simulation, pushing target_percentile high makes HE cover most y==0,
            # which often stabilizes training.
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 99.0, "hn_percentile": 10.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 95.0, "hn_percentile": 10.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 90.0, "hn_percentile": 10.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 80.0, "hn_percentile": 10.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 70.0, "hn_percentile": 10.0},
        ]
    return warmups


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_trial(trial_id: int, args, params: dict, trial_dir: str) -> tuple[float, dict]:
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, "run.log")

    cmd = [
        sys.executable,
        "models_debias_pu/benchmark_counterif.py",
        "--data_name",
        args.data_name,
        "--alpha",
        str(args.alpha),
        "--rerun",
        "True",
        "--use_tqdm",
        "False",
        "--output_dir",
        args.work_dir,
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--lr",
        str(params["lr"]),
        "--l2_reg",
        str(params["l2_reg"]),
        "--lambda_point",
        str(params["lambda_point"]),
        "--lambda_pair",
        str(params["lambda_pair"]),
        "--lambda_ipm",
        str(params["lambda_ipm"]),
        "--ipm_lam",
        str(params["ipm_lam"]),
        "--ipm_its",
        str(params["ipm_its"]),
        "--ipm_p",
        str(params["ipm_p"]),
    ]
    if args.tune_groups:
        cmd += [
            "--target_percentile",
            str(params["target_percentile"]),
            "--hn_percentile",
            str(params["hn_percentile"]),
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for reproducibility

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    perf_path = os.path.join(args.work_dir, "performance.yaml")
    if proc.returncode != 0 or (not os.path.exists(perf_path)):
        raise RuntimeError(f"Trial {trial_id} failed (code={proc.returncode}). See {log_path}")

    metrics = _read_yaml(perf_path)
    score = float(metrics.get(args.metric, float("-inf")))
    metrics["_trial_time_sec"] = float(dt)
    return score, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, default="hs")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="AUROC on eval", help="Metric key in performance.yaml")
    p.add_argument("--tune_groups", action="store_true", help="Also tune OCSVM percentiles (slower)")
    p.add_argument("--output_root", type=str, default="./results/tune/counterif")
    args = p.parse_args()

    # shared work_dir to reuse OCSVM group cache across trials (when hyperparams fixed)
    args.work_dir = os.path.join(args.output_root, "work", f"{args.data_name}_alpha{args.alpha}")
    trials_root = os.path.join(args.output_root, "trials", f"{args.data_name}_alpha{args.alpha}")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(trials_root, exist_ok=True)

    # Optional Optuna
    has_optuna = False
    try:
        import optuna  # noqa: F401
        has_optuna = True
    except Exception:
        has_optuna = False

    if not has_optuna:
        print("Optuna not installed; falling back to random search.")
        print("Tip: install optuna then rerun for smarter search.")

    rng = np.random.default_rng(args.seed)
    space = SearchSpace()
    warmups = _warmup_configs(tune_groups=bool(args.tune_groups))

    best = {"score": float("-inf"), "trial_id": None, "params": None, "metrics": None}
    results_path = os.path.join(args.output_root, f"tune_results_{args.data_name}_alpha{args.alpha}.jsonl")

    for t in range(int(args.n_trials)):
        trial_id = t + 1
        if t < len(warmups):
            params = warmups[t]
        else:
            params = _sample_params(rng, space, tune_groups=bool(args.tune_groups))
        trial_dir = os.path.join(trials_root, f"trial_{trial_id:03d}")

        print(f"[{trial_id:03d}/{args.n_trials:03d}] params={params}")
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

        print(f"[{trial_id:03d}] {args.metric}={score:.4f}  AUROC_test={metrics.get('AUROC on test', float('nan')):.4f}")

        if score > best["score"]:
            best = {"score": score, "trial_id": trial_id, "params": params, "metrics": metrics}

            # Snapshot best artifacts from work_dir
            for name in ("best_model.pth", "performance.yaml", "config.yaml", "counterif_groups_cache.npz"):
                src = os.path.join(args.work_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(trial_dir, name))

            best_path = os.path.join(args.output_root, f"best_{args.data_name}_alpha{args.alpha}.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

            print(f"[{trial_id:03d}] NEW BEST -> {best_path}")

    if best["trial_id"] is None:
        raise SystemExit("No successful trials.")

    print("\n=== Best Trial ===")
    print(f"trial_id: {best['trial_id']}")
    print(f"{args.metric}: {best['score']:.4f}")
    print(f"params: {best['params']}")
    if best["metrics"] is not None:
        print(f"AUROC on test: {best['metrics'].get('AUROC on test')}")
        print(f"NLL on test: {best['metrics'].get('NLL on test')}")


if __name__ == "__main__":
    main()
