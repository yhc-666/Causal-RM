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
    lr_low: float = 5e-5
    lr_high: float = 5e-3
    l2_low: float = 1e-8
    l2_high: float = 1e-3

    w_reg: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    clip_min: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
    batch_size: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    num_neg: tuple[int, ...] = (1, 2, 5, 10, 20, 50)

    num_epochs: tuple[int, ...] = (60, 80, 120, 200)
    patience: tuple[int, ...] = (5, 8, 10, 15, 20, 30)
    monitor_on: tuple[str, ...] = ("val", "train")


def _warmup_configs() -> list[dict]:
    base = {
        "lr": 5e-4,
        "l2_reg": 1e-6,
        "w_reg": 1.0,
        "clip_min": 1e-8,
        "batch_size": 512,
        "num_neg": 10,
        "num_epochs": 120,
        "patience": 10,
        "monitor_on": "val",
    }
    return [
        base,
        {**base, "lr": 2e-4},
        {**base, "l2_reg": 1e-5},
        {**base, "w_reg": 0.5},
        {**base, "num_neg": 5},
        {**base, "batch_size": 256},
    ]


def _sample_params(rng: np.random.Generator, space: SearchSpace) -> dict:
    return {
        "lr": _loguniform(rng, space.lr_low, space.lr_high),
        "l2_reg": _loguniform(rng, space.l2_low, space.l2_high),
        "w_reg": float(rng.choice(space.w_reg)),
        "clip_min": float(rng.choice(space.clip_min)),
        "batch_size": int(rng.choice(space.batch_size)),
        "num_neg": int(rng.choice(space.num_neg)),
        "num_epochs": int(rng.choice(space.num_epochs)),
        "patience": int(rng.choice(space.patience)),
        "monitor_on": str(rng.choice(space.monitor_on)),
    }


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_trial(trial_id: int, args, params: dict, trial_dir: str) -> tuple[float, dict]:
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, "run.log")

    cmd = [
        sys.executable,
        "models_pu/benchmark_ubpr.py",
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
        "--seed",
        str(args.seed),
        "--lr",
        str(params["lr"]),
        "--l2_reg",
        str(params["l2_reg"]),
        "--w_reg",
        str(params["w_reg"]),
        "--clip_min",
        str(params["clip_min"]),
        "--batch_size",
        str(params["batch_size"]),
        "--num_neg",
        str(params["num_neg"]),
        "--num_epochs",
        str(params["num_epochs"]),
        "--patience",
        str(params["patience"]),
        "--monitor_on",
        str(params["monitor_on"]),
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
    if args.metric not in metrics:
        raise KeyError(f"Metric '{args.metric}' not found in {perf_path}. Available: {list(metrics.keys())}")
    value = float(metrics[args.metric])
    metrics["_trial_time_sec"] = float(dt)
    return value, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, default="hs")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="MAE on eval", help="Metric key in performance.yaml")
    p.add_argument("--output_root", type=str, default="./results/tune/ubpr")
    args = p.parse_args()

    # shared work_dir (ubpr doesn't have reusable caches, but keeps outputs simple)
    args.work_dir = os.path.join(args.output_root, "work", f"{args.data_name}_alpha{args.alpha}")
    trials_root = os.path.join(args.output_root, "trials", f"{args.data_name}_alpha{args.alpha}")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(trials_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    space = SearchSpace()
    warmups = _warmup_configs()

    best = {"value": float("inf"), "trial_id": None, "params": None, "metrics": None}
    results_path = os.path.join(args.output_root, f"tune_results_{args.data_name}_alpha{args.alpha}.jsonl")

    for t in range(int(args.n_trials)):
        trial_id = t + 1
        params = warmups[t] if t < len(warmups) else _sample_params(rng, space)
        trial_dir = os.path.join(trials_root, f"trial_{trial_id:03d}")

        print(f"[{trial_id:03d}/{args.n_trials:03d}] params={params}")
        try:
            value, metrics = _run_trial(trial_id, args, params, trial_dir)
        except Exception as e:
            print(f"[{trial_id:03d}] FAILED: {e}")
            rec = {"trial_id": trial_id, "status": "failed", "params": params, "error": str(e)}
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        rec = {
            "trial_id": trial_id,
            "status": "ok",
            "metric": args.metric,
            "value": value,
            "params": params,
            "metrics": metrics,
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mae_test = metrics.get("MAE on test", float("nan"))
        print(f"[{trial_id:03d}] {args.metric}={value:.6f}  MAE_test={mae_test:.6f}")

        if value < best["value"]:
            best = {"value": value, "trial_id": trial_id, "params": params, "metrics": metrics}

            # Snapshot best artifacts from work_dir
            for name in ("best_model.pth", "performance.yaml", "config.yaml"):
                src = os.path.join(args.work_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(trial_dir, name))

            best_path = os.path.join(args.output_root, f"best_{args.data_name}_alpha{args.alpha}.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

            print(f"[{trial_id:03d}] NEW BEST -> {best_path}")

    if best["trial_id"] is None:
        raise SystemExit("No successful trials.")

    print("\n=== Best Trial (min) ===")
    print(f"trial_id: {best['trial_id']}")
    print(f"{args.metric}: {best['value']:.6f}")
    print(f"params: {best['params']}")
    if best["metrics"] is not None:
        print(f"MAE on test: {best['metrics'].get('MAE on test')}")


if __name__ == "__main__":
    main()

