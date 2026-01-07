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
    # CounterIF paper hyperparams: alpha ~ target_percentile/100, beta ~ hn_percentile/100.
    # In this repo's PU simulation, very high target_percentile and/or very high hn_percentile
    # can collapse UN (or reduce pairwise terms), so include some extremes for robustness.
    target_percentile: tuple[float, ...] = (70.0, 80.0, 85.0, 90.0, 95.0, 97.0, 99.0, 99.5)
    hn_percentile: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 95.0, 99.0)
    # Training/runtime knobs (excluding hidden_dim).
    num_epochs: tuple[int, ...] = (60, 80, 120, 200)
    patience: tuple[int, ...] = (5, 8, 10, 15, 20)
    monitor_on: tuple[str, ...] = ("val", "train")
    batch_size_point: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    batch_size_pair: tuple[int, ...] = (256, 512, 1024, 2048)
    batch_size_ipm: tuple[int, ...] = (32, 64, 128, 256, 512)
    ocsvm_batch_size: tuple[int, ...] = (2048, 4096, 8192, 16384)
    pair_max_dp_he: tuple[int, ...] = (0, 2000, 5000, 10000, 20000)
    pair_max_un_he: tuple[int, ...] = (0, 2000, 5000, 10000, 20000)
    pair_max_hu_un: tuple[int, ...] = (0, 2000, 5000, 10000, 20000)


def _sample_params(
    rng: np.random.Generator,
    space: SearchSpace,
    *,
    tune_groups: bool,
    tune_training: bool,
    tune_batches: bool,
    tune_pairs: bool,
    tune_ocsvm: bool,
) -> dict:
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
    if tune_training:
        params["num_epochs"] = int(rng.choice(space.num_epochs))
        params["patience"] = int(rng.choice(space.patience))
        params["monitor_on"] = str(rng.choice(space.monitor_on))
    if tune_batches:
        params["batch_size_point"] = int(rng.choice(space.batch_size_point))
        params["batch_size_pair"] = int(rng.choice(space.batch_size_pair))
        params["batch_size_ipm"] = int(rng.choice(space.batch_size_ipm))
    if tune_ocsvm:
        params["ocsvm_batch_size"] = int(rng.choice(space.ocsvm_batch_size))
    if tune_pairs:
        params["pair_max_dp_he"] = int(rng.choice(space.pair_max_dp_he))
        params["pair_max_un_he"] = int(rng.choice(space.pair_max_un_he))
        params["pair_max_hu_un"] = int(rng.choice(space.pair_max_hu_un))
    return params


def _warmup_configs(*, tune_groups: bool, tune_training: bool, tune_batches: bool, tune_pairs: bool, tune_ocsvm: bool) -> list[dict]:
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
    if tune_training:
        base.update({"num_epochs": 120, "patience": 10, "monitor_on": "val"})
    if tune_batches:
        base.update({"batch_size_point": 512, "batch_size_pair": 1024, "batch_size_ipm": 256})
    if tune_ocsvm:
        base.update({"ocsvm_batch_size": 8192})
    if tune_pairs:
        base.update({"pair_max_dp_he": 20000, "pair_max_un_he": 20000, "pair_max_hu_un": 20000})

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
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 99.0, "hn_percentile": 5.0},
            {**base, "lambda_pair": 0.0, "lambda_ipm": 0.0, "target_percentile": 99.0, "hn_percentile": 20.0},
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

    num_epochs = int(params.get("num_epochs", args.num_epochs))
    patience = int(params.get("patience", args.patience))

    cmd = [
        sys.executable,
        "models_debias_pu/benchmark_counterif.py",
        "--data_name",
        args.data_name,
        "--unbiased",
        "True" if args.unbiased else "False",
        "--alpha",
        str(args.alpha),
        "--rerun",
        "True",
        "--use_tqdm",
        "False",
        "--output_dir",
        args.work_dir,
        "--num_epochs",
        str(num_epochs),
        "--patience",
        str(patience),
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
    if "monitor_on" in params:
        cmd += ["--monitor_on", str(params["monitor_on"])]
    if "batch_size_point" in params:
        cmd += ["--batch_size_point", str(params["batch_size_point"])]
    if "batch_size_pair" in params:
        cmd += ["--batch_size_pair", str(params["batch_size_pair"])]
    if "batch_size_ipm" in params:
        cmd += ["--batch_size_ipm", str(params["batch_size_ipm"])]
    if "ocsvm_batch_size" in params:
        cmd += ["--ocsvm_batch_size", str(params["ocsvm_batch_size"])]
    for k in ("pair_max_dp_he", "pair_max_un_he", "pair_max_hu_un"):
        if k in params:
            cmd += [f"--{k}", str(params[k])]

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
    p.add_argument("--unbiased", action="store_true", help="Tune CounterIF on unbiased PU data (pu_unbiased.safetensors)")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="AUROC on eval", help="Metric key in performance.yaml")
    p.add_argument("--tune_groups", action="store_true", help="Also tune OCSVM percentiles (slower)")
    p.add_argument("--tune_training", action="store_true", help="Also tune num_epochs/patience/monitor_on")
    p.add_argument("--tune_batches", action="store_true", help="Also tune batch sizes")
    p.add_argument("--tune_pairs", action="store_true", help="Also tune pair_max_*")
    p.add_argument("--tune_ocsvm", action="store_true", help="Also tune ocsvm_batch_size")
    p.add_argument("--tune_all", action="store_true", help="Shortcut: enable all tuning knobs except hidden_dim")
    p.add_argument("--output_root", type=str, default="./results/tune/counterif")
    args = p.parse_args()

    if args.tune_all:
        args.tune_groups = True
        args.tune_training = True
        args.tune_batches = True
        args.tune_pairs = True
        args.tune_ocsvm = True

    # shared work_dir to reuse OCSVM group cache across trials (when hyperparams fixed)
    run_id = f"{args.data_name}_unbiased" if args.unbiased else f"{args.data_name}_alpha{args.alpha}"
    args.work_dir = os.path.join(args.output_root, "work", run_id)
    trials_root = os.path.join(args.output_root, "trials", run_id)
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
    warmups = _warmup_configs(
        tune_groups=bool(args.tune_groups),
        tune_training=bool(args.tune_training),
        tune_batches=bool(args.tune_batches),
        tune_pairs=bool(args.tune_pairs),
        tune_ocsvm=bool(args.tune_ocsvm),
    )

    best = {"score": float("-inf"), "trial_id": None, "params": None, "metrics": None}
    results_path = os.path.join(args.output_root, f"tune_results_{run_id}.jsonl")

    for t in range(int(args.n_trials)):
        trial_id = t + 1
        if t < len(warmups):
            params = warmups[t]
        else:
            params = _sample_params(
                rng,
                space,
                tune_groups=bool(args.tune_groups),
                tune_training=bool(args.tune_training),
                tune_batches=bool(args.tune_batches),
                tune_pairs=bool(args.tune_pairs),
                tune_ocsvm=bool(args.tune_ocsvm),
            )
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

            best_path = os.path.join(args.output_root, f"best_{run_id}.json")
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
