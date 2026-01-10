"""
Optuna multi-objective tuner for this repo's benchmark_*.py scripts.

Goals:
  - Tune (potentially many) benchmarks on a given dataset + alpha.
  - Keep model backbone fixed: do NOT tune hidden_dim (pass a fixed value).
  - Optimize multi-objective: MAE on test (min), RMSE on test (min), R2 on test (max).

Example:
  python scripts/tune_benchmarks_optuna.py --data_name hs --alpha 0.5 --models ips,dr,counterif --n_trials 50
  python scripts/tune_benchmarks_optuna.py --data_name hs --alpha 0.2 --models all --n_trials 200

Notes:
  - This script *tunes on test metrics* (as requested). This leaks test information; use with care.
  - Each Optuna study is stored as a sqlite DB under `--out_dir`, so you can resume with --resume True.
  - Default sampler is NSGA-II (recommended for multi-objective).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import optuna
import yaml


OBJECTIVES_DEFAULT = ["MAE on test", "RMSE on test", "R2 on test"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resolve_path(path_str: str | os.PathLike[str]) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _objective_directions(objectives: list[str]) -> list[str]:
    directions: list[str] = []
    for obj in objectives:
        key = str(obj).lower()
        if "r2" in key or "auroc" in key or "ndcg" in key or "pearson" in key or "f1" in key or "recall" in key:
            directions.append("maximize")
        else:
            directions.append("minimize")
    return directions


def _select_representative_trial(
    pareto_trials: list[optuna.trial.FrozenTrial],
    objectives: list[str],
) -> optuna.trial.FrozenTrial:
    """
    Pick ONE trial from Pareto front.

    Default rule for requested objectives:
      - min RMSE on test
      - tie-break: min MAE on test
      - tie-break: max R2 on test
    """
    name_to_idx = {name: i for i, name in enumerate(objectives)}
    idx_mae = name_to_idx.get("MAE on test", 0)
    idx_rmse = name_to_idx.get("RMSE on test", 1 if len(objectives) > 1 else 0)
    idx_r2 = name_to_idx.get("R2 on test", 2 if len(objectives) > 2 else 0)

    def key(t: optuna.trial.FrozenTrial):
        vals = list(t.values) if t.values is not None else []
        mae = float(vals[idx_mae]) if idx_mae < len(vals) else float("inf")
        rmse = float(vals[idx_rmse]) if idx_rmse < len(vals) else float("inf")
        r2 = float(vals[idx_r2]) if idx_r2 < len(vals) else float("-inf")
        return (rmse, mae, -r2)

    return min(pareto_trials, key=key)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    script_path: str
    fixed_args: Tuple[str, ...] = ()
    suggest_params: Callable[[optuna.Trial], Dict[str, object]] | None = None


def _suggest_common(
    trial: optuna.Trial,
    *,
    lr_low: float = 1e-5,
    lr_high: float = 3e-3,
    l2_low: float = 1e-8,
    l2_high: float = 1e-4,
    batch_sizes: Tuple[int, ...] = (256, 512, 1024, 2048),
) -> dict:
    return {
        "lr": float(trial.suggest_float("lr", lr_low, lr_high, log=True)),
        "l2_reg": float(trial.suggest_float("l2_reg", l2_low, l2_high, log=True)),
        "batch_size": int(trial.suggest_categorical("batch_size", list(batch_sizes))),
    }


def _suggest_with_w_reg(trial: optuna.Trial, *, w_reg: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) -> dict:
    params = _suggest_common(trial)
    params["w_reg"] = float(trial.suggest_categorical("w_reg", list(w_reg)))
    return params


def _suggest_with_clip_min(trial: optuna.Trial, *, clip_min: Tuple[float, ...]) -> dict:
    params = _suggest_common(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", list(clip_min)))
    return params


def _suggest_rmf_like(trial: optuna.Trial) -> dict:
    params = _suggest_with_w_reg(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2]))
    return params


def _suggest_upu_like(trial: optuna.Trial) -> dict:
    params = _suggest_with_w_reg(trial)
    params["class_prior"] = float(trial.suggest_float("class_prior", 0.05, 0.95))
    return params


def _suggest_bpr_like(trial: optuna.Trial) -> dict:
    params = _suggest_with_w_reg(trial)
    params["num_neg"] = int(trial.suggest_categorical("num_neg", [1, 2, 5, 10, 20, 50]))
    return params


def _suggest_ubpr_like(trial: optuna.Trial) -> dict:
    params = _suggest_bpr_like(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2]))
    return params


def _suggest_mtips(trial: optuna.Trial) -> dict:
    params = _suggest_common(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", [0.01, 0.05, 0.1, 0.2]))
    params["w_prop"] = float(trial.suggest_categorical("w_prop", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    params["w_reg"] = float(trial.suggest_categorical("w_reg", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    return params


def _suggest_mtdr(trial: optuna.Trial) -> dict:
    params = _suggest_mtips(trial)
    params["w_imp"] = float(trial.suggest_categorical("w_imp", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    return params


def _suggest_dr(trial: optuna.Trial) -> dict:
    params = _suggest_common(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", [0.01, 0.05, 0.1, 0.2]))
    params["w_imp"] = float(trial.suggest_categorical("w_imp", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    params["w_reg"] = float(trial.suggest_categorical("w_reg", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    params["l2_imp"] = float(trial.suggest_float("l2_imp", 1e-8, 1e-4, log=True))
    return params


def _suggest_sdr2(trial: optuna.Trial) -> dict:
    params = _suggest_common(trial)
    params["clip_min"] = float(trial.suggest_categorical("clip_min", [0.01, 0.05, 0.1, 0.2]))
    params["w_prop"] = float(trial.suggest_categorical("w_prop", [0.1, 0.2, 0.5, 1.0, 2.0]))
    params["w_imp"] = float(trial.suggest_categorical("w_imp", [0.1, 0.2, 0.5, 1.0, 2.0]))
    params["w_reg"] = float(trial.suggest_categorical("w_reg", [0.1, 0.2, 0.5, 1.0, 2.0]))
    params["eta"] = float(trial.suggest_float("eta", 0.0, 5.0))
    params["l2_prop"] = float(trial.suggest_float("l2_prop", 1e-8, 1e-4, log=True))
    params["l2_imp"] = float(trial.suggest_float("l2_imp", 1e-8, 1e-4, log=True))
    return params


def _suggest_counterif(trial: optuna.Trial) -> dict:
    params = {
        "lr": float(trial.suggest_float("lr", 1e-5, 2e-3, log=True)),
        "l2_reg": float(trial.suggest_float("l2_reg", 1e-8, 1e-4, log=True)),
        "lambda_point": float(trial.suggest_categorical("lambda_point", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])),
    }
    params["lambda_pair"] = float(trial.suggest_categorical("lambda_pair", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]))
    params["lambda_ipm"] = float(trial.suggest_categorical("lambda_ipm", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]))
    params["ipm_lam"] = float(trial.suggest_categorical("ipm_lam", [1.0, 10.0, 50.0]))
    params["ipm_its"] = int(trial.suggest_categorical("ipm_its", [5, 10, 20]))
    params["ipm_p"] = float(trial.suggest_categorical("ipm_p", [0.2, 0.5, 0.8]))
    params["target_percentile"] = float(trial.suggest_categorical("target_percentile", [70.0, 80.0, 85.0, 90.0, 95.0, 97.0, 99.0, 99.5]))
    params["hn_percentile"] = float(trial.suggest_categorical("hn_percentile", [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 95.0, 99.0]))
    params["batch_size_point"] = int(trial.suggest_categorical("batch_size_point", [128, 256, 512, 1024, 2048]))
    params["batch_size_pair"] = int(trial.suggest_categorical("batch_size_pair", [256, 512, 1024, 2048]))
    params["batch_size_ipm"] = int(trial.suggest_categorical("batch_size_ipm", [32, 64, 128, 256, 512]))
    params["ocsvm_batch_size"] = int(trial.suggest_categorical("ocsvm_batch_size", [2048, 4096, 8192, 16384]))
    params["pair_max_un_he"] = int(trial.suggest_categorical("pair_max_un_he", [0, 2000, 5000, 10000, 20000]))
    params["pair_max_hu_un"] = int(trial.suggest_categorical("pair_max_hu_un", [0, 2000, 5000, 10000, 20000]))
    params["pair_max_dp_he"] = int(trial.suggest_categorical("pair_max_dp_he", [0, 2000, 5000, 10000, 20000]))
    return params


def _suggest_recrec(trial: optuna.Trial, *, variant: str) -> dict:
    params = _suggest_common(trial, lr_low=1e-5, lr_high=3e-3, batch_sizes=(256, 512, 1024, 2048))
    params["calibration_sharpen_k"] = float(trial.suggest_float("calibration_sharpen_k", 1.0, 10.0))
    if variant.upper() == "F":
        params["lamp"] = float(trial.suggest_float("lamp", 1e-3, 100.0, log=True))
    return params


def _all_model_specs() -> Dict[str, ModelSpec]:
    return {
        # Debias models
        "naive": ModelSpec("naive", "models_debias/benchmark_naive.py", suggest_params=_suggest_with_w_reg),
        "ips": ModelSpec("ips", "models_debias/benchmark_ips.py", suggest_params=lambda t: _suggest_with_clip_min(t, clip_min=(0.01, 0.05, 0.1, 0.2))),
        "dr": ModelSpec("dr", "models_debias/benchmark_dr.py", suggest_params=_suggest_dr),
        "mtips": ModelSpec("mtips", "models_debias/benchmark_mtips.py", suggest_params=_suggest_mtips),
        "mtdr": ModelSpec("mtdr", "models_debias/benchmark_mtdr.py", suggest_params=_suggest_mtdr),
        "sdr2": ModelSpec("sdr2", "models_debias/benchmark_sdr2.py", suggest_params=_suggest_sdr2),

        # PU models
        "pu_naive": ModelSpec("pu_naive", "models_pu/benchmark_pu_naive.py", suggest_params=_suggest_with_w_reg),
        "upu": ModelSpec("upu", "models_pu/benchmark_upu.py", suggest_params=_suggest_upu_like),
        "nnpu": ModelSpec("nnpu", "models_pu/benchmark_nnpu.py", suggest_params=_suggest_upu_like),
        "rmf": ModelSpec("rmf", "models_pu/benchmark_rmf.py", suggest_params=_suggest_rmf_like),
        "ncrmf": ModelSpec("ncrmf", "models_pu/benchmark_ncrmf.py", suggest_params=_suggest_rmf_like),
        "bpr": ModelSpec("bpr", "models_pu/benchmark_bpr.py", suggest_params=_suggest_bpr_like),
        "ubpr": ModelSpec("ubpr", "models_pu/benchmark_ubpr.py", suggest_params=_suggest_ubpr_like),
        "cubpr": ModelSpec("cubpr", "models_pu/benchmark_cubpr.py", suggest_params=_suggest_ubpr_like),
        "uprl": ModelSpec("uprl", "models_pu/benchmark_uprl.py", suggest_params=_suggest_ubpr_like),

        # Debias + PU models
        "counterif": ModelSpec("counterif", "models_debias_pu/benchmark_counterif.py", suggest_params=_suggest_counterif),
        "recrec_i": ModelSpec("recrec_i", "models_debias_pu/benchmark_recrec.py", fixed_args=("--variant", "I"), suggest_params=lambda t: _suggest_recrec(t, variant="I")),
        "recrec_f": ModelSpec("recrec_f", "models_debias_pu/benchmark_recrec.py", fixed_args=("--variant", "F"), suggest_params=lambda t: _suggest_recrec(t, variant="F")),
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


def _make_base_cmd(args, spec: ModelSpec, *, work_dir: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_resolve_path(spec.script_path)),
        "--data_name",
        args.data_name,
        "--alpha",
        str(args.alpha),
        "--data_root",
        str(args.data_root),
        "--output_dir",
        work_dir,
        "--rerun",
        "True",
        "--use_tqdm",
        "False",
        "--seed",
        str(args.seed),
        "--binary",
        "True",
        "--hidden_dim",
        args.hidden_dim,
    ]
    if args.num_epochs is not None:
        cmd += ["--num_epochs", str(int(args.num_epochs))]
    if args.patience is not None:
        cmd += ["--patience", str(int(args.patience))]
    if args.monitor_on is not None:
        cmd += ["--monitor_on", str(args.monitor_on)]
    cmd += list(spec.fixed_args)
    return cmd


def _params_to_cli(params: dict) -> list[str]:
    cli: list[str] = []
    for k, v in params.items():
        cli.append(f"--{k}")
        cli.append(str(v))
    return cli


def _load_objectives(metrics: dict, objectives: list[str]) -> tuple[float, ...]:
    values: list[float] = []
    for obj in objectives:
        if obj not in metrics:
            raise KeyError(f"Objective '{obj}' not found in performance.yaml. Available keys: {list(metrics.keys())}")
        values.append(float(metrics[obj]))
    return tuple(values)


def _make_sampler(name: str, *, seed: int) -> optuna.samplers.BaseSampler:
    key = str(name).strip().lower()
    if key == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed)
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if key == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    raise ValueError(f"Unknown sampler: {name}. Choose from: nsga2, random, tpe.")


def _tune_one_model(args, spec: ModelSpec, *, objectives: list[str], out_dir: str) -> dict:
    if spec.suggest_params is None:
        raise ValueError(f"Model {spec.name} does not define suggest_params.")

    directions = _objective_directions(objectives)
    study_name = f"{spec.name}_{args.data_name}_alpha{args.alpha}"
    storage_path = os.path.join(out_dir, "studies", spec.name, f"{args.data_name}_alpha{args.alpha}.db")
    _ensure_dir(os.path.dirname(storage_path))
    storage = f"sqlite:///{storage_path}"

    sampler = _make_sampler(args.sampler, seed=int(args.seed))
    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(args.resume),
    )

    work_dir = os.path.join(out_dir, "work", spec.name, f"{args.data_name}_alpha{args.alpha}")
    trials_root = os.path.join(out_dir, "trials", spec.name, f"{args.data_name}_alpha{args.alpha}")
    _ensure_dir(work_dir)
    _ensure_dir(trials_root)

    results_jsonl = os.path.join(out_dir, "trials", spec.name, f"tune_results_{args.data_name}_alpha{args.alpha}.jsonl")

    env = os.environ.copy()
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    # Optuna TPESampler (multi-objective) can crash if trials are PRUNED (values=None).
    # Workaround: mark benchmark failures as FAILED trials (not PRUNED) when using TPE.
    failure_mode = "prune"
    optimize_catch: tuple[type[BaseException], ...] = ()
    if str(args.sampler).strip().lower() == "tpe" and len(directions) > 1:
        failure_mode = "fail"
        optimize_catch = (Exception,)

    def objective(trial: optuna.Trial):
        trial_dir = os.path.join(trials_root, f"trial_{trial.number:05d}")
        log_path = os.path.join(trial_dir, "run.log")
        params: dict[str, object] = {}

        try:
            params = spec.suggest_params(trial)
            cmd = _make_base_cmd(args, spec, work_dir=work_dir) + _params_to_cli(params)
            _run_benchmark(cmd, log_path=log_path, env=env)

            perf_path = os.path.join(work_dir, "performance.yaml")
            if not os.path.exists(perf_path):
                raise RuntimeError(f"Missing performance.yaml at {perf_path} (trial {trial.number}).")
            metrics = _read_yaml(perf_path)

            values = _load_objectives(metrics, objectives)

            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", params)
            trial.set_user_attr("log_path", log_path)

            rec = {
                "trial_number": int(trial.number),
                "status": "ok",
                "params": params,
                "values": list(map(float, values)),
                "objectives": objectives,
                "metrics": metrics,
                "log_path": log_path,
            }
            _ensure_dir(os.path.dirname(results_jsonl))
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            return values
        except Exception as e:
            trial.set_user_attr("params", params)
            trial.set_user_attr("log_path", log_path)
            trial.set_user_attr("error", str(e))
            rec = {
                "trial_number": int(trial.number),
                "status": "failed",
                "params": params,
                "objectives": objectives,
                "error": str(e),
                "log_path": log_path,
            }
            _ensure_dir(os.path.dirname(results_jsonl))
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if failure_mode == "prune":
                raise optuna.TrialPruned(f"trial failed: {e}")
            raise

    study.optimize(objective, n_trials=int(args.n_trials), catch=optimize_catch)

    pareto_trials = list(study.best_trials)
    if not pareto_trials:
        raise RuntimeError("Optuna returned empty Pareto front.")

    chosen = _select_representative_trial(pareto_trials, objectives=objectives)
    best_params = dict(chosen.params)
    best_metrics = dict(chosen.user_attrs.get("metrics", {}))
    best_values = list(map(float, chosen.values)) if chosen.values is not None else []

    summary = {
        "model": spec.name,
        "data_name": args.data_name,
        "alpha": float(args.alpha),
        "hidden_dim_fixed": args.hidden_dim,
        "objectives": objectives,
        "directions": directions,
        "n_trials": int(args.n_trials),
        "study_name": study_name,
        "storage": storage,
        "chosen_trial_number": int(chosen.number),
        "chosen_values": best_values,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "pareto_size": int(len(pareto_trials)),
    }

    out_best = os.path.join(out_dir, "best", spec.name, f"best_{args.data_name}_alpha{args.alpha}.json")
    _ensure_dir(os.path.dirname(out_best))
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_pareto = os.path.join(out_dir, "pareto", spec.name, f"pareto_{args.data_name}_alpha{args.alpha}.json")
    _ensure_dir(os.path.dirname(out_pareto))
    pareto_records: list[dict] = []
    for t in pareto_trials:
        pareto_records.append(
            {
                "trial_number": int(t.number),
                "values": list(map(float, t.values)) if t.values is not None else [],
                "params": dict(t.params),
                "metrics": dict(t.user_attrs.get("metrics", {})),
                "log_path": str(t.user_attrs.get("log_path", "")),
            }
        )
    with open(out_pareto, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": spec.name,
                "data_name": args.data_name,
                "alpha": float(args.alpha),
                "objectives": objectives,
                "directions": directions,
                "pareto": pareto_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, default="hs")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--data_root", type=str, default="embeddings/biased_pu")
    p.add_argument("--models", type=str, default="all", help="Comma-separated model keys, or 'all'.")
    p.add_argument("--objectives", type=str, default=",".join(OBJECTIVES_DEFAULT))
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="nsga2", help="Optuna sampler: nsga2|random|tpe.")
    p.add_argument("--hidden_dim", type=str, default="256,64", help="Fixed hidden_dim passed to all benchmarks.")
    p.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs for all models (optional).")
    p.add_argument("--patience", type=int, default=None, help="Override patience for all models (optional).")
    p.add_argument("--monitor_on", type=str, default=None, help="Override monitor_on for all models (optional).")
    p.add_argument("--force_cpu", type=str, default="True", help="Force CPU by setting CUDA_VISIBLE_DEVICES= (True/False).")
    p.add_argument("--resume", type=str, default="False", help="Resume existing Optuna study (True/False).")
    p.add_argument(
        "--require_data",
        type=str,
        default="True",
        help="Fail fast if the biased PU safetensors file is missing (True/False).",
    )
    p.add_argument("--out_dir", type=str, default="results/tune/benchmarks_optuna")
    p.add_argument("--list_models", action="store_true")
    args = p.parse_args()

    args.force_cpu = str(args.force_cpu).lower() in {"1", "true", "yes", "y"}
    args.resume = str(args.resume).lower() in {"1", "true", "yes", "y"}
    args.require_data = str(args.require_data).lower() in {"1", "true", "yes", "y"}
    args.sampler = str(args.sampler).strip().lower()

    args.data_root = _resolve_path(args.data_root)
    args.out_dir = _resolve_path(args.out_dir)

    specs = _all_model_specs()
    if args.list_models:
        for k in sorted(specs.keys()):
            print(k)
        return

    if str(args.models).strip().lower() == "all":
        model_keys = sorted(specs.keys())
    else:
        model_keys = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]

    unknown = [m for m in model_keys if m not in specs]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Use --list_models to see available.")

    objectives = [o.strip() for o in str(args.objectives).split(",") if o.strip()]
    if not objectives:
        raise SystemExit("--objectives must be a non-empty comma-separated list.")

    # Quick dataset file check (all benchmarks use the same biased_pu naming convention)
    expected = args.data_root / f"FsfairX-LLaMA3-RM-v0.1_{args.data_name}_{args.alpha}_pu.safetensors"
    if not os.path.exists(expected):
        msg = (
            f"Missing biased PU file: {expected}\n"
            "Run `python simulate_bias_pu.py --data_name ... --alpha ... --output_dir ...` first, "
            "or pass the correct `--data_root`."
        )
        if args.require_data:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")

    _ensure_dir(str(args.out_dir))

    all_summaries: list[dict] = []
    for m in model_keys:
        spec = specs[m]
        print("=" * 70)
        print(f"Tuning: {spec.name}  data={args.data_name}  alpha={args.alpha}")
        print(f"Sampler: {args.sampler}")
        print(f"Objectives: {objectives}")
        print(f"hidden_dim fixed: {args.hidden_dim}")
        print(f"Trials: {args.n_trials}  resume={args.resume}")
        print("=" * 70)
        try:
            summary = _tune_one_model(args, spec, objectives=objectives, out_dir=str(args.out_dir))
            all_summaries.append(summary)
            vals = summary.get("chosen_values", [])
            print(f"[DONE] {spec.name}: chosen_values={vals}")
        except Exception as e:
            print(f"[FAIL] {spec.name}: {e}")
            all_summaries.append(
                {
                    "model": spec.name,
                    "data_name": args.data_name,
                    "alpha": float(args.alpha),
                    "hidden_dim_fixed": args.hidden_dim,
                    "objectives": objectives,
                    "error": str(e),
                }
            )

    out_summary = str(args.out_dir / f"summary_{args.data_name}_alpha{args.alpha}.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(out_summary)


if __name__ == "__main__":
    main()
