import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import optuna
import yaml


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _objective_directions(objectives: list[str]) -> list[str]:
    directions: list[str] = []
    for obj in objectives:
        key = str(obj).lower()
        if any(k in key for k in ["mae", "rmse", "loss", "nll"]):
            directions.append("minimize")
        elif "r2" in key or "auroc" in key or "ndcg" in key or "pearson" in key or "f1" in key or "recall" in key:
            directions.append("maximize")
        else:
            # Default: maximize
            directions.append("maximize")
    return directions


def _trial_output_dir(base_dir: str, data_name: str, variant: str) -> str:
    return os.path.join(base_dir, f"recrec_{variant.lower()}", data_name)


def _run_recrec_trial(args, params: dict, *, output_dir: str, subsample_train: int | None, subsample_val: int | None) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "models_debias_pu/benchmark_recrec.py",
        "--data_name",
        args.data_name,
        "--alpha",
        str(args.alpha),
        "--variant",
        args.variant,
        "--rerun",
        "True",
        "--use_tqdm",
        "False",
        "--output_dir",
        output_dir,
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        str(args.patience),
        "--eval_every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--hidden_dim",
        params["hidden_dim"],
        "--batch_size",
        str(params["batch_size"]),
        "--lr",
        str(params["lr"]),
        "--l2_reg",
        str(params["l2_reg"]),
        "--pred_target",
        "gamma",
        "--calibration",
        "isotonic",
        "--calibration_fit_on",
        "val_true",
        "--calibration_sharpen_k",
        str(params["calibration_sharpen_k"]),
    ]
    if args.variant.upper() == "F":
        cmd += ["--lamp", str(params["lamp"])]

    if subsample_train is not None:
        cmd += ["--subsample_train", str(int(subsample_train))]
    if subsample_val is not None:
        cmd += ["--subsample_val", str(int(subsample_val))]

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    log_path = os.path.join(output_dir, "tune_last.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    perf_path = os.path.join(output_dir, "performance.yaml")
    if proc.returncode != 0 or (not os.path.exists(perf_path)):
        raise RuntimeError(f"Trial failed (code={proc.returncode}). See {log_path}")

    metrics = _read_yaml(perf_path)
    metrics["_trial_time_sec"] = float(dt)
    metrics["_trial_log_path"] = log_path
    return metrics


def _suggest_params(trial: optuna.Trial, *, variant: str) -> dict:
    hidden_dim = trial.suggest_categorical("hidden_dim", ["256,64", "128,64", "256,128", "512,128"])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-8, 1e-4, log=True)
    calibration_sharpen_k = trial.suggest_float("calibration_sharpen_k", 1.0, 8.0)
    params = {
        "hidden_dim": hidden_dim,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "l2_reg": float(l2_reg),
        "calibration_sharpen_k": float(calibration_sharpen_k),
    }
    if variant.upper() == "F":
        params["lamp"] = float(trial.suggest_float("lamp", 1e-3, 100.0, log=True))
    return params


def _select_representative_trial(pareto_trials: list[optuna.trial.FrozenTrial], objectives: list[str]) -> optuna.trial.FrozenTrial:
    """
    Pick ONE trial from Pareto front to write back into benchmark defaults.

    Default rule for the requested objectives:
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


def tune_one(args) -> tuple[dict, dict, list[dict]]:
    """
    Returns:
      (best_params, best_metrics, pareto_front_records)
    """
    objectives = [o.strip() for o in str(args.objectives).split(",") if o.strip()]
    if not objectives:
        raise ValueError("--objectives must be a non-empty comma-separated list.")
    directions = _objective_directions(objectives)
    study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(seed=args.seed))

    work_dir = _trial_output_dir(args.work_dir, args.data_name, args.variant)

    # Heuristic subsampling: keep HS full, subsample larger datasets by default.
    subsample_train = args.subsample_train
    subsample_val = args.subsample_val

    def objective(trial: optuna.Trial):
        params = _suggest_params(trial, variant=args.variant)
        metrics = _run_recrec_trial(
            args,
            params,
            output_dir=work_dir,
            subsample_train=subsample_train,
            subsample_val=subsample_val,
        )
        values = []
        for obj in objectives:
            v = float(metrics.get(obj, float("nan")))
            values.append(v)
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise RuntimeError(f"Objectives contain non-finite values: {dict(zip(objectives, values))}")
        trial.set_user_attr("metrics", metrics)
        return tuple(values)

    import numpy as np  # local import for optuna serialization friendliness

    study.optimize(objective, n_trials=int(args.n_trials))

    pareto_trials = list(study.best_trials)
    if len(pareto_trials) == 0:
        raise RuntimeError("Optuna returned empty Pareto front.")

    chosen = _select_representative_trial(pareto_trials, objectives=objectives)

    best_params = dict(chosen.params)
    best_metrics = dict(chosen.user_attrs.get("metrics", {}))
    best_metrics["_objectives"] = objectives
    best_metrics["_directions"] = directions
    best_metrics["_chosen_trial_number"] = int(chosen.number)
    best_metrics["_chosen_values"] = list(map(float, chosen.values))

    # Ensure all required keys exist in best_params
    if args.variant.upper() == "F" and "lamp" not in best_params:
        best_params["lamp"] = float(args.lamp_default)

    pareto_records: list[dict] = []
    for t in pareto_trials:
        pareto_records.append(
            {
                "trial_number": int(t.number),
                "values": list(map(float, t.values)),
                "params": dict(t.params),
                "metrics": dict(t.user_attrs.get("metrics", {})),
            }
        )

    return best_params, best_metrics, pareto_records


def _embedding_file_exists(data_name: str, alpha: float) -> bool:
    path = f"embeddings/biased_pu/FsfairX-LLaMA3-RM-v0.1_{data_name}_{alpha}_pu.safetensors"
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", type=str, default="hs,saferlhf,ufb", help="Comma-separated datasets.")
    parser.add_argument("--variants", type=str, default="F,I", help="Comma-separated variants: F,I.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--objectives",
        type=str,
        default="MAE on test,RMSE on test,R2 on test",
        help="Comma-separated objective names (read from performance.yaml).",
    )
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--final_num_epochs", type=int, default=200)
    parser.add_argument("--final_patience", type=int, default=20)
    parser.add_argument("--final_eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_train", type=int, default=30000)
    parser.add_argument("--subsample_val", type=int, default=10000)
    parser.add_argument("--work_dir", type=str, default="results/tune/_tmp")
    parser.add_argument("--out_dir", type=str, default="results/tune/recrec_pareto")
    parser.add_argument("--final_full_train", type=str, default="True", help="Run final full training with best params.")
    parser.add_argument("--lamp_default", type=float, default=1.0)
    args = parser.parse_args()

    final_full_train = str(args.final_full_train).lower() in {"true", "1", "yes", "y"}

    data_names = [d.strip() for d in str(args.data_names).split(",") if d.strip()]
    variants = [v.strip().upper() for v in str(args.variants).split(",") if v.strip()]

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_best = []

    for data_name in data_names:
        if not _embedding_file_exists(data_name, args.alpha):
            print(f"[SKIP] Missing embeddings for {data_name} alpha={args.alpha}: run `simulate_bias_pu.sh` first.")
            continue

        for variant in variants:
            if variant not in {"F", "I"}:
                raise ValueError(f"Unknown variant: {variant}")

            subsample_train = int(args.subsample_train) if int(args.subsample_train) > 0 else None
            subsample_val = int(args.subsample_val) if int(args.subsample_val) > 0 else None

            run_args = argparse.Namespace(**vars(args))
            run_args.data_name = data_name
            run_args.variant = variant
            run_args.subsample_train = subsample_train
            run_args.subsample_val = subsample_val

            print("=" * 80)
            print(
                f"Tuning ReCRec-{variant} on {data_name} (alpha={args.alpha}) | objectives={args.objectives} | trials={args.n_trials}"
            )
            print("=" * 80)

            best_params, best_metrics, pareto_front = tune_one(run_args)

            record = {
                "data_name": data_name,
                "alpha": float(args.alpha),
                "variant": variant,
                "objectives": [o.strip() for o in str(args.objectives).split(",") if o.strip()],
                "best_params": best_params,
                "best_metrics": best_metrics,
            }

            out_path = os.path.join(args.out_dir, f"best_{data_name}_alpha{args.alpha}_{variant.lower()}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            print(f"[Saved] {out_path}")

            pareto_path = os.path.join(args.out_dir, f"pareto_{data_name}_alpha{args.alpha}_{variant.lower()}.json")
            with open(pareto_path, "w", encoding="utf-8") as f:
                json.dump(pareto_front, f, indent=2, ensure_ascii=False)
            print(f"[Saved] {pareto_path}")

            all_best.append(record)

            if final_full_train:
                # Run full training once with best params into default cache folder.
                out_subdir = f"recrec_{variant.lower()}"
                final_out = f"results/cache/{out_subdir}/{data_name}"
                if os.path.exists(final_out):
                    shutil.rmtree(final_out)
                os.makedirs(final_out, exist_ok=True)

                cmd = [
                    sys.executable,
                    "models_debias_pu/benchmark_recrec.py",
                    "--data_name",
                    data_name,
                    "--alpha",
                    str(args.alpha),
                    "--variant",
                    variant,
                    "--rerun",
                    "True",
                    "--use_tqdm",
                    "False",
                    "--output_dir",
                    final_out,
                    "--num_epochs",
                    str(args.final_num_epochs),
                    "--patience",
                    str(args.final_patience),
                    "--eval_every",
                    str(args.final_eval_every),
                    "--hidden_dim",
                    best_params["hidden_dim"],
                    "--batch_size",
                    str(best_params["batch_size"]),
                    "--lr",
                    str(best_params["lr"]),
                    "--l2_reg",
                    str(best_params["l2_reg"]),
                    "--pred_target",
                    "gamma",
                    "--calibration",
                    "isotonic",
                    "--calibration_fit_on",
                    "val_true",
                    "--calibration_sharpen_k",
                    str(best_params["calibration_sharpen_k"]),
                ]
                if variant == "F":
                    cmd += ["--lamp", str(best_params.get("lamp", args.lamp_default))]

                print(f"[Final] Training best ReCRec-{variant} on {data_name} -> {final_out}")
                subprocess.run(cmd, check=True)

    # Write a combined summary file
    summary_path = os.path.join(args.out_dir, f"summary_alpha{args.alpha}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
