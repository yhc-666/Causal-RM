import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _format_alpha(alpha: float) -> str:
    s = str(alpha)
    if "e" in s or "E" in s:
        s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s


def _bar_or_hist(ax, values_all, values_obs, title: str, xlabel: str):
    values_all = np.asarray(values_all)
    values_obs = np.asarray(values_obs)
    values_all = values_all[np.isfinite(values_all)]
    values_obs = values_obs[np.isfinite(values_obs)]

    if values_all.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    rounded = np.round(values_all.astype(np.float64), 3)
    uniq = np.unique(rounded)

    if uniq.size <= 60:
        counts_all = np.array([(rounded == v).sum() for v in uniq], dtype=np.int64)
        counts_obs = np.array([(np.round(values_obs.astype(np.float64), 3) == v).sum() for v in uniq], dtype=np.int64)
        x = np.arange(uniq.size)
        width = 0.42
        ax.bar(x - width / 2, counts_all, width=width, label="All train", color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, counts_obs, width=width, label="Observed (mask=1)", color="#DD8452", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in uniq], rotation=45, ha="right")
    else:
        bins = 50
        ax.hist(values_all, bins=bins, alpha=0.65, color="#4C72B0", label="All train")
        ax.hist(values_obs, bins=bins, alpha=0.65, color="#DD8452", label="Observed (mask=1)")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./embeddings/biased_pu")
    parser.add_argument("--model_name", type=str, default="FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--data_name", type=str, default="hs")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for output filename")
    parser.add_argument("--out_dir", type=str, default="./results/plots")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    alpha_str = _format_alpha(args.alpha)
    path = f"{args.data_root}/{args.model_name}_{args.data_name}_{alpha_str}_pu.safetensors"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Biased PU file not found: {path}")

    data = load_file(path)
    y_train = _to_numpy(data["y_train"]).astype(np.float64)
    propensity_train = _to_numpy(data["propensity_train"]).astype(np.float64)
    mask_train = _to_numpy(data["mask_train"]).astype(bool)

    y_val = _to_numpy(data["y_val"]).astype(np.float64)
    propensity_val = _to_numpy(data["propensity_val"]).astype(np.float64)
    mask_val = _to_numpy(data["mask_val"]).astype(bool)

    obs_rate_train = float(mask_train.mean()) if mask_train.size else float("nan")
    obs_rate_val = float(mask_val.mean()) if mask_val.size else float("nan")
    prop_mean_train = float(np.mean(propensity_train)) if propensity_train.size else float("nan")
    prop_mean_val = float(np.mean(propensity_val)) if propensity_val.size else float("nan")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Rating distribution shift due to sampling
    _bar_or_hist(
        axes[0],
        values_all=y_train,
        values_obs=y_train[mask_train],
        title="Rating distribution (train)",
        xlabel="rating / y_train",
    )

    # (2) Propensity distribution
    bins = np.linspace(0.0, 1.0, 51)
    axes[1].hist(propensity_train, bins=bins, alpha=0.7, label="train", color="#4C72B0")
    axes[1].hist(propensity_val, bins=bins, alpha=0.55, label="val", color="#55A868")
    axes[1].axvline(prop_mean_train, color="#4C72B0", linestyle="--", linewidth=2, label=f"train mean={prop_mean_train:.3f}")
    axes[1].axvline(prop_mean_val, color="#55A868", linestyle="--", linewidth=2, label=f"val mean={prop_mean_val:.3f}")
    axes[1].set_title("Propensity distribution")
    axes[1].set_xlabel("propensity")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    # (3) Propensity vs rating
    y_round = np.round(y_train, 3)
    uniq = np.unique(y_round[np.isfinite(y_round)])
    if uniq.size <= 30:
        import pandas as pd

        df = pd.DataFrame({"rating": y_round, "propensity": propensity_train})
        df = df[np.isfinite(df["rating"]) & np.isfinite(df["propensity"])].copy()
        df["rating"] = df["rating"].astype(str)
        order = sorted(df["rating"].unique(), key=lambda s: float(s))
        sns.boxplot(data=df, x="rating", y="propensity", ax=axes[2], color="#DD8452", order=order)
        axes[2].set_title("Propensity vs rating (train)")
        axes[2].set_xlabel("rating (binned)")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].set_ylabel("propensity")
    else:
        axes[2].hexbin(y_train, propensity_train, gridsize=40, cmap="viridis", mincnt=1)
        axes[2].set_title("Propensity vs rating (train)")
        axes[2].set_xlabel("rating / y_train")
        axes[2].set_ylabel("propensity")

    fig.suptitle(
        f"{args.data_name}  alpha={alpha_str}  "
        f"train n={y_train.size} (obs={obs_rate_train:.3f})  "
        f"val n={y_val.size} (obs={obs_rate_val:.3f})",
        y=1.05,
        fontsize=16,
    )
    fig.tight_layout()

    _ensure_dir(args.out_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"propensity_rating_{args.data_name}_alpha{alpha_str}{suffix}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
