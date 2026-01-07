"""
Simulate PU Learning WITHOUT Selection Bias for Causal Reward Model Research

This script processes reward model embeddings to simulate:
1. Random sampling (no selection bias - all samples have equal probability of observation)
2. PU (Positive-Unlabeled) Learning setup: unobserved samples are treated as unlabeled (y=0)

=== INPUT ===
Files:
    - {data_root}/{model_name}_{data_name}_train.safetensors
    - {data_root}/{model_name}_{data_name}_test.safetensors

Input format (safetensors):
    {
        "embeddings": Tensor[N, D],  # N samples, D-dimensional embeddings
        "labels": Tensor[N],         # Continuous reward labels
        "user_id": Tensor[N],        # (Optional) int64 group id, for group-wise sampling/pairwise
    }

=== OUTPUT ===
File:
    - {output_dir}/{model_name}_{data_name}_pu_unbiased.safetensors

Output format (safetensors):
    {
        # Embeddings (80/20 train/val split from train file, test from test file)
        "X_train": Tensor[N_train, D],
        "X_val": Tensor[N_val, D],
        "X_test": Tensor[N_test, D],

        # Original continuous labels
        "y_train": Tensor[N_train],
        "y_val": Tensor[N_val],
        "y_test": Tensor[N_test],

        # Binary labels (PU setting for train/val)
        "y_train_binary": Tensor[N_train],  # observed: true label (0/1), unobserved: 0 (unlabeled)
        "y_val_binary": Tensor[N_val],      # observed: true label (0/1), unobserved: 0 (unlabeled)
        "y_val_binary_true": Tensor[N_val], # clean binary label (0/1), before PU masking (oracle, for monitoring)
        "y_test_binary": Tensor[N_test],    # 0 or 1, clean (no bias)

        # Observation masks (True = observed/sampled)
        "mask_train": Tensor[N_train],  # Boolean
        "mask_val": Tensor[N_val],      # Boolean

        # (Optional) group ids aligned with X_*/y_* splits
        "user_id_train": Tensor[N_train],  # int64
        "user_id_val": Tensor[N_val],      # int64
        "user_id_test": Tensor[N_test],    # int64
    }

=== PARAMETERS ===
    --target_observation_rate: Controls the fraction of samples that are observed (default: 0.2)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import yaml
from safetensors.torch import load_file, save_file
from sklearn.model_selection import train_test_split


data2levels = {
    'hs': [0., 1., 2., 3., 4.],
    'ufb': [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.3, 8.5, 9.0, 9.3, 9.5, 9.8, 10.0],
    'saferlhf': [0., 1., 2., 3.],
}


def binarize_labels(labels, data_name):
    """
    Binarize labels based on data2levels.

    Args:
        labels: Original continuous labels
        data_name: Name of the dataset

    Returns:
        binary_labels: Binarized labels (0 or 1)
    """
    levels = data2levels[data_name]
    median_level = np.median(levels)

    # Convert to binary: 0 for below or equal to median, 1 for above median
    binary_labels = (labels > median_level).astype(float)

    print(f"Binarization threshold (median): {median_level}")
    print(f"Binary distribution: {np.sum(binary_labels == 0)} negative, {np.sum(binary_labels == 1)} positive")

    return binary_labels


def apply_random_sampling(X, y, observation_rate, seed=0):
    """
    Apply uniform random sampling to create a mask (no selection bias).

    Args:
        X: Features
        y: Labels
        observation_rate: Probability of observing each sample (uniform for all)
        seed: Random seed

    Returns:
        X_sampled: Sampled features
        y_sampled: Sampled labels
        mask: Boolean mask indicating which samples were selected
    """
    np.random.seed(seed)

    # Generate random numbers and compare with observation rate (uniform for all samples)
    random_values = np.random.random(len(y))
    mask = random_values <= observation_rate

    # Apply mask
    X_sampled = X[mask]
    y_sampled = y[mask]

    return X_sampled, y_sampled, mask


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _split_stats(y_binary_true, mask):
    y_binary_true = np.asarray(y_binary_true)
    mask = np.asarray(mask).astype(bool)

    n_total = int(y_binary_true.shape[0])
    n_pos = int(np.sum(y_binary_true == 1))
    n_neg = int(np.sum(y_binary_true == 0))

    n_observed = int(np.sum(mask))
    n_unobserved = int(n_total - n_observed)

    masked_pos = int(np.sum((~mask) & (y_binary_true == 1)))
    observed_pos = int(np.sum(mask & (y_binary_true == 1)))
    observed_neg = int(np.sum(mask & (y_binary_true == 0)))

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": _safe_float(n_pos / n_total) if n_total > 0 else float("nan"),
        "n_observed": n_observed,
        "n_unobserved": n_unobserved,
        "observed_rate": _safe_float(n_observed / n_total) if n_total > 0 else float("nan"),
        "masked_pos": masked_pos,
        "masked_pos_ratio_over_pos": _safe_float(masked_pos / n_pos) if n_pos > 0 else float("nan"),
        "observed_pos": observed_pos,
        "observed_neg": observed_neg,
    }


def _unique_count(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    return int(np.unique(arr).shape[0])


parser = ArgumentParser(description="PU Learning Simulation (Unbiased/Random Sampling)")
parser.add_argument("--model_name", type=str, default="FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--data_name", type=str, default="saferlhf")
parser.add_argument("--data_root", type=str, default="./embeddings/normal")
parser.add_argument("--output_dir", type=str, default="./embeddings/biased_pu")
parser.add_argument("--target_observation_rate", type=float, default=0.2, help="Fraction of samples to observe (uniform probability)")
args = parser.parse_args()

data = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_train.safetensors")
embeddings = data["embeddings"].float().numpy()
labels = data["labels"].float().numpy()
user_id = data["user_id"].numpy() if "user_id" in data else None
print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")
if user_id is not None:
    print(f"Total user_id loaded: {user_id.shape[0]}")

# Mask data where target_label is nan
embeddings_filtered = embeddings[~np.isnan(labels)]
target_labels_filtered = labels[~np.isnan(labels)]
user_id_filtered = user_id[~np.isnan(labels)] if user_id is not None else None
print(f"Original data size: {embeddings.shape[0]}")
print(f"Data size after filtering NaN values: {embeddings_filtered.shape[0]}")


# Data split
if user_id_filtered is not None:
    X_train, X_val, y_train, y_val, user_id_train, user_id_val = train_test_split(
        embeddings_filtered,
        target_labels_filtered,
        user_id_filtered,
        test_size=0.2,
        random_state=42,
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings_filtered,
        target_labels_filtered,
        test_size=0.2,
        random_state=42,
    )
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")

# Apply random sampling (uniform probability - no selection bias)
X_train_sampled, y_train_sampled, mask_train = apply_random_sampling(X_train, y_train, args.target_observation_rate, seed=0)
X_val_sampled, y_val_sampled, mask_val = apply_random_sampling(X_val, y_val, args.target_observation_rate, seed=0)
print(f"Training set after sampling: {X_train_sampled.shape[0]} samples ({X_train_sampled.shape[0]/X_train.shape[0]*100:.1f}% of original)")
print(f"Valid set after sampling: {X_val_sampled.shape[0]} samples ({X_val_sampled.shape[0]/X_val.shape[0]*100:.1f}% of original)")

# Binarize - compute true binary labels
y_train_binary_true = binarize_labels(y_train, args.data_name)
y_val_binary_true = binarize_labels(y_val, args.data_name)

# PU Learning setup:
# - Observed samples (mask=True): use true binary labels
# - Unobserved samples (mask=False): set to 0 (unlabeled)
y_train_binary = y_train_binary_true.copy()
y_train_binary[~mask_train] = 0  # Unobserved -> unlabeled (0)

y_val_binary = y_val_binary_true.copy()
y_val_binary[~mask_val] = 0  # Unobserved -> unlabeled (0)

# PU Learning Statistics
print("\n=== PU Learning Statistics (Train) ===")
total_pos_train = np.sum(y_train_binary_true == 1)
masked_pos_train = np.sum((~mask_train) & (y_train_binary_true == 1))
observed_pos_train = np.sum(mask_train & (y_train_binary_true == 1))
observed_neg_train = np.sum(mask_train & (y_train_binary_true == 0))
print(f"Total positive samples: {total_pos_train}")
print(f"Masked (unlabeled) positive samples: {masked_pos_train}")
print(f"Masked positive ratio: {masked_pos_train/total_pos_train*100:.1f}%")
print(f"Observed positive samples: {observed_pos_train}")
print(f"Observed negative samples: {observed_neg_train}")

print("\n=== PU Learning Statistics (Val) ===")
total_pos_val = np.sum(y_val_binary_true == 1)
masked_pos_val = np.sum((~mask_val) & (y_val_binary_true == 1))
observed_pos_val = np.sum(mask_val & (y_val_binary_true == 1))
observed_neg_val = np.sum(mask_val & (y_val_binary_true == 0))
print(f"Total positive samples: {total_pos_val}")
print(f"Masked (unlabeled) positive samples: {masked_pos_val}")
print(f"Masked positive ratio: {masked_pos_val/total_pos_val*100:.1f}%")
print(f"Observed positive samples: {observed_pos_val}")
print(f"Observed negative samples: {observed_neg_val}")

data_test = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_test.safetensors")
y_test = data_test["labels"].float().numpy()
y_test_binary = binarize_labels(y_test, args.data_name)
user_id_test = data_test["user_id"] if "user_id" in data_test else None

os.makedirs(args.output_dir, exist_ok=True)
output = {
    # subsets that are used for unbiased evaluation
    "X_train": torch.from_numpy(X_train),
    "X_val": torch.from_numpy(X_val),
    "X_test": data_test["embeddings"],
    "y_train": torch.from_numpy(y_train),
    "y_val": torch.from_numpy(y_val),
    "y_test": torch.from_numpy(y_test),
    "y_train_binary": torch.from_numpy(y_train_binary),
    "y_val_binary": torch.from_numpy(y_val_binary),
    "y_val_binary_true": torch.from_numpy(y_val_binary_true),
    "y_test_binary": torch.from_numpy(y_test_binary),
    "mask_train": torch.from_numpy(mask_train),
    "mask_val": torch.from_numpy(mask_val)
}
if user_id_filtered is not None:
    output["user_id_train"] = torch.from_numpy(user_id_train)
    output["user_id_val"] = torch.from_numpy(user_id_val)
if user_id_test is not None:
    output["user_id_test"] = user_id_test

output_path = f"{args.output_dir}/{args.model_name}_{args.data_name}_pu_unbiased.safetensors"
save_file(output, output_path)

# Save dataset statistics for later analysis (grouping/pairwise sampling, PU rates, etc.)
stats = {
    "model_name": args.model_name,
    "data_name": args.data_name,
    "sampling_method": "random_uniform",
    "output_path": output_path,
    "stage1_train_path": f"{args.data_root}/{args.model_name}_{args.data_name}_train.safetensors",
    "stage1_test_path": f"{args.data_root}/{args.model_name}_{args.data_name}_test.safetensors",
    "user_id_available": user_id_filtered is not None,
    "user_id_test_available": user_id_test is not None,
    "target_observation_rate": args.target_observation_rate,
    "n_stage1_train_total": int(embeddings.shape[0]),
    "n_stage1_train_nan_labels": int(np.isnan(labels).sum()),
    "n_stage1_train_after_nan_filter": int(embeddings_filtered.shape[0]),
    "n_split_train": int(X_train.shape[0]),
    "n_split_val": int(X_val.shape[0]),
    "n_test": int(data_test["embeddings"].shape[0]),
    "binary_threshold": float(np.median(data2levels[args.data_name])),
    "train": _split_stats(y_train_binary_true, mask_train),
    "val": _split_stats(y_val_binary_true, mask_val),
    "test": {
        "n_total": int(y_test_binary.shape[0]),
        "n_pos": int(np.sum(y_test_binary == 1)),
        "n_neg": int(np.sum(y_test_binary == 0)),
        "pos_rate": _safe_float(np.mean(y_test_binary == 1)) if y_test_binary.size > 0 else float("nan"),
    },
}

if user_id_filtered is not None:
    stats["user_id"] = {
        "n_unique_train": _unique_count(user_id_train),
        "n_unique_val": _unique_count(user_id_val),
        "n_unique_test": _unique_count(user_id_test.cpu().numpy() if user_id_test is not None else None),
        "n_unique_train_observed": _unique_count(user_id_train[mask_train]),
        "n_unique_val_observed": _unique_count(user_id_val[mask_val]),
        "n_unique_train_pos": _unique_count(user_id_train[y_train_binary_true == 1]),
        "n_unique_val_pos": _unique_count(user_id_val[y_val_binary_true == 1]),
        "n_unique_train_masked_pos": _unique_count(user_id_train[(~mask_train) & (y_train_binary_true == 1)]),
        "n_unique_val_masked_pos": _unique_count(user_id_val[(~mask_val) & (y_val_binary_true == 1)]),
    }

stats_path = f"{args.output_dir}/{args.model_name}_{args.data_name}_pu_unbiased_stats.yaml"
tmp_stats_path = stats_path + ".tmp"
with open(tmp_stats_path, "w") as f:
    yaml.safe_dump(stats, f, sort_keys=False)
os.replace(tmp_stats_path, stats_path)
print(f"\nSaved stats to: {stats_path}")
