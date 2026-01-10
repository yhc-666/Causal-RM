"""
Simulate Selection Bias with PU Learning for Causal Reward Model Research

This script processes reward model embeddings to simulate:
1. Selection bias via propensity-based sampling (lower reward -> lower observation probability)
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
    - {output_dir}/{model_name}_{data_name}_{alpha}_pu.safetensors

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

        # Propensity scores for bias correction (e.g., IPS)
        "propensity_train": Tensor[N_train],  # P(observed | features)
        "propensity_val": Tensor[N_val],

        # Observation masks (True = observed/sampled)
        "mask_train": Tensor[N_train],  # Boolean
        "mask_val": Tensor[N_val],      # Boolean

        # (Optional) group ids aligned with X_*/y_* splits
        "user_id_train": Tensor[N_train],  # int64
        "user_id_val": Tensor[N_val],      # int64
        "user_id_test": Tensor[N_test],    # int64
    }

=== PARAMETERS ===
    --alpha: Controls selection bias strength (lower = stronger bias toward high rewards)
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


# NOTE: Label noise functionality removed - now using PU Learning instead
# def add_noise_to_labels(labels, noise_rates, seed=0):
#     """
#     Add noise to binary labels by flipping them.
#
#     Args:
#         labels: Binary labels (0 or 1)
#         noise_rates: [rate_pos_to_neg, rate_neg_to_pos] - probabilities of flipping
#         seed: Random seed
#
#     Returns:
#         noisy_labels: Labels with noise added
#     """
#     np.random.seed(seed)
#
#     labels_noisy = labels.copy()
#
#     # Find positive and negative samples
#     pos_mask = labels == 1
#     neg_mask = labels == 0
#
#     pos_count = np.sum(pos_mask)
#     neg_count = np.sum(neg_mask)
#
#     print(f"Before noise: {neg_count} negative, {pos_count} positive")
#
#     # Generate random flips
#     if pos_count > 0:
#         pos_flip = np.random.binomial(1, noise_rates[0], pos_count)
#         labels_noisy[pos_mask] = labels_noisy[pos_mask] * (1 - pos_flip)  # positive to negative
#         pos_flipped = np.sum(pos_flip)
#         print(f"Flipped {pos_flipped} positive samples to negative (rate: {noise_rates[0]})")
#
#     if neg_count > 0:
#         neg_flip = np.random.binomial(1, noise_rates[1], neg_count)
#         labels_noisy[neg_mask] = labels_noisy[neg_mask] + neg_flip * (1 - labels_noisy[neg_mask])  # negative to positive
#         neg_flipped = np.sum(neg_flip)
#         print(f"Flipped {neg_flipped} negative samples to positive (rate: {noise_rates[1]})")
#
#     final_pos_count = np.sum(labels_noisy == 1)
#     final_neg_count = np.sum(labels_noisy == 0)
#     print(f"After noise: {final_neg_count} negative, {final_pos_count} positive")
#
#     return labels_noisy


def calculate_propensity(labels, alpha, target_observation_rate=0.2):
    """
    Calculate propensity scores for each rating.

    Args:
        labels: Array of ratings
        alpha: Alpha parameter for propensity calculation
    
    Returns:
        propensity: Array of propensity scores
    """
    propensity = np.ones_like(labels)
    labels = 1 + (labels - labels.min()) * 4 / (labels.max() - labels.min())

    mask_lt_4 = labels < labels.max()
    # reward越低，propensity score越小
    propensity[mask_lt_4] = alpha ** (labels.max() - labels[mask_lt_4])

    mask_ge_4 = labels >= labels.max()
    propensity[mask_ge_4] = 1.0
    # propensity = alpha ** (0.9 - labels)

    # Calculate k such that sum(propensity) = expected_observations
    expected_observations = target_observation_rate * len(labels)
    k = expected_observations / np.sum(propensity)
    propensity = propensity * k

    return propensity


def apply_propensity_sampling(X, y, propensity, seed=0):
    """
    Apply propensity-based sampling to create a mask.

    Args:
        X: Features
        y: Labels
        propensity: Propensity scores
        seed: Random seed

    Returns:
        X_sampled: Sampled features
        y_sampled: Sampled labels
        mask: Boolean mask indicating which samples were selected
    """
    np.random.seed(seed)

    # Generate random numbers and compare with propensity
    random_values = np.random.random(len(propensity))
    # 模拟propensity score越高越容易选中
    mask = random_values <= propensity

    # Apply mask
    X_sampled = X[mask]
    y_sampled = y[mask]
    propensity_sampled = propensity[mask]

    return X_sampled, y_sampled, propensity_sampled, mask


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _split_stats(y_binary_true, mask, propensity):
    y_binary_true = np.asarray(y_binary_true)
    mask = np.asarray(mask).astype(bool)
    propensity = np.asarray(propensity)

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
        "propensity_min": _safe_float(np.min(propensity)) if propensity.size > 0 else float("nan"),
        "propensity_max": _safe_float(np.max(propensity)) if propensity.size > 0 else float("nan"),
        "propensity_mean": _safe_float(np.mean(propensity)) if propensity.size > 0 else float("nan"),
        "propensity_sum_expected_obs": _safe_float(np.sum(propensity)) if propensity.size > 0 else float("nan"),
    }


def _unique_count(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    return int(np.unique(arr).shape[0])


parser = ArgumentParser(description="Linear Probing on Precomputed Embeddings")
parser.add_argument("--model_name", type=str, default="FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--data_name", type=str, default="saferlhf")
parser.add_argument("--data_root", type=str, default="../embeddings/normal")
parser.add_argument("--output_dir", type=str, default="../embeddings/biased_pu")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for propensity calculation")
parser.add_argument("--target_obs_rate", type=float, default=0.2, help="Target observation rate")
# NOTE: Label noise parameters removed - now using PU Learning instead
# parser.add_argument("--r10", type=float, default=0.1, help="Noise rate for positive to negative")
# parser.add_argument("--r01", type=float, default=0.1, help="Noise rate for negative to positive")
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
propensity_filtered = calculate_propensity(target_labels_filtered, args.alpha, args.target_obs_rate)
print(f"Original data size: {embeddings.shape[0]}")
print(f"Data size after filtering NaN values: {embeddings_filtered.shape[0]}")
print(f"Propensity range: [{propensity_filtered.min():.6f}, {propensity_filtered.max():.6f}]")


# Data split
if user_id_filtered is not None:
    X_train, X_val, y_train, y_val, propensity_train, propensity_val, user_id_train, user_id_val = train_test_split(
        embeddings_filtered,
        target_labels_filtered,
        propensity_filtered,
        user_id_filtered,
        test_size=0.2,
        random_state=42,
    )
else:
    X_train, X_val, y_train, y_val, propensity_train, propensity_val = train_test_split(
        embeddings_filtered,
        target_labels_filtered,
        propensity_filtered,
        test_size=0.2,
        random_state=42,
    )
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")

# Apply propensity sampling to training set
X_train_sampled, y_train_sampled, propensity_train_sampled, mask_train = apply_propensity_sampling(X_train, y_train, propensity_train, seed=0)
X_val_sampled, y_val_sampled, propensity_val_sampled, mask_val = apply_propensity_sampling(X_val, y_val, propensity_val, seed=0)
print(f"Training set after sampling: {X_train_sampled.shape[0]} samples ({X_train_sampled.shape[0]/X_train.shape[0]*100:.1f}% of original)")
print(f"Valid set after sampling: {X_val_sampled.shape[0]} samples ({X_val_sampled.shape[0]/X_val.shape[0]*100:.1f}% of original)")
print(f"Expected training observations: {np.sum(propensity_train):.1f}")

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
    "propensity_train": torch.from_numpy(propensity_train),
    "propensity_val": torch.from_numpy(propensity_val),
    "mask_train": torch.from_numpy(mask_train),
    "mask_val": torch.from_numpy(mask_val)
}
if user_id_filtered is not None:
    output["user_id_train"] = torch.from_numpy(user_id_train)
    output["user_id_val"] = torch.from_numpy(user_id_val)
if user_id_test is not None:
    output["user_id_test"] = user_id_test

output_path = f"{args.output_dir}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
save_file(output, output_path)

# Save dataset statistics for later analysis (grouping/pairwise sampling, PU rates, etc.)
stats = {
    "model_name": args.model_name,
    "data_name": args.data_name,
    "alpha": float(args.alpha),
    "output_path": output_path,
    "stage1_train_path": f"{args.data_root}/{args.model_name}_{args.data_name}_train.safetensors",
    "stage1_test_path": f"{args.data_root}/{args.model_name}_{args.data_name}_test.safetensors",
    "user_id_available": user_id_filtered is not None,
    "user_id_test_available": user_id_test is not None,
    "target_observation_rate": 0.2,
    "n_stage1_train_total": int(embeddings.shape[0]),
    "n_stage1_train_nan_labels": int(np.isnan(labels).sum()),
    "n_stage1_train_after_nan_filter": int(embeddings_filtered.shape[0]),
    "n_split_train": int(X_train.shape[0]),
    "n_split_val": int(X_val.shape[0]),
    "n_test": int(data_test["embeddings"].shape[0]),
    "binary_threshold": float(np.median(data2levels[args.data_name])),
    "train": _split_stats(y_train_binary_true, mask_train, propensity_train),
    "val": _split_stats(y_val_binary_true, mask_val, propensity_val),
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

stats_path = f"{args.output_dir}/{args.model_name}_{args.data_name}_{args.alpha}_pu_stats.yaml"
tmp_stats_path = stats_path + ".tmp"
with open(tmp_stats_path, "w") as f:
    yaml.safe_dump(stats, f, sort_keys=False)
os.replace(tmp_stats_path, stats_path)
print(f"\nSaved stats to: {stats_path}")
