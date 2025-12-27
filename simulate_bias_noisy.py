import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
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


def add_noise_to_labels(labels, noise_rates, seed=0):
    """
    Add noise to binary labels by flipping them.
    
    Args:
        labels: Binary labels (0 or 1)
        noise_rates: [rate_pos_to_neg, rate_neg_to_pos] - probabilities of flipping
        seed: Random seed
    
    Returns:
        noisy_labels: Labels with noise added
    """
    np.random.seed(seed)
    
    labels_noisy = labels.copy()
    
    # Find positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)
    
    print(f"Before noise: {neg_count} negative, {pos_count} positive")
    
    # Generate random flips
    if pos_count > 0:
        pos_flip = np.random.binomial(1, noise_rates[0], pos_count)
        labels_noisy[pos_mask] = labels_noisy[pos_mask] * (1 - pos_flip)  # positive to negative
        pos_flipped = np.sum(pos_flip)
        print(f"Flipped {pos_flipped} positive samples to negative (rate: {noise_rates[0]})")
    
    if neg_count > 0:
        neg_flip = np.random.binomial(1, noise_rates[1], neg_count)
        labels_noisy[neg_mask] = labels_noisy[neg_mask] + neg_flip * (1 - labels_noisy[neg_mask])  # negative to positive
        neg_flipped = np.sum(neg_flip)
        print(f"Flipped {neg_flipped} negative samples to positive (rate: {noise_rates[1]})")
    
    final_pos_count = np.sum(labels_noisy == 1)
    final_neg_count = np.sum(labels_noisy == 0)
    print(f"After noise: {final_neg_count} negative, {final_pos_count} positive")
    
    return labels_noisy


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


parser = ArgumentParser(description="Linear Probing on Precomputed Embeddings")
parser.add_argument("--model_name", type=str, default="FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--data_name", type=str, default="saferlhf")
parser.add_argument("--data_root", type=str, default="../embeddings/normal")
parser.add_argument("--output_dir", type=str, default="../embeddings/biased_noisy")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for propensity calculation")
parser.add_argument("--r10", type=float, default=0.1, help="Noise rate for positive to negative")
parser.add_argument("--r01", type=float, default=0.1, help="Noise rate for negative to positive")
args = parser.parse_args()

data = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_train.safetensors")
embeddings = data["embeddings"].float().numpy()
labels = data["labels"].float().numpy()
print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")

# Mask data where target_label is nan
embeddings_filtered = embeddings[~np.isnan(labels)]
target_labels_filtered = labels[~np.isnan(labels)]
propensity_filtered = calculate_propensity(target_labels_filtered, args.alpha)
print(f"Original data size: {embeddings.shape[0]}")
print(f"Data size after filtering NaN values: {embeddings_filtered.shape[0]}")
print(f"Propensity range: [{propensity_filtered.min():.6f}, {propensity_filtered.max():.6f}]")


# Data split
X_train, X_val, y_train, y_val, propensity_train, propensity_val = train_test_split(embeddings_filtered, target_labels_filtered, propensity_filtered, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/embeddings_filtered.shape[0]*100:.1f}%)")

# Apply propensity sampling to training set
X_train_sampled, y_train_sampled, propensity_train_sampled, mask_train = apply_propensity_sampling(X_train, y_train, propensity_train, seed=0)
X_val_sampled, y_val_sampled, propensity_val_sampled, mask_val = apply_propensity_sampling(X_val, y_val, propensity_val, seed=0)
print(f"Training set after sampling: {X_train_sampled.shape[0]} samples ({X_train_sampled.shape[0]/X_train.shape[0]*100:.1f}% of original)")
print(f"Valid set after sampling: {X_val_sampled.shape[0]} samples ({X_val_sampled.shape[0]/X_val.shape[0]*100:.1f}% of original)")
print(f"Expected training observations: {np.sum(propensity_train):.1f}")

# Binarize
y_train_binary = binarize_labels(y_train, args.data_name)
y_val_binary = binarize_labels(y_val, args.data_name)
noise_rates = [args.r10, args.r01]
###### Add noise to observed positions only
y_train_binary[mask_train] = add_noise_to_labels(y_train_binary[mask_train], noise_rates, seed=0)
y_val_binary[mask_val] = add_noise_to_labels(y_val_binary[mask_val], noise_rates, seed=0)
###### Add noise to all positions
# y_train_binary = add_noise_to_labels(y_train_binary, noise_rates, seed=0)
# y_val_binary = add_noise_to_labels(y_val_binary, noise_rates, seed=0)

data_test = load_file(f"{args.data_root}/{args.model_name}_{args.data_name}_test.safetensors")
y_test = data_test["labels"].float().numpy()
y_test_binary = binarize_labels(y_test, args.data_name)

os.makedirs(args.output_dir, exist_ok=True)
save_file({
    # subsets that are used for unbiased evaluation
    "X_train": torch.from_numpy(X_train),
    "X_val": torch.from_numpy(X_val),
    "X_test": data_test["embeddings"],
    "y_train": torch.from_numpy(y_train),
    "y_val": torch.from_numpy(y_val),
    "y_test": torch.from_numpy(y_test),
    "y_train_binary": torch.from_numpy(y_train_binary),
    "y_val_binary": torch.from_numpy(y_val_binary),
    "y_test_binary": torch.from_numpy(y_test_binary),
    "propensity_train": torch.from_numpy(propensity_train),
    "propensity_val": torch.from_numpy(propensity_val),
    "mask_train": torch.from_numpy(mask_train),
    "mask_val": torch.from_numpy(mask_val)
}, f"{args.output_dir}/{args.model_name}_{args.data_name}_{args.alpha}_{args.r10}_{args.r01}.safetensors")
