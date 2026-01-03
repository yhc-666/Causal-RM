"""
Doubly Robust (DR) Reward Modeling (PU baseline)

This script adapts a DR-style objective to the PU setting where the algorithm cannot use
the observation mask:
  - Train on `y_train_binary` and treat all `0` (UNK) as negative (missing-as-negative baseline).
  - Compute per-sample propensity from the (noisy) 0/1 labels using `calculate_propensity`
    (copied from `simulate_bias_pu.py`) and clip to `[clip_min, 1]`.
  - Use a PU-consistent DR form where only labeled positives are treated as "observed":
      L = y * (error - error_hat) / pi + error_hat
    where:
      - `y ∈ {0,1}` is the PU label (1 = labeled positive, 0 = unlabeled)
      - `error = ℓ(p, y)` and `error_hat = ℓ(p, p_baseline)`
      - `p = sigmoid(r)` is the model probability.

Evaluation reports mask-blind metrics on full train/val PU labels (y_*_binary) and
test metrics on the clean test split.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from argparse import ArgumentParser
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, log_loss, precision_recall_curve
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tools.utils import seed_everything, str2bool, drop_params, f1_score, load_data, save_metrics, refine_dict, compute_nll, compute_ndcg_binary, compute_recall_binary, add_tuned_recall_metrics


def calculate_propensity(labels, alpha, target_observation_rate=0.2):
    """
    Copy from simulate_bias_pu.py (do not import/modify simulate_bias_pu.py).
    """
    propensity = np.ones_like(labels)
    labels = 1 + (labels - labels.min()) * 4 / (labels.max() - labels.min())

    mask_lt_4 = labels < labels.max()
    propensity[mask_lt_4] = alpha ** (labels.max() - labels[mask_lt_4])

    mask_ge_4 = labels >= labels.max()
    propensity[mask_ge_4] = 1.0

    expected_observations = target_observation_rate * len(labels)
    k = expected_observations / np.sum(propensity)
    propensity = propensity * k

    return propensity


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(',')))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        x = self.output_layer(x)
        return x


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults if dataset not listed
    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/dr/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "dr",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 200,
        "batch_size": 512,
        "batch_size_prop": 512,
        "hidden_dim": "256,64",
        "hidden_dim_prop": "256,64",
        "patience": 20,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization for DR model
        "l2_prop": 1e-6,  # L2 regularization for propensity model
        "l2_imp": 1e-6,  # L2 regularization for imputation (baseline) model
        "w_prop": 1.0,  # Task weight for propensity model training
        "w_imp": 1.0,  # Task weight for imputation (baseline) model training
        "w_reg": 1.0,  # Task weight for DR model training
        "rerun": False,
        "monitor_on": "test",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.1,
            "batch_size": 512,
            "lr": 0.0005,
        },
        "hs": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
        },
        "ufb": {
        }
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    # Full parser
    parser = ArgumentParser(description="")
    parser.add_argument("--desc", type=str, default="foo")
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--alpha", type=float, help="Alpha parameter for propensity calculation")
    parser.add_argument("--lr", type=float) # key parameter
    parser.add_argument("--clip_min", type=float, help="Minimum clip value for propensity weights")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int) # key parameter
    parser.add_argument("--batch_size_prop", type=int) # second key parameter
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--hidden_dim_prop", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient for DR model") # key parameter
    parser.add_argument("--l2_prop", type=float, help="L2 regularization coefficient for propensity model") # key parameter
    parser.add_argument("--l2_imp", type=float, help="L2 regularization coefficient for imputation (baseline) model") # key parameter
    parser.add_argument("--w_prop", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--w_imp", type=float, help="Task weight for imputation (baseline) model training") # key parameter
    parser.add_argument("--w_reg", type=float, help="Task weight for DR model training") # key parameter
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or test set")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary or continuous rewards")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train_baseline_model(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    """
    PU setting (mask-invisible): train a baseline model on the full PU dataset
    (y_train_binary with UNK treated as 0).
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0
    criterion = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        bar = tqdm(train_loader, desc=f"Training Baseline Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()
            loss = criterion(reward_pred, batch_y.squeeze())
            weighted_loss = loss * args.w_imp
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Track unweighted loss for reporting

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Baseline Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/len(train_loader):.5f}, Val loss: {val_loss:.5f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_baseline_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for baseline model after {epoch + 1} epochs.")
                break


def train(model, baseline_model, train_loader, optimizer, num_epochs, val_data, patience, args):
    """
    PU setting (mask is NOT visible to the model):
      - Treat all `y_train_binary == 0` as negative (UNK -> negative).
      - Compute sample-wise propensity via `calculate_propensity(y_train_binary, alpha)` (noisy baseline).
      - Use a PU-consistent DR-style objective with implicit observation indicator m=y:
          L = y * (error - error_hat) / clip(pi) + error_hat

    Note: This is a baseline adaptation; it is not the original DR estimator for missing-at-random labels.
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.MSELoss(reduction="none") if not args.binary else None
    eps = 1e-6

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()
        baseline_model.eval()  # Baseline model is frozen during DR training

        bar = tqdm(train_loader, desc=f"Training DR Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_propensity in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()
            batch_y = batch_y.float().squeeze()

            with torch.no_grad():
                # Get baseline predictions (no gradient)
                baseline_pred = baseline_model(batch_X).squeeze()
                if args.binary:
                    baseline_pred = torch.clamp(torch.sigmoid(baseline_pred), eps, 1.0 - eps)

            prop_clipped = torch.clip(batch_propensity.float(), args.clip_min, 1.0).detach()

            if args.binary:
                # Use probability-space BCE with clamping to avoid unbounded objectives.
                p = torch.clamp(torch.sigmoid(reward_pred), eps, 1.0 - eps)
                error = F.binary_cross_entropy(p, batch_y, reduction="none")
                error_hat = F.binary_cross_entropy(p, baseline_pred, reduction="none")

                # PU-consistent: only labeled positives are treated as observed (m = y).
                dr_loss_per_sample = batch_y * (error - error_hat) / prop_clipped + error_hat
            else:
                # Regression: DR in MSE space (baseline_pred is raw prediction).
                error = criterion(reward_pred, batch_y)
                error_hat = criterion(reward_pred, baseline_pred)
                dr_loss_per_sample = batch_y * (error - error_hat) / prop_clipped + error_hat

            # Average over batch
            loss = torch.mean(dr_loss_per_sample)
            # Apply task weight
            weighted_loss = loss * args.w_reg
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Track unweighted loss for reporting

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/len(train_loader):.5f}, Val loss: {val_loss:.5f}')

        monitor_loss = epoch_loss / len(train_loader) if args.monitor_on == "train" else val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def evaluate(model, val_data, args, propensity=False):
    model.eval()
    with torch.no_grad():
        X, y, mask = val_data
        outputs = model(X).squeeze()

        if propensity:  # evaluation of the propensity model
            criterion_mean = nn.MSELoss()
            loss = criterion_mean(F.sigmoid(outputs), mask.float())
        else:  # evaluation of the reward prediction or imputation/baseline model
            criterion_mean = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()
            loss = criterion_mean(outputs, y.float())
    return loss.item()


def main():
    args = parse_arguments()

    if args.is_training and os.path.exists(f"{args.output_dir}/performance.yaml") and not args.rerun:
        print(f"The path {args.output_dir}/performance.yaml exists!!")
        sys.exit()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*70)
    print("Doubly Robust (DR) Reward Modeling (PU setting: UNK->0, mask-invisible)")
    print("="*70)
    print("Loading embeddings and labels from Safetensors file...")
    if args.binary:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, y_val_true, mask_val, X_test, y_test = \
            load_data(embedding_file, device, keys=["X_train", "y_train_binary", "mask_train", "X_val", "y_val_binary", "y_val_binary_true", "mask_val", "X_test", "y_test_binary"])
    else:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, mask_val, X_test, y_test = \
            load_data(embedding_file, device, keys=["X_train", "y_train", "mask_train", "X_val", "y_val", "mask_val", "X_test", "y_test"])

    X_train, y_train = X_train_full, y_train_full
    X_val, y_val = X_val_full, y_val_full
    print(f"Training on {X_train.shape[0]} samples (full PU dataset).")
    if args.binary:
        print(f"  - y=1 (labeled positives): {(y_train == 1).sum().item()}")
        print(f"  - y=0 (UNK treated as negative): {(y_train == 0).sum().item()}")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    val_data = (X_val_full, y_val_full, mask_val.float())
    val_data_true = (X_val_full, y_val_true, mask_val.float()) if args.binary else val_data
    test_data = (X_test, y_test, torch.ones_like(y_test))  # mask not used for test

    propensity_train_np = calculate_propensity(y_train_full.detach().cpu().numpy(), args.alpha)
    propensity_train = torch.from_numpy(propensity_train_np).to(device=device, dtype=torch.float32)

    # Step 1: Train baseline reward model on the full PU dataset
    print("\n" + "="*70)
    print("Step 1: Training Baseline Reward Model")
    print("="*70)
    train_loader_baseline = DataLoader(TensorDataset(X_train_full, y_train_full), batch_size=args.batch_size, shuffle=True)
    baseline_model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=args.lr, weight_decay=args.l2_imp)

    train_baseline_model(
        model=baseline_model,
        train_loader=train_loader_baseline,
        optimizer=optimizer_baseline,
        num_epochs=args.num_epochs,
        val_data=val_data_true,
        patience=args.patience,
        args=args
    )
    del train_loader_baseline, optimizer_baseline
    baseline_model.load_state_dict(torch.load(f'{args.output_dir}/best_baseline_model.pth'))
    baseline_model.eval()

    # Step 2: Train final reward model using DR-style loss + label-based propensity
    print("\n" + "="*70)
    print("Step 2: Training Reward Model with Doubly Robust (DR)")
    print("="*70)
    print("Using DR-style loss (PU-consistent): L = y * (error - error_hat) / π + error_hat")
    train_loader = DataLoader(TensorDataset(X_train_full, y_train_full, propensity_train), batch_size=args.batch_size, shuffle=True)
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
        baseline_model=baseline_model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        val_data=val_data_true,
        patience=args.patience,
        args=args
    )
    model.load_state_dict(torch.load(f'{args.output_dir}/best_model.pth'))
    model.eval()

    with torch.no_grad():
        # For all splits, get outputs
        def get_preds(X, y, mask):
            reward_pred = F.sigmoid(model(X).squeeze()) if args.binary else model(X).squeeze()
            reward_pred = reward_pred.cpu().numpy()
            y_cpu = y.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            return reward_pred, y_cpu, mask_cpu

        y_train_pred, y_train_cpu, mask_train_cpu = get_preds(X_train_full, y_train_full, mask_train.float())
        y_val_pred, y_val_cpu, mask_val_cpu = get_preds(*val_data)
        y_test_pred, y_test_cpu, _ = get_preds(*test_data)

    # Mask-blind metrics on full train/val PU labels (y_*_binary).
    metrics = {
        "R2 on train": r2_score(y_train_cpu, y_train_pred),
        "R2 on val": r2_score(y_val_cpu, y_val_pred),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "MAE on eval": mean_absolute_error(y_val_cpu, y_val_pred),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "RMSE on eval": np.sqrt(mean_squared_error(y_val_cpu, y_val_pred)),
        "RMSE on test": np.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on eval": roc_auc_score(y_val_cpu, y_val_pred),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),
        "Pearson on eval": pearsonr(y_val_cpu, y_val_pred)[0],
        "Pearson on test": pearsonr(y_test_cpu, y_test_pred)[0],
        "NLL on eval": compute_nll(y_val_cpu, y_val_pred),
        "NLL on test": compute_nll(y_test_cpu, y_test_pred),
        "NDCG on eval": compute_ndcg_binary(y_val_cpu, y_val_pred),
        "NDCG on test": compute_ndcg_binary(y_test_cpu, y_test_pred),
    }
    add_tuned_recall_metrics(metrics, y_val_cpu, y_val_pred, y_test_cpu, y_test_pred)
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
