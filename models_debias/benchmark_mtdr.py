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
    """
    Multitask neural network with three outputs:
    1. Propensity score: π(x) = P(mask=1 | x)
    2. Imputation reward: r_baseline(x)
    3. Reward prediction: r(x)
    """
    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(',')))
        # Shared layers
        self.shared_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        # Task-specific output heads
        self.propensity_head = nn.Linear(hidden_dims[-1], 1)  # Propensity score output
        self.baseline_head = nn.Linear(hidden_dims[-1], 1)   # Baseline/imputation output
        self.reward_head = nn.Linear(hidden_dims[-1], 1)     # Reward output

        nn.init.normal_(self.propensity_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.propensity_head.bias, 0.0)
        nn.init.normal_(self.baseline_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.baseline_head.bias, 0.0)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reward_head.bias, 0.0)

    def forward(self, x):
        # Shared feature extraction
        for layer in self.shared_layers:
            x = torch.nn.functional.leaky_relu(layer(x))

        # Task-specific outputs
        return {
            'propensity': self.propensity_head(x),
            'baseline': self.baseline_head(x),
            'reward': self.reward_head(x)
        }


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults if dataset not listed
    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/mtdr/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "mtdr",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 200,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 20,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization for DR model
        "w_prop": 1.0,  # Task weight for propensity model training
        "w_imp": 1.0,  # Task weight for imputation (baseline) model training
        "w_reg": 1.0,  # Task weight for DR model training
        "rerun": False,
        "monitor_on": "val",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.5,
            "lr": 0.0005,
            "l2_reg": 1e-06,
            "batch_size": 512,
            "clip_min": 0.1,
            "w_prop": 1.0,
            "w_imp": 1.0,
            "w_reg": 1.0,
        },
        "hs": {
            "alpha": 0.5,
            "lr": 0.000182006,
            "l2_reg": 1.07e-08,
            "batch_size": 256,
            "clip_min": 0.2,
            "w_prop": 2.0,
            "w_imp": 2.0,
            "w_reg": 1.0,
            "hidden_dim": "512,128",
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
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient for DR model") # key parameter
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


def train(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    """
    Train multitask model with combined loss:
      PU setting (mask-invisible):
        - Treat all `y_train_binary==0` as negative (UNK->0).
        - Use label-based propensity target computed by `calculate_propensity(y_train_binary, alpha)`.

      L_total = w_prop * L_prop + w_imp * L_imp + w_reg * L_DR
        - L_prop: MSE(π̂(x), π_target)
        - L_imp: baseline/imputation loss on full PU dataset
        - L_DR: DR-style loss with implicit observation indicator m=y (mask-blind PU):
            L_DR = y * (error - error_hat) / clip(π̂(x)) + error_hat
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0
    eps = 1e-6

    # Loss functions
    criterion_prop = nn.MSELoss()
    criterion_reward = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()
    criterion_reward_none = nn.MSELoss(reduction='none') if not args.binary else nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_prop = 0.0
        epoch_loss_imp = 0.0
        epoch_loss_dr = 0.0

        bar = tqdm(train_loader, desc=f"Training Multitask Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_propensity_target in bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            prop_logits = outputs['propensity'].squeeze()
            baseline_pred = outputs['baseline'].squeeze()
            reward_pred = outputs['reward'].squeeze()

            prop_pred = F.sigmoid(prop_logits)
            prop_clipped = torch.clip(prop_pred, args.clip_min, 1.0).detach()

            # Task 1: Propensity loss L_prop = ℓ(π̂(x), π_target)
            target = torch.clip(batch_propensity_target.float(), 0.0, 1.0)
            loss_prop = criterion_prop(prop_pred, target)

            # Task 2: Imputation loss L_imp = ℓ(r_baseline(x), y) on full PU dataset
            loss_imp = criterion_reward(baseline_pred, batch_y)

            # Task 3: Doubly Robust loss L_DR = mask * (error - \hat{error}) / π + \hat{error}
            # error = ℓ(r(x), y), \hat{error} = ℓ(r(x), r_baseline(x))
            if args.binary:
                batch_y_float = batch_y.float().squeeze()
                p = torch.clamp(torch.sigmoid(reward_pred), eps, 1.0 - eps)
                baseline_p = torch.clamp(torch.sigmoid(baseline_pred.detach()), eps, 1.0 - eps)
                error = F.binary_cross_entropy(p, batch_y_float, reduction="none")
                error_hat = F.binary_cross_entropy(p, baseline_p, reduction="none")
                error_diff = error - error_hat
                dr_loss_per_sample = batch_y_float * error_diff / prop_clipped + error_hat
            else:
                error = criterion_reward_none(reward_pred, batch_y)
                error_hat = criterion_reward_none(reward_pred, baseline_pred.detach())
                error_diff = error - error_hat
                dr_loss_per_sample = error_diff / prop_clipped + error_hat
            loss_dr = torch.mean(dr_loss_per_sample)

            # Combined multitask loss
            total_loss = (
                args.w_prop * loss_prop +
                args.w_imp * loss_imp +
                args.w_reg * loss_dr
            )
            total_loss = args.w_prop * loss_prop if epoch < 10 else total_loss
            # total_loss = args.w_prop * loss_prop

            total_loss.backward()
            optimizer.step()

            # Track losses for reporting
            epoch_loss += total_loss.item()
            epoch_loss_prop += loss_prop.item()
            epoch_loss_imp += loss_imp.item() if isinstance(loss_imp, torch.Tensor) else 0.0
            epoch_loss_dr += loss_dr.item()

        # Evaluate on validation set
        val_loss = evaluate_multitask(model, val_data, args)

        if epoch % 4 == 0:
            avg_total = epoch_loss / len(train_loader)
            avg_prop = epoch_loss_prop / len(train_loader)
            avg_imp = epoch_loss_imp / len(train_loader)
            avg_dr = epoch_loss_dr / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Total: {avg_total:.5f}, '
                  f'Prop: {avg_prop:.5f}, '
                  f'Imp: {avg_imp:.5f}, '
                  f'DR: {avg_dr:.5f}, '
                  f'Val: {val_loss:.5f}')

        monitor_loss = avg_dr if args.monitor_on == "train" else val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def evaluate_multitask(model, val_data, args):
    """
    Validation loss for early-stopping/monitoring.

    For binary PU experiments we use the *clean* oracle label `y_val_binary_true` with BCE on the reward head,
    and treat it as the single `val loss` across all methods for fair comparison.
    """
    model.eval()
    criterion_reward = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()

    with torch.no_grad():
        X, y, _mask, _propensity_target = val_data
        outputs = model(X)
        reward_pred = outputs['reward'].squeeze()
        loss_reward = criterion_reward(reward_pred, y.float())
    return loss_reward.item()


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
    print("Multitask Doubly Robust (DR) Reward Modeling (PU setting: UNK->0, mask-invisible)")
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

    propensity_train_np = calculate_propensity(y_train_full.detach().cpu().numpy(), args.alpha)
    propensity_val_np = calculate_propensity(y_val_full.detach().cpu().numpy(), args.alpha)
    propensity_train_np = np.clip(propensity_train_np, 0.0, 1.0)
    propensity_val_np = np.clip(propensity_val_np, 0.0, 1.0)
    propensity_train = torch.from_numpy(propensity_train_np).to(device=device, dtype=torch.float32)
    propensity_val = torch.from_numpy(propensity_val_np).to(device=device, dtype=torch.float32)

    val_data = (X_val_full, y_val_full, mask_val.float(), propensity_val)
    val_data_true = (X_val_full, y_val_true, mask_val.float(), propensity_val) if args.binary else val_data
    test_data = (X_test, y_test, torch.ones_like(y_test))  # mask not used for test

    # Multitask learning: single model with three outputs
    print("\n" + "="*70)
    print("Multitask Doubly Robust (DR) Training")
    print("="*70)
    print(f"Task weights: w_prop={args.w_prop}, w_imp={args.w_imp}, w_reg={args.w_reg}")
    print("Combined loss: L_total = w_prop * L_prop + w_imp * L_imp + w_reg * L_DR")
    train_loader = DataLoader(
        TensorDataset(X_train_full, y_train_full, propensity_train),
        batch_size=args.batch_size, 
        shuffle=True
    )
    # Single multitask model
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
    # Use combined L2 regularization (could also use separate per-head regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
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
        # For all splits, get outputs from multitask model
        def get_preds(X, y, mask):
            outputs = model(X)
            prop_logits = outputs['propensity'].squeeze()
            prop_pred = F.sigmoid(prop_logits).cpu().numpy()
            reward_pred = outputs['reward'].squeeze()
            if args.binary:
                reward_pred = F.sigmoid(reward_pred)
            reward_pred = reward_pred.cpu().numpy()
            y_cpu = y.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            return prop_pred, reward_pred, y_cpu, mask_cpu

        prop_train_pred, y_train_pred, y_train_cpu, mask_train_cpu = get_preds(
            X_train_full, y_train_full, mask_train.float()
        )
        prop_val_pred, y_val_pred, y_val_cpu, mask_val_cpu = get_preds(
            X_val_full, y_val_full, mask_val.float()
        )
        prop_test_pred, y_test_pred, y_test_cpu, _ = get_preds(*test_data)

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
        "R2 prop_target on train": r2_score(propensity_train_np, prop_train_pred),
        "R2 prop_target on val": r2_score(propensity_val_np, prop_val_pred),
        "MAE prop_target on train": mean_absolute_error(propensity_train_np, prop_train_pred),
        "MAE prop_target on val": mean_absolute_error(propensity_val_np, prop_val_pred),
    }
    add_tuned_recall_metrics(metrics, y_val_cpu, y_val_pred, y_test_cpu, y_test_pred)
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
