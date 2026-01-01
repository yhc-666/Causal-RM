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

from tools.utils import seed_everything, str2bool, drop_params, f1_score, load_data, save_metrics, refine_dict, compute_nll, compute_ndcg_binary, compute_recall_binary


class Model(nn.Module):
    """
    Multitask neural network with three outputs:
    1. Propensity score: π(x) = P(mask=1 | x)
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
        self.reward_head = nn.Linear(hidden_dims[-1], 1)     # Reward output

        nn.init.normal_(self.propensity_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.propensity_head.bias, 0.0)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reward_head.bias, 0.0)

    def forward(self, x):
        # Shared feature extraction
        for layer in self.shared_layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return {
            'propensity': self.propensity_head(x),
            'reward': self.reward_head(x)
        }


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="saferlhf")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults if dataset not listed
    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/mtips/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "mtips",
        "data_name": pre_args.data_name,
        "alpha": 0.5,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 600,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization for the model
        "w_prop": 1.0,  # Task weight for propensity model training
        "w_reg": 1.0,  # Task weight for IPS model training
        "rerun": True,
        "monitor_on": "train",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
        },
        "hs": {
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
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient for model") # key parameter
    parser.add_argument("--w_prop", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--w_reg", type=float, help="Task weight for IPS model training") # key parameter
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
    L_total = w_prop * L_prop + w_reg * L_IPS
    
    where:
        - L_prop: Propensity prediction loss
        - L_IPS: IPS loss
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    # Loss functions
    criterion_prop = nn.MSELoss()
    criterion_reward = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()
    criterion_reward_none = nn.MSELoss(reduction='none') if not args.binary else nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_prop = 0.0
        epoch_loss_ips = 0.0

        bar = tqdm(train_loader, desc=f"Training Multitask Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            prop_logits = outputs['propensity'].squeeze()
            reward_pred = outputs['reward'].squeeze()

            prop_pred = F.sigmoid(prop_logits)
            prop_clipped = torch.clip(prop_pred, args.clip_min, 1.0).detach()

            # Task 1: Propensity loss L_prop = ℓ(π(x), mask)
            loss_prop = criterion_prop(prop_pred, batch_mask.float())

            # Task 2: IPS loss L_IPS = mask * (error) / π
            error = criterion_reward_none(reward_pred, batch_y)
            mask_float = batch_mask.float()
            ips_loss_per_sample = mask_float * error / prop_clipped
            loss_ips = torch.mean(ips_loss_per_sample)

            # Combined multitask loss
            total_loss = (
                args.w_prop * loss_prop +
                args.w_reg * loss_ips
            )
            total_loss = args.w_prop * loss_prop if epoch < 10 else total_loss

            total_loss.backward()
            optimizer.step()

            # Track losses for reporting
            epoch_loss += total_loss.item()
            epoch_loss_prop += loss_prop.item()
            epoch_loss_ips += loss_ips.item()

        # Evaluate on validation set
        val_loss = evaluate_multitask(model, val_data, args)

        if epoch % 4 == 0:
            avg_total = epoch_loss / len(train_loader)
            avg_prop = epoch_loss_prop / len(train_loader)
            avg_ips = epoch_loss_ips / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Total: {avg_total:.5f}, '
                  f'Prop: {avg_prop:.5f}, '
                  f'IPS: {avg_ips:.5f}, '
                  f'Val: {val_loss:.5f}')

        monitor_loss = avg_ips if args.monitor_on == "train" else val_loss
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
    Evaluate multitask model on validation data.
    Returns the combined validation loss.
    """
    model.eval()
    criterion_prop = nn.MSELoss()
    criterion_reward = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()

    with torch.no_grad():
        X, y, mask = val_data
        outputs = model(X)

        prop_pred = outputs['propensity'].squeeze()
        prop_pred = F.sigmoid(prop_pred)
        reward_pred = outputs['reward'].squeeze()

        # Propensity loss
        loss_prop = criterion_prop(prop_pred, mask.float())

        # Reward loss (on observed samples)
        observed = mask > 0.5
        loss_reward = torch.tensor(0.0, device=X.device)
        if observed.sum() > 0:
            loss_reward = criterion_reward(reward_pred[observed], y[observed])

        # Combined validation loss (weighted)
        val_loss = (
            args.w_prop * loss_prop.item() +
            args.w_reg * loss_reward.item()
        )

    return val_loss


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
    print("Multitask Inverse Propensity Score (IPS) Reward Modeling")
    print("="*70)
    print("Loading embeddings and labels from Safetensors file...")
    if args.binary:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, mask_val, X_test, y_test = \
            load_data(embedding_file, device, keys=["X_train", "y_train_binary", "mask_train", "X_val", "y_val_binary", "mask_val", "X_test", "y_test_binary"])
    else:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, mask_val, X_test, y_test = \
            load_data(embedding_file, device, keys=["X_train", "y_train", "mask_train", "X_val", "y_val", "mask_val", "X_test", "y_test"])

    X_train, y_train = X_train_full[mask_train], y_train_full[mask_train]
    X_val, y_val = X_val_full[mask_val], y_val_full[mask_val]
    print(f"Training on {X_train.shape[0]} samples.")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    val_data = (X_val_full, y_val_full, mask_val.float())
    test_data = (X_test, y_test, torch.ones_like(y_test))  # mask not used for test

    # Multitask learning: single model with three outputs
    print("\n" + "="*70)
    print("Multitask IPS Training")
    print("="*70)
    print(f"Task weights: w_prop={args.w_prop}, w_reg={args.w_reg}")
    print("Combined loss: L_total = w_prop * L_prop + w_reg * L_IPS")
    train_loader = DataLoader(
        TensorDataset(X_train_full, y_train_full, mask_train.float()), 
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
        val_data=val_data,
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
        prop_val_pred, y_val_pred, y_val_cpu, mask_val_cpu = get_preds(*val_data)
        prop_test_pred, y_test_pred, y_test_cpu, _ = get_preds(*test_data)

    # Only evaluate reward metrics on observed samples
    obs_train = mask_train_cpu > 0.5
    obs_val = mask_val_cpu > 0.5

    metrics = {
        "R2 on train": r2_score(y_train_cpu[obs_train], y_train_pred[obs_train]) if obs_train.sum() > 0 else float('nan'),
        "R2 on val": r2_score(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "MAE on eval": mean_absolute_error(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "RMSE on eval": np.sqrt(mean_squared_error(y_val_cpu[obs_val], y_val_pred[obs_val])) if obs_val.sum() > 0 else float('nan'),
        "RMSE on test": np.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on eval": roc_auc_score(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),
        "Pearson on eval": pearsonr(y_val_cpu[obs_val], y_val_pred[obs_val])[0] if obs_val.sum() > 0 else float('nan'),
        "Pearson on test": pearsonr(y_test_cpu, y_test_pred)[0],
        "NLL on eval": compute_nll(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "NLL on test": compute_nll(y_test_cpu, y_test_pred),
        "NDCG on eval": compute_ndcg_binary(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "NDCG on test": compute_ndcg_binary(y_test_cpu, y_test_pred),
        "Recall on eval": compute_recall_binary(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float('nan'),
        "Recall on test": compute_recall_binary(y_test_cpu, y_test_pred),
        "R2 prop on train": r2_score(mask_train_cpu, prop_train_pred),
        "R2 prop on val": r2_score(mask_val_cpu, prop_val_pred),
        "MAE prop on train": mean_absolute_error(mask_train_cpu, prop_train_pred),
        "MAE prop on val": mean_absolute_error(mask_val_cpu, prop_val_pred),
        "Max error prop on train": np.max(np.abs(mask_train_cpu - prop_train_pred)),
        "Max error prop on val": np.max(np.abs(mask_val_cpu - prop_val_pred)),
    }
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
