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
        "output_dir": f"./results/cache/upu/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "upu",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "class_prior": 0.5,  # Prior probability of positive class
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 200,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 20,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization
        "w_reg": 1.0,  # Task weight
        "rerun": False,
        "monitor_on": "val",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-6,  # L2 regularization
            "w_reg": 1.0,  # Task weight
        },
        "hs": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,  # L2 regularization
            "w_reg": 1.0,  # Task weight
        },
        "ufb": {
            "alpha": 0.5,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,  # L2 regularization
            "w_reg": 0.2,  # Task weight
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
    parser.add_argument("--class_prior", type=float, help="Prior probability of positive class for PU learning")
    parser.add_argument("--lr", type=float) # key parameter
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int) # key parameter
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient") # key parameter
    parser.add_argument("--w_reg", type=float, help="Task weight") # key parameter
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or test set")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary or continuous rewards")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    """
    Train reward model using classical PU learning loss:
    L = π * (loss_pp - loss_pn) + loss_un

    This is the unbiased risk estimator for PU learning.

    Where:
        - loss_pp = BCE(positive samples, target=1) - predicting positives as positive
        - loss_pn = BCE(positive samples, target=0) - predicting positives as negative
        - loss_un = BCE(unknown samples, target=0) - predicting unknowns as negative
        - π = class_prior (probability of positive class)

    In PU learning context:
        - y=1: observed positive samples (labeled positives)
        - y=0: unlabeled samples (mix of positives and negatives)
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.MSELoss(reduction='none') if not args.binary else nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()

        bar = tqdm(train_loader, desc=f"Training PU Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()

            # Separate positive (y=1) and unlabeled/negative (y=0) samples
            pos_mask = batch_y == 1
            neg_mask = batch_y == 0

            # Compute losses on positive samples
            if pos_mask.sum() > 0:
                pos_pred = reward_pred[pos_mask]
                # loss_pp: loss for classifying positives as positive (target=1)
                loss_pp = criterion(pos_pred, torch.ones_like(pos_pred)).mean()
                # loss_pn: loss for classifying positives as negative (target=0)
                loss_pn = criterion(pos_pred, torch.zeros_like(pos_pred)).mean()
            else:
                loss_pp = torch.tensor(0.0, device=batch_X.device)
                loss_pn = torch.tensor(0.0, device=batch_X.device)

            # Compute loss on unlabeled/negative samples
            if neg_mask.sum() > 0:
                neg_pred = reward_pred[neg_mask]
                # loss_un: loss for classifying unknowns as negative (target=0)
                loss_un = criterion(neg_pred, torch.zeros_like(neg_pred)).mean()
            else:
                loss_un = torch.tensor(0.0, device=batch_X.device)

            # Classical PU loss: π * (loss_pp - loss_pn) + loss_un
            loss = args.class_prior * (loss_pp - loss_pn) + loss_un

            # Apply task weight
            weighted_loss = args.w_reg * loss

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
    print("Unbiased PU (uPU) Learning Reward Modeling")
    print(f"Class Prior (π): {args.class_prior}")
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

    # For PU learning: use ALL samples, not just observed
    # y_train_binary already has: observed→true label, unobserved→0
    X_train = X_train_full  # Use all samples
    y_train = y_train_full  # y=1 for observed pos, y=0 for (observed neg + unobserved)
    X_val = X_val_full
    y_val = y_val_full

    print(f"Training on {X_train.shape[0]} samples (full dataset, no masking).")
    print(f"  - Positive samples (y=1): {(y_train == 1).sum().item()}")
    print(f"  - Unlabeled samples (y=0): {(y_train == 0).sum().item()}")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    val_data = (X_val_full, y_val_full, mask_val.float())
    test_data = (X_test, y_test, torch.ones_like(y_test))  # mask not used for test

    # Train reward model on all data (PU learning approach)
    print("\n" + "="*70)
    print("Step 1: Training uPU Reward Model")
    print("="*70)
    train_loader = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size, shuffle=True)
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
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
        # For all splits, get outputs
        def get_preds(X, y, mask):
            reward_pred = F.sigmoid(model(X).squeeze()) if args.binary else model(X).squeeze()
            reward_pred = reward_pred.detach().cpu().numpy()
            y_cpu = y.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            return reward_pred, y_cpu, mask_cpu

        y_train_pred, y_train_cpu, mask_train_cpu = get_preds(X_train_full, y_train_full, mask_train.float())
        y_val_pred, y_val_cpu, mask_val_cpu = get_preds(*val_data)
        y_test_pred, y_test_cpu, _ = get_preds(*test_data)

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
    }
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
