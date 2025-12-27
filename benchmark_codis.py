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

from tools.utils import seed_everything, str2bool, drop_params, f1_score, load_data, save_metrics, refine_dict


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
    pre_parser.add_argument("--data_name", type=str, default="saferlhf")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults if dataset not listed
    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/codis/{pre_args.data_name}",
        "data_root": "../embeddings/biased_noisy",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "codis",
        "data_name": pre_args.data_name,
        "alpha": 0.5,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 600,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization
        "w_reg": 1.0,  # Task weight
        "rerun": False,
        "monitor_on": "train",
        "binary": True,
        "r10": 0.1,
        "r01": 0.2,
        "use_tqdm": True,
        "forget_rate": 0.2,  # Default noise rate/forget rate
        "num_gradual": 10,  # How many epochs to reach the final forget rate
        "co_lambda": 0.1,
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
            "alpha": 0.5,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,  # L2 regularization
            "w_reg": 10.0,  # Task weight
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
    parser.add_argument("--r10", type=float, help="Noise ratio for positive to negative")
    parser.add_argument("--r01", type=float, help="Noise ratio for negative to positive")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")
    parser.add_argument("--forget_rate", type=float, help="Expected noise rate for Co-teaching")
    parser.add_argument("--num_gradual", type=int, help="Epochs to ramp up forget rate")
    parser.add_argument("--co_lambda", type=float, help="Coefficient for JS divergence in CoDis selection")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def gen_forget_schedule(num_epochs, forget_rate, num_gradual):
    forget_rate_schedule = np.ones(num_epochs) * forget_rate
    forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
    return forget_rate_schedule


def compute_js_divergence(pred1, pred2, binary=True):
    """
    Compute Jensen-Shannon Divergence between two predictions.
    If binary=True, pred1/pred2 are logits, we calculate JS on Sigmoid probabilities.
    If binary=False, we use L2 distance as a proxy for discrepancy (since JS is for distributions).
    """
    if binary:
        # Calculate probabilities
        p1 = torch.sigmoid(pred1)
        p2 = torch.sigmoid(pred2)

        # Mean distribution
        m = 0.5 * (p1 + p2)

        # Binary KL Divergence: P(x)log(P(x)/M(x)) + (1-P(x))log((1-P(x))/(1-M(x)))
        eps = 1e-7
        kl1 = p1 * torch.log((p1 + eps) / (m + eps)) + (1 - p1) * torch.log((1 - p1 + eps) / (1 - m + eps))
        kl2 = p2 * torch.log((p2 + eps) / (m + eps)) + (1 - p2) * torch.log((1 - p2 + eps) / (1 - m + eps))

        return 0.5 * (kl1 + kl2)
    else:
        # For regression, use squared difference as discrepancy
        return (pred1 - pred2) ** 2


def loss_codis(loss1, loss2, pred1, pred2, forget_rate, mask, args):
    """
    CoDis loss logic: Select based on (Loss - lambda * Discrepancy).
    """
    # 1. Calculate Discrepancy (JS Divergence or L2)
    discrepancy = compute_js_divergence(pred1, pred2, args.binary)

    # 2. Compute CoDis Selection Metric: CE - lambda * JS
    metric1 = loss1 - args.co_lambda * discrepancy
    metric2 = loss2 - args.co_lambda * discrepancy

    # IMPORTANT: We must ignore unobserved data (mask=0).
    metric1[mask == 0] = float('inf')
    metric2[mask == 0] = float('inf')

    # Sort by metric
    ind1_sorted = torch.argsort(metric1.data)
    ind2_sorted = torch.argsort(metric2.data)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss1))

    # Determine indices to update
    ind1_update = ind1_sorted[:num_remember]
    ind2_update = ind2_sorted[:num_remember]

    # Filter out infinite (masked)
    ind1_update = ind1_update[metric1[ind1_update] != float('inf')]
    ind2_update = ind2_update[metric2[ind2_update] != float('inf')]

    # Cross-Update: Net1 uses samples selected by Net2 (based on Net2's metric)
    if len(ind2_update) == 0:
        loss1_mean = torch.tensor(0.0, device=loss1.device, requires_grad=True)
    else:
        # Note: We minimize the original Classification Loss, not the metric
        loss1_mean = torch.mean(loss1[ind2_update])

    # Cross-Update: Net2 uses samples selected by Net1
    if len(ind1_update) == 0:
        loss2_mean = torch.tensor(0.0, device=loss2.device, requires_grad=True)
    else:
        loss2_mean = torch.mean(loss2[ind1_update])

    return loss1_mean, loss2_mean


def train(net1, net2, train_loader, optimizer1, optimizer2, num_epochs, val_data, patience, args):
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    # Generate forget rate schedule
    rate_schedule = gen_forget_schedule(num_epochs, args.forget_rate, args.num_gradual)

    criterion = nn.MSELoss(reduction='none') if not args.binary else nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        epoch_loss1 = 0
        epoch_loss2 = 0

        net1.train()
        net2.train()

        bar = tqdm(train_loader, desc=f"Training Naive Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            reward_pred1 = net1(batch_X).squeeze()
            reward_pred2 = net2(batch_X).squeeze()

            # Calculate Co-teaching loss with cross-update logic
            loss1_raw, loss2_raw = criterion(reward_pred1, batch_y), criterion(reward_pred2, batch_y)
            loss1, loss2 = loss_codis(loss1_raw, loss2_raw, reward_pred1, reward_pred2, rate_schedule[epoch], batch_mask, args)

            # Apply task weight
            weighted_loss1 = args.w_reg * loss1
            weighted_loss2 = args.w_reg * loss2

            weighted_loss1.backward()
            optimizer1.step()
            weighted_loss2.backward()
            optimizer2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        val_loss1 = evaluate(net1, val_data, args)
        val_loss2 = evaluate(net2, val_data, args)
        val_loss = (val_loss1 + val_loss2) / 2
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Net1 loss: {epoch_loss1/len(train_loader):.5f}, Net2 loss: {epoch_loss2/len(train_loader):.5f}, Val loss: {val_loss:.5f}')

        epoch_loss = (epoch_loss1 + epoch_loss2) / 2
        monitor_loss = epoch_loss / len(train_loader) if args.monitor_on == "train" else val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(net1.state_dict(), f'{args.output_dir}/best_model_net1.pth')
            torch.save(net2.state_dict(), f'{args.output_dir}/best_model_net2.pth')
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
    print("Naive Reward Modeling")
    print("="*70)
    print("Loading embeddings and labels from Safetensors file...")
    if args.binary:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_{args.r10}_{args.r01}.safetensors"
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

    # Train reward model on observed data only
    print("\n" + "="*70)
    print("Step 2: Training Reward Model")
    print("="*70)
    train_loader = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size, shuffle=True)
    net1 = Model(X_train.shape[1], args.hidden_dim).to(device)
    net2 = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        net1=net1,
        net2=net2,
        train_loader=train_loader,
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        args=args
    )
    net1.load_state_dict(torch.load(f'{args.output_dir}/best_model_net1.pth'))
    net2.load_state_dict(torch.load(f'{args.output_dir}/best_model_net2.pth'))
    net1.eval()
    net2.eval()

    with torch.no_grad():
        # For all splits, get outputs
        def get_preds(X, y, mask):
            reward_pred1 = F.sigmoid(net1(X).squeeze()) if args.binary else net1(X).squeeze()
            reward_pred2 = F.sigmoid(net2(X).squeeze()) if args.binary else net2(X).squeeze()
            reward_pred = (reward_pred1 + reward_pred2) / 2
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
    }
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
