import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from argparse import ArgumentParser
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tools.utils import (
    seed_everything,
    str2bool,
    drop_params,
    f1_score,
    load_data,
    save_metrics,
    refine_dict,
    compute_nll,
    compute_ndcg_binary,
    compute_recall_binary,
)


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


class Model(nn.Module):
    """
    UPRL has two prediction branches:
      - pointwise branch: used to estimate P(y=1|x) and provide correction terms
      - pairwise branch: used for ranking / final prediction

    In the original recommender implementation, these are two independent embedding models.
    Here we keep the same separation by using two independent MLPs.
    """

    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(",")))

        self.point_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.point_head = nn.Linear(hidden_dims[-1], 1)

        self.pair_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.pair_head = nn.Linear(hidden_dims[-1], 1)

    def forward_point(self, x):
        for layer in self.point_layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return self.point_head(x)

    def forward_pair(self, x):
        for layer in self.pair_layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return self.pair_head(x)

    def forward(self, x):
        # For evaluation we follow the original UPRL.predict(): use the pairwise branch.
        return self.forward_pair(x)


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/uprl/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "uprl",
        "data_name": pre_args.data_name,
        "alpha": 0.1,
        "lr": 0.0002,
        "clip_min": 1e-8,
        "num_epochs": 600,
        "batch_size": 512,
        "num_neg": 10,  # number of j samples per i (global pairwise)
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,
        "w_reg": 1.0,
        "rerun": False,
        "monitor_on": "train",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-6,
            "w_reg": 1.0,
        },
        "hs": {
            "alpha": 0.5,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,
            "w_reg": 10.0,
        },
        "ufb": {
            "alpha": 0.5,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,
            "w_reg": 0.2,
        },
    }
    ds_defaults = dataset_defaults.get(pre_args.data_name, {})
    merged_defaults = {**base_defaults, **ds_defaults}

    parser = ArgumentParser(description="")
    parser.add_argument("--desc", type=str, default="foo")
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--alpha", type=float, help="Alpha parameter for propensity calculation")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--clip_min", type=float, help="Minimum clip value for propensity weights")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_neg", type=int, help="Number of j samples per i (global pairwise)")
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient")
    parser.add_argument("--w_reg", type=float, help="Task weight")
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or test set")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary or continuous rewards")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train(model, train_data, optimizer, num_epochs, val_data, patience, args):
    """
    Global pairwise UPRL adapted from old recommender UPRL:

      - i sampled from positives (y=1)
      - j sampled from all samples
      - pairwise loss uses pair-branch scores and a correction factor from point-branch predictions
      - pointwise loss trains the point-branch with an IPS-like objective

    This implementation follows the original formula structure but replaces (user,item) scoring
    with MLP scoring on embeddings (global pairs, no user grouping).
    """
    if not args.is_training:
        return
    if not args.binary:
        raise NotImplementedError("UPRL baseline currently supports binary (0/1) labels only.")

    X_train_full, y_train_full, propensity_train = train_data
    device = X_train_full.device
    n = y_train_full.shape[0]

    pos_idx = torch.nonzero(y_train_full > 0.5, as_tuple=False).squeeze(-1)
    if pos_idx.numel() == 0 or n <= 1:
        print(f"Skip training: pos={pos_idx.numel()} n={n}")
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
        return

    best_loss = float("inf")
    patience_counter = 0

    eps = 1e-5

    for epoch in range(num_epochs):
        perm = torch.randperm(pos_idx.numel(), device=device)
        pos_shuf = pos_idx[perm]

        model.train()
        epoch_loss = 0.0

        steps = max(1, math.ceil(pos_shuf.numel() / args.batch_size))
        iterator = range(0, pos_shuf.numel(), args.batch_size)
        if args.use_tqdm:
            iterator = tqdm(iterator, desc=f"Training UPRL Model Epoch {epoch + 1}/{num_epochs}", leave=False)

        for start in iterator:
            batch_pos_idx = pos_shuf[start : start + args.batch_size]
            if batch_pos_idx.numel() == 0:
                continue
            batch_size = batch_pos_idx.numel()

            num_neg = max(1, int(args.num_neg))
            j_idx = torch.randint(0, n, (batch_size, num_neg), device=device)
            same = j_idx == batch_pos_idx.view(-1, 1)
            if same.any():
                j_idx = torch.where(same, (j_idx + 1) % n, j_idx)

            X_i = X_train_full[batch_pos_idx]
            X_j = X_train_full[j_idx.reshape(-1)]

            # Pair branch scores
            s_pair_i = model.forward_pair(X_i).squeeze(-1).view(batch_size, 1)
            s_pair_j = model.forward_pair(X_j).squeeze(-1).view(batch_size, num_neg)
            base = -F.logsigmoid(s_pair_i - s_pair_j)  # positive

            # Point branch predictions on j (detached in the correction term)
            s_point_j = model.forward_point(X_j).squeeze(-1).view(batch_size, num_neg)
            q = torch.sigmoid(s_point_j).detach()

            flat_j = j_idx.reshape(-1)
            y_j = y_train_full[flat_j].float().view(batch_size, num_neg)

            pi_i = torch.clamp(propensity_train[batch_pos_idx].float(), args.clip_min, 1.0).view(batch_size, 1)
            pi_j = torch.clamp(propensity_train[flat_j].float(), args.clip_min, 1.0).view(batch_size, num_neg)

            # Original UPRL pairwise weight: (1/pi_i) * (1 - y_j) / pi_j
            w = (1.0 / pi_i) * (1.0 - y_j) / pi_j

            numerator = pi_j * (1.0 - q)
            denominator = 1.0 - q * pi_j + eps
            corr = numerator / denominator

            pair_loss = torch.mean(torch.clamp(corr * w * base, 0.0, 1e2))

            # Pointwise IPS-like term (original formula structure, with clipping)
            p = torch.sigmoid(s_point_j)
            y_over_p = y_j / pi_j
            coeff_neg = torch.clamp(1.0 - y_over_p, -1e8, 1e8)
            ll = y_over_p * torch.log(p + eps) + coeff_neg * torch.log(1.0 - p + eps)
            point_ce = -torch.mean(torch.clamp(ll, -1e8, 1e8))

            loss = pair_loss + point_ce
            weighted_loss = args.w_reg * loss

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/steps:.5f}, Val loss: {val_loss:.5f}"
            )

        monitor_loss = epoch_loss / steps if args.monitor_on == "train" else val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
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

        if propensity:
            criterion_mean = nn.MSELoss()
            loss = criterion_mean(torch.sigmoid(outputs), mask.float())
        else:
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

    print("=" * 70)
    print("UPRL Reward Modeling (Global Pairwise)")
    print("=" * 70)
    print("Loading embeddings and labels from Safetensors file...")
    if args.binary:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, mask_val, X_test, y_test = load_data(
            embedding_file,
            device,
            keys=[
                "X_train",
                "y_train_binary",
                "mask_train",
                "X_val",
                "y_val_binary",
                "mask_val",
                "X_test",
                "y_test_binary",
            ],
        )
    else:
        embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}.safetensors"
        X_train_full, y_train_full, mask_train, X_val_full, y_val_full, mask_val, X_test, y_test = load_data(
            embedding_file,
            device,
            keys=["X_train", "y_train", "mask_train", "X_val", "y_val", "mask_val", "X_test", "y_test"],
        )

    print(f"Training on {X_train_full.shape[0]} samples (global pairwise sampling).")
    print(f"Validating on {X_val_full.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    propensity_train_np = calculate_propensity(y_train_full.detach().cpu().numpy(), args.alpha)
    propensity_train = torch.from_numpy(propensity_train_np).to(device=device, dtype=torch.float32)

    val_data = (X_val_full, y_val_full, mask_val.float())
    test_data = (X_test, y_test, torch.ones_like(y_test))

    model = Model(X_train_full.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
        train_data=(X_train_full, y_train_full, propensity_train),
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        args=args,
    )
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    model.eval()

    with torch.no_grad():
        def get_preds(X, y, mask):
            reward_pred = torch.sigmoid(model(X).squeeze()) if args.binary else model(X).squeeze()
            reward_pred = reward_pred.detach().cpu().numpy()
            y_cpu = y.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            return reward_pred, y_cpu, mask_cpu

        y_train_pred, y_train_cpu, mask_train_cpu = get_preds(X_train_full, y_train_full, mask_train.float())
        y_val_pred, y_val_cpu, mask_val_cpu = get_preds(*val_data)
        y_test_pred, y_test_cpu, _ = get_preds(*test_data)

    obs_train = mask_train_cpu > 0.5
    obs_val = mask_val_cpu > 0.5

    metrics = {
        "R2 on train": r2_score(y_train_cpu[obs_train], y_train_pred[obs_train]) if obs_train.sum() > 0 else float("nan"),
        "R2 on val": r2_score(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "MAE on eval": mean_absolute_error(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "RMSE on eval": np.sqrt(mean_squared_error(y_val_cpu[obs_val], y_val_pred[obs_val])) if obs_val.sum() > 0 else float("nan"),
        "RMSE on test": np.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on eval": roc_auc_score(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),
        "Pearson on eval": pearsonr(y_val_cpu[obs_val], y_val_pred[obs_val])[0] if obs_val.sum() > 0 else float("nan"),
        "Pearson on test": pearsonr(y_test_cpu, y_test_pred)[0],
        "NLL on eval": compute_nll(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "NLL on test": compute_nll(y_test_cpu, y_test_pred),
        "NDCG on eval": compute_ndcg_binary(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "NDCG on test": compute_ndcg_binary(y_test_cpu, y_test_pred),
        "Recall on eval": compute_recall_binary(y_val_cpu[obs_val], y_val_pred[obs_val]) if obs_val.sum() > 0 else float("nan"),
        "Recall on test": compute_recall_binary(y_test_cpu, y_test_pred),
    }
    metrics = refine_dict(metrics)
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == "__main__":
    main()

