import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from collections import defaultdict

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
    add_tuned_recall_metrics,
)


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
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(",")))
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
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/ubpr/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "ubpr",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "lr": 0.0002,
        "clip_min": 1e-8,
        "num_epochs": 200,
        "batch_size": 512,  # number of positive-i per step
        "num_neg": 10,  # number of j samples per i
        "hidden_dim": "256,64",
        "patience": 20,
        "seed": 42,
        "l2_reg": 1e-6,
        "w_reg": 1.0,
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
            "l2_reg": 1e-6,
            "w_reg": 1.0,
        },
        "hs": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,
            "w_reg": 1.0,
        },
        "ufb": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
            "l2_reg": 1e-5,
            "w_reg": 0.2,
        },
    }
    merged_defaults = {**base_defaults, **dataset_defaults.get(pre_args.data_name, {})}

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
    parser.add_argument("--batch_size", type=int, help="Number of positive samples per step")
    parser.add_argument("--num_neg", type=int, help="Number of j samples per i")
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient")
    parser.add_argument("--w_reg", type=float, help="Task weight")
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or val loss")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary labels (must be True for UBPR)")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")

    parser.set_defaults(**merged_defaults)
    return parser.parse_args()


def _build_user_to_indices(user_id_np: np.ndarray):
    user_to_indices = defaultdict(list)
    for idx, uid in enumerate(user_id_np.tolist()):
        user_to_indices[int(uid)].append(idx)
    return {uid: np.asarray(idxs, dtype=np.int64) for uid, idxs in user_to_indices.items()}


def _sample_other_user_uniform(n: int, user_id_np: np.ndarray, uid: int, k: int, rng):
    if n <= 1 or k <= 0:
        return np.empty((0,), dtype=np.int64)
    out = []
    rounds = 0
    while len(out) < k and rounds < 50:
        need = k - len(out)
        cand = rng.integers(0, n, size=max(need * 3, 8), endpoint=False, dtype=np.int64)
        cand = cand[user_id_np[cand] != uid]
        out.extend(cand.tolist())
        rounds += 1
    if len(out) < k:
        cand = rng.integers(0, n, size=k - len(out), endpoint=False, dtype=np.int64)
        out.extend(cand.tolist())
    return np.asarray(out[:k], dtype=np.int64)


def _precompute_pairs_ubpr(user_id_np, y_np, num_neg, seed):
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_np > 0.5)[0].astype(np.int64)
    user_to_indices = _build_user_to_indices(user_id_np)

    js_per_pos = []
    n_within_pairs = 0
    n_fallback_pairs = 0
    n_fallback_pos = 0

    n = int(y_np.shape[0])
    for i in pos_idx:
        uid = int(user_id_np[i])
        group = user_to_indices.get(uid, np.asarray([i], dtype=np.int64))
        cand = group[group != i]
        if cand.size > 0:
            k = min(int(num_neg), int(cand.size))
            js = rng.choice(cand, size=k, replace=False).astype(np.int64)
            n_within_pairs += int(js.size)
        else:
            n_fallback_pos += 1
            js = _sample_other_user_uniform(n, user_id_np, uid, int(num_neg), rng)
            n_fallback_pairs += int(js.size)
        js_per_pos.append(js)

    stats = {
        "n_pos": int(pos_idx.size),
        "n_pairs_within_user": int(n_within_pairs),
        "n_pairs_fallback_other_user": int(n_fallback_pairs),
        "n_pos_fallback_other_user": int(n_fallback_pos),
        "fallback_pair_ratio": float(n_fallback_pairs / max(1, (n_within_pairs + n_fallback_pairs))),
    }
    return pos_idx, js_per_pos, stats


def train(model, train_data, optimizer, num_epochs, val_data, patience, args):
    """
    User-wise UBPR (binary only):
      - i sampled from positives (y=1)
      - j sampled from same user (all samples, y in {0,1})
      - if a user has no within-user j candidates, fallback: sample j from other users (no explicit switch)

    Loss (same structure as old MF UBPR baseline):
      base = -log σ(s_i - s_j)
      w = (1/pi_i) * (1 - y_j/pi_j)
      loss = mean(w * base)
    """
    if not args.is_training:
        return
    if not args.binary:
        raise NotImplementedError("UBPR baseline currently supports binary (0/1) labels only.")

    X_train_full, y_train_full, user_id_train, propensity_train = train_data
    device = X_train_full.device

    y_np = y_train_full.detach().cpu().numpy()
    user_id_np = user_id_train.detach().cpu().numpy()

    pos_idx_np, js_per_pos, pair_stats = _precompute_pairs_ubpr(
        user_id_np=user_id_np,
        y_np=y_np,
        num_neg=max(1, int(args.num_neg)),
        seed=args.seed,
    )
    if pos_idx_np.size == 0:
        print("Skip training: no positive samples.")
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
        return

    total_pairs = pair_stats["n_pairs_within_user"] + pair_stats["n_pairs_fallback_other_user"]
    if total_pairs == 0:
        print(f"Skip training: no valid pairs. stats={pair_stats}")
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
        return

    print(f"[UBPR] Pair stats: {pair_stats}")

    best_loss = float("inf")
    patience_counter = 0

    n_pos = int(pos_idx_np.size)
    steps = max(1, math.ceil(n_pos / int(args.batch_size)))

    for epoch in range(num_epochs):
        perm = torch.randperm(n_pos)
        pos_order = perm.detach().cpu().numpy()

        model.train()
        epoch_loss = 0.0

        iterator = range(0, n_pos, int(args.batch_size))
        if args.use_tqdm:
            iterator = tqdm(iterator, desc=f"Training UBPR Model Epoch {epoch + 1}/{num_epochs}", leave=False)

        for start in iterator:
            batch_pos_pos = pos_order[start : start + int(args.batch_size)]
            if batch_pos_pos.size == 0:
                continue

            i_list = []
            j_list = []
            for p in batch_pos_pos.tolist():
                i = pos_idx_np[p]
                js = js_per_pos[p]
                if js.size == 0:
                    continue
                i_list.append(np.full((js.size,), i, dtype=np.int64))
                j_list.append(js)
            if not i_list:
                continue

            i_idx = torch.from_numpy(np.concatenate(i_list)).to(device=device, dtype=torch.long)
            j_idx = torch.from_numpy(np.concatenate(j_list)).to(device=device, dtype=torch.long)

            X_i = X_train_full[i_idx]
            X_j = X_train_full[j_idx]

            s_i = model(X_i).squeeze(-1)
            s_j = model(X_j).squeeze(-1)

            diff = s_i - s_j
            base = -F.logsigmoid(diff)

            pi_i = torch.clamp(propensity_train[i_idx].float(), args.clip_min, 1.0)
            pi_j = torch.clamp(propensity_train[j_idx].float(), args.clip_min, 1.0)
            y_j = y_train_full[j_idx].float()

            w = (1.0 / pi_i) * (1.0 - (y_j / pi_j))
            loss = torch.mean(w * base)

            weighted_loss = args.w_reg * loss
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/steps:.5f}, Val loss: {val_loss:.5f}")

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
    print("UBPR Reward Modeling (User-wise Pairwise + Fallback Other-User)")
    print("=" * 70)
    print("Loading embeddings and labels from Safetensors file...")

    if not args.binary:
        raise NotImplementedError("UBPR baseline currently supports --binary True only.")

    embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
    try:
        (
            X_train_full,
            y_train_full,
            mask_train,
            user_id_train,
            X_val_full,
            y_val_full,
            y_val_true,
            mask_val,
            X_test,
            y_test,
        ) = load_data(
            embedding_file,
            device,
            keys=[
                "X_train",
                "y_train_binary",
                "mask_train",
                "user_id_train",
                "X_val",
                "y_val_binary",
                "y_val_binary_true",
                "mask_val",
                "X_test",
                "y_test_binary",
            ],
        )
    except KeyError as e:
        raise KeyError(
            f"{embedding_file} is missing `user_id_train`. Please rerun Stage 1/2 to generate user_id fields."
        ) from e

    print(f"Training on {X_train_full.shape[0]} samples (user-wise pairwise).")
    print(f"Validating on {X_val_full.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    propensity_train_np = calculate_propensity(y_train_full.detach().cpu().numpy(), args.alpha)
    propensity_train = torch.from_numpy(propensity_train_np).to(device=device, dtype=torch.float32)

    val_data = (X_val_full, y_val_full, mask_val.float())
    val_data_true = (X_val_full, y_val_true, mask_val.float())
    test_data = (X_test, y_test, torch.ones_like(y_test))

    model = Model(X_train_full.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
        train_data=(X_train_full, y_train_full, user_id_train, propensity_train),
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        val_data=val_data_true,
        patience=args.patience,
        args=args,
    )
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    model.eval()

    with torch.no_grad():
        def get_preds(X, y, mask):
            reward_pred = torch.sigmoid(model(X).squeeze())
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
    metrics = refine_dict(metrics)
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == "__main__":
    main()
