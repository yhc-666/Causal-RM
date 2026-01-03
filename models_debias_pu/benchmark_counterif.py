import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from argparse import ArgumentParser
from safetensors.torch import load_file
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tools.utils import (
    seed_everything,
    str2bool,
    save_metrics,
    refine_dict,
    compute_nll,
    compute_ndcg_binary,
    compute_recall_binary,
    add_tuned_recall_metrics,
)


@dataclass(frozen=True)
class Groups:
    # Match paper naming: DP / HE / HU / UN
    DP: int = 0
    HE: int = 1
    HU: int = 2
    UN: int = 3


GROUPS = Groups()


def _safe_percentile(a: np.ndarray, q: float, default: float) -> float:
    a = np.asarray(a)
    if a.size == 0:
        return float(default)
    return float(np.percentile(a, q))


def _impute_counterif_groups_ocsvm(
    X_np: np.ndarray,
    y_binary_np: np.ndarray,
    *,
    target_percentile: float,
    hn_percentile: float,
    batch_size: int,
    seed: int,
):
    """
    Strictly follows old/counterIF Ours.assign_labels() rule:
      - y==1 => (R=1, O=1) => DP
      - y==0 and ocsvm_distance > 0 => (R=1, O=1) => HE
      - y==0 and ocsvm_distance <= percentile(y==0 distances, hn_percentile) => (R=1, O=0) => HU
      - else => (R=0, O=-1) => UN

    Implementation detail kept from old code: fit OneClassSVM per batch on the positive
    samples within that batch (rbf kernel, nu = 1 - target_percentile/100).
    """
    n = int(y_binary_np.shape[0])
    if X_np.shape[0] != n:
        raise ValueError(f"X/y size mismatch: X={X_np.shape[0]}, y={n}")

    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)

    r_labels = np.zeros(n, dtype=np.int64)
    o_labels = np.full(n, -1, dtype=np.int64)
    group = np.full(n, GROUPS.UN, dtype=np.int64)

    nu = 1.0 - float(target_percentile) / 100.0
    nu = float(np.clip(nu, 1e-4, 0.9999))

    if batch_size <= 0:
        batch_size = n

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = order[start:end]
        Xb = X_np[idx]
        yb = y_binary_np[idx].astype(np.int64)

        pos_mask = yb == 1
        if int(pos_mask.sum()) == 0:
            # No positives in this chunk -> cannot fit OCSVM; default all negatives to UN.
            continue

        ocsvm = OneClassSVM(kernel="rbf", nu=nu)
        ocsvm.fit(Xb[pos_mask])
        distances = ocsvm.decision_function(Xb)

        neg_mask = ~pos_mask
        threshold = _safe_percentile(distances[neg_mask], hn_percentile, default=-np.inf)

        # DP
        dp = pos_mask
        r_labels[idx[dp]] = 1
        o_labels[idx[dp]] = 1
        group[idx[dp]] = GROUPS.DP

        # For y==0: HE / HU / UN by distance
        he = neg_mask & (distances > 0)
        hu = neg_mask & (~he) & (distances <= threshold)
        un = neg_mask & (~he) & (~hu)

        r_labels[idx[he]] = 1
        o_labels[idx[he]] = 1
        group[idx[he]] = GROUPS.HE

        r_labels[idx[hu]] = 1
        o_labels[idx[hu]] = 0
        group[idx[hu]] = GROUPS.HU

        r_labels[idx[un]] = 0
        o_labels[idx[un]] = -1
        group[idx[un]] = GROUPS.UN

    stats = {
        "n_total": int(n),
        "n_dp": int(np.sum(group == GROUPS.DP)),
        "n_he": int(np.sum(group == GROUPS.HE)),
        "n_hu": int(np.sum(group == GROUPS.HU)),
        "n_un": int(np.sum(group == GROUPS.UN)),
        "nu": float(nu),
    }
    return group, r_labels, o_labels, stats


def _build_user_to_indices(user_id_np: np.ndarray):
    user_to_indices = defaultdict(list)
    for idx, uid in enumerate(user_id_np.tolist()):
        user_to_indices[int(uid)].append(int(idx))
    return {uid: np.asarray(idxs, dtype=np.int64) for uid, idxs in user_to_indices.items()}


def _make_userwise_pairs(pos_idx: np.ndarray, neg_idx: np.ndarray, user_id_np: np.ndarray | None, max_pairs: int | None, seed: int):
    pos_idx = np.asarray(pos_idx, dtype=np.int64)
    neg_idx = np.asarray(neg_idx, dtype=np.int64)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.empty((0, 2), dtype=np.int64), {
            "n_pos": int(pos_idx.size),
            "n_pairs_within_user": 0,
            "n_pairs_fallback_other_user": 0,
            "n_pos_fallback_other_user": 0,
            "fallback_pair_ratio": 0.0,
        }

    rng = np.random.default_rng(seed)

    if user_id_np is None:
        # global pairing
        if max_pairs is not None:
            pos_idx = pos_idx[: int(max_pairs)]
        chosen_neg = rng.choice(neg_idx, size=int(pos_idx.size), replace=True).astype(np.int64)
        pairs = np.stack([pos_idx, chosen_neg], axis=1)
        stats = {
            "n_pos": int(pos_idx.size),
            "n_pairs_within_user": 0,
            "n_pairs_fallback_other_user": int(pos_idx.size),
            "n_pos_fallback_other_user": int(pos_idx.size),
            "fallback_pair_ratio": 1.0,
        }
        return pairs, stats

    # Build neg candidates per user (global indices)
    user_to_neg_global = _build_user_to_indices(user_id_np[neg_idx])
    for uid, local in user_to_neg_global.items():
        user_to_neg_global[uid] = neg_idx[local]
    neg_users = user_id_np[neg_idx]

    pairs = []
    n_within = 0
    n_fallback = 0
    n_fallback_pos = 0

    rng.shuffle(pos_idx)
    for i in pos_idx:
        uid = int(user_id_np[i])
        cand = user_to_neg_global.get(uid)
        if cand is not None and cand.size > 0:
            j = int(rng.choice(cand, size=1, replace=False)[0])
            n_within += 1
        else:
            n_fallback_pos += 1
            other = neg_idx[neg_users != uid]
            if other.size > 0:
                j = int(rng.choice(other, size=1, replace=False)[0])
            else:
                j = int(rng.choice(neg_idx, size=1, replace=False)[0])
            n_fallback += 1
        pairs.append((int(i), int(j)))
        if max_pairs is not None and len(pairs) >= int(max_pairs):
            break

    pairs = np.asarray(pairs, dtype=np.int64)
    stats = {
        "n_pos": int(pairs.shape[0]),
        "n_pairs_within_user": int(n_within),
        "n_pairs_fallback_other_user": int(n_fallback),
        "n_pos_fallback_other_user": int(n_fallback_pos),
        "fallback_pair_ratio": float(n_fallback / max(1, (n_within + n_fallback))),
    }
    return pairs, stats


def _wasserstein_ipm(x: torch.Tensor, t: torch.Tensor, p: float, lam: float, its: int, sq: bool, backpropT: bool):
    """
    Ported from old/counterIF/code/src/models/recommenders.py (TensorFlow) to PyTorch.
    Computes an entropic-regularized Wasserstein distance as an IPM term.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [n,d], got {tuple(x.shape)}")
    if t.ndim != 1:
        t = t.reshape(-1)
    it = torch.where(t == 1)[0]
    ic = torch.where(t == 0)[0]
    if it.numel() == 0 or ic.numel() == 0:
        return x.new_tensor(0.0)

    xt = x[it]
    xc = x[ic]
    nt = float(xt.shape[0])
    nc = float(xc.shape[0])

    # Pairwise distance matrix
    xt2 = (xt * xt).sum(dim=1, keepdim=True)  # [nt,1]
    xc2 = (xc * xc).sum(dim=1, keepdim=True).transpose(0, 1)  # [1,nc]
    dist2 = torch.clamp(xt2 + xc2 - 2.0 * (xt @ xc.transpose(0, 1)), min=0.0)
    if sq:
        m = dist2
    else:
        m = torch.sqrt(torch.clamp(dist2, min=1e-10))

    m_mean = m.mean()
    delta = m.max().detach()
    eff_lam = (float(lam) / (m_mean + 1e-12)).detach()

    # Pad Mt with an extra row/col of delta; bottom-right set to 0
    mt = F.pad(m, (0, 1, 0, 1), value=float(delta))
    mt[-1, -1] = 0.0

    # Marginals
    a = torch.cat(
        [
            x.new_full((int(nt), 1), float(p) / max(1.0, nt)),
            x.new_full((1, 1), 1.0 - float(p)),
        ],
        dim=0,
    )
    b = torch.cat(
        [
            x.new_full((int(nc), 1), (1.0 - float(p)) / max(1.0, nc)),
            x.new_full((1, 1), float(p)),
        ],
        dim=0,
    )

    mlam = eff_lam * mt
    k = torch.exp(-mlam) + 1e-6
    ainvk = k / (a + 1e-12)

    u = a
    for _ in range(int(its)):
        denom = (u.transpose(0, 1) @ k).transpose(0, 1)
        u = 1.0 / (ainvk @ (b / (denom + 1e-12)) + 1e-12)
    v = b / ((u.transpose(0, 1) @ k).transpose(0, 1) + 1e-12)

    tplan = u * (v.transpose(0, 1) * k)
    if not backpropT:
        tplan = tplan.detach()

    e = tplan * mt
    d = 2.0 * e.sum()
    return d


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_dim_str: str):
        super().__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(",")))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor, *, return_rep: bool = False):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        rep = x
        logits = self.output_layer(rep)
        if return_rep:
            return logits, rep
        return logits


def parse_arguments():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_args, _ = pre_parser.parse_known_args()

    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/counterif/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "counterif",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "lr": 5e-4,
        "num_epochs": 200,
        "batch_size_point": 512,
        "batch_size_pair": 1024,
        "batch_size_ipm": 256,
        "hidden_dim": "256,64",
        "patience": 20,
        "seed": 42,
        "l2_reg": 1e-6,
        "lambda_point": 1.0,
        "lambda_pair": 1.0,
        "lambda_ipm": 1.0,
        "ipm_lam": 10.0,
        "ipm_its": 10,
        "ipm_p": 0.5,
        "target_percentile": 90.0,
        "hn_percentile": 10.0,
        "ocsvm_batch_size": 8192,
        "pair_max_dp_he": None,
        "pair_max_un_he": 20000,
        "pair_max_hu_un": 20000,
        "rerun": False,
        "monitor_on": "val",
        "binary": True,
        "use_tqdm": True,
    }

    dataset_defaults = {
        "hs": {"alpha": 0.2, "lr": 5e-4},
        "saferlhf": {"alpha": 0.2, "lr": 5e-4},
        "ufb": {"alpha": 0.2, "lr": 5e-4},
    }
    merged_defaults = {**base_defaults, **dataset_defaults.get(pre_args.data_name, {})}

    parser = ArgumentParser(description="")
    parser.add_argument("--desc", type=str, default="foo")
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size_point", type=int)
    parser.add_argument("--batch_size_pair", type=int)
    parser.add_argument("--batch_size_ipm", type=int)
    parser.add_argument("--hidden_dim", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--lambda_point", type=float)
    parser.add_argument("--lambda_pair", type=float)
    parser.add_argument("--lambda_ipm", type=float)
    parser.add_argument("--ipm_lam", type=float)
    parser.add_argument("--ipm_its", type=int)
    parser.add_argument("--ipm_p", type=float)
    parser.add_argument("--target_percentile", type=float)
    parser.add_argument("--hn_percentile", type=float)
    parser.add_argument("--ocsvm_batch_size", type=int)
    parser.add_argument("--pair_max_dp_he", type=int)
    parser.add_argument("--pair_max_un_he", type=int)
    parser.add_argument("--pair_max_hu_un", type=int)
    parser.add_argument("--rerun", type=str2bool)
    parser.add_argument("--monitor_on", type=str)
    parser.add_argument("--binary", type=str2bool)
    parser.add_argument("--use_tqdm", type=str2bool)

    parser.set_defaults(**merged_defaults)
    return parser.parse_args()


def _bpr_pair_loss(logits_pos: torch.Tensor, logits_neg: torch.Tensor):
    # -log σ(pos - neg) = softplus(-(pos-neg))
    return F.softplus(-(logits_pos - logits_neg)).mean()


def _compute_total_loss(
    model: Model,
    X: torch.Tensor,
    y_binary: torch.Tensor,
    group_np: np.ndarray,
    r_np: np.ndarray,
    o_np: np.ndarray,
    user_id_np: np.ndarray | None,
    pairs_dp_he: np.ndarray,
    pairs_un_he: np.ndarray,
    pairs_hu_un: np.ndarray,
    args,
    device: torch.device,
):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    eps = 1e-12

    group = torch.from_numpy(group_np).to(device=device, dtype=torch.long)
    r = torch.from_numpy(r_np).to(device=device, dtype=torch.long)
    o = torch.from_numpy(o_np).to(device=device, dtype=torch.long)

    with torch.no_grad():
        # Pointwise: DP=1, HE=0
        mask_point = (group == GROUPS.DP) | (group == GROUPS.HE)
        if int(mask_point.sum()) > 0:
            logits, rep = model(X[mask_point], return_rep=True)
            labels = (group[mask_point] == GROUPS.DP).float()
            loss_point = bce(logits.squeeze(), labels)
        else:
            loss_point = X.new_tensor(0.0)

        # Pairwise: three strata-ranking terms
        loss_pair = X.new_tensor(0.0)
        for pairs in (pairs_dp_he, pairs_un_he, pairs_hu_un):
            if pairs.size == 0:
                continue
            idx_pos = torch.from_numpy(pairs[:, 0]).to(device=device, dtype=torch.long)
            idx_neg = torch.from_numpy(pairs[:, 1]).to(device=device, dtype=torch.long)
            logits_pos = model(X[idx_pos]).squeeze()
            logits_neg = model(X[idx_neg]).squeeze()
            loss_pair = loss_pair + _bpr_pair_loss(logits_pos, logits_neg)

        # IPM/Wasserstein
        loss_ipm = X.new_tensor(0.0)
        if args.batch_size_ipm > 0:
            n = int(X.shape[0])
            k = min(int(args.batch_size_ipm), n)
            idx = torch.randint(0, n, (k,), device=device)
            _, rep = model(X[idx], return_rep=True)

            r_b = r[idx]
            # IPM(R=0 vs R=1)
            loss_ipm = loss_ipm + _wasserstein_ipm(
                rep,
                r_b,
                p=float(args.ipm_p),
                lam=float(args.ipm_lam),
                its=int(args.ipm_its),
                sq=False,
                backpropT=False,
            )

            # IPM(R=1,O=0 vs R=1,O=1) within R=1 only
            r1_mask = r_b == 1
            if int(r1_mask.sum()) > 0:
                rep_r1 = rep[r1_mask]
                o_r1 = o[idx][r1_mask]
                # within R=1, O is 0 or 1 (HU vs DP/HE)
                t_o = (o_r1 == 1).long()
                loss_ipm = loss_ipm + _wasserstein_ipm(
                    rep_r1,
                    t_o,
                    p=float(args.ipm_p),
                    lam=float(args.ipm_lam),
                    its=int(args.ipm_its),
                    sq=False,
                    backpropT=False,
                )

        total = (
            float(args.lambda_point) * loss_point
            + float(args.lambda_pair) * loss_pair
            + float(args.lambda_ipm) * loss_ipm
        )
        return total.item()


def train(
    model: Model,
    X_train: torch.Tensor,
    y_train_binary: torch.Tensor,
    train_group: np.ndarray,
    train_r: np.ndarray,
    train_o: np.ndarray,
    user_id_train_np: np.ndarray | None,
    pairs_train_dp_he: np.ndarray,
    pairs_train_un_he: np.ndarray,
    pairs_train_hu_un: np.ndarray,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    X_val: torch.Tensor,
    y_val_binary: torch.Tensor,
    y_val_binary_true: torch.Tensor,
    val_group: np.ndarray,
    val_r: np.ndarray,
    val_o: np.ndarray,
    user_id_val_np: np.ndarray | None,
    pairs_val_dp_he: np.ndarray,
    pairs_val_un_he: np.ndarray,
    pairs_val_hu_un: np.ndarray,
    patience: int,
    args,
    device: torch.device,
):
    if not args.is_training:
        return

    best_loss = float("inf")
    patience_counter = 0

    # Pointwise dataset: DP/HE only
    train_group_t = torch.from_numpy(train_group).long()
    idx_point = torch.where((train_group_t == GROUPS.DP) | (train_group_t == GROUPS.HE))[0]
    y_point = (train_group_t[idx_point] == GROUPS.DP).float()
    point_loader = DataLoader(
        TensorDataset(idx_point, y_point),
        batch_size=max(1, int(args.batch_size_point)),
        shuffle=True,
    )

    # Pairwise datasets (indices only)
    def _pair_loader(pairs: np.ndarray):
        if pairs.size == 0:
            return None
        t = torch.from_numpy(pairs).long()
        return DataLoader(TensorDataset(t[:, 0], t[:, 1]), batch_size=max(1, int(args.batch_size_pair)), shuffle=True)

    pair_loader_dp_he = _pair_loader(pairs_train_dp_he)
    pair_loader_un_he = _pair_loader(pairs_train_un_he)
    pair_loader_hu_un = _pair_loader(pairs_train_hu_un)

    # IPM batches sampled from full train set (indices only)
    full_idx = torch.arange(int(X_train.shape[0]), dtype=torch.long)
    full_loader = DataLoader(TensorDataset(full_idx), batch_size=max(1, int(args.batch_size_ipm)), shuffle=True)

    bce = nn.BCEWithLogitsLoss()

    def _cycle(loader):
        if loader is None:
            while True:
                yield None
        while True:
            for batch in loader:
                yield batch

    it_pair_dp_he = _cycle(pair_loader_dp_he)
    it_pair_un_he = _cycle(pair_loader_un_he)
    it_pair_hu_un = _cycle(pair_loader_hu_un)
    it_full = _cycle(full_loader)

    train_r_t = torch.from_numpy(train_r).to(device=device, dtype=torch.long)
    train_o_t = torch.from_numpy(train_o).to(device=device, dtype=torch.long)

    for epoch in range(int(num_epochs)):
        model.train()
        epoch_loss = 0.0

        bar = tqdm(point_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else point_loader
        for (idx_p, y_p) in bar:
            idx_p = idx_p.to(device=device)
            y_p = y_p.to(device=device)

            optimizer.zero_grad()

            # Pointwise loss on DP/HE
            logits_p, _ = model(X_train[idx_p], return_rep=True)
            loss_point = bce(logits_p.squeeze(), y_p)

            # Pairwise losses (three terms)
            loss_pair = X_train.new_tensor(0.0)
            for it_pair in (it_pair_dp_he, it_pair_un_he, it_pair_hu_un):
                batch = next(it_pair)
                if batch is None:
                    continue
                pos_idx, neg_idx = batch
                pos_idx = pos_idx.to(device=device)
                neg_idx = neg_idx.to(device=device)
                logits_pos = model(X_train[pos_idx]).squeeze()
                logits_neg = model(X_train[neg_idx]).squeeze()
                loss_pair = loss_pair + _bpr_pair_loss(logits_pos, logits_neg)

            # IPM/Wasserstein on a random full batch
            loss_ipm = X_train.new_tensor(0.0)
            batch_full = next(it_full)
            if batch_full is not None:
                (idx_full,) = batch_full
                idx_full = idx_full.to(device=device)
                _, rep_full = model(X_train[idx_full], return_rep=True)

                r_b = train_r_t[idx_full]
                loss_ipm = loss_ipm + _wasserstein_ipm(
                    rep_full,
                    r_b,
                    p=float(args.ipm_p),
                    lam=float(args.ipm_lam),
                    its=int(args.ipm_its),
                    sq=False,
                    backpropT=False,
                )

                r1_mask = r_b == 1
                if int(r1_mask.sum()) > 0:
                    rep_r1 = rep_full[r1_mask]
                    o_r1 = train_o_t[idx_full][r1_mask]
                    t_o = (o_r1 == 1).long()
                    loss_ipm = loss_ipm + _wasserstein_ipm(
                        rep_r1,
                        t_o,
                        p=float(args.ipm_p),
                        lam=float(args.ipm_lam),
                        its=int(args.ipm_its),
                        sq=False,
                        backpropT=False,
                    )

            loss = (
                float(args.lambda_point) * loss_point
                + float(args.lambda_pair) * loss_pair
                + float(args.lambda_ipm) * loss_ipm
            )

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        # Validation loss: oracle clean binary labels + BCE (for monitoring/early-stopping only).
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze(-1)
            val_loss = float(bce(val_logits, y_val_binary_true.float()).item())
        model.train()

        if epoch % 4 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/len(point_loader):.5f}, Val loss: {val_loss:.5f}")

        monitor_loss = epoch_loss / max(1, len(point_loader)) if args.monitor_on == "train" else val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= int(patience):
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def main():
    args = parse_arguments()

    if args.is_training and os.path.exists(f"{args.output_dir}/performance.yaml") and not args.rerun:
        print(f"The path {args.output_dir}/performance.yaml exists!!")
        sys.exit(0)
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 70)
    print("Counter-IF Reward Modeling (Debias+PU) (mask-blind PU: UNK->0)")
    print("=" * 70)

    if not args.binary:
        raise ValueError("Counter-IF benchmark currently supports --binary True only.")

    embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
    data = load_file(embedding_file)

    def _get(key: str, dtype: torch.dtype):
        if key not in data:
            return None
        return data[key].to(device=device, dtype=dtype)

    X_train = _get("X_train", torch.float32)
    y_train_binary = _get("y_train_binary", torch.float32)
    X_val = _get("X_val", torch.float32)
    y_val_binary = _get("y_val_binary", torch.float32)
    y_val_binary_true = _get("y_val_binary_true", torch.float32)
    X_test = _get("X_test", torch.float32)
    y_test_binary = _get("y_test_binary", torch.float32)

    user_id_train = _get("user_id_train", torch.long)
    user_id_val = _get("user_id_val", torch.long)
    user_id_test = _get("user_id_test", torch.long)

    if any(v is None for v in (X_train, y_train_binary, X_val, y_val_binary, y_val_binary_true, X_test, y_test_binary)):
        missing = [
            k
            for k in (
                "X_train",
                "y_train_binary",
                "X_val",
                "y_val_binary",
                "y_val_binary_true",
                "X_test",
                "y_test_binary",
            )
            if k not in data
        ]
        raise KeyError(f"Missing required keys in {embedding_file}: {missing}")

    print(f"Training on {X_train.shape[0]} samples.")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    # Group imputation (cache by output_dir + hyperparams)
    cache_path = os.path.join(args.output_dir, "counterif_groups_cache.npz")
    cache_ok = False
    # Group cache is independent from training reruns; reuse when hyperparams match.
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if (
                float(cache["target_percentile"]) == float(args.target_percentile)
                and float(cache["hn_percentile"]) == float(args.hn_percentile)
                and int(cache["ocsvm_batch_size"]) == int(args.ocsvm_batch_size)
            ):
                train_group = cache["train_group"]
                train_r = cache["train_r"]
                train_o = cache["train_o"]
                val_group = cache["val_group"]
                val_r = cache["val_r"]
                val_o = cache["val_o"]
                cache_ok = True
        except Exception:
            cache_ok = False

    if not cache_ok:
        print("\n" + "=" * 70)
        print("Step 0: Missing-treatment imputation + stratification (OneClassSVM)")
        print("=" * 70)
        X_train_np = X_train.detach().cpu().numpy()
        y_train_np = y_train_binary.detach().cpu().numpy().astype(np.int64)
        train_group, train_r, train_o, train_stats = _impute_counterif_groups_ocsvm(
            X_train_np,
            y_train_np,
            target_percentile=float(args.target_percentile),
            hn_percentile=float(args.hn_percentile),
            batch_size=int(args.ocsvm_batch_size),
            seed=int(args.seed),
        )
        print(f"[Train] group stats: {train_stats}")

        X_val_np = X_val.detach().cpu().numpy()
        y_val_np = y_val_binary.detach().cpu().numpy().astype(np.int64)
        val_group, val_r, val_o, val_stats = _impute_counterif_groups_ocsvm(
            X_val_np,
            y_val_np,
            target_percentile=float(args.target_percentile),
            hn_percentile=float(args.hn_percentile),
            batch_size=int(args.ocsvm_batch_size),
            seed=int(args.seed) + 1,
        )
        print(f"[Val] group stats: {val_stats}")

        np.savez(
            cache_path,
            target_percentile=float(args.target_percentile),
            hn_percentile=float(args.hn_percentile),
            ocsvm_batch_size=int(args.ocsvm_batch_size),
            train_group=train_group,
            train_r=train_r,
            train_o=train_o,
            val_group=val_group,
            val_r=val_r,
            val_o=val_o,
        )

    # Pairwise datasets (user-wise when user_id exists; else global)
    user_id_train_np = user_id_train.detach().cpu().numpy() if user_id_train is not None else None
    user_id_val_np = user_id_val.detach().cpu().numpy() if user_id_val is not None else None
    idx_dp_train = np.where(train_group == GROUPS.DP)[0]
    idx_he_train = np.where(train_group == GROUPS.HE)[0]
    idx_hu_train = np.where(train_group == GROUPS.HU)[0]
    idx_un_train = np.where(train_group == GROUPS.UN)[0]

    pairs_train_dp_he, stats_dp_he = _make_userwise_pairs(
        idx_dp_train, idx_he_train, user_id_train_np, args.pair_max_dp_he, seed=args.seed
    )
    pairs_train_un_he, stats_un_he = _make_userwise_pairs(
        idx_un_train, idx_he_train, user_id_train_np, args.pair_max_un_he, seed=args.seed + 1
    )
    pairs_train_hu_un, stats_hu_un = _make_userwise_pairs(
        idx_hu_train, idx_un_train, user_id_train_np, args.pair_max_hu_un, seed=args.seed + 2
    )

    idx_dp_val = np.where(val_group == GROUPS.DP)[0]
    idx_he_val = np.where(val_group == GROUPS.HE)[0]
    idx_hu_val = np.where(val_group == GROUPS.HU)[0]
    idx_un_val = np.where(val_group == GROUPS.UN)[0]
    pairs_val_dp_he, _ = _make_userwise_pairs(idx_dp_val, idx_he_val, user_id_val_np, args.pair_max_dp_he, seed=args.seed)
    pairs_val_un_he, _ = _make_userwise_pairs(idx_un_val, idx_he_val, user_id_val_np, args.pair_max_un_he, seed=args.seed + 1)
    pairs_val_hu_un, _ = _make_userwise_pairs(idx_hu_val, idx_un_val, user_id_val_np, args.pair_max_hu_un, seed=args.seed + 2)

    print("\n" + "=" * 70)
    print("Pair stats (train)")
    print("=" * 70)
    print(f"DP>HE: {stats_dp_he}")
    print(f"UN>HE: {stats_un_he}")
    print(f"HU>UN: {stats_hu_un}")

    # Train model
    print("\n" + "=" * 70)
    print("Training Counter-IF Model")
    print("=" * 70)
    print(
        f"Loss: L = λ_point*L_point(DP,HE) + λ_pair*L_pair + λ_ipm*L_ipm(Wasserstein)\n"
        f"  - λ_point={args.lambda_point}, λ_pair={args.lambda_pair}, λ_ipm={args.lambda_ipm}\n"
        f"  - ipm_lam={args.ipm_lam}, ipm_its={args.ipm_its}, ipm_p={args.ipm_p}\n"
        f"  - OCSVM: target_percentile={args.target_percentile}, hn_percentile={args.hn_percentile}, nu={1-args.target_percentile/100:.4f}"
    )

    model = Model(int(X_train.shape[1]), args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.l2_reg))

    train(
        model=model,
        X_train=X_train,
        y_train_binary=y_train_binary,
        train_group=train_group,
        train_r=train_r,
        train_o=train_o,
        user_id_train_np=user_id_train_np,
        pairs_train_dp_he=pairs_train_dp_he,
        pairs_train_un_he=pairs_train_un_he,
        pairs_train_hu_un=pairs_train_hu_un,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        X_val=X_val,
        y_val_binary=y_val_binary,
        y_val_binary_true=y_val_binary_true,
        val_group=val_group,
        val_r=val_r,
        val_o=val_o,
        user_id_val_np=user_id_val_np,
        pairs_val_dp_he=pairs_val_dp_he,
        pairs_val_un_he=pairs_val_un_he,
        pairs_val_hu_un=pairs_val_hu_un,
        patience=args.patience,
        args=args,
        device=device,
    )

    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        def preds(X):
            return torch.sigmoid(model(X).squeeze()).cpu().numpy()

        y_train_pred = preds(X_train)
        y_val_pred = preds(X_val)
        y_test_pred = preds(X_test)

        y_train_cpu = y_train_binary.cpu().numpy()
        y_val_cpu = y_val_binary.cpu().numpy()
        y_test_cpu = y_test_binary.cpu().numpy()

    metrics = {
        "R2 on train": r2_score(y_train_cpu, y_train_pred),
        "R2 on val": r2_score(y_val_cpu, y_val_pred),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "MAE on eval": mean_absolute_error(y_val_cpu, y_val_pred),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "RMSE on eval": math.sqrt(mean_squared_error(y_val_cpu, y_val_pred)),
        "RMSE on test": math.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on eval": roc_auc_score(y_val_cpu, y_val_pred),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),
        "Pearson on eval": pearsonr(y_val_cpu, y_val_pred)[0],
        "Pearson on test": pearsonr(y_test_cpu, y_test_pred)[0],
        "NLL on eval": compute_nll(y_val_cpu, y_val_pred),
        "NLL on test": compute_nll(y_test_cpu, y_test_pred),
        "NDCG on eval": compute_ndcg_binary(y_val_cpu, y_val_pred),
        "NDCG on test": compute_ndcg_binary(y_test_cpu, y_test_pred),
        "Group train n_dp": int(np.sum(train_group == GROUPS.DP)),
        "Group train n_he": int(np.sum(train_group == GROUPS.HE)),
        "Group train n_hu": int(np.sum(train_group == GROUPS.HU)),
        "Group train n_un": int(np.sum(train_group == GROUPS.UN)),
        "Group val n_dp": int(np.sum(val_group == GROUPS.DP)),
        "Group val n_he": int(np.sum(val_group == GROUPS.HE)),
        "Group val n_hu": int(np.sum(val_group == GROUPS.HU)),
        "Group val n_un": int(np.sum(val_group == GROUPS.UN)),
    }
    add_tuned_recall_metrics(metrics, y_val_cpu, y_val_pred, y_test_cpu, y_test_pred)
    metrics = refine_dict(metrics)
    print("\n--- Final Performance ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    save_metrics(args, metrics)


if __name__ == "__main__":
    main()
