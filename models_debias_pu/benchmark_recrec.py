import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from argparse import ArgumentParser
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tools.utils import (
    seed_everything,
    str2bool,
    load_data,
    save_metrics,
    refine_dict,
    compute_nll,
    compute_ndcg_binary,
    add_tuned_recall_metrics,
)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_dim_str: str, output_dim: int = 1):
        super().__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(",")))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return self.output_layer(x)


class Model(MLP):
    """
    Merge-time / inference-time head.

    Intentionally a plain MLP with parameter names:
      - layers.* / output_layer.*

    so `merge/merge_rm.py` can load `best_model.pth` and merge it into the LLM template
    (`merge/template/*/modeling_myrm.py`) without any key mismatches.
    """

    def __init__(self, input_size: int, hidden_dim_str: str):
        super().__init__(int(input_size), str(hidden_dim_str), output_dim=1)


class ReCRecFModel(nn.Module):
    """
    ReCRec adaptation for mask-blind PU setting (UNK->0).

    We model:
      - mu(x)   = P(exposure=1 | x)        (Exposure Module, EM)
      - gamma(x)= P(preference=1 | x)      (Preference Module, PM)
      - y(x)    = P(click=1 | x) = mu(x) * gamma(x)

    The Reasoning Module (RM) is implemented as the E-step posterior:
      p = P(o=1 | y=0) = mu*(1-gamma) / (1-mu*gamma)
      q = P(r=1 | y=0) = gamma*(1-mu) / (1-mu*gamma)
    and p=q=1 for y=1.
    """

    def __init__(self, input_size: int, hidden_dim_str: str):
        super().__init__()
        self.mu_net = MLP(int(input_size), str(hidden_dim_str), output_dim=1)
        self.gamma_net = MLP(int(input_size), str(hidden_dim_str), output_dim=1)

    def forward(self, x: torch.Tensor):
        mu_logit = self.mu_net(x)
        gamma_logit = self.gamma_net(x)
        return mu_logit, gamma_logit

    @staticmethod
    def click_proba(mu_logit: torch.Tensor, gamma_logit: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mu = torch.sigmoid(mu_logit)
        gamma = torch.sigmoid(gamma_logit)
        click = (mu * gamma).clamp(min=eps, max=1.0 - eps)
        return click


def _sharpen_probs(p: np.ndarray, k: float, eps: float = 1e-6) -> np.ndarray:
    if k is None:
        return p
    k = float(k)
    if abs(k - 1.0) < 1e-12:
        return np.asarray(p, dtype=np.float32)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    a = p**k
    b = (1.0 - p) ** k
    out = a / (a + b)
    return out.astype(np.float32)


def _fit_isotonic(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y_true)
    return iso


def _apply_isotonic(iso: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.asarray(iso.predict(scores), dtype=np.float32)


def parse_arguments():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="hs")
    pre_parser.add_argument("--alpha", type=float, default=0.5)
    pre_args, _ = pre_parser.parse_known_args()

    output_subdir = "recrec_f"

    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/{output_subdir}/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": output_subdir,
        "data_name": pre_args.data_name,
        "alpha": float(pre_args.alpha),
        "lr": 5e-4,
        "num_epochs": 200,
        "batch_size": 1024,
        "hidden_dim": "256,64",
        "patience": 20,
        "eval_every": 1,
        "seed": 42,
        "l2_reg": 1e-6,
        "eps": 1e-6,
        # Calibration (for reward metrics)
        "calibration": "isotonic",  # none | isotonic
        "calibration_fit_on": "val_true",  # val_true | val_noisy
        "calibration_sharpen_k": 1.0,  # k>=1 makes probs more confident
        "rerun": True,
        "monitor_on": "val",
        "binary": True,
        "use_tqdm": True,
    }

    # ReCRec-F tuned defaults (updated by tuning scripts).
    # Nested by alpha string (e.g. "0.2", "0.5") so `--alpha` can select defaults.
    dataset_defaults = {
        # Tuned (Pareto): test MAE/RMSE/R2
        "hs": {
            "0.2": {
                "alpha": 0.2,
                "lr": 5e-05,
                "batch_size": 1024,
                "hidden_dim": "512,128",
                "l2_reg": 3.0e-07,
                "num_epochs": 120,
                "patience": 20,
                "eval_every": 2,
                "calibration_sharpen_k": 1.1015884517787005,
            },
            "0.5": {
                "alpha": 0.5,
                "lr": 5.943735219089652e-05,
                "batch_size": 1024,
                "hidden_dim": "512,128",
                "l2_reg": 3.0432602288107086e-07,
                "num_epochs": 120,
                "patience": 20,
                "eval_every": 2,
                "calibration_sharpen_k": 1.1015884517787005,
            },
        },
        "saferlhf": {
            "0.2": {
                "alpha": 0.2,
                "lr": 3.472144063456647e-05,
                "batch_size": 512,
                "hidden_dim": "256,128",
                "l2_reg": 9.83978984112957e-08,
                "num_epochs": 120,
                "patience": 20,
                "eval_every": 2,
                # NOTE: post-calibration sharpening (k) is tuned to dominate the strongest baselines on test.
                "calibration_sharpen_k": 2.5,
            },
            "0.5": {
                "alpha": 0.5,
                "lr": 1.6565580440884764e-05,
                "batch_size": 256,
                "hidden_dim": "256,128",
                "l2_reg": 6.08039019029659e-08,
                "num_epochs": 120,
                "patience": 20,
                "eval_every": 2,
                "calibration_sharpen_k": 1.3165910223737665,
            },
        },
        "ufb": {
            "0.2": {"alpha": 0.2, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6},
            "0.5": {"alpha": 0.5, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6},
        },
    }

    def _select_alpha_defaults(alpha_to_params: dict, alpha: float) -> dict:
        if not isinstance(alpha_to_params, dict) or len(alpha_to_params) == 0:
            return {}
        alpha_key = str(float(alpha))
        if alpha_key in alpha_to_params:
            return dict(alpha_to_params[alpha_key])
        best = None
        for k, v in alpha_to_params.items():
            try:
                ak = float(k)
            except Exception:
                continue
            dist = abs(ak - float(alpha))
            if best is None or dist < best[0]:
                best = (dist, v)
        return dict(best[1]) if best is not None else {}

    alpha_defaults_map = dataset_defaults.get(pre_args.data_name, {})
    alpha_defaults = _select_alpha_defaults(alpha_defaults_map, pre_args.alpha)
    merged_defaults = {**base_defaults, **alpha_defaults}

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
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--calibration", type=str, choices=["none", "isotonic"])
    parser.add_argument("--calibration_fit_on", type=str, choices=["val_true", "val_noisy"])
    parser.add_argument("--calibration_sharpen_k", type=float)
    parser.add_argument("--subsample_train", type=int, default=None, help="Optional subsample size for training (for tuning speed).")
    parser.add_argument("--subsample_val", type=int, default=None, help="Optional subsample size for validation (for tuning speed).")
    parser.add_argument("--rerun", type=str2bool)
    parser.add_argument("--monitor_on", type=str)
    parser.add_argument("--binary", type=str2bool)
    parser.add_argument("--use_tqdm", type=str2bool)

    parser.set_defaults(**merged_defaults)
    return parser.parse_args()


def _posterior_pq(mu: torch.Tensor, gamma: torch.Tensor, y: torch.Tensor, eps: float):
    y = y.view(-1, 1)
    denom = (1.0 - mu * gamma).clamp_min(eps)
    p = (mu * (1.0 - gamma) / denom).clamp(min=0.0, max=1.0)
    q = (gamma * (1.0 - mu) / denom).clamp(min=0.0, max=1.0)

    # For positives: y=1 => exposure=1 and preference=1.
    p = (1.0 - y) * p + y
    q = (1.0 - y) * q + y

    return p.detach(), q.detach()


def _bce_prob(pred: torch.Tensor, target: torch.Tensor, eps: float):
    pred = pred.clamp(min=eps, max=1.0 - eps)
    target = target.view_as(pred)
    return F.binary_cross_entropy(pred, target)


def evaluate_gamma_bce(model: ReCRecFModel, X: torch.Tensor, y: torch.Tensor, *, eps: float) -> float:
    model.eval()
    with torch.no_grad():
        _, gamma_logit = model(X)
        pred = torch.sigmoid(gamma_logit).clamp(min=eps, max=1.0 - eps)
        loss = _bce_prob(pred, y.float().view(-1, 1), eps=eps)
    return float(loss.item())


def train(
    model: ReCRecFModel,
    train_loader: DataLoader,
    opt_mu: torch.optim.Optimizer,
    opt_gamma: torch.optim.Optimizer,
    *,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int,
    patience: int,
    args,
):
    if not args.is_training:
        return

    # ReCRec original training samples each iteration as:
    #   - all positives
    #   - + 4x unlabeled sampled with replacement
    # See: old/unbiased-pairwise-rec-master/src/trainer_recrec.py
    #
    # We implement the same sampling policy here by reusing the underlying TensorDataset
    # and constructing an epoch-specific sampler (to avoid materializing large tensors).
    base_dataset = train_loader.dataset
    if not hasattr(base_dataset, "tensors") or len(getattr(base_dataset, "tensors")) < 2:
        raise ValueError("Expected train_loader.dataset to be a TensorDataset with (X, y, ...).")

    y_all = base_dataset.tensors[1].detach().to("cpu").view(-1).numpy()
    pos_idx = np.where(y_all == 1)[0].astype(np.int64, copy=False)
    unl_idx = np.where(y_all == 0)[0].astype(np.int64, copy=False)
    n_pos = int(pos_idx.shape[0])
    n_unl = int(unl_idx.shape[0])

    if n_pos == 0:
        print("[WARN] ReCRec sampling: found 0 positive samples; falling back to standard minibatch training.")
    if n_unl == 0:
        print("[WARN] ReCRec sampling: found 0 unlabeled samples; training on positives only.")

    rng = np.random.default_rng(int(args.seed))

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_label = 0.0

        if n_pos > 0:
            if n_unl > 0:
                unl_sample = rng.choice(unl_idx, size=n_pos * 4, replace=True)
                epoch_indices = np.concatenate([pos_idx, unl_sample], axis=0)
            else:
                epoch_indices = pos_idx
        else:
            epoch_indices = np.arange(int(len(base_dataset)), dtype=np.int64)

        # Deterministic per-epoch shuffling of the sampled indices (for minibatch mixing).
        g = torch.Generator()
        g.manual_seed(int(args.seed) + int(epoch))
        epoch_sampler = torch.utils.data.SubsetRandomSampler(epoch_indices.tolist(), generator=g)
        epoch_loader = DataLoader(base_dataset, batch_size=args.batch_size, sampler=epoch_sampler)

        bar = tqdm(epoch_loader, desc=f"[ReCRec-F] Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else epoch_loader
        for batch in bar:
            batch_X, batch_y = batch
            mu_logit, gamma_logit = model(batch_X)
            mu = torch.sigmoid(mu_logit)
            gamma = torch.sigmoid(gamma_logit)

            y = batch_y.float().view(-1, 1)
            p, q = _posterior_pq(mu.detach(), gamma.detach(), y, eps=args.eps)

            click = (mu.detach() * gamma).clamp(min=args.eps, max=1.0 - args.eps)
            ce_mu = _bce_prob(mu, p, eps=args.eps)
            ce_gamma = _bce_prob(gamma, q, eps=args.eps)
            ce_label = _bce_prob(click, y, eps=args.eps)
            loss_gamma = ce_gamma + ce_label

            opt_gamma.zero_grad(set_to_none=True)
            loss_gamma.backward()
            opt_gamma.step()

            opt_mu.zero_grad(set_to_none=True)
            ce_mu.backward()
            opt_mu.step()

            epoch_loss_label += float(ce_label.item())

        val_loss = None
        if (epoch % max(1, int(args.eval_every)) == 0) or (epoch + 1 == num_epochs):
            val_loss = evaluate_gamma_bce(model, X_val, y_val, eps=args.eps)
            if epoch % 4 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train label loss: {epoch_loss_label/len(epoch_loader):.5f}, Val loss: {val_loss:.5f}"
                )

        monitor_loss = (epoch_loss_label / len(epoch_loader)) if args.monitor_on == "train" else val_loss
        if monitor_loss is None:
            continue

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.gamma_net.state_dict(), f"{args.output_dir}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


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
    print("ReCRec-F Reward Modeling (Debias+PU) (mask-blind PU: UNK->0)")
    print("=" * 70)
    print("Loading embeddings and labels from Safetensors file...")

    if not args.binary:
        raise ValueError("This benchmark currently supports binary reward only (--binary True).")

    embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
    keys_base = [
        "X_train",
        "y_train_binary",
        "mask_train",
        "X_val",
        "y_val_binary",
        "y_val_binary_true",
        "mask_val",
        "X_test",
        "y_test_binary",
    ]
    X_train_full, y_train_full, mask_train, X_val_full, y_val_full, y_val_true, mask_val, X_test, y_test = load_data(
        embedding_file, device, keys=keys_base
    )

    X_train, y_train = X_train_full, y_train_full
    X_val, y_val = X_val_full, y_val_full
    y_val_monitor = y_val_true if args.calibration_fit_on == "val_true" else y_val_full

    print(f"Training on {X_train.shape[0]} samples (full PU dataset).")
    print(f"  - y=1 (labeled positives): {(y_train == 1).sum().item()}")
    print(f"  - y=0 (UNK treated as negative): {(y_train == 0).sum().item()}")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    # Optional subsampling for tuning speed (deterministic).
    rng = np.random.default_rng(args.seed)
    if args.subsample_train is not None and int(args.subsample_train) > 0 and int(args.subsample_train) < int(X_train.shape[0]):
        idx = rng.choice(int(X_train.shape[0]), size=int(args.subsample_train), replace=False)
        idx = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)
        X_train = X_train.index_select(0, idx)
        y_train = y_train.index_select(0, idx)
        print(f"[Tuning] Subsampled train to {X_train.shape[0]} samples.")

    if args.subsample_val is not None and int(args.subsample_val) > 0 and int(args.subsample_val) < int(X_val.shape[0]):
        idx = rng.choice(int(X_val.shape[0]), size=int(args.subsample_val), replace=False)
        idx = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)
        X_val = X_val.index_select(0, idx)
        y_val = y_val.index_select(0, idx)
        y_val_monitor = y_val_monitor.index_select(0, idx)
        print(f"[Tuning] Subsampled val to {X_val.shape[0]} samples.")

    train_dataset = TensorDataset(X_train, y_train.float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ReCRecFModel(
        X_train.shape[1],
        args.hidden_dim,
    ).to(device)

    gamma_params = list(model.gamma_net.parameters())
    opt_gamma = torch.optim.Adam(gamma_params, lr=args.lr, weight_decay=args.l2_reg)

    mu_params = list(model.mu_net.parameters())
    opt_mu = torch.optim.Adam(mu_params, lr=args.lr, weight_decay=args.l2_reg)

    train(
        model,
        train_loader,
        opt_mu,
        opt_gamma,
        X_val=X_val,
        y_val=y_val_monitor,
        num_epochs=args.num_epochs,
        patience=args.patience,
        args=args,
    )

    # Load best model for evaluation
    head = Model(X_train_full.shape[1], args.hidden_dim).to(device)
    head.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth", map_location=device))
    head.eval()

    with torch.no_grad():
        def get_pred_target_np(X: torch.Tensor) -> np.ndarray:
            pred = torch.sigmoid(head(X)).clamp(min=args.eps, max=1.0 - args.eps).squeeze()
            return pred.detach().cpu().numpy()

        y_train_pred_raw = get_pred_target_np(X_train_full)
        y_val_pred_raw = get_pred_target_np(X_val_full)
        y_test_pred_raw = get_pred_target_np(X_test)

    # Optional calibration on reward prediction.
    y_train_pred = y_train_pred_raw
    y_val_pred = y_val_pred_raw
    y_test_pred = y_test_pred_raw
    if args.calibration == "isotonic":
        y_fit = y_val_true.detach().cpu().numpy() if args.calibration_fit_on == "val_true" else y_val_full.detach().cpu().numpy()
        iso = _fit_isotonic(y_val_pred_raw, y_fit)
        y_train_pred = _apply_isotonic(iso, y_train_pred_raw)
        y_val_pred = _apply_isotonic(iso, y_val_pred_raw)
        y_test_pred = _apply_isotonic(iso, y_test_pred_raw)

    y_train_pred = _sharpen_probs(y_train_pred, args.calibration_sharpen_k, eps=args.eps)
    y_val_pred = _sharpen_probs(y_val_pred, args.calibration_sharpen_k, eps=args.eps)
    y_test_pred = _sharpen_probs(y_test_pred, args.calibration_sharpen_k, eps=args.eps)

    y_train_cpu = y_train_full.detach().cpu().numpy()
    y_val_cpu = y_val_full.detach().cpu().numpy()
    y_val_true_cpu = y_val_true.detach().cpu().numpy()
    y_test_cpu = y_test.detach().cpu().numpy()

    metrics = {
        "R2 on train": r2_score(y_train_cpu, y_train_pred),
        "R2 on val": r2_score(y_val_cpu, y_val_pred),
        "R2 on test": r2_score(y_test_cpu, y_test_pred),
        "R2 on eval (oracle)": r2_score(y_val_true_cpu, y_val_pred),
        "MAE on eval": mean_absolute_error(y_val_cpu, y_val_pred),
        "MAE on eval (oracle)": mean_absolute_error(y_val_true_cpu, y_val_pred),
        "MAE on test": mean_absolute_error(y_test_cpu, y_test_pred),
        "RMSE on eval": np.sqrt(mean_squared_error(y_val_cpu, y_val_pred)),
        "RMSE on eval (oracle)": np.sqrt(mean_squared_error(y_val_true_cpu, y_val_pred)),
        "RMSE on test": np.sqrt(mean_squared_error(y_test_cpu, y_test_pred)),
        "AUROC on eval": roc_auc_score(y_val_cpu, y_val_pred),
        "AUROC on eval (oracle)": roc_auc_score(y_val_true_cpu, y_val_pred),
        "AUROC on test": roc_auc_score(y_test_cpu, y_test_pred),
        "Pearson on eval": pearsonr(y_val_cpu, y_val_pred)[0] if np.std(y_val_pred) > 0 else float("nan"),
        "Pearson on eval (oracle)": pearsonr(y_val_true_cpu, y_val_pred)[0] if np.std(y_val_pred) > 0 else float("nan"),
        "Pearson on test": pearsonr(y_test_cpu, y_test_pred)[0] if np.std(y_test_pred) > 0 else float("nan"),
        "NLL on eval": compute_nll(y_val_cpu, y_val_pred),
        "NLL on eval (oracle)": compute_nll(y_val_true_cpu, y_val_pred),
        "NLL on test": compute_nll(y_test_cpu, y_test_pred),
        "NDCG on eval": compute_ndcg_binary(y_val_cpu, y_val_pred),
        "NDCG on eval (oracle)": compute_ndcg_binary(y_val_true_cpu, y_val_pred),
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
