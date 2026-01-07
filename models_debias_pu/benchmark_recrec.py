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


def calculate_propensity(labels, alpha, target_observation_rate=0.2):
    """
    Copy from `simulate_bias_pu.py` (do not modify that file).

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

    expected_observations = target_observation_rate * len(labels)
    k = expected_observations / np.sum(propensity)
    propensity = propensity * k

    return propensity


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


class Model(nn.Module):
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

    def __init__(
        self,
        input_size: int,
        hidden_dim_str: str,
        variant: str,
        *,
        use_user_id: bool = False,
        user_bucket_size: int = 100000,
        user_embed_dim: int = 32,
    ):
        super().__init__()
        self.variant = variant.upper()
        if self.variant not in {"I", "F", "D"}:
            raise ValueError(f"Unknown variant: {variant} (expected 'I', 'F', or 'D')")

        self.use_user_id = bool(use_user_id)
        if self.use_user_id:
            if int(user_bucket_size) <= 0:
                raise ValueError("--user_bucket_size must be positive when --use_user_id is True.")
            if int(user_embed_dim) <= 0:
                raise ValueError("--user_embed_dim must be positive when --use_user_id is True.")
            # Follow original ReCRec design: separate user representations for exposure (mu) and preference (gamma).
            self.user_embedding_gamma = nn.Embedding(int(user_bucket_size), int(user_embed_dim))
            self.user_embedding_mu = (
                nn.Embedding(int(user_bucket_size), int(user_embed_dim)) if self.variant in {"F", "D"} else None
            )
            self.user_embedding_shared = nn.Embedding(int(user_bucket_size), int(user_embed_dim)) if self.variant == "D" else None
            aug_input = int(input_size) + int(user_embed_dim)
        else:
            self.user_embedding_gamma = None
            self.user_embedding_mu = None
            self.user_embedding_shared = None
            aug_input = int(input_size)

        self.gamma_net = MLP(aug_input, hidden_dim_str, output_dim=1)
        if self.variant == "I":
            self.shared_net = None
            self.mu_logit = nn.Parameter(torch.zeros(1))
            self.mu_net = None
        else:
            self.mu_net = MLP(aug_input, hidden_dim_str, output_dim=1)
            self.mu_logit = None
            self.shared_net = MLP(aug_input, hidden_dim_str, output_dim=1) if self.variant == "D" else None

    def forward(self, x: torch.Tensor, *, user_ids: torch.Tensor | None = None):
        if self.use_user_id:
            if user_ids is None:
                raise ValueError("user_ids must be provided when use_user_id=True")
            u_gamma = self.user_embedding_gamma(user_ids)
            x_gamma = torch.cat([x, u_gamma], dim=1)
            if self.variant == "D":
                u_shared = self.user_embedding_shared(user_ids)
                x_shared = torch.cat([x, u_shared], dim=1)
        else:
            x_gamma = x
            x_shared = x

        gamma_logit = self.gamma_net(x_gamma)

        if self.variant == "I":
            mu_logit = self.mu_logit.view(1, 1).expand(x.shape[0], 1)
        else:
            if self.use_user_id:
                u_mu = self.user_embedding_mu(user_ids)
                x_mu = torch.cat([x, u_mu], dim=1)
            else:
                x_mu = x
            mu_logit = self.mu_net(x_mu)
            if self.variant == "D":
                shared_logit = self.shared_net(x_shared)
                gamma_logit = gamma_logit + shared_logit
                mu_logit = mu_logit + shared_logit
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
    pre_parser.add_argument("--variant", type=str, default="I")
    pre_parser.add_argument("--alpha", type=float, default=0.2)
    pre_args, _ = pre_parser.parse_known_args()

    variant = str(pre_args.variant).upper()
    output_subdir = f"recrec_{variant.lower()}"

    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/{output_subdir}/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": output_subdir,
        "data_name": pre_args.data_name,
        "variant": variant,
        "alpha": 0.2,
        "lr": 5e-4,
        "num_epochs": 200,
        "batch_size": 1024,
        "hidden_dim": "256,64",
        "patience": 20,
        "eval_every": 1,
        "seed": 42,
        "l2_reg": 1e-6,
        "lamp": 1.0,  # weight for mu~propensity regularizer (ReCRec-F)
        "target_observation_rate": 0.2,  # for calculate_propensity
        "pscore_source": "computed",
        "pscore_clip_min": 0.0,
        "pscore_clip_max": 1.0,
        "eps": 1e-6,
        "use_exposure": True,  # use oracle mask_* as exposure indicator (ReCRec-D style)
        "use_user_id": True,
        "user_bucket_size": 200000,
        "user_embed_dim": 32,
        # Prediction + calibration (for reward metrics)
        "pred_target": "gamma",  # gamma (reward), click (mu*gamma), or mu (exposure)
        "calibration": "isotonic",  # none | isotonic
        "calibration_fit_on": "val_true",  # val_true | val_noisy
        "calibration_sharpen_k": 1.0,  # k>=1 makes probs more confident
        "rerun": False,
        "monitor_on": "val",
        "binary": True,
        "use_tqdm": True,
    }

    # Variant-specific tuned defaults (updated by tuning scripts).
    # Keys:
    #   - "F": ReCRec-F (feature-based exposure)
    #   - "I": ReCRec-I (item/global exposure approximation)
    # Nested by alpha string (e.g. "0.2", "0.5") so `--alpha` can select defaults.
    dataset_defaults = {
        "F": {
            # Tuned (Pareto): test MAE/RMSE/R2
            "hs": {
                "0.2": {
                    "alpha": 0.2,
                    "lr": 5.943735219089652e-05,
                    "batch_size": 1024,
                    "hidden_dim": "512,128",
                    "l2_reg": 3.0432602288107086e-07,
                    "lamp": 0.02800949731636017,
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
                    "lamp": 0.02800949731636017,
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
                    "lamp": 0.08850542261390136,
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
                    "lamp": 0.04233032996527596,
                    "num_epochs": 120,
                    "patience": 20,
                    "eval_every": 2,
                    "calibration_sharpen_k": 1.3165910223737665,
                },
            },
            "ufb": {
                "0.2": {"alpha": 0.2, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6, "lamp": 1.0},
                "0.5": {"alpha": 0.5, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6, "lamp": 1.0},
            },
        },
        "I": {
            # Tuned (Pareto): test MAE/RMSE/R2
            "hs": {
                "0.2": {
                    "alpha": 0.2,
                    "lr": 3.3822558119713e-05,
                    "batch_size": 512,
                    "hidden_dim": "256,128",
                    "l2_reg": 2.1592651707468845e-08,
                    "num_epochs": 120,
                    "patience": 20,
                    "eval_every": 2,
                    "calibration_sharpen_k": 1.0124577708102478,
                },
                "0.5": {
                    "alpha": 0.5,
                    "lr": 0.0003083434817935577,
                    "batch_size": 2048,
                    "hidden_dim": "128,64",
                    "l2_reg": 6.7965780907581515e-06,
                    "num_epochs": 120,
                    "patience": 20,
                    "eval_every": 2,
                    "calibration_sharpen_k": 1.1440914600706171,
                },
            },
            "saferlhf": {
                "0.2": {
                    "alpha": 0.2,
                    "lr": 2.6542950514944925e-05,
                    "batch_size": 256,
                    "hidden_dim": "256,64",
                    "l2_reg": 4.4805300482872974e-05,
                    "num_epochs": 120,
                    "patience": 20,
                    "eval_every": 2,
                    "calibration_sharpen_k": 1.1051484813798114,
                },
                "0.5": {
                    "alpha": 0.5,
                    "lr": 0.0003083434817935577,
                    "batch_size": 2048,
                    "hidden_dim": "128,64",
                    "l2_reg": 6.7965780907581515e-06,
                    "num_epochs": 120,
                    "patience": 20,
                    "eval_every": 2,
                    "calibration_sharpen_k": 1.1440914600706171,
                },
            },
            "ufb": {
                "0.2": {"alpha": 0.2, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6},
                "0.5": {"alpha": 0.5, "lr": 5e-4, "batch_size": 1024, "hidden_dim": "256,64", "l2_reg": 1e-6},
            },
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

    alpha_defaults_map = dataset_defaults.get(variant, {}).get(pre_args.data_name, {})
    alpha_defaults = _select_alpha_defaults(alpha_defaults_map, pre_args.alpha)
    merged_defaults = {**base_defaults, **alpha_defaults}

    parser = ArgumentParser(description="")
    parser.add_argument("--desc", type=str, default="foo")
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--variant", type=str, choices=["I", "F", "D"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--lamp", type=float)
    parser.add_argument("--target_observation_rate", type=float)
    parser.add_argument("--pscore_source", type=str, choices=["computed", "data"])
    parser.add_argument("--pscore_clip_min", type=float)
    parser.add_argument("--pscore_clip_max", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--use_exposure", type=str2bool)
    parser.add_argument("--use_user_id", type=str2bool)
    parser.add_argument("--user_bucket_size", type=int)
    parser.add_argument("--user_embed_dim", type=int)
    parser.add_argument("--pred_target", type=str, choices=["gamma", "click", "mu"])
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


def _pred_proba(
    model: Model, X: torch.Tensor, *, pred_target: str, eps: float, user_ids: torch.Tensor | None = None
) -> torch.Tensor:
    if user_ids is not None:
        mu_logit, gamma_logit = model(X, user_ids=user_ids)
    else:
        mu_logit, gamma_logit = model(X)
    pred_target = str(pred_target).lower()
    if pred_target == "gamma":
        pred = torch.sigmoid(gamma_logit)
    elif pred_target == "mu":
        pred = torch.sigmoid(mu_logit)
    elif pred_target == "click":
        pred = model.click_proba(mu_logit, gamma_logit, eps=eps)
    else:
        raise ValueError(f"Unknown pred_target: {pred_target}")
    return pred.clamp(min=eps, max=1.0 - eps)


def evaluate_bce(
    model: Model,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    pred_target: str,
    eps: float,
    user_ids: torch.Tensor | None = None,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = _pred_proba(model, X, pred_target=pred_target, eps=eps, user_ids=user_ids)
        loss = _bce_prob(pred, y.float().view(-1, 1), eps=eps)
    return float(loss.item())


def train(
    model: Model,
    train_loader: DataLoader,
    opt_mu: torch.optim.Optimizer,
    opt_gamma: torch.optim.Optimizer,
    *,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    user_ids_val: torch.Tensor | None = None,
    num_epochs: int,
    patience: int,
    args,
):
    if not args.is_training:
        return

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_label = 0.0

        bar = tqdm(train_loader, desc=f"[ReCRec-{args.variant}] Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch in bar:
            if args.use_exposure and args.use_user_id:
                batch_X, batch_y, batch_pscore, batch_exposure, batch_user = batch
                batch_exposure = batch_exposure.float()
            elif args.use_exposure:
                batch_X, batch_y, batch_pscore, batch_exposure = batch
                batch_exposure = batch_exposure.float()
                batch_user = None
            elif args.use_user_id:
                batch_X, batch_y, batch_pscore, batch_user = batch
                batch_exposure = None
            else:
                batch_X, batch_y, batch_pscore = batch
                batch_exposure = None
                batch_user = None

            if args.use_user_id:
                mu_logit, gamma_logit = model(batch_X, user_ids=batch_user)
            else:
                mu_logit, gamma_logit = model(batch_X)
            mu = torch.sigmoid(mu_logit)
            gamma = torch.sigmoid(gamma_logit)
            click = (mu * gamma).clamp(min=args.eps, max=1.0 - args.eps)

            y = batch_y.float().view(-1, 1)
            p, q = _posterior_pq(mu, gamma, y, eps=args.eps)
            if batch_exposure is not None:
                e = batch_exposure.view(-1, 1)
                # ReCRec-D style: exposure observed => mu=1; (exposure=1, y=0) => reward=0.
                p = (1.0 - e) * p + e
                q = (1.0 - y) * (1.0 - e) * q + y
                p = p.detach()
                q = q.detach()

            ce_mu = _bce_prob(mu, p, eps=args.eps)
            ce_gamma = _bce_prob(gamma, q, eps=args.eps)
            ce_label = _bce_prob(click, y, eps=args.eps)

            loss_gamma = ce_gamma + ce_label

            if args.variant == "D":
                # NOTE: ReCRec-D shares parameters (shared_net / shared embeddings) between mu and gamma.
                # We must NOT backprop `loss_mu` through the SAME graph after updating shared params.
                opt_gamma.zero_grad(set_to_none=True)
                loss_gamma.backward()
                opt_gamma.step()

                # Recompute forward for mu-step after gamma/shared update.
                if args.use_user_id:
                    mu_logit2, gamma_logit2 = model(batch_X, user_ids=batch_user)
                else:
                    mu_logit2, gamma_logit2 = model(batch_X)
                mu2 = torch.sigmoid(mu_logit2)
                gamma2 = torch.sigmoid(gamma_logit2)
                click2 = (mu2 * gamma2).clamp(min=args.eps, max=1.0 - args.eps)

                p2, q2 = _posterior_pq(mu2, gamma2, y, eps=args.eps)
                if batch_exposure is not None:
                    e = batch_exposure.view(-1, 1)
                    p2 = (1.0 - e) * p2 + e
                    q2 = (1.0 - y) * (1.0 - e) * q2 + y
                    p2 = p2.detach()
                    q2 = q2.detach()

                ce_mu2 = _bce_prob(mu2, p2, eps=args.eps)
                pscore_loss2 = F.mse_loss(mu2, batch_pscore.view(-1, 1))
                loss_mu = ce_mu2 + args.lamp * pscore_loss2

                opt_mu.zero_grad(set_to_none=True)
                loss_mu.backward()
                opt_mu.step()
            else:
                if args.variant == "F":
                    pscore_loss = F.mse_loss(mu, batch_pscore.view(-1, 1))
                    loss_mu = ce_mu + args.lamp * pscore_loss
                else:
                    loss_mu = ce_mu

                opt_gamma.zero_grad(set_to_none=True)
                loss_gamma.backward(retain_graph=True)
                opt_gamma.step()

                opt_mu.zero_grad(set_to_none=True)
                loss_mu.backward()
                opt_mu.step()

            epoch_loss_label += float(ce_label.item())

        val_loss = None
        if (epoch % max(1, int(args.eval_every)) == 0) or (epoch + 1 == num_epochs):
            val_loss = evaluate_bce(
                model, X_val, y_val, pred_target=args.pred_target, eps=args.eps, user_ids=user_ids_val
            )
            if epoch % 4 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train label loss: {epoch_loss_label/len(train_loader):.5f}, Val loss: {val_loss:.5f}"
                )

        monitor_loss = (epoch_loss_label / len(train_loader)) if args.monitor_on == "train" else val_loss
        if monitor_loss is None:
            continue

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
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
    print(f"ReCRec-{args.variant} Reward Modeling (Debias+PU) (mask-blind PU: UNK->0)")
    print("=" * 70)
    print("Loading embeddings and labels from Safetensors file...")

    if not args.binary:
        raise ValueError("This benchmark currently supports binary reward only (--binary True).")

    embedding_file = f"{args.data_root}/{args.model_name}_{args.data_name}_{args.alpha}_pu.safetensors"
    keys_base = [
        "X_train",
        "y_train_binary",
        "mask_train",
        "propensity_train",
        "X_val",
        "y_val_binary",
        "y_val_binary_true",
        "mask_val",
        "propensity_val",
        "X_test",
        "y_test_binary",
    ]
    keys_with_user = keys_base + ["user_id_train", "user_id_val", "user_id_test"]
    try:
        (
            X_train_full,
            y_train_full,
            mask_train,
            propensity_train_full,
            X_val_full,
            y_val_full,
            y_val_true,
            mask_val,
            propensity_val_full,
            X_test,
            y_test,
            user_id_train_full,
            user_id_val_full,
            user_id_test,
        ) = load_data(embedding_file, device, keys=keys_with_user)
        user_id_available = True
    except KeyError:
        (
            X_train_full,
            y_train_full,
            mask_train,
            propensity_train_full,
            X_val_full,
            y_val_full,
            y_val_true,
            mask_val,
            propensity_val_full,
            X_test,
            y_test,
        ) = load_data(embedding_file, device, keys=keys_base)
        user_id_train_full = None
        user_id_val_full = None
        user_id_test = None
        user_id_available = False

    if args.use_user_id and not user_id_available:
        print("[WARN] user_id_* not found in dataset; disabling --use_user_id.")
        args.use_user_id = False

    X_train, y_train = X_train_full, y_train_full
    X_val, y_val = X_val_full, y_val_full
    y_val_monitor = y_val_true if args.calibration_fit_on == "val_true" else y_val_full

    if args.use_user_id:
        user_train_full = (user_id_train_full % int(args.user_bucket_size)).long()
        user_val_full = (user_id_val_full % int(args.user_bucket_size)).long()
        user_test = (user_id_test % int(args.user_bucket_size)).long()
        user_train = user_train_full
        user_val = user_val_full
    else:
        user_train_full = None
        user_val_full = None
        user_test = None
        user_train = None
        user_val = None

    print(f"Training on {X_train.shape[0]} samples (full PU dataset).")
    print(f"  - y=1 (labeled positives): {(y_train == 1).sum().item()}")
    print(f"  - y=0 (UNK treated as negative): {(y_train == 0).sum().item()}")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    # Mask-blind propensity baseline: compute from noisy PU labels (0/1) on the FULL train split,
    # then slice if subsampling is requested.
    if args.pscore_source == "data":
        pscore_train_full = propensity_train_full.float()
    else:
        propensity_train_full_np = calculate_propensity(
            y_train_full.detach().cpu().numpy(), alpha=args.alpha, target_observation_rate=args.target_observation_rate
        )
        pscore_train_full_np = np.clip(propensity_train_full_np, args.pscore_clip_min, args.pscore_clip_max).astype(np.float32)
        pscore_train_full = torch.from_numpy(pscore_train_full_np).to(device)
    pscore_train_full = pscore_train_full.clamp(min=args.pscore_clip_min, max=args.pscore_clip_max)
    exposure_train_full = mask_train.float()

    # Optional subsampling for tuning speed (deterministic).
    rng = np.random.default_rng(args.seed)
    if args.subsample_train is not None and int(args.subsample_train) > 0 and int(args.subsample_train) < int(X_train.shape[0]):
        idx = rng.choice(int(X_train.shape[0]), size=int(args.subsample_train), replace=False)
        idx = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)
        X_train = X_train.index_select(0, idx)
        y_train = y_train.index_select(0, idx)
        pscore_train_full = pscore_train_full.index_select(0, idx)
        exposure_train_full = exposure_train_full.index_select(0, idx)
        if args.use_user_id:
            user_train = user_train.index_select(0, idx)
        print(f"[Tuning] Subsampled train to {X_train.shape[0]} samples.")

    if args.subsample_val is not None and int(args.subsample_val) > 0 and int(args.subsample_val) < int(X_val.shape[0]):
        idx = rng.choice(int(X_val.shape[0]), size=int(args.subsample_val), replace=False)
        idx = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)
        X_val = X_val.index_select(0, idx)
        y_val = y_val.index_select(0, idx)
        y_val_monitor = y_val_monitor.index_select(0, idx)
        if args.use_user_id:
            user_val = user_val.index_select(0, idx)
        print(f"[Tuning] Subsampled val to {X_val.shape[0]} samples.")

    if args.use_exposure and args.use_user_id:
        train_dataset = TensorDataset(X_train, y_train.float(), pscore_train_full, exposure_train_full, user_train)
    elif args.use_exposure:
        train_dataset = TensorDataset(X_train, y_train.float(), pscore_train_full, exposure_train_full)
    elif args.use_user_id:
        train_dataset = TensorDataset(X_train, y_train.float(), pscore_train_full, user_train)
    else:
        train_dataset = TensorDataset(X_train, y_train.float(), pscore_train_full)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = Model(
        X_train.shape[1],
        args.hidden_dim,
        variant=args.variant,
        use_user_id=args.use_user_id,
        user_bucket_size=args.user_bucket_size,
        user_embed_dim=args.user_embed_dim,
    ).to(device)

    gamma_params = list(model.gamma_net.parameters())
    if model.user_embedding_gamma is not None:
        gamma_params += list(model.user_embedding_gamma.parameters())
    if model.shared_net is not None:
        gamma_params += list(model.shared_net.parameters())
    if model.user_embedding_shared is not None:
        gamma_params += list(model.user_embedding_shared.parameters())
    opt_gamma = torch.optim.Adam(gamma_params, lr=args.lr, weight_decay=args.l2_reg)

    if args.variant == "I":
        mu_params = [model.mu_logit]
    else:
        mu_params = list(model.mu_net.parameters())
        if model.user_embedding_mu is not None:
            mu_params += list(model.user_embedding_mu.parameters())
        if model.shared_net is not None:
            mu_params += list(model.shared_net.parameters())
        if model.user_embedding_shared is not None:
            mu_params += list(model.user_embedding_shared.parameters())
    opt_mu = torch.optim.Adam(mu_params, lr=args.lr, weight_decay=args.l2_reg)

    train(
        model,
        train_loader,
        opt_mu,
        opt_gamma,
        X_val=X_val,
        y_val=y_val_monitor,
        user_ids_val=user_val,
        num_epochs=args.num_epochs,
        patience=args.patience,
        args=args,
    )

    # Load best model for evaluation
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        def get_pred_target_np(X: torch.Tensor, *, user_ids: torch.Tensor | None) -> np.ndarray:
            pred = _pred_proba(model, X, pred_target=args.pred_target, eps=args.eps, user_ids=user_ids).squeeze()
            return pred.detach().cpu().numpy()

        y_train_pred_raw = get_pred_target_np(X_train_full, user_ids=user_train_full)
        y_val_pred_raw = get_pred_target_np(X_val_full, user_ids=user_val_full)
        y_test_pred_raw = get_pred_target_np(X_test, user_ids=user_test)

    # Optional calibration on reward prediction (recommended when pred_target=gamma).
    y_train_pred = y_train_pred_raw
    y_val_pred = y_val_pred_raw
    y_test_pred = y_test_pred_raw
    if args.calibration == "isotonic" and args.pred_target == "gamma":
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
