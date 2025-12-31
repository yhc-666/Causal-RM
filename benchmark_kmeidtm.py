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


class TransitionModel(nn.Module):
    """
    Estimates the transition matrix T(x).
    Strictly adapted from 'sig_t' class in kMEIDTM_Code.txt.
    """
    def __init__(self, num_classes=2):
        super(TransitionModel, self).__init__()
        self.num_classes = num_classes

        # Linear layer mapping: num_classes -> num_classes * num_classes
        # Input to this model is the output probability/logit of the base model
        self.fc = nn.Linear(num_classes, num_classes * num_classes, bias=False)

        # Initialize weights to be close to Identity Matrix (Diagonal dominance)
        self.ones = torch.ones(num_classes)
        self.zeros = torch.zeros([num_classes, num_classes])
        self.w = torch.Tensor([])

        for i in range(num_classes):
            temp = self.zeros.clone()
            temp[i] = self.ones - 0.1
            temp = temp + 0.1 / self.num_classes
            self.w = torch.cat([self.w, temp.detach()], 0)

        self.fc.weight.data = self.w

    def forward(self, x):
        # 1. Identity Matrix Construction for regularization (from original code)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = out.view(x.size(0), self.num_classes, -1)

        # Constraints from original code
        # out = torch.sigmoid(out) # Commented in original, but clamp used
        out = torch.clamp(out, min=1e-5, max=1-1e-5)

        # Row normalization (T matrix rows sum to 1)
        out = F.normalize(out, p=1, dim=2)
        return out


def parse_arguments():
    # Pre-parse only data_name to select dataset defaults
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--data_name", type=str, default="saferlhf")
    pre_args, _ = pre_parser.parse_known_args()

    # Base defaults if dataset not listed
    base_defaults = {
        "desc": "foo",
        "is_training": True,
        "output_dir": f"./results/cache/kmeidtm/{pre_args.data_name}",
        "data_root": "../embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "kmeidtm",
        "data_name": pre_args.data_name,
        "alpha": 0.5,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 600,
        "warmup_epochs": 10,
        "batch_size": 512,
        "hidden_dim": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,  # L2 regularization
        "l2_trans": 1e-6,  # L2 regularization
        "w_reg": 1.0,  # Task weight
        "rerun": False,
        "monitor_on": "train",
        "binary": True,
        "use_tqdm": True,
        "lam": 0.3,    # Lambda for manifold regularization
        "gamma": 0.1,  # Sigma for RBF kernel
        "u": 0.8,      # Threshold ratio for distillation
        "knn_k": 7     # k for knn (k1=k2=7 in paper)
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
    parser.add_argument("--warmup_epochs", type=int) # key parameter
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int) # key parameter
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float, help="L2 regularization coefficient") # key parameter
    parser.add_argument("--l2_trans", type=float, help="L2 regularization coefficient") # key parameter
    parser.add_argument("--w_reg", type=float, help="Task weight") # key parameter
    parser.add_argument("--w_trans", type=float, help="Task weight") # key parameter
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or test set")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary or continuous rewards")
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")
    parser.add_argument("--lam", type=float, help="Lambda for manifold regularization")
    parser.add_argument("--gamma", type=float, help="gamma for RBF kernel")
    parser.add_argument("--u", type=float, help="Threshold ratio for distillation")
    parser.add_argument("--knn_k", type=int, help="k for knn (k1=k2=7 in paper)")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train_warmup(model, train_loader, optimizer, num_epochs, args):
    if not args.is_training: return

    if num_epochs <= 0: return

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()

        bar = tqdm(train_loader, desc=f"Training warmup Epoch {epoch+1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()

            # Loss = BCE + NegEntropy, then masked
            loss_vec = criterion(reward_pred, batch_y)
            loss_vec = loss_vec * batch_mask
            loss = loss_vec.mean()
            weighted_loss = loss * args.w_reg
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Warmup loss: {epoch_loss/len(train_loader):.5f}')


def distilling(model, train_data, args):
    """
    Selects confident samples based on current model predictions.
    Operates on the Full dataset but only selects from Observed.
    """
    X_full, y_full, mask_full = train_data

    model.eval()
    with torch.no_grad():
        # Only check observed samples for distillation candidate
        obs_idx = torch.where(mask_full > 0.5)[0]
        X_obs = X_full[obs_idx]
        y_obs = y_full[obs_idx] 

        # Get Clean Predictions
        logits = model(X_obs)
        probs = torch.sigmoid(logits)
        probs_2d = torch.cat([1-probs, probs], dim=1) # (N, 2)

        # Get confidence of the predicted class that matches the NOISY label
        # y_obs_long = y_obs.long().view(-1, 1)
        # clean_confidence = probs_2d.gather(1, y_obs_long).squeeze()
        # sorted_conf, sorted_indices = torch.sort(clean_confidence, descending=True)
        max_confidence, _ = torch.max(probs_2d, dim=1)
        sorted_conf, sorted_indices = torch.sort(max_confidence, descending=True)

        # Select Top-U%
        num_select = int(args.u * len(obs_idx))
        selected_rel_indices = sorted_indices[:num_select]
        selected_abs_indices = obs_idx[selected_rel_indices]

    return selected_abs_indices


def compute_manifold_loss(clean_features, T, batch_y, args):
    """
    Calculates the manifold regularization loss based on RBF kernel affinity.
    Ref: loss_our function in attachment
    """
    N = clean_features.size(0)
    if N <= 1: return torch.tensor(0.0).to(clean_features.device)

    clean_features = F.normalize(clean_features, p=2, dim=1)

    # 1. Feature Space Similarity (RBF Kernel)
    dist_mat = torch.cdist(clean_features, clean_features, p=2).pow(2)
    s_ij1 = torch.exp(-dist_mat * args.gamma)

    # k = min(args.knn_k + 1, N) # +1 因为包含自身
    # _, indices = torch.topk(s_ij1, k=k, dim=1)
    # mask_knn = torch.zeros_like(s_ij1)
    # mask_knn.scatter_(1, indices, 1.0)
    # s_ij1 = s_ij1 * mask_knn

    # 2. Label Consistency (Intrinsic vs Extrinsic)
    y_col = batch_y.view(-1, 1).long()
    label_match = (y_col == y_col.T)

    # S_ij = 1 if same noisy label, -1 if different
    s_ij = -torch.ones(N, N).to(clean_features.device)
    s_ij[label_match] = 1.0

    # Combine
    s_ij = s_ij * s_ij1

    # 3. Transition Matrix Distance
    T_flat = T.view(N, -1)
    ij_dist = torch.cdist(T_flat, T_flat, p=2).pow(2)

    # 4. Loss
    manifold_loss = torch.mean(s_ij.detach() * ij_dist)
    return manifold_loss


def train(model, trans_model, train_data, optimizer, optimizer_trans, num_epochs, val_data, patience, args):
    """
    Train reward model using naive loss:
    L = mask * (error)
    
    where:
        error = ℓ(r(x), y) - loss between predicted and true reward
        mask = observation indicator (1 if observed, 0 otherwise)
    """
    if not args.is_training: return

    X_train_full, y_train_full, mask_train = train_data

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 1. Distill Data
        if epoch % 10 == 0:
            distilled_indices = distilling(model, train_data, args)
            train_dataset = TensorDataset(X_train_full[distilled_indices], y_train_full[distilled_indices])
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model.train()
        trans_model.train()

        epoch_loss = 0
        epoch_manifold_loss = 0

        bar = tqdm(train_loader, desc=f"Training kMEIDTM Epoch {epoch + 1}/{args.num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y in bar:
            optimizer.zero_grad()
            optimizer_trans.zero_grad()

            # 1. Base Model Prediction (Clean)
            logits = model(batch_X) # (B, 1)
            probs = torch.sigmoid(logits)
            clean_probs_2d = torch.cat([1-probs, probs], dim=1) 

            # 2. Estimate Transition Matrix T(x)
            T = trans_model(clean_probs_2d) # (B, 2, 2)

            # 3. Compute Noisy Prediction
            # P(y_noisy) = P(y_clean) * T
            out_noisy = torch.bmm(clean_probs_2d.unsqueeze(1), T).squeeze(1) # (B, 1, 2) bmm (B, 2, 2) -> (B, 1, 2)

            # 4. Losses
            # CE Loss on Noisy Labels
            ce_loss = criterion(out_noisy, batch_y.long())

            # Manifold Loss
            # Note: We use raw features batch_X for distance calculation in manifold
            manifold_loss = compute_manifold_loss(batch_X, T, batch_y, args)

            loss = ce_loss + args.lam * manifold_loss
            weighted_loss = loss * args.w_reg
            weighted_loss.backward()

            optimizer.step()
            optimizer_trans.step()

            epoch_loss += ce_loss.item()
            epoch_manifold_loss += manifold_loss.item()

        # Evaluation
        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}, Train loss: {epoch_loss/len(train_loader):.5f}, Manifold Loss: {epoch_manifold_loss/len(train_loader):.5f}, Val Loss: {val_loss:.5f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
            torch.save(trans_model.state_dict(), f'{args.output_dir}/best_trans_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
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

    # Train reward model on observed data only
    print("\n" + "="*70)
    print("Step 1: Training warmup")
    print("="*70)
    train_data = (X_train_full, y_train_full, mask_train.float())
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train_warmup(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.warmup_epochs,
        args=args
    )

    # Train reward model on observed data only
    print("\n" + "="*70)
    print("Step 2: Training Reward Model")
    print("="*70)
    trans_model = TransitionModel(num_classes=2).to(device)
    optimizer_trans = torch.optim.Adam(trans_model.parameters(), lr=args.lr, weight_decay=args.l2_trans)

    train(
        model=model,
        trans_model=trans_model,
        train_data=train_data,
        optimizer=optimizer,
        optimizer_trans=optimizer_trans,
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
