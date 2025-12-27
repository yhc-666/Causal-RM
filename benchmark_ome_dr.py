import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from argparse import ArgumentParser
from itertools import cycle
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
        "output_dir": f"./results/cache/ome_dr/{pre_args.data_name}",
        "data_root": "../embeddings/biased_noisy",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "ome_dr",
        "data_name": pre_args.data_name,
        "alpha": 0.5,
        "lr": 0.0002,
        "clip_min": 0.1,
        "num_epochs": 600,
        "batch_size": 512,
        "batch_size_prop": 512,
        "batch_size_full": 2048,
        "hidden_dim": "256,64",
        "hidden_dim_prop": "256,64",
        "patience": 30,
        "seed": 42,
        "l2_reg": 1e-6,
        "l2_prop": 1e-6,
        "l2_noise": 1e-6,
        "l2_imp": 1e-6,
        "w_reg": 1.0,
        "w_prop": 1.0,  # Task weight for propensity model training
        "w_noise": 1.0,  # Task weight for noise model training
        "w_imp": 1.0,  # Task weight for noise model training
        "rerun": False,
        "monitor_on": "train",
        "binary": True,
        "r10": 0.1,
        "r01": 0.2,
        "quant": 0.97,
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
    parser.add_argument("--output_dir", type=str, default="./results/nn_biasedfoo")
    parser.add_argument("--data_root", type=str, default="../embeddings/biased")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--alpha", type=float, help="Alpha parameter for propensity calculation")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--clip_min", type=float, help="Minimum clip value for propensity weights")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--batch_size_prop", type=int)
    parser.add_argument("--batch_size_full", type=int)
    parser.add_argument("--hidden_dim", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--hidden_dim_prop", type=str, help="Hidden dimensions, e.g., '128,64'")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--l2_prop", type=float)
    parser.add_argument("--l2_noise", type=float)
    parser.add_argument("--l2_imp", type=float)
    parser.add_argument("--w_reg", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--w_prop", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--w_noise", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--w_imp", type=float, help="Task weight for propensity model training") # key parameter
    parser.add_argument("--rerun", type=str2bool, help="Whether to rerun the experiment")
    parser.add_argument("--monitor_on", type=str, help="Whether to monitor on train or test set")
    parser.add_argument("--binary", type=str2bool, help="Whether to use binary or continuous rewards")
    parser.add_argument("--r10", type=float, help="Noise ratio for positive to negative")
    parser.add_argument("--r01", type=float, help="Noise ratio for negative to positive")
    parser.add_argument("--quant", type=float)
    parser.add_argument("--use_tqdm", type=str2bool, help="Whether to use tqdm progress bar")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


def train_propensity_model(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    criterion_mean = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()
        bar = tqdm(train_loader, desc=f"Training Propensity Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, _, batch_mask in bar:
            optimizer.zero_grad()
            propensity_scores = model(batch_X).squeeze()
            propensity_scores = F.sigmoid(propensity_scores)
            loss = criterion_mean(propensity_scores, batch_mask.float())
            # Apply task weight
            weighted_loss = loss * args.w_prop
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = evaluate(model, val_data, args, propensity=True)
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {loss.item():.5f}, Val loss: {val_loss:.5f}')

        monitor_loss = val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_propensity_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for propensity model after {epoch + 1} epochs.")
                break


def train_noise_prediction_model(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    criterion_mean = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()

        bar = tqdm(train_loader, desc=f"Training Noise Prediction Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            observed = batch_mask > 0.5
            if observed.sum() > 0:
                loss = criterion_mean(outputs[observed], batch_y[observed])
                weighted_loss = loss * args.w_noise
                weighted_loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            else:
                continue

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {loss.item():.5f}, Val loss: {val_loss:.5f}')

        monitor_loss = val_loss
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_noise_pred_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for noise prediction model after {epoch + 1} epochs.")
                break


def train_baseline_model(model, train_loader, optimizer, num_epochs, val_data, patience, args):
    """
    Train a baseline reward model on observed data only.
    This baseline is used in the Doubly Robust estimator.
    """
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0
    criterion = nn.MSELoss() if not args.binary else nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        bar = tqdm(train_loader, desc=f"Training Baseline Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()
            observed = batch_mask > 0.5
            if observed.sum() > 0:
                reward_pred = model(batch_X[observed]).squeeze()
                loss = criterion(reward_pred, batch_y[observed].squeeze())
                # Apply task weight
                weighted_loss = loss * args.w_imp
                weighted_loss.backward()
                optimizer.step()
                epoch_loss += loss.item()  # Track unweighted loss for reporting
            else:
                continue

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Baseline Epoch {epoch + 1}/{num_epochs}, Train loss: {epoch_loss/len(train_loader):.5f}, Val loss: {val_loss:.5f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_baseline_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for baseline model after {epoch + 1} epochs.")
                break


def train(model, propensity_model, noise_pred_model, baseline_model, train_loader, train_full_iter, optimizer, num_epochs, val_data, patience, args):
    if not args.is_training: return

    best_loss = float('inf')
    patience_counter = 0

    criterion = nn.MSELoss(reduction='none') if not args.binary else nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()
        propensity_model.eval()
        noise_pred_model.eval()
        baseline_model.eval()

        bar = tqdm(train_loader, desc=f"Training Unified Model Epoch {epoch + 1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, batch_mask in bar:
            optimizer.zero_grad()

            with torch.no_grad():
                X_sampled, y_sampled, mask_sampled = next(train_full_iter)
                pred_scores = noise_pred_model(X_sampled).squeeze().detach()
                quantiles = torch.quantile(pred_scores, torch.tensor([args.quant, 1-args.quant], device=batch_X.device))
                quantiles = F.sigmoid(quantiles)
                rho10 = quantiles[1].item()
                rho01 = 1 - quantiles[0].item()

                prop_pred = propensity_model(batch_X).squeeze()
                prop_pred = F.sigmoid(prop_pred)
                # Clip propensity scores to avoid extreme values
                prop_clipped = torch.clip(prop_pred, args.clip_min, 1.0).detach()

                baseline_pred = baseline_model(batch_X).squeeze().detach()
                if args.binary:
                    baseline_pred = F.sigmoid(baseline_pred)

            reward_pred = model(batch_X).squeeze()

            mask_float = batch_mask.float()
            denom = max(1 - rho10 - rho01, 1e-6)
            loss_1 = (
                (1 - rho10) * criterion(reward_pred, torch.ones_like(reward_pred)) - 
                rho01 * criterion(reward_pred, torch.zeros_like(reward_pred))
            ) / denom
            loss_0 = (
                (1 - rho01) * criterion(reward_pred, torch.zeros_like(reward_pred)) - 
                rho10 * criterion(reward_pred, torch.ones_like(reward_pred))
            ) / denom
            error = batch_y * loss_1 + (1 - batch_y) * loss_0
            error_hat = criterion(reward_pred, baseline_pred)
            error_diff = error - error_hat
            ips_correction = mask_float * error_diff / prop_clipped

            loss = ips_correction + error_hat

            # Average over batch
            loss = torch.mean(loss)
            # Apply task weight
            weighted_loss = loss * args.w_reg
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = evaluate(model, val_data, args)
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {loss.item():.5f}, Val loss: {val_loss:.5f}, rho10: {rho10:.5f}, rho01: {rho01:.5f}')

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
    print("OME DR Reward Modeling")
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

    # Step 1: Train propensity model
    print("\n" + "="*70)
    print("Step 1: Training Propensity Model")
    print("="*70)
    train_loader_prop = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size_prop, shuffle=True)
    model_prop = Model(X_train.shape[1], args.hidden_dim_prop).to(device)
    optimizer_prop = torch.optim.Adam(model_prop.parameters(), lr=args.lr, weight_decay=args.l2_prop)

    train_propensity_model(
        model=model_prop,
        train_loader=train_loader_prop,
        optimizer=optimizer_prop,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        args=args
    )
    del train_loader_prop, optimizer_prop
    model_prop.load_state_dict(torch.load(f'{args.output_dir}/best_propensity_model.pth'))
    model_prop.eval()

    # Step 2: Train noise prediction model
    print("\n" + "="*70)
    print("Step 2: Training Noise Prediction Model")
    print("="*70)
    train_loader_noise = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size, shuffle=True)
    model_noise = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer_noise = torch.optim.Adam(model_noise.parameters(), lr=args.lr, weight_decay=args.l2_noise)

    train_noise_prediction_model(
        model=model_noise,
        train_loader=train_loader_noise,
        optimizer=optimizer_noise,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        args=args
    )
    del train_loader_noise, optimizer_noise
    model_noise.load_state_dict(torch.load(f'{args.output_dir}/best_noise_pred_model.pth'))
    model_noise.eval()

    # Step 3: Train baseline reward model on observed data only
    print("\n" + "="*70)
    print("Step 3: Training Baseline Reward Model")
    print("="*70)
    print("Training baseline model on observed samples only (for DR correction term)")
    train_loader_baseline = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size, shuffle=True)
    baseline_model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=args.lr, weight_decay=args.l2_imp)

    train_baseline_model(
        model=baseline_model,
        train_loader=train_loader_baseline,
        optimizer=optimizer_baseline,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        args=args
    )
    del train_loader_baseline, optimizer_baseline
    baseline_model.load_state_dict(torch.load(f'{args.output_dir}/best_baseline_model.pth'))
    baseline_model.eval()

    # Step 4: Train final reward model using OME DR
    print("\n" + "="*70)
    print("Step 4: Training Reward Model with OME DR")
    print("="*70)
    train_loader = DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size, shuffle=True)
    train_full_iter = cycle(DataLoader(TensorDataset(X_train_full, y_train_full, mask_train.float()), batch_size=args.batch_size_full, shuffle=True))
    model = Model(X_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train(
        model=model,
        propensity_model=model_prop,
        noise_pred_model=model_noise,
        baseline_model=baseline_model,
        train_loader=train_loader,
        train_full_iter=train_full_iter,
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
            reward_pred = reward_pred.cpu().numpy()
            prop_pred = F.sigmoid(model_prop(X).squeeze())
            prop_pred = prop_pred.cpu().numpy()
            y_cpu = y.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            return prop_pred, reward_pred, y_cpu, mask_cpu

        prop_train_pred, y_train_pred, y_train_cpu, mask_train_cpu = get_preds(X_train_full, y_train_full, mask_train.float())
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
