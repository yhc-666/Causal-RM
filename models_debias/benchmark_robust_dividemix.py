import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch

from argparse import ArgumentParser
from itertools import cycle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, log_loss, precision_recall_curve
from sklearn.mixture import BayesianGaussianMixture
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
        "output_dir": f"./results/cache/robust_dividemix/{pre_args.data_name}",
        "data_root": "./embeddings/biased_pu",
        "model_name": "FsfairX-LLaMA3-RM-v0.1",
        "estimator_name": "robust_dividemix",
        "data_name": pre_args.data_name,
        "alpha": 0.2,
        "lr": 0.0002,
        "clip_min": 0.1,
        "warmup_epochs": 10,
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
        "p_threshold": 0.5,
        "lambda_u": 25,
        "T": 0.5,
        "alpha_mix": 4,
        "perturb_step": 0.2,
        "num_steps": 1,
    }

    dataset_defaults = {
        "saferlhf": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
        },
        "hs": {
            "alpha": 0.2,
            "batch_size": 512,
            "lr": 0.0005,
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
    parser.add_argument("--warmup_epochs", type=int) # key parameter
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
    parser.add_argument("--p_threshold", type=float, help="Threshold for propensity score")
    parser.add_argument("--lambda_u", type=float, help="Weight for unobserved data")
    parser.add_argument("--T", type=float, help="Temperature for softmax")
    parser.add_argument("--alpha_mix", type=float, help="Alpha for mixup")
    parser.add_argument("--perturb_step", type=float, help="Step size for label flip")
    parser.add_argument("--num_steps", type=int, help="Number of steps for label flip")

    parser.set_defaults(**merged_defaults)
    args = parser.parse_args()
    return args


class NegEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(NegEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, outputs):
        probs = torch.sigmoid(outputs)
        return -F.binary_cross_entropy(probs, probs, reduction=self.reduction)


def label_flip(model, data, target, num_classes, step_size, num_steps=1):
    """
    BadLabel 对抗性标签扰动 (适配 Binary Output + 纯前向计算)
    """
    model.eval()

    # 构造 one-hot soft label (N, 2)
    target_long = target.long().view(-1)
    soft_label = F.one_hot(target_long, num_classes).float().to(data.device)

    with torch.no_grad(): # 修改点 2: 不需要 requires_grad
        output = model(data) # logits (N, 1)
        prob_1 = torch.sigmoid(output)
        probs = torch.cat([1 - prob_1, prob_1], dim=1)

        # 计算 BadLabel 定义的 "Gradient" (Heuristic)
        grad = -torch.log(probs.add(1e-10))
        perturbed_label = soft_label + step_size * grad * num_steps
        target_flipped = torch.argmax(perturbed_label, dim=1).float()

    return target_flipped, output.squeeze()


def eval_train_perturbed(model, eval_loader, args, last_prob=None):
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    losses = []

    model.eval()

    with torch.no_grad():
        bar = tqdm(eval_loader, desc=f"Eval train perturbed", leave=False) if args.use_tqdm else eval_loader
        for batch_X, batch_y, _ in bar:
            # 1. Label Flip
            noisy_targets, reward_pred = label_flip(model, batch_X, batch_y, 2, args.perturb_step, args.num_steps)
            # 2. Calc Loss
            loss_batch = criterion(reward_pred, noisy_targets)

            losses.append(loss_batch.detach().cpu())

        losses = torch.cat(losses, dim=0).numpy()

    # Fit GMM on the full PU dataset (mask-invisible baseline)
    min_l, max_l = losses.min(), losses.max()
    losses = (losses - min_l) / (max_l - min_l + 1e-8)
    input_loss = losses.reshape(-1, 1)

    # 3. BayesGMM
    gmm = BayesianGaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4, weight_concentration_prior_type='dirichlet_process')
    gmm.fit(input_loss)

    if not gmm.converged_:
        if last_prob is not None:
            print("BayesGMM not converged, using last probability.")
            return last_prob
        else:
            print("BayesGMM not converged at epoch 0. Defaulting to high confidence in clean data.")
            full_probs = np.ones(len(losses)) 
            return full_probs

    prob_all = gmm.predict_proba(input_loss)
    clean_idx = gmm.means_.argmin()
    prob_clean = prob_all[:, clean_idx]
    return prob_clean


def train_warmup(model, train_loader, optimizer, num_epochs, args):
    if not args.is_training: return

    if num_epochs <= 0: return

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    conf_penalty = NegEntropy(reduction='none')

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()

        bar = tqdm(train_loader, desc=f"Training warmup Epoch {epoch+1}/{num_epochs}", leave=False) if args.use_tqdm else train_loader
        for batch_X, batch_y, _ in bar:
            optimizer.zero_grad()
            reward_pred = model(batch_X).squeeze()

            # Loss = BCE + NegEntropy
            loss_vec = criterion(reward_pred, batch_y) + conf_penalty(reward_pred)
            loss = loss_vec.mean()
            weighted_loss = loss * args.w_reg
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Warmup loss: {epoch_loss/len(train_loader):.5f}')


def train_mixmatch_epoch(net, net2, optimizer, labeled_loader, unlabeled_loader, epoch, args):
    net.train()
    net2.eval()

    epoch_loss = 0
    unlabeled_iter = cycle(unlabeled_loader)

    bar = tqdm(labeled_loader, desc=f"Training MixMatch Epoch {epoch}/{args.num_epochs}", leave=False) if args.use_tqdm else labeled_loader
    for batch_idx, (inputs_x, labels_x, w_x) in enumerate(bar):
        optimizer.zero_grad()

        labels_x = labels_x.view(-1, 1).float()
        w_x = w_x.view(-1, 1).float()
        inputs_u, _ = next(unlabeled_iter)

        batch_size = inputs_x.size(0)
        with torch.no_grad():
            # 1. Co-Guessing (Unlabeled / Noisy Set)
            outputs_u1 = torch.sigmoid(net(inputs_u))
            outputs_u2 = torch.sigmoid(net2(inputs_u))
            p = (outputs_u1 + outputs_u2) / 2
            pt = p**(1/args.T)
            targets_u = pt / (pt + (1-p)**(1/args.T))
            targets_u = targets_u.detach()

            # 2. Refinement (Labeled / Clean Set)
            outputs_x1 = torch.sigmoid(net(inputs_x))
            outputs_x2 = torch.sigmoid(net2(inputs_x))
            p_x = (outputs_x1 + outputs_x2) / 2

            targets_x = w_x * labels_x + (1 - w_x) * p_x
            targets_x = targets_x.detach()

        # 3. MixUp
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(args.alpha_mix, args.alpha_mix)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # 4. Forward
        logits = net(mixed_input)
        logits_x, logits_u = logits[:batch_size], logits[batch_size:]
        target_x, target_u = mixed_target[:batch_size], mixed_target[batch_size:]

        # Labeled Loss
        Lx = F.binary_cross_entropy_with_logits(logits_x, target_x)
        # Unlabeled Loss
        probs_u = torch.sigmoid(logits_u)
        Lu = F.mse_loss(probs_u, target_u)

        curr_progress = epoch + batch_idx / len(labeled_loader)
        curr_lambda = args.lambda_u * min(1.0, curr_progress / args.num_epochs)

        # Regularization
        probs = torch.sigmoid(logits)
        probs_2d = torch.cat([1-probs, probs], dim=1) 
        pred_mean = probs_2d.mean(0)
        prior = torch.tensor([0.5, 0.5]).to(inputs_x.device)
        penalty = torch.sum(prior * torch.log(prior / (pred_mean + 1e-12)))

        loss = Lx + curr_lambda * Lu + penalty
        weighted_loss = loss * args.w_reg
        weighted_loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(labeled_loader)


def train(net1, net2, train_data, optimizer1, optimizer2, num_epochs, val_data, patience, device, args):
    if not args.is_training: return

    eval_loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=False)

    # 获取 Tensor 数据用于动态切分
    X_full, y_full, _ = train_data
    prob1, prob2 = None, None 
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # 1. Calc Prob(Clean) on the full PU dataset (mask-invisible baseline)
        prob1 = eval_train_perturbed(net1, eval_loader, args, last_prob=prob1)
        prob2 = eval_train_perturbed(net2, eval_loader, args, last_prob=prob2)

        # 2. Split Data
        def get_split_loaders(probs_arr):
            probs_arr = np.asarray(probs_arr)
            labeled_idx = np.where(probs_arr > args.p_threshold)[0]
            unlabeled_idx = np.where(probs_arr <= args.p_threshold)[0]

            if labeled_idx.size == 0:
                k = max(1, min(args.batch_size, len(probs_arr)))
                labeled_idx = np.argsort(-probs_arr)[:k]
            if unlabeled_idx.size == 0:
                k = max(1, min(args.batch_size, len(probs_arr)))
                unlabeled_idx = np.argsort(probs_arr)[:k]

            labeled_ds = TensorDataset(
                X_full[labeled_idx],
                y_full[labeled_idx],
                torch.tensor(probs_arr[labeled_idx], device=device).float(),
            )
            unlabeled_ds = TensorDataset(X_full[unlabeled_idx], y_full[unlabeled_idx])

            l_bs = min(args.batch_size, len(labeled_idx))
            u_bs = min(args.batch_size, len(unlabeled_idx))
            l_l = DataLoader(labeled_ds, batch_size=l_bs, shuffle=True, num_workers=0)
            u_l = DataLoader(unlabeled_ds, batch_size=u_bs, shuffle=True, num_workers=0)
            return l_l, u_l

        l_loader1, u_loader1 = get_split_loaders(prob2)
        loss_mm1 = train_mixmatch_epoch(net1, net2, optimizer1, l_loader1, u_loader1, epoch, args)

        l_loader2, u_loader2 = get_split_loaders(prob1)
        loss_mm2 = train_mixmatch_epoch(net2, net1, optimizer2, l_loader2, u_loader2, epoch, args)

        val_loss1 = evaluate(net1, val_data, args)
        val_loss2 = evaluate(net2, val_data, args)
        val_loss = (val_loss1 + val_loss2) / 2
        if epoch % 4 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Net1 Loss: {loss_mm1:.4f}, Net2 Loss: {loss_mm2:.4f}, Val Loss: {val_loss:.4f}')

        monitor_loss = (loss_mm1 + loss_mm2) / 2 if args.monitor_on == "train" else val_loss
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
    print("Robust DivideMix Reward Modeling (PU setting: UNK->0, mask-invisible)")
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

    X_train, y_train = X_train_full, y_train_full
    X_val, y_val = X_val_full, y_val_full
    print(f"Training on {X_train.shape[0]} samples (full PU dataset).")
    if args.binary:
        print(f"  - y=1 (labeled positives): {(y_train == 1).sum().item()}")
        print(f"  - y=0 (UNK treated as negative): {(y_train == 0).sum().item()}")
    print(f"Validating on {X_val.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")

    val_data = (X_val_full, y_val_full, mask_val.float())
    test_data = (X_test, y_test, torch.ones_like(y_test))  # mask not used for test

    # Train reward model on full PU dataset (no mask used for training)
    print("\n" + "="*70)
    print("Step 1: Training warmup")
    print("="*70)
    train_indices = torch.arange(X_train_full.shape[0])
    train_data = (X_train_full, y_train_full, train_indices)
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    net1 = Model(X_train_full.shape[1], args.hidden_dim).to(device)
    net2 = Model(X_train_full.shape[1], args.hidden_dim).to(device)
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    train_warmup(
        model=net1,
        train_loader=train_loader,
        optimizer=optimizer1,
        num_epochs=args.warmup_epochs,
        args=args
    )
    train_warmup(
        model=net2,
        train_loader=train_loader,
        optimizer=optimizer2,
        num_epochs=args.warmup_epochs,
        args=args
    )

    # Train reward model on full PU dataset (no mask used for training)
    print("\n" + "="*70)
    print("Step 2: Training Robust DivideMix Reward Model")
    print("="*70)
    train(
        net1=net1,
        net2=net2,
        train_data=train_data,
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        num_epochs=args.num_epochs,
        val_data=val_data,
        patience=args.patience,
        device=device,
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

    # Mask-blind metrics on full train/val PU labels (y_*_binary).
    metrics = {
        "R2 on train": r2_score(y_train_cpu, y_train_pred),
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
        "Recall on eval": compute_recall_binary(y_val_cpu, y_val_pred),
        "Recall on test": compute_recall_binary(y_test_cpu, y_test_pred),
    }
    metrics = refine_dict(metrics)  # avoid .item() error w.r.t version of numpy
    print("\n--- Final Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    save_metrics(args, metrics)


if __name__ == '__main__':
    main()
