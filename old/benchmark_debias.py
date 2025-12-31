# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import yaml
import argparse
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

# Import custom modules
from matrix_factorization_recsys import MF_DR_JL, MF_DR, MF_CDR_JL
from mf_sdr import MF_Stable_DR, MF_TDR, MF_TDR_JL, MF_UDR
from dataset import load_data
from utils import (
    gini_index,
    ndcg_func,
    get_user_wise_ctr,
    recall_func,
    seed_everything,
    precision_func,
    binarize,
    test
)


def evaluate_model(model, x_test, y_test):
    test_pred = model.predict(x_test)
    mse, mae, auc = mean_squared_error(y_test, test_pred), mean_absolute_error(Y_TEST, test_pred),roc_auc_score(Y_TEST, test_pred)
    ndcg = ndcg_func(model, x_test, y_test, top_k_list = [1, 3, 5, 8, 10])
    recall = recall_func(model, x_test, y_test, top_k_list = [1, 3, 5, 8, 10])
    precision = precision_func(model, x_test, y_test, top_k_list = [1, 3, 5, 8, 10])
    user_wise_ctr = get_user_wise_ctr(x_test, y_test, test_pred)
    gi,gu = gini_index(user_wise_ctr)

    metrics = {'mse': round(mse.tolist(), 5), 'mae': round(mae.tolist(), 5), 'auc': round(auc.tolist(), 5), 'train_loss': round(loss.tolist(), 5), 'gini': round(gi.tolist(), 5), 'utility': round(gu.tolist(), 5)}
    ndcg = {key: round(np.mean(ndcg[key]).tolist(), 5) for key in ndcg}
    recall = {key: round(np.mean(recall[key]).tolist(), 5) for key in recall}
    precision = {key: round(np.mean(precision[key]).tolist(), 5) for key in precision}
    metrics.update(ndcg)
    metrics.update(recall)
    metrics.update(precision)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Basic')
    parser.add_argument('--model', default='dr-jl')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data', default='coat')
    parser.add_argument('--outpath', default='debug0415')
    parser.add_argument('--verbose', default=1)

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--thres', default=4, type=float)
    parser.add_argument('--embed', default=8, type=int)
    
    parser.add_argument('--use_ukn_ratio', default=1, type=float)
    parser.add_argument('--pos_batch_ratio', default=0.1, type=float)
    parser.add_argument('--class_prior', default=0.05, type=float)
    parser.add_argument('--pureg', default='nnpu4', type=str)
    parser.add_argument('--puell', default='logistic',  type=str)
    parser.add_argument('--eta_threshold', default=1.5,  type=float)
    parser.add_argument('--lamb', default=1e-4,  type=float)
    parser.add_argument('--gamma', default=0.1,  type=float)
    args = parser.parse_args()

    args.outpath = f"./{args.outpath}/{args.data}_{args.model}_{args.batch_size}_{args.lr}_{args.eta_threshold}_{args.lamb}_{args.gamma}_{args.seed}"
    args.datapath = f"./data/{args.data}.txt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    seed_everything(args.seed)

    # Load and Preprocess Data
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, NUM_USER, NUM_ITEM = load_data(args.data, "./data")
    Y_TRAIN, Y_TEST = binarize(Y_TRAIN, thres=args.thres), binarize(Y_TEST, thres=args.thres)
    ips_idxs = np.arange(len(Y_TEST))
    np.random.shuffle(ips_idxs)
    prior_y = Y_TEST[ips_idxs[:int(0.05 * len(ips_idxs))]]
    # Model Initialization and Training
    if args.model == 'drjl':
        model = MF_DR_JL(NUM_USER, NUM_ITEM, embedding_k=args.embed, device=device, batch_size=args.batch_size, batch_size_prop=args.batch_size)
        model._compute_IPS(X_TRAIN, lr = 0.01, lamb = 5e-4)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, lamb=args.lamb, gamma=args.gamma, G=1, tol=1e-4, verbose=False)
    elif args.model == 'dr':
        model = MF_DR(NUM_USER, NUM_ITEM, embedding_k=args.embed, device=device, batch_size=args.batch_size, batch_size_prop=args.batch_size)
        model._compute_IPS(X_TRAIN, lr = 0.01, lamb = 5e-4)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, lamb=args.lamb, gamma=args.gamma, G=1, tol=1e-4, verbose=False, prior_y=prior_y)
    elif args.model == 'cdrjl':
        model = MF_CDR_JL(NUM_USER, NUM_ITEM, embedding_k=args.embed, device=device, batch_size=args.batch_size, batch_size_prop=args.batch_size)
        model._compute_IPS(X_TRAIN, lr = 0.01, lamb = 5e-4)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, lamb=args.lamb, gamma=args.gamma, G=1, tol=1e-4, verbose=False, eta_threshold=args.eta_threshold)
    elif args.model == 'cdrjl0':
        model = MF_CDR_JL(NUM_USER, NUM_ITEM, embedding_k=args.embed, device=device, batch_size=args.batch_size, batch_size_prop=args.batch_size)
        model._compute_IPS(X_TRAIN, lr = 0.01, lamb = 5e-4)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, lamb=args.lamb, gamma=args.gamma, G=1, tol=1e-4, verbose=False, eta_threshold=100000)
    elif args.model == 'tdr':
        model = MF_TDR(NUM_USER, NUM_ITEM, batch_size=args.batch_size, embedding_k=args.embed)
        loss = model.fit(X_TRAIN, Y_TRAIN, prior_y, lr=args.lr, lamb=args.lamb, gamma=args.gamma, G=1, tol=1e-4, verbose=False)
    elif args.model == 'sdr':
        model = MF_Stable_DR(NUM_USER, NUM_ITEM, embedding_k=args.embed)
        loss = model.fit(X_TRAIN, Y_TRAIN, prior_y, lr=args.lr, lamb=args.lamb, batch_size=args.batch_size, eta=1000, lr1=100, G=5, tol=1e-4, verbose=False)
    elif args.model == 'udr':
        model = MF_UDR(NUM_USER, NUM_ITEM, embedding_k_pred=args.embed, embedding_k_impu=args.embed, embedding_k_prop=args.embed)
        loss = model.fit(X_TRAIN[:, 0], X_TRAIN[:, 1], Y_TRAIN, lr_pred=args.lr, lamb_pred=args.lamb, batch_size=256)


    metrics = test(model, X_TEST, Y_TEST, loss)
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    with open(f"{args.outpath}/configs.yaml", 'w') as f:
        yaml.dump(vars(args), f, encoding='utf-8', allow_unicode=True)
    with open(f"{args.outpath}/results.yaml", 'w') as f:
        yaml.dump(metrics, f, encoding='utf-8', allow_unicode=True)

    print(",".join([f"{key} {metrics[key]}" for key in metrics]))
    torch.save(model.state_dict(), f"{args.outpath}/checkpoint.pt")
