# -*- coding: utf-8 -*-
import numpy as np
import torch
import yaml
import os
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from matrix_factorization_recsys import MF, MF_implicit_sampled, MF_reg_implicit, RMF_sample, ExpoMF, PairwiseRecommender, UPRL, SVD
from dataset import load_data
from utils import gini_index, ndcg_func, get_user_wise_ctr, recall_func, seed_everything, precision_func
from utils import binarize
import argparse
from benchmark_fullot import MF_explicit_sampled



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Basic')
    parser.add_argument('--model_name', default='mf')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--data', default='coat')
    parser.add_argument('--outpath', default='debug0910')
    parser.add_argument('--verbose', default=1)

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--thres', default=4, type=float)

    
    parser.add_argument('--use_ukn_ratio', default=1, type=float)
    parser.add_argument('--pos_batch_ratio', default=0.5, type=float)

    parser.add_argument('--class_prior', default=0.1, type=float)
    parser.add_argument('--pureg', default='nnpu', type=str)
    parser.add_argument('--puell', default='sigmoid',  type=str)
    args = parser.parse_args()
    # args.outpath = f"./{args.outpath}/{args.data}_{args.model}_{args.batch_size}_{args.lr}_{args.use_ukn_ratio}_{args.pos_batch_ratio}_{args.class_prior}_{args.pureg}_{args.puell}_{args.seed}"
    args.outpath = f"./{args.outpath}/{args.data}_{args.model_name}_{args.seed}"
    args.datapath = f"./data/{args.data}.txt"
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    seed_everything(args.seed)
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, NUM_USER, NUM_ITEM = load_data(args.data, "./data")
    Y_TRAIN, Y_TEST = binarize(Y_TRAIN, thres=args.thres), binarize(Y_TEST, thres=args.thres)

    if args.model_name == 'mf':
        model = MF(NUM_USER, NUM_ITEM, embedding_k=8, device=device)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size)
    if args.model_name == 'wmf':
        model = RMF_sample(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='wmf')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='squared', weight=0.8, clip_min=0.1, pos_batch_ratio=0.5, protocol='full')
    if args.model_name == 'rmf':
        model = RMF_sample(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='rmf')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='squared', weight=0.8, clip_min=1e-8, pos_batch_ratio=0.5, protocol='full')
    if args.model_name == 'crmf':
        model = RMF_sample(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='crmf')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='squared', weight=0.8, clip_min=1e-1, pos_batch_ratio=0.5, protocol='full')
    if args.model_name == 'expomf':
        model = ExpoMF(NUM_USER, NUM_ITEM, embedding_k=8, device=device, max_iter=10)
        loss = model.fit(X_TRAIN, Y_TRAIN, batch_size=args.batch_size)
    if args.model_name == 'bpr':
        model = PairwiseRecommender(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='bpr')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, weight=0.8, protocol='full')
    if args.model_name == 'ubpr':
        model = PairwiseRecommender(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='ubpr')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, weight=0.8, protocol='full', clip_min=1e-8)
    if args.model_name == 'cubpr':
        model = PairwiseRecommender(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='cubpr')
        if args.data=='coat': # the maxixmum pscore = 0.13
            loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, weight=0.8, protocol='full', clip_min=1.5e-1)
        else:
            loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, weight=0.8, protocol='full', clip_min=1e-1)
    if args.model_name == 'uprl':
        model = UPRL(NUM_USER, NUM_ITEM, embedding_k=8, device=device, model_name='uprl')
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, protocol='full')
    if args.model_name == 'purl':
        model = MF_implicit_sampled(NUM_USER, NUM_ITEM, embedding_k=8, device=device)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='logistic', pureg='nnpu4', pos_batch_ratio=0.1, class_prior=0.05, use_pt=True, pt_kappa=0.3, protocol='full')
    if args.model_name == 'bmf':
        model = MF_implicit_sampled(NUM_USER, NUM_ITEM, embedding_k=8, device=device)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='logistic', pureg='wnaive', exp_batch_ratio=0.1, class_prior=0.05, use_pt=False, pt_kappa=0.3, protocol='full')
    if args.model_name == 'bmffull':
        model = MF_reg_implicit(NUM_USER, NUM_ITEM, embedding_k=8, device=device)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size, puell='logistic', pureg='wnaive', use_pt=False, protocol='full')
    if args.model_name == 'svd':
        model = SVD(NUM_USER, NUM_ITEM, embedding_k=8, device=device)
        loss = model.fit(X_TRAIN, Y_TRAIN, num_epoch=args.n_epochs, lr=args.lr, batch_size=args.batch_size)

    test_pred = model.predict(X_TEST)
    mse, mae, auc = mean_squared_error(Y_TEST, test_pred), mean_absolute_error(Y_TEST, test_pred),roc_auc_score(Y_TEST, test_pred)
    logloss = -Y_TEST*np.log(np.clip(test_pred, 1e-3, 1-1e-3)) - (1-Y_TEST)*np.log(np.clip(1-test_pred, 1e-3, 1-1e-3))
    ndcg = ndcg_func(model, X_TEST, Y_TEST, top_k_list = [1, 3, 5, 8, 10])
    recall = recall_func(model, X_TEST, Y_TEST, top_k_list = [1, 3, 5, 8, 10])
    precision = precision_func(model, X_TEST, Y_TEST, top_k_list = [1, 3, 5, 8, 10])
    user_wise_ctr = get_user_wise_ctr(X_TEST, Y_TEST, test_pred)
    gi,gu = gini_index(user_wise_ctr)

    metrics = {'mse': round(mse.tolist(), 5), 'mae': round(mae.tolist(), 5), 'auc': round(auc.tolist(), 5), 'train_loss': round(loss.tolist(), 5), 'gini': round(gi.tolist(), 5), 'utility': round(gu.tolist(), 5)}
    ndcg = {key: round(np.mean(ndcg[key]).tolist(), 5) for key in ndcg}
    recall = {key: round(np.mean(recall[key]).tolist(), 5) for key in recall}
    precision = {key: round(np.mean(precision[key]).tolist(), 5) for key in precision}
    metrics.update(ndcg)
    metrics.update(recall)
    metrics.update(precision)

    if os.path.exists(args.outpath) is False:
        os.makedirs(args.outpath)
    with open(f"{args.outpath}/configs.yaml", 'w') as f:
        yaml.dump(vars(args), f, encoding='utf-8', allow_unicode=True)
    with open(f"{args.outpath}/results.yaml", 'w') as f:
        yaml.dump(metrics, f, encoding='utf-8', allow_unicode=True)
    
    print(",".join([f"{key} {metrics[key]}" for key in metrics]))
    if args.model_name != 'expomf':
        torch.save(model.state_dict(), f"{args.outpath}/checkpoint.pt")
