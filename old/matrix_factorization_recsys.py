# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import pandas as pd
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
import ot
# from ray.air import session

from utils import EarlyStopping

from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from joblib import Parallel, delayed


mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def implicit_data_generation(x, y, user_ukn_ratio):
    x_pos = x[y==1]
    x_ukn = [(user, item) for user in range(x[:, 0].min(), x[:, 0].max()+1) for item in range(x[:, 1].min(), x[:, 1].max()+1)]
    x_ukn = np.array(list(set(x_ukn)-set(map(tuple, x_pos))), dtype=int)
    np.random.shuffle(x_ukn)
    x_ukn = x_ukn[:int(len(x_ukn)*user_ukn_ratio)] # select the ratio of unknown samples
    return x_pos, x_ukn

def explicit_data_generation(x, y, user_ukn_ratio):
    x_ukn = [(user, item) for user in range(x[:, 0].min(), x[:, 0].max()+1) for item in range(x[:, 1].min(), x[:, 1].max()+1)]
    x_ukn = np.array(list(set(x_ukn)-set(map(tuple, x))), dtype=int)
    np.random.shuffle(x_ukn)
    x_ukn = x_ukn[:int(len(x_ukn)*user_ukn_ratio)] # select the ratio of unknown samples
    return x, x_ukn

def implicit_data_generation_semi(x, y, class_prior):
    """
    |x_flip|/(|x_flip|+|x_neg|)=k -> |x_flip|=(k/(1-k))|x_neg|
    kuairec: class_prior<=0.2
    yahoo: class_prior<=0.35
    coat: class_prior<=0.2
    """
    
    x_pos, x_neg = x[y==1], x[y==0]
    prior = int((class_prior/(1-class_prior))*len(x_neg))
    assert int(class_prior/(1-class_prior)*len(x_neg)) < 0.85*len(x_pos) # |x_flip| <= 0.5 |x_pos|
    print(prior)
    np.random.shuffle(x_pos)
    x_flip = x_pos[:prior]
    x_pos = x_pos[prior:]
    x_ukn = np.concatenate([x_neg,x_flip])
    return x_pos, x_ukn


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = torch.device(device)
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)
        # out = self.sigmoid(out)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, labda_emb=0, batch_size=1024, 
        tol=1e-4, verbose=False, ell='logistic', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        num_sample = len(x)
        total_batch = num_sample // batch_size
        earlystop = EarlyStopping(patience=5, delta=tol)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.tensor(sub_y, device=self.device, dtype=torch.float)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                if ell=='logistic':
                    xent_loss = F.binary_cross_entropy_with_logits(pred, sub_y)
                elif ell=='squared':
                    xent_loss = F.mse_loss(torch.sigmoid(pred), sub_y)
                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            import copy
            earlystop(epoch_loss, state_dict=copy.deepcopy(self.state_dict()))
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                self.load_state_dict(earlystop.state)
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss
      
    def predict(self, x):
        pred = self.forward(x)
        pred = torch.sigmoid(pred)
        return pred.detach().cpu().numpy()


class SVD(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.W_b = torch.nn.Embedding(self.num_users, 1, device=self.device)
        self.H_b = torch.nn.Embedding(self.num_items, 1, device=self.device)
        self.mu = torch.tensor([0.0], requires_grad=True, device=self.device)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1) + self.W_b(user_idx).flatten() + self.H_b(item_idx).flatten() + self.mu

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        
class SVDPlus(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.W_b = torch.nn.Embedding(self.num_users, 1, device=self.device)
        self.H_b = torch.nn.Embedding(self.num_items, 1, device=self.device)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1) + self.W_b(user_idx) + self.H_b(item_idx) + self.mu + V_emb.mul()
        # out = self.sigmoid(out)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

def ell_cal(pred, test, type='sigmoid'):
    assert type in ['squared', 'logistic', 'sigmoid', 'bce']
    z = pred * (test*2-1) # from [0,1] to [-1,1]
    if type == 'squared':
        loss = (z-1)**2/4
    if type == 'logistic':
        loss = torch.log(1+torch.exp(-z))
    if type == 'sigmoid':
        loss = 1/(1+torch.exp(z))
    if type == 'bce': # same as logistic given sigmoid activation
        loss = F.binary_cross_entropy_with_logits(pred, test, reduction='none')

    return loss
    

class MF_reg_implicit(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu'):
        super().__init__(num_users, num_items, embedding_k, device)
  
           
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, pureg=None, puell='bce', class_prior=0.5, user_ukn_ratio=1.0, semi_prior=0.2, protocol='full', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)
        
        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)
        x = np.r_[x_pos, x_ukn]
        y = np.r_[np.ones(len(x_pos)), np.zeros(len(x_ukn))]

        total_batch =  len(x) // batch_size
        for epoch in range(num_epoch):

            all_idx = np.arange(len(x))            
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                _idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                _x = x[_idx]
                _y = y[_idx]
                _pos_pred, _, _ = self.forward(_x[_y==1], True)
                _ukn_pred, _, _ = self.forward(_x[_y==0], True)

                #/*------ loss calculate ------*/ 
                loss_pp = ell_cal(_pos_pred, torch.ones_like(_pos_pred), type=puell)
                loss_pn = ell_cal(_pos_pred, torch.zeros_like(_pos_pred), type=puell)
                loss_un = ell_cal(_ukn_pred, torch.zeros_like(_ukn_pred), type=puell)
                if pureg == 'naive': # bce without reweight
                    xent_loss = torch.mean(loss_pp) + torch.mean(loss_un)
                elif pureg == 'wnaive': # ordinary bce. should be equal to xent_loss = F.binary_cross_entropy(torch.cat([_pos_pred, _ukn_pred]), torch.cat([_pos_y, _ukn_y]))
                    xent_loss = torch.mean(torch.concat([loss_pp, loss_un])) # TODO check the correctness with ordinary BCE.
                elif pureg == 'pu':
                    xent_loss = class_prior*(torch.mean(loss_pp)-torch.mean(loss_pn)) + torch.mean(loss_un)
                elif pureg == 'nnpu':
                    xent_loss = class_prior*torch.mean(torch.abs(loss_pp-loss_pn)) + torch.mean(loss_un)
                elif pureg == 'nnpu2':
                    xent_loss = class_prior*torch.mean(loss_pp) + torch.abs(torch.mean(loss_un)-class_prior*torch.mean(loss_pn))
                elif pureg == 'nnpu3':
                    xent_loss = class_prior*torch.mean(torch.clip(loss_pp-loss_pn, min=0.0)) + torch.mean(loss_un)
                elif pureg == 'nnpulab':
                    xent_loss = 2*class_prior*torch.abs(torch.mean(torch.sigmoid(_pos_pred))-1) + torch.abs(torch.mean(torch.sigmoid(_ukn_pred))-class_prior)
                # loss = xent_loss
                # assert (loss_pp*len(_pos_pred) + loss_un * len(_ukn_pred)) / (len(_pos_pred) + len(_ukn_pred)) == F.binary_cross_entropy(torch.sigmoid(torch.cat([_pos_pred, _ukn_pred])), torch.cat([_pos_y, _ukn_y]))
                # assert F.binary_cross_entropy(torch.sigmoid(torch.cat([_pos_pred, _ukn_pred])), torch.cat([_pos_y, _ukn_y])) == F.binary_cross_entropy_with_logits(torch.cat([_pos_pred, _ukn_pred]), torch.cat([_pos_y, _ukn_y]))
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            import copy
            earlystop(epoch_loss, state_dict=copy.deepcopy(self.state_dict()))
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                self.load_state_dict(earlystop.state)
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss

def kl_divergence(p, q):
    return np.sum(np.where(q*p != 0, p * np.log(p / q), 0))


class MF_implicit_sampled(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.costs=None
           
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=True, user_ukn_ratio=1, pos_batch_ratio=0.1, class_prior=0.025, pureg=None, puell='bce', semi_prior=0.2, protocol='full',  use_pt=False, pt_kappa=0.5, **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)

        pos_batch_size = int(batch_size * pos_batch_ratio)
        ukn_batch_size = int(batch_size - pos_batch_size)

        total_batch =  len(x_pos) // pos_batch_size
        for epoch in range(num_epoch):

            pos_idx = np.arange(len(x_pos))
            np.random.shuffle(pos_idx)

            epoch_loss = 0

            for idx in range(total_batch):
                _pos_idx = pos_idx[pos_batch_size*idx:(idx+1)*pos_batch_size]
                _ukn_idx = np.random.choice(len(x_ukn), ukn_batch_size)
                _pos_x = x_pos[_pos_idx]
                _ukn_x = x_ukn[_ukn_idx]

                _pos_pred, _pos_u, _pos_v = self.forward(_pos_x, True)
                _ukn_pred, _ukn_u, _ukn_v = self.forward(_ukn_x, True)
                if epoch > 40 and use_pt is True and idx == 0:
                    _pos_data = torch.concat([_pos_u, _pos_v], dim=-1).detach().cpu().numpy()
                    _ukn_data = torch.concat([_ukn_u, _ukn_v], dim=-1).detach().cpu().numpy()
                    w_set = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
                    costs = []
                   
                    for w in w_set:
                        a = np.ones(shape=(len(_pos_data)))/len(_pos_data)
                        b = w*np.ones(shape=(len(_ukn_data)))/len(_ukn_data)
                        M = ot.dist(_pos_data, _ukn_data)
                        G1 = ot.partial.partial_wasserstein(a=a, b=b, m=1-1e-5, M=M) 
                        b1 = np.ones(shape=(len(_ukn_data)))/len(_ukn_data)
                        b1_hat = np.matmul(G1.transpose(), a*len(_pos_data))
                        cost = np.sum(G1*M)+ pt_kappa*(kl_divergence(b1, b1_hat) + kl_divergence(b1_hat, b1))
                        costs += [np.round(cost, 4)]
                    
                    costs = np.array(costs)
                    if self.costs is None:
                        self.costs = costs.reshape(1, -1)
                    else:
                        self.costs = np.concatenate([self.costs, costs.reshape(1, -1)], axis=0)
                    w = w_set[np.argmin(costs)]
                    class_prior = 1/w
                    

                loss_pp = ell_cal(_pos_pred, torch.ones_like(_pos_pred), type=puell)
                loss_pn = ell_cal(_pos_pred, torch.zeros_like(_pos_pred), type=puell)
                loss_un = ell_cal(_ukn_pred, torch.zeros_like(_ukn_pred), type=puell)
                if pureg == 'naive': # bce without reweight
                    xent_loss = torch.mean(loss_pp) + torch.mean(loss_un)
                elif pureg == 'wnaive': # ordinary bce. should be equal to xent_loss = F.binary_cross_entropy(torch.cat([_pos_pred, _ukn_pred]), torch.cat([_pos_y, _ukn_y]))
                    xent_loss = torch.mean(torch.concat([loss_pp, loss_un])) # TODO check the correctness with ordinary BCE.
                elif pureg == 'pu':
                    xent_loss = class_prior*(torch.mean(loss_pp)-torch.mean(loss_pn)) + torch.mean(loss_un)
                elif pureg == 'nnpu':
                    xent_loss = class_prior*torch.mean(torch.abs(loss_pp-loss_pn)) + torch.mean(loss_un)
                elif pureg == 'nnpu2':
                    xent_loss = class_prior*torch.mean(loss_pp) + torch.abs(torch.mean(loss_un)-class_prior*torch.mean(loss_pn))
                elif pureg == 'nnpu3':
                    xent_loss = class_prior*torch.mean(loss_pp) + torch.clip(torch.mean(loss_un) - class_prior*torch.mean(loss_pn), min=0.0)
                elif pureg == 'nnpu4':
                    xent_loss = class_prior*torch.mean(torch.clip(loss_pp-loss_pn, min=0.0)) + torch.mean(loss_un)
                elif pureg == 'nnpulab':
                    xent_loss = 2*class_prior*torch.abs(torch.mean(torch.sigmoid(_pos_pred))-1) + torch.abs(torch.mean(torch.sigmoid(_ukn_pred))-class_prior)
                # loss = xent_loss
                # assert (loss_pp*len(_pos_pred) + loss_un * len(_ukn_pred)) / (len(_pos_pred) + len(_ukn_pred)) == F.binary_cross_entropy(torch.sigmoid(torch.cat([_pos_pred, _ukn_pred])), torch.cat([_pos_y, _ukn_y]))
                # assert F.binary_cross_entropy(torch.sigmoid(torch.cat([_pos_pred, _ukn_pred])), torch.cat([_pos_y, _ukn_y])) == F.binary_cross_entropy_with_logits(torch.cat([_pos_pred, _ukn_pred]), torch.cat([_pos_y, _ukn_y]))

                # def mixup_bce(u, v, y, alpha=6.0, device=None):

                #     lam = np.random.beta(alpha, alpha) if alpha>0 else 1
                #     index = torch.randperm(len(u)).to(device)

                #     mixed_u = lam * u + (1-lam) * u[index, :]
                #     mixed_v = lam * v + (1-lam) * v[index, :]
                #     mixed_score = torch.sum(u.mul(mixed_v), 1)

                #     y_a, y_b = y, y[index]

                #     mixup_loss_a = F.binary_cross_entropy_with_logits(mixed_score, y_a)
                #     mixup_loss_b = F.binary_cross_entropy_with_logits(mixed_score, y_b)
                    
                #     mixup_loss = lam * mixup_loss_a + (1-lam) * mixup_loss_b
                #     return mixup_loss
                
                # loss_mixup = mixup_bce(torch.concat([_pos_u, _ukn_u]), torch.concat([_pos_v, _ukn_v]), torch.concat([torch.ones_like(_pos_pred), torch.ones_like(_ukn_pred)]), alpha=1.0, device=self.device)

                # loss = xent_loss + kwargs.get('mix_ratio', 0.0) * loss_mixup
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()     
                epoch_loss += xent_loss.detach().cpu().numpy()
            if epoch>12 and pureg=='pu': # for coat dataset
                    break
            # print(epoch_loss)
            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 5 == 0 and verbose:
                print("Epoch:{}, xent:{}, class:{}".format(epoch, epoch_loss, class_prior))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss


class RMF(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', model_name='crmf', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.model_name = model_name
           
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, puell='bce', clip_min=1e-8, weight=1.0, user_ukn_ratio=1.0, semi_prior=0.2, protocol='full', **kwargs):

        _, item_freq = np.unique(x[y == 1, 1], return_counts=True)
        freq_map = {_[i]: item_freq[i] for i in range(len(_))}
        freq_max = max(freq_map.values())
        pscores = {key: (freq_map[key]/freq_max)**0.5 for key in freq_map}

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)

        x = np.r_[x_pos, x_ukn]
        y = np.r_[np.ones(len(x_pos)), np.zeros(len(x_ukn))]

        total_batch =  len(x) // batch_size
        for epoch in range(num_epoch):

            all_idx = np.arange(len(x))            
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                _idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                _x = x[_idx]
                _y = y[_idx]
                _pred, _, _ = self.forward(_x, True)

                #/*------ loss calculate ------*/ 
                # _pscore = pscore[_x[:, 1].astype(np.int)] 
                _pscore = [pscores.get(_x[i, 1].astype(np.int), 0) for i in range(len(_x))]
                _y, _pscore = torch.tensor(_y, device=self.device),  torch.tensor(_pscore, device=self.device)
                if self.model_name == 'crmf':
                    _pscore = torch.clip(_pscore, clip_min, 1.0)
                elif self.model_name =='rmf':
                    _pscore = torch.clip(_pscore, 1e-8, 1.0)
                elif self.model_name == 'wmf':
                    _pscore = torch.ones_like(_pscore)
        
                if self.model_name != 'wmf':
                    weight = 1.0

                xent_loss = (_y/_pscore) * torch.square(1.0 - torch.sigmoid(_pred)) + weight * (1 - _y/_pscore) * torch.square(torch.sigmoid(_pred))
                xent_loss = torch.sum(xent_loss) / torch.sum(_y + weight * (1-_y))    
          
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss


class RMF_sample(MF):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', model_name='crmf', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.model_name = model_name
           
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, puell='bce', clip_min=1e-1, weight=1.0, user_ukn_ratio=1, pos_batch_ratio=0.5, semi_prior=0.2, protocol='full', **kwargs):

        _, item_freq = np.unique(x[y == 1, 1], return_counts=True)
        freq_map = {_[i]: item_freq[i] for i in range(len(_))}
        freq_max = max(freq_map.values())
        pscores = {key: (freq_map[key]/freq_max)**0.5 for key in freq_map}

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)

        pos_batch_size = int(batch_size * pos_batch_ratio)
        ukn_batch_size = int(batch_size - pos_batch_size)

        total_batch =  len(x_pos) // pos_batch_size
        for epoch in range(num_epoch):

            pos_idx = np.arange(len(x_pos))
            np.random.shuffle(pos_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                _pos_idx = pos_idx[pos_batch_size*idx: (idx+1)*pos_batch_size]
                _ukn_idx = np.random.choice(len(x_ukn), ukn_batch_size)
                _pos_x= x_pos[_pos_idx]
                _ukn_x = x_ukn[_ukn_idx]
                _x = np.r_[_pos_x, _ukn_x]
                _y = np.r_[np.ones(shape=len(_pos_x), dtype=int), np.ones(shape=len(_ukn_x), dtype=int)]
                _pred, _, _ = self.forward(_x, True)

                #/*------ loss calculate ------*/ 
                _pscore = [pscores.get(_x[i, 1].astype(int), 0) for i in range(len(_x))]
                _y, _pscore = torch.tensor(_y, device=self.device),  torch.tensor(_pscore, device=self.device)
                if self.model_name == 'crmf':
                    _pscore = torch.clip(_pscore, clip_min, 1.0)
                elif self.model_name =='rmf':
                    _pscore = torch.clip(_pscore, 1e-8, 1.0)
                elif self.model_name == 'wmf' or self.model_name == 'mf':
                    _pscore = torch.ones_like(_pscore)
        
                if self.model_name != 'wmf':
                    weight = 1.0

                xent_loss = (_y/_pscore) * torch.square(1.0 - torch.sigmoid(_pred)) + weight * (1 - _y/_pscore) * torch.square(torch.sigmoid(_pred))
                xent_loss = torch.sum(xent_loss) / torch.sum(_y + weight * (1-_y))    
          
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss
    

def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'click', 'propensity'])
    positive = df.query("click == 1")
    negative = df.query("click == 0")
    ret = positive.merge(negative, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'click_y', 'propensity_x', 'propensity_y']].values

def _ubpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'click', 'propensity'])
    positive = df.query("click == 1")
    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'propensity_x', 'propensity_y']].values



def _bpr_test(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'gamma', 'propensity'])
    ret = df.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'gamma_x', 'gamma_y']].values


# def _ubpr(data: np.ndarray, pscore: np.ndarray, n_samples: int) -> np.ndarray:
#     """Generate training data for the unbiased bpr."""
#     data = np.c_[data, pscore[data[:, 1].astype(int)]]
#     df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta'])
#     positive = df.query("click == 1")
#     ret = positive.merge(df, on="user")\
#         .sample(frac=1, random_state=12345)\
#         .groupby(["user", "item_x"])\
#         .head(n_samples)
#     ret = ret[ret["item_x"] != ret["item_y"]]

#     return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_x', 'theta_y']].values

class MF_denoised(MF_implicit_sampled):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', **kwargs):
        super().__init__(num_users, num_items, embedding_k, device)
        self.costs=None
           
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, user_ukn_ratio=1, pos_batch_ratio=0.1, class_prior=0.025, pureg=None, puell='bce', semi_prior=0.2, protocol='full',  use_pt=False, pt_kappa=0.5, **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)

        pos_batch_size = int(batch_size * pos_batch_ratio)
        ukn_batch_size = int(batch_size - pos_batch_size)

        total_batch =  len(x_pos) // pos_batch_size
        for epoch in range(num_epoch):

            pos_idx = np.arange(len(x_pos))
            np.random.shuffle(pos_idx)

            epoch_loss = 0

            for idx in range(total_batch):
                _pos_idx = pos_idx[pos_batch_size*idx:(idx+1)*pos_batch_size]
                _ukn_idx = np.random.choice(len(x_ukn), ukn_batch_size)
                _pos_x = x_pos[_pos_idx]
                _ukn_x = x_ukn[_ukn_idx]

                _pos_pred, _pos_u, _pos_v = self.forward(_pos_x, True)
                _ukn_pred, _ukn_u, _ukn_v = self.forward(_ukn_x, True)
                
                loss_pp = ell_cal(_pos_pred, torch.ones_like(_pos_pred), type=puell)
                loss_pn = ell_cal(_pos_pred, torch.zeros_like(_pos_pred), type=puell)
                loss_un = ell_cal(_ukn_pred, torch.zeros_like(_ukn_pred), type=puell)
                loss_pp = loss_pp[loss_pp<3]
                if pureg == 'naive': # bce without reweight
                    xent_loss = torch.mean(loss_pp) + torch.mean(loss_un)
                elif pureg == 'wnaive': # ordinary bce. should be equal to xent_loss = F.binary_cross_entropy(torch.cat([_pos_pred, _ukn_pred]), torch.cat([_pos_y, _ukn_y]))
                    xent_loss = torch.mean(torch.concat([loss_pp, loss_un])) # TODO check the correctness with ordinary BCE.                
               
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()     
                epoch_loss += xent_loss.detach().cpu().numpy()
            if epoch>12 and pureg=='pu': # for coat dataset
                    break
            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss


class PairwiseRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', model_name='bpr', **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = torch.device(device)
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)
        self.model_name = model_name

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    def predict(self, x):
        pred = self.forward(x)
        pred = torch.sigmoid(pred)
        return pred.detach().cpu().numpy() 
    
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, clip_min=1e-1, user_ukn_ratio=1, semi_prior=0.2, protocol='full', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        _, item_freq = np.unique(x[y == 1, 1], return_counts=True)
        freq_map = {_[i]: item_freq[i] for i in range(len(_))}
        freq_max = max(freq_map.values())
        pscores = {key: (freq_map[key]/freq_max)**0.5 for key in freq_map}

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)
        x, y = np.r_[x_pos,x_ukn], np.r_[np.ones(len(x_pos)), np.zeros(len(x_ukn))]

        pscore = [pscores.get(x[i, 1].astype(int), 0) for i in range(len(x))]
        data = np.c_[x, y, pscore]
        data = _bpr(data, 10)
        total_batch =  len(data) // batch_size

        for epoch in range(num_epoch):

            all_idx = np.arange(len(data))
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                _idx = all_idx[batch_size*idx: (idx+1)*batch_size]
                _data= data[_idx]
                
                #/*------ loss calculate ------*/
                # 'user', 'item_x', 'item_y', 'click_y', 'peopensity_x', 'propensity_y'
                _user, _item_1, _item_2, _y2, _pscore1, _pscore2 = _data[:, 0], _data[:, 1], _data[:, 2], _data[:, 3], _data[:, 4], _data[:, 5]
                _pred_1, _, _ = self.forward(np.c_[_user,_item_1], True)
                _pred_2, _, _ = self.forward(np.c_[_user,_item_2], True)

                _pscore1, _pscore2, _y2 = torch.tensor(_pscore1, device=self.device), torch.tensor(_pscore2, device=self.device), torch.tensor(_y2, device=self.device)
                if self.model_name =='cubpr':
                    _pscore1 = torch.clip(_pscore1, clip_min, 1.0)
                    _pscore2 = torch.clip(_pscore2, clip_min, 1.0)
                    weight = -(1/_pscore1) * (1-(_y2/_pscore2))
                elif self.model_name == 'ubpr':
                    _pscore1 = torch.clip(_pscore1, 1e-8, 1.0)
                    _pscore2 = torch.clip(_pscore2, 1e-8, 1.0)
                    weight = -1.0*(1/_pscore1) * (1-(_y2/_pscore2))
                elif self.model_name == 'bpr':
                    weight = -1.0
                
                xent_loss = weight*torch.log(torch.sigmoid(_pred_1-_pred_2))
                xent_loss = torch.mean(xent_loss) 
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss
    

class UBPR(PairwiseRecommender):
    
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, clip_min=1e-8, user_ukn_ratio=1, semi_prior=0.2, protocol='full', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        _, item_freq = np.unique(x[y == 1, 1], return_counts=True)
        freq_map = {_[i]: item_freq[i] for i in range(len(_))}
        freq_max = max(freq_map.values())
        pscores = {key: (freq_map[key]/freq_max)**0.5 for key in freq_map}

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)
        x, y = np.r_[x_pos,x_ukn], np.r_[np.ones(len(x_pos)), np.zeros(len(x_ukn))]

        pscore = [pscores.get(x[i, 1].astype(np.int), 0) for i in range(len(x))]
        data = np.c_[x, y, pscore]
        data = _ubpr(data, 10)
        total_batch =  len(data) // batch_size

        for epoch in range(num_epoch):

            all_idx = np.arange(len(data))
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                _idx = all_idx[batch_size*idx: (idx+1)*batch_size]
                _data= data[_idx]
                
                #/*------ loss calculate ------*/
                # 'user', 'item_x', 'item_y', 'click_y', 'peopensity_x', 'propensity_y'
                _user, _item_1, _item_2, _y2, _pscore1, _pscore2 = _data[:, 0], _data[:, 1], _data[:, 2], _data[:, 3], _data[:, 4], _data[:, 5]
                _pred_1, _, _ = self.forward(np.c_[_user,_item_1], True)
                _pred_2, _, _ = self.forward(np.c_[_user,_item_2], True)

                _pscore1, _pscore2, _y2 = torch.tensor(_pscore1, device=self.device), torch.tensor(_pscore2, device=self.device), torch.tensor(_y2, device=self.device)
                if self.model_name =='cubpr':
                    _pscore1 = torch.clip(_pscore1, clip_min, 1.0)
                    _pscore2 = torch.clip(_pscore2, clip_min, 1.0)
                    weight = -1.0*(1/_pscore1) * (1-(_y2/_pscore2))
                elif self.model_name == 'ubpr':
                    _pscore1 = torch.clip(_pscore1, 1e-8, 1.0)
                    _pscore2 = torch.clip(_pscore2, 1e-8, 1.0)
                    weight = -1.0*(1/_pscore1) * (1-(_y2/_pscore2))
                elif self.model_name == 'bpr':
                    weight = -1.0
                
                xent_loss = weight*torch.log(torch.sigmoid(_pred_1-_pred_2))
                xent_loss = torch.mean(xent_loss) 
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            earlystop(epoch_loss)
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss

class UPRL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', model_name='bpr', **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = torch.device(device)
        self.W_point = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H_point = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)
        self.W_pair = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H_pair = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)
        self.model_name = model_name

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_point = self.W_point(user_idx)
        V_point = self.H_point(item_idx)
        U_pair = self.W_pair(user_idx)
        V_pair = self.H_pair(item_idx)
        out_point = torch.sum(U_point.mul(V_point), 1)
        out_pair = torch.sum(U_pair.mul(V_pair), 1)

        if is_training:
            return out_point, out_pair, U_point, V_point, U_pair, V_pair
        else:
            return out_pair
        
    def predict(self, x):
        pred = self.forward(x)
        pred = torch.sigmoid(pred)
        return pred.detach().cpu().numpy() 
    
    def fit(self, x, y, num_epoch=1000, lr=0.01, labda_emb=0, batch_size=1024, tol=1e-4, verbose=False, user_ukn_ratio=1, semi_prior=0.2, protocol='full', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=labda_emb)
        earlystop = EarlyStopping(patience=5, delta=tol)

        _, item_freq = np.unique(x[y == 1, 1], return_counts=True)
        freq_map = {_[i]: item_freq[i] for i in range(len(_))}
        freq_max = max(freq_map.values())
        pscores = {key: (freq_map[key]/freq_max)**0.5 for key in freq_map}

        if protocol == 'full':
            x_pos, x_ukn = implicit_data_generation(x, y, user_ukn_ratio)
        elif protocol == 'semi':
            x_pos, x_ukn = implicit_data_generation_semi(x, y, class_prior=semi_prior)
        x, y = np.r_[x_pos,x_ukn], np.r_[np.ones(len(x_pos)), np.zeros(len(x_ukn))]

        pscore = [pscores.get(x[i, 1].astype(np.int), 0) for i in range(len(x))]
        data = np.c_[x, y, pscore]
        data = _ubpr(data, 10)
        total_batch =  len(data) // batch_size

        for epoch in range(num_epoch):

            all_idx = np.arange(len(data))
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                _idx = all_idx[batch_size*idx: (idx+1)*batch_size]
                _data= data[_idx]
                
                #/*------ loss calculate ------*/
                # 'user', 'item_x', 'item_y', 'click_y', 'peopensity_x', 'propensity_y'
                _user, _item_1, _item_2, _y2, _pscore1, _pscore2 = _data[:, 0], _data[:, 1], _data[:, 2], _data[:, 3], _data[:, 4], _data[:, 5]
                _pred_point_1, _pred_pair_1, _,_,_,_ = self.forward(np.c_[_user,_item_1], True)
                _pred_point_2, _pred_pair_2, _,_,_,_ = self.forward(np.c_[_user,_item_2], True)

                _pscore1, _pscore2, _y2 = torch.tensor(_pscore1, device=self.device), torch.tensor(_pscore2, device=self.device), torch.tensor(_y2, device=self.device)
                
                _pscore1 = torch.clip(_pscore1, 1e-8, 1.0)
                _pscore2 = torch.clip(_pscore2, 1e-8, 1.0)
                weight = -1.0*(1/_pscore1) * (1-_y2)/_pscore2
                _local_loss = weight*torch.log(torch.sigmoid(_pred_pair_1-_pred_pair_2))
                _pred_point_neg = torch.sigmoid(_pred_point_2).detach()
                numerator = _pscore2 * (1-_pred_point_neg)
                denominator = 1 - _pred_point_neg * _pscore2 + 1e-5
                _local_loss = torch.mean(torch.clip(numerator / denominator * _local_loss, 0.0, 1e2))
                # _local_ce = torch.mean(F.binary_cross_entropy_with_logits(_pred_point_2, _y2, reduction='none'))
                _local_ce = _y2/_pscore2 * torch.log(torch.sigmoid(_pred_point_2)+1e-5) + torch.clip((1-_y2/_pscore2), -1e8, 1e8) * torch.log(1.0-torch.sigmoid(_pred_point_2)+1e-5)
                _local_ce = -1.0* torch.mean(torch.clip(_local_ce, -1e8, 1e8))  # Compared with the vanilla version, we add a clip operator here, for convergence.
                xent_loss = _local_loss + _local_ce 
                if torch.any(torch.isnan(xent_loss)):
                    print(_local_loss, _local_ce, denominator)
                
                loss = xent_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            earlystop(epoch_loss, self.state_dict())
            
            if earlystop.early_stop is True:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
                self.load_state_dict(earlystop.state_dict)
                break
            if epoch % 10 == 0 and verbose:
                print("Epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")
            
            # session.report({'loss_curve': epoch_loss})

        return epoch_loss


class ExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, num_users, num_items, embedding_k=30, max_iter=10, device='cpu', **kwargs):

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.max_iter = max_iter
        self.device = torch.device(device)
        self.init_std = 1e-2
        self.init_mu = 1e-2
        self.n_jobs = 4

        self.lam_theta = 1e-5
        self.lam_beta = 1e-5
        self.lam_y = 1.0
        
        self.a = 1.0
        self.b = 1.0

        self.W = self.init_std * np.random.randn(num_users, self.embedding_k)
        self.H = self.init_std * np.random.randn(num_items, self.embedding_k)
        self.mu = self.init_mu * np.ones(num_items, dtype=float)


    def tocsr(self, data) -> sparse.csr_matrix:
        matrix = sparse.lil_matrix((self.num_users, self.num_items))
        for (u, i, r) in data[:, :3]:
            matrix[u, i] = r
        return sparse.csr_matrix(matrix) 
    
    def get_row(self, Y, i):
        '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
        non-zero values in i_th row'''
        lo, hi = Y.indptr[i], Y.indptr[i + 1]
        return Y.data[lo:hi], Y.indices[lo:hi]


    def a_row_batch(self, Y_batch, theta_batch, beta, lam_y, mu):
        '''Compute the posterior of exposure latent variables A by batch'''
        pEX = sqrt(lam_y / 2 * np.pi) * \
            np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
        A = (pEX + 1e-8) / (pEX + 1e-8 + (1 - mu) / mu)
        A[Y_batch.nonzero()] = 1.
        return A


    def _solve(self, k, A_k, X, Y, f, lam, lam_y, mu):
        '''Update one single factor'''
        s_u, i_u = self.get_row(Y, k)
        a = np.dot(s_u * A_k[i_u], X[i_u])
        B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
        return np.linalg.solve(B, a)


    def _solve_batch(self, lo, hi, X, X_old_batch, Y, m, f, lam, lam_y, mu):
        '''Update factors by batch, will eventually call _solve() on each factor to
        keep the parallel process busy'''
        assert X_old_batch.shape[0] == hi - lo

        if mu.size == X.shape[0]:  # update users
            A_batch = self.a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu)
        else:  # update items
            A_batch = self.a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu[lo:hi,
                                                                    np.newaxis])

        X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)
        for ib, k in enumerate(np.arange(lo, hi)):
            X_batch[ib] = self._solve(k, A_batch[ib], X, Y, f, lam, lam_y, mu)
        return X_batch


    def recompute_factors(self, X, X_old, Y, lam, lam_y, mu, n_jobs, batch_size=1000):
        '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
        ridge term lam by embarrassingly parallelization. All the comments below
        are in the view of computing user factors'''
        m, n = Y.shape  # m = number of users, n = number of items
        assert X.shape[0] == n
        assert X_old.shape[0] == m
        f = X.shape[1]  # f = number of factors

        start_idx = np.arange(0, m, batch_size).tolist()
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs=n_jobs)(delayed(self._solve_batch)(
            lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
            for lo, hi in zip(start_idx, end_idx))
        X_new = np.vstack(res)
        return X_new

    def fit(self, x, y, batch_size=1024, **kwargs):

        X = self.tocsr(np.c_[x, y])
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf
        for i in range(self.max_iter):
            self.W = self.recompute_factors(self.H, self.W, X,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=batch_size)
            self.H = self.recompute_factors(self.W, self.H, XT,
                                      self.lam_beta / self.lam_y,
                                      self.lam_y,
                                      self.mu,
                                      self.n_jobs,
                                      batch_size=batch_size)
            start_idx = np.arange(0, self.num_users-1, batch_size).tolist() # 1:15400-1
            end_idx = start_idx[1:] + [self.num_users] # 2:15400

            A_sum = np.zeros_like(self.mu)
            for lo, hi in zip(start_idx, end_idx):
                A_sum += self.a_row_batch(X[lo:hi], self.W[lo:hi], self.H,
                                    self.lam_y, self.mu).sum(axis=0)
            self.mu = (self.a + A_sum - 1) / (self.a + self.b + self.num_users - 2)
        return np.array(0.0)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)
        # out = self.sigmoid(out)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    def predict(self, x):

        user_idx = torch.LongTensor(x[:,0]).cpu()
        item_idx = torch.LongTensor(x[:,1]).cpu()
        U_emb = self.W[user_idx]
        V_emb = self.H[item_idx]
        pred = np.sum(U_emb*V_emb, 1)
        pred = 1 / (1 + np.exp(-pred)) # sigmoid
        return pred


class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        print(kwargs)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    def forward_MCMC(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.dropout(self.W(user_idx))
        V_emb = self.dropout(self.H(item_idx))

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class MF_BaseModel_ui(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=8, *args, **kwargs):
        super(MF_BaseModel_ui, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, y, is_training=True):
        user_idx = torch.LongTensor(x).cuda()
        item_idx = torch.LongTensor(y).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()
    
    
class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=8, device='cpu', *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k, device=self.device)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k, device=self.device)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True, device=self.device)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.sigmoid(self.linear_1(z_emb))

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()
    
class Embedding_Sharing(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=8, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

#         out = self.linear_2(h1)

        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)    
    
class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=8, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")        

    
    def fit(self, x, y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        
        num_sample = len(x)
        total_batch = num_sample // self.batch_size
#         x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0              
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                #inv_prop = torch.squeeze(inv_prop).detach()

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)
#                 pred = self.sigmoid(pred)
                #print(pred)
                #print(sub_y)
                #print(inv_prop)
                #print(pred.shape)
                #print(sub_y.shape)
                #print(inv_prop.shape)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()        

class MF_ASIPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=8, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction1_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction2_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS-PS] Reach preset epochs, it seems does not converge.")        

    
    def fit(self, x, y, gamma, tao, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer_prediction1 = torch.optim.Adam(
            self.prediction1_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction2 = torch.optim.Adam(
            self.prediction2_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction1_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction1.zero_grad()
                loss.backward()
                optimizer_prediction1.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred1] Reach preset epochs, it seems does not converge.")

        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.prediction2_model.forward(sub_x, True)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer_prediction2.zero_grad()
                loss.backward()
                optimizer_prediction2.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred2] Reach preset epochs, it seems does not converge.")
        
        early_stop = 0
        last_loss = 1e9
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                pred_u1 = self.prediction1_model.forward(x_sampled)
                pred_u2 = self.prediction2_model.forward(x_sampled)

                x_sampled_common = x_sampled[(pred_u1.detach().cpu().numpy() - pred_u2.detach().cpu().numpy()) < tao]

                pred_u3 = self.prediction_model.forward(x_sampled_common)

                sub_y = self.prediction1_model.forward(x_sampled_common)
                #print(sub_y)
                #sub_y = torch.Tensor(sub_y).cuda()
                
                xent_loss = F.binary_cross_entropy(pred_u3, sub_y.detach())

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()    
    
class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=8, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        #self.logistic_model = LogisticRegression().cuda()

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS-PS] Reach preset epochs, it seems does not converge.")        
                
    def fit(self, x, y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                #inv_prop = torch.squeeze(inv_prop).detach()

                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop, reduction = "sum")
                
                xent_loss = xent_loss / (torch.sum(inv_prop))

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy() # 得到数字

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()        
    
    
    
    
class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, device='cpu', batch_size=1024, batch_size_prop=1024, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop

        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, device=self.device, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, device=self.device, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs, device=self.device)
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, prior_y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        #for param in self.parameters():
            #print(param, param.shape)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items) # list 有 290*300元素 

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size
        
        prior_y = prior_y.mean()
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y, device=self.device)

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  
                #print(x_all)
                #print(x_all.shape[0])
                #print(x_all[0])
                x_sampled = x_all[ul_idxs[G * idx* self.batch_size: G * (idx+1)*self.batch_size]] # batch size

                pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                
                #print(self.state_dict())
                #print(self.W.weight.shape)
                #print(H.weight.shape)
                
                #print(selected_idx)
                
                imputation_y = torch.Tensor([prior_y]* G *selected_idx.shape[0], device=self.device)
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="sum") # e^ui

                ips_loss = (xent_loss - imputation_loss) # batch size

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="sum") # 290*300/total_batch个

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

            return epoch_loss
    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()
    
    

"""        
        for name, param in self.prediction_model.named_parameters():
            print(name,param)
        print()
        print()
        for name, param in self.imputation.named_parameters():
            print(name,param)
        print()
        print()
        for name, param in self.named_parameters():
            print(name,param)
"""


class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, device='cpu', batch_size=1024, batch_size_prop=1024, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop

        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, device=self.device, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, device=self.device)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, device=self.device, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, verbose=False):
        
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs, device=self.device)
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DRJL-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 


        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch): 
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                sub_y = torch.Tensor(sub_y, device=self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)           
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)

              #  for name, param in self.named_parameters():
              #          print(name, param.grad)                
                
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                
              #  for name, param in self.named_parameters():
              #          print(name, param.grad)
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                                
                # direct loss                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                
                # propensity loss
                
                #pred_prop = 1/(one_over_zl)
                #print(pred_prop)
                #print(prediction)
                #prediction = torch.squeeze(prediction)
                #prop_loss = F.binary_cross_entropy(pred_prop, prediction)
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]
               # print(prop_loss)

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                     
               # for i in range(prop_update):
               # for name, param in self.named_parameters():
               #     print(name, param.grad)                                       
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).to(self.device)
                imputation_y = self.imputation_model.forward(sub_x)

                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")

        return epoch_loss
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()


class MF_CDR_JL(MF_DR_JL):
    
    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, verbose=False):
        
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs, device=self.device)
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DRJL-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, eta_threshold=0.1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch): 
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                sub_y = torch.Tensor(sub_y, device=self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)           
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)

                confidence = []
                for _ in range(10):
                    confidence.append(self.imputation_model.forward_MCMC(x_sampled).detach().cpu().numpy())
                confidence = np.array(confidence)
                mean, std = confidence.mean(axis=0), confidence.std(axis=0)
                eta = std / mean

                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum((imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6)
                # print(eta)
                direct_loss = direct_loss[eta < eta_threshold]
                direct_loss = -torch.sum(direct_loss)
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                                    
                epoch_loss += xent_loss.detach().cpu().numpy()         


                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)       
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")

        return epoch_loss
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
#         one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
#                 inv_prop = one_over_zl_obs[selected_idx].detach()
                #inv_prop = torch.squeeze(inv_prop).detach()
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
#                 print(min(prop))
#                 prop_loss = F.binary_cross_entropy(prop, sub_obs)
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDRJL-PS] Reach preset epochs, it seems does not converge.")        


    def fit(self, x, y, stop = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch): 
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)             
                
                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()

              #  for name, param in self.named_parameters():
              #          print(name, param.grad)                
          
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                
              #  for name, param in self.named_parameters():
              #          print(name, param.grad)
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                
                # direct loss
                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                 
                # propensity loss
                
                #pred_prop = 1/(one_over_zl)
                #print(pred_prop)
                #print(prediction)
                #prediction = torch.squeeze(prediction)
                #prop_loss = F.binary_cross_entropy(pred_prop, prediction)
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]
               # print(prop_loss)

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                     
               # for i in range(prop_update):
               # for name, param in self.named_parameters():
               #     print(name, param.grad)                                       
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2
                            ) * (inv_prop.detach())**2 *(1-1/inv_prop.detach())).sum()   

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")
                
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()            
        
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)


class MF_Multi_IPS(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.embedding_sharing = Embedding_Sharing(self.num_users, self.num_items, self.embedding_k)
        self.propensity_model = MLP(input_size = 2 * embedding_k)
        self.prediction_model = MLP(input_size = 2 * embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
#         obs = sps.csr_matrix((1, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
#         y = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        # generate all counterfactuals and factuals
#         x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                commom_emb = self.embedding_sharing.forward(sub_x)
                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(commom_emb), gamma, 1)
                #inv_prop = torch.squeeze(inv_prop).detach()

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(commom_emb)

                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
#                 print(xent_loss)
#                 print(min(pred))
#                 print(max(pred))
                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Multi-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Multi-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Multi-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.embedding_sharing.forward(x)
        pred = self.prediction_model.forward(pred)
        return pred.detach().cpu().numpy()        

    
class MF_Multi_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.embedding_sharing = Embedding_Sharing(self.num_users, self.num_items, self.embedding_k)
        self.propensity_model = MLP(input_size = 2 * embedding_k)
        self.prediction_model = MLP(input_size = 2 * embedding_k)
        self.imputation_model = MLP(input_size = 2 * embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, G = 4,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
#         obs = sps.csr_matrix((1, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
#         y = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                commom_emb = self.embedding_sharing.forward(sub_x)
                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(commom_emb), gamma, 1)
                #inv_prop = torch.squeeze(inv_prop).detach()

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(commom_emb)
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                
                imputation_y = self.imputation_model.forward(commom_emb)                
                imputation_loss = -torch.sum(pred * torch.log(imputation_y + 1e-6) + (1-pred) * torch.log(1 - imputation_y + 1e-6))
                
                ips_loss = xent_loss - imputation_loss
                
                x_all_idx = ul_idxs[G * idx * self.batch_size : G * (idx+1) * self.batch_size]
                x_sampled = x_all[x_all_idx]
                
                commom_emb_u = self.embedding_sharing.forward(x_sampled)
                pred_u = self.prediction_model.forward(commom_emb_u)
                imputation_y1 = self.imputation_model.forward(commom_emb_u)
                
                direct_loss = -torch.sum(pred_u * torch.log(imputation_y1 + 1e-6) + (1-pred_u) * torch.log(1 - imputation_y1 + 1e-6))
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Multi-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Multi-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Multi-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.embedding_sharing.forward(x)
        pred = self.prediction_model.forward(pred)
        return pred.detach().cpu().numpy()       
    
    
class MF_ESMM(nn.Module):
    def __init__(self, num_users, num_items, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, alpha = 1, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(obs)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch): 
            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size : (idx+1) * self.batch_size]
                x_sampled = x_all[x_all_idx]
                
                # ctr loss
                
                prop = torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_y = torch.Tensor(y[x_all_idx]).cuda()
                
                prop_loss = F.binary_cross_entropy(prop, sub_obs)                                    
                
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(prop * pred, sub_y)                          
                
                loss = alpha * prop_loss + pred_loss

                optimizer_prediction.zero_grad()
                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                optimizer_propensity.step()
                     
               # for i in range(prop_update):
               # for name, param in self.named_parameters():
               #     print(name, param.grad)                                       
                epoch_loss += loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESMM] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESMM] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESMM] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    
    
class MF_ESCM2(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)        
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()             
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum((imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6))
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                                                  
                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = -sub_y * torch.log(pred + 1e-6) - (1-sub_y) * torch.log(1 - pred + 1e-6)
                e_hat_loss = -imputation_y * torch.log(pred + 1e-6) - (1-imputation_y) * torch.log(1 - pred + 1e-6)
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
#                 print(sub_obs)
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
#                 print(inv_prop_all.shape)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
#                 print(prop_loss.shape)
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)
                
                loss = alpha * prop_loss + beta * pred_loss + theta * imp_loss + dr_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()