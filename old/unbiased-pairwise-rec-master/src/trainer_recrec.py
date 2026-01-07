from asyncio import FastChildWatcher
from cmath import exp
import imp
import json
import time
from pathlib import Path
from token import CIRCUMFLEX
from typing import Dict, List, Optional, Tuple
from urllib.request import UnknownHandler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import data
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from models.recommenders_recrec import ReCRec_I,ReCRec_D,ReCRec_F


def sigmoid(x):
    return 1./(1+np.exp(-x))

def ReCRec_trainer(sess: tf.Session, model,
                      train: np.ndarray,exposure: np.ndarray, test: np.ndarray, vad: np.ndarray, pscore: np.ndarray,item_freq :np.ndarray,
                      max_iters:int , batch_size: int, model_name: str ) -> Tuple[np.ndarray, np.ndarray,np.ndarray]:
    
    vad_metrics = ['Recall@1','MAP@1','NDCG@1']
    metrics5 = ['Recall@5','MAP@5','NDCG@5']
    best_iter = 0
    max_interval  = 300
    max_sum = 0.
    best_ib = None
    best_mu = None
        
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    
    train = train[np.lexsort((train[:,0],train[:,1]))]
    exposure = exposure[np.lexsort((exposure[:,0],exposure[:,1]))]
    pscore = pscore[train[:, 1].astype(np.int)]
    pos_train = train[train[:, 2] == 1]
    pscore_pos_train = pscore[train[:, 2] == 1]
    exposure_pos_train = exposure[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    pscore_unlabeled_train = pscore[train[:, 2] == 0]
    exposure_unlabeled_train = exposure[train[:, 2] == 0]
    num_unlabeled = np.sum(1 - train[:, 2])
    
    np.random.seed(12345)
    for i in np.arange(max_iters):
        #pos_idx = np.random.choice(np.arange(num_pos), size=batch_size//2)
        pos_idx = np.arange(num_pos)
        unlabeled_idx = np.random.choice(np.arange(num_unlabeled), size=num_pos*4)
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unlabeled_idx]]
        train_label = train_batch[:, 2]
        pscore_ = np.r_[pscore_pos_train[pos_idx],
                        pscore_unlabeled_train[unlabeled_idx]]
        exposure_ = np.r_[exposure_pos_train[pos_idx],
                        exposure_unlabeled_train[unlabeled_idx]]
        
        train_score = pscore_ 

        _,_,loss_gama,loss_mu,gama,mu = sess.run([model.apply_grads_gama,model.apply_grads_mu,model.loss_gama,model.loss_mu,model.gama,model.mu],
                        feed_dict={model.users: train_batch[:, 0],
                                    model.items: train_batch[:, 1],
                                    model.labels: np.expand_dims(train_label, 1),
                                    model.exposure: np.expand_dims(exposure_[:,2], 1),
                                    model.scores: np.expand_dims(train_score, 1)})
        
        if i % 20 == 0:

            u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
            u_bias = None
            i_bias = None
            u_bias, i_bias = sess.run([model.user_bias_rel, model.item_bias_rel])
            
            ret = aoa_evaluator( user_embed=u_emb,item_embed=i_emb, model_name=model_name, test=vad, at_k=[1],user_bias = u_bias,item_bias = i_bias )
            
            #ret = aoa_evaluator( user_embed=u_emb,item_embed=i_emb, model_name=model_name, test=test, at_k=[5],user_bias = u_bias,item_bias = i_bias )
           
            sum = 0.
            for metric in vad_metrics:
                sum = sum + ret.loc[metric,model_name]
            
            if sum > max_sum:
                best_iter = i
                best_uemb = u_emb
                best_iemb = i_emb
                best_ib = i_bias
                max_sum = sum
                
            if i - best_iter > max_interval:
                break


    sess.close()

    return best_uemb, best_iemb,best_ib

class Trainer:
    """Trainer Class for ImplicitRecommender."""

    def __init__(self, max_iters: int, lam: float,lamp: float, batch_size: int, eta: float , model_name: str, dataset_name: str, dim : int) -> None:
        self.dim = dim
        self.lam = lam
        self.lamp = lamp
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rec = {}


    def run(self) -> None:
        

        train = np.load(f'../data/{self.dataset_name}_st_1/point/train.npy',allow_pickle=True)
        # exposure = np.load(f'../data/{self.dataset_name}_st_1/point/exposure.npy',allow_pickle=True)#有大问题！
        vad = np.load(f'../data/{self.dataset_name}_st_1/point/val.npy',allow_pickle=True)
        test = np.load(f'../data/{self.dataset_name}_st_1/point/test.npy',allow_pickle=True)
        pscore = np.load(f'../data/{self.dataset_name}_st_1/point/pscore.npy',allow_pickle=True)
        item_freq = np.load(f'../data/{self.dataset_name}_st_1/point/item_freq.npy',allow_pickle=True)
        num_users = np.int(train[:, 0].max() + 1)
        num_items = np.int(train[:, 1].max() + 1)
        
        np.random.seed(12345)
        ############################3
        exposure = train.copy()
        # Find indices of zeros in the third column
        zero_indices = np.where(exposure[:, 2] == 0)[0]
        # Randomly select 100 indices
        random_indices = np.random.choice(zero_indices, size=100, replace=False)
        # Set those 100 zeros to ones
        exposure[random_indices, 2] = 1
        ############################3
        
        
        
        
        tf.set_random_seed(12345)
        ops.reset_default_graph()
        
        sess = tf.Session()


        if  self.model_name == 'ReCRec-I':
            model = ReCRec_I(
                num_users=num_users, num_items=num_items,pscore=pscore,
                 dim=self.dim, lam=self.lam, eta=self.eta)
            
        elif self.model_name == 'ReCRec-F':
            model = ReCRec_F(
                num_users=num_users, num_items=num_items,pscore=pscore,
                 dim=self.dim, lam=self.lam, lamp=self.lamp,eta=self.eta) 

        else :
            model = ReCRec_D(
                num_users=num_users, num_items=num_items,pscore=pscore,
                 dim=self.dim, lam=self.lam, eta=self.eta)
        
        best_uemb, best_iemb, best_ib = ReCRec_trainer(
                sess, model=model, train=train,exposure = exposure, test=test,vad= vad, pscore=pscore, item_freq= item_freq,
                max_iters=self.max_iters, batch_size=2**self.batch_size,
                model_name=self.model_name)
        self.ret = aoa_evaluator( user_embed=best_uemb,item_embed=best_iemb, model_name=self.model_name, test=test, at_k = [3, 5, 8],item_bias=best_ib)
        ret_path = Path(f'../logs/{self.dataset_name}_st_1/{self.model_name}/results')
        ret_path.mkdir(parents=True, exist_ok=True)
        pd.concat([self.ret], 1).to_csv(ret_path / f'aoa_all.csv')
    # def get_performance_metric(self):
    #     """Return the desired performance metric for optimization."""
    #     # Example: Return the mean of a specific metric (modify according to the actual evaluation results structure)
    #     return self.rec['final_results'].mean().values  # Modify as per actual metric

    def get_performance_metric(self):
        """Return the desired performance metric for optimization."""
        # 假设你希望优化 DCG@3
        metric_to_optimize = 'NDCG@3'  # 指定你要优化的指标

        # 确保 'DCG@3' 存在于 DataFrame 的 index 中
        if metric_to_optimize in self.ret.index:
            # 返回该指标的平均值
            return self.ret.loc[metric_to_optimize].mean()  # 根据索引选择行
        else:
            raise ValueError(f"Metric {metric_to_optimize} not found in final results.")
    
        
