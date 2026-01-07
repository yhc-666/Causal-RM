"""
Codes for training recommenders used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import Tuple
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import sparse
from tensorflow.python.framework import ops
from trainer_cjmf import cjmf_trainer
from trainer_BISER import ae_trainer
from evaluate.Ours_evaluate import aoa_evaluator as ours_evaluator
from evaluate.evaluator import aoa_evaluator
from models.expomf import ExpoMF
from models.recommenders import PairwiseRecommender, PointwiseRecommender, UPLPairwiseRecommender, PairwiseRecommender_ours, PointwiseRecommender_ours
from models.Ours import Ours
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
def train_expomf(data: str, train: np.ndarray, num_users: int, num_items: int,
                 n_components: int = 100, lam: float = 1e-6) -> Tuple:
    """Train the expomf model."""
    def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
        """Convert data to csr_matrix."""
        matrix = sparse.lil_matrix((num_users, num_items))
        for (u, i, r) in data[:, :3]:
            matrix[u, i] = r
        return sparse.csr_matrix(matrix)

    path = Path(f'../logs/{data}_st_{self.sample_times}/expomf/emb')
    path.mkdir(parents=True, exist_ok=True)
    model = ExpoMF(n_components=n_components, random_state=12345, save_params=False,
                   early_stopping=True, verbose=False, lam_theta=lam, lam_beta=lam)
    model.fit(tocsr(train, num_users, num_items))
    np.save(file=str(path / 'user_embed.npy'), arr=model.theta)
    np.save(file=str(path / 'item_embed.npy'), arr=model.beta)

    return model.theta, model.beta

def train_ours(sess: tf.Session, model: Ours, save_path: str,
               labeled_train: np.ndarray, labeled_val: np.ndarray,
               max_iters: int = 1000, batch_size: int = 1024,
               model_name: str = 'Ours', is_optuna: bool = False) -> Tuple:
    """Train and evaluate pairwise recommenders."""
    labeled_data = labeled_train
    train_loss_list = []
    val_loss_list = []
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # # 初始化特定的变量


    # 创建一个 Saver 对象，用于保存模型
    saver = tf.train.Saver()

    best_val_loss = float('inf')  # 初始时，验证集的最小损失设为无穷大
    
    
    # Early stopping parameters
    last_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    
    # 训练循环
    for epoch in range(max_iters):
        labeled_data = np.array(labeled_data)
        # Shuffle the labeled_data using numpy
        np.random.shuffle(labeled_data)

        total_train_loss = 0.0
        total_val_loss = 0.0
        
        # Training on the training set
        pbar = tqdm(total=len(labeled_data), desc ="train:")
        for start in range(0, len(labeled_data), batch_size):
            pbar.update(batch_size)
            end = min(start + batch_size, len(labeled_data))
            batch_data = labeled_data[start:end]  # Direct slicing of labeled_data

            user_ids = batch_data[:, 0]
            item_ids = batch_data[:, 1]
            labels = batch_data[:, 2]
            R_labels = batch_data[:, 3]
            O_labels = batch_data[:, 4]

            feed_dict = {
                model.user_input: user_ids,
                model.item_input: item_ids,
                model.Y_labels: labels,
                model.R_labels: R_labels,
                model.O_labels: O_labels
            }

            # loss = self.compute_loss()

            # l_rate = self.eta
            # optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)

            # minimize = optimizer.minimize(loss)

            # prediction, train_loss, _ = self.sess.run([self.prediction, loss, minimize], feed_dict=feed_dict)
            _, train_loss = sess.run(
                [model.apply_grads, model.loss], 
                feed_dict=feed_dict
            )

            total_train_loss += train_loss
            
            val_loss = 0.0
            # 计算整个验证集上的损失
            for start in range(0, len(labeled_val), batch_size):
                end = min(start + batch_size, len(labeled_val))
                batch_data_val = np.array(labeled_val[start:end])  # Convert to NumPy array

                user_ids_val = batch_data_val[:, 0]
                item_ids_val = batch_data_val[:, 1]
                labels_val = batch_data_val[:, 2]
                R_labels_val = batch_data_val[:, 3]
                O_labels_val = batch_data_val[:, 4]

                feed_dict_val = {
                    model.user_input: user_ids_val,
                    model.item_input: item_ids_val,
                    model.Y_labels: labels_val,
                    model.R_labels: R_labels_val,
                    model.O_labels: O_labels_val
                }

                # 计算验证损失（不进行优化）
                val_loss_0 = sess.run(model.unbiased_loss, feed_dict=feed_dict_val)
                val_loss += val_loss_0
            total_val_loss += val_loss

        # 打印训练集和验证集的损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: Training Loss = {total_train_loss:.4f}, Validation Loss = {total_val_loss:.4f}")

        # 如果验证损失比之前的最小值小，则保存当前模型
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            saver.save(sess, save_path)  # 保存当前模型的参数
            print(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
        if total_val_loss < last_val_loss:
            last_val_loss = total_val_loss
        else:
            if epoch > 10:
                patience_counter += 1
            last_val_loss = total_val_loss
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    return best_val_loss


def train_pointwise(sess: tf.Session, model: PointwiseRecommender, data: str,
                    train: np.ndarray, val: np.ndarray, test: np.ndarray, pscore: np.ndarray,
                    max_iters: int = 1000, batch_size: int = 256,
                    model_name: str = 'wmf', is_optuna: bool = False) -> Tuple:
    """Train and evaluate implicit recommender."""
    train_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ips = 'relmf' in model_name
    # pscore for train
    pscore = pscore[train[:, 1].astype(int)]
    # positive and unlabeled data for training set
    pos_train = train[train[:, 2] == 1]
    pscore_pos_train = pscore[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    pscore_unlabeled_train = pscore[train[:, 2] == 0]
    num_unlabeled = np.sum(1 - train[:, 2])
    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        # positive mini-batch sampling
        # the same num. of postive and negative samples are used in each batch
        sample_size = np.int(batch_size / 2)
        pos_idx = np.random.choice(np.arange(num_pos), size=sample_size)
        unl_idx = np.random.choice(np.arange(num_unlabeled), size=sample_size)
        # mini-batch samples
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unl_idx]]
        pscore_ = np.r_[pscore_pos_train[pos_idx], pscore_unlabeled_train[unl_idx]] if ips else np.ones(batch_size)
        # update user-item latent factors and calculate training loss
        _, train_loss = sess.run([model.apply_grads, model.unbiased_loss],
                                 feed_dict={model.users: train_batch[:, 0],
                                            model.items: train_batch[:, 1],
                                            model.labels: np.expand_dims(train_batch[:, 2], 1),
                                            model.scores: np.expand_dims(pscore_, 1)})
        train_loss_list.append(train_loss)
    # calculate a validation score
    unl_idx = np.random.choice(np.arange(num_unlabeled), size=val.shape[0])
    val_batch = np.r_[val, unlabeled_train[unl_idx]]
    pscore_ = np.r_[pscore[val[:, 1].astype(int)], pscore_unlabeled_train[unl_idx]]
    val_loss = sess.run(model.unbiased_loss,
                        feed_dict={model.users: val_batch[:, 0],
                                   model.items: val_batch[:, 1],
                                   model.labels: np.expand_dims(val_batch[:, 2], 1),
                                   model.scores: np.expand_dims(pscore_, 1)})

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    if ~is_optuna:
        path = Path(f'../logs/{data}_st_{self.sample_times}/{model_name}')
        (path / 'loss').mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / 'loss/train.npy'), arr=train_loss_list)
        np.save(file=str(path / 'loss/test.npy'), arr=test_loss_list)
        (path / 'emb').mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / 'emb/user_embed.npy'), arr=u_emb)
        np.save(file=str(path / 'emb/item_embed.npy'), arr=i_emb)
    sess.close()

    return u_emb, i_emb, val_loss


def train_pointwise_ours(sess: tf.Session, model: PointwiseRecommender_ours, data: str,
                    train: np.ndarray, val: np.ndarray, test: np.ndarray, train_pairwise: np.ndarray, val_pairwise: np.ndarray, pscore: np.ndarray,
                    max_iters: int = 1000, batch_size: int = 256,
                    model_name: str = 'wmf', is_optuna: bool = False) -> Tuple:
    """Train and evaluate implicit recommender."""
    train_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ips = 'relmf' in model_name
    # pscore for train
    pscore = pscore[train[:, 1].astype(int)]
    # positive and unlabeled data for training set
    pos_train = train[train[:, 2] == 1]
    pscore_pos_train = pscore[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    pscore_unlabeled_train = pscore[train[:, 2] == 0]
    num_unlabeled = np.sum(1 - train[:, 2])

    num_pairwise = len(train_pairwise)
    # batch_num_pair = num_pairwise // batch_size

    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        # positive mini-batch sampling
        # the same num. of postive and negative samples are used in each batch
        sample_size = np.int(batch_size / 2)
        pos_idx = np.random.choice(np.arange(num_pos), size=sample_size)
        unl_idx = np.random.choice(np.arange(num_unlabeled), size=sample_size)
        pair_idx = np.random.choice(np.arange(num_pairwise), size=sample_size)
        # mini-batch samples
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unl_idx]]
        # print(sum(train_batch[:, 2] < 0)) # 0
        train_batch_pair = train_pairwise[pair_idx]
        pscore_ = np.r_[pscore_pos_train[pos_idx], pscore_unlabeled_train[unl_idx]] if ips else np.ones(batch_size)
        # update user-item latent factors and calculate training loss
        _, train_loss = sess.run([model.apply_grads, model.loss],
                                 feed_dict={model.users: train_batch[:, 0],
                                            model.items: train_batch[:, 1],
                                            model.labels: np.expand_dims(train_batch[:, 2], 1),
                                            model.users_pair: train_batch_pair[:, 0],
                                            model.pos_items: train_batch_pair[:, 1],
                                            model.items2: train_batch_pair[:, 2],
                                            model.point_R: np.expand_dims(train_batch[:, 3], 1),
                                            model.point_O: np.expand_dims(train_batch[:, 4], 1),                                            
                                            model.scores: np.expand_dims(pscore_, 1)})
        train_loss_list.append(train_loss)
    # calculate a validation score
    # print(train_loss_list)
    unl_idx = np.random.choice(np.arange(num_unlabeled), size=val.shape[0])
    val_batch = np.r_[val, unlabeled_train[unl_idx]]  

    pair_idx = np.random.choice(np.arange(len(val_pairwise)), size=min(sample_size, len(val_pairwise)))
    val_batch_pair = val_pairwise[pair_idx]
    # print(sum(val_batch[:, 2] < 0)) # 0
    pscore_ = np.r_[pscore[val[:, 1].astype(int)], pscore_unlabeled_train[unl_idx]]
    # print(pscore_)
    val_loss = sess.run(model.unbiased_loss,
                        feed_dict={model.users: val_batch[:, 0],
                                   model.items: val_batch[:, 1],
                                   model.labels: np.expand_dims(val_batch[:, 2], 1),
                                   model.users_pair: val_batch_pair[:, 0],
                                   model.pos_items: val_batch_pair[:, 1],
                                   model.items2: val_batch_pair[:, 2],
                                   model.point_R: np.expand_dims(val_batch[:, 3], 1),
                                   model.point_O: np.expand_dims(val_batch[:, 4], 1),                                     
                                   model.scores: np.expand_dims(pscore_, 1)})

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    # if ~is_optuna:
    #     path = Path(f'../logs/{data}_st_{self.sample_times}/{model_name}')
    #     (path / 'loss').mkdir(parents=True, exist_ok=True)
    #     np.save(file=str(path / 'loss/train.npy'), arr=train_loss_list)
    #     np.save(file=str(path / 'loss/test.npy'), arr=test_loss_list)
    #     (path / 'emb').mkdir(parents=True, exist_ok=True)
    #     np.save(file=str(path / 'emb/user_embed.npy'), arr=u_emb)
    #     np.save(file=str(path / 'emb/item_embed.npy'), arr=i_emb)
    sess.close()
    print(val_loss)
    return u_emb, i_emb, val_loss

def train_pairwise(sess: tf.Session, model: PairwiseRecommender, data: str,
                   train: np.ndarray, val: np.ndarray, test: np.ndarray,
                   max_iters: int = 1000, batch_size: int = 1024,
                   model_name: str = 'bpr', is_optuna: bool = False) -> Tuple:
    """Train and evaluate pairwise recommenders."""
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data.
    num_train, num_val = train.shape[0], val.shape[0]
    np.random.seed(12345)
    
    # Early stopping parameters
    last_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    total_batch = int(num_train / batch_size)
    for i in np.arange(max_iters):
        idx = np.arange(num_train)
        np.random.shuffle(idx)
        
        
        val_loss = 0
        for j in np.arange(total_batch):
            train_batch = train[idx[j * batch_size: (j + 1) * batch_size]]
            # update user-item latent factors
            if model_name in 'bpr':
                _, loss = sess.run([model.apply_grads, model.loss],
                                feed_dict={model.users: train_batch[:, 0],
                                            model.pos_items: train_batch[:, 1],
                                            model.scores1: np.ones((batch_size, 1)),
                                            model.items2: train_batch[:, 2],
                                            model.labels2: np.zeros((batch_size, 1)),
                                            model.scores2: np.ones((batch_size, 1))})
            elif 'ubpr' in model_name or 'upl_bpr' in model_name:
                _, loss = sess.run([model.apply_grads, model.loss],
                                feed_dict={model.users: train_batch[:, 0],
                                            model.pos_items: train_batch[:, 1],
                                            model.scores1: np.expand_dims(train_batch[:, 4], 1),
                                            model.items2: train_batch[:, 2],
                                            model.labels2: np.expand_dims(train_batch[:, 3], 1),
                                            model.scores2: np.expand_dims(train_batch[:, 5], 1)})
            train_loss_list.append(loss)
            # calculate a validation loss
            if model_name in 'bpr':
                val_loss_0 = sess.run(model.unbiased_loss,
                                    feed_dict={model.users: val[:, 0],
                                            model.pos_items: val[:, 1],
                                            model.scores1: np.ones((num_val, 1)),
                                            model.items2: val[:, 2],
                                            model.labels2: np.zeros((num_val, 1)),
                                            model.scores2: np.ones((num_val, 1))})
            elif 'ubpr' in model_name or 'upl_bpr' in model_name:
                val_loss_0 = sess.run(model.unbiased_loss,
                                    feed_dict={model.users: val[:, 0],
                                            model.pos_items: val[:, 1],
                                            model.scores1: np.expand_dims(val[:, 4], 1),
                                            model.items2: val[:, 2],
                                            model.labels2: np.expand_dims(val[:, 3], 1),
                                            model.scores2: np.expand_dims(val[:, 5], 1)})
            val_loss += val_loss_0
            # print(val_loss_0)
        # calculate a test loss
        test_loss = sess.run(model.ideal_loss,
                             feed_dict={model.users: test[:, 0],
                                        model.pos_items: test[:, 1],
                                        model.rel1: np.expand_dims(test[:, 3], 1),
                                        model.items2: test[:, 2],
                                        model.rel2: np.expand_dims(test[:, 4], 1)})
        test_loss_list.append(test_loss)
        # calculate a validation loss

        val_loss_list.append(val_loss)
        if i % 10 == 0:
            print("epoch:", i, "val_loss:", val_loss)
        # Early stopping check
        if val_loss < last_val_loss:
            # best_val_loss = val_loss
            # patience_counter = 0
            last_val_loss = val_loss
        else:
            if i > 10:
                patience_counter += 1
            last_val_loss = val_loss
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {i}.")
                break

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    if ~is_optuna:
        path = Path(f'../logs/{data}/{model_name}')
        (path / 'loss').mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / 'loss/train.npy'), arr=train_loss_list)
        np.save(file=str(path / 'loss/val.npy'), arr=val_loss_list)
        np.save(file=str(path / 'loss/test.npy'), arr=test_loss_list)
        (path / 'emb').mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / 'emb/user_embed.npy'), arr=u_emb)
        np.save(file=str(path / 'emb/item_embed.npy'), arr=i_emb)
    sess.close()

    return u_emb, i_emb, val_loss
    

def train_pairwise_ours(sess: tf.Session, model: PairwiseRecommender_ours, data: str,
                   train: np.ndarray, val: np.ndarray, test: np.ndarray, train_point: np.ndarray, val_point: np.ndarray,
                   max_iters: int = 1000, batch_size: int = 1024,
                   model_name: str = 'bpr', is_optuna: bool = False) -> Tuple:
    """Train and evaluate pairwise recommenders."""
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data.
    temp_train1 = train_point[train_point[:, 2] == 1] # DP
    temp_train0 = train_point[(train_point[:, 2] == 0) & (train_point[:, 3] == 1) & (train_point[:, 4] == 1)] # HP
    temp_val1 = val_point[val_point[:, 2] == 1] # DP
    temp_val0 = val_point[val_point[:, 2] == 0 & (val_point[:, 3] == 1) & (val_point[:, 4] == 1)] # HP

    temp_train_point = np.r_[temp_train1, temp_train0]
    temp_val_point = np.r_[temp_val1, temp_val0]

    print(train.shape[0], temp_train_point.shape[0]) # 438, 3644
    print(val.shape[0], temp_val_point.shape[0]) # 92, 445

    # temp_train_all = np.c_[temp_train_pair, temp_train_point] # 8 columns, U I1 I2 U I Y R O
    # temp_val_all = np.c_[temp_val_pair, temp_val_point] # 8 columns

    num_train, num_val = train.shape[0], val.shape[0]
    num_train_point, num_val_point = temp_train_point.shape[0], temp_val_point.shape[0]
    np.random.seed(12345)
    
    # Early stopping parameters
    last_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    total_batch_pair = int(num_train / batch_size)
    batch_size_point = batch_size * len(temp_train_point) // num_train
    # total_batch_point = int(num_train_point / batch_size)
    for i in np.arange(max_iters):
        idx = np.arange(num_train)
        np.random.shuffle(idx)

        idx_all = np.arange(num_train_point)
        np.random.shuffle(idx_all)
        val_loss = 0
        for j in np.arange(total_batch_pair):
            train_batch = train[idx[j * batch_size: (j + 1) * batch_size]]
            train_batch_point = temp_train_point[idx_all[j * batch_size_point: (j + 1) * batch_size_point]]
            # update user-item latent factors
            if model_name in 'ours1_bpr':
                _, loss = sess.run([model.apply_grads, model.loss],
                                feed_dict={model.users: train_batch[:, 0],
                                            model.pos_items: train_batch[:, 1],
                                            model.scores1: np.ones((batch_size, 1)),
                                            model.items2: train_batch[:, 2],
                                            model.point_users: train_batch_point[:, 0],
                                            model.point_items: train_batch_point[:, 1],
                                            model.point_labels: np.expand_dims(train_batch_point[:, 2], 1),
                                            model.point_R: np.expand_dims(train_batch_point[:, 3], 1),
                                            model.point_O: np.expand_dims(train_batch_point[:, 4], 1),
                                            model.labels2: np.zeros((batch_size, 1)),
                                            model.scores2: np.ones((batch_size, 1))})
            train_loss_list.append(loss)
            # calculate a validation loss
            if model_name in 'ours1_bpr':
                val_loss_0 = sess.run(model.unbiased_loss,
                                    feed_dict={model.users: val[:, 0],
                                            model.pos_items: val[:, 1],
                                            model.scores1: np.ones((num_val, 1)),
                                            model.items2: val[:, 2],
                                            model.point_users: temp_val_point[:, 0],
                                            model.point_items: temp_val_point[:, 1],
                                            model.point_labels: np.expand_dims(temp_val_point[:, 2], 1),
                                            model.point_R: np.expand_dims(temp_val_point[:, 3], 1),
                                            model.point_O: np.expand_dims(temp_val_point[:, 4], 1),
                                            model.labels2: np.zeros((num_val, 1)),
                                            model.scores2: np.ones((num_val, 1))})
            val_loss += val_loss_0
            # print(val_loss_0)
        # calculate a test loss
        test_loss = sess.run(model.ideal_loss,
                             feed_dict={model.users: test[:, 0],
                                        model.pos_items: test[:, 1],
                                        model.rel1: np.expand_dims(test[:, 3], 1),
                                        model.items2: test[:, 2],
                                        model.rel2: np.expand_dims(test[:, 4], 1)})
        test_loss_list.append(test_loss)
        # calculate a validation loss

        val_loss_list.append(val_loss)
        if i % 10 == 0:
            print("epoch:", i, "val_loss:", val_loss)
        # Early stopping check
        if val_loss < last_val_loss:
            # best_val_loss = val_loss
            # patience_counter = 0
            last_val_loss = val_loss
        else:
            if i > 10:
                patience_counter += 1
            last_val_loss = val_loss
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {i}.")
                break

    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    # if ~is_optuna:
    #     path = Path(f'../logs/{data}/{model_name}')
    #     (path / 'loss').mkdir(parents=True, exist_ok=True)
    #     np.save(file=str(path / 'loss/train.npy'), arr=train_loss_list)
    #     np.save(file=str(path / 'loss/val.npy'), arr=val_loss_list)
    #     np.save(file=str(path / 'loss/test.npy'), arr=test_loss_list)
    #     (path / 'emb').mkdir(parents=True, exist_ok=True)
    #     np.save(file=str(path / 'emb/user_embed.npy'), arr=u_emb)
    #     np.save(file=str(path / 'emb/item_embed.npy'), arr=i_emb)
    sess.close()

    return u_emb, i_emb, val_loss    


class Trainer:

    suffixes = ['cold-user', 'rare-item']
    at_k = [3, 5, 8]
    cold_user_threshold = 6
    rare_item_threshold = 100

    def __init__(self, data: str, max_iters: int = 1000, batch_size: int = 12,
                 eta: float = 0.1, model_name: str = 'bpr', params: dict = None) -> None:
        """Initialize class for different models."""

        self.data = data
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        # self.wd = wd
        self.metrics = {}  # Dictionary to store performance metrics
        # print(params)
        if model_name not in ['expomf', 'cjmf', 'BISER','Ours', 'ours1_bpr', 'ours2_wmf']:
            # Load general hyper-parameters from 'hyper_params.yaml'
            if params is None:
                hyper_params = yaml.safe_load(open(f'../conf/hyper_params.yaml', 'r'))[f'{data}1'][model_name]
            else:
                hyper_params = params  # Use external params if provided
            self.dim = int(hyper_params['dim'])
            self.lam = hyper_params['lam']
            self.eta = hyper_params['eta']
            self.weight = hyper_params['weight'] if model_name == 'wmf' else 1.
            self.clip = hyper_params['clip'] if model_name == 'relmf' else 0.
            self.beta = hyper_params['beta'] if 'ubpr' in model_name else 0.
            self.pair_weight = hyper_params['pair_weight'] if 'upl' in model_name else 0.
            self.dual_unbias = hyper_params['dual_unbias'] if 'bpr' not in model_name else False
            # self.sample_times = hyper_params['sample_times']
        elif model_name in ['ours1_bpr']:
            self.lam1 = params['lam1']
            self.lam2 = params['lam2']
            self.lam3 = params['lam3']
            self.eta = params['eta']
            self.dim = params['dim']
            self.beta = 0            
            self.weight = params['weight']
            self.ind = params['ind']
        elif model_name in ['ours2_wmf']:
            self.lam1 = params['lam1']
            self.lam2 = params['lam2']
            self.lam3 = params['lam3']
            self.eta = params['eta']
            self.dim = params['dim']
            self.beta = 0            
            self.weight = params['weight']     
            self.clip = params['clip']
            self.dual_unbias = False   
        elif model_name in ['Ours']:  # 新增 'Ours' 参数处理
            if params is None:
                # 打开并读取 config.yaml 文件
                with open(f'../conf/Ours.yaml', 'r') as file:
                    config = yaml.safe_load(file)
                params = config['Ours']  # 确保读取 'Ours' 部分的配置
            self.lambda1 = params['lambda1']  # 正则化系数
            self.lambda2 = params['lambda2']  # 正则化系数
            self.wd1 = params['weight_decay1']  # 权重衰减
            self.wd2 = params['weight_decay2']  # 权重衰减
            self.hidden_dim = params['hidden_dim']  # 表征层维度
            self.embedding_dim = params['embedding_dim']  # Embedding 层的维度
            self.eta = params['eta']  # 学习率
            # self.reg = params['reg']  # 正则化系数
            self.max_iters = params['max_iters']  # 最大迭代次数
            self.batch_size = params['batch_size']  # 批量大小
            # self.r_ratio = params['r_ratio']  # R2 和 R1 的比例
            self.random_state = params.get('random_state', 12345)   # 随机种子
            self.nu = params['nu']
            # self.radius = params['radius']
            self.percentile = params['percentile']
            self.HN_percentile = params['HN_percentile']
            self.subsample = params['subsample']
            self.hidden_pre = params['prediction_hidden_dim']
            
        else:
            # Load model-specific parameters from CJMF or BISER YAML files via external config
            if params is None:
                # If params not provided, load from the specific YAML file
                if model_name == 'cjmf':
                    params = yaml.safe_load(open(f'../conf/cjmf.yaml', 'r'))['cjmf'][data]
                elif model_name == 'BISER':
                    params = yaml.safe_load(open(f'../conf/BISER.yaml', 'r'))['BISER'][data]

            # Set model-specific parameters


            if model_name == 'BISER':
                self.wu = params['wu']
                self.wi = params['wi']
                self.alpha = params['alpha']
                self.neg_sample = params['neg_sample']
                self.dim = params['hidden']  # 必须在params里定义
                self.lam = float(params['lam'])  # 必须在params里定义，确保是float类型
                self.clip = params['clip']  # 必须在params里定义
                self.batch_size = params['batch_size']  # 必须在params里定义
                self.max_iters = params['max_iters']  # 必须在params里定义
                self.eta = params['eta']  # 必须在params里定义
                self.unbiased_eval = params.get('unbiased_eval', True)  # 默认 True，只有 cjmf 和 BISER 使用

            if model_name == 'cjmf':
                self.C = params['C']
                self.alpha_cjmf = params['alpha_cjmf']
                self.beta_cjmf = params['beta_cjmf']
                self.dim = params['hidden']  # 必须在params里定义
                self.lam = float(params['lam'])  # 必须在params里定义，确保是float类型
                self.clip = params['clip']  # 必须在params里定义
                self.batch_size = params['batch_size']  # 必须在params里定义
                self.max_iters = params['max_iters']  # 必须在params里定义
                self.eta = params['eta']  # 必须在params里定义
                self.unbiased_eval = params.get('unbiased_eval', True)  # 默认 True，只有 cjmf 和 BISER 使用


            self.best_model_save = params.get('best_model_save', True)
            self.date_now = params.get('date_now', '')
            self.random_state = params.get('random_state', [1])
        

    def get_performance_metric(self):
        """Return the desired performance metric for optimization."""
        return np.mean(self.val_loss_list)  # 返回验证集的平均损失
        
        
        # 假设你希望优化 DCG@3
        
        # metric_to_optimize = 'NDCG@3'  # 指定你要优化的指标
        # print(self.metrics['val_results'])
        # exit()
        # # 确保 'DCG@3' 存在于 DataFrame 的 index 中
        # if metric_to_optimize in self.metrics['val_results'].index:
        #     # 返回该指标的平均值
        #     return self.metrics['val_results'].loc[metric_to_optimize].mean()  # 根据索引选择行
        # else:
        #     raise ValueError(f"Metric {metric_to_optimize} not found in final results.")

    def run(self, num_sims: int = 10) -> None:
        """Train implicit recommenders."""
        item_freq = np.load(f'../data/{self.data}_st_1/point/item_freq.npy')
        train_point = np.load(f'../data/{self.data}_st_1/point/train.npy')
        val_point = np.load(f'../data/{self.data}_st_1/point/val.npy')
        test_point = np.load(f'../data/{self.data}_st_1/point/test.npy')
        pscore = np.load(f'../data/{self.data}_st_1/point/pscore.npy')
        num_users = np.int(train_point[:, 0].max() + 1)
        num_items = np.int(train_point[:, 1].max() + 1)
        print(num_items)
        print(num_users)
        if 'ours1' in self.model_name:
            train = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_train_30_30.npy')
            val = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_val_30_30.npy')
            test = np.load(f'../data/{self.data}_st_1/pair/test.npy')
            train_point = np.load(f'../data/{self.data}_st_1/point/labeled_train_30_30.npy')
            val_point = np.load(f'../data/{self.data}_st_1/point/labeled_val_30_30.npy')  
            # print(train.shape)
            # print(val.shape)
            # print(train[:10])
            # print(val[:10])
            # print(train_point.shape)
            # print(val_point.shape)
            # print(train_point[:10])
            # print(val_point[:10])
            # exit()            
        elif 'bpr' in self.model_name:
            data_name = self.model_name.split('_')[0]
            # train = np.load(f'../data/{self.data}_st_1/pair/{data_name}_train.npy')
            # val = np.load(f'../data/{self.data}_st_1/pair/{data_name}_val.npy')
            train = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_train_90_10.npy')
            val = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_val_90_10.npy')
            test = np.load(f'../data/{self.data}_st_1/pair/test.npy')
            # train_point = np.load(f'../data/{self.data}_st_1/point/labeled_train_90_10.npy')
            # val_point = np.load(f'../data/{self.data}_st_1/point/labeled_val_90_10.npy')                    
            # print(train.shape)
            # print(val.shape)
            # print(train[:10])
            # print(val[:10])
            # exit()
        # if self.data == 'yahoo' or self.data == 'kuai':
        #     user_freq = np.load(f'../data/{self.data}/point/user_freq.npy')
        #     item_freq = np.load(f'../data/{self.data}/point/item_freq.npy')

        result_list = list()
        val_loss_list = list()
        # if self.data == 'yahoo' or self.data =='kuai':
        #     cold_user_result_list = list()
        #     rare_item_result_list = list()
        for seed in np.arange(num_sims):
            set_seed(seed)
            
            ops.reset_default_graph()
            
            sess = tf.Session()
            
            
            
            # config = tf.ConfigProto()
            # config.intra_op_parallelism_threads = 1  # CPU 并行线程数
            # config.inter_op_parallelism_threads = 1
            # sess = tf.Session(config=config)
            
            
            
            if 'bpr' in self.model_name:
                if 'ours1' in self.model_name:
                    pair_rec = PairwiseRecommender_ours(num_users=num_users, num_items=num_items, dim=self.dim,
                                               lam1=self.lam1, lam2=self.lam2, lam3 = self.lam3, eta=self.eta, beta=self.beta, ind = self.ind, weight = self.weight)                
                elif 'upl_bpr' not in self.model_name:
                    pair_rec = PairwiseRecommender(num_users=num_users, num_items=num_items, dim=self.dim,
                                               lam=self.lam, eta=self.eta, beta=self.beta)
                else:
                    pair_rec = UPLPairwiseRecommender(num_users=num_users, num_items=num_items, dim=self.dim,
                                               lam=self.lam, eta=self.eta, beta=self.beta, pair_weight=self.pair_weight)

                if 'ours1' in self.model_name:
                    u_emb, i_emb, val_loss = train_pairwise_ours(sess, model=pair_rec, data=f"{self.data}_st_1",
                                                     train=train, val=val, test=test, train_point=train_point, val_point=val_point,
                                                     max_iters=self.max_iters, batch_size=self.batch_size,
                                                     model_name=self.model_name) 
                else:
                    u_emb, i_emb, val_loss = train_pairwise(sess, model=pair_rec, data=f"{self.data}_st_1",
                                                 train=train, val=val, test=test,
                                                 max_iters=self.max_iters, batch_size=self.batch_size,
                                                 model_name=self.model_name)
                
            elif self.model_name == 'Ours':  # 针对 Ours 模型进行训练
                # 初始化 Ours 模型

                model = Ours(sess,
                             train=train_point,
                             val=val_point,
                             num_users=num_users,
                             num_items=num_items,
                             hidden_dim=self.hidden_dim,
                             hidden_pre=self.hidden_pre,
                             embedding_dim=self.embedding_dim,
                             eta=self.eta,
                             max_iters=self.max_iters,
                             batch_size=self.batch_size,
                             random_state=self.random_state,
                             nu=self.nu,
                             percentile=self.percentile,
                             HN_percentile=self.HN_percentile,
                             subsample=self.subsample,
                             data_name=self.data,
                             wd1=self.wd1,
                             wd2=self.wd2,
                             lambda1=self.lambda1,
                             lambda2=self.lambda2
                             )
                labeled_train_path = Path(f'../data/{self.data}_st_1/point/labeled_train_{self.percentile}_{self.HN_percentile}.npy')
                labeled_val_path = Path(f'../data/{self.data}_st_1/point/labeled_val_{self.percentile}_{self.HN_percentile}.npy')

                # if os.path.exists(labeled_train_path):
                    # U,  I, Y, R, O
                #     labeled_train = np.load(labeled_train_path)
                #     labeled_train_pos = labeled_train[labeled_train[:, 2] == 1]
                #     labeled_train_neg = labeled_train[(labeled_train[:, 2] == 0) & (labeled_train[:, 3] == 1) & (labeled_train[:, 4] == 1)]
                #     labeled_train_pairs = []

                #     for user in np.unique(labeled_train[:, 0]):
                #         pos_items = labeled_train_pos[labeled_train_pos[:, 0] == user]
                #         neg_items = labeled_train_neg[labeled_train_neg[:, 0] == user]
                #         for pos_item in pos_items:
                #             if len(neg_items) < 2:
                #                 additional_neg_items = np.random.choice(len(labeled_train_neg), size=2 - len(neg_items), replace=False)
                #                 additional_neg_items = labeled_train_neg[additional_neg_items]
                #                 selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                #             else:
                #                 selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                #                 selected_neg_items = neg_items[selected_neg_items]
                #             for neg_item in selected_neg_items:
                #                 labeled_train_pairs.append([user, pos_item[1], neg_item[1]])   

                                                            
                #     labeled_train_pairs = np.array(labeled_train_pairs)
                #     np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_train_{self.percentile}_{self.HN_percentile}.npy', labeled_train_pairs)
                #     print(labeled_train_pairs.shape)
                #     print(labeled_train_pairs[:30])                    
                #     # exit()
                # if os.path.exists(labeled_val_path):
                #     labeled_val = np.load(labeled_val_path)
                #     labeled_val_pos = labeled_val[labeled_val[:, 2] == 1]
                #     labeled_val_neg = labeled_val[(labeled_val[:, 2] == 0) & (labeled_val[:, 3] == 1) & (labeled_val[:, 4] == 1)] 
                #     labeled_val_pairs = []

                #     for user in np.unique(labeled_val[:, 0]):
                #         pos_items = labeled_val_pos[labeled_val_pos[:, 0] == user]
                #         neg_items = labeled_val_neg[labeled_val_neg[:, 0] == user]
                #         for pos_item in pos_items:
                #             if len(neg_items) < 2:
                #                 additional_neg_items = np.random.choice(len(labeled_val_neg), size=2 - len(neg_items), replace=False)
                #                 additional_neg_items = labeled_val_neg[additional_neg_items]
                #                 selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                #             else:
                #                 selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                #                 selected_neg_items = neg_items[selected_neg_items]
                #             for neg_item in selected_neg_items:
                #                 labeled_val_pairs.append([user, pos_item[1], neg_item[1]])                                
               
                #     labeled_val_pairs = np.array(labeled_val_pairs)
                #     np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_val_{self.percentile}_{self.HN_percentile}.npy', labeled_val_pairs)
                #     print(labeled_val_pairs.shape)
                #     print(labeled_val_pairs[:30])                    
                #     exit()   

                if os.path.exists(labeled_train_path):
                    # U,  I, Y, R, O
                    labeled_train = np.load(labeled_train_path)

                    labeled_train_pos = labeled_train[labeled_train[:, 4] == -1]
                    labeled_train_neg = labeled_train[(labeled_train[:, 2] == 0) & (labeled_train[:, 3] == 1) & (labeled_train[:, 4] == 1)]
                    labeled_train_pairs = []

                    # print(len(labeled_train_pos))
                    # print(len(labeled_train_neg))
                    # exit()

                    # 20
                    # print(labeled_train_pos[labeled_train_pos[:, 0] == 57])
                    # print(labeled_train_pos[labeled_train_pos[:, 0] == 101])
                    # print((labeled_train_neg[labeled_train_neg[:, 0] == 57])) # 4
                    # print((labeled_train_neg[labeled_train_neg[:, 0] == 101])) # 7
                    # exit()
                    for user in np.unique(labeled_train[:, 0]):
                        pos_items = labeled_train_pos[labeled_train_pos[:, 0] == user]
                        neg_items = labeled_train_neg[labeled_train_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_train_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_train_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                            for neg_item in selected_neg_items:
                                labeled_train_pairs.append([user, pos_item[1], neg_item[1]])   

                    labeled_train_pos = labeled_train[labeled_train[:, 4] == 0] # 180 180
                    labeled_train_neg = labeled_train[labeled_train[:, 4] == -1] # 20
                    # 90 / 10
                    # print(len(labeled_train_neg))
                    # labeled_train_pairs = []                    
                    for user in np.unique(labeled_train[:, 0]):
                        pos_items = labeled_train_pos[labeled_train_pos[:, 0] == user]
                        neg_items = labeled_train_neg[labeled_train_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_train_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_train_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                                # ind = 0
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                                # ind = 1
                            for neg_item in selected_neg_items:
                                labeled_train_pairs.append([user, pos_item[1], neg_item[1]])        

                    labeled_train_pos = labeled_train[labeled_train[:, 2] == 1]
                    labeled_train_neg = labeled_train[(labeled_train[:, 2] == 0) & (labeled_train[:, 3] == 1) & (labeled_train[:, 4] == 1)]
                    # labeled_train_pairs = []

                    for user in np.unique(labeled_train[:, 0]):
                        pos_items = labeled_train_pos[labeled_train_pos[:, 0] == user]
                        neg_items = labeled_train_neg[labeled_train_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_train_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_train_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                            for neg_item in selected_neg_items:
                                labeled_train_pairs.append([user, pos_item[1], neg_item[1]])   

                                                            
                    labeled_train_pairs = np.array(labeled_train_pairs)
                    np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_train_{self.percentile}_{self.HN_percentile}.npy', labeled_train_pairs)
                    print(labeled_train_pairs.shape)
                    print(labeled_train_pairs[:30])
                    # exit()
                if os.path.exists(labeled_val_path):
                    labeled_val = np.load(labeled_val_path)
                    labeled_val_pos = labeled_val[labeled_val[:, 4] == -1]
                    labeled_val_neg = labeled_val[(labeled_val[:, 2] == 0) & (labeled_val[:, 3] == 1) & (labeled_val[:, 4] == 1)] 
                    labeled_val_pairs = []

                    for user in np.unique(labeled_val[:, 0]):
                        pos_items = labeled_val_pos[labeled_val_pos[:, 0] == user]
                        neg_items = labeled_val_neg[labeled_val_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_val_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_val_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                            for neg_item in selected_neg_items:
                                labeled_val_pairs.append([user, pos_item[1], neg_item[1]])                                

                    labeled_val_pos = labeled_val[labeled_val[:, 4] == 0]
                    labeled_val_neg = labeled_val[labeled_val[:, 4] == -1] 
                    # labeled_val_pairs = []

                    for user in np.unique(labeled_val[:, 0]):
                        pos_items = labeled_val_pos[labeled_val_pos[:, 0] == user]
                        neg_items = labeled_val_neg[labeled_val_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_val_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_val_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                            for neg_item in selected_neg_items:
                                labeled_val_pairs.append([user, pos_item[1], neg_item[1]])                                

                    labeled_val_pos = labeled_val[labeled_val[:, 2] == 1]
                    labeled_val_neg = labeled_val[(labeled_val[:, 2] == 0) & (labeled_val[:, 3] == 1) & (labeled_val[:, 4] == 1)] 

                    for user in np.unique(labeled_val[:, 0]):
                        pos_items = labeled_val_pos[labeled_val_pos[:, 0] == user]
                        neg_items = labeled_val_neg[labeled_val_neg[:, 0] == user]
                        for pos_item in pos_items:
                            if len(neg_items) < 2:
                                additional_neg_items = np.random.choice(len(labeled_val_neg), size=2 - len(neg_items), replace=False)
                                additional_neg_items = labeled_val_neg[additional_neg_items]
                                selected_neg_items = np.concatenate((neg_items, additional_neg_items))
                            else:
                                selected_neg_items = np.random.choice(len(neg_items), size=2, replace=False)
                                selected_neg_items = neg_items[selected_neg_items]
                            for neg_item in selected_neg_items:
                                labeled_val_pairs.append([user, pos_item[1], neg_item[1]])                                
               
                    labeled_val_pairs = np.array(labeled_val_pairs)
                    np.save(f'/home/zhouchuan/unbiased-pairwise-rec-master/data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_val_{self.percentile}_{self.HN_percentile}.npy', labeled_val_pairs)
                    print(labeled_val_pairs.shape)
                    print(labeled_val_pairs[:30])                    
                    exit()
                else:
                    print('No labeled data found, training classifier...')
                    labeled_train, labeled_val = model.train_classifier(target_percentile=self.percentile, subsample_size=self.subsample)
                    exit()
                val_loss = train_ours(sess, model, save_path=f'../logs/{self.data}_st_1/Ours/{self.percentile}_{self.HN_percentile}/best_model.ckpt', 
                                      labeled_train=labeled_train, labeled_val=labeled_val, 
                                      max_iters=self.max_iters, batch_size=self.batch_size, 
                                      model_name=self.model_name)
                # prediction = model.load_and_test_model(test_data=test_point[:, :2])  # 对测试集进行预测

            elif self.model_name in ['BISER']:
                weights_enc_u, weights_dec_u, bias_enc_u, bias_dec_u, weights_enc_i, weights_dec_i, bias_enc_i, bias_dec_i = ae_trainer(
                    sess, data=self.data, train=train_point, val=val_point, test=test_point,
                    num_users=num_users, num_items=num_items, n_components=self.dim, wu=self.wu, wi=self.wi,
                    eta=self.eta, lam=self.lam, max_iters=self.max_iters, batch_size=self.batch_size,
                    model_name=self.model_name, item_freq=item_freq, unbiased_eval = self.unbiased_eval, random_state=seed)
            elif self.model_name in ['cjmf']:
                u_emb, i_emb = cjmf_trainer(sess=sess, data=self.data, n_components=self.dim, num_users=num_users,
                                            num_items=num_items, \
                                            batch_size=self.batch_size, max_iters=self.max_iters, item_freq=item_freq, \
                                            unbiased_eval=self.unbiased_eval, C=self.C, lr=self.eta, reg=self.lam, \
                                            alpha=self.alpha_cjmf, beta=self.beta_cjmf, train=train_point, val=val_point,
                                            seed=seed, model_name=self.model_name)
            elif self.model_name in ['ours2_wmf']:
                train = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_train_30_30.npy')
                val = np.load(f'../data/{self.data}_st_1/pair/ours_bak_ocsvm_formal/Ours_val_30_30.npy')
                test = np.load(f'../data/{self.data}_st_1/pair/test.npy')
                train_point = np.load(f'../data/{self.data}_st_1/point/labeled_train_30_30.npy')
                val_point = np.load(f'../data/{self.data}_st_1/point/labeled_val_30_30.npy')          

                point_rec = PointwiseRecommender_ours(num_users=num_users, num_items=num_items, weight=self.weight,
                                                 clip=self.clip, dim=self.dim, lam1=self.lam1, lam2 = self.lam2, lam3 = self.lam3, eta=self.eta, dual_unbias=self.dual_unbias)

                u_emb, i_emb, val_loss = train_pointwise_ours(sess, model=point_rec, data=self.data,
                                                  train=train_point, val=val_point, test=test_point, train_pairwise = train, val_pairwise = val, pscore=pscore,
                                                  max_iters=self.max_iters, batch_size=self.batch_size,
                                                  model_name=self.model_name)

            elif self.model_name in ['wmf', 'relmf', 'relmf_du']:
                point_rec = PointwiseRecommender(num_users=num_users, num_items=num_items, weight=self.weight,
                                                 clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta, dual_unbias=self.dual_unbias)
                u_emb, i_emb, _ = train_pointwise(sess, model=point_rec, data=self.data,
                                                  train=train_point, val=val_point, test=test_point, pscore=pscore,
                                                  max_iters=self.max_iters, batch_size=self.batch_size,
                                                  model_name=self.model_name)
            elif self.model_name == 'expomf':
                u_emb, i_emb = train_expomf(data=self.data, train=train_point, num_users=num_users, num_items=num_items)
                
            val_loss_list.append(val_loss)
            
            
            # Evaluate the model
            if self.model_name in ['BISER']:
                results = aoa_evaluator(user_embed=[weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i],
                                        item_embed=[bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i],
                                        train=train_point, test=test_point, num_users=num_users, num_items=num_items,
                                        model_name=self.model_name, at_k=self.at_k)

            elif self.model_name in ['cjmf']:
                results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                        train=train_point, test=test_point,num_users=num_users, num_items=num_items,
                                        model_name=self.model_name, at_k=self.at_k)
            elif self.model_name in ['Ours']:
                results = aoa_evaluator(user_embed=None, item_embed=None,test=test_point,
                                        model_name=self.model_name, at_k=self.at_k, model=model)
            else:
                results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                   test=test_point, model_name=self.model_name, at_k=self.at_k)
            result_list.append(results)
            print(results)
            # self.metrics['final_results'] = pd.concat(result_list, axis=1)

        self.val_loss_list = val_loss_list

        # val metrics
        #     if self.model_name in ['BISER']:
        #         val_results = aoa_evaluator(user_embed=[weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i],
        #                                 item_embed=[bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i],
        #                                 train=train_point, test=val_point, num_users=num_users, num_items=num_items,
        #                                 model_name=self.model_name, at_k=self.at_k)

        #     elif self.model_name in ['cjmf']:
        #         val_results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
        #                                 train=train_point, test=val_point,num_users=num_users, num_items=num_items,
        #                                 model_name=self.model_name, at_k=self.at_k)
        #     elif self.model_name in ['Ours']:
        #         val_results = aoa_evaluator(user_embed=None, item_embed=None,test=val_point,
        #                                 model_name=self.model_name, at_k=self.at_k, model=model)
        #     else:
        #         val_results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
        #                            test=val_point, model_name=self.model_name, at_k=[1])

        #     val_result_list.append(val_results)
        # self.metrics['val_results'] = pd.concat(val_result_list, axis=1)
        
        
        
        
        
            # if self.data == 'yahoo' or self.data == 'kuai':
            #     user_idx, item_idx = test_point[:, 0].astype(int), test_point[:, 1].astype(int)
            #     cold_user_idx = user_freq[user_idx] <= self.cold_user_threshold
            #     rare_item_idx = item_freq[item_idx] <= self.rare_item_threshold
            #     cold_user_result = aoa_evaluator(user_embed=u_emb, item_embed=i_emb, at_k=self.at_k,
            #                                      test=test_point[cold_user_idx], model_name=self.model_name)
            #     rare_item_result = aoa_evaluator(user_embed=u_emb, item_embed=i_emb, at_k=self.at_k,
            #                                      test=test_point[rare_item_idx], model_name=self.model_name)
            #     cold_user_result_list.append(cold_user_result)
            #     rare_item_result_list.append(rare_item_result)
            #
            # print(f'#{seed+1}: {self.model_name}...')

        ret_path = Path(f'../logs/{self.data}_st_1/{self.model_name}/results')
        ret_path.mkdir(parents=True, exist_ok=True)
        pd.concat(result_list, axis=1).to_csv(ret_path / f'aoa_all.csv')
        # if self.data == 'yahoo' or self.data == 'kuai':
        #     pd.concat(cold_user_result_list, axis=1).to_csv(ret_path / f'aoa_cold-user.csv')
        #     pd.concat(rare_item_result_list, axis=1).to_csv(ret_path / f'aoa_rare-item.csv')