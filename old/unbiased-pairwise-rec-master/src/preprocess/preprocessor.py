"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import codecs
from pathlib import Path
import pdb

import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information."""
    ratings -= 1
    return eps + (1. - eps) * (2 ** ratings - 1) / (2 ** np.max(ratings) - 1)

def kuai_transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information."""
    # ratings -= 1
    return eps + (1. - eps) * (2 ** ratings - 1) / (2 ** np.max(ratings) - 1)


def preprocess_dataset(data: str, sample_times: int):
    """Load and preprocess datasets."""
    np.random.seed(12345)
    if data == 'yahoo':
        cols = {0: 'user', 1: 'item', 2: 'rate'}
        with codecs.open(f'../data/yahoo/raw/train.txt', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter='\t', header=None)
            train_.rename(columns=cols, inplace=True)
        with codecs.open(f'../data/yahoo/raw/test.txt', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter='\t', header=None)
            test_.rename(columns=cols, inplace=True)
        for _data in [train_, test_]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    elif data == 'kuai':
        # 读取数据
        with open("../data/kuai/raw/kuairec.pkl", "rb") as f:
            x_train = pickle.load(f)
            y_train = pickle.load(f)
            x_test = pickle.load(f)
            y_test = pickle.load(f) 
        def reindex_by_column(data, column_idx):
            unique_vals, new_indices = np.unique(data[:, column_idx], return_inverse=True)
            data[:, column_idx] = new_indices
            return data
        train_ = pd.DataFrame(np.c_[x_train, y_train], columns=['user', 'item', 'rate'])
        test_ = pd.DataFrame(np.c_[x_test, y_test], columns=['user', 'item', 'rate'])
        # 强制将 'user' 和 'item' 列转换为整数格式
        train_['user'] = train_['user'].astype(int)
        train_['item'] = train_['item'].astype(int)
        test_['user'] = test_['user'].astype(int)
        test_['item'] = test_['item'].astype(int)

        # 确保 'rate' 不大于 5
        train_['rate'] = train_['rate'].apply(lambda x: min(float(x), 5))
        test_['rate'] = test_['rate'].apply(lambda x: min(float(x), 5))


    # # Create train_ and test_ from the updated `d`
        # train_ = pd.DataFrame(d[d[:, 3] == 1, :3], columns=['user', 'item', 'rate'])
        # test_ = pd.DataFrame(d[d[:, 3] == 0, :3], columns=['user', 'item', 'rate'])
    elif data == 'coat':
        cols = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/coat/raw/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter=' ', header=None)
            train_ = train_.stack().reset_index().rename(columns=cols)
            train_ = train_[train_.rate != 0].reset_index(drop=True)
        with codecs.open(f'../data/coat/raw/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter=' ', header=None)
            test_ = test_.stack().reset_index().rename(columns=cols)
            test_ = test_[test_.rate != 0].reset_index(drop=True)

    # count the num. of users and items.
    num_users, num_items = int(train_.user.max() + 1), int(train_.item.max() + 1)
    train, test = train_.values, test_.values
    # transform rating into (0,1)-scale.

    if data == 'kuai':
        rel_train = np.random.binomial(n=1, p=kuai_transform_rating(ratings=train[:, 2], eps=0.1))
        train = train[rel_train == 1,:2]
        # train = np.c_[train, np.ones(train.shape[0])]
        train = reindex_by_column(train, 0)  # Re-index
        train = reindex_by_column(train, 1)  # Re-index
        test = reindex_by_column(test, 0)  # Re-index
        test = reindex_by_column(test, 1)  #
        num_users = int(train[:, 0].max() + 1)
        num_items = int(train[:, 1].max() + 1)
        print("num_users:", num_users)
        print("num_items:", num_items)
        # test[:,2] = np.random.binomial(n=1, p=transform_rating(ratings=test[:, 2],eps=0.0))
        test[:, 2] = kuai_transform_rating(ratings=test[:, 2], eps=0.0)
    else:
        rel_train = np.random.binomial(n=1, p=transform_rating(ratings=train[:, 2], eps=0.1))
        train = train[rel_train == 1, :2]
        test[:, 2] = transform_rating(ratings=test[:, 2], eps=0.0)

    
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2]
    print("num_users:", num_users)
    print("num_items:", num_items)
    
    np.random.shuffle(all_data)
    all_num = int(len(train[:, 0]) * sample_times)
    all_data = all_data[:all_num]
        
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    if data == 'yahoo' or data == 'kuai':
        user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1]

        # item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
        # 计算item_freq时，检查所有item是否都有参与计算
        print(len(np.unique(train[:,1])))
        item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
        print(len(item_freq))
        # 调试输出
        pscore = (item_freq / item_freq.max()) ** 0.5
        # 11
    elif data == 'coat':
        user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1]

        # item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
        # 计算item_freq时，检查所有item是否都有参与计算
        item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
        matrix = sparse.lil_matrix((num_users, num_items))
        for (u, i) in train[:, :2]:
            matrix[int(u), int(i)] = 1
        pscore = np.clip(np.array(matrix.mean(axis=0)).flatten() ** 0.5, a_max=1.0, a_min=1e-6)

    # exstract only positive (relevant) user-item pairs
    # creating training data
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    # train_2 = train[:, :2]
    # train_1 = train[train[:, 2] == 1, :2]
    
    # unlabeled_data = np.array(list(set(map(tuple, train_2)) - set(map(tuple, train_1))), dtype=int)
    
    
    
    
    # n_train = int(train_2[:,0].max() + 1)
    # # 从未点击的样本中进行负采样，假设这些样本被暴露了，但没有点击 (exposure = 1, click = 0)
    # neg_samples_per_user = 5  # 你可以根据需要调整采样数量
    # negative_samples = []
    # for user_id in range(n_train):
    #     user_unobserved_items = unlabeled_data[unlabeled_data[:, 0] == user_id]
    #     if len(user_unobserved_items) > 0:
    #         sampled_items = user_unobserved_items[
    #             np.random.choice(len(user_unobserved_items), size=min(neg_samples_per_user, len(user_unobserved_items)),
    #                              replace=False)]
    #         negative_samples.append(sampled_items)

    # # 将负采样的样本合并成一个数组
    # negative_samples = np.vstack(negative_samples)

    # # 为正样本打标签：1 表示暴露并点击
    # labeled_positive_samples = np.c_[train_1, np.ones(train_1.shape[0])]

    # # 为负采样的样本打标签：1 表示暴露，但未点击
    # labeled_negative_samples = np.c_[negative_samples, np.ones(negative_samples.shape[0])]

    # # 合并正样本和负采样的负样本
    # labeled_data = np.r_[labeled_positive_samples, labeled_negative_samples]

    # unlabeled_exposure_data = np.array(
    #     list(set(map(tuple, train_2)) - set(map(tuple, labeled_data[:, :2]))), dtype=int)
    # # 为未暴露的样本打标签：0 表示未暴露
    # labeled_unexposed_samples = np.c_[unlabeled_exposure_data, np.zeros(unlabeled_exposure_data.shape[0])]

    # 最终合并所有样本，包含暴露和未暴露的样本
    # exposed = np.r_[labeled_data, labeled_unexposed_samples]  # 将结果存储到 `exposed`
    # print(f"exposure shape: {exposed.shape}")  # 打印exposure的维度

    # estimate propensities and user-item frequencies.

    # train-val split using the raw training datasets

    # save preprocessed datasets
    path_data = Path(f'../data/{data}_st_{sample_times}')
    (path_data / 'point').mkdir(parents=True, exist_ok=True)
    (path_data / 'pair').mkdir(parents=True, exist_ok=True)
    # pointwise
    # 保存为整数类型
    np.save(file=path_data / 'point/train.npy', arr=train.astype(np.int))
    np.save(file=path_data / 'point/val.npy', arr=val.astype(np.int))
    np.save(file=path_data / 'point/test.npy', arr=test.astype(np.int))  # 转换为int
    np.save(file=path_data / 'point/pscore.npy', arr=pscore.astype(np.int))  # 转换为int
    # np.save(file=path_data / 'point/exposure.npy', arr=exposed.astype(np.int))  # 转换为int

    # if data == 'yahoo' or data == 'kuai':
    np.save(file=path_data / 'point/user_freq.npy', arr=user_freq.astype(np.int))  # 转换为int
    np.save(file=path_data / 'point/item_freq.npy', arr=item_freq.astype(np.int))  # 转换为int

    # pairwise
    samples = 1
    bpr_train = _bpr(data=train, n_samples=samples)
    print("bpr_train_shape:", bpr_train.shape)
    print("train_shape:", train.shape)
    ubpr_train = _ubpr(data=train, pscore=pscore, n_samples=samples, name=data)
    bpr_val = _bpr(data=val, n_samples=samples)
    ubpr_val = _ubpr(data=val, pscore=pscore, n_samples=samples, name=data)
    pair_test = _bpr_test(data=test, n_samples=samples)

    # 保存pairwise数据为整数类型
    np.save(file=path_data / 'pair/bpr_train.npy', arr=bpr_train.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/ubpr_train.npy', arr=ubpr_train.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/upl_bpr_train.npy', arr=ubpr_train.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/bpr_val.npy', arr=bpr_val.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/ubpr_val.npy', arr=ubpr_val.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/upl_bpr_val.npy', arr=ubpr_val.astype(np.int))  # 转换为int
    np.save(file=path_data / 'pair/test.npy', arr=pair_test.astype(np.int))  # 转换为int


def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'click'])
    positive = df.query("click == 1")
    negative = df.query("click == 0")
    ret = positive.merge(negative, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y']].values


def _bpr_test(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'gamma'])
    ret = df.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'gamma_x', 'gamma_y']].values


def _ubpr(data: np.ndarray, pscore: np.ndarray, n_samples: int, name="kuai") -> np.ndarray:
    """Generate training data for the unbiased bpr."""
    data = np.c_[data, pscore[data[:, 1].astype(int)]]
    print("Data Column 1 (before int cast):", data[:, 1])  # To check the original values
    print("Data Column 1 (after int cast):", data[:, 1].astype(int))  # To check the integer conversion
    print("Max Index for pscore:", len(pscore))  # To check the size of pscore

    df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta'])
    positive = df.query("click == 1")
    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_x', 'theta_y']].values
