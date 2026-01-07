"""评估隐式推荐模型。"""
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluate.metrics import average_precision_at_k, ndcg_at_k, recall_at_k


class PredictRankings:
    """通过训练好的推荐模型预测排序。"""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray) -> None:
        """初始化类。"""
        self.user_embed = user_embed
        self.item_embed = item_embed

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """预测用户-物品对的分数。"""
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()
        return scores

class PredictRankings_recrec:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray , mu:np.ndarray = None, user_bias:np.ndarray = None, item_bias:np.ndarray = None) -> None:
        """Initialize Class."""
        # latent embeddings
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.mu = mu
        self.user_bias = user_bias
        self.item_bias = item_bias


    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1]) # 1*dim
        item_emb = self.item_embed[items]  #n* dim


        scores = (user_emb @ item_emb.T).flatten()  # 1*dim * dim* n  ->  1*n


        if self.item_bias is not None:
            item_b = self.item_bias[items].flatten()
            scores = (item_b + scores)
        return scores    


class PredictRankings_Ours:
    """通过训练好的推荐模型预测排序。"""

    def __init__(self, model) -> None:
        """初始化类。"""
        self.model = model
        self.sess = model.sess
        self.data_name = model.data_name
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, f'../logs/{self.data_name}_st_1/Ours/{self.model.percentile}_{self.model.HN_percentile}/best_model.ckpt')        
        print("Best model restored.")

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """预测用户-物品对的分数。"""
        
        
        user_ids = np.array([users] * len(items))
        item_ids = items
        feed_dict = {
            self.model.user_input: user_ids,
            self.model.item_input: item_ids
        }

        # 获取预测结果
        scores = self.sess.run(self.model.prediction, feed_dict=feed_dict).flatten()
        
        
        # user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        # item_emb = self.item_embed[items]
        # scores = (user_emb @ item_emb.T).flatten()
        return scores

class PredictRankings_i_u_AE:
    """proposed模型的排序预测"""

    def __init__(self, train, num_users, num_items, weights, biases) -> None:
        """初始化类。"""
        # 初始化与proposed模型相关的变量
        # 省略详细实现，假设已完成
        pass

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """根据proposed模型预测分数"""
        # 实现proposed模型的预测逻辑
        pass


class PredictRankings_cjmf:
    """cjmf模型的排序预测"""

    def __init__(self, user_embed: list, item_embed: list, num_users: int, num_items: int, pred=None) -> None:
        """初始化类。"""
        # 初始化与cjmf模型相关的变量
        # 省略详细实现，假设已完成
        pass

    def predict(self, users: int, items: np.array) -> np.ndarray:
        """根据cjmf模型预测分数"""
        pass


metrics = {'NDCG': ndcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}


def aoa_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  test: np.ndarray,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5],
                  model = None,
                  mu = None,
                  user_bias = None,
                  item_bias = None) -> pd.DataFrame:
    """使用平均评估器计算排序指标。"""

    # 定义模型，根据传入的模型名称选择合适的模型
    if model_name == 'BISER':
        model = PredictRankings_i_u_AE(weights=user_embed, biases=item_embed, num_users=len(user_embed), num_items=len(item_embed), train=test)
    elif model_name == 'cjmf':
        model = PredictRankings_cjmf(user_embed=user_embed, item_embed=item_embed, num_users=len(user_embed), num_items=len(item_embed))
    elif model_name == 'Ours':
        model = PredictRankings_Ours(model)
    elif 'ReCRec' in model_name:
        model = PredictRankings_recrec(user_embed=user_embed, item_embed=item_embed, mu=mu, user_bias=user_bias, item_bias=item_bias)
    else:
        model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # 准备排序指标
    users = test[:, 0]
    items = test[:, 1]
    relevances = test[:, 2]

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # 计算排序指标
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        rel = relevances[indices]

        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                result = metric_func(rel, scores, k)
                if result < 0:
                    continue
                results[f'{metric}@{k}'].append(result)

        # 汇总结果
        results_df = pd.DataFrame(index=results.keys())
        results_df[model_name] = list(map(np.mean, list(results.values())))
        # results_df[model_name] = [np.mean([x for x in values if x >= 0]) for values in results.values()]

    return results_df


def unbiased_evaluator(user_embed: np.ndarray,
                       item_embed: np.ndarray,
                       train: np.ndarray,
                       val: np.ndarray,
                       pscore: np.ndarray,
                       model_name: str,
                       metric: str = 'NDCG',
                       num_negatives: int = 100,
                       k: int = 5) -> float:
    """使用无偏评估器计算排序指标。"""

    # 根据模型名称选择对应的模型
    if model_name == 'BISER':
        model = PredictRankings_i_u_AE(weights=user_embed, biases=item_embed, num_users=len(user_embed), num_items=len(item_embed), train=train)
    elif model_name == 'cjmf':
        model = PredictRankings_cjmf(user_embed=user_embed, item_embed=item_embed, num_users=len(user_embed), num_items=len(item_embed))
    elif model_name == 'Ours':
        model = PredictRankings_Ours(model)
    elif 'ReCRec' in model_name:
        model = PredictRankings_recrec(user_embed=user_embed, item_embed=item_embed, mu=mu, user_bias=user_bias, item_bias=item_bias)    
    else:
        model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # 计算无偏DCG得分
    users = val[:, 0]
    items = val[:, 1]
    positive_pairs = np.r_[train[train[:, 2] == 1, :2], val]

    dcg_values = list()
    unique_items = np.unique(items)
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user]
        neg_items = np.random.permutation(np.setdiff1d(unique_items, all_pos_items))[:num_negatives]
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[used_items]
        relevances = np.r_[np.ones_like(pos_items), np.zeros(num_negatives)]

        scores = model.predict(users=user, items=used_items)
        dcg_values.append(metrics[metric](relevances, scores, k, pscore_))

    return np.mean(dcg_values)
