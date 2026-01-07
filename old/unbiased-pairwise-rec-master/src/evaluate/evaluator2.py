"""Evaluate Implicit Recommendation models."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.ma.extras import flatten_inplace
import pandas as pd

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k

def sigmoid(x):
    return 1./(1+np.exp(-x))

class PredictRankings:
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


metrics = {'Recall': recall_at_k,
           'MAP': average_precision_at_k,
           'DCG':dcg_at_k}


def aoa_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  test: np.ndarray,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5],
                  mu:np.ndarray = None,
                  user_bias:np.ndarray = None,
                  item_bias:np.ndarray = None) -> pd.DataFrame:
    users = test[:, 0]
    items = test[:, 1]
    relevances = 0.001 + 0.999 * test[:, 2]
    test_score = test[:, 2]

    model = PredictRankings(user_embed=user_embed, item_embed=item_embed,mu=mu,user_bias = user_bias,item_bias = item_bias)

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    for user in set(users):
        indices = users == user
        pos_items = items[indices]
        rel = relevances[indices]
        if np.sum(test_score[indices]) == 0:
            continue
        # predict ranking score for each user
        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[model_name] = \
            list(map(np.mean, list(results.values())))

    if mu is not None:
        print('\nwith mu:')

    #results_df[model_name]
    return results_df