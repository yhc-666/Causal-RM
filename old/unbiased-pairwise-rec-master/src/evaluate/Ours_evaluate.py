from typing import List
import numpy as np
import pandas as pd

from .metrics import ndcg_at_k, average_precision_at_k, recall_at_k

from typing import Callable, Dict, List, Optional, Tuple
def aoa_evaluator(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  at_k: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """Compute ranking metrics at specified cutoff positions."""

    metrics: Dict[str, Callable[[np.ndarray, np.ndarray, int], float]] = {
        'NDCG': ndcg_at_k,
        'AP': average_precision_at_k,
        'Recall': recall_at_k,
    }

    results = {}
    for k in at_k:
        for metric_name, metric_func in metrics.items():
            results[f'{metric_name}@{k}'] = []
    print("y_true:", len(y_true))
    print("y_pred:", y_pred.shape)
    # Calculate metrics for each user/item pair
    for user_idx in range(len(y_true)):
        true_labels = y_true[user_idx]
        pred_scores = y_pred[user_idx]

        for k in at_k:
            for metric_name, metric_func in metrics.items():
                results[f'{metric_name}@{k}'].append(metric_func(true_labels, pred_scores, k))

    # Convert results to a DataFrame with mean values across users
    results_df = pd.DataFrame({metric: np.mean(values) for metric, values in results.items()}, index=[0])

    return results_df