import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr


def rank_normalize(x: np.ndarray) -> np.ndarray:
    ranks = x.argsort().argsort().astype(float)
    return ranks / len(ranks) - 0.5


def rank_l2_loss(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    y = dtrain.get_label()
    ranked_pred = rank_normalize(pred)
    y_normalized = rank_normalize(y)
    grad = 2 * (ranked_pred - y_normalized) / len(y)
    hess = 2 * np.ones_like(y) / len(y)
    return grad, hess


def rank_l2_objective():
    return rank_l2_loss


def rank_l2_metric(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    y = dtrain.get_label()
    ic, _ = spearmanr(pred, y)
    return "RankIC", ic, True
