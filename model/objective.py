"""
自定义目标函数模块

定义 LightGBM 的自定义目标函数和评估指标。

当前实现：
1. Rank-L2 损失函数：基于排名的 L2 损失
2. 自定义 RankIC 评估指标

这些自定义函数用于替代 LightGBM 内置的目标函数，
以更好地适应量化排序任务。
"""

import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """
    将数组归一化为 [-0.5, 0.5] 范围的排名值

    使用 argsort 的两次应用来获取排名，然后归一化。

    Args:
        x: 输入数组

    Returns:
        归一化后的排名数组，范围 [-0.5, 0.5]
    """
    ranks = x.argsort().argsort().astype(float)
    return ranks / len(ranks) - 0.5


def rank_l2_loss(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    """
    Rank-L2 损失函数

    原理：
    - 将预测值和真实值都转换为排名
    - 计算排名之间的 L2 损失

    损失 = 2 * (rank_pred - rank_true) / n

    Args:
        pred: 预测值数组
        dtrain: LightGBM 数据集对象

    Returns:
        (gradient, hessian) 元组
    """
    y = dtrain.get_label()
    ranked_pred = rank_normalize(pred)
    y_normalized = rank_normalize(y)
    grad = 2 * (ranked_pred - y_normalized) / len(y)
    hess = 2 * np.ones_like(y) / len(y)
    return grad, hess


def rank_l2_objective():
    """
    获取 Rank-L2 目标函数

    用于 LightGBM 的 objective 参数。

    Returns:
        rank_l2_loss 函数
    """
    return rank_l2_loss


def rank_l2_metric(pred: np.ndarray, dtrain: lgb.Dataset) -> tuple:
    """
    Rank-L2 自定义评估指标

    计算预测与真实值的 Spearman 秩相关系数（RankIC）。

    Args:
        pred: 预测值数组
        dtrain: LightGBM 数据集对象

    Returns:
        (metric_name, metric_value, higher_is_better) 元组
    """
    y = dtrain.get_label()
    ic, _ = spearmanr(pred, y)
    return "RankIC", ic, True