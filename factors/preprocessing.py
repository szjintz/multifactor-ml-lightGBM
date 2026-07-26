"""
因子预处理模块

提供因子预处理的四个核心步骤：
1. 异常值处理（Winsorize）：使用 3σ 原则或分位数截断
2. 截面标准化（Standardize）：每个日期独立 Z-score 标准化
3. 中性化（Neutralize）：对市值、行业等外生变量回归取残差
4. 正交化（Orthogonalize）：因子间去相关（Gram-Schmidt 或 PCA）

关键设计原则：
- 所有处理都是按截面（每日）进行的，避免前视偏差
- 中性化和正交化使用线性回归，需要足够的样本量
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def winsorize(factor: pd.Series, method="3sigma") -> pd.Series:
    """
    因子缩尾处理

    将超出统计范围的极端值截断到边界值。
    - 3sigma：均值 ± 3 倍标准差
    - quantile：1% ~ 99% 分位数

    Args:
        factor: 输入因子序列
        method: 缩尾方法，"3sigma" 或 "quantile"

    Returns:
        缩尾后的因子序列
    """
    if method == "3sigma":
        mean, std = factor.mean(), factor.std()
        lower, upper = mean - 3 * std, mean + 3 * std
        logger.debug(f"[PREPROCESS] Winsorize(3sigma): mean={mean:.4f}, std={std:.4f}, bounds=[{lower:.4f}, {upper:.4f}]")
    elif method == "quantile":
        lower, upper = factor.quantile(0.01), factor.quantile(0.99)
        logger.debug(f"[PREPROCESS] Winsorize(quantile): bounds=[{lower:.4f}, {upper:.4f}]")
    else:
        logger.warning(f"[PREPROCESS] Unknown winsorize method: {method}, returning unchanged")
        return factor
    return factor.clip(lower, upper)


def cross_sectional_standardize(factor: pd.DataFrame) -> pd.DataFrame:
    """
    截面标准化（Z-score）

    对每个截面（每个日期），将所有股票的因子值标准化为均值为0、标准差为1。
    必须使用截面统计量，避免引入未来信息。
    支持 MultiIndex (instrument, date) 格式的因子DataFrame：

    Args:
        factor: 因子 DataFrame，index 为 (instrument, datetime)

    Returns:
        标准化后的因子 DataFrame
    """
    before_mean = factor.mean(axis=1).mean()
    before_std = factor.std(axis=1).mean()
    result = factor.copy()
    for col in result.columns:
        grouped = result[col].groupby(level=1)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        result[col] = (result[col] - mean) / std
    after_mean = result.mean(axis=1).mean()
    after_std = result.std(axis=1).mean()
    logger.debug(f"[PREPROCESS] Cross-sectional standardize: before mean={before_mean:.4f}, std={before_std:.4f}; after mean={after_mean:.4f}, std={after_std:.4f}")
    return result


def neutralize(factor: pd.Series, exog: pd.DataFrame) -> pd.Series:
    """
    因子中性化

    对因子与外生变量（如市值、行业）进行回归，取残差作为中性化后的因子。
    中性化可以去除因子中与外生变量线性相关的部分，使因子更纯粹。

    数学表达：
    factor_neutral = factor - X @ (X'X)^{-1} @ X' @ factor

    其中 X 是外生变量矩阵（如 [1, log(market_cap), industry_dummies]）

    Args:
        factor: 待中性化的因子序列
        exog: 外生变量 DataFrame，必须与 factor 索引对齐

    Returns:
        中性化后的因子序列
    """
    common_idx = exog.index.intersection(factor.index)
    exog_aligned = exog.loc[common_idx]
    factor_aligned = factor.loc[common_idx]
    valid = exog_aligned.notna().all(axis=1) & factor_aligned.notna()
    if valid.sum() < 10:
        logger.warning(f"[PREPROCESS] Neutralize: only {valid.sum()} valid points, skipping")
        return factor
    X = exog_aligned.loc[valid].values
    y = factor_aligned.loc[valid].values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    logger.debug(f"[PREPROCESS] Neutralize: {valid.sum()} valid points, R²={r2:.4f}")
    residuals = y - model.predict(X)
    result = factor.copy().astype(float)
    result.loc[valid.index[valid]] = residuals
    return result


def orthogonalize(factors: pd.DataFrame, method="gram_schmidt") -> pd.DataFrame:
    """
    因子正交化

    去除因子之间的线性相关性，使每个因子互相独立。
    两种方法：
    - Gram-Schmidt：顺序正交化，因子顺序影响结果
    - PCA：主成分分析，保留最大方差方向

    Args:
        factors: 因子 DataFrame
        method: 正交化方法，"gram_schmidt" 或 "pca"

    Returns:
        正交化后的因子 DataFrame
    """
    logger.info(f"[PREPROCESS] Orthogonalizing {factors.shape[1]} factors using method={method}")
    if method == "gram_schmidt":
        result = factors.copy()
        for i in range(1, factors.shape[1]):
            for j in range(i):
                col_i = result.iloc[:, i]
                col_j = result.iloc[:, j]
                mask = col_i.notna() & col_j.notna()
                if mask.sum() < 10:
                    continue
                col_i_v = col_i[mask]
                col_j_v = col_j[mask]
                col_norm_sq = col_j_v @ col_j_v
                if col_norm_sq < 1e-10:
                    continue
                coeff = (col_i_v @ col_j_v) / col_norm_sq
                col_i_new = col_i.copy()
                col_i_new[mask] = col_i_v - col_j_v * coeff
                result.iloc[:, i] = col_i_new
        logger.info(f"[PREPROCESS] Gram-Schmidt orthogonalization complete")
        return result
        logger.info(f"[PREPROCESS] Gram-Schmidt orthogonalization complete")
        return result
    elif method == "pca":
        from sklearn.decomposition import PCA
        logger.info(f"[PREPROCESS] PCA orthogonalization: {factors.shape[1]} -> {min(factors.shape[1], factors.shape[0])} components")
        pca = PCA(n_components=min(factors.shape[1], factors.shape[0]))
        components = pca.fit_transform(factors.fillna(0))
        logger.info(f"[PREPROCESS] PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}...")
        return pd.DataFrame(components, index=factors.index, columns=factors.columns[:components.shape[1]])
    logger.warning(f"[PREPROCESS] Unknown orthogonalize method: {method}, returning unchanged")
    return factors