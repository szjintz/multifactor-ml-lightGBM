"""
特征变换模块

在基础特征之上构建衍生特征，包括：
1. 交叉特征（因子 × 市值）：捕捉因子与市值的非线性关系
2. 动量特征：因子的滚动均值和滚动波动率
3. IC 时序特征：用于分析目的，非训练特征

这些衍生特征可以增强模型的预测能力。
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def build_cross_features(factor_df: pd.DataFrame, market_cap: pd.Series) -> pd.DataFrame:
    """
    构建交叉特征

    交叉特征 = 原始因子 × log(市值)
    作用：捕捉因子与市值的非线性关系
    例如：小市值股票的因子效应可能与大市值股票不同

    Args:
        factor_df: 原始因子 DataFrame，MultiIndex
        market_cap: 市值 Series，与因子索引对齐

    Returns:
        增加了交叉特征的 DataFrame
    """
    logger.info(f"[TRANSFORMER] Building cross features with market_cap: {len(factor_df.columns)} base factors")
    new_cols = {}
    market_cap_aligned = market_cap.reindex(factor_df.index)
    for col in factor_df.columns:
        new_cols[f"{col}_x_CAP"] = factor_df[col] * market_cap_aligned
    result = pd.concat([factor_df, pd.DataFrame(new_cols, index=factor_df.index)], axis=1)
    logger.info(f"[TRANSFORMER] Cross features created: {len(result.columns)} total columns")
    return result


def build_momentum_features(factor_df: pd.DataFrame, windows=[7, 15, 30]) -> pd.DataFrame:
    """
    构建动量特征

    对每个原始因子，计算其在不同窗口期的滚动均值和滚动标准差。
    - 滚动均值：反映因子动量（过去 N 天平均表现）
    - 滚动标准差：反映因子波动率（稳定性）

    动量特征可以帮助模型捕捉因子的趋势和周期性。

    Args:
        factor_df: 原始因子 DataFrame
        windows: 滚动窗口列表，默认 [7, 15, 30] 天

    Returns:
        增加了动量特征的 DataFrame
    """
    logger.info(f"[TRANSFORMER] Building momentum features: {len(factor_df.columns)} factors, windows={windows}")
    new_cols = {}
    for col in factor_df.columns:
        s = factor_df[col]
        for w in windows:
            mom = (s.groupby(level=0)
                   .rolling(w, min_periods=w)
                   .mean()
                   .reset_index(level=0, drop=True))
            vol = (s.groupby(level=0)
                   .rolling(w, min_periods=w)
                   .std()
                   .reset_index(level=0, drop=True))
            new_cols[f"{col}_MOM_{w}"] = mom
            new_cols[f"{col}_VOL_{w}"] = vol
    result = pd.concat([factor_df, pd.DataFrame(new_cols, index=factor_df.index)], axis=1)
    logger.info(f"[TRANSFORMER] Momentum features created: {len(result.columns)} total columns")
    return result


def build_ic_time_series(factor_df: pd.DataFrame, returns: pd.DataFrame,
                         ic_windows=[20, 60]) -> pd.DataFrame:
    """
    构建 IC 时序特征

    注意：此函数目前被跳过，不生成训练特征，仅用于分析目的。
    IC 时序特征可以帮助分析因子的时效性和衰减特性。

    IC 时序特征包括：
    - 滚动 IC：因子在最近 N 天的 IC 均值
    - IC 衰减：不同持有期的 IC 变化

    Args:
        factor_df: 因子 DataFrame
        returns: 收益率 DataFrame
        ic_windows: 滚动 IC 窗口列表

    Returns:
        原始 factor_df（未添加新特征）
    """
    logger.info(f"[TRANSFORMER] build_ic_time_series skipped (not a proper training feature, use for analysis only)")
    return factor_df


def build_cross_sectional_rank_features(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建截面排名特征

    对每个日期截面内的因子值进行排名并标准化到 [0, 1]。
    截面排名比原始值更有预测力，因为它消除了截面尺度差异。

    生成的特征以 _RANK 后缀命名，替代原始特征的部分信息。

    Args:
        factor_df: 因子 DataFrame，MultiIndex

    Returns:
        增加了截面排名特征的 DataFrame
    """
    logger.info(f"[TRANSFORMER] Building cross-sectional rank features: {len(factor_df.columns)} factors")
    idx = factor_df.index
    date_level = 1 if idx.nlevels == 2 and (
        idx.names[0] in ("instrument", "Instrument", "code", "asset") or
        (idx.names[0] is None and not np.issubdtype(idx.get_level_values(0).dtype, np.datetime64))
    ) else 0

    ranked_cols = {}
    for col in factor_df.columns:
        ranked = factor_df.groupby(level=date_level)[col].rank(pct=True)
        ranked_cols[f"{col}_RANK"] = ranked

    result = pd.concat([factor_df, pd.DataFrame(ranked_cols, index=factor_df.index)], axis=1)
    logger.info(f"[TRANSFORMER] Cross-sectional rank features created: {len(result.columns)} total columns")
    return result


def build_reversal_features(factor_df: pd.DataFrame, windows=[3, 5, 10]) -> pd.DataFrame:
    """
    构建短期反转特征

    基于动量因子的排名，构建反转信号：
    - 取 N 日收益排名的低分位（即将反弹的股票）
    - 加入更短窗口的反转信号

    Args:
        factor_df: 因子 DataFrame
        windows: 反转窗口列表

    Returns:
        增加了反转特征的 DataFrame
    """
    logger.info(f"[TRANSFORMER] Building reversal features: {len(factor_df.columns)} factors, windows={windows}")
    rev_cols = {}
    # 通过对原始动量因子取负，构建反转信号
    momentum_cols = [col for col in factor_df.columns if col.startswith("RET_")]
    for col in momentum_cols:
        rev_cols[f"{col}_REV"] = -factor_df[col]

    result = pd.concat([factor_df, pd.DataFrame(rev_cols, index=factor_df.index)], axis=1)
    logger.info(f"[TRANSFORMER] Reversal features created: {len(result.columns)} total columns")
    return result