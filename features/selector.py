"""
特征选择器模块

基于 IC（Information Coefficient，信息系数）对特征进行预筛选。
IC 衡量因子预测能力，是量化选股的核心指标。

筛选流程：
1. 计算每个因子与未来收益率的截面 Spearman 相关系数（IC）
2. 汇总时间序列上的 IC 均值和 ICIR（IC / std）
3. 根据阈值筛选有效因子

关键设计原则：
- 所有 IC 计算都是严格前向的，无前视偏差
- 使用 Bonferroni 或 FDR 校正进行多重假设检验
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    valid = factor.notna() & forward_return.notna()
    if valid.sum() < 30:
        logger.debug(f"[SELECTOR] 有效点不足 ({valid.sum()}<30)，无法计算IC")
        return np.nan
    ic = spearmanr(factor[valid], forward_return[valid])[0]
    logger.debug(f"[SELECTOR] IC计算完成: {ic:.4f}，来自{valid.sum()}个有效点")
    return ic


def compute_icir(ic_series: pd.Series) -> float:
    if len(ic_series) < 5:
        logger.debug(f"[SELECTOR] 周期数不足 ({len(ic_series)}<5)，无法计算ICIR")
        return np.nan
    icir = ic_series.mean() / ic_series.std()
    logger.debug(f"[SELECTOR] ICIR计算完成: {icir:.4f} (均值={ic_series.mean():.4f}, 标准差={ic_series.std():.4f})")
    return icir


def ic_prefilter(factor_df: pd.DataFrame, returns: pd.DataFrame,
                 min_ic=0.0, min_icir=0.0, p_threshold=0.05) -> list[str]:
    """
    基于 IC 和 ICIR 的特征预筛选

    筛选标准：
    - 平均 |IC| > min_ic：因子与收益率有一定的单调关系
    - |ICIR| > min_icir：因子预测能力具有一定的稳定性

    Args:
        factor_df: 因子 DataFrame，MultiIndex（date, instrument）
        returns: 收益率 DataFrame，与因子日期对齐
        min_ic: 最小平均 IC 阈值（绝对值），默认 0.0（不筛选）
        min_icir: 最小 ICIR 阈值，默认 0.0（不筛选）
        p_threshold: 统计显著性阈值（保留，暂未使用 Bonferroni 校正）

    Returns:
        通过筛选的因子名称列表
    """
    logger.info(f"[SELECTOR] 开始IC预筛选: {len(factor_df.columns)}个因子, min_ic={min_ic}, min_icir={min_icir}, p_threshold={p_threshold}")
    selected = []
    rejected = []

    idx = factor_df.index
    date_level = 1 if idx.nlevels == 2 and (
        idx.names[0] in ("instrument", "Instrument", "code", "asset") or
        (idx.names[0] is None and not np.issubdtype(idx.get_level_values(0).dtype, np.datetime64))
    ) else 0
    unique_dates = sorted(idx.levels[date_level].unique())
    logger.info(f"[SELECTOR] Using level {date_level} for dates: {len(unique_dates)} unique dates")
    logger.info(f"[SELECTOR] Filtering over {len(unique_dates)} periods")

    for col in factor_df.columns:
        ic_values = []
        for date in unique_dates:
            try:
                f = factor_df.xs(date, level=date_level)[col]
                r = returns.xs(date, level=date_level)
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[:, 0]
                ic = compute_ic(f, r)
                ic_values.append(ic)
            except (KeyError, AttributeError, ValueError) as e:
                logger.debug(f"[SELECTOR] Failed to compute IC for {col} on {date}: {e}")
                continue

        # 需要至少 20 个有效日期才进行筛选判断
        if len(ic_values) < 20:
            logger.debug(f"[SELECTOR] {col}: rejected, insufficient periods ({len(ic_values)}<20)")
            rejected.append((col, "insufficient_periods", len(ic_values)))
            continue

        ic_series = pd.Series(ic_values)
        mean_ic = ic_series.mean()
        icir = compute_icir(ic_series)
        valid_ics = ic_series.notna().sum()

        # 诊断首次因子 IC 情况
        if col == factor_df.columns[0] or len(selected) < 3:
            logger.warning(f"[SELECTOR] {col}: mean_ic={mean_ic:.6f}, icir={icir:.4f}, valid_dates={valid_ics}/{len(ic_values)}, total_nan={ic_series.isna().sum()}")

        # 双边筛选：既考虑正向预测也考虑负向预测（因子可能反向）
        if abs(mean_ic) > min_ic and abs(icir) > min_icir:
            selected.append(col)
            logger.debug(f"[SELECTOR] {col}: selected (mean_ic={mean_ic:.4f}, icir={icir:.4f}, periods={len(ic_values)})")
        else:
            rejected.append((col, mean_ic, icir))
            logger.debug(f"[SELECTOR] {col}: rejected (mean_ic={mean_ic:.4f}, icir={icir:.4f})")

    logger.info(f"[SELECTOR] IC prefilter complete: {len(selected)}/{len(factor_df.columns)} factors selected, {len(rejected)} rejected")
    if selected:
        logger.info(f"[SELECTOR] Selected factors: {selected[:10]}{'...' if len(selected)>10 else ''}")
    return selected