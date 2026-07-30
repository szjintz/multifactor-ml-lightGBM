"""
特征标签生成模块

用于计算机器学习模型的标签（目标变量）。
在多因子量化策略中，标签通常是未来一段时间的股票收益率。

关键设计原则：
- 所有标签都是前向计算的（shift 为负），确保无前视偏差
- 支持多种标签计算方式：原始收益率、降噪收益率
- 支持自定义预测周期和跳过天数
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_forward_return(close: pd.DataFrame, periods: int = 20, skip: int = 1) -> pd.DataFrame:
    """
    计算未来持有期的收益率

    标签 = (T+skip+periods 收盘价 - T+skip 收盘价) / T+skip 收盘价

    这代表了从 T+skip+1 日到 T+skip+periods 日的累计收益率。
    跳过 skip 天的原因：
    - T 日收盘价已经确定，但 T+1 日开盘价可能有跳空
    - 避免 T 日收盘价和 T+1 日开盘价之间的日内效应污染标签

    Args:
        close: 收盘价 DataFrame，宽格式（列为股票，行为日期）
        periods: 持有期天数，默认20天（约一个月）
        skip: 跳过天数，默认1天，避免 T+1 开盘跳空影响

    Returns:
        未来收益率 DataFrame，形状与 close 相同
    """
    logger.info(f"[LABEL] Computing forward returns: close shape={close.shape}, periods={periods}, skip={skip}")
    # shifted_close 是持有期开始的参考价格
    shifted_close = close.shift(-skip)
    # future_close 是持有期结束的价格
    future_close = close.shift(-periods - skip + 1)
    # 计算收益率
    returns = (future_close - shifted_close) / shifted_close
    valid_pct = returns.notna().sum().sum() / returns.size * 100
    logger.info(f"[LABEL] Forward returns computed: shape={returns.shape}, {valid_pct:.1f}% non-NaN")
    return returns


def denoise_label(labels: pd.Series, method: str = None) -> pd.Series:
    """
    标签降噪处理

    当标签（收益率）噪声过高时，可以使用平滑方法减少噪声。
    常用场景：高频数据噪声大，需要平滑处理。

    Args:
        labels: 原始标签序列
        method: 降噪方法，目前支持 "ewm"（指数加权移动平均）

    Returns:
        降噪后的标签序列
    """
    if method == "ewm":
        logger.info(f"[LABEL] Denoising labels with EWM span=5")
        return labels.ewm(span=5).mean()
    logger.debug(f"[LABEL] No denoising applied (method={method})")
    return labels


def compute_excess_forward_return(close: pd.DataFrame, periods: int = 20, skip: int = 1) -> pd.DataFrame:
    """
    计算市场超额前向收益（个股收益 - 截面均值收益）

    在横截面上对每个日期减去市场平均收益，消除市场整体波动的影响，
    使标签聚焦于股票自身的 alpha（选股能力）。

    Args:
        close: 收盘价 DataFrame，宽格式（列为股票，行为日期）
        periods: 持有期天数，默认20天
        skip: 跳过天数，默认1天

    Returns:
        超额收益率 DataFrame，形状与 close 相同
    """
    logger.info(f"[LABEL] Computing excess forward returns: close shape={close.shape}, periods={periods}, skip={skip}")
    raw_returns = compute_forward_return(close, periods, skip)
    market_mean = raw_returns.mean(axis=1)
    excess = raw_returns.sub(market_mean, axis=0)
    valid_pct = excess.notna().sum().sum() / excess.size * 100
    logger.info(f"[LABEL] Excess forward returns computed: shape={excess.shape}, {valid_pct:.1f}% non-NaN")
    return excess


def compute_labels(close: pd.DataFrame, periods=20, skip=1, denoise=None, relative=False) -> pd.DataFrame:
    """
    计算标签的主函数

    整合了前向收益率计算和降噪处理。
    支持绝对收益和相对收益（市场超额）两种模式。

    默认使用原始前向收益（relative=False）：
    - 截面 Z-score 已在 WalkForwardTrainer 中应用（按level=1分组标准化）
    - 在标签层再减去截面均值会使信号过度去均值，损害训练效果
    - relative=True 仅在需要纯 alpha 标签且不在 trainer 内做 Z-score 时使用

    Args:
        close: 收盘价 DataFrame
        periods: 持有期天数
        skip: 跳过天数
        denoise: 降噪方法，None 表示不降噪
        relative: 是否使用市场超额收益（默认 False），True 则使用原始前向收益

    Returns:
        标签 DataFrame
    """
    logger.info(f"[LABEL] Computing labels: close shape={close.shape}, periods={periods}, skip={skip}, denoise={denoise}, relative={relative}")
    if relative:
        labels = compute_excess_forward_return(close, periods, skip)
    else:
        labels = compute_forward_return(close, periods, skip)
    if denoise:
        labels = denoise_label(labels, denoise)
    logger.info(f"[LABEL] Labels computed: shape={labels.shape}, date range: {labels.index.min()} to {labels.index.max()}")
    return labels