"""
特征处理器模块

提供金融数据特征的各种预处理功能，包括：
- 缩尾处理（Winsorize）：使用 3σ 原则或分位数截断异常值
- 截面标准化（Z-score）：每个截面（日期）独立标准化
- 分位数变换：将特征值转换为均匀分布的排名

注意：
- 所有处理都是按截面（每日）独立进行的，避免前视偏差
- 标准化使用截面均值和标准差，而非全局值
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    特征预处理器

    提供多种特征处理方法，用于清洗和标准化输入特征。
    所有处理方法都是时序安全的，不会引入前视偏差。
    """

    @staticmethod
    def winsorize_series(s: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
        """
        分位数缩尾处理

        将特征值截断到指定分位数范围内，超出范围的数值被替换为边界值。

        Args:
            s: 输入的特征序列
            limits: 分位数边界，例如 (0.01, 0.99) 表示截断到 1%~99% 分位数

        Returns:
            缩尾处理后的序列
        """
        lower, upper = s.quantile(limits[0]), s.quantile(limits[1])
        logger.debug(f"[PROCESSOR] winsorize_series: bounds=[{lower:.4f}, {upper:.4f}]")
        return s.clip(lower, upper)

    @staticmethod
    def winsorize_3sigma(s: pd.Series) -> pd.Series:
        """
        3σ 原则缩尾处理

        将超出均值 ± 3 倍标准差范围的数值截断到边界值。
        这是统计学中常用的异常值处理方法。

        Args:
            s: 输入的特征序列

        Returns:
            缩尾处理后的序列
        """
        mean, std = s.mean(), s.std()
        lower, upper = mean - 3 * std, mean + 3 * std
        logger.debug(f"[PROCESSOR] winsorize_3sigma: mean={mean:.4f}, std={std:.4f}, bounds=[{lower:.4f}, {upper:.4f}]")
        return s.clip(lower, upper)

    @staticmethod
    def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """
        截面 Z-score 标准化

        对每个截面（每个日期），将其所有股票的特征值标准化为均值为0、标准差为1的分布。
        注意：必须使用截面标准差而非全局标准差，避免引入未来信息。

        Args:
            df: 特征 DataFrame，index 为 (datetime, instrument) 的 MultiIndex

        Returns:
            标准化后的 DataFrame
        """
        result = df.subtract(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, 1), axis=0)
        logger.debug(f"[PROCESSOR] cs_zscore: {df.shape[1]} columns normalized")
        return result

    @staticmethod
    def quantile_transform(s: pd.Series, n_quantiles=100) -> pd.Series:
        """
        分位数变换

        将特征值转换为 1 到 n_quantiles 的整数排名，
        使得变换后的值近似均匀分布，有助于减少极端值的影响。

        Args:
            s: 输入的特征序列
            n_quantiles: 分位数的数量，默认100

        Returns:
            分位数变换后的序列（整数排名）
        """
        ranks = s.rank(method="min")
        result = (ranks / ranks.max() * n_quantiles).astype(int)
        logger.debug(f"[PROCESSOR] quantile_transform: {n_quantiles} quantiles")
        return result

    def process(self, df: pd.DataFrame, method: str = "3sigma") -> pd.DataFrame:
        """
        完整的特征处理流程

        处理步骤：
        1. 缩尾处理（去除异常值）
        2. 分位数变换（统一尺度）
        3. 截面 Z-score 标准化

        Args:
            df: 输入的特征 DataFrame
            method: 缩尾方法，"3sigma" 或 "quantile"

        Returns:
            处理后的特征 DataFrame
        """
        logger.info(f"[PROCESSOR] Processing {len(df.columns)} features with method={method}")
        result = df.copy()
        for col in result.columns:
            # 第一步：缩尾处理
            if method == "3sigma":
                result[col] = self.winsorize_3sigma(result[col])
            elif method == "quantile":
                result[col] = self.winsorize_series(result[col])
            # 第二步：分位数变换
            result[col] = self.quantile_transform(result[col])
        # 第三步：截面 Z-score 标准化
        result = self.cs_zscore(result)
        logger.info(f"[PROCESSOR] Processing complete: {len(result.columns)} features")
        return result