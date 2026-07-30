"""
因子预处理管线模块

整合预处理流程，提供一键式因子预处理。
预处理步骤：
1. 异常值处理（Winsorize）
2. 截面标准化（Cross-sectional Standardization）
3. 中性化（Neutralization）- 可选
4. 正交化（Orthogonalization）- 可选

设计原则：
- 所有处理都是按截面进行的，保证时序安全性
- 中性化和正交化是可选的，由配置控制
"""

import logging
import numpy as np
import pandas as pd
from .base import FactorPipeline
from .preprocessing import winsorize, cross_sectional_standardize, neutralize, orthogonalize

logger = logging.getLogger(__name__)


class FactorPreprocessingPipeline:
    """
    因子预处理管线

    整合所有预处理步骤，提供端到端的因子预处理能力。
    配置参数通过 config 字典传入。
    """

    def __init__(self, config: dict):
        """
        初始化预处理管线

        Args:
            config: 预处理配置字典，结构示例：
                {
                    "winsorize": "3sigma",  # 或 "quantile"
                    "neutralize": ["market_cap", "industry"],  # 可选
                    "orthogonalize": "gram_schmidt"  # 可选
                }
        """
        self.config = config
        logger.info(f"[PREPROCESS] Initializing preprocessing pipeline, config: {config}")

    def process(self, factor_df: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        执行完整的预处理流程

        步骤：
        1. Winsorize：缩尾处理异常值
        2. Standardize：截面 Z-score 标准化
        3. Neutralize：市值/行业中性化（可选）
        4. Orthogonalize：因子正交化（可选）

        Args:
            factor_df: 原始因子 DataFrame
            market_data: 市场数据（用于中性化），包含市值和行业信息

        Returns:
            预处理后的因子 DataFrame
        """
        logger.info(f"[PREPROCESS] Starting preprocessing: {len(factor_df.columns)} factors, {len(factor_df)} rows")
        result = factor_df.copy()

        # 步骤 1：Winsorize 缩尾处理
        winsorize_cfg = self.config.get("winsorize", "3sigma")
        logger.info(f"[PREPROCESS] Step 1/4: Winsorizing, method={winsorize_cfg}")
        for col in result.columns:
            result[col] = winsorize(result[col], winsorize_cfg)
        logger.info(f"[PREPROCESS] Winsorization complete")

        # 步骤 2：截面标准化
        logger.info(f"[PREPROCESS] Step 2/4: Cross-sectional standardization")
        result = cross_sectional_standardize(result)
        # 截面分组（按日期分组填充）
        result = result.groupby(level=0).transform(lambda x: x.fillna(x.median()))
        logger.info(f"[PREPROCESS] Standardization complete: shape={result.shape}")
        # 标准化完成后立刻打印因子状态
        total_num = result.size
        nan_num = np.isnan(result.values).sum()
        inf_num = np.isinf(result.values).sum()
        zero_num = np.sum(result.values == 0)
        logger.info(f"[DEBUG] After standardization: total={total_num}, NaN={nan_num}, Inf={inf_num}, all-zero={zero_num}")
        logger.info(f"[DEBUG] Valid non-null data count: {total_num - nan_num - inf_num}")


        # 步骤 3：中性化（可选）
        if self.config.get("neutralize") and market_data is not None:
            exog_cols = self.config["neutralize"]
            logger.info(f"[PREPROCESS] Step 3/4: Neutralizing, exogenous variables={exog_cols}")
            exog_data = pd.DataFrame(index=market_data.index)
            found = []
            missing = []
            # 日志：显示 market_data 中有哪些列可用（用于调试）
            logger.info(f"[PREPROCESS] market_data available columns: {list(market_data.columns)}")
            # 构建不区分大小写的列名映射
            col_map = {c.lower(): c for c in market_data.columns}
            for col_name in exog_cols:
                if col_name == "market_cap":
                    if "market_cap" in market_data.columns:
                        exog_data["market_cap"] = np.log(market_data["market_cap"].replace(0, np.nan))
                        found.append("market_cap")
                    elif "close" in market_data.columns and "outstanding_share" in market_data.columns:
                        exog_data["market_cap"] = np.log(
                            (market_data["close"] * market_data["outstanding_share"]).replace(0, np.nan)
                        )
                        found.append("market_cap")
                    else:
                        missing.append("market_cap")
                elif col_name == "industry":
                    if "industry" in market_data.columns:
                        dummies = pd.get_dummies(market_data["industry"], prefix="ind", drop_first=True)
                        for dcol in dummies.columns:
                            exog_data[dcol] = dummies[dcol]
                        found.append("industry (one-hot)")
                    else:
                        missing.append("industry")
                elif col_name in market_data.columns:
                    exog_data[col_name] = market_data[col_name]
                    found.append(col_name)
                elif col_name.lower() in col_map:
                    actual_name = col_map[col_name.lower()]
                    exog_data[col_name] = market_data[actual_name]
                    found.append(f"{col_name}->{actual_name}")
                else:
                    missing.append(col_name)
            if missing:
                logger.warning(f"[PREPROCESS] Neutralization columns not found: {missing}")
            if len(found) == 0:
                logger.warning(f"[PREPROCESS] No neutralization data available, skipping")
            else:
                logger.info(f"[PREPROCESS] Neutralization data prepared: {found}")
                # Log how many market_cap values are valid before neutralizing
                if "market_cap" in exog_data.columns:
                    valid_mc = exog_data["market_cap"].notna().sum()
                    logger.info(f"[PREPROCESS] market_cap non-NaN: {valid_mc}/{len(exog_data)}")
                for col in result.columns:
                    result[col] = neutralize(result[col], exog_data)
                logger.info(f"[PREPROCESS] Neutralization complete")
        else:
            logger.info(f"[PREPROCESS] Step 3/4: Skipped (no neutralization config or market_data)")

        # 步骤 4：正交化（可选）
        orth = self.config.get("orthogonalize")
        if orth and orth != "none":
            logger.info(f"[PREPROCESS] Step 4/4: Orthogonalization against {orth}")
            result = orthogonalize(result, orth)
            logger.info(f"[PREPROCESS] Orthogonalization complete")
        else:
            logger.info(f"[PREPROCESS] Step 4/4: Skipped (no orthogonalize config)")

        # 记录 NaN 比例用于监控
        nan_pct = result.isna().sum().sum() / result.size * 100
        logger.info(f"[PREPROCESS] Preprocessing complete: {len(result.columns)} factors, {len(result)} rows, {nan_pct:.2f}% NaN")
        return result