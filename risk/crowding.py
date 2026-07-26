"""
因子拥挤度监控模块

检测市场拥挤情况，避免在过度拥挤的因子上下注。

拥挤度衡量：
- 因子收益与资金流的相关性
- 高度拥挤的因子可能面临拥挤交易风险

使用方式：
- 监控因子收益与资金流的相关性
- 相关性高的因子自动降权或剔除
"""

import pandas as pd


class CrowdingMonitor:
    """
    因子拥挤度监控器

    通过监测因子收益与资金流的相关性来判断因子拥挤程度。
    """

    def __init__(self, lookback=60):
        """
        初始化拥挤度监控器

        Args:
            lookback: 计算相关性的回望期（天）
        """
        self.lookback = lookback

    def compute_crowding(self, factor_returns: pd.DataFrame, flow_data: pd.DataFrame = None) -> pd.Series:
        """
        计算每个因子的拥挤度得分

        拥挤度 = 因子收益与资金流相关性的绝对值在回望期内的均值

        Args:
            factor_returns: 因子收益 DataFrame
            flow_data: 资金流数据 DataFrame（如果有）

        Returns:
            拥挤度 Series，因子名为索引
        """
        if flow_data is not None:
            crowding = {}
            for col in factor_returns.columns:
                if col in flow_data.columns:
                    corr = factor_returns[col].rolling(self.lookback).corr(flow_data[col])
                    crowding[col] = float(corr.abs().mean())
                else:
                    crowding[col] = 0.0
            return pd.Series(crowding)
        # 如果没有资金流数据，返回零
        return pd.Series(0.0, index=factor_returns.columns)

    def filter_crowded(self, factor_list: list, crowding_scores: pd.Series, threshold=0.6) -> list:
        """
        过滤拥挤因子

        移除拥挤度超过阈值的因子。

        Args:
            factor_list: 因子名称列表
            crowding_scores: 拥挤度得分 Series
            threshold: 拥挤度阈值，默认 0.6

        Returns:
            过滤后的因子列表
        """
        return [f for f in factor_list if crowding_scores.get(f, 0) < threshold]