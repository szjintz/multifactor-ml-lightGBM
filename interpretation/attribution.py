"""
归因分析模块

实现 Brinson 归因分析，将组合超额收益分解为：
1. 行业配置收益：偏离基准的行业权重带来的收益
2. 个股选择收益：行业内选股带来的收益
3. 交互效应：行业配置和选股效应的交叉

Brinson 归因公式：
    超额收益 = 行业配置收益 + 个股选择收益 + 交互效应

    行业配置收益 = Σ (w_p - w_b) * R_b
    个股选择收益 = Σ w_b * (R_p - R_b)
    交互效应   = Σ (w_p - w_b) * (R_p - R_b)

其中 w_p 是组合权重，w_b 是基准权重，R_p 是组合收益，R_b 是基准收益
"""

import pandas as pd
import numpy as np


class BrinsonAttribution:
    """
    Brinson 归因分析器

    将组合超额收益分解为行业配置和个股选择效应。
    """

    def __init__(self, sector_map: dict):
        """
        初始化归因分析器

        Args:
            sector_map: 股票到行业的映射，如 {"SH600000": "金融", "SZ000001": "科技"}
        """
        self.sector_map = sector_map

    def attribute(self, portfolio_weights: pd.Series, benchmark_weights: pd.Series,
                  stock_returns: pd.Series) -> dict:
        """
        执行归因分析

        Args:
            portfolio_weights: 组合权重 Series（股票名为索引）
            benchmark_weights: 基准权重 Series（股票名为索引）
            stock_returns: 股票收益 Series（股票名为索引）

        Returns:
            归因结果字典，包含：
            - allocation: 行业配置收益
            - selection: 个股选择收益
            - interaction: 交互效应
            - total: 总超额收益
            - allocation_detail: 各行业配置收益详情
            - selection_detail: 各行业选择收益详情
        """
        sectors = sorted(set(self.sector_map.values()))  # 排序确保可复现

        # 按行业汇总权重
        pw = portfolio_weights.groupby(self.sector_map).sum()
        bw = benchmark_weights.groupby(self.sector_map).sum()

        # 按行业计算收益率
        sr = stock_returns.groupby(self.sector_map).mean()
        # 组合在行业内的加权收益，避免除零
        pr = (portfolio_weights * stock_returns).groupby(self.sector_map).sum() / pw.replace(0, np.nan)

        # 计算各项收益
        allocation = (pw - bw) * sr  # 行业配置收益
        selection = bw * (pr - sr)  # 个股选择收益
        interaction = (pw - bw) * (pr - sr)  # 交互效应

        return {
            "allocation": float(allocation.sum()),
            "selection": float(selection.sum()),
            "interaction": float(interaction.sum()),
            "total": float(allocation.sum() + selection.sum() + interaction.sum()),
            "allocation_detail": allocation.to_dict(),
            "selection_detail": selection.to_dict(),
        }