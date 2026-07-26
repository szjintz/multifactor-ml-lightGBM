"""
特征重要性跟踪模块

在 Walk-Forward 训练过程中跟踪特征重要性的变化，
帮助识别：
1. 哪些因子在长期有效
2. 哪些因子随时间衰减
3. 市场环境变化时哪些因子发生突变

功能：
1. 记录每个 fold 的特征重要性
2. 绘制重要性热力图（时间 × 特征）
3. 识别重要性衰减的因子
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ImportanceTracker:
    """
    特征重要性跟踪器

    在滚动训练过程中记录每个模型的特征重要性，
    用于时序分析和因子衰减检测。
    """

    def __init__(self):
        """
        初始化跟踪器
        """
        self.importance_history = []

    def record(self, model, date: str, feature_names: list):
        """
        记录单个模型的特征重要性

        Args:
            model: 训练好的 LightGBM 模型
            date: 对应的日期
            feature_names: 特征名称列表
        """
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=feature_names
        )
        importance = importance / importance.sum()  # 归一化为概率分布
        importance["date"] = date
        self.importance_history.append(importance)

    def get_importance_matrix(self) -> pd.DataFrame:
        """
        获取重要性矩阵

        Returns:
            DataFrame，行=日期，列=特征（不含 date 列）
        """
        df = pd.DataFrame(self.importance_history)
        df = df.set_index("date")
        return df

    def plot_heatmap(self, top_n=20):
        """
        绘制特征重要性热力图

        纵轴：特征名称
        横轴：时间（日期）
        颜色：重要性强度

        Args:
            top_n: 显示前 N 个重要性的特征
        """
        imp_matrix = self.get_importance_matrix()
        top_features = imp_matrix.mean().nlargest(top_n).index
        plt.figure(figsize=(12, 8))
        sns.heatmap(imp_matrix[top_features].T, cmap="YlOrRd", cbar_kws={"label": "Importance"})
        plt.title(f"Feature Importance Time-Series (Top {top_n})")
        plt.xlabel("Date")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("results/importance_heatmap.png", dpi=150)
        plt.close()

    def get_decaying_features(self, window=10) -> list:
        """
        识别重要性衰减的因子

        比较最近 window 期与最初 window 期的平均重要性。
        重要性下降明显的因子可能已经失效。

        Args:
            window: 比较的期数

        Returns:
            衰减最严重的5个因子名称列表
        """
        imp_matrix = self.get_importance_matrix()
        if len(imp_matrix) < window * 2:
            return []
        recent = imp_matrix.iloc[-window:].mean()
        early = imp_matrix.iloc[:window].mean()
        decay = early - recent  # 正值表示早期高，近期低（衰减）
        return list(decay.nlargest(5).index)