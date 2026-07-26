"""
预测器模块

使用训练好的 Walk-Forward 模型进行批量预测。
支持多模型并行预测，结果按日期拼接。
"""

import pandas as pd


class Predictor:
    """
    预测器

    封装多个 Walk-Forward 模型，提供批量预测接口。
    """

    def __init__(self, models: list, dates: list):
        """
        初始化预测器

        Args:
            models: 训练好的 LightGBM 模型列表
            dates: 每个模型对应的测试起始日期列表
        """
        self.models = models
        self.dates = dates

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        批量预测

        对每个模型，在其对应日期的特征子集上进行预测，
        并拼接所有预测结果。

        Args:
            features: 特征 DataFrame，MultiIndex (date, instrument)

        Returns:
            预测 Series，MultiIndex (date, instrument)
        """
        preds = []
        for model, date in zip(self.models, self.dates):
            X = features.loc[features.index.get_level_values(0) == date]
            if len(X) == 0:
                continue
            pred = model.predict(X.values)
            preds.append(pd.Series(pred, index=X.index))
        return pd.concat(preds) if preds else pd.Series(dtype=float)