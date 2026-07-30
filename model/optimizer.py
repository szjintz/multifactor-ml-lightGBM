"""
模型超参数优化模块

使用 Optuna 框架对 LightGBM 模型进行超参数调优。
采用 TPE (Tree-structured Parzen Estimator) 采样器最大化 IC (Information Coefficient)。
"""

import logging
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    基于 Optuna 的 LightGBM 超参数调优器

    使用贝叶斯优化方法（TPE 采样器）在给定的超参数空间内搜索最优参数组合，
    目标是最小化验证集上的负 IC（即最大化 IC 绝对值）。

    Attributes:
        n_trials: 搜索的试验次数，默认50次
        best_params: 搜索到的最优超参数字典
    """

    def __init__(self, n_trials: int = 10):
        """
        初始化 Optuna 调优器

        Args:
            n_trials: 最大试验次数，默认10次（内存优化）
        """
        self.n_trials = n_trials
        self.best_params = None
        logger.info(f"[OPTUNA] OptunaTuner initialized: {n_trials} trials")

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna 目标函数：单次试验评估

        每次试验从预定义的超参数空间采样一组参数，训练 LightGBM 模型，
        并在验证集上计算 IC 作为目标值。

        Args:
            trial: Optuna 试验对象，用于采样超参数
            X_train: 训练特征矩阵
            y_train: 训练标签向量
            X_val: 验证特征矩阵
            y_val: 验证标签向量

        Returns:
            IC 值（保留符号，Optuna 最大化）
        """
        # 定义超参数搜索空间
        # 正则化上限设为 0.1 —— 过强的 reg 会让 best_iter=1、模型学到几乎常数，
        # 信号被压制变成「不再学习」。0.1 已足够防止过拟合，且保留 fold-1 训练空间。
        params = {
            "objective": "regression",  # 回归目标更适合 IC 优化
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 0.1, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.3),
            "verbosity": -1,
        }

        # 创建 LightGBM 数据集（回归模式不需要 group）
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # 训练 LightGBM 模型
        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # 在验证集上预测并计算 IC（使用 Pearson 相关衡量预测与实际收益的相关性）
        pred = model.predict(X_val)
        ic, _ = spearmanr(pred, y_val)
        if np.isnan(ic):
            ic = 0.0
        logger.debug(f"[OPTUNA] Trial {trial.number}: IC={ic:.4f}")

        # 返回 IC（保留符号，Optuna 最大化）
        return ic

    def tune(self, X_train, y_train, X_val, y_val):
        """
        执行完整的超参数搜索过程

        使用 TPE 采样器进行贝叶斯优化搜索最优超参数。
        采样器使用随机种子42确保结果可复现。

        Args:
            X_train: 训练特征矩阵
            y_train: 训练标签向量
            X_val: 验证特征矩阵
            y_val: 验证标签向量

        Returns:
            最优超参数字典
        """
        logger.info(f"[OPTUNA] Starting hyperparameter tuning: {self.n_trials} trials, train={len(X_train)}, val={len(X_val)}")

        # 创建优化研究，direction="maximize" 表示最大化目标函数值
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        # 执行优化搜索，gc_after_trial=True 在每次试验后强制垃圾回收以节省内存
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            gc_after_trial=True,
        )

        # 保存最优参数和目标值
        self.best_params = study.best_params
        logger.info(f"[OPTUNA] Tuning complete: best_value={study.best_value:.4f}, best_params={self.best_params}")
        return self.best_params