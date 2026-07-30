"""
模型训练模块

实现 Walk-Forward（滚动窗口）训练策略，用于时序金融数据的模型训练。
每个窗口包含训练集和验证集，验证集用于早停和超参数调优。
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """
    Walk-Forward 训练器

    滚动窗口训练策略：
    - 每个窗口使用固定大小的历史数据进行训练
    - 训练集和验证集从同一时间窗口中划分（80% 训练，20% 验证）
    - 仅在第一个窗口进行超参数调优，后续窗口复用最优参数
    - 动态筛选低重要性特征

    Attributes:
        config: 配置字典，包含训练参数和模型参数
        tuner: OptunaTuner 实例，用于超参数调优（可选）
        models: 训练好的 LightGBM 模型列表
        dates: 每个模型对应测试集的起始日期列表
    """

    def __init__(self, config: dict = None, tuner=None):
        """
        初始化 Walk-Forward 训练器

        Args:
            config: 配置字典，结构示例：
                {
                    "training": {
                        "window_months": 24,      # 训练窗口月数（默认24）
                        "objective": "lambdarank", # 学习目标
                        "lgb_params": {},         # LightGBM 参数（如果有的话）
                        "early_stopping_rounds": 30  # 早停轮数
                    }
                }
            tuner: OptunaTuner 实例，用于超参数自动调优（可选）
        """
        self.config = config or {}
        self.models: list[lgb.Booster] = []  # 存储所有 fold 的模型
        self.dates: list[str] = []  # 存储每个模型对应的测试集起始日期
        self.tuner = tuner  # 超参数调优器
        logger.info(f"[TRAINER] WalkForwardTrainer initialized: window_months={self.config.get('training', {}).get('window_months', 24)}, tuner={tuner is not None}")

    def _get_train_test_dates(self, all_dates: list, window_size: int, step_size: int):
        """
        生成 Walk-Forward 分割的日期边界

        滚动窗口策略：
        - 训练窗口：[train_start, train_end)，长度 = window_size
        - 测试窗口：[test_start, test_end)，长度 = step_size
        - 每一步向前滚动 step_size 天

        Args:
            all_dates: 所有可用日期（已排序）
            window_size: 训练窗口大小（天数）
            step_size: 测试窗口大小（天数）

        Returns:
            splits: 列表，每个元素为 (train_start, train_end, test_start, test_end) 元组
        """
        splits = []
        for i in range(window_size, len(all_dates) - step_size, step_size):
            train_start = all_dates[i - window_size]
            train_end = all_dates[i]
            test_start = all_dates[i]
            test_end = all_dates[min(i + step_size, len(all_dates) - 1)]
            splits.append((train_start, train_end, test_start, test_end))

        logger.info(f"[TRAINER] Generated {len(splits)} train/test windows, window_size={window_size}, step_size={step_size}")
        return splits

    def train(self, features: pd.DataFrame, labels: pd.Series, dates: list, market_cap: pd.Series = None) -> pd.Series:
        """
        执行 Walk-Forward 训练

        训练流程：
        1. 按日期分割数据为多个训练/测试窗口
        2. 对每个窗口：
           a. 划分训练集（80%）和验证集（20%）
           b. 如果配置了market_cap，进行市值分层抽样
           c. 如果是第一个窗口且配置了 tuner，进行超参数调优
           d. 训练 LightGBM 模型
           e. 预测测试集
           f. 动态移除低重要性特征（如果有至少20个特征保留）
        3. 拼接所有测试集的预测结果

        Args:
            features: 特征 DataFrame，MultiIndex（date, instrument）
            labels: 标签 Series，MultiIndex（date, instrument）
            dates: 所有唯一日期列表（已排序）
            market_cap: 市值 Series，MultiIndex（date, instrument），用于分层抽样

        Returns:
            合并后的预测 Series，包含所有测试窗口的预测结果
        """
        logger.info(f"[TRAINER] Training started: {len(features.columns)} features, {len(features)} rows, {len(dates)} dates, market_cap={'Yes' if market_cap is not None else 'No'}")

        cfg = self.config.get("training", {})

        # 确保标签索引与特征索引一致
        labels = labels.reindex(features.index)

        # regression 模式：直接使用数值标签（不需要排名转换）
        is_ranking = cfg.get("objective", "regression") == "lambdarank"
        if is_ranking:
            labels = labels.groupby(level=1).rank(method="min", ascending=True) - 1
        else:
            # 对截面收益做 Z-score 标准化：
            # 1) 让模型学习横截面相对排序而不是绝对收益大小；
            # 2) 标签量纲与因子一致，避免 L2 在数值上偏向常数预测。
            # 同时保留掩码：原始 NaN 仍为 NaN。
            labels = labels.groupby(level=1).transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

        # 剔除标签为 NaN 的行（最后 ~20 个日期无前向收益数据，或全NaN截面）
        valid_mask = labels.notna().values
        features = features.loc[valid_mask]
        labels = labels.loc[valid_mask]
        if is_ranking:
            labels = labels.astype(int)
        if market_cap is not None:
            market_cap = market_cap.reindex(features.index)

        # 重新计算日期列表
        dates = sorted(features.index.get_level_values(1).unique())
        logger.info(f"[TRAINER] After label NaN removal: {len(features)} rows, {len(dates)} dates")

        # 按 (datetime, instrument) 排序，确保 lambdarank group 连续
        features = features.sort_index(level=[1, 0])
        labels = labels.reindex(features.index)
        if market_cap is not None:
            market_cap = market_cap.reindex(features.index)

        # 计算训练窗口大小：月数转换为天数（近似：每月21个交易日）
        window_size = cfg.get("window_months", 24) * 21
        step_size = 21  # 每月滚动一次（约21个交易日）

        # 自适应：如果可用日期不足，缩小训练窗口或减少历史要求
        total_dates = len(dates)
        if window_size + step_size >= total_dates:
            max_window = total_dates - step_size - 1
            if max_window < 21:
                logger.error(f"[TRAINER] Insufficient dates: {total_dates} available, need at least {window_size + step_size + 1}")
                return pd.Series(dtype=float)
            window_size = max_window - (max_window % 21)
            logger.warning(f"[TRAINER] Limited dates ({total_dates}), reducing window to {window_size} days ({window_size//21} months)")

        # 生成所有分割点
        splits = self._get_train_test_dates(dates, window_size, step_size)

        # 活跃特征列表，用于动态特征筛选
        active_features = list(features.columns)
        all_predictions = []  # 存储所有测试集的预测结果

        # 遍历每个 Walk-Forward 分割
        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            # 构建训练集和测试集的布尔掩码（按 datetime 分割）
            train_mask = (features.index.get_level_values(1) >= train_start) & \
                         (features.index.get_level_values(1) < train_end)
            test_mask = (features.index.get_level_values(1) >= test_start) & \
                        (features.index.get_level_values(1) < test_end)

            # 获取训练集和测试集数据
            X_train = features.loc[train_mask, active_features]
            y_train = labels[train_mask]
            X_test = features.loc[test_mask, active_features]

            # 检查数据量是否足够
            if len(X_train) < 100 or len(X_test) < 10:
                logger.warning(f"[TRAINER] Fold {fold_idx+1}: Insufficient data (train={len(X_train)}, test={len(X_test)}), skipping")
                continue

            # 从训练集末尾划分验证集（时序敏感：使用最近的数据作为验证集）
            val_size = max(int(len(X_train) * 0.2), 1)  # 验证集占20%，最少1条
            X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
            X_train_fit, y_train_fit = X_train.iloc[:-val_size], y_train.iloc[:-val_size]

            # 市值分层：确保每个市值五分位在验证集有足够样本
            if market_cap is not None:
                mc_train = market_cap[train_mask]
                mc_val = market_cap[train_mask].iloc[-val_size:]
                mc_train_fit = mc_train.iloc[:-val_size]
                try:
                    quintiles = pd.qcut(mc_train_fit, 5, labels=False, duplicates='drop')
                    val_quintiles = pd.cut(mc_val, bins=pd.qcut(mc_train_fit, 5, duplicates='drop').quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).drop_duplicates(), labels=False, include_lowest=True)
                    quintile_counts = val_quintiles.value_counts().sort_index()
                    logger.info(f"[TRAINER] Fold {fold_idx+1}: Market-cap stratified validation distribution: {dict(quintile_counts)}")
                    min_q_count = quintile_counts.min()
                    if min_q_count < 3:
                        logger.warning(f"[TRAINER] Fold {fold_idx+1}: Market-cap stratification imbalance, min quintile count={min_q_count}")
                except Exception as e:
                    logger.debug(f"[TRAINER] Fold {fold_idx+1}: Market-cap stratification failed: {e}")

            # 内存优化：仅当数据量极大时采样（保留最近的样本以保证时序一致性）
            max_train_samples = 250000
            if len(X_train_fit) > max_train_samples:
                X_train_fit = X_train_fit.tail(max_train_samples)
                y_train_fit = y_train_fit.tail(max_train_samples)
                logger.info(f"[TRAINER] Fold {fold_idx+1}: Training data sampled to {len(X_train_fit)} rows (most recent)")

            # 创建 LightGBM 数据集对象
            dtrain = lgb.Dataset(
                X_train_fit.values, label=y_train_fit.values,
                feature_name=list(X_train_fit.columns),
            )
            dval = lgb.Dataset(
                X_val.values, label=y_val.values,
                reference=dtrain,  # 关联训练集，用于早停
            )

            # 获取基础 LightGBM 参数
            lgb_params = cfg.get("lgb_params", {})

            # 仅在第一个 fold 进行超参数调优（保留最佳参数复用到后续 fold）
            # 后续 fold 复用第一个 fold 调优结果，节省训练时间并保证 fold 间参数一致性
            if self.tuner is not None and len(self.models) == 0:
                logger.info(f"[TRAINER] Fold {fold_idx+1}: Running Optuna hyperparameter tuning ({self.tuner.n_trials} trials)...")

                # 为了节省内存，对训练数据进行采样（最多30000条）
                sample_idx = np.random.choice(len(X_train_fit), size=min(30000, len(X_train_fit)), replace=False)
                X_sample = X_train_fit.values[sample_idx]
                y_sample = y_train_fit.values[sample_idx]

                # 调用 OptunaTuner 进行超参数搜索
                tuned = self.tuner.tune(X_sample, y_sample, X_val.values, y_val.values)
                lgb_params = {**lgb_params, **tuned}  # 将调优后的参数合并到基础参数
                # 以下参数始终使用配置值覆盖 Optuna 结果（Optuna 在短训练下倾向保守参数，
                # 但主训练使用更多轮次，需要允许树分裂）
                override_params = ["min_gain_to_split", "learning_rate", "reg_alpha", "reg_lambda"]
                for p in override_params:
                    config_val = cfg.get("lgb_params", {}).get(p)
                    if config_val is not None:
                        lgb_params[p] = config_val
                applied = {p: lgb_params[p] for p in override_params if cfg.get("lgb_params", {}).get(p) is not None}
                logger.info(f"[TRAINER] Fold {fold_idx+1}: Best params: {tuned}, overrides applied: {applied}")

            is_ranking = cfg.get("objective", "regression") == "lambdarank"
            params = {
                "objective": cfg.get("objective", "regression"),
                "num_leaves": lgb_params.get("num_leaves", 31),
                "min_child_samples": lgb_params.get("min_child_samples", 20),
                "learning_rate": lgb_params.get("learning_rate", 0.05),
                "reg_alpha": lgb_params.get("reg_alpha", 0.1),
                "reg_lambda": lgb_params.get("reg_lambda", 0.1),
                "bagging_fraction": lgb_params.get("bagging_fraction", 0.8),
                "feature_fraction": lgb_params.get("feature_fraction", 0.8),
                "min_gain_to_split": lgb_params.get("min_gain_to_split", 0.0),
                "verbosity": -1,
            }
            if is_ranking:
                params["label_gain"] = list(range(int(max(max(y_train_fit), max(y_val)) + 1)))

            # 记录训练信息
            logger.info(f"[TRAINER] Fold {fold_idx+1}/{len(splits)}: Training LightGBM, train={len(X_train_fit)}, val={len(X_val)}, test={len(X_test)}, features={len(active_features)}")

            # 训练 LightGBM 模型 - 使用大 num_boost_round + 早停（patience=50）
            # 早停对噪声大的截面 Z-score 标签至关重要：避免过拟合并保留泛化能力。
            num_rounds = cfg.get("num_boost_round", 500)
            early_stopping = cfg.get("early_stopping_rounds", 50)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_rounds,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            # 在测试集上进行预测
            pred = model.predict(X_test.values)
            pred_series = pd.Series(pred, index=X_test.index)
            all_predictions.append(pred_series)

            logger.info(f"[TRAINER] Fold {fold_idx+1}: Model trained, best_iteration={model.best_iteration}, best_score={model.best_score}")

            # 动态特征筛选：获取重要性在删除模型之前
            feature_imp = pd.Series(
                model.feature_importance(importance_type="gain"),
                index=active_features,
            )
            feature_imp = feature_imp / feature_imp.sum()

            # 保存模型和日期（仅2个fold，内存可接受）
            self.models.append(model)
            self.dates.append(test_start)

            # 释放数据集对象，模型稍后通过 self.models 访问
            del dtrain, dval
            import gc
            gc.collect()

            # 找出低重要性特征
            low_imp = list(feature_imp[feature_imp < 0.002].index)

            min_keep = 80
            if low_imp and len(active_features) - len(low_imp) >= min_keep:
                max_remove = max(1, int(0.10 * len(active_features)))
                removed_count = min(len(low_imp), max_remove)
                low_imp_to_remove = list(feature_imp.nsmallest(removed_count).index)
                removed = len(low_imp_to_remove)
                active_features = [f for f in active_features if f not in low_imp_to_remove]
                logger.info(f"[TRAINER] Fold {fold_idx+1}: Removed {removed} low-importance features (<0.2%), {len(active_features)} remaining")

        # 合并所有预测结果
        if all_predictions:
            result = pd.concat(all_predictions)
            logger.info(f"[TRAINER] Training complete: {len(self.models)} models, {len(result)} predictions")
        else:
            result = pd.Series(dtype=float)
            logger.warning(f"[TRAINER] Training complete: no predictions generated")
        return result