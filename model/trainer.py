import numpy as np
import pandas as pd
import lightgbm as lgb


class WalkForwardTrainer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.models: list[lgb.Booster] = []
        self.dates: list[str] = []

    def _get_train_test_dates(self, all_dates: list, window_size: int, step_size: int):
        splits = []
        for i in range(window_size, len(all_dates) - step_size, step_size):
            train_start = all_dates[i - window_size]
            train_end = all_dates[i]
            test_start = all_dates[i]
            test_end = all_dates[min(i + step_size, len(all_dates) - 1)]
            splits.append((train_start, train_end, test_start, test_end))
        return splits

    def train(self, features: pd.DataFrame, labels: pd.Series, dates: list) -> pd.Series:
        cfg = self.config.get("training", {})
        window_size = cfg.get("window_months", 24) * 21
        step_size = 21
        splits = self._get_train_test_dates(dates, window_size, step_size)

        all_predictions = []
        for train_start, train_end, test_start, test_end in splits:
            train_mask = (features.index.get_level_values(0) >= train_start) & \
                         (features.index.get_level_values(0) < train_end)
            test_mask = (features.index.get_level_values(0) >= test_start) & \
                        (features.index.get_level_values(0) < test_end)

            X_train = features[train_mask]
            y_train = labels[train_mask]
            X_test = features[test_mask]

            if len(X_train) < 100 or len(X_test) < 10:
                continue

            val_size = max(int(len(X_train) * 0.2), 1)
            X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
            X_train, y_train = X_train.iloc[:-val_size], y_train.iloc[:-val_size]

            dtrain = lgb.Dataset(
                X_train.values, label=y_train.values,
                feature_name=list(X_train.columns)
            )
            dval = lgb.Dataset(
                X_val.values, label=y_val.values,
                reference=dtrain,
            )

            lgb_params = cfg.get("lgb_params", {})
            params = {
                "objective": cfg.get("objective", "lambdarank"),
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

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(
                    cfg.get("early_stopping_rounds", 30),
                    first_metric_only=True
                )],
            )

            self.models.append(model)
            self.dates.append(test_start)
            pred = model.predict(X_test.values)
            pred_series = pd.Series(pred, index=X_test.index)
            all_predictions.append(pred_series)

        return pd.concat(all_predictions) if all_predictions else pd.Series(dtype=float)
