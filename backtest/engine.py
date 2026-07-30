"""
回测引擎模块

实现组合回测的核心逻辑：
- 每日根据预测信号构建投资组合
- 使用 CVXPY 优化器进行组合优化
- 计算组合收益和交易成本
- 支持等权组合和优化组合两种模式

交易假设：
- T日收盘前根据 T 日收盘价生成预测
- T+1 日开盘执行交易（不考虑盘后交易）
- 交易成本：滑点 + 市场冲击成本
"""

import logging
import numpy as np
import pandas as pd

from portfolio.optimizer import CVXPYOptimizer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎

    模拟投资组合的日频调仓过程。
    支持两种组合构建方式：
    1. 优化组合：使用 CVXPY 求解均值-方差优化问题
    2. 等权组合：top_n 只股票等权配置
    """

    def __init__(self, config: dict):
        """
        初始化回测引擎

        Args:
            config: 配置字典，包含回测参数和组合参数
        """
        self.config = config
        backtest_cfg = config.get("backtest", {})
        training_cfg = config.get("training", {})
        self.slippage = backtest_cfg.get("slippage", 0.003)  # 滑点（双边）
        self.market_impact = backtest_cfg.get("market_impact", 0.001)  # 市场冲击
        self.holding_period = training_cfg.get("predict_days", 1)  # 持有期（与标签周期一致）
        self.optimizer = CVXPYOptimizer(config.get("portfolio", {}))
        self.lookback = 120  # 协方差矩阵估计的回望期（6个月，约120个交易日）
        self.LIMIT_THRESHOLD = 0.099  # 涨跌停阈值（9.9%，留0.1%安全边际）
        logger.info(f"[BACKTEST] BacktestEngine initialized: slippage={self.slippage}, market_impact={self.market_impact}, top_n={config.get('portfolio', {}).get('top_n', 50)}, holding_period={self.holding_period}, limit_threshold={self.LIMIT_THRESHOLD}")

    def _get_limit_stocks(self, prices: pd.DataFrame, date, all_dates, direction="up"):
        """
        获取涨跌停股票列表

        Args:
            prices: 价格 DataFrame
            date: 当前日期
            all_dates: 所有日期列表
            direction: "up" 表示涨停股票, "down" 表示跌停股票

        Returns:
            set of stock codes that hit limit up/down
        """
        date_idx = all_dates.index(date) if date in all_dates else -1
        if date_idx <= 0:
            return set()
        prev_date = all_dates[date_idx - 1]
        if prev_date not in prices.index:
            return set()

        try:
            curr_prices = prices.loc[date]
            prev_prices = prices.loc[prev_date]
            ret_change = (curr_prices / prev_prices - 1).dropna()
            if direction == "up":
                limit_stocks = set(ret_change[ret_change >= self.LIMIT_THRESHOLD].index)
                logger.debug(f"[BACKTEST] Date {date}: {len(limit_stocks)} limit-up stocks (from {len(ret_change)})")
                return limit_stocks
            else:
                limit_stocks = set(ret_change[ret_change <= -self.LIMIT_THRESHOLD].index)
                logger.debug(f"[BACKTEST] Date {date}: {len(limit_stocks)} limit-down stocks (from {len(ret_change)})")
                return limit_stocks
        except Exception as e:
            logger.debug(f"[BACKTEST] Failed to get limit stocks for {date}: {e}")
            return set()

    def _get_covariance(self, prices: pd.DataFrame, date, stock_names):
        n_stocks = len(stock_names)
        hist = prices.ffill().pct_change(fill_method=None).loc[:date]
        if len(hist) < self.lookback + 1:
            daily_ret = hist.iloc[1:]
        else:
            daily_ret = hist.iloc[-self.lookback:]
        if len(daily_ret) < 2:
            logger.debug(f"[BACKTEST] Insufficient history for covariance, using identity matrix")
            return np.eye(n_stocks) * 0.01

        common = daily_ret.columns.intersection(pd.Index(stock_names))
        if len(common) < 2:
            logger.debug(f"[BACKTEST] Insufficient common stocks for covariance, using identity matrix")
            return np.eye(n_stocks) * 0.01

        n = len(daily_ret)
        half_life = 60
        lam = np.exp(-np.log(2) / half_life)
        w = np.array([lam ** (n - 1 - i) for i in range(n)])
        w = w / w.sum()

        rets = daily_ret[common].fillna(0).values
        mean_ret = (rets.T @ w)
        centered = rets - mean_ret
        weighted_cov = (centered * w[:, None]).T @ centered

        n_features = len(common)
        sample_cov = weighted_cov
        prior = np.trace(sample_cov) / n_features * np.eye(n_features)
        shrinkage = max(0, min(1, (n_features + 1) / (n + 1)))
        shrunk_cov = shrinkage * prior + (1 - shrinkage) * sample_cov

        cov = pd.DataFrame(shrunk_cov, index=common, columns=common)
        result = cov.reindex(index=stock_names, columns=stock_names).fillna(0).values
        logger.debug(f"[BACKTEST] Covariance (EWMA, hl={half_life}d) computed for {len(common)}/{n_stocks} stocks, lookback={len(daily_ret)}d")
        return result

    def run(self, predictions: pd.Series, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None, use_optimizer=True,
            market_cap: pd.Series = None) -> tuple:
        """
        执行回测

        遍历每个预测日期：
        1. 选择 top_n 只预测最强的股票
        2. 使用优化器或等权方式确定权重
        3. 计算换手率和交易成本
        4. 计算组合收益

        Args:
            predictions: 预测 Series，MultiIndex (date, instrument)
            prices: 价格 DataFrame
            benchmark_returns: 基准收益 Series（可选）
            use_optimizer: 是否使用优化器，False 则使用等权

        Returns:
            (portfolio_returns, benchmark_returns, weights_history)
        """
        # 确保 MultiIndex 为 (date, instrument) 顺序
        idx = predictions.index
        if isinstance(idx, pd.MultiIndex) and idx.nlevels == 2:
            # 尝试通过 names 判断，如果名为 "instrument" 的级别在第一层则交换
            if idx.names[0] in ("instrument", "Instrument", "code", "asset"):
                predictions = predictions.swaplevel().sort_index()
            # 无 names 时通过 dtype 判断：第一层是 datetime 类型则无需交换
            elif idx.names[0] is None and not np.issubdtype(idx.get_level_values(0).dtype, np.datetime64):
                predictions = predictions.swaplevel().sort_index()
            idx = predictions.index

        # 如果传入了市值数据，对其进行标准化（对齐 index 顺序）
        if market_cap is not None:
            mc = market_cap.copy()
            mc_idx = mc.index
            if isinstance(mc_idx, pd.MultiIndex) and mc_idx.nlevels == 2:
                if mc_idx.names[0] in ("instrument", "Instrument", "code", "asset"):
                    mc = mc.swaplevel().sort_index()
                elif mc_idx.names[0] is None and not np.issubdtype(mc_idx.get_level_values(0).dtype, np.datetime64):
                    mc = mc.swaplevel().sort_index()
            market_cap = mc

        dates = sorted(idx.get_level_values(0).unique())
        hp = max(1, self.holding_period)
        rebalance_dates = dates[::hp]
        logger.info(f"[BACKTEST] Starting backtest: {len(rebalance_dates)} rebalance periods, {len(predictions)} predictions, holding_period={hp}, use_optimizer={use_optimizer}")

        portfolio_returns = []
        weights_history = []
        current_weights_series = None

        for i, date in enumerate(rebalance_dates):
            # 获取当日预测
            date_preds = predictions.loc[idx.get_level_values(0) == date]
            if len(date_preds) == 0:
                logger.warning(f"[BACKTEST] No predictions for date {date}, skipping")
                continue

            # 选择 top_n 只股票（排除涨停股票：涨停不开仓）
            n = self.config.get("portfolio", {}).get("top_n", 50)
            limit_up_stocks = self._get_limit_stocks(prices, date, dates, "up")
            date_preds_filtered = date_preds.drop(limit_up_stocks, errors="ignore")
            if len(date_preds_filtered) == 0:
                logger.warning(f"[BACKTEST] Date {date}: All top stocks hit limit-up, skipping rebalance")
                continue
            top_stocks = date_preds_filtered.nlargest(min(n, len(date_preds_filtered)))
            stock_names = top_stocks.index.get_level_values(-1)

            # 确定权重：支持三种方法
            weight_method = self.config.get("portfolio", {}).get("weight_method", "mean_variance")
            if use_optimizer and len(stock_names) >= 5 and weight_method == "mean_variance":
                logger.debug(f"[BACKTEST] Date {date}: Using CVXPY optimizer for {len(stock_names)} stocks")
                pred_arr = date_preds.loc[top_stocks.index].values
                cov = self._get_covariance(prices, date, stock_names)
                current_w_arr = (
                    current_weights_series.reindex(stock_names).fillna(0).values
                    if current_weights_series is not None else None
                )
                # 市值中性约束：计算 size_exposure 和 benchmark_size
                size_exp = None
                bench_size = None
                if market_cap is not None:
                    try:
                        mc_date = market_cap.xs(date, level=0, dropna=False)
                        mc_stocks = mc_date.reindex(stock_names)
                        valid = mc_stocks.dropna()
                        if len(valid) >= 3:
                            log_mc = np.log(valid + 1)
                            size_exp = log_mc.values
                            bench_size = float(np.median(log_mc))
                    except Exception:
                        size_exp = None
                        bench_size = None

                w_arr = self.optimizer.optimize(
                    predicted_returns=pred_arr,
                    cov_matrix=cov,
                    current_weights=current_w_arr,
                    size_exposure=size_exp,
                    benchmark_size=bench_size,
                )
                if w_arr is not None:
                    w_arr = w_arr / w_arr.sum()
                    w = pd.Series(w_arr, index=stock_names)
                    logger.debug(f"[BACKTEST] Date {date}: Optimizer returned weights, sum={w_arr.sum():.4f}")
                else:
                    w = pd.Series(1.0 / len(stock_names), index=stock_names)
                    logger.debug(f"[BACKTEST] Date {date}: Optimizer failed, using equal weights")
            elif use_optimizer and len(stock_names) >= 5 and weight_method == "rank":
                pred_arr = date_preds.loc[top_stocks.index].values
                w_arr = self.optimizer.rank_weights(pred_arr)
                w = pd.Series(w_arr, index=stock_names)
                logger.debug(f"[BACKTEST] Date {date}: Using rank-weighted portfolio for {len(stock_names)} stocks")
            else:
                w = pd.Series(1.0 / len(stock_names), index=stock_names)
                logger.debug(f"[BACKTEST] Date {date}: Using equal weights for {len(stock_names)} stocks")

            # 计算交易成本（跌停股票不平仓）
            cost = 0.0
            if current_weights_series is not None:
                limit_down_stocks = self._get_limit_stocks(prices, date, dates, "down")
                old_w_aligned = current_weights_series.reindex(w.index).fillna(0).values
                new_w_arr = w.values
                weight_diff = new_w_arr - old_w_aligned
                # 跌停股：不能卖出（如果旧持仓>0且新持仓<=旧持仓，则强制保留旧持仓）
                for j, s in enumerate(w.index):
                    if s in limit_down_stocks and old_w_aligned[j] > 0 and new_w_arr[j] <= old_w_aligned[j]:
                        new_w_arr[j] = old_w_aligned[j]
                        logger.debug(f"[BACKTEST] Date {date}: {s} hit limit-down, skipping sell")
                w = pd.Series(new_w_arr, index=w.index)
                w = w / w.sum()  # 重新归一化
                turnover_val = np.abs(new_w_arr - old_w_aligned).sum() / 2
                cost = turnover_val * (self.slippage + self.market_impact)
                logger.debug(f"[BACKTEST] Date {date}: turnover={turnover_val:.4f}, cost={cost:.6f}")

            weights_history.append(w)
            current_weights_series = w.copy()

            # 计算持有期收益
            next_idx = dates.index(date) + hp
            if next_idx < len(dates):
                next_date = dates[next_idx]
            else:
                next_date = dates[-1]
            ret = prices.loc[next_date] / prices.loc[date] - 1
            ret_aligned = ret.reindex(stock_names).fillna(0).values
            portfolio_ret = float(w.values @ ret_aligned) - cost
            portfolio_returns.append(portfolio_ret)
            logger.debug(f"[BACKTEST] Date {date} -> {next_date}: portfolio_return={portfolio_ret:.6f}")

        # 构建收益序列
        returns_idx = rebalance_dates[:len(portfolio_returns)]
        returns = pd.Series(portfolio_returns, index=returns_idx)

        # 计算实际年化参数（基于交易日历）
        if len(returns_idx) >= 2 and isinstance(returns_idx[0], pd.Timestamp):
            total_trading_days = sum(
                len([d for d in dates if returns_idx[i] <= d < returns_idx[min(i+1, len(returns_idx)-1)]])
                for i in range(len(returns_idx) - 1)
            )
            actual_periods_per_year = total_trading_days / (len(returns_idx) - 1) if len(returns_idx) > 1 else hp
            ann_periods_per_year = 252.0 / actual_periods_per_year if actual_periods_per_year > 0 else 252.0 / hp
        else:
            ann_periods_per_year = 252.0 / hp  # 后备：使用 hp 估算

        # 处理基准收益（按持有期复合，对齐到回测日期）
        # 组合持有期收益 = P[next_date] / P[date] - 1 = ∏(1 + r_{date+1..next_date}) - 1
        # 因此基准要剥离 r_date（即从 date 前一交易日到 date 的收益），从 date 后一天起算
        if benchmark_returns is not None:
            bench_list = []
            for i, date in enumerate(returns_idx):
                start_idx = dates.index(date)
                end_idx = min(start_idx + hp, len(dates) - 1)
                end_date = dates[end_idx]
                # 从 date 的下一个交易日开始（date+1 ~ end_date）复合收益
                if start_idx + 1 <= end_idx:
                    bench_slice = benchmark_returns.loc[pd.Timestamp(dates[start_idx + 1]):pd.Timestamp(end_date)]
                    period_ret = float((1 + bench_slice).prod() - 1)
                else:
                    period_ret = 0.0
                bench_list.append(period_ret)
            bench = pd.Series(bench_list, index=returns_idx)
        else:
            bench = pd.Series(0, index=returns.index)

        weights_df = pd.DataFrame(weights_history, index=rebalance_dates[:len(weights_history)])
        total_return = (1 + returns).prod() - 1
        periods_per_year_for_sharpe = ann_periods_per_year
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year_for_sharpe) if returns.std() > 0 else 0
        max_dd = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()
        logger.info(f"[BACKTEST] Backtest complete: {len(rebalance_dates)} rebalances, total_return={total_return:.4f}, ann_periods_per_year={periods_per_year_for_sharpe:.2f}, sharpe={sharpe:.4f}, max_drawdown={max_dd:.4f}")
        return returns, bench, weights_df, ann_periods_per_year