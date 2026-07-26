"""
组合优化器模块

使用 CVXPY 实现均值-方差优化组合构建。

优化目标：
    max w'μ - λ*w'Σw - tc*|w-w0|_1

约束条件：
    - w >= 0（仅做多）
    - sum(w) <= 1（总权重上限）
    - w <= max_weight（单股权重上限）
    - |w - w0|_1 <= turnover_limit（换手率约束，可选）
    - |sectorExposure'w - sectorBenchmark| <= sector_dev（行业约束，可选）
"""

import logging
import cvxpy as cp
import numpy as np

from .constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class CVXPYOptimizer:
    """
    CVXPY 均值-方差优化器

    使用二次规划求解最优组合权重。
    目标函数包含三个部分：
    1. 预期收益：predicted_returns @ w
    2. 风险惩罚：λ * w'Σw
    3. 交易成本：tc * |w - w0|_1

    如果优化失败（不可行或求解失败），回退到等权组合。
    """

    def __init__(self, config: dict):
        """
        初始化优化器

        Args:
            config: 配置字典，包含：
                - risk_aversion: 风险厌恶系数 λ
                - transaction_cost: 交易成本系数 tc
                - turnover_limit: 最大换手率
                - max_weight: 单股最大权重
                - sector_dev: 行业偏离上限
        """
        self.config = config
        self.constraints_obj = PortfolioConstraints(config)
        logger.info(f"[OPTIMIZER] CVXPYOptimizer initialized: risk_aversion={config.get('risk_aversion', 1.0)}, transaction_cost={config.get('transaction_cost', 0.003)}")

    def optimize(self, predicted_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 current_weights: np.ndarray = None,
                 sector_exposure: np.ndarray = None,
                 benchmark_sector: np.ndarray = None,
                 size_exposure: np.ndarray = None,
                 benchmark_size: float = None) -> np.ndarray:
        """
        执行组合优化

        Args:
            predicted_returns: 预期收益向量（n,）
            cov_matrix: 协方差矩阵（n, n）
            current_weights: 当前权重向量（n,），用于计算换手率
            sector_exposure: 行业暴露矩阵（n,），股票×行业的指示矩阵
            benchmark_sector: 基准行业权重向量（n,），行业中性基准

        Returns:
            优化后的权重向量（n,）
        """
        n = len(predicted_returns)
        logger.debug(f"[OPTIMIZER] Optimize called: n={n}, risk_aversion={self.config.get('risk_aversion', 1.0)}, current_weights={'Yes' if current_weights is not None else 'No'}")

        # 定义优化变量
        w = cp.Variable(n)
        risk_aversion = self.config.get("risk_aversion", 1.0)
        tc = self.config.get("transaction_cost", 0.003)

        # 目标函数：收益 - 风险惩罚 - 交易成本
        ret = predicted_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        if current_weights is not None:
            turnover = cp.norm1(w - current_weights)
            objective = cp.Maximize(ret - risk_aversion * risk - tc * turnover)
        else:
            objective = cp.Maximize(ret - risk_aversion * risk)

        # 约束条件列表
        constraints_list = [
            w >= 0,  # 仅做多
            cp.sum(w) <= 1.0,  # 总权重 <= 1
            w <= self.constraints_obj.max_weight,  # 单股权重上限
        ]

        # 换手率约束（如果有当前权重）
        if current_weights is not None:
            constraints_list.append(cp.norm1(w - current_weights) <= self.constraints_obj.turnover_limit)

        # 行业约束（如果有行业暴露数据）
        if sector_exposure is not None and benchmark_sector is not None:
            sector_diff = sector_exposure.T @ w - benchmark_sector
            constraints_list.append(cp.norm_inf(sector_diff) <= self.constraints_obj.sector_dev)

        # 市值中性约束（如果有市值暴露数据）
        if size_exposure is not None and benchmark_size is not None:
            size_diff = w @ size_exposure - benchmark_size
            constraints_list.append(cp.abs(size_diff) <= self.constraints_obj.cap_dev)
            logger.debug(f"[OPTIMIZER] Size neutral constraint: benchmark_size={benchmark_size:.4f}, cap_dev={self.constraints_obj.cap_dev}")

        # 求解优化问题
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS, verbose=False)

        if w.value is not None:
            logger.debug(f"[OPTIMIZER] Optimization succeeded: status={problem.status}, optimal value={problem.value:.6f}, sum(w)={w.value.sum():.4f}")
            return w.value
        else:
            logger.warning(f"[OPTIMIZER] Optimization failed: status={problem.status}, using clipped equal weights")
            # 等权时需要裁剪到 max_weight 以下
            equal_w = np.ones(n) / n
            equal_w = np.clip(equal_w, 0, self.constraints_obj.max_weight)
            equal_w = equal_w / equal_w.sum()  # 重新归一化
            return equal_w