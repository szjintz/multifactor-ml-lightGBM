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
import pandas as pd

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
            # 标准化 size_exposure 使其具有单位标准差
            size_std = np.std(size_exposure)
            if size_std > 1e-8:
                size_exposure_scaled = (size_exposure - benchmark_size) / size_std
                # 约束：组合加权的标准化市值暴露接近 0（即与基准对齐）
                size_diff = w @ size_exposure_scaled
                constraints_list.append(cp.abs(size_diff) <= self.constraints_obj.cap_dev)
                logger.debug(f"[OPTIMIZER] Size neutral constraint (scaled): benchmark_size={benchmark_size:.4f}, size_std={size_std:.4f}, cap_dev={self.constraints_obj.cap_dev}")

        # 求解优化问题
        problem = cp.Problem(objective, constraints_list)
        # 尝试 ECOS，如失败回退到 SCS（更鲁棒但精度略低）
        try:
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=200)
        except Exception:
            logger.warning(f"[OPTIMIZER] ECOS solver failed, falling back to SCS")
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)

        if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
            # 清理数值噪声并裁剪到合规范围
            w_val = np.maximum(w.value, 0)
            w_sum = w_val.sum()
            if w_sum > 1e-6:
                w_val = w_val / w_sum
            logger.debug(f"[OPTIMIZER] Optimization succeeded: status={problem.status}, optimal value={problem.value:.6f}, sum(w)={w_val.sum():.4f}")
            return w_val
        else:
            logger.warning(f"[OPTIMIZER] Optimization failed: status={problem.status}, falling back to rank weights")
            try:
                return self.rank_weights(predicted_returns)
            except Exception:
                logger.warning(f"[OPTIMIZER] Rank weights also failed, using equal weights")
                equal_w = np.ones(n) / n
                equal_w = np.clip(equal_w, 0, self.constraints_obj.max_weight)
                equal_w = equal_w / equal_w.sum()
                return equal_w

    @staticmethod
    def rank_weights(predicted_returns: np.ndarray) -> np.ndarray:
        """
        基于预测排序的线性加权（仅使用排序信息，忽略预测幅度）

        对弱信号（IC~0.01）更鲁棒：排名比预测值更稳定、可迁移。
        权重 = rank / sum(rank)，最高预测获得最大权重（线性衰减）。

        Args:
            predicted_returns: 预测收益向量（n,）

        Returns:
            归一化权重向量（n,），总和为1
        """
        n = len(predicted_returns)
        if n == 0:
            return np.array([])
        ranks = pd.Series(predicted_returns).rank(method="min").values
        weights = ranks / ranks.sum()
        # 线性加权后通常已有max_weight约束
        return weights