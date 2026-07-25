import cvxpy as cp
import numpy as np

from .constraints import PortfolioConstraints


class CVXPYOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.constraints_obj = PortfolioConstraints(config)

    def optimize(self, predicted_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 current_weights: np.ndarray = None,
                 sector_exposure: np.ndarray = None,
                 benchmark_sector: np.ndarray = None) -> np.ndarray:
        n = len(predicted_returns)
        w = cp.Variable(n)
        risk_aversion = self.config.get("risk_aversion", 1.0)
        tc = self.config.get("transaction_cost", 0.003)

        ret = predicted_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        if current_weights is not None:
            turnover = cp.norm1(w - current_weights)
            objective = cp.Maximize(ret - risk_aversion * risk - tc * turnover)
        else:
            objective = cp.Maximize(ret - risk_aversion * risk)

        constraints_list = [
            w >= 0,
            cp.sum(w) <= 1.0,
            w <= self.constraints_obj.max_weight,
        ]

        if current_weights is not None:
            constraints_list.append(cp.norm1(w - current_weights) <= self.constraints_obj.turnover_limit)

        if sector_exposure is not None and benchmark_sector is not None:
            sector_diff = sector_exposure.T @ w - benchmark_sector
            constraints_list.append(cp.norm_inf(sector_diff) <= self.constraints_obj.sector_dev)

        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS, verbose=False)

        return w.value if w.value is not None else np.ones(n) / n
