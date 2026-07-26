"""
组合约束模块

定义投资组合的各种约束条件，包括：
- 换手率约束：限制每次调仓的换手率
- 权重约束：单股权重上限、总权重上限
- 行业约束：行业权重相对基准的偏离限制
- 市值约束：市值权重相对基准的偏离限制

这些约束用于控制组合风险和交易成本。
"""

import logging

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """
    组合约束定义

    所有约束都是可选的，如果不设置则使用默认值。
    """

    def __init__(self, config: dict):
        """
        初始化组合约束

        Args:
            config: 配置字典，包含约束参数：
                {
                    "turnover_limit": 0.30,    # 最大换手率（单边）
                    "max_weight": 0.08,        # 单股最大权重
                    "sector_neutral": True,    # 是否行业中性
                    "size_neutral": True,       # 是否市值中性
                    "sector_dev": 0.03,        # 行业权重偏离上限
                    "cap_dev": 0.03            # 市值权重偏离上限
                }
        """
        self.turnover_limit = config.get("turnover_limit", 0.30)
        self.max_weight = config.get("max_weight", 0.08)
        self.sector_neutral = config.get("sector_neutral", True)
        self.size_neutral = config.get("size_neutral", True)
        self.sector_dev = config.get("sector_dev", 0.03)
        self.cap_dev = config.get("cap_dev", 0.03)