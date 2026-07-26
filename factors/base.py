"""
因子基类模块

定义了因子计算的抽象基类和因子管线（Pipeline）。
所有具体因子类都应继承 BaseFactor 并实现 compute 方法。

设计模式：
- 策略模式：每个因子是独立的计算策略
- 管线模式：FactorPipeline 统一调度多个因子的计算

因子命名规范：
- RET_{N}: N日收益率
- WMA_{N}: N日加权动量
- RS_{N}: N日 RSI
- STD_{N}: N日波动率
- BETA_{N}: N日市场贝塔
等
"""

from abc import ABC, abstractmethod
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """
    因子抽象基类

    所有因子类都必须继承此类并实现：
    - compute: 计算因子值
    - name: 返回因子名称

    设计原则：
    - compute 方法接收原始市场数据，返回因子 Series
    - 因子计算应该是向量化的，支持批量处理
    - 因子名称应该唯一，用于标识
    """

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: 原始市场数据，至少包含 close, high, low, volume 等列

        Returns:
            因子值 Series，index 与输入数据对齐
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        返回因子名称

        Returns:
            因子名称字符串
        """
        pass


class FactorPipeline:
    """
    因子管线

    统一管理多个因子的注册和计算。
    支持链式注册（返回 self），方便 fluent API。

    使用示例：
        pipeline = FactorPipeline()
        pipeline.register(MomentumFactor(20))
        pipeline.register(RSIFactor(14))
        factor_df = pipeline.compute_all(market_data)
    """

    def __init__(self):
        """
        初始化因子管线
        """
        self.factors: list[BaseFactor] = []
        logger.debug(f"[因子管线] 初始化FactorPipeline，当前注册因子数：{len(self.factors)}")

    def register(self, factor: BaseFactor):
        """
        注册因子

        Args:
            factor: 继承自 BaseFactor 的因子实例

        Returns:
            self，支持链式调用
        """
        self.factors.append(factor)
        logger.info(f"[因子管线] 注册因子：{factor.name}（总计：{len(self.factors)})")
        return self

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有已注册的因子

        遍历所有因子，依次计算并合并为 DataFrame。

        Args:
            data: 原始市场数据

        Returns:
            包含所有因子值的 DataFrame，列名为因子名称
        """
        logger.info(f"[因子管线] 开始计算全部{len(self.factors)}个因子，数据形状={data.shape}")
        results = {}
        t_start = time.time()
        for i, f in enumerate(self.factors):
            t0 = time.time()
            logger.debug(f"[因子管线] 正在计算因子{i+1}/{len(self.factors)}：{f.name}")
            result = f.compute(data)
            elapsed = time.time() - t0
            results[f.name] = result
            nan_pct = result.isna().sum() / len(result) * 100
            logger.info(f"[因子管线] 因子{f.name}：{len(result)}行，{nan_pct:.1f}%缺失值，耗时{elapsed:.1f}秒")
            if elapsed > 10:
                logger.warning(f"[因子管线] 因子计算缓慢（{elapsed:.1f}秒）：{f.name}")
        total = time.time() - t_start
        df = pd.DataFrame(results)
        logger.info(f"[因子管线] 全部因子计算完成：{len(df.columns)}个因子，{len(df)}行，形状={df.shape}，总耗时{total:.1f}秒")
        return df