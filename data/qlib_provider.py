"""
Qlib 数据提供者模块

封装 Qlib 数据接口，提供：
- 日线行情数据（OHLCV、VWAP、复权因子等）
- 交易日历
- 股票池（可投资范围）
- 市值数据
- 行业分类

数据特点：
- 使用 Qlib 作为基础数据源
- 支持 CSI300、CSI500、CSI800 等指数成分
- 数据格式为 MultiIndex (instrument, datetime)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from qlib.data import D
import logging

from .fundamental_provider import FundamentalProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QlibDataProvider:
    """
    Qlib 数据提供者

    封装 Qlib 的数据获取接口，提供统一的数据访问层。
    支持从 Qlib 获取行情数据，并可叠加基本面数据。
    """

    def __init__(self, market="csi300", start_date="2022-01-01", end_date="2026-06-30",
                 cache_dir=None):
        """
        初始化数据提供者

        Args:
            market: 股票池标识，如 "csi300"（沪深300）、"csi500"（中证500）
            start_date: 数据开始日期
            end_date: 数据结束日期
            cache_dir: 缓存目录路径（用于基本面数据）
        """
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self._fundamental_provider = None

    @property
    def fundamental_provider(self):
        """
        获取基本面数据提供者（延迟初始化）

        只有在首次访问时才创建实例。
        """
        if self._fundamental_provider is None:
            logger.info(f"[QLIB数据] 初始化基本面数据提供器，标的数量: {len(self.get_all_instruments())}")
            all_inst = self.get_all_instruments()
            self._fundamental_provider = FundamentalProvider(
                instruments=all_inst,
                start_date=self.start_date,
                end_date=self.end_date,
                cache_dir=self.cache_dir,
            )
            logger.info(f"[QLIB数据] 基本面数据提供器初始化完成: {len(all_inst)} 个标的, 缓存目录={self.cache_dir}")
        return self._fundamental_provider

    def get_all_instruments(self):
        """
        获取股票池内所有标的代码

        Returns:
            股票代码列表，如 ["SH600000", "SZ000001", ...]
        """
        logger.info(f"[QLIB数据] 获取所有标的, 市场={self.market}, 日期范围: {self.start_date} 到 {self.end_date}")
        instruments = D.instruments(market=self.market)
        logger.info(f"[QLIB数据] 从Qlib获取到 {len(instruments)} 个标的")
        return list(D.list_instruments(instruments, self.start_date))

    def get_trade_dates(self):
        """
        获取交易日历

        Returns:
            交易日期列表（已排序）
        """
        logger.info(f"[QLIB数据] 获取交易日历, 范围: {self.start_date} 到 {self.end_date}")
        trade_dates = D.calendar(start_time=self.start_date, end_time=self.end_date)
        logger.info(f"[QLIB数据] 交易日历: {len(trade_dates)} 个交易日, 范围: {trade_dates[0]} 到 {trade_dates[-1]}")
        return trade_dates

    def get_universe(self, date):
        """
        获取某日期的股票池

        Args:
            date: 查询日期

        Returns:
            该日期可投资的股票代码列表
        """
        instruments = D.instruments(market=self.market)
        universe = D.list_instruments(instruments, date)
        logger.debug(f"[QLIB数据] 股票池 {date}: {len(universe)} 个标的")
        return universe

    def get_daily_data(self, fields=None):
        """
        获取日线行情数据

        默认字段：开盘价、最高价、最低价、收盘价、成交量、成交额、VWAP、涨跌幅、复权因子

        Args:
            fields: 字段列表，None 表示使用默认字段

        Returns:
            日线行情 DataFrame，MultiIndex (instrument, datetime)
        """
        if fields is None:
            fields = [
                "$open", "$high", "$low", "$close", "$volume", "$amount",
                "$vwap", "$change", "$factor",
            ]
        logger.info(f"[QLIB数据] 获取日线数据: {len(fields)} 个字段, 范围: {self.start_date} 到 {self.end_date}")
        instruments = D.instruments(market=self.market)
        data = D.features(instruments, fields, self.start_date, self.end_date)
        data.columns = data.columns.map(lambda x: x.lstrip("$"))  # 去除字段名前缀 $
        logger.info(f"[QLIB数据] 日线数据: {len(data)} 行, {len(data.columns)} 列, 日期范围: {data.index.get_level_values(1).min()} 到 {data.index.get_level_values(1).max()}")
        return data

    def get_augmented_data(self):
        """
        获取增强数据：行情 + 基本面

        整合 Qlib 行情数据和 Akshare 基本面数据。

        数据来源：
        - Qlib：行情数据（OHLCV、VWAP 等）
        - Akshare：基本面数据（每股收益、每股净资产、ROE、ROA、资产负债率、营收增长、毛利率等）
        - 计算：市值 = 收盘价 × 流通股本

        Returns:
            增强后的 DataFrame，包含行情和基本面数据
        """
        logger.info("[QLIB数据] 开始数据增强流程...")
        qlib_data = self.get_daily_data()
        trade_dates = self.get_trade_dates()

        logger.info("[QLIB数据] 获取基本面数据...")
        fundamental = self.fundamental_provider.get_all(trade_dates)

        if fundamental.empty:
            logger.warning("[QLIB数据] 无基本面数据可用，仅返回Qlib数据")
            return qlib_data

        logger.info(f"[QLIB数据] 基本面数据合并完成: {len(fundamental)} 行")

        # 对齐索引格式
        fundamental.index.names = qlib_data.index.names
        aligned = qlib_data.join(fundamental, how="left")

        logger.info(f"[QLIB数据] 合并后: {len(aligned)} 行, {len(aligned.columns)} 列")

        # 计算市值：如果有流通股本数据
        if "outstanding_share" in aligned.columns and "close" in aligned.columns:
            logger.info("[QLIB数据] 计算市值: 流通股本 × 收盘价")
            before_count = len(aligned)
            aligned["market_cap"] = aligned["close"] * aligned["outstanding_share"]
            nan_count = aligned["market_cap"].isna().sum()
            logger.info(f"[QLIB数据] 市值计算完成: {before_count - nan_count}/{before_count} 非空值 ({100*(before_count - nan_count)/before_count:.1f}%)")
        else:
            logger.warning("[QLIB数据] 无法计算市值: 缺少流通股本或收盘价列")

        logger.info(f"[QLIB数据] 增强数据准备完成: {len(aligned)} 行, {len(aligned.columns)} 列, 日期范围: {aligned.index.get_level_values(1).min()} 到 {aligned.index.get_level_values(1).max()}")
        return aligned

    def get_market_cap(self, field="$market_cap"):
        """
        获取市值数据

        Args:
            field: Qlib 字段名，默认市值

        Returns:
            市值 DataFrame
        """
        data = D.features(
            D.instruments(market=self.market),
            [field],
            self.start_date,
            self.end_date,
        )
        data.columns = data.columns.map(lambda x: x.lstrip("$"))
        logger.info(f"[QLIB数据] 市值数据: {len(data)} 行, {len(data.columns)} 列")
        return data

    def get_industry(self):
        """
        获取行业分类数据

        Returns:
            行业分类 DataFrame
        """
        data = D.features(
            D.instruments(market=self.market),
            ["$industry"],
            self.start_date,
            self.end_date,
        )
        data.columns = data.columns.map(lambda x: x.lstrip("$"))
        logger.info(f"[QLIB数据] 行业分类数据: {len(data)} 行, {len(data.columns)} 列")
        return data