"""
基本面数据提供者模块

从 Akshare 获取中国A股基本面数据，包括：
- 日线数据：流通股数、换手率
- 季度财务数据：每股收益、每股净资产、ROE、ROA、资产负债率、营收增长、毛利率

数据缓存：
- 所有数据会缓存到本地文件系统，避免重复请求
- 缓存路径默认 qlib_fundamental_cache
- 支持增量更新，只获取缺失的数据

关键设计：
- 使用多线程并行获取数据，加速数据拉取
- 支持重试机制，应对网络波动
- 季度数据按报告日期对齐到下一个交易日（财务数据有发布延迟）
"""

import os
os.environ.setdefault("AKSHARE_VERBOSE", "0")

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import time
import logging

logging.getLogger("akshare").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import akshare as ak


class FundamentalProvider:
    """
    基本面数据提供者

    从 Akshare 获取A股基本面数据，支持缓存和增量更新。
    数据分为两类：
    - 日线数据（每日更新）：流通股数、换手率
    - 季度数据（季度更新）：财务指标
    """

    # 季度财务指标列名映射：Akshare 中文列名 -> 内部英文列名
    QUARTERLY_COLUMNS = {
        "earnings_per_share": "每股收益_调整后(元)",
        "book_value_per_share": "每股净资产_调整后(元)",
        "ROE_TTM": "净资产收益率(%)",
        "ROA": "总资产净利润率(%)",
        "debt_ratio": "资产负债率(%)",
        "revenue_growth_yoy": "主营业务收入增长率(%)",
    }

    def __init__(self, instruments, start_date, end_date, cache_dir=None, max_workers=8):
        """
        初始化基本面数据提供者

        Args:
            instruments: 股票代码列表
            start_date: 数据开始日期
            end_date: 数据结束日期
            cache_dir: 缓存目录，None 则使用默认目录
            max_workers: 并行获取的线程数
        """
        self.instruments = instruments
        if not isinstance(start_date, str):
            start_date = start_date.isoformat()
        if not isinstance(end_date, str):
            end_date = end_date.isoformat()
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir or Path(__file__).parent / "qlib_fundamental_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        # 财务数据往前多取两年，确保季度数据完整
        self.start_year = max(int(self.start_date[:4]) - 2, 2015)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    @staticmethod
    def _inst_to_akshare_symbol(inst):
        """
        将内部股票代码转换为 Akshare 格式

        内部格式：SH600000、SZ000001、BJxxxxxx
        Akshare 格式：sh600000、sz000001、bjxxxxxx
        """
        if inst.startswith("SZ"):
            return f"sz{inst[2:]}"
        elif inst.startswith("SH"):
            return f"sh{inst[2:]}"
        elif inst.startswith("BJ"):
            return f"bj{inst[2:]}"
        return inst

    @staticmethod
    def _inst_to_code(inst):
        """
        将内部股票代码转换为纯数字代码

        SH600000 -> 600000
        """
        return inst[2:]

    def _cache_path(self, inst, data_type):
        """
        获取缓存文件路径

        Args:
            inst: 股票代码
            data_type: 数据类型（如 "daily", "quarterly"）

        Returns:
            缓存文件路径
        """
        return self.cache_dir / f"{data_type}_{inst}.pkl"

    def _load_inst_cache(self, inst, data_type):
        """
        从缓存加载单个股票的数据

        Returns:
            缓存的数据 DataFrame，如果缓存不存在则返回 None
        """
        p = self._cache_path(inst, data_type)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_inst_cache(self, inst, data_type, data):
        """
        保存单个股票的数据到缓存
        """
        with open(self._cache_path(inst, data_type), "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def _fetch_single_stock_daily(symbol, start_date, end_date, retries=3):
        """
        获取单只股票的日线数据

        Args:
            symbol: Akshare 格式的股票代码
            start_date: 开始日期
            end_date: 结束日期
            retries: 重试次数

        Returns:
            日线数据 DataFrame，包含 date, outstanding_share, turnover 列
        """
        for attempt in range(retries):
            try:
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    @staticmethod
    def _fetch_single_financial(symbol, start_year, retries=3):
        """
        获取单只股票的财务指标

        Args:
            symbol: 纯数字股票代码
            start_year: 开始年份
            retries: 重试次数

        Returns:
            财务指标 DataFrame
        """
        for attempt in range(retries):
            try:
                df = ak.stock_financial_analysis_indicator(
                    symbol=symbol, start_year=str(start_year)
                )
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    @staticmethod
    def _fetch_single_gross_margin(code, retries=3):
        """
        获取单只股票的毛利率数据

        Args:
            code: 纯数字股票代码
            retries: 重试次数

        Returns:
            财务摘要 DataFrame
        """
        for attempt in range(retries):
            try:
                df = ak.stock_financial_abstract(symbol=code)
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_daily_fundamentals(self):
        """
        获取日线基本面数据（流通股数、换手率）

        使用缓存机制，只拉取缺失的数据。
        并行获取所有股票的日线数据。

        Returns:
            日线数据 DataFrame，MultiIndex (instrument, date)
        """
        cached = self._load_inst_cache("_ALL_", "daily_fundamentals")
        records = []
        to_fetch = []

        if cached is not None and not cached.empty:
            cached_dates = cached.index.get_level_values(1)
            date_ok = cached_dates.max() >= pd.Timestamp(self.end_date) and cached_dates.min() <= pd.Timestamp(self.start_date) + pd.Timedelta(days=30)

            if date_ok:
                cached_insts = set(cached.index.get_level_values(0))
                all_insts = set(self.instruments)
                missing = sorted(all_insts - cached_insts)
                if not missing:
                    self.logger.info(f"[DAILY_FUNDAMENTALS] Cache hit: {len(cached)} rows, {cached_dates.min()} to {cached_dates.max()}")
                    return cached
                previously_failed = []
                active_missing = []
                for inst in missing:
                    inst_cache = self._load_inst_cache(inst, "daily")
                    if inst_cache is not None and inst_cache.empty:
                        previously_failed.append(inst)
                    else:
                        active_missing.append(inst)
                if previously_failed:
                    self.logger.warning(f"[DAILY_FUNDAMENTALS] Skipping {len(previously_failed)} previously failed instruments: {previously_failed}")
                if not active_missing:
                    self.logger.info(f"[DAILY_FUNDAMENTALS] All missing instruments previously failed, returning cached data")
                    return cached
                self.logger.warning(f"[DAILY_FUNDAMENTALS] Global cache missing {len(active_missing)} instruments: {active_missing[:5]}... Will refetch")
                records.append(cached.reset_index())
                to_fetch = active_missing
            else:
                self.logger.info(f"[DAILY_FUNDAMENTALS] Cache stale: cached={cached_dates.min()} to {cached_dates.max()}, requested={self.start_date} to {self.end_date}")
                self.logger.info(f"[DAILY_FUNDAMENTALS] Trying to rebuild from individual caches...")
                rebuilt, to_fetch = self._rebuild_from_individual_caches("daily")
                if rebuilt is not None:
                    self.logger.info(f"[DAILY_FUNDAMENTALS] Rebuilt from individual caches, need to fetch {len(to_fetch)} more instruments")
                    records.append(rebuilt.reset_index())
                else:
                    to_fetch = list(self.instruments)
        else:
            self.logger.info(f"[DAILY_FUNDAMENTALS] Global aggregate cache not found, trying to rebuild from individual caches...")
            rebuilt, to_fetch = self._rebuild_from_individual_caches("daily")
            if rebuilt is not None:
                records.append(rebuilt.reset_index())
            else:
                to_fetch = list(self.instruments)

        if to_fetch:
            self.logger.info(f"[DAILY_FUNDAMENTALS] Fetching daily data for {len(to_fetch)} instruments (from cache miss)")
            total = len(to_fetch)
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fut_map = {}
                for inst in to_fetch:
                    symbol = self._inst_to_akshare_symbol(inst)
                    fut = ex.submit(
                        self._fetch_single_stock_daily,
                        symbol, self.start_date, self.end_date,
                    )
                    fut_map[fut] = inst

                done_count = 0
                for fut in as_completed(fut_map):
                    inst = fut_map[fut]
                    done_count += 1
                    df = fut.result()
                    if df is None or len(df) == 0:
                        self._save_inst_cache(inst, "daily", pd.DataFrame())
                        self.logger.warning(f"[DAILY_FUNDAMENTALS] Failed to fetch daily data for {inst} (attempt {done_count}/{total})")
                        continue
                    temp = df[["date", "outstanding_share", "turnover"]].copy()
                    temp["instrument"] = inst
                    temp["date"] = pd.to_datetime(temp["date"])
                    temp = temp.dropna(subset=["outstanding_share"])
                    self._save_inst_cache(inst, "daily", temp)
                    if not temp.empty:
                        records.append(temp)
                    self.logger.debug(f"[DAILY_FUNDAMENTALS] Fetched daily data for {inst}: {len(temp)} rows, outstanding_share={temp['outstanding_share'].mean():.2e}, turnover={temp['turnover'].mean():.4f}")

        if not records:
            self.logger.warning("[DAILY_FUNDAMENTALS] No daily data fetched for any instrument")
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result = result.drop_duplicates(subset=["instrument", "date"])
        result = result.set_index(["instrument", "date"]).sort_index()
        self.logger.info(f"[DAILY_FUNDAMENTALS] Combined daily data: {len(result)} rows, {len(result.index.get_level_values(0).unique())} instruments, date range: {result.index.get_level_values(1).min()} to {result.index.get_level_values(1).max()}")
        self._save_inst_cache("_ALL_", "daily_fundamentals", result)
        return result

    def _fetch_all_gross_margin(self):
        """
        获取所有股票的毛利率数据

        毛利率是季度财务数据的一部分，需要单独获取。
        """
        cached = self._load_inst_cache("_ALL_", "gross_margin")
        gm_records = []
        gm_to_fetch = []

        if cached is not None and not cached.empty:
            cached_dates = cached.index.get_level_values(1)
            date_ok = cached_dates.max() >= pd.Timestamp(self.end_date) and cached_dates.min() <= pd.Timestamp(self.start_date) + pd.Timedelta(days=30)

            if date_ok:
                cached_insts = set(cached.index.get_level_values(0))
                all_insts = set(self.instruments)
                missing = sorted(all_insts - cached_insts)
                if not missing:
                    self.logger.info(f"[GROSS_MARGIN] Cache hit: {len(cached)} rows, {cached_dates.min()} to {cached_dates.max()}")
                    return cached
                previously_failed = []
                active_missing = []
                for inst in missing:
                    inst_cache = self._load_inst_cache(inst, "gross_margin")
                    if inst_cache is not None and inst_cache.empty:
                        previously_failed.append(inst)
                    else:
                        active_missing.append(inst)
                if previously_failed:
                    self.logger.warning(f"[GROSS_MARGIN] Skipping {len(previously_failed)} previously failed instruments: {previously_failed}")
                if not active_missing:
                    self.logger.info(f"[GROSS_MARGIN] All missing instruments previously failed, returning cached data")
                    return cached
                self.logger.warning(f"[GROSS_MARGIN] Global cache missing {len(active_missing)} instruments: {active_missing[:5]}... Will refetch")
                gm_records.append(cached)
                gm_to_fetch = active_missing
            else:
                self.logger.info(f"[GROSS_MARGIN] Cache stale: cached={cached_dates.min()} to {cached_dates.max()}, requested={self.start_date} to {self.end_date}")
                self.logger.info(f"[GROSS_MARGIN] Trying to rebuild from individual caches...")
                rebuilt, gm_to_fetch = self._rebuild_from_individual_caches("gross_margin")
                if rebuilt is not None:
                    self.logger.info(f"[GROSS_MARGIN] Rebuilt from individual caches, need to fetch {len(gm_to_fetch)} more instruments")
                    gm_records.append(rebuilt)
                else:
                    gm_to_fetch = list(self.instruments)
        else:
            self.logger.info(f"[GROSS_MARGIN] Global aggregate cache not found, trying to rebuild from individual caches...")
            rebuilt, gm_to_fetch = self._rebuild_from_individual_caches("gross_margin")
            if rebuilt is not None:
                gm_records.append(rebuilt)
            else:
                gm_to_fetch = list(self.instruments)

        if gm_to_fetch:
            self.logger.info(f"[GROSS_MARGIN] Fetching gross margin data for {len(gm_to_fetch)} instruments (from cache miss)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fut_map = {}
                for inst in gm_to_fetch:
                    code = self._inst_to_code(inst)
                    fut = ex.submit(self._fetch_single_gross_margin, code)
                    fut_map[fut] = inst

                for fut in as_completed(fut_map):
                    inst = fut_map[fut]
                    df = fut.result()
                    if df is None or df.empty:
                        self._save_inst_cache(inst, "gross_margin", pd.DataFrame())
                        self.logger.warning(f"[GROSS_MARGIN] Failed to fetch gross margin for {inst}")
                        continue
                    gm_row = df[df['指标'] == '毛利率']
                    if gm_row.empty:
                        self._save_inst_cache(inst, "gross_margin", pd.DataFrame())
                        self.logger.warning(f"[GROSS_MARGIN] No gross margin data found in abstract for {inst}")
                        continue
                    gm = gm_row.iloc[[0]]
                    date_cols = [c for c in gm.columns if c not in ['选项', '指标']]
                    gm_long = gm[date_cols].T.reset_index()
                    gm_long.columns = ['日期', 'gross_margin']
                    gm_long['日期'] = pd.to_datetime(gm_long['日期'], errors='coerce')
                    gm_long = gm_long.dropna(subset=['日期'])
                    gm_long['gross_margin'] = pd.to_numeric(gm_long['gross_margin'], errors='coerce')
                    gm_long = gm_long.dropna(subset=['gross_margin'])
                    gm_long['instrument'] = inst
                    gm_long = gm_long[['instrument', '日期', 'gross_margin']]
                    gm_long = gm_long.set_index(["instrument", "日期"]).sort_index()
                    self._save_inst_cache(inst, "gross_margin", gm_long)
                    if not gm_long.empty:
                        gm_records.append(gm_long)
                    #   self.logger.info(f"[GROSS_MARGIN] Successfully fetched gross margin for {inst}: {len(gm_long)} periods, range: {gm_long.index.get_level_values(1).min()} to {gm_long.index.get_level_values(1).max()}, avg={gm_long['gross_margin'].mean():.2f}%")

        if not gm_records:
            result = pd.DataFrame()
            self.logger.warning("[GROSS_MARGIN] No gross margin data fetched for any instrument")
        else:
            result = pd.concat(gm_records)
            result = result[~result.index.duplicated(keep="last")]
            self.logger.info(f"[GROSS_MARGIN] Combined gross margin data: {len(result)} rows, {len(result.index.get_level_values(0).unique())} instruments, date range: {result.index.get_level_values(1).min()} to {result.index.get_level_values(1).max()}")

        self._save_inst_cache("_ALL_", "gross_margin", result)
        return result

    def _rebuild_from_individual_caches(self, data_type):
        """
        尝试从个体缓存重建全局数据

        当全局聚合缓存缺失但个体缓存存在时，从个体缓存重建。

        Args:
            data_type: 个体缓存的数据类型前缀（如 "quarterly", "daily", "gross_margin"）

        Returns:
            (聚合DataFrame或None, 仍未缓存的股票列表)
        """
        records = []
        missing = []
        for inst in self.instruments:
            inst_cache = self._load_inst_cache(inst, data_type)
            if inst_cache is not None and not inst_cache.empty:
                if isinstance(inst_cache.index, pd.MultiIndex):
                    inst_cache = inst_cache.reset_index()
                records.append(inst_cache)
            else:
                missing.append(inst)
        if not records:
            return None, self.instruments
        result = pd.concat(records, ignore_index=True)
        if "日期" in result.columns:
            result = result.set_index(["instrument", "日期"]).sort_index()
        elif "date" in result.columns:
            result = result.set_index(["instrument", "date"]).sort_index()
        self.logger.info(f"[{data_type.upper()}] Rebuilt from {len(records)} individual caches: {len(result)} rows, {len(result.index.get_level_values(0).unique())} instruments")
        return result, missing

    def get_quarterly_fundamentals(self):
        """
        获取季度财务数据

        包含：每股收益、每股净资产、ROE、ROA、资产负债率、营收增长、毛利率

        Returns:
            季度财务数据 DataFrame，MultiIndex (instrument, 日期)
        """
        cached = self._load_inst_cache("_ALL_", "quarterly_fundamentals")
        records = []
        to_fetch = []

        if cached is not None and not cached.empty:
            cached_dates = cached.index.get_level_values(1)
            date_ok = cached_dates.max() >= pd.Timestamp(self.end_date) and cached_dates.min() <= pd.Timestamp(self.start_date) + pd.Timedelta(days=30)

            if date_ok:
                cached_insts = set(cached.index.get_level_values(0))
                all_insts = set(self.instruments)
                missing = sorted(all_insts - cached_insts)
                if not missing:
                    self.logger.info(f"[QUARTERLY] Cache hit: {len(cached)} rows, {cached_dates.min()} to {cached_dates.max()}")
                    return cached
                previously_failed = []
                active_missing = []
                for inst in missing:
                    inst_cache = self._load_inst_cache(inst, "quarterly")
                    if inst_cache is not None and inst_cache.empty:
                        previously_failed.append(inst)
                    else:
                        active_missing.append(inst)
                if previously_failed:
                    self.logger.warning(f"[QUARTERLY] Skipping {len(previously_failed)} previously failed instruments: {previously_failed}")
                if not active_missing:
                    self.logger.info(f"[QUARTERLY] All missing instruments previously failed, returning cached data")
                    return cached
                self.logger.warning(f"[QUARTERLY] Global cache missing {len(active_missing)} instruments: {active_missing[:5]}... Will refetch")
                records.append(cached.reset_index())
                to_fetch = active_missing
            else:
                self.logger.info(f"[QUARTERLY] Cache stale: cached={cached_dates.min()} to {cached_dates.max()}, requested={self.start_date} to {self.end_date}")
                self.logger.info(f"[QUARTERLY] Trying to rebuild from individual caches...")
                rebuilt, to_fetch = self._rebuild_from_individual_caches("quarterly")
                if rebuilt is not None:
                    self.logger.info(f"[QUARTERLY] Rebuilt from individual caches, need to fetch {len(to_fetch)} more instruments")
                    records.append(rebuilt.reset_index())
                else:
                    to_fetch = list(self.instruments)
        else:
            self.logger.info(f"[QUARTERLY] Global aggregate cache not found, trying to rebuild from individual caches...")
            rebuilt, to_fetch = self._rebuild_from_individual_caches("quarterly")
            if rebuilt is not None:
                records.append(rebuilt.reset_index())
            else:
                to_fetch = list(self.instruments)

        if to_fetch:
            self.logger.info(f"[QUARTERLY] Fetching quarterly fundamentals for {len(to_fetch)} instruments (from cache miss)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fut_map = {}
                for inst in to_fetch:
                    code = self._inst_to_code(inst)
                    fut = ex.submit(
                        self._fetch_single_financial,
                        code, self.start_year,
                    )
                    fut_map[fut] = inst

                for fut in as_completed(fut_map):
                    inst = fut_map[fut]
                    df = fut.result()
                    if df is None or len(df) == 0:
                        self._save_inst_cache(inst, "quarterly", pd.DataFrame())
                        self.logger.warning(f"[QUARTERLY] Failed to fetch quarterly fundamentals for {inst}")
                        continue
                    cols = [c for c in ["日期"] + list(self.QUARTERLY_COLUMNS.values()) if c in df.columns]
                    if not cols:
                        self._save_inst_cache(inst, "quarterly", pd.DataFrame())
                        self.logger.warning(f"[QUARTERLY] No valid columns in quarterly data for {inst}")
                        continue
                    temp = df[cols].copy()
                    temp["日期"] = pd.to_datetime(temp["日期"])
                    temp["instrument"] = inst
                    renaming = {v: k for k, v in self.QUARTERLY_COLUMNS.items()}
                    temp = temp.rename(columns=renaming)
                    for col in temp.columns:
                        if col not in ["instrument", "日期"]:
                            temp[col] = pd.to_numeric(temp[col], errors="coerce")
                    self._save_inst_cache(inst, "quarterly", temp)
                    if not temp.empty:
                        records.append(temp)
                        missing_cols = set(self.QUARTERLY_COLUMNS.keys()) - set(temp.columns)
                        if missing_cols:
                            self.logger.warning(f"[QUARTERLY] Missing columns in fetched data for {inst}: {missing_cols}")
                        else:
                            self.logger.debug(f"[QUARTERLY] Fetched quarterly data for {inst}: {len(temp)} rows, date range: {temp['日期'].min()} to {temp['日期'].max()}")

        if not records:
            self.logger.warning("[QUARTERLY] No quarterly fundamentals fetched for any instrument")
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result = result.set_index(["instrument", "日期"]).sort_index()

        # 合并毛利率数据
        gm_data = self._fetch_all_gross_margin()
        if not gm_data.empty:
            result["gross_margin"] = gm_data["gross_margin"]
            result = result.sort_index()
            self.logger.info(f"[QUARTERLY] Merged gross margin: {len(result)} rows")

        self._save_inst_cache("_ALL_", "quarterly_fundamentals", result)
        return result

    def get_all(self, trade_calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """
        获取所有基本面数据并对齐到交易日

        这是主入口函数，将日线和季度数据合并，
        并将季度数据对齐到交易日期（季度财务数据在报告期后第一个交易日生效）。

        Args:
            trade_calendar: 交易日期索引

        Returns:
            合并后的基本面数据 DataFrame，MultiIndex (instrument, datetime)
        """
        self.logger.info(f"[FUNDAMENTAL_MERGE] Merging data: {len(trade_calendar)} trading days, {len(self.instruments)} instruments")
        daily = self.get_daily_fundamentals()
        quarterly = self.get_quarterly_fundamentals()

        if daily.empty and quarterly.empty:
            self.logger.warning("[FUNDAMENTAL_MERGE] No fundamental data available (both daily and quarterly empty)")
            return pd.DataFrame()

        inst_list = sorted(set(
            list(daily.index.get_level_values(0)) if not daily.empty else []
        ) | set(
            list(quarterly.index.get_level_values(0)) if not quarterly.empty else []
        ))
        self.logger.info(f"[FUNDAMENTAL_MERGE] Instruments in data: {len(inst_list)}")
        self.logger.info(f"[FUNDAMENTAL_MERGE] Daily rows: {len(daily)}, Quarterly rows: {len(quarterly)}")

        # 创建完整的 (instrument, datetime) 索引
        full_idx = []
        for inst in inst_list:
            for dt in trade_calendar:
                full_idx.append((inst, dt))
        full_idx = pd.MultiIndex.from_tuples(full_idx, names=["instrument", "datetime"])

        combined = pd.DataFrame(index=full_idx)

        # 合并日线数据
        if not daily.empty:
            daily_idx = daily.copy()
            daily_idx.index.names = ["instrument", "datetime"]
            combined["turnover"] = daily_idx["turnover"].reindex(full_idx)
            combined["outstanding_share"] = daily_idx["outstanding_share"].reindex(full_idx)
            # 前向填充：假设流通股数在两次报告之间不变
            combined["outstanding_share"] = combined.groupby(level=0)["outstanding_share"].ffill()
            outstanding_count = combined["outstanding_share"].notna().sum()
            self.logger.info(f"[FUNDAMENTAL_MERGE] Daily data merged: {len(daily_idx)} rows -> {outstanding_count} non-NaN outstanding_share values")

        # 合并季度数据
        if not quarterly.empty:
            self.logger.info("[FUNDAMENTAL_MERGE] Aligning quarterly financial data...")
            q_idx = quarterly.copy()
            q_idx.index.names = ["instrument", "datetime"]

            # 将季度报告日期 shift 到下一个交易日（财务数据有发布延迟）
            shifted_quarters = []
            q_insts = set(q_idx.index.get_level_values(0))
            for inst in inst_list:
                if inst not in q_insts:
                    continue
                inst_q = q_idx.loc[[inst]].copy()
                inst_q = inst_q.droplevel("instrument")
                # 去重，保留最后一个
                inst_q = inst_q[~inst_q.index.duplicated(keep="last")]
                shifted_dates = []
                for d in inst_q.index:
                    mask = trade_calendar >= d
                    if mask.any():
                        shifted_dates.append(trade_calendar[mask][0])
                    else:
                        shifted_dates.append(d)
                inst_q.index = pd.DatetimeIndex(shifted_dates, name="datetime")
                inst_q["instrument"] = inst
                inst_q = inst_q.reset_index().set_index(["instrument", "datetime"])
                shifted_quarters.append(inst_q)
                self.logger.debug(f"[FUNDAMENTAL_MERGE] {inst} quarterly dates shifted: {len(shifted_dates)} report dates")

            if shifted_quarters:
                q_shifted = pd.concat(shifted_quarters).sort_index()
                q_shifted = q_shifted[~q_shifted.index.duplicated(keep="last")]
                self.logger.info(f"[FUNDAMENTAL_MERGE] Shifted quarterly data: {len(q_shifted)} rows, {len(q_shifted.index.get_level_values(0).unique())} instruments")
            else:
                q_shifted = pd.DataFrame()

            # 合并季度数据到完整索引
            if not q_shifted.empty:
                for col in list(self.QUARTERLY_COLUMNS.keys()) + (["gross_margin"] if "gross_margin" in q_shifted.columns else []):
                    if col in q_shifted.columns:
                        combined[col] = q_shifted[[col]].reindex(full_idx)[col]
                        # 前向填充：季度数据在下一个报告前保持不变
                        combined[col] = combined.groupby(level=0)[col].ffill()
                        non_na_count = combined[col].notna().sum()
                        total_count = len(combined)
                        pct = 100 * non_na_count / total_count
                        self.logger.info(f"[FUNDAMENTAL_MERGE] Column '{col}' merged: {non_na_count}/{total_count} ({pct:.1f}%) non-NaN values")
                        if pct < 50:
                            self.logger.warning(f"[FUNDAMENTAL_MERGE] Low coverage for '{col}': only {pct:.1f}% non-NaN")

        combined = combined.dropna(how="all")
        self.logger.info(f"[FUNDAMENTAL_MERGE] Final combined data: {len(combined)} rows, {len(combined.index.get_level_values(0).unique())} instruments")
        return combined

    def clear_cache(self):
        """
        清除所有缓存文件
        """
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink()