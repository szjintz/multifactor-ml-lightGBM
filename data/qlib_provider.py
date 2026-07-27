"""
Qlib data provider module

Wraps Qlib data interfaces, providing:
- Daily market data (OHLCV, VWAP, adjustment factors, etc.)
- Trade calendar
- Stock universe (investable range)
- Market cap data
- Industry classification

Data characteristics:
- Uses Qlib as the base data source
- Supports CSI300, CSI500, CSI800 index constituents
- Data format: MultiIndex (instrument, datetime)
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
    Qlib data provider

    Wraps Qlib's data fetching interfaces to provide a unified data access layer.
    Supports fetching market data from Qlib and augmenting with fundamental data.
    """

    def __init__(self, market="csi300", start_date="2022-01-01", end_date="2026-06-30",
                 cache_dir=None):
        """
        Initialize data provider

        Args:
            market: Stock pool identifier, e.g. "csi300", "csi500"
            start_date: Data start date
            end_date: Data end date
            cache_dir: Cache directory path (for fundamental data)
        """
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self._fundamental_provider = None

    @property
    def fundamental_provider(self):
        """
        Get fundamental data provider (lazy initialization)

        Creates instance only on first access.
        """
        if self._fundamental_provider is None:
            logger.info(f"[QLIB] Initializing fundamental provider with {len(self.get_all_instruments())} instruments")
            all_inst = self.get_all_instruments()
            self._fundamental_provider = FundamentalProvider(
                instruments=all_inst,
                start_date=self.start_date,
                end_date=self.end_date,
                cache_dir=self.cache_dir,
            )
            logger.info(f"[QLIB] Fundamental provider initialized: {len(all_inst)} instruments, cache_dir={self.cache_dir}")
        return self._fundamental_provider

    def get_all_instruments(self):
        """
        Get all instrument codes in the stock pool

        Returns:
            List of instrument codes, e.g. ["SH600000", "SZ000001", ...]
        """
        logger.info(f"[QLIB] Getting all instruments, market={self.market}, date range: {self.start_date} to {self.end_date}")
        instruments = D.instruments(market=self.market)
        logger.info(f"[QLIB] Retrieved {len(instruments)} instruments from Qlib")
        return list(D.list_instruments(instruments, self.start_date))

    def get_trade_dates(self):
        """
        Get trade calendar

        Returns:
            Sorted list of trade dates
        """
        logger.info(f"[QLIB] Getting trade calendar, range: {self.start_date} to {self.end_date}")
        trade_dates = D.calendar(start_time=self.start_date, end_time=self.end_date)
        logger.info(f"[QLIB] Trade calendar: {len(trade_dates)} trading days, range: {trade_dates[0]} to {trade_dates[-1]}")
        return trade_dates

    def get_universe(self, date):
        """
        Get stock universe for a given date

        Args:
            date: Query date

        Returns:
            List of investable instrument codes for that date
        """
        instruments = D.instruments(market=self.market)
        universe = D.list_instruments(instruments, date)
        logger.debug(f"[QLIB] Universe {date}: {len(universe)} instruments")
        return universe

    def get_daily_data(self, fields=None):
        """
        Get daily market data

        Default fields: open, high, low, close, volume, amount, VWAP, change, factor

        Args:
            fields: List of field names, None uses defaults

        Returns:
            Daily market data DataFrame, MultiIndex (instrument, datetime)
        """
        if fields is None:
            fields = [
                "$open", "$high", "$low", "$close", "$volume", "$amount",
                "$vwap", "$change", "$factor",
            ]
        logger.info(f"[QLIB] Getting daily data: {len(fields)} fields, range: {self.start_date} to {self.end_date}")
        instruments = D.instruments(market=self.market)
        data = D.features(instruments, fields, self.start_date, self.end_date)
        data.columns = data.columns.map(lambda x: x.lstrip("$"))
        logger.info(f"[QLIB] Daily data: {len(data)} rows, {len(data.columns)} cols, date range: {data.index.get_level_values(1).min()} to {data.index.get_level_values(1).max()}")
        return data

    def get_augmented_data(self):
        """
        Get augmented data: market + fundamentals

        Integrates Qlib market data and Akshare fundamental data.

        Data sources:
        - Qlib: market data (OHLCV, VWAP, etc.)
        - Akshare: fundamental data (EPS, BVPS, ROE, ROA, debt ratio, revenue growth, gross margin, etc.)
        - Computed: market cap = close price * outstanding shares

        Returns:
            Augmented DataFrame with market and fundamental data
        """
        logger.info("[QLIB] Starting data augmentation pipeline...")
        qlib_data = self.get_daily_data()
        trade_dates = self.get_trade_dates()

        logger.info("[QLIB] Fetching fundamental data...")
        fundamental = self.fundamental_provider.get_all(trade_dates)

        if fundamental.empty:
            logger.warning("[QLIB] No fundamental data available, returning Qlib data only")
            return qlib_data

        logger.info(f"[QLIB] Fundamental data merged: {len(fundamental)} rows")

        # Align index formats
        fundamental.index.names = qlib_data.index.names
        aligned = qlib_data.join(fundamental, how="left")

        logger.info(f"[QLIB] After merge: {len(aligned)} rows, {len(aligned.columns)} cols")

        # Compute market cap if outstanding share data is available
        if "outstanding_share" in aligned.columns and "close" in aligned.columns:
            logger.info("[QLIB] Computing market cap: outstanding_share * close")
            before_count = len(aligned)
            aligned["market_cap"] = aligned["close"] * aligned["outstanding_share"]
            nan_count = aligned["market_cap"].isna().sum()
            logger.info(f"[QLIB] Market cap computed: {before_count - nan_count}/{before_count} non-NaN ({100*(before_count - nan_count)/before_count:.1f}%)")
        else:
            logger.warning("[QLIB] Cannot compute market cap: missing outstanding_share or close column")

        logger.info(f"[QLIB] Augmented data ready: {len(aligned)} rows, {len(aligned.columns)} cols, date range: {aligned.index.get_level_values(1).min()} to {aligned.index.get_level_values(1).max()}")
        return aligned

    def get_market_cap(self, field="$market_cap"):
        """
        Get market cap data

        Args:
            field: Qlib field name, default market cap

        Returns:
            Market cap DataFrame
        """
        data = D.features(
            D.instruments(market=self.market),
            [field],
            self.start_date,
            self.end_date,
        )
        data.columns = data.columns.map(lambda x: x.lstrip("$"))
        logger.info(f"[QLIB] Market cap data: {len(data)} rows, {len(data.columns)} cols")
        return data

    def get_industry(self):
        """
        Get industry classification data

        Returns:
            Industry classification DataFrame
        """
        data = D.features(
            D.instruments(market=self.market),
            ["$industry"],
            self.start_date,
            self.end_date,
        )
        data.columns = data.columns.map(lambda x: x.lstrip("$"))
        logger.info(f"[QLIB] Industry classification data: {len(data)} rows, {len(data.columns)} cols")
        return data
