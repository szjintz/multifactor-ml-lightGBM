import pandas as pd
import numpy as np
from qlib.data import D
from qlib.config import REG_CN


class QlibDataProvider:
    def __init__(self, market="csi300", start_date="2018-01-01", end_date="2025-06-30"):
        self.market = market
        self.start_date = start_date
        self.end_date = end_date

    def get_trade_dates(self):
        return D.calendar(start_time=self.start_date, end_time=self.end_date)

    def get_universe(self, date):
        instruments = D.instruments(market=self.market)
        return D.list_instruments(instruments, date)

    def get_daily_data(self, fields=None):
        if fields is None:
            fields = [
                "open", "high", "low", "close", "volume", "amount",
                "vwap", "change", "factor",
            ]
        instruments = D.instruments(market=self.market)
        return D.features(instruments, fields, self.start_date, self.end_date)

    def get_market_cap(self):
        return D.features(
            D.instruments(market=self.market),
            ["$close", "$volume"],
            self.start_date,
            self.end_date,
        )

    def get_industry(self):
        return D.features(
            D.instruments(market=self.market),
            ["$industry"],
            self.start_date,
            self.end_date,
        )
