from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cwd = os.
cwd = os.chdir("/home/madhav/Documents/github/equities_trading_bot/sensitive")
instrument_df_loc = "/home/madhav/Documents/github/equities_trading_bot/sensitive/nse_tickers.csv"


class FinancialInstrument:
    def __init__(
        self, ticker, start=dt.date.today() - dt.timedelta(30), end=dt.date.today()
    ):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.get_access()
        self.get_data()
        self.log_returns()

    def __repr__(self):
        return "Financial Instrument (ticker = {}, start={}, end = {})".format(
            self._ticker, self.start, self.end
        )

    def get_access(self):
        access_token = open("access_token.txt", "r").read()
        key_secret = open("api_key.txt", "r").read().split()
        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)
        instrument_df = pd.read_csv(instrument_df_loc)
        self.kite = kite
        self.instrument_df = instrument_df

    def instrumentLookup(self):
        """Looks up instrument token for a given script from instrument dump"""
        try:
            return self.instrument_df[
                self.instrument_df.tradingsymbol == self._ticker
            ].instrument_token.values[0]
        except:
            return -1

    def get_data(self, interval="day"):
        """interval
        The candle record interval. Possible values are:
                · minute
                · day
                · 3minute
                · 5minute
                · 10minute
                · 15minute
                · 30minute
                · 60minute
                extracts historical data and outputs in the form of dataframe
                """
        instrument = self.instrumentLookup()
        data = pd.DataFrame(
            self.kite.historical_data(instrument, self.start, self.end, interval)
        )
        data.set_index("date", inplace=True)
        self.data = data

    def get_data_extended(self, inception_date, interval):
        """extracts historical data and outputs in the form of dataframe
           inception date string format - yyyy-mm-dd"""
        instrument = self.instrumentLookup()
        from_date = dt.datetime.strptime(inception_date, "%Y-%m-%d")
        to_date = dt.date.today()
        data = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        while True:
            if from_date.date() >= (dt.date.today() - dt.timedelta(100)):
                data = data.append(
                    pd.DataFrame(
                        self.kite.historical_data(
                            instrument, from_date, dt.date.today(), interval
                        )
                    ),
                    ignore_index=True,
                )
                break
            else:
                to_date = from_date + dt.timedelta(100)
                data = data.append(
                    pd.DataFrame(
                        self.kite.historical_data(
                            instrument, from_date, to_date, interval
                        )
                    ),
                    ignore_index=True,
                )
                from_date = to_date
        data.set_index("date", inplace=True)
        self.data_df = data

    def log_returns(self):
        self.data["log_returns"] = np.log(self.data.close / self.data.close.shift(1))

    def plot_prices(self):
        self.data.close.plot(figsize=(12, 6))
        plt.title("Price chart : {}".format(self._ticker), fontsize=15)

    def plot_returns(self, kind="ts"):
        if kind == "ts":
            self.data.log_returns.plot(figsize=(12, 8))
            plt.title("Returns : {} ".format(self._ticker), fontsize=15)
        elif kind == "hist":
            self.data.log_returns.hist(
                figsize=(12, 8), bins=int(np.sqrt(len(self.data)))
            )
            plt.title("Frequency of Returns : {}".format(self._ticker), fontsize=15)

    def set_ticker(self, ticker=None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()

    def mean_return(self, freq=None):
        if freq is None:
            return self.data.log_returns.mean()
        else:
            resampled_price = self.data.close.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()

    def std_returns(self, freq=None):
        if freq is None:
            return self.data.log_returns.std()
        else:
            resampled_price = self.data.close.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    def annualized_perf(self):
        """calculates annualized return and risk
        """
        mean_return = round(self.data.log_returns.mean() * 252, 4)
        risk = round(self.data.log_returns.std() * np.sqrt(252), 4)
        print("Return: {} | Risk: {}".format(mean_return, risk))


if __name__ == "__main__":
    stock = FinancialInstrument("RELIANCE")
    stock.get_data_extended("2019-01-01", "15minute")
    print("head of extended data")
    print(stock.data_df.head())
    print("tail of extended data")
    print(stock.data_df.tail())
