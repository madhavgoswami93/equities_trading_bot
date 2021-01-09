import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class MLBacktester():

    def __init__(self, symbol, start, end, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.model = LogisticRegression(C=1e6, max_iter = 100000, multi_class='ovr')
        self.results = None
        self.get_data()

    def __repr__(self):
        rep = "MLBacktester symbol : {} , model : {} , start : {}, end : {}, tc = {}"
        return rep.format(self.symbol,'Logistic Regression',self.start, self.end, self.tc)

    def get_data(self):
        raw = pd.read_csv("Part3_Materials/five_minute_pairs.csv",parse_dates=['time'],index_col = 'time')
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns = {self.symbol:'price'},inplace = True)
        raw['returns'] = np.log(raw.div(raw.shift(1)))
        self.data = raw.dropna()
        return raw

    def select_data(self,start, end):
        data = self.data.loc[start:end].copy()
        return data

    def prepare_features(self,start,end):
        self.data_subset = self.data.loc[start:end].copy()
        self.feature_columns = []
        for lag in range(1,self.lags):
            col = "lag{}".format(lag)
            self.data_subset[col] =self.data_subset['returns'].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace = True)

    def fit_model(self,start, end):
        self.prepare_features( start, end)
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset['returns']))

    def test_strategy(self,start_train, end_train, start_test, end_test, lags = 5):
        self.lags = lags

        self.fit_model(start_train, end_train)

        self.prepare_features(start_test, end_test)

        prediction = self.model.predict(self.data_subset[self.feature_columns])

        self.data_subset["pred"] = prediction

        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]

        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()

        self.data_subset['strategy_minus_ptc'] = self.data_subset.strategy - self.data_subset.trades * self.tc

        # calculate cumulative returns for strategy & buy and hold
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
        self.data_subset["cstrategy_actual"] = self.data_subset['strategy_minus_ptc'].cumsum().apply(np.exp)
        self.results = self.data_subset

        # absolute performance of the strategy

        perf_actual = self.results['cstrategy_actual'].iloc[-1]
        perf = self.results["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf_actual - self.results["creturns"].iloc[-1]

        return round(perf_actual,6), round(perf, 6), round(outperf, 6)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy","cstrategy_actual"]].plot(title=title, figsize=(12, 8))

if __name__ == '__main__':

    symbol = 'EURUSD'
    ptc = 0.007

    mlb = MLBacktester(symbol, '2019-01-01','2020-08-31',ptc)

    print(mlb)

    print(mlb.test_strategy('2019-01-01','2020-01-01','2020-01-01','2020-08-31',15))

    mlb.plot_results()
