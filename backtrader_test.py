import datetime  # For datetime objects

# Import the backtrader platform
import backtrader as bt
import pandas as pd
import pyfolio as pf

import FinancialInstrument as FI
from orb_strategy import OpeningRangeBreakout
from strategies import ProphetStrategy, TestStrategy

if __name__ == "__main__":
    # Create a cerebro entity

    ticker = ["RELIANCE", "ACC", "DMART", "HDFC", "HDFCBANK"]
    cerebro = bt.Cerebro()

    for tick in ticker[0:1]:
        # for tick in [ticker[0]]:

        stock = FI.FinancialInstrument(tick)
        stock.get_data_extended("2019-01-01", "15minute")

        # Create a Data Feed
        data = bt.feeds.PandasData(
            dataname=stock.data_df,
            fromdate=datetime.datetime(2019, 1, 1),
            todate=datetime.datetime(2020, 8, 31),
            name=tick,
            plot=False,
        )

        # Add the Data Feed to Cerebro
        cerebro.adddata(data)

    # Add the TestStrategy Class
    cerebro.addstrategy(OpeningRangeBreakout)

    # Set our desired cash start
    cerebro.broker.setcash(10000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    # Set the commission
    comm = 0.0
    cerebro.broker.setcommission(commission=comm)

    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    # cerebro.addanalyzer(bt.analyzers.PyFolio)
    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Run over everything
    results = cerebro.run()

    # pyfoliozer = results[0].analyzers.getbyname('pyfolio')

    print(
        f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}"
    )
    print(
        f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%"
    )
    print(
        f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%"
    )

    # returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    print(results[0].analyzers.tradeanalyzer.get_analysis())
    # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # fig = pf.create_returns_tear_sheet(returns, return_fig=True)
    # fig.savefig('returns_tear_sheet.pdf')

    # f = pf.create_returns_tear_sheet(returns, benchmark_rets=benchmark_rets, return_fig=True)
    # f.savefig('pyfolio_returns_tear_sheet.png')

    # pystats_df = pyfolio.timeseries.perf_stats(returns, positions=positions, transactions=transactions)
    # print(pystats_df)

    # Plot the result
    cerebro.plot(iplot=False, style="candlestick")

