import backtrader as bt
import pandas as pd
from fbprophet import Prophet
import datetime
import matplotlib.pyplot as plt
# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        print('initiated the strategy {}'.format(self.dataclose[0]))

        # To keep track of pending orders
        self.order = None
        # for i, d in enumerate(self.datas):
        #     if i > 0: #Check we are not on the first loop of data feed:
        #         if self.p.oneplot == True:
        #             d.plotinfo.plotmaster = self.datas[0]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()



class ProphetStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # keep array of dates and closes
        self.date_array = []
        self.close_array = []

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.datas[0].close[0])
        
        # append date and close to arrays
        self.date_array.append(self.datas[0].datetime.date(0))
        self.close_array.append(self.datas[0].close[0])
        
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        
        # make sure we have a decent amount of data
        if len(self.date_array) < 90:
            return
        
        # only invest once a week
        if len(self.date_array) % 5 != 0:
            return
        
        # get predictions
        max_move, expected_move, min_move = self.get_prophet_moves(7, False)
        
        # if the predicted movement is mostly positive, buy
        if max_move > 0 and abs(max_move) > abs(min_move):
            self.log('BUY CREATE, %.2f' % self.datas[0].close[0])

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()
        
        # if the predicted movement is mostly negative, sell
        elif min_move < 0 and abs(min_move) > abs(max_move):
            
            # make sure we have some stock to sell
            if self.position:
                self.log('SELL CREATE, %.2f' % self.datas[0].close[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
                
    def get_prophet_moves(self, daysOut=7, showCharts=False):
        # create stock dataframe for prophet
        stock_df = pd.DataFrame({
            'ds': self.date_array,
            'y': self.close_array
        })

        # fit data using prophet model
        m = Prophet()
        m.fit(stock_df)

        # create future dates
        future_prices = m.make_future_dataframe(periods=365)

        # predict prices
        forecast = m.predict(future_prices)

        # view results
        if showCharts:
            fig = m.plot(forecast)
            ax1 = fig.add_subplot(111)
            ax1.set_title("Stock Price Forecast", fontsize=16)
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel("Close Price", fontsize=12)

            fig2 = m.plot_components(forecast)
            plt.show()

        # calculate predicted returns
        end_of_period = self.datas[0].datetime.date(0) + datetime.timedelta(days=daysOut)

        future_close_max = forecast[forecast['ds'] > end_of_period].iloc[0].yhat_upper
        future_close_expected = forecast[forecast['ds'] > end_of_period].iloc[0].yhat
        future_close_min = forecast[forecast['ds'] > end_of_period].iloc[0].yhat_lower

        # calculate percent changes based on predictions
        max_move = (future_close_max - self.datas[0].close[0])/self.datas[0].close[0]
        expected_move = (future_close_expected - self.datas[0].close[0])/self.datas[0].close[0]
        min_move = (future_close_min - self.datas[0].close[0])/self.datas[0].close[0]

        return (max_move, expected_move, min_move)


