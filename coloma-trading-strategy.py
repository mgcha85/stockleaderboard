import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt
from backtrader import Cerebro, Strategy, analyzers
from database import Database
import csv

db = Database()

# 1. Algorithm Selection: Using Python with essential libraries
# (Note: This script will not include the actual import and setup of these libraries)

# 3. Indicator Calculation
def calculate_ema(prices, days):
    return prices.ewm(span=days, adjust=False).mean()

def calculate_stochastic_oscillator(low, high, close, k_period, d_period):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    K = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    D = K.rolling(window=d_period).mean()
    return K, D

# 4. Strategy Logic
class ColomaStrategy(bt.Strategy):
    params = (
        ('ema_short_period', 10),
        ('ema_medium_period', 20),
        ('ema_long_period', 50),
        ('stochastic_period', 14),
        ('stochastic_d_period', 3),
        ('rise_period_short', 30),  # 1 month
        ('rise_period_long', 90),  # 3 months
        ('consolidation_period', 60),  # 2 months
        ('rise_threshold', 0.05),  # 30%
    )

    def __init__(self):
        # Define indicators
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.params.ema_short_period)
        self.ema_medium = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.params.ema_medium_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.params.ema_long_period)
        self.stochastic = bt.indicators.Stochastic(self.datas[0], period=self.params.stochastic_period, period_dfast=self.params.stochastic_d_period)

        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(self.datas[0].close)

        # ATR for Keltner Channel width
        self.atr = bt.indicators.AverageTrueRange(self.datas[0])
        # Use backtrader's moving average for the midband of Keltner Channels
        self.midband = bt.indicators.MovingAverageSimple(self.datas[0].close, period=self.params.ema_long_period)
        # Define upper and lower bands for Keltner Channels
        self.upperband = self.midband + 2 * self.atr
        self.lowerband = self.midband - 2 * self.atr

        self.order = None
        self.entry_high = None
        self.entry_low = None
        self.position_opened = False
        self.partially_closed = False
        self.stop_loss_hit = False

        self.trade_count = 0
        self.trade_list = []

        # CSV 파일 헤더 작성
        self.csvfile = open(f'{self.__class__.__name__}.csv', 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['Entry Date', 'Entry Price', 'Quantity', 'Entry Value', 'Exit Date1', 'Exit Price1', 'Exit Date2', 'Exit Price2', 'Sell Reason', 'Abs Profit', 'Pct Profit'])
        self.current_trade_info = {}
        self.in_position = False
        self.exit_count = 0
        self.trade_count = 0

    def log(self, text, dt=None):
        """ 로그 메시지를 출력하는 함수 """
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), text))

    def notify_order(self, order):
        if order.isbuy():
            self.in_position = True
            self.stop_loss_hit = False

            self.current_trade_info = {
                'entry_date': self.data.datetime.date(0).isoformat(),
                'entry_price': order.executed.price,
                'quantity': order.executed.size,
                'entry_value': order.executed.value,
                'exit_price1': 0,
                'exit_price2': 0,
                'exit_date1': None,
                'exit_date2': None,
                'sell_reason': None
            }
        elif order.issell() and self.in_position:
            if self.exit_count < 2:
                self.exit_count += 1
                self.current_trade_info[f'exit_date{self.exit_count}'] = self.data.datetime.date(0).isoformat()
                self.current_trade_info[f'exit_price{self.exit_count}'] = order.executed.price
                if self.stop_loss_hit:
                    self.current_trade_info['sell_reason'] = 'stop_loss'
                else:
                    self.current_trade_info['sell_reason'] = 'normal'
                self.trade_count += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            required_keys = ['entry_date', 'entry_price', 'quantity', 'entry_value']
            if all(key in self.current_trade_info for key in required_keys):

                # Calculate profit percentage and absolute profit
                exit_value1 = self.current_trade_info.get('exit_price1', 0) * self.current_trade_info['quantity']
                exit_value2 = self.current_trade_info.get('exit_price2', 0) * self.current_trade_info['quantity']
                pnl = exit_value1 + exit_value2 - self.current_trade_info['entry_value']
                pct_profit = (pnl / self.current_trade_info['entry_value']) if self.current_trade_info['entry_value'] != 0 else 0

                # Write to CSV
                self.writer.writerow([
                    self.current_trade_info['entry_date'],
                    self.current_trade_info['entry_price'],
                    self.current_trade_info['quantity'],
                    self.current_trade_info['entry_value'],
                    self.current_trade_info['exit_date1'],
                    self.current_trade_info['exit_price1'],
                    self.current_trade_info['exit_date2'],
                    self.current_trade_info['exit_price2'],
                    self.current_trade_info['sell_reason'],
                ] + [pnl, pct_profit])

                self.current_trade_info = {}
                self.exit_count = 0
                self.trade_count += 1  # Increment trade count here
            else:
                # Log a message if the trade closure is missing information
                print(f"Trade closure missing information: {self.current_trade_info}")

    def stop(self):
        # Close CSV file and print total trades
        self.csvfile.close()
        print(f'Total Trades: {self.trade_count}')

    def next(self):
        # Check for rise and consolidation conditions for entry
        if not self.position_opened and len(self) >= self.params.rise_period_long + self.params.consolidation_period:
            start_price = self.data.close[-self.params.rise_period_long - self.params.consolidation_period]
            end_price = self.data.close[-self.params.consolidation_period]
            rise_percentage = (end_price - start_price) / start_price
            rise_condition = self.params.rise_threshold <= rise_percentage <= 1.0

            consolidation_condition = all([
                self.bollinger.lines.top[-1] <= self.upperband[-1],
                self.bollinger.lines.bot[-1] >= self.lowerband[-1]
            ])

            if rise_condition and consolidation_condition and self.data.close[0] > self.ema_short[0]:
                self.buy(size=(self.broker.getcash() / self.data.close[0]) * 0.1)
                self.entry_high = self.data.high[0]
                self.entry_low = self.data.low[0]
                self.position_opened = True

        # Partial and full exit conditions
        if self.position_opened:
            # Update stop loss if current price is higher than entry high
            if self.data.close[0] > self.entry_high:
                self.entry_high = self.data.close[0]

            # Partial exit on first bearish candle after bullish one
            if not self.partially_closed and self.data.close[-1] > self.data.open[-1] and self.data.close[0] < self.data.open[0]:
                if self.data.low[0] > self.entry_low:
                    self.sell(size=self.position.size * 0.3)
                    self.partially_closed = True

            if self.data.close[0] < self.entry_low:
                self.close()
                self.stop_loss_hit = True  # Set flag if stop loss condition met
                self.position_opened = False

            # Full exit if price drops below EMA 20-day line
            if self.data.close[0] < self.ema_medium[0]:
                self.close()
                self.position_opened = False

# 5. Risk Management
# This will be part of the strategy's logic in the ColomaStrategy class

# 6. Backtesting
def run_backtest(strategy, data_feed, initial_cash=100000):
    cerebro = Cerebro()
    cerebro.addstrategy(strategy)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.addanalyzer(analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(analyzers.Returns, _name='returns')

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    strategy_instance = results[0]
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print("Sharpe Ratio:", strategy_instance.analyzers.sharpe_ratio.get_analysis())
    print("Drawdown:", strategy_instance.analyzers.drawdown.get_analysis())
    print("Total Returns:", strategy_instance.analyzers.returns.get_analysis())
    print(f'Total Trades: {strategy_instance.trade_count}')

# 7. Optimization
# Further optimization can be done based on backtesting results

# 8. Validation
# This involves running the strategy on out-of-sample data


if __name__ == '__main__':
    # Fetch data from the database
    data_feed_df = pd.read_sql("SELECT * FROM `BTCUSDT_1m` where time between '2022-01-01' and '2023-12-31'", db.engine)
    data_feed_df['time'] = pd.to_datetime(data_feed_df['time'])
    data_feed_df.set_index('time', inplace=True)
    data_feed_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data_feed_df = data_feed_df.resample('15T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    data_feed = bt.feeds.PandasData(dataname=data_feed_df)
    run_backtest(ColomaStrategy, data_feed)
