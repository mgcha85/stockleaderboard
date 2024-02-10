import backtrader as bt
import pandas as pd
import math
from database import Database
import csv
import json
from datetime import datetime

db = Database()


class KeltnerChannel(bt.Indicator):
    lines = ('mid', 'top', 'bot',)
    params = (
        ('period', 20),
        ('devfactor', 1.5),
        ('movav', bt.indicators.MovAv.EMA),
    )

    def __init__(self):
        self.l.mid = self.p.movav(self.data, period=self.p.period)
        self.l.top = self.l.mid + self.p.devfactor * bt.indicators.AverageTrueRange(self.data, period=self.p.period)
        self.l.bot = self.l.mid - self.p.devfactor * bt.indicators.AverageTrueRange(self.data, period=self.p.period)

class TTMSqueeze(bt.Indicator):
    lines = ('momentum', 'squeeze_on', 'squeeze_off',)
    params = dict(
        bollinger_period=20,
        bollinger_dev=2.0,
        keltner_period=20,
        keltner_dev=1.5,
        momentum_period=12,
    )

    def __init__(self):
        # Bollinger Bands
        self.boll = bt.indicators.BollingerBands(self.data, period=self.p.bollinger_period, devfactor=self.p.bollinger_dev)

        # Keltner Channel
        self.keltner = KeltnerChannel(self.data, period=self.p.keltner_period, devfactor=self.p.keltner_dev)

        # Momentum
        self.l.momentum = bt.indicators.MomentumOscillator(self.data, period=self.p.momentum_period)

        # Squeeze logic
        self.l.squeeze_on = bt.And(
            self.boll.lines.bot > self.keltner.lines.bot,
            self.boll.lines.top < self.keltner.lines.top
        )
        
        # Squeeze off logic
        self.l.squeeze_off = bt.indicators.CrossOver(self.boll.lines.bot, self.keltner.lines.bot)

    def next(self):
        self.l.momentum[0] = self.l.momentum[0]

# Define the strategy
class TTMStrategy(bt.Strategy):
    params = (
        ('ema_period', 20),
        ('risk_reward_ratio', 1.5),
        ('stop_loss', 0.02),  # Define the stop loss percentage
    )

    def __init__(self):
        self.ema = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_period)
        self.ttm_squeeze = TTMSqueeze(self.datas[0])
        self.momentum = self.ttm_squeeze.momentum
        self.squeeze_on = self.ttm_squeeze.squeeze_on
        self.squeeze_off = self.ttm_squeeze.squeeze_off

        self.order = None
        self.entry_price = None
        self.stop_loss_price = None
        self.trade_count = 0  # Initialize trade count

    def log(self, text, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {text}')
        # Add additional logs for debugging
        # print(f'Close: {self.datas[0].close[0]}, EMA: {self.ema[0]}, Squeeze Off: {self.squeeze_off[0]}, Momentum: {self.momentum[0]}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
                self.trade_count += 1  # Increment trade count on buy
                self.entry_price = order.executed.price
                self.stop_loss_price = self.entry_price * (1 - self.params.stop_loss)
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                self.trade_count += 1  # Increment trade count on sell
                self.entry_price = order.executed.price
                self.stop_loss_price = self.entry_price * (1 + self.params.stop_loss)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Example of additional logging
        # self.log(f'Current Close: {self.datas[0].close[0]}')
        # self.log(f'EMA: {self.ema[0]}')
        # self.log(f'Squeeze off: {self.squeeze_off[0]}')
        # self.log(f'Momentum: {self.momentum[0]}')

        # Cancel existing orders
        if self.order:
            self.cancel(self.order)

        # Update stop loss price for open positions
        if self.position.size > 0 and self.datas[0].close[0] < self.stop_loss_price:
            self.log('STOP LOSS HIT ON LONG POSITION')
            self.close()  # Close long position

        if self.position.size < 0 and self.datas[0].close[0] > self.stop_loss_price:
            self.log('STOP LOSS HIT ON SHORT POSITION')
            self.close()  # Close short position

        # Long Position Entry
        if self.squeeze_off and self.momentum[0] > 0 and self.datas[0].close[0] > self.ema[0]:
            if not self.position or self.position.size < 0:
                self.order = self.close()  # Close any short positions
                self.order = self.buy()
                self.entry_price = self.datas[0].close[0]

        # Short Position Entry
        elif self.squeeze_off and self.momentum[0] < 0 and self.datas[0].close[0] < self.ema[0]:
            if not self.position or self.position.size > 0:
                self.order = self.close()  # Close any long positions
                self.order = self.sell()
                self.entry_price = self.datas[0].close[0]

        # Long Position Exit
        if self.position.size > 0:
            if (self.datas[0].close[0] - self.entry_price) / self.entry_price >= self.params.risk_reward_ratio:
                self.order = self.sell()  # Take Profit based on risk-reward ratio
            elif self.momentum[0] < 0 or self.squeeze_on:  # Momentum turns gray or squeeze turns red
                self.order = self.sell()  # Exit long position

        # Short Position Exit
        if self.position.size < 0:
            if (self.entry_price - self.datas[0].close[0]) / self.entry_price >= self.params.risk_reward_ratio:
                self.order = self.close()  # Take Profit based on risk-reward ratio
            elif self.momentum[0] > 0 or self.squeeze_on:  # Momentum turns blue or squeeze turns red
                self.order = self.close()  # Exit short position


def run_backtest(strategy, data_feed, initial_cash=100000):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    strategy_instance = results[0]

    # Collect results
    final_portfolio_value = cerebro.broker.getvalue()
    sharpe_ratio = strategy_instance.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    drawdown = strategy_instance.analyzers.drawdown.get_analysis()['max']['drawdown']
    total_returns = strategy_instance.analyzers.returns.get_analysis()['rtot']
    total_trades = strategy_instance.trade_count

    conditions = json.dumps({
        'ema_period': strategy_instance.params.ema_period,
        'risk_reward_ratio': strategy_instance.params.risk_reward_ratio,
        'stop_loss': strategy_instance.params.stop_loss
    })

    # Prepare data for database insertion
    result_data = pd.Series({
        'strategy_name': strategy.__name__,
        'conditions': conditions,
        'invest_type': 'backtest',  # Assuming backtest as invest_type
        'side': 'both',  # Assuming strategy trades both sides
        'initial_capital': initial_cash,
        'final_portfolio_value': final_portfolio_value,
        'profit': total_returns,
        'SharpeRatio': sharpe_ratio,
        'Drawdown': drawdown,
        'total_trades': total_trades,
        'take_profit': strategy_instance.params.risk_reward_ratio,
        'stoploss': strategy_instance.params.stop_loss,
        'interval': interval
    })

    # Save to database
    db.insert_by_series('cryptocurrency.leaderboard', result_data)

    # Print results
    print("Final Portfolio Value: %.2f" % final_portfolio_value)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Drawdown:", drawdown)
    print("Total Returns:", total_returns)
    print(f'Total Trades: {total_trades}')


if __name__ == '__main__':
    db = Database()
    for interval in ['15T', '30T', '1h', '1d']:
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
        data_feed_df = data_feed_df.resample(interval).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        data_feed = bt.feeds.PandasData(dataname=data_feed_df)

        # Choose the strategy you want to run
        chosen_strategy = TTMStrategy  # or ColomaStrategy
        run_backtest(chosen_strategy, data_feed)
