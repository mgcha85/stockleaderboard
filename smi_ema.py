import backtrader as bt
import pandas as pd
from database import Database
import json

db = Database()

# Stochastic Momentum Index Indicator
class StochasticMomentumIndex(bt.Indicator):
    lines = ('smi', 'signal',)
    params = (('period', 14), ('smooth_period', 3),)

    def __init__(self):
        stoch = bt.indicators.Stochastic(self.data, period=self.p.period)
        self.lines.smi = stoch.lines.percK
        self.lines.signal = bt.indicators.EMA(self.lines.smi, period=self.p.smooth_period)

# EMA Trend Cloud Indicator
class EMATrendCloud(bt.Indicator):
    lines = ('ema', )
    params = (('period', 9),)

    def __init__(self):
        self.lines.ema = bt.indicators.EMA(self.data, period=self.p.period)


# Define the strategy
class StochasticRSIStrategy(bt.Strategy):
    params = (
        ('smi_period', 14),
        ('ema_period', 9),
        ('stop_loss_factor', 0.02),  # Define the stop loss percentage
        ('profit_to_loss_ratio', 2),  # Target profit-to-loss ratio
    )

    def __init__(self):
        self.smi = StochasticMomentumIndex(self.data, period=self.p.smi_period)
        self.ema_trend = EMATrendCloud(self.data, period=self.p.ema_period)
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
                self.stop_loss_price = self.entry_price * (1 - self.params.stop_loss_factor)
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                self.trade_count += 1  # Increment trade count on sell
                self.entry_price = order.executed.price
                self.stop_loss_price = self.entry_price * (1 + self.params.stop_loss_factor)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Entry conditions
        if not self.position:  # No open position
            if self.smi.smi[0] > self.ema_trend.ema[0]:  # Long entry condition
                self.order = self.buy()
                self.entry_price = self.data.close[0]
            elif self.smi.smi[0] < self.ema_trend.ema[0]:  # Short entry condition
                self.order = self.sell()
                self.entry_price = self.data.close[0]

        # Exit conditions
        if self.position.size > 0:  # For long positions
            if (self.data.close[0] - self.entry_price) / self.entry_price >= self.p.profit_to_loss_ratio * self.p.stop_loss_factor:
                self.order = self.close()  # Take profit
            elif self.data.close[0] < self.entry_price * (1 - self.p.stop_loss_factor):
                self.order = self.close()  # Stop loss

        elif self.position.size < 0:  # For short positions
            if (self.entry_price - self.data.close[0]) / self.entry_price >= self.p.profit_to_loss_ratio * self.p.stop_loss_factor:
                self.order = self.close()  # Take profit
            elif self.data.close[0] > self.entry_price * (1 + self.p.stop_loss_factor):
                self.order = self.close()  # Stop loss

# Backtesting function
def run_backtest(strategy, data_feed, initial_cash=100000, interval='1d'):
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
    profit = final_portfolio_value - initial_cash
    sharpe_ratio = strategy_instance.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    drawdown = strategy_instance.analyzers.drawdown.get_analysis()['max']['drawdown']
    total_returns = strategy_instance.analyzers.returns.get_analysis()['rtot']
    total_trades = strategy_instance.trade_count

    # Serialize strategy parameters
    conditions = json.dumps({
        'smi_period': strategy_instance.params.smi_period,
        'ema_period': strategy_instance.params.ema_period,
        'stop_loss_factor': strategy_instance.params.stop_loss_factor,
        'profit_to_loss_ratio': strategy_instance.params.profit_to_loss_ratio
    })

    # Prepare data for database insertion
    result_data = {
        'strategy_name': strategy.__name__,
        'conditions': conditions,
        'invest_type': 'backtest',
        'side': 'both',
        'initial_capital': initial_cash,
        'final_portfolio_value': final_portfolio_value,
        'profit': total_returns,
        'SharpeRatio': sharpe_ratio,
        'Drawdown': drawdown,
        'total_trades': total_trades,
        'take_profit': strategy_instance.params.profit_to_loss_ratio,
        'stoploss': strategy_instance.params.stop_loss_factor,
        'interval': interval
    }

    # Save to database
    db.insert_by_series('cryptocurrency.leaderboard', pd.Series(result_data))

    # Print results
    print("Final Portfolio Value: %.2f" % final_portfolio_value)
    print("Profit: %.2f" % profit)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Drawdown:", drawdown)
    print("Total Returns:", total_returns)
    print(f'Total Trades: {total_trades}')


if __name__ == '__main__':
    db = Database()
    interval = '1d'

    if interval == '1d':
        # Fetch data from the database
        data_feed_df = pd.read_sql("SELECT * FROM `BTCUSDT_1d`", db.engine)
    else:
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
    
    if interval != '1d':
        data_feed_df = data_feed_df.resample(interval).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

    data_feed = bt.feeds.PandasData(dataname=data_feed_df)

    chosen_strategy = StochasticRSIStrategy
    run_backtest(chosen_strategy, data_feed)
