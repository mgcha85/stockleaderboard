import backtrader as bt
import pandas as pd
from database import Database
import json
import math

db = Database()

# Define the trading strategy
class EmaCrossStrategy(bt.Strategy):
    params = (
        ('ema_fast', 20),
        ('ema_slow', 200),
        ('stop_loss_factor', 0.02),  # Define the stop loss percentage
        ('profit_to_loss_ratio', 3),  # Target profit-to-loss ratio
    )

    def __init__(self):
        self.ema_fast = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_fast)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_slow)
        self.order = None
        self.entry_price = None
        self.stop_loss_price = None
        self.trade_count = 0  # Initialize trade count

    def log(self, text, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {text}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
                self.entry_price = order.executed.price
                self.stop_loss_price = self.entry_price * (1 - self.params.stop_loss_factor)
                self.trade_count += 1
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                self.entry_price = order.executed.price
                self.stop_loss_price = self.entry_price * (1 + self.params.stop_loss_factor)
                self.trade_count += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Entry logic
        if not self.position:
            # Long Entry
            if self.ema_fast[0] > self.ema_slow[0] and self.data.close[0] > max(self.ema_fast[0], self.ema_slow[0]):
                self.buy()
            # Short Entry
            elif self.ema_fast[0] < self.ema_slow[0] and self.data.close[0] < min(self.ema_fast[0], self.ema_slow[0]):
                self.sell()

        # Exit logic
        else:
            if self.position.size > 0:
                # Long exit
                profit = self.data.close[0] - self.entry_price
                if profit / self.entry_price >= self.params.profit_to_loss_ratio * self.params.stop_loss_factor:
                    self.close()  # Take profit for long position
                elif self.data.close[0] < self.stop_loss_price:
                    self.close()  # Stop loss for long position

            elif self.position.size < 0:
                # Short exit
                profit = self.entry_price - self.data.close[0]
                if profit / self.entry_price >= self.params.profit_to_loss_ratio * self.params.stop_loss_factor:
                    self.close()  # Take profit for short position
                elif self.data.close[0] > self.stop_loss_price:
                    self.close()  # Stop loss for short position

def run_backtest(strategy, data_feed, initial_cash=100000, interval='1d', symbol='BTCUSDT'):
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

    # Check for nan values and handle them
    def handle_nan(value, default=0):
        return value if not math.isnan(value) else default
    
    # Collect results
    final_portfolio_value = handle_nan(cerebro.broker.getvalue())
    sharpe_ratio = handle_nan(strategy_instance.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0))
    drawdown = handle_nan(strategy_instance.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0))
    total_returns = handle_nan(strategy_instance.analyzers.returns.get_analysis().get('rtot', 0))
    total_trades = strategy_instance.trade_count

    # Serialize strategy parameters
    conditions = json.dumps({
        'ema_fast': strategy_instance.params.ema_fast,
        'ema_slow': strategy_instance.params.ema_slow,
        'stop_loss_factor': strategy_instance.params.stop_loss_factor,
        'profit_to_loss_ratio': strategy_instance.params.profit_to_loss_ratio,
    })

    # Prepare data for database insertion
    result_data = pd.Series({
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
        'interval': interval,
        'symbol': symbol
    })

    # Save to database
    db = Database()  # Ensure you have a Database instance
    db.insert_by_series('cryptocurrency.leaderboard', result_data)

    # Print results
    print("Final Portfolio Value: %.2f" % final_portfolio_value)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Drawdown:", drawdown)
    print("Total Returns:", total_returns)
    print(f'Total Trades: {total_trades}')


if __name__ == '__main__':
    db = Database()
    interval = '5T'
    symbol = 'BTCUSDT'

    # Fetch data from the database
    if interval[-1].lower() in ['d', 'w', 'm']:
        # Fetch data from the database
        data_feed_df = pd.read_sql(f"SELECT * FROM `{symbol}_1d` where time between '2022-01-01' and '2023-12-31'", db.engine)
    else:
        data_feed_df = pd.read_sql(f"SELECT * FROM `{symbol}_1m` where time between '2022-01-01' and '2023-12-31'", db.engine)
    data_feed_df['time'] = pd.to_datetime(data_feed_df['time'])
    data_feed_df.set_index('time', inplace=True)
    data_feed_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    if interval not in ['1d', '1m']:
        data_feed_df = data_feed_df.resample(interval).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    data_feed = bt.feeds.PandasData(dataname=data_feed_df)

    chosen_strategy = EmaCrossStrategy
    run_backtest(chosen_strategy, data_feed, initial_cash=100000, interval=interval, symbol=symbol)
