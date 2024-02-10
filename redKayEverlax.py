import backtrader as bt
import pandas as pd
import math
import json
from database import Database

# Custom Indicator: RedKay Everlax
class RedKayEverlax(bt.Indicator):
    lines = ('smooth_line', 'signal_line')

    def __init__(self):
        # Hypothetical implementation - replace with actual logic
        self.lines.smooth_line = bt.indicators.SMA(self.data.close, period=20)
        self.lines.signal_line = bt.indicators.SMA(self.data.close, period=50)

# Strategy Class
class RedKayEverlaxStrategy(bt.Strategy):
    params = (
        ('ema_period', 200),
        ('volume_threshold', 1000000),  # Hypothetical volume threshold
        ('take_profit_ratio', 4),
        ('stop_loss', 0.1),
        ('profit_to_loss_ratio', 3),
    )

    def __init__(self):
        self.ema200 = bt.indicators.EMA(self.data, period=self.params.ema_period)
        self.redkay = RedKayEverlax(self.data)
        self.volume = self.data.volume  # Assuming volume is part of the data feed
        self.trade_count = 0

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
        if not self.position:
            # Long Entry
            if self.redkay.smooth_line > self.redkay.signal_line and self.redkay.smooth_line[-1] < -50 and self.volume[0] > self.params.volume_threshold and self.data.close[0] > self.ema200[0]:
                self.buy()

            # Short Entry
            elif self.redkay.smooth_line < self.redkay.signal_line and self.redkay.smooth_line[-1] > 50 and self.volume[0] > self.params.volume_threshold and self.data.close[0] < self.ema200[0]:
                self.sell()

        else:
            # Long Exit
            if self.position.size > 0 and (self.data.close[0] >= self.position.price * self.params.take_profit_ratio or self.redkay.smooth_line < self.redkay.signal_line):
                self.close()

            # Short Exit
            elif self.position.size < 0 and (self.data.close[0] <= self.position.price * (1 - self.params.take_profit_ratio) or self.redkay.smooth_line > self.redkay.signal_line):
                self.close()


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

    # Check for nan values and handle them
    def handle_nan(value, default=0):
        return value if not math.isnan(value) else default
    
    # Collect results
    final_portfolio_value = handle_nan(cerebro.broker.getvalue())
    sharpe_ratio = handle_nan(strategy_instance.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0))
    drawdown = handle_nan(strategy_instance.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0))
    total_returns = handle_nan(strategy_instance.analyzers.returns.get_analysis().get('rtot', 0))
    total_trades = strategy_instance.trade_count
    
    conditions = json.dumps({
        'ema_period': strategy_instance.params.ema_period,
        'volume_threshold': strategy_instance.params.volume_threshold,
        'take_profit_ratio': strategy_instance.params.take_profit_ratio,
        'stop_loss': strategy_instance.params.stop_loss,
        'profit_to_loss_ratio': strategy_instance.params.profit_to_loss_ratio
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
        'take_profit': strategy_instance.params.profit_to_loss_ratio,
        'stoploss': strategy_instance.params.stop_loss,
        'interval': interval,
        'symbol': symbol
    })

    # Save to database
    db.insert_by_series('cryptocurrency.leaderboard', pd.Series(result_data))

    # Print results
    print("Final Portfolio Value: %.2f" % final_portfolio_value)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Drawdown:", drawdown)
    print("Total Returns:", total_returns)
    print(f'Total Trades: {total_trades}')


if __name__ == '__main__':
    db = Database()
    interval = '1d'
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

    chosen_strategy = RedKayEverlaxStrategy
    run_backtest(chosen_strategy, data_feed)
