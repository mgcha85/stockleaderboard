import backtrader as bt
import pandas as pd
from database import Database
import json
import math

db = Database()

# Assuming the existence of PurpleCloudIndicator and EMAHistogram
# These should be implemented according to their specific algorithms
class PurpleCloudIndicator(bt.Indicator):
    lines = ('signal',)

    def __init__(self):
        # Example logic for generating signals
        # 1 for buy, -1 for sell, 0 for no signal
        self.lines.signal = bt.If(self.data.close > self.data.close(-1), 1, bt.If(self.data.close < self.data.close(-1), -1, 0))

class EMAHistogram(bt.Indicator):
    lines = ('color',)

    def __init__(self, ema_period=50):
        ema = bt.indicators.EMA(self.data, period=ema_period)
        # 1 for bullish (green), -1 for bearish (red)
        self.lines.color = bt.If(self.data.close > ema, 1, -1)

class PurpleCloudStrategy(bt.Strategy):
    params = (
        ('ema_period', 60),
        ('stop_loss', 0.1),
        ('profit_to_loss_ratio', 3),
        ('risk_reward_ratio', 2),
    )

    def __init__(self):
        self.ema200 = bt.indicators.EMA(self.data, period=self.params.ema_period)
        self.purple_cloud = PurpleCloudIndicator(self.data)
        self.ema_histogram = EMAHistogram(self.data)
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
        # Long Entry
        if not self.position and self.purple_cloud.lines.signal[0] == 1 and self.ema_histogram.lines.color[0] == 1:
            self.buy()

        # Long Exit
        elif self.position.size > 0 and self.purple_cloud.lines.signal[0] == -1:
            self.close()

        # Short Entry
        elif not self.position and self.purple_cloud.lines.signal[0] == -1 and self.ema_histogram.lines.color[0] == -1:
            self.sell()

        # Short Exit
        elif self.position.size < 0 and self.purple_cloud.lines.signal[0] == 1:
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
    interval = '15T'
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

    chosen_strategy = PurpleCloudStrategy
    run_backtest(chosen_strategy, data_feed)
