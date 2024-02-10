import backtrader as bt
import pandas as pd
from database import Database
import json
import math

db = Database()

# Define the Strategy
class StochRSIMACDStrategy(bt.Strategy):
    params = (
        ('stoch_overbought', 80), 
        ('stoch_oversold', 20), 
        ('rsi_period', 14), 
        ('macd_fast', 12), 
        ('macd_slow', 26), 
        ('macd_signal', 9), 
        ('risk_reward_ratio', 1.5),
        ('stop_loss', 0.02),
        ('profit_to_loss_ratio', 3),
    )

    def __init__(self):
        self.stochastic = bt.indicators.Stochastic(self.datas[0])
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
        self.stop_loss_price = None
        self.trade_count = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.trade_count += 1
                self.stop_loss_price = order.executed.price * (1 - self.params.stop_loss)
            elif order.issell():
                self.trade_count += 1
                self.stop_loss_price = order.executed.price * (1 + self.params.stop_loss)
            self.order = None

    def next(self):
        if not self.position:
            if self.stochastic.lines.percK[0] < self.params.stoch_oversold and self.rsi[0] > self.params.rsi_period and self.macd.lines.macd[0] > self.macd.lines.signal[0]:
                self.buy()
            elif self.stochastic.lines.percK[0] > self.params.stoch_overbought and self.rsi[0] < self.params.rsi_period and self.macd.lines.macd[0] < self.macd.lines.signal[0]:
                self.sell()

        else:
            if self.position.size > 0 and (self.datas[0].close[0] < self.stop_loss_price or self.macd.lines.macd[0] < self.macd.lines.signal[0]):
                self.close()
            elif self.position.size < 0 and (self.datas[0].close[0] > self.stop_loss_price or self.macd.lines.macd[0] > self.macd.lines.signal[0]):
                self.close()


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
        if value is None:  # Check if the value is None
            return default
        return value if not math.isnan(value) else default

    # Collect results
    final_portfolio_value = handle_nan(cerebro.broker.getvalue())
    sharpe_ratio = handle_nan(strategy_instance.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0))
    drawdown = handle_nan(strategy_instance.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0))
    total_returns = handle_nan(strategy_instance.analyzers.returns.get_analysis().get('rtot', 0))
    total_trades = strategy_instance.trade_count

    # Serialize strategy parameters
    conditions = json.dumps({
        'stoch_overbought': strategy_instance.params.stoch_overbought,
        'stoch_oversold': strategy_instance.params.stoch_oversold,
        'rsi_period': strategy_instance.params.rsi_period,
        'macd_fast': strategy_instance.params.macd_fast,
        'macd_slow': strategy_instance.params.macd_slow,
        'macd_signal': strategy_instance.params.macd_signal,
        'risk_reward_ratio': strategy_instance.params.risk_reward_ratio,
        'stop_loss': strategy_instance.params.stop_loss,
        'profit_to_loss_ratio': strategy_instance.params.profit_to_loss_ratio
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
        'stoploss': strategy_instance.params.stop_loss,
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

    chosen_strategy = StochRSIMACDStrategy
    run_backtest(chosen_strategy, data_feed, initial_cash=100000, interval=interval, symbol=symbol)
