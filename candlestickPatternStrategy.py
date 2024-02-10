import backtrader as bt
import pandas as pd
from database import Database
import json
import math


# Define the strategy class
class CandlestickPatternStrategy(bt.Strategy):
    params = (
        ('stop_loss_percent', 0.02),  # Stop loss at 2% of the entry price
        ('take_profit_percent', 0.05),  # Take profit at 5% of the entry price
    )

    def __init__(self):
        # To identify the patterns, we'll need the open, high, low, and close prices
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low
        self.close = self.datas[0].close
        self.trade_count = 0  # Initialize trade count
        self.trade_log = []  # Initialize trade log

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
                self.trade_count += 1  # Increment trade count on buy
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            # Extracting trade details
            entry_date = bt.num2date(trade.dtopen).date()
            exit_date = bt.num2date(trade.dtclose).date()
            entry_price = trade.price
            quantity = trade.size
            exit_price = trade.price + (trade.pnl / trade.size if trade.size != 0 else 0)
            entry_value = entry_price * quantity
            sell_reason = 'stoploss' if trade.pnl < 0 else 'takeprofit'
            abs_profit = trade.pnl
            pct_profit = (exit_price - entry_price) / entry_price if entry_price != 0 else 0

            self.trade_log.append({
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Quantity': quantity,
                'Entry Value': entry_value,
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'Sell Reason': sell_reason,
                'Abs Profit': abs_profit,
                'Pct Profit': pct_profit
            })

    def next(self):
        # Check for bullish engulfing pattern
        if self.close[-1] < self.open[-1] and self.close[0] > self.open[0] and \
           self.close[0] > self.open[-1] and self.open[0] < self.close[-1]:
            self.log('Bullish engulfing pattern found')
            # Enter long position
            self.buy()

        # Check for bearish engulfing pattern
        elif self.close[-1] > self.open[-1] and self.close[0] < self.open[0] and \
             self.close[0] < self.open[-1] and self.open[0] > self.close[-1]:
            self.log('Bearish engulfing pattern found')
            # Enter short position
            self.sell()

        # Check for momentum candle
        if (self.close[0] - self.open[0]) > 2 * (self.close[-1] - self.open[-1]):
            self.log('Momentum candle found')
            # Enter in the direction of momentum
            self.buy() if self.close[0] > self.open[0] else self.sell()

        # Check for hammer pattern
        if self.close[0] > self.open[0] and (self.low[0] - self.open[0]) > 2 * (self.close[0] - self.open[0]):
            self.log('Hammer pattern found')
            # Enter long position
            self.buy()

        # Check for shooting star pattern
        if self.close[0] < self.open[0] and (self.high[0] - self.close[0]) > 2 * (self.open[0] - self.close[0]):
            self.log('Shooting star pattern found')
            # Enter short position
            self.sell()


def run_backtest(strategy, data_feed, initial_cash=100000):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(initial_cash)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.broker.setcommission(commission=0.001)

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    strategy_instance = results[0]

    # Convert trade log to DataFrame and save to Excel
    trade_log_df = pd.DataFrame(strategy_instance.trade_log)
    trade_log_df.to_excel('CandlestickPatternStrategy_Trades.xlsx', index=False)

    # Handle NaN and Inf values
    def handle_nan(value, default=0):
        if math.isnan(value) or math.isinf(value):
            return default
        return value

    final_portfolio_value = handle_nan(cerebro.broker.getvalue(), 100000)
    sharpe_ratio = handle_nan(strategy_instance.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0), 0)
    drawdown = handle_nan(strategy_instance.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0), 0)
    total_returns = handle_nan(strategy_instance.analyzers.returns.get_analysis().get('rtot', 0), 0)
    total_trades = strategy_instance.trade_count

    # There are no 'boll' parameters in this strategy, so conditions should reflect the actual strategy parameters
    conditions = json.dumps({
        'stop_loss_percent': strategy_instance.params.stop_loss_percent,
        'take_profit_percent': strategy_instance.params.take_profit_percent,
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
    })

    # Save to database
    db.insert_by_series('cryptocurrency.leaderboard', result_data)

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
    chosen_strategy = CandlestickPatternStrategy
    run_backtest(chosen_strategy, data_feed)
