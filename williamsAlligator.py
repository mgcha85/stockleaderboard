import backtrader as bt
import pandas as pd
from database import Database
import json
import math
from datetime import datetime, timedelta

db = Database()


class AlligatorIndicator(bt.Indicator):
    lines = ('jaw', 'teeth', 'lips',)
    params = (('jaw_period', 13), ('teeth_period', 8), ('lips_period', 5),)

    def __init__(self):
        self.l.jaw = bt.indicators.SimpleMovingAverage(self.data, period=self.p.jaw_period)
        self.l.teeth = bt.indicators.SimpleMovingAverage(self.data, period=self.p.teeth_period)
        self.l.lips = bt.indicators.SimpleMovingAverage(self.data, period=self.p.lips_period)

# Define the strategy
class AlligatorStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.1),  # Define the stop loss percentage
        ('profit_to_loss_ratio', 3),  # Target profit-to-loss ratio
    )

    def __init__(self):
        self.alligator = AlligatorIndicator(self.datas[0])
        self.order = None
        self.entry_price = None
        self.stop_loss_price = None
        self.trade_count = 0  # Initialize trade count
        self.trade_log = []

    def log(self, text, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {text}')
        # Add additional logs for debugging
        # print(f'Close: {self.datas[0].close[0]}, EMA: {self.ema[0]}, Squeeze Off: {self.squeeze_off[0]}, Momentum: {self.momentum[0]}')

    def log_trade(self, entry_date, entry_price, quantity, entry_value, exit_date, exit_price, sell_reason, pnl):
        entry_datetime = bt.num2date(entry_date)
        exit_datetime = bt.num2date(exit_date)
        
        self.trade_log.append({
            'Entry Date': entry_datetime,
            'Entry Price': entry_price,
            'Quantity': quantity,
            'Entry Value': entry_value,
            'Exit Date': exit_datetime,
            'Exit Price': exit_price,
            'Sell Reason': sell_reason,
            'Abs Profit': pnl,
            'Pct Profit': pnl / entry_value if entry_value else 0
        })


    def notify_trade(self, trade):
        if trade.isclosed:
            entry_date = bt.num2date(trade.dtopen).date()
            exit_date = bt.num2date(trade.dtclose).date()
            entry_price = trade.price
            exit_price = trade.pnl / trade.size + trade.price if trade.size != 0 else 0
            quantity = trade.size
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
        # Cancel existing orders
        if self.order:
            self.cancel(self.order)

        # Long Position Entry Logic
        if self.datas[0].close[0] > self.alligator.teeth[0] and self.alligator.lips[0] < self.alligator.teeth[0] and self.alligator.teeth[0] < self.alligator.jaw[0]:
            self.close_short_positions()
            self.order = self.buy()
            self.entry_price = self.datas[0].close[0]
            self.stop_loss_price = self.alligator.lips[0]

        # Short Position Entry Logic
        elif self.datas[0].close[0] < self.alligator.teeth[0] and self.alligator.lips[0] > self.alligator.teeth[0] and self.alligator.teeth[0] > self.alligator.jaw[0]:
            self.close_long_positions()
            self.order = self.sell()
            self.entry_price = self.datas[0].close[0]
            self.stop_loss_price = self.alligator.lips[0]

        # Exit conditions
        self.check_exit_conditions()

    def close_short_positions(self):
        if self.position.size < 0:
            self.order = self.close()

    def close_long_positions(self):
        if self.position.size > 0:
            self.order = self.close()

    def check_exit_conditions(self):
        if self.position.size > 0:  # For long positions
            # Calculate the stop loss and take profit prices
            stop_loss_price = self.entry_price * (1 - self.params.stop_loss)
            take_profit_price = self.entry_price * (1 + self.params.profit_to_loss_ratio * self.params.stop_loss)

            # Exit if stop loss or take profit conditions are met
            if self.data.close[0] <= stop_loss_price or self.data.close[0] >= take_profit_price:
                self.order = self.close()  # Close the long position

        elif self.position.size < 0:  # For short positions
            # Calculate the stop loss and take profit prices
            stop_loss_price = self.entry_price * (1 + self.params.stop_loss)
            take_profit_price = self.entry_price * (1 - self.params.profit_to_loss_ratio * self.params.stop_loss)

            # Exit if stop loss or take profit conditions are met
            if self.data.close[0] >= stop_loss_price or self.data.close[0] <= take_profit_price:
                self.order = self.close()  # Close the short position

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

    # Extract the trade log from the strategy instance
    trade_log_df = pd.DataFrame(strategy_instance.trade_log)
    trade_log_df.to_excel('AlligatorStrategy.xlsx', index=False, engine='xlsxwriter')

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
        'jaw_period': strategy_instance.alligator.params.jaw_period,
        'teeth_period': strategy_instance.alligator.params.teeth_period,
        'lips_period': strategy_instance.alligator.params.lips_period,
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

    chosen_strategy = AlligatorStrategy
    run_backtest(chosen_strategy, data_feed)
