import backtrader as bt
import pandas as pd
from database import Database
import json
import math

db = Database()

class BollingerVolumeStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('devfactor', 2.5),
        ('volume_multiplier', 1.5),
    )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
        self.order = None
        self.trade_count = 0  # Initialize trade count
        self.trade_log = []  # Initialize trade log

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

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
        if self.order:
            return

        if not self.position:
            if self.data.close[-1] > self.boll.lines.top[-1] and \
               self.data.volume > self.data.volume[-1] * self.p.volume_multiplier:
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.order = self.buy()
            elif self.data.close[-1] < self.boll.lines.bot[-1] and \
                 self.data.volume > self.data.volume[-1] * self.p.volume_multiplier:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.order = self.sell()
        else:
            if len(self) >= (self.bar_executed + 5):  # Exit after 5 bars
                self.log('CLOSE CREATE, %.2f' % self.data.close[0])
                self.order = self.close()

def run_backtest(strategy, data_feed, initial_cash=100000):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
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
    trade_log_df.to_excel('BollingerVolumeStrategy_Trades.xlsx', index=False)

    # Check for nan values and handle them
    def handle_nan(value, default=0):
        return value if not math.isnan(value) else default
    
    def sanitize_value(value, default=0.0):
        if math.isinf(value) or math.isnan(value):
            return default
        return value
    
    # Collect results
    final_portfolio_value = sanitize_value(cerebro.broker.getvalue(), default=100000)
    sharpe_ratio = sanitize_value(strategy_instance.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0), default=0)
    drawdown = sanitize_value(strategy_instance.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0), default=0)
    total_returns = sanitize_value(strategy_instance.analyzers.returns.get_analysis().get('rtot', 0), default=0)
    total_trades = strategy_instance.trade_count

    # Correctly access strategy parameters for conditions
    conditions = json.dumps({
        'period': strategy_instance.params.period,
        'devfactor': strategy_instance.params.devfactor,
        'volume_multiplier': strategy_instance.params.volume_multiplier,
    })

    # Prepare data for database insertion and print results
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
    }
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
    chosen_strategy = BollingerVolumeStrategy
    run_backtest(chosen_strategy, data_feed)
