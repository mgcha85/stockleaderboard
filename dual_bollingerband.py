import backtrader as bt
import pandas as pd
from database import Database
import json
import math

db = Database()


class DualBollingerBandStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('devfactor_2sigma', 2.5),
        ('devfactor_07sigma', 0.7),
        ('stop_loss', 0.1),
        ('profit_to_loss_ratio', 1),
    )

    def __init__(self):
        self.trade_count = 0  # Initialize trade count

        # Bollinger Bands with 2-sigma
        self.bollinger_2sigma = bt.indicators.BollingerBands(
            self.datas[0], 
            period=self.p.period, 
            devfactor=self.p.devfactor_2sigma
        )

        # Bollinger Bands with 0.7-sigma
        self.bollinger_07sigma = bt.indicators.BollingerBands(
            self.datas[0], 
            period=self.p.period, 
            devfactor=self.p.devfactor_07sigma
        )
        self.trade_log = []  # Initialize trade log
        self.order = None

    def log(self, text, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {text}')

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

    def next(self):
        # Cancel existing orders
        if self.order:
            self.cancel(self.order)

        # Check if we are already in the market
        if not self.position:
            # Buy condition - price is below the lower band of the 2-sigma Bollinger Band
            if self.data.close[0] < self.bollinger_2sigma.lines.bot:
                self.order = self.buy()
        else:
            # Sell condition - price is above the upper band of the 0.7-sigma Bollinger Band
            if self.data.close[0] > self.bollinger_07sigma.lines.top:
                self.order = self.sell()

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

    # Convert trade log to DataFrame and save to Excel
    strategy_instance = results[0]
    trade_log_df = pd.DataFrame(strategy_instance.trade_log)
    trade_log_df.to_excel('DualBollingerBandStrategy_Trades.xlsx', index=False)

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
        '2s_period': strategy_instance.bollinger_2sigma.params.period,
        '2s_sigma': strategy_instance.bollinger_2sigma.params.devfactor,
        '07s_period': strategy_instance.bollinger_2sigma.params.period,
        '07_sigma': strategy_instance.bollinger_07sigma.params.devfactor,
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

    chosen_strategy = DualBollingerBandStrategy
    run_backtest(chosen_strategy, data_feed)

