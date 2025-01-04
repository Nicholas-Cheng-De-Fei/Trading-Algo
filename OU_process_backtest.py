import pandas as pd
import numpy as np
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.stats import linregress

# Fetch data
def fetch_spy_data(start_date, end_date):
    ticker = 'SPY'
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Handle MultiIndex Columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Ensure 'Date' column exists
    data.reset_index(inplace=True)
    if 'Datetime' in data.columns:  # Intraday data uses 'Datetime'
        data.rename(columns={'Datetime': 'Date'}, inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Keep only required columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data.dropna()

data = fetch_spy_data('2020-12-01', '2024-12-31')

print(data.head())
print(data.columns)
print(data.index)


# 🟢 Indicator: Ornstein-Uhlenbeck Mean Reversion
def ornstein_uhlenbeck_signal(close):
    """Generate a signal based on the Ornstein-Uhlenbeck model."""
    price_change = np.diff(close)
    close_lagged = close[:-1]  # Lagged close prices (same length as price_change)
    
    if len(price_change) < 2 or np.std(close_lagged) == 0:
        return np.zeros(len(close))  # Return zeros if insufficient data

    slope, _, _, _, _ = linregress(close_lagged, price_change)
    if abs(slope) >= 1:
        return np.zeros(len(close))

    half_life = -np.log(0.5) / np.log(abs(slope))
    theta = 1 / half_life

    mu = np.mean(close)
    sigma = np.std(close)
    noise = np.random.normal(0, sigma * np.sqrt(1), len(close))

    ou_price = np.roll(close, 1) + theta * (mu - np.roll(close, 1)) + noise
    signal = np.where(close > ou_price, -1, np.where(close < ou_price, 1, 0))

    return signal



# 🟢 Strategy Class
class OUReversionStrategy(Strategy):
    def init(self):
        # Indicator
        self.ou_signal = self.I(ornstein_uhlenbeck_signal, self.data.Close)

    def next(self):
        # Mean-reversion signal
        if self.ou_signal[-1] == 1:
            self.buy()
        elif self.ou_signal[-1] == -1:
            self.sell()

# 🟢 Run Backtest
bt = Backtest(data, OUReversionStrategy,
              cash=10000, commission=0.02,
              exclusive_orders=True)

output = bt.run()
bt.plot()
print(output)
