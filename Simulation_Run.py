"""
if unable to install backtesting package do this instead:
pip install git+https://github.com/kernc/backtesting.py.git
"""

"""
Users should run ADF test first to observe whether the instrument does indeed exhibit
mean-reverting behaviour.
"""

import pandas as pd
import numpy as np
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.stats import linregress

# for simulation
import os

ticker = 'NZDUSD=X' # Configure ticker

# 1. Fetch data
def fetch_spy_data(start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1wk') # Configure candlestick interval
    
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

data = fetch_spy_data('2010-01-01', '2024-12-31') # Configure data collection period

# print(data.head())
# print(data.columns)
# print(data.index)

# 2. Calculate Theta
def calculate_mean_reversion_speed(close_prices):

    if len(close_prices) < 2:
        raise ValueError("❌ Insufficient data for mean-reversion speed calculation.")

    # Calculate price change and lagged close prices
    price_change = pd.Series(close_prices).diff().dropna()
    close_lagged = pd.Series(close_prices).shift(1).dropna()

    # Ensure the same length for regression
    regression_data = pd.DataFrame({'price_change': price_change, 'close_lagged': close_lagged}).dropna()
    if len(regression_data) < 2:
        raise ValueError("❌ Not enough valid data points for regression.")

    try:
        # Perform linear regression
        slope, _, _, _, _ = linregress(regression_data['close_lagged'], regression_data['price_change'])
        
        if abs(slope) >= 1:
            raise ValueError("❌ Invalid regression slope. Slope must be less than 1.")

        # Calculate half-life and theta
        half_life = -np.log(0.5) / np.log(abs(slope))
        theta = 1 / half_life
        # print(f"📊 Mean-Reversion Speed (Theta): {theta:.4f}")
        return theta
    except Exception as e:
        raise ValueError(f"❌ Error in regression calculation: {e}")
    
def find_jumps(data):
    data["Log_Return"] = np.log(data['Close'] / data["Close"].shift(1))
    data.dropna(inplace=True)

    log_mean = data["Log_Return"].mean()
    log_std = data["Log_Return"].std()
    jump_threshold = log_mean + 2 * log_std # 2 stds from the mean

    jumps = data[data['Log_Return'].abs() > jump_threshold]['Log_Return']

    return jumps

# def log_likelihood(mu, sigma, data):
#     if sigma <= 0:
#         return np.inf
#     return -np.sum(norm.logpdf(np.log(data), loc=mu, scale=sigma))


# 3. Generate Ornstein-Uhlenbeck price
def generate_ou_price(close):

    if len(close) < 2:
        raise ValueError("❌ Insufficient data for generating OU price.")

    mu = np.mean(close)
    sigma = np.std(close)
    noise = np.random.normal(0, sigma * np.sqrt(1), len(close))
    theta = calculate_mean_reversion_speed(close)

    # jumps = find_jumps(data) # To account for BS

    jump_occurrence = np.random.poisson(0.1, len(close))  # Adjust λ (jump frequency)
    jump_size = np.random.lognormal(0, sigma, len(close)) * jump_occurrence
    
    ou_price = np.roll(close, 1) + theta * (mu - np.roll(close, 1)) + noise + jump_size
    return ou_price

# 4. Generate trading signal
def ornstein_uhlenbeck_signal(close):
    """
    Generate a signal based on the Ornstein-Uhlenbeck model.

    Parameters:
        close (np.ndarray): Array of close prices.

    Returns:
        np.ndarray: Signal array (-1 for SELL, 1 for BUY, 0 for HOLD).
    """
    if len(close) < 2:
        return np.zeros(len(close))  # Insufficient data

    theta = calculate_mean_reversion_speed(close)
    if theta is None:
        return np.zeros(len(close))  # Invalid or insufficient data

    ou_price = generate_ou_price(close)
    signal = np.where(close > ou_price, -1, np.where(close < ou_price, 1, 0))
    # print(f'SIGNAL STRUCTURE: {signal}')
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
              cash=10000, commission=0.0000, # set comms to zero to ensure that overcomms doesnt affect performance
              exclusive_orders=True)



if __name__ == '__main__':
    print("🚀 Starting Ornstein-Uhlenbeck Backtest...")
    initial_equity = 10000
    results = []

    for i in range(1000):
        output = bt.run()

        # Extract specific values from the _Stats object
        equity_final = output['Equity Final [$]']
        equity_peak = output['Equity Peak [$]']
        max_drawdown = output['Max. Drawdown [%]']
        max_drawdown_duration = output['Max. Drawdown Duration']

        # Calculate percentages
        final_equity_pct = (equity_final / initial_equity) * 100
        peak_equity_pct = (equity_peak / initial_equity) * 100


        final_equity_pct_formatted = round(final_equity_pct, 3)  
        peak_equity_pct_formatted = round(peak_equity_pct, 3)

        results.append({
            "Peak Equity [%]": peak_equity_pct_formatted,
            "Final Equity [%]": final_equity_pct_formatted,
            "Max Drawdown [%]": max_drawdown,
            "Max Drawdown Duration": max_drawdown_duration
        })

    df = pd.DataFrame(results)
    output_file = "backtest_results.xlsx"
    df.to_excel(output_file)