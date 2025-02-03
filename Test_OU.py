import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, util, MarketOrder, Forex, Stock
import sys
import datetime
from scipy.stats import linregress

# Trading bot class
class TradingBot:

    def __init__(self):
        self.buyOrders = {}
        self.sellOrders = {}

    # ✅ Connect to IBKR
    def connect_ibkr(self):
        ib = IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)
            print("✅ Connected to IBKR")
        except TimeoutError:
            print("❌ Connection timed out.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        return ib

    # ✅ Fetch Historical Data
    def fetch_historical_data(self, ib, contract, interval):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='3 M',
            barSizeSetting=interval,
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=1
        )
        data = util.df(bars)
        data.set_index('date', inplace=True)
        
        if data.empty:
            raise ValueError("❌ No historical data retrieved. Check IBKR API connection or contract details.")
        
        return data.dropna()

    # ✅ Calculate Mean-Reversion Speed (Theta)
    def calculate_mean_reversion_speed(self, data):
        """Estimate mean-reversion speed (theta) using AR(1) regression."""
        data['price_change'] = data['close'].diff()
        data['close_lagged'] = data['close'].shift(1)
        data.dropna(subset=['price_change', 'close_lagged'], inplace=True)
        
        if len(data) < 2:
            raise ValueError("❌ Not enough data points for regression.")
        
        if data['close_lagged'].std() == 0 or data['price_change'].std() == 0:
            raise ValueError("❌ Zero variance detected in lagged or price change columns.")
        
        try:
            slope, _, _, _, _ = linregress(data['close_lagged'], data['price_change'])
        except Exception as e:
            raise ValueError("❌ Regression failed due to numerical instability.") from e
        
        if abs(slope) >= 1:
            raise ValueError("❌ Slope coefficient must be less than 1 for mean reversion.")
        
        half_life = -np.log(0.5) / np.log(abs(slope))
        theta = 1 / half_life
        
        print(f"✅ Estimated Mean-Reversion Speed (Theta): {theta:.4f}")
        return theta

    # ✅ Exponentially Weighted Moving Average (EWMA)
    def calculate_ewma(self, data, span=30):
        """Compute Exponentially Weighted Moving Average."""
        return data['close'].ewm(span=span, adjust=False).mean()

    # ✅ Volatility Calculation
    def calculate_volatility(self, data, window=30):
        """Compute rolling standard deviation for volatility bands."""
        return data['close'].rolling(window=window).std()

    # ✅ Ornstein-Uhlenbeck Process Strategy with EWMA & Volatility
    def ornstein_uhlenbeck_strategy(self, data):
        """Apply the Ornstein-Uhlenbeck model with EWMA & volatility-based cloud."""
        length = len(data) - 2
        theta = self.calculate_mean_reversion_speed(data)
        
        # Compute EWMA & Volatility Bands
        ewma = self.calculate_ewma(data)
        volatility = self.calculate_volatility(data)
        
        # Set
