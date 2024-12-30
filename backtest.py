import importlib.util
import pandas as pd
import numpy as np
from ib_insync import IB, util, MarketOrder, Forex, Stock
import sys
import datetime

# ✅ Main Function
def main():
    spec = importlib.util.spec_from_file_location("Ornstein_Uhlenbeck", "./Ornstein-Uhlenbeck.py")
    Ornstein_Uhlenbeck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Ornstein_Uhlenbeck)

    print("✅ Testing started")

    # Initialise parameters
    trading_bot = Ornstein_Uhlenbeck.TradingBot()
    ib = trading_bot.connect_ibkr()
    contract = Stock('SPY', 'SMART', 'USD')  # Correct contract for SPY ETF
    open_position = {'price': 0}

    data = trading_bot.fetch_historical_data(ib, contract, "4 hours")
    print(data.head(6))

    # Disconnect from IBKR
    ib.disconnect()
    print("🔌 Disconnected from IBKR")

    print("🛑 Testing ended")

# ✅ Entry Point
if __name__ == '__main__':
    main()