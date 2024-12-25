import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, util, MarketOrder, Forex, Stock
import sys
import datetime
from scipy.stats import linregress


# ✅ Connect to IBKR
def connect_ibkr():
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
def fetch_historical_data(ib, contract):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 Y',
        barSizeSetting='4 hours',
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
# https://forums.eviews.com/viewtopic.php?t=1478
def calculate_mean_reversion_speed(data):
    """Estimate mean-reversion speed (theta) using AR(1) regression."""
    data['price_change'] = data['close'].diff()
    data['close_lagged'] = data['close'].shift(1)
    data.dropna(subset=['price_change', 'close_lagged'], inplace=True)
    
    if len(data) < 2:
        raise ValueError("❌ Not enough data points for regression.")
    
    if data['close_lagged'].std() == 0 or data['price_change'].std() == 0:
        raise ValueError("❌ Zero variance detected in lagged or price change columns.")
    
    # Use scipy's linregress for better numerical stability
    try:
        slope, intercept, _, _, _ = linregress(data['close_lagged'], data['price_change'])
    except Exception as e:
        raise ValueError("❌ Regression failed due to numerical instability.") from e
    
    if abs(slope) >= 1:
        raise ValueError("❌ Slope coefficient must be less than 1 for mean reversion.")
    
    half_life = -np.log(0.5) / np.log(abs(slope))
    theta = 1 / half_life
    
    print(f"✅ Estimated Mean-Reversion Speed (Theta): {theta:.4f}")
    return theta


# ✅ Ornstein-Uhlenbeck Process Strategy
def ornstein_uhlenbeck_strategy(data):
    """Apply the Ornstein-Uhlenbeck model to identify trading signals."""
    theta = calculate_mean_reversion_speed(data)
    mu = data['close'].mean()
    sigma = data['close'].std()

    data['OU_Process'] = 0.0
    for i in range(1, len(data)):
        dt = 1  # Time step
        prev_price = data['close'].iloc[i - 1]
        noise = np.random.normal(0, sigma * np.sqrt(dt))
        data['OU_Process'].iloc[i] = prev_price + theta * (mu - prev_price) * dt + noise

    # Generate trading signals
    data['OU_Signal'] = 0
    data.loc[data['close'] > data['OU_Process'], 'OU_Signal'] = -1  # Sell Signal
    data.loc[data['close'] < data['OU_Process'], 'OU_Signal'] = 1  # Buy Signal

    print(f"✅ Mean-Reversion Speed (Theta) used: {theta:.4f}")
    return data.dropna()


# ✅ Trade Execution
def execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position):
    """Execute trade based on the signal."""
    order = None
    ticker = contract.symbol
    profit = 0

    if signal == 1 and ticker not in buyOrders:
        order = MarketOrder('BUY', 1000)
        buyOrders[ticker] = 1
        open_position['price'] = ib.reqMktData(contract).last

    elif signal == -1 and ticker not in sellOrders:
        order = MarketOrder('SELL', 1000)
        sellOrders[ticker] = 1
        open_position['price'] = ib.reqMktData(contract).last

    elif signal == 0:
        if ticker in buyOrders:
            order = MarketOrder('SELL', 1000)
            profit = (ib.reqMktData(contract).last - open_position['price']) * 10000
            del buyOrders[ticker]
        elif ticker in sellOrders:
            order = MarketOrder('BUY', 1000)
            profit = (open_position['price'] - ib.reqMktData(contract).last) * 10000
            del sellOrders[ticker]

    if order:
        trade = ib.placeOrder(contract, order)
        print(f"✅ Trade executed: {trade}")
        if profit != 0:
            print(f"💰 Profit: {profit:.2f}")


# ✅ Main Function
def main():
    ib = connect_ibkr()
    contract = Stock('SPY', 'SMART', 'USD')  # Correct contract for SPY ETF
    buyOrders = {}
    sellOrders = {}
    open_position = {'price': 0}

    try:
        data = fetch_historical_data(ib, contract)
        data = ornstein_uhlenbeck_strategy(data)  # Apply OU Process

        for index, row in data.iterrows():
            signal = row['OU_Signal']
            execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position)
            ib.sleep(1)  # Prevent API overload

    except ValueError as e:
        print(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("🛑 Trading halted by user.")
    finally:
        ib.disconnect()
        print("🔌 Disconnected from IBKR")


# ✅ Entry Point
if __name__ == '__main__':
    main()
