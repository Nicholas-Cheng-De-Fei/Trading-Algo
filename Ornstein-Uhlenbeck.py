import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, util, MarketOrder, Forex
import sys
import datetime

# Connect to IBKR
def connect_ibkr():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)
        print("Connected to IBKR")
    except TimeoutError:
        print("Connection timed out.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    return ib

# Fetch Historical Data
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
    return data.dropna()

# Ornstein-Uhlenbeck Process Strategy
def ornstein_uhlenbeck_strategy(data):
    """Apply the Ornstein-Uhlenbeck model to identify trading signals."""
    # Parameters for OU process
    theta = 0.15  # Mean-reversion speed
    mu = data['close'].mean()  # Long-term mean
    sigma = data['close'].std()  # Volatility

    # OU Simulation
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

    return data.dropna()

# Trade Execution
def execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position):
    """Execute trade based on the signal."""
    order = None
    ticker = contract.symbol
    action = None
    profit = 0

    if signal == 1 and ticker not in buyOrders:
        order = MarketOrder('BUY', 10000)
        buyOrders[ticker] = 1
        open_position['price'] = ib.reqMktData(contract).last

    elif signal == -1 and ticker not in sellOrders:
        order = MarketOrder('SELL', 10000)
        sellOrders[ticker] = 1
        open_position['price'] = ib.reqMktData(contract).last

    elif signal == 0:
        if ticker in buyOrders:
            order = MarketOrder('SELL', 10000)
            profit = (ib.reqMktData(contract).last - open_position['price']) * 10000
            del buyOrders[ticker]
        elif ticker in sellOrders:
            order = MarketOrder('BUY', 10000)
            profit = (open_position['price'] - ib.reqMktData(contract).last) * 10000
            del sellOrders[ticker]

    if order:
        trade = ib.placeOrder(contract, order)
        print(f"Trade executed: {trade}")
        if profit != 0:
            print(f"Profit: {profit:.2f}")

# Main Function
def main():
    ib = connect_ibkr()
    contract = Forex('NZDUSD')
    buyOrders = {}
    sellOrders = {}
    open_position = {'price': 0}

    data = fetch_historical_data(ib, contract)
    data = ornstein_uhlenbeck_strategy(data)  # Apply OU Process

    for index, row in data.iterrows():
        signal = row['OU_Signal']
        execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position)
        ib.sleep(1)  # Prevent API overload

if __name__ == '__main__':
    main()
