import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, util, MarketOrder, Forex
import sys
import datetime

def connect_ibkr():
    """Connect to IBKR."""
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)
        print("Connected to IBKR")
    except TimeoutError:
        print("Connection to IBKR timed out. Please check TWS/Gateway settings and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    return ib

def fetch_historical_data(ib, contract):
    """Fetch historical data from IBKR."""
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

def identify_support_resistance(data, window=20):
    """Identify support and resistance levels."""
    data['Support'] = data['low'].rolling(window=window).min()
    data['Resistance'] = data['high'].rolling(window=window).max()
    return data

def ichimoku_strategy(data):
    """Apply the Ichimoku strategy with flipped signals."""
    high_9 = data['high'].rolling(window=9).max()
    low_9 = data['low'].rolling(window=9).min()
    data['Tenkan-sen'] = (high_9 + low_9) / 2

    high_26 = data['high'].rolling(window=26).max()
    low_26 = data['low'].rolling(window=26).min()
    data['Kijun-sen'] = (high_26 + low_26) / 2

    data['Senkou Span A'] = ((data['Tenkan-sen'] + data['Kijun-sen']) / 2).shift(26)

    high_52 = data['high'].rolling(window=52).max()
    low_52 = data['low'].rolling(window=52).min()
    data['Senkou Span B'] = ((high_52 + low_52) / 2).shift(26)

    data['Chikou Span'] = data['close'].shift(-26)

    # Generate Ichimoku-based trading signals with flipped logic
    data['Ichimoku_Signal'] = 0
    data.loc[(data['Tenkan-sen'] > data['Kijun-sen']) & (data['close'] > data['Senkou Span A']) & (data['close'] > data['Senkou Span B']), 'Ichimoku_Signal'] = -1
    data.loc[(data['Tenkan-sen'] < data['Kijun-sen']) & (data['close'] < data['Senkou Span A']) & (data['close'] < data['Senkou Span B']), 'Ichimoku_Signal'] = 1

    return data.dropna()

def price_action_confirmation(data, min_movement=0.0010, confirmation_period=3):
    """Apply price action confirmation to signals with additional filters."""
    data['Position'] = 0
    last_position = 0  # Track the last position to avoid instant reversals
    last_price = 0  # Track the price at which the last trade was made

    for i in range(confirmation_period, len(data)):
        # Check if the price has moved sufficiently from the last trade price
        price_movement = abs(data['close'].iloc[i] - last_price)
        
        if last_position == 0:  # No open position
            # Bullish price action confirmation
            if (data['Ichimoku_Signal'].iloc[i] == 1 and 
                data['close'].iloc[i] > data['Resistance'].iloc[i-confirmation_period] and
                price_movement >= min_movement):
                data.loc[data.index[i], 'Position'] = 1
                last_position = 1
                last_price = data['close'].iloc[i]
                
            # Bearish price action confirmation
            elif (data['Ichimoku_Signal'].iloc[i] == -1 and 
                  data['close'].iloc[i] < data['Support'].iloc[i-confirmation_period] and
                  price_movement >= min_movement):
                data.loc[data.index[i], 'Position'] = -1
                last_position = -1
                last_price = data['close'].iloc[i]

        # Check if we should exit or reverse the position
        elif last_position == 1:  # Currently in a buy position
            if (data['Ichimoku_Signal'].iloc[i] == -1 or 
                data['close'].iloc[i] < data['Tenkan-sen'].iloc[i] or
                price_movement >= min_movement):
                data.loc[data.index[i], 'Position'] = 0  # Close position
                last_position = 0

        elif last_position == -1:  # Currently in a sell position
            if (data['Ichimoku_Signal'].iloc[i] == 1 or 
                data['close'].iloc[i] > data['Tenkan-sen'].iloc[i] or
                price_movement >= min_movement):
                data.loc[data.index[i], 'Position'] = 0  # Close position
                last_position = 0

    return data

def execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position):
    """Execute trade based on the signal and track profit for closed positions."""
    order = None
    ticker = contract.symbol
    action = None
    profit = 0

    if signal == 1:  # Buy Signal
        if ticker not in buyOrders:
            action = 'BUY'
            order = MarketOrder('BUY', 10000)  # 10,000 units is a common size for Forex trades
            if ticker in sellOrders:
                del sellOrders[ticker]
            else:
                buyOrders[ticker] = 1
                open_position['price'] = ib.reqMktData(contract).last  # Store the open price

    elif signal == -1:  # Sell Signal
        if ticker not in sellOrders:
            action = 'SELL'
            order = MarketOrder('SELL', 10000)
            if ticker in buyOrders:
                del buyOrders[ticker]
            else:
                sellOrders[ticker] = 1
                open_position['price'] = ib.reqMktData(contract).last  # Store the open price

    elif signal == 0:  # Exit Signal
        if ticker in buyOrders:
            action = 'CLOSE BUY'
            order = MarketOrder('SELL', 10000)  # Adjust for closing positions
            closing_price = ib.reqMktData(contract).last
            profit = (closing_price - open_position['price']) * 10000  # Calculate profit for closing buy
            del buyOrders[ticker]

        elif ticker in sellOrders:
            action = 'CLOSE SELL'
            order = MarketOrder('BUY', 10000)  # Adjust for closing positions
            closing_price = ib.reqMktData(contract).last
            profit = (open_position['price'] - closing_price) * 10000  # Calculate profit for closing sell
            del sellOrders[ticker]

    if order:
        trade = ib.placeOrder(contract, order)
        if action:
            print(f"Placed order: {action}")
        if profit != 0:
            print(f"Profit from trade: {profit:.2f}")

def main():
    ib = connect_ibkr()
    contract = Forex('NZDUSD')  # Correct Forex contract for NZD/USD
    buyOrders = {}
    sellOrders = {}
    open_position = {'price': 0}
    data = fetch_historical_data(ib, contract)
    data = identify_support_resistance(data)
    data = ichimoku_strategy(data)
    data = price_action_confirmation(data, min_movement=0.0010, confirmation_period=3)

    for index, row in data.iterrows():
        signal = row['Position']
        execute_trade(signal, contract, ib, buyOrders, sellOrders, open_position)

        ib.sleep(1)  # Respect rate limits and avoid overloading IBKR API

if __name__ == '__main__':
    main()
