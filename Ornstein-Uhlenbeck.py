import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import IB, util, MarketOrder, Forex, Stock
import sys
import datetime
from scipy.stats import linregress

class TradingBot:

    def __init__(self):
        # Key : Ticker, Value tuple of tuple with time of purchase, price, amount
        self.buyOrder = {}
        self.sellOrder = {}

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
    def fetch_historical_data(self, ib, contract):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='3 M',
            barSizeSetting='5 mins',
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
    def calculate_mean_reversion_speed(self, data):
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
    def ornstein_uhlenbeck_strategy(self, data):
        """Apply the Ornstein-Uhlenbeck model to identify trading signals."""
        length = len(data) - 2
        theta = self.calculate_mean_reversion_speed(data)
        mu = data['close'].mean()
        sigma = data['close'].std()

        # Get the signal from the most recent data point

        """
        fixed dt of 1 not true representation of time step of data freq,
        derive dt dynamically from the time diff btn price obs
        """
        dt = 1  # Time step
        prev_price = data['close'].iloc[length - 1]
        current_price = data['close'].iloc[length]

        '''
        Adds randomness to OU price, use historical volatility / deterministic methods for noise.
        '''
        noise = np.random.normal(0, sigma * np.sqrt(dt))

        OU_price = prev_price + theta * (mu - prev_price) * dt + noise


        """
        direct comparison btwn prices. introduce a threshold factor to avoid over-trading on small deviations
        """
        signal = 0
        if (current_price > OU_price):
            signal = -1 # Sell Signal
        elif (current_price < OU_price):
            signal = 1 # Buy Signal

        # insert risk management here

        print(f"✅ Mean-Reversion Speed (Theta) used: {theta:.4f}")
        return signal, OU_price


    # ✅ Trade Execution
    def execute_trade(self, signal, contract, ib, buyOrders, sellOrders, open_position):
        """Execute trade based on the signal."""
        order = None
        ticker = contract.symbol
        profit = 0

        if signal == 1 and ticker not in buyOrders:
            order = MarketOrder('BUY', 100)
            if (ticker not in buyOrders):
                buyOrders[ticker] = ((datetime.now(), ib.reqMktData(contract).last, 100),)
            else:
                buyOrders[ticker] += ((datetime.now(), ib.reqMktData(contract).last, 100),)
            open_position['price'] = ib.reqMktData(contract).last

        elif signal == -1 and ticker not in sellOrders:
            order = MarketOrder('SELL', 100)
            if (ticker not in buyOrders):
                sellOrders[ticker] = ((datetime.now(), ib.reqMktData(contract).last, 100),)
            else:
                sellOrders[ticker] += ((datetime.now(), ib.reqMktData(contract).last, 100),)
            open_position['price'] = ib.reqMktData(contract).last

        if order:
            trade = ib.placeOrder(contract, order)
            print(f"✅ Trade executed: {trade}")
            if profit != 0:
                print(f"💰 Profit: {profit:.2f}")

    def exitBuyOrders(self, OU_price, MoS=0.10, SL=0.15, time_limit=360):
        current_time = datetime.now()
        orders = self.buyOrders["SPY"]
        for order in orders:
            time_elapsed = (current_time - order[0]).total_seconds()
            if time_elapsed >= time_limit:
                # ✅ Take-Profit Logic
                # # 10% before it hits OU price
                if order[1] >= OU_price * (1 - MoS):
                    pass

                # ❌ Stop-Loss Logic
                # if price falls 15% from cost price

# ✅ Main Function
def main():
    tradingBot = TradingBot()
    ib = tradingBot.connect_ibkr()
    contract = Stock('SPY', 'SMART', 'USD')  # Correct contract for SPY ETF
    open_position = {'price': 0}

    try:
        data = tradingBot.fetch_historical_data(ib, contract)
        signal, OU_price = tradingBot.ornstein_uhlenbeck_strategy(data)  # Apply OU Process and get the signal
        tradingBot.execute_trade(signal, contract, ib, open_position)
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
