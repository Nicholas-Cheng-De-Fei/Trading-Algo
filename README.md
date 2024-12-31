# 📊 **Mean Reversion Trading Strategy using Ornstein-Uhlenbeck Process**  

A quantitative trading strategy utilizing the **Ornstein-Uhlenbeck process** to identify mean-reverting opportunities between **S&P500 (SPY)**, optimizing trade execution for profitable arbitrage when the market falls out of conventional sense.

---

## 📝 **Description**

This projects attempts to find market arbitrage especially when the market enters into states of disorientations using the **Ornstein-Uhlenbeck (OU) stochastic differential equation**. The OU price generated by the equation creates an artificial price target to exit when the stock approaches the OU price.

### **Key Highlights:**
- Statistical modeling of ETF/Currencies.
- Signal generation based on mean-reverting properties.
- Backtesting framework to evaluate performance.
- Integration with live trading platforms (Interactive Brokers API).

---

## ⚙️ **Mathematical Framework**

The relationship between securities' prices and its conventional market sentiment is modeled using the **Ornstein-Uhlenbeck (OU) Process**:

$$
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
$$

Where:
- $X_t$: Current price of security.
- $\theta$: Speed of mean reversion where $\theta$ = ln(0.5)/ln(b)
- $\mu$: Long-term mean of the spread.
- $\sigma$: Volatility of the spread.
- $dW_t$: Brownian motion term.



---
