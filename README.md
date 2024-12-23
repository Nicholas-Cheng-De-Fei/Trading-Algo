# 📊 **Mean Reversion Trading Strategy using Ornstein-Uhlenbeck Process**  

A quantitative trading strategy utilizing the **Ornstein-Uhlenbeck process** to identify mean-reverting opportunities between **GLD (Gold ETF)** and **GDX (Gold Miners ETF)**, optimizing trade execution for profitable arbitrage.

---

## 📝 **Description**

This project models the price relationship between **GLD** and **GDX** using the **Ornstein-Uhlenbeck stochastic differential equation** to exploit statistical arbitrage opportunities. The mean-reverting properties of the spread between these two ETFs are used to generate trading signals for buying or selling.

### **Key Highlights:**
- Statistical modeling of ETF spreads.
- Signal generation based on mean-reverting properties.
- Backtesting framework to evaluate performance.
- Integration with live trading platforms (Interactive Brokers API).

---

## ⚙️ **Mathematical Framework**

The relationship between **GLD** and **GDX** is modeled using the **Ornstein-Uhlenbeck (OU) Process**:

\[
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
\]

Where:
- \( X_t \): Spread between GLD and GDX prices.
- \( \theta \): Speed of mean reversion.
- \( \mu \): Long-term mean of the spread.
- \( \sigma \): Volatility of the spread.
- \( dW_t \): Brownian motion term.

---

## 🚀 **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/mean-reversion-GLD-GDX.git
cd mean-reversion-GLD-GDX
