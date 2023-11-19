# code generated with the use of GH Copilot and GH Copilot Chat

import pandas as pd
import yfinance as yf
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm
import math

# Define the ticker symbol
tickerSymbol = 'AAPL'

# Get the data
tickerData = yf.download(tickerSymbol, start='2023-01-01', end='2023-11-18')

# Print the closing prices
print("Closing Prices:")
print(tickerData['Close'])

# Compute daily returns
daily_returns = tickerData['Close'].pct_change()

# Compute cumulative returns
cumulative_returns = (1 + daily_returns).cumprod()

# Combine into a single dataframe
combined_df = pd.DataFrame({
    'Closing Prices': tickerData['Close'],
    'Daily Returns': daily_returns,
    'Cumulative Returns': cumulative_returns
})

# Print the combined dataframe
print("\nCombined Dataframe:")
print(combined_df)

# Plot the daily and cumulative returns
combined_df.plot(y=["Daily Returns", "Cumulative Returns"], kind="line", title=tickerSymbol)

# Save the plot as a PNG image
plt.savefig("./codex/capital_markets/images/stock_returns.png")

# Display the plot
plt.show()



# Get the last closing price
S = combined_df['Closing Prices'].iloc[-1]

# Compute the volatility
volatility = combined_df['Daily Returns'].std()

# Define the Black-Scholes formula for call and put options
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) - S * norm.cdf(-d1) + S * norm.cdf(-d2)

# Compute the call and put options
K = 185  # strike price
T = 1  # one year
r = 0.02  # risk-free rate

# print parameters for black-scholes formula
print(f"Stock Price: {S}")
print(f"Strike Price: {K}")
print(f"Time to Maturity: {T}")
print(f"Risk-Free Rate: {r}")
print(f"Volatility: {volatility}")


call_option = black_scholes_call(S, K, T, r, volatility)
put_option = black_scholes_put(S, K, T, r, volatility)

print(f"Call Option: {call_option}")
print(f"Put Option: {put_option}")