import pandas as pd

# Read the CSV data
data = pd.read_csv('/workspaces/gh_copilot/codex/capital_markets/AAPL.csv')

# Calculate and print statistics for the 'Close' column
close_stats = data['Close'].describe()
print(close_stats)

import numpy as np
from scipy.stats import norm

# Assumed values
S = 100  # Stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate
T = 1  # Time to maturity
sigma = 0.2  # Volatility

# Calculate d1 and d2 parameters
d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# Calculate call and put prices
call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print(f"Call Price: {call_price}")
print(f"Put Price: {put_price}")