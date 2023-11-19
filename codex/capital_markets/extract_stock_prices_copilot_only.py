# code generated using only GH Copilot

from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# extract stock prices from yahoo finance for AAPL from
# 1/1/2023 to 11/18/2023

# Define the start and end dates
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 11, 18)

# Specify the company's ticker symbol
ticker_symbol = "AAPL"  # Replace with the desired company's ticker symbol

# Fetch the stock data for the specified company and period
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Extract the closing prices
closing_prices = stock_data["Close"]


# Print or process the extracted stock prices as needed
print(closing_prices)

# compute the daily returns
daily_returns = closing_prices.pct_change()

# compute the cumulative returns
cumulative_returns = (1 + daily_returns).cumprod()

# combine the closing prices, daily returns, and cumulative returns into a single DataFrame
combined_df = pd.concat([closing_prices, daily_returns, cumulative_returns], axis=1)

# rename the columns
combined_df.columns = ["Closing Price", "Daily Return", "Cumulative Return"]

# print the combined DataFrame
print(combined_df)

# plot the daily and cumulative returns and save as a PNG image
combined_df.plot(y=["Daily Return", "Cumulative Return"], kind="line", title=ticker_symbol)
plt.savefig("./codex/capital_markets/images/stock_returns2.png")
plt.show()


# cmpute call and put options using Black-Scholes formula
# for last closing price of AAPL from 11/18/2023 ith strike price of 185
# compute volativity from daily returns

# Define the parameters
# get last closing price from the stock data
S = closing_prices.iloc[-1]
K = 185
T = 1
r = 0.02
sigma = daily_returns.std() * np.sqrt(252)

# these functions created by GH Copilot by typing start of function definition
def black_scholes_call(S, K, T, r, sigma):
    """Computes the Black-Scholes call option price for a European option."""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Computes the Black-Scholes put option price for a European option."""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Print the paramters from the Black-Scholes formula
print(f"Last closing price: {S}")
print(f"Strike price: {K}")
print(f"Risk-free rate: {r}")
print(f"Time to expiration: {T}")
print(f"Volatility: {sigma}")


# Compute the call and put option prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Print the call and put option prices
print(f"The call option price is ${call_price:,.2f}")
print(f"The put option price is ${put_price:,.2f}")

