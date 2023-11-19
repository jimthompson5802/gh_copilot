
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import math
from scipy.stats import norm

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
plt.savefig("./codex/capital_markets/images/stock_returns.png")
plt.show()


# Define the parameters
# get last closing price from the stock data
S = closing_prices.iloc[-1]
K = 185  # strike price
r = 0.05  # risk-free rate
days = 30
T = days / 365  # time to expiration in years
# compute volatility from the daily returns
sigma = daily_returns.std() * math.sqrt(252)

# print the paramters from the Black-Scholes formula
print(f"Last closing price: {S}")
print(f"Strike price: {K}")
print(f"Risk-free rate: {r}")
print(f"Days to expiration: {days}")
print(f"Time to expiration: {T}")
print(f"Volatility: {sigma}")

# Black-Scholes formula for call option
d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
call_option = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# Black-Scholes formula for put option
put_option = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print(f"The price of the {days}-day call option with a strike price of ${K} is: {call_option}")
print(f"The price of the {days}-day put option with a strike price of ${K} is: {put_option}")