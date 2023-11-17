
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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

