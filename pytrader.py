import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Technical Analysis library

# Load stock data
def get_stock_data(symbol, start, end):
    print(f"Fetching data for {symbol} from {start} to {end}")
    data = yf.download(symbol, start=start, end=end, auto_adjust=False)
    print(f"Data fetched: {data.shape[0]} rows")
    print(data.head())  # Print the first few rows of the data
    return data

# Compute indicators
def add_indicators(df):
    print("Adding indicators")
    print(df.head())  # Print the first few rows of the data before adding indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)

    # Ensure 'Close' is a 1D Series
    close_series = df['Close'].squeeze()
    rsi_indicator = ta.momentum.RSIIndicator(close_series, window=14)
    df['RSI'] = rsi_indicator.rsi().squeeze()  # Make sure the output is a 1D Series
    
    print("Indicators added")
    print(df[['SMA_50', 'SMA_200', 'RSI']].head(60))  # Print the first 60 rows to check for NaN values
    return df

# Simple trading strategy
def simple_strategy(df):
    print("Applying simple strategy")
    df['Signal'] = np.where((df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 30), 1, 0)  # Buy signal
    df['Signal'] = np.where((df['SMA_50'] < df['SMA_200']) & (df['RSI'] > 70), -1, df['Signal'])  # Sell signal
    print("Strategy applied")
    print(df[['SMA_50', 'SMA_200', 'RSI', 'Signal']].head(60))  # Print the first 60 rows to check signals
    return df

# Backtest strategy
def backtest(df, initial_balance=10000):
    print("Starting backtest")
    balance = initial_balance
    position = 0
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1:  # Buy
            position = balance / df['Close'].iloc[i]
            balance = 0
            print(f"Bought at {df['Close'].iloc[i]}, position: {position}")
        elif df['Signal'].iloc[i] == -1 and position > 0:  # Sell
            balance = position * df['Close'].iloc[i]
            position = 0
            print(f"Sold at {df['Close'].iloc[i]}, balance: {balance}")
    final_value = float(balance + (position * df['Close'].iloc[-1].iloc[0]))
    print(f"Backtest completed, final value: {final_value}")
    return final_value

# NEW FUNCTION: uses the AI you shared to provide a simple report/score
def get_ai_recommendation(symbol, final_value):
    from openai import OpenAI
    # Use the same API info you provided
    client = OpenAI(api_key="PASTE_KEY_HERE", base_url="https://api.deepseek.com")

    prompt_text = f"Please give a brief recommendation or report for {symbol}, based on the final backtest value: {final_value}."
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        stream=False
    )
    
    print("\n===== AI Recommendation =====")
    print(response.choices[0].message.content)
    print("=============================\n")

# Run the bot
if __name__ == "__main__":
    stock = input("Enter the stock symbol: ")
    df = get_stock_data(stock, "2023-01-01", "2024-01-01")
    df = add_indicators(df)
    df = simple_strategy(df)
    final_balance = backtest(df)
    print(f"Final balance after backtesting: ${final_balance:.2f}")
    
    # Call the new function that uses AI to provide a recommendation
    get_ai_recommendation(stock, final_balance)

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price', alpha=0.6)
    plt.plot(df['SMA_50'], label='50-day SMA', linestyle='dashed')
    plt.plot(df['SMA_200'], label='200-day SMA', linestyle='dotted')
    plt.legend()
    plt.title(f"Trading Strategy for {stock}")
    plt.show()