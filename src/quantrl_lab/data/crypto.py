import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_crypto_data(
    symbol: str = "BTC-USD", 
    period: str = "2y", 
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetches historical crypto data from Yahoo Finance and formats it 
    specifically for the QuantRL-Lab environment.
    
    Args:
        symbol: The cryptocurrency ticker (e.g., 'BTC-USD', 'ETH-USD').
        period: Data period to download (e.g., '1mo', '1y', 'max').
        interval: Data interval (e.g., '1d', '1h', '15m').
        
    Returns:
        pd.DataFrame: OHLCV dataframe formatted with float32 for RL agents.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}. Check if the ticker is correct.")

    # Drop non-price columns that yfinance sometimes injects
    cols_to_drop = [col for col in ['Dividends', 'Stock Splits'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Standardize column names to lowercase (open, high, low, close, volume)
    df.columns = [col.lower() for col in df.columns]

    # Convert timezone-aware datetimes to timezone-naive to prevent Gymnasium environment crashes
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Cast to float32 for Neural Network memory efficiency
    return df.astype("float32")