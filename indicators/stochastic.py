import pandas as pd
import numpy as np

def getStochastic(df, k_window = 14, d_window = 3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()

    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].ewm(span=d_window, adjust=False).mean()