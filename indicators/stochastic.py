import pandas as pd
import numpy as np

def getStochastic(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window, min_periods=1).min()
    high_max = df['High'].rolling(window=k_window, min_periods=1).max()

    stoch_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.ewm(span=d_window, adjust=False).mean()

    stoch_k_arr = stoch_k.values.flatten()
    stoch_d_arr = stoch_d.values.flatten()

    return pd.DataFrame({
        'Stoch_K': stoch_k_arr,
        'Stoch_D': stoch_d_arr
    }, index=df.index)
