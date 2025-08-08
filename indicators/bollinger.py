import pandas as pd
import numpy as np

def getBollinger(history, window=20, num_std=2):
    rolling_mean = history['Close'].rolling(window=window).mean()
    rolling_std = history['Close'].rolling(window=window).std()

    # garantir que sejam arrays 1D para uso no DataFrame
    bb_middle = rolling_mean.values.flatten()
    bb_upper = (rolling_mean + (rolling_std * num_std)).values.flatten()
    bb_lower = (rolling_mean - (rolling_std * num_std)).values.flatten()

    return pd.DataFrame({
        'BB_Middle': bb_middle,
        'BB_Upper': bb_upper,
        'BB_Lower': bb_lower
    }, index=history.index)
