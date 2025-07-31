import pandas as pd
import numpy as np

def getBollingerBands(history, window=20, num_std=2):
    rolling_mean = history['Close'].rolling(window=window).mean()
    rolling_std = history['Close'].rolling(window=window).std()

    history['BB_Middle'] = rolling_mean
    history['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    history['BB_Lower'] = rolling_mean - (rolling_std * num_std)
