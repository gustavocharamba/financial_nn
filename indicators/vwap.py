import pandas as pd
import numpy as np

def getVWAP(history):
    typical_price = (history['High'] + history['Low'] + history['Close']) / 3
    volume = history['Volume']

    cum_typical_volume = (typical_price * volume).cumsum()
    cum_volume = volume.cumsum()

    vwap = cum_typical_volume / cum_volume

    return pd.DataFrame({'VWAP': vwap}, index=history.index)
