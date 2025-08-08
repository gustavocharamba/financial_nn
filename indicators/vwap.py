import pandas as pd
import numpy as np

def getVWAP(history):
    typical_price = (history['High'] + history['Low'] + history['Close']) / 3
    volume = history['Volume']

    typical_volume = typical_price * volume
    cum_typical_volume = typical_volume.cumsum()
    cum_volume = volume.cumsum()

    vwap = cum_typical_volume / cum_volume

    vwap_arr = vwap.values.flatten()

    return pd.DataFrame({'VWAP': vwap_arr}, index=history.index)
