import pandas as pd
import numpy as np

def getVWAP(history):
    typical_price = (history['High'] + history['Low'] + history['Close']) / 3
    volume = history['Volume']

    cum_typical_volume = (typical_price * volume).cumsum()
    cum_volume = volume.cumsum()

    history['VWAP'] = cum_typical_volume / cum_volume
