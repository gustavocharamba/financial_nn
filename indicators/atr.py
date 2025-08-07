import pandas as pd
import numpy as np

def getATR(history, period=14):
    high = history['High']
    low = history['Low']
    close = history['Close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({"ATR": atr}, index=history.index)
