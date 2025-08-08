import pandas as pd
import numpy as np

def getATR(history, period=14):
    high = history['High'].values.flatten()
    low = history['Low'].values.flatten()
    close = history['Close'].values.flatten()

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr2[0] = np.nan  # o primeiro valor nao tem close anterior
    tr3 = np.abs(low - np.roll(close, 1))
    tr3[0] = np.nan

    tr_df = pd.DataFrame({
        'tr1': tr1,
        'tr2': tr2,
        'tr3': tr3
    }, index=history.index)

    tr = tr_df.max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()

    atr_arr = atr.values.flatten()

    return pd.DataFrame({"ATR": atr_arr}, index=history.index)
