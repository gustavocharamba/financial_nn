import pandas as pd
import numpy as np

def getMACD(history):
    ema_12 = history['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = history['Close'].ewm(span=26, adjust=False).mean()

    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Garante que os arrays usados nas colunas são 1D
    macd_arr = macd.values.flatten()
    signal_arr = signal.values.flatten()

    return pd.DataFrame({
        'MACD': macd_arr,
        'MACD_Signal': signal_arr
    }, index=history.index)
