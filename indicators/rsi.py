import pandas as pd
import numpy as np

def getRSI(history, period = 14):
    delta = history['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    history['RSI'] = 100 - (100 / (1 + rs))