import pandas as pd
import numpy as np

def getRSI(history, period=14):
    delta = history['Close'].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = gain.flatten() if hasattr(gain, 'flatten') else gain
    loss = loss.flatten() if hasattr(loss, 'flatten') else loss

    avg_gain = pd.Series(gain, index=history.index).ewm(span=period, adjust=False).mean()
    avg_loss = pd.Series(loss, index=history.index).ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.values.flatten() if hasattr(rsi, 'values') else rsi

    return pd.DataFrame({"RSI": rsi}, index=history.index)
