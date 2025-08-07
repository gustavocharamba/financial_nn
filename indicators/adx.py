import pandas as pd
import numpy as np

def getADX(history, period=14):
    high = history['High']
    low = history['Low']
    close = history['Close']

    # Calcula as diferenças direcional positivas e negativas
    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range (ATR)
    atr = pd.Series(tr, index=history.index).ewm(span=period, adjust=False).mean()

    # Direcional Indicators (DI)
    plus_di = 100 * pd.Series(plus_dm, index=history.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=history.index).ewm(span=period, adjust=False).mean() / atr

    # DX e ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({
        'ADX': adx,
        '+DI': plus_di,
        '-DI': minus_di
    }, index=history.index)
