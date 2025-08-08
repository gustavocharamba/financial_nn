import pandas as pd
import numpy as np

def getADX(history, period=14):
    high = history['High']
    low = history['Low']
    close = history['Close']

    # Diferenças de alta e baixa
    up_move = high.diff()
    down_move = low.diff().abs()

    # Ajuste para evitar arrays 2D
    up_move = np.ravel(up_move.values) if hasattr(up_move, "values") else np.ravel(up_move)
    down_move = np.ravel(down_move.values) if hasattr(down_move, "values") else np.ravel(down_move)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range (ATR)
    atr = tr.ewm(span=period, adjust=False).mean()

    # Criar pandas Series unidimensionais para movimentos direcionais exponenciais
    plus_di_ewm = pd.Series(plus_dm, index=history.index).ewm(span=period, adjust=False).mean()
    minus_di_ewm = pd.Series(minus_dm, index=history.index).ewm(span=period, adjust=False).mean()

    plus_di = 100 * plus_di_ewm / atr
    minus_di = 100 * minus_di_ewm / atr

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({
        'ADX': adx,
        '+DI': plus_di,
        '-DI': minus_di
    }, index=history.index)
