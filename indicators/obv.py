import pandas as pd
import numpy as np

def getOBV(history):
    direction = np.sign(history['Close'].diff())
    obv = (direction * history['Volume']).fillna(0).cumsum()

    return pd.DataFrame({
        'OBV': obv
    }, index=history.index)
