import pandas as pd
import numpy as np

def getOBV(history):
    direction = np.sign(history['Close'].diff())
    obv = (direction * history['Volume']).fillna(0).cumsum()

    obv_arr = obv.values.flatten()  # garante vetor 1D

    return pd.DataFrame({
        'OBV': obv_arr
    }, index=history.index)
