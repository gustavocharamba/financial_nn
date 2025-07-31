import pandas as pd
import numpy as np

def getOBV(history):
    direction = np.sign(history['Close'].diff())
    obv = (direction * history['Volume']).cumsum()

    history['OBV'] = obv.fillna(0)  # Preenche o primeiro valor com 0
