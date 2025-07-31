import pandas as pd
import numpy as np

def getMACD(history):
    ema_12 = history['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = history['Close'].ewm(span=26, adjust=False).mean()

    history['MACD'] = ema_12 - ema_26
    history['MACD_Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
