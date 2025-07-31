import pandas as pd
import numpy as np

def getSMA(history):

    sma_3 = history['Close'].rolling(window = 3).median()
    sma_7 = history['Close'].rolling(window = 7).median()
    sma_21 = history['Close'].rolling(window = 21).median()
    sma_50 = history['Close'].rolling(window = 50).median()
    sma_77 = history['Close'].rolling(window = 77).median()
    sma_231 = history['Close'].rolling(window = 231).median()

    history['SMA_3'] = sma_3
    history['SMA_7'] = sma_7
    history['SMA_21'] = sma_21
    history['SMA_50'] = sma_50
    history['SMA_77'] = sma_77
    history['SMA_231'] = sma_3