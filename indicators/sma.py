import pandas as pd
import numpy as np

def getSMA(history):
    sma_3   = history['Close'].rolling(window=3).median()
    sma_7   = history['Close'].rolling(window=7).median()
    sma_21  = history['Close'].rolling(window=21).median()
    sma_50  = history['Close'].rolling(window=50).median()
    #sma_77  = history['Close'].rolling(window=77).median()
    #sma_231 = history['Close'].rolling(window=231).median()

    sma_3_arr   = sma_3.values.flatten()
    sma_7_arr   = sma_7.values.flatten()
    sma_21_arr  = sma_21.values.flatten()
    sma_50_arr  = sma_50.values.flatten()
   # sma_77_arr  = sma_77.values.flatten()
    #sma_231_arr = sma_231.values.flatten()

    return pd.DataFrame({
        'SMA_3': sma_3_arr,
        'SMA_7': sma_7_arr,
        'SMA_21': sma_21_arr,
        'SMA_50': sma_50_arr,
        #'SMA_77': sma_77_arr,
        #'SMA_231': sma_231_arr
    }, index=history.index)
