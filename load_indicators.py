import pandas as pd
import numpy as np

from load_datasets import load_financial_datasets

from indicators.adx import getADX
from indicators.atr import getATR
from indicators.bollinger import getBollinger
from indicators.macd import getMACD
from indicators.obv import getOBV
from indicators.psar import getPSAR
from indicators.rsi import getRSI
from indicators.sma import getSMA
from indicators.stochastic import getStochastic
from indicators.vwap import getVWAP

def load_indicators(df):

 getADX(df)
 getATR(df)
 getBollinger(df)
 getMACD(df)
 getOBV(df)
 getPSAR(df)
 getRSI(df)
 getSMA(df)
 getStochastic(df)
 getVWAP(df)

 df = df.fillna(method='ffill').fillna(0)

 return df