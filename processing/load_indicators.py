import os
import pandas as pd
import numpy as np

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

def load_indicators(market):
    base_path = os.path.dirname(__file__)

    if market == "SP500":
        df = pd.read_csv(os.path.join(base_path, "../data/SP500.csv"), parse_dates=["Date"], index_col="Date")
        start_date = "1977-02-01"

    elif market == "NASDAQ":
        df = pd.read_csv(os.path.join(base_path, "../data/NASDAQ.csv"), parse_dates=["Date"], index_col="Date")
        start_date = "1977-02-01"

    elif market == "BTC":
        df = pd.read_csv(os.path.join(base_path, "../data/BTCUSD.csv"), parse_dates=["Date"], index_col="Date")
        df = df.drop(columns=[col for col in ["Unix", "Symbol", "VolumeBTC"] if col in df.columns])
        start_date = "2016-01-01"

    # Filtrar data
    df = df.loc[start_date:]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Calcula os indicadores
    adx        = getADX(df)
    atr        = getATR(df)
    bollinger  = getBollinger(df)
    macd       = getMACD(df)
    obv        = getOBV(df)
    psar       = getPSAR(df)
    rsi        = getRSI(df)
    sma        = getSMA(df)
    stochastic = getStochastic(df)
    vwap       = getVWAP(df)

    # Junta os indicadores
    indicators_dfs = [adx, atr, bollinger, macd, obv, psar, rsi, sma, stochastic, vwap]
    indicators_cols = [col for indi_df in indicators_dfs for col in indi_df.columns]

    df_full = pd.concat([df] + indicators_dfs, axis=1)

    # Aplica shift para evitar lookahead
    df_full[indicators_cols] = df_full[indicators_cols].shift(1)

    # Resample diário com ffill
    df_full = df_full.resample('D').ffill()

    # Remove NaNs e reseta o índice
    df_full = df_full.dropna().reset_index()

    return df_full
