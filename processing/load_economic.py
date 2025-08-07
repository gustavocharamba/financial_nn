import os
import pandas as pd

def load_economic_datasets(market):
    base_path = os.path.dirname(__file__)

    # CPI (Consumer Price Index)
    cpi = pd.read_csv(os.path.join(base_path, "data/CPI.csv"), parse_dates=["Date"], index_col="Date")
    cpiCore = pd.read_csv(os.path.join(base_path, "data/CPICore.csv"), parse_dates=["Date"], index_col="Date")

    # PPI (Producer Price Index)
    ppi = pd.read_csv(os.path.join(base_path, "data/PPI.csv"), parse_dates=["Date"], index_col="Date")

    # IPI ( Industrial Production Index)
    ipi = pd.read_csv(os.path.join(base_path, "data/IPI.csv"), parse_dates=["Date"], index_col="Date")

    # Payroll and Unemployment
    payroll = pd.read_csv(os.path.join(base_path, "data/Payroll.csv"), parse_dates=["Date"], index_col="Date")
    unemploy = pd.read_csv(os.path.join(base_path, "data/Unemployment.csv"), parse_dates=["Date"], index_col="Date")

    # Yields
    yield5 = pd.read_csv(os.path.join(base_path, "data/Yield5.csv"), parse_dates=["Date"], index_col="Date")
    yield10 = pd.read_csv(os.path.join(base_path, "data/Yield10.csv"), parse_dates=["Date"], index_col="Date")
    yield30 = pd.read_csv(os.path.join(base_path, "data/Yield30.csv"), parse_dates=["Date"], index_col="Date")

    # Interest Rates
    rates = pd.read_csv(os.path.join(base_path, "data/FEDRates.csv"), parse_dates=["Date"], index_col="Date")

    # Monetary Aggregates (M1 - Narrow Money) (M2 - Broad Money)
    m1 = pd.read_csv(os.path.join(base_path, "data/M1.csv"), parse_dates=["Date"], index_col="Date")
    m2 = pd.read_csv(os.path.join(base_path, "data/M2.csv"), parse_dates=["Date"], index_col="Date")

    # Economic Columns
    economic_dfs = [cpi, cpiCore, ppi, ipi, payroll, unemploy, yield5, yield10, yield30, rates, m1, m2]
    economic_cols = [col for df in economic_dfs for col in df.columns]

    # SP500, Nasdaq, Bitcoin (Daily)
    sp500 = pd.read_csv(os.path.join(base_path, "data/SP500.csv"), parse_dates=["Date"], index_col="Date")
    nasdaq = pd.read_csv(os.path.join(base_path, "data/NASDAQ.csv"), parse_dates=["Date"], index_col="Date")
    bitcoin = pd.read_csv(os.path.join(base_path, "data/BTCUSD.csv"), parse_dates=["Date"], index_col="Date")
    bitcoin = bitcoin.drop(columns=["Unix", "Symbol", "VolumeBTC"])

    if market == "SP500":
        df_sp500 = pd.concat(economic_dfs + [sp500], axis=1, join="outer")
        df_sp500 = df_sp500.ffill()
        start_date = '1977-02-01'
        df_sp500 = df_sp500.loc[start_date:]
        df_sp500[economic_cols] = df_sp500[economic_cols].shift(1)
        df_sp500 = df_sp500.resample('D').ffill()
        df_sp500 = df_sp500.dropna().reset_index()
        return df_sp500

    elif market == "NASDAQ":
        df_nasdaq = pd.concat(economic_dfs + [nasdaq], axis=1, join="outer")
        df_nasdaq = df_nasdaq.ffill()
        start_date = '1977-02-01'
        df_nasdaq = df_nasdaq.loc[start_date:]
        df_nasdaq[economic_cols] = df_nasdaq[economic_cols].shift(1)
        df_nasdaq = df_nasdaq.resample('D').ffill()
        df_nasdaq = df_nasdaq.dropna().reset_index()
        return df_nasdaq

    elif market == "BTC":
        df_bitcoin = pd.concat(economic_dfs + [bitcoin], axis=1, join="outer")
        df_bitcoin = df_bitcoin.ffill()
        start_date = '2016-01-01'
        df_bitcoin = df_bitcoin.loc[start_date:]
        df_bitcoin[economic_cols] = df_bitcoin[economic_cols].shift(1)
        df_bitcoin = df_bitcoin.resample('D').ffill()
        df_bitcoin = df_bitcoin.dropna().reset_index()
        return df_bitcoin
